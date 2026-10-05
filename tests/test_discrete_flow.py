"""Exact autoregressive density, sampling, training, and checkpoint identity."""

from itertools import product

import numpy as np
import pytest
import torch

from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    torch.manual_seed(123)
    return ConditionalAutoregressiveBernoulli(2, 3, hidden_dim=9)


def states(width):
    return torch.tensor(list(product((0, 1), repeat=width)), dtype=torch.uint8)


def test_exhaustive_normalization_and_exact_log_probability(model):
    z = states(3)
    for s in states(2):
        probabilities = model(z, s)
        assert torch.all((probabilities >= model.eps) & (probabilities <= 1 - model.eps))
        expected = torch.where(z.bool(), probabilities, 1 - probabilities).prod(dim=-1)
        log_probability = model.log_prob(z, s)
        assert torch.isfinite(log_probability).all()
        assert (expected > 0).all()
        torch.testing.assert_close(log_probability.exp(), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(log_probability.exp().sum(), torch.tensor(1.0, dtype=torch.float64))


def test_masks_exclude_current_and_future_bits(model):
    z = states(3)
    s = torch.tensor([1, 0])
    original = model(z, s)
    for j in range(3):
        changed = z.clone()
        changed[:, j:] = 1 - changed[:, j:]
        torch.testing.assert_close(model(changed, s)[:, :j + 1], original[:, :j + 1], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_binary_sampling_repeated_syndromes_and_rng_replay(model, backend):
    def generator():
        return np.random.default_rng(71) if backend == "numpy" else torch.Generator().manual_seed(71)

    s = torch.tensor([[1, 0]]).repeat(300, 1)
    z = model.sample(s, generator())
    assert z.shape == (300, 3)
    assert z.dtype == torch.uint8
    assert torch.all((z == 0) | (z == 1))
    torch.testing.assert_close(z, model.sample(s, generator()))
    assert model.log_prob(z, s).shape == (300,)
    assert torch.isfinite(model.log_prob(z, s)).all()
    torch.testing.assert_close(model.log_prob(z, s), model.log_prob(z, s[0]))
    single = model.sample(s[0], generator())
    assert single.shape == (3,)
    assert model.log_prob(single, s[0]).shape == ()


def test_sampling_uses_same_conditionals_as_log_prob(model):
    s = torch.tensor([[0, 1]]).repeat(100, 1)
    actual = model.sample(s, np.random.default_rng(92))
    rng = np.random.default_rng(92)
    expected = torch.zeros_like(actual)
    for j in range(model.z_dim):
        p = model(expected, s)[:, j].detach().numpy()
        expected[:, j] = torch.from_numpy((rng.random(100) < p).astype(np.uint8))
    torch.testing.assert_close(actual, expected)


def test_empirical_samples_match_exact_joint(model):
    s = torch.tensor([0, 1])
    samples = model.sample(s.repeat(20000, 1), np.random.default_rng(51))
    for z, probability in zip(states(3), model.log_prob(states(3), s).exp()):
        observed = (samples == z).all(dim=-1).double().mean()
        assert abs(float(observed - probability.detach())) < 0.015


def test_saturation_preserves_full_support(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.hidden_output.bias.copy_(torch.tensor([-1e6, 1e6, -1e6]))
    z = states(3)
    p = model(z, [0, 0])
    assert torch.all(p[:, 0] == model.eps)
    assert torch.all(p[:, 1] == 1 - model.eps)
    log_probability = model.log_prob(z, [0, 0])
    assert torch.isfinite(log_probability).all()
    assert (log_probability.exp() > 0).all()
    torch.testing.assert_close(log_probability.exp().sum(), torch.tensor(1.0, dtype=torch.float64))


def test_batch_training_and_syndrome_conditioning(model):
    s = states(2).repeat(32, 1)
    z = torch.column_stack((s[:, 0], s[:, 1], s[:, 0] ^ s[:, 1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    before = -model.log_prob(z, s).mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = -model.log_prob(z, s).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert -model.log_prob(z, s).mean().item() < before * 0.25
    assert not torch.allclose(model(states(3), [0, 0]), model(states(3), [1, 1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_round_trip_and_fingerprint_rejection(model, tmp_path, dtype):
    model.to(dtype=dtype)
    fingerprints = {"edge_ordering_fingerprint": "a" * 64, "h_nullspace_fingerprint": "b" * 64}
    path = tmp_path / "model.pt"
    model.save_checkpoint(path, fingerprints=fingerprints)
    loaded = ConditionalAutoregressiveBernoulli.load_checkpoint(path, expected_fingerprints=fingerprints)
    assert loaded.context_hidden.weight.dtype == dtype
    torch.testing.assert_close(loaded.log_prob(states(3), [1, 0]), model.log_prob(states(3), [1, 0]), rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.sample([[1, 0]] * 20, np.random.default_rng(7)),
        model.sample([[1, 0]] * 20, np.random.default_rng(7)),
    )
    for key in fingerprints:
        with pytest.raises(ValueError, match="fingerprints do not match"):
            ConditionalAutoregressiveBernoulli.load_checkpoint(
                path, expected_fingerprints={**fingerprints, key: "c" * 64},
            )


def test_empty_coordinates_and_batches():
    model = ConditionalAutoregressiveBernoulli(2, 0)
    s = torch.zeros((3, 2))
    z = model.sample(s, np.random.default_rng(1))
    assert z.shape == (3, 0)
    torch.testing.assert_close(model.log_prob(z, s), torch.zeros(3, dtype=torch.float64))
    model = ConditionalAutoregressiveBernoulli(0, 3)
    assert model.sample([], np.random.default_rng(1)).shape == (3,)
    assert model.log_prob(torch.zeros((0, 3)), torch.zeros((0, 0))).shape == (0,)


@pytest.mark.parametrize("kwargs", [{"eps": 0}, {"eps": 0.5}, {"eps": 1e-20}, {"eps": float("nan")}, {"z_dim": -1}, {"hidden_dim": 0}])
def test_invalid_architecture(kwargs):
    with pytest.raises(ValueError):
        ConditionalAutoregressiveBernoulli(**{"syndrome_dim": 2, "z_dim": 3, **kwargs})


@pytest.mark.parametrize("z,s", [([0, 1, 2], [0, 1]), ([0, 0.5, 1], [0, 1]), ([0, 1], [0, 1]), ([0, 1, 0], [0]), ([0, 1, 0], [float("nan"), 0])])
def test_invalid_inputs(model, z, s):
    with pytest.raises(ValueError):
        model.log_prob(z, s)
