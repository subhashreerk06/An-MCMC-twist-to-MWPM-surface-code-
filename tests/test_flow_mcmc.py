"""Exact Hastings decisions, frozen proposals, and uniform-start invariants."""

from itertools import product
import subprocess
import sys

import numpy as np
import pytest
import stim
import torch

from surface_code import flow_mcmc
from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.gf2 import build_incidence_matrix, AffineCoordinates, sample_uniform_solution


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Uniforms:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


class ScriptedModel(torch.nn.Module):
    """Full-support two-bit joint distribution with scripted proposal draws."""

    def __init__(self, proposals):
        super().__init__()
        self.syndrome_dim, self.z_dim = 0, 2
        self.register_buffer("probabilities", torch.tensor([0.9, 0.04, 0.01, 0.05], dtype=torch.float64))
        self.proposals = iter(proposals)
        self.sample_calls = self.log_calls = 0

    def sample(self, s, rng):
        assert not torch.is_grad_enabled() and not self.training
        self.sample_calls += 1
        return torch.tensor(next(self.proposals))

    def log_prob(self, z, s):
        assert not torch.is_grad_enabled() and not self.training
        self.log_calls += 1
        return self.probabilities[int(z[0]) * 2 + int(z[1])].log()


def logical_graph():
    return build_decoding_graph(stim.DetectorErrorModel("error(0.2) L0\nerror(0.1) L0"))


@pytest.mark.parametrize("ratio,u,expected", [
    (1000, 0.999, True), (0, 0.999, True), (-np.log(2), 0.49, True),
    (-np.log(2), 0.5, False), (-1000, 0.01, False), (-700, np.exp(-701), True),
])
def test_strict_log_acceptance(ratio, u, expected):
    rng = Uniforms([u])
    assert flow_mcmc.independent_metropolis_accept(ratio, rng) is expected
    assert rng.calls == 1


def test_zero_draw_is_redrawn_and_nan_rejected():
    rng = Uniforms([0, 0, 0.25])
    assert flow_mcmc.independent_metropolis_accept(-np.log(2), rng)
    assert rng.calls == 3
    with pytest.raises(ValueError, match="NaN"):
        flow_mcmc.independent_metropolis_accept(np.nan, Uniforms([]))


def test_scripted_hastings_chain_best_excludes_rejected_proposals(monkeypatch):
    graph = logical_graph()
    proposals = [[0, 0], [1, 1], [1, 0], [0, 1], [1, 0]]
    model = ScriptedModel(proposals)
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, model, flow_probability=1)
    calls = []
    rng = Uniforms([0.5, 0.001, 0.9, 0.9, 0.9])

    def initial(H, s, supplied_rng):
        assert supplied_rng is rng
        assert decoder._model.sample_calls == decoder._model.log_calls == 0
        calls.append(1)
        return np.array([1, 0], dtype=np.uint8)

    monkeypatch.setattr(flow_mcmc, "sample_uniform_solution", initial)
    result = decoder.decode([], 5, rng=rng)
    assert len(calls) == 1
    assert decoder._model.sample_calls == 5
    assert decoder._model.log_calls == 6  # Current density is cached until acceptance.
    assert model.sample_calls == model.log_calls == 0
    assert [row.accepted for row in result.trace] == [False, True, True, False, True]
    assert result.accepted_moves == 3 and result.rejected_moves == 2
    assert result.acceptance_rate == 0.6
    np.testing.assert_array_equal(result.initial_configuration, [1, 0])
    np.testing.assert_array_equal(result.final_configuration, [1, 0])
    # Proposal 00 has weight zero, but is rejected and never visited.
    np.testing.assert_array_equal(result.best_configuration, [1, 0])
    assert result.best_weight == graph.edges[0].weight and result.prediction == 1
    previous_weight = result.initial_weight
    previous_log = np.log(0.01)
    q_proposed = [0.9, 0.05, 0.01, 0.04, 0.01]
    weights = np.array([edge.weight for edge in graph.edges])
    for iteration, (row, z, q) in enumerate(zip(result.trace, proposals, q_proposed), 1):
        assert row.iteration == iteration
        assert row.proposed_weight == np.sum(np.array(z) * weights)
        assert row.log_q_current == pytest.approx(previous_log)
        assert row.log_q_proposed == pytest.approx(np.log(q))
        assert row.log_acceptance_ratio == pytest.approx(
            previous_weight - row.proposed_weight + previous_log - np.log(q)
        )
        assert row.current_weight == (row.proposed_weight if row.accepted else previous_weight)
        assert row.best_weight_so_far == result.initial_weight
        previous_weight = row.current_weight
        if row.accepted:
            previous_log = row.log_q_proposed
    # A weight-only rule would have accepted the first proposal; this chain rejects it.
    assert result.trace[0].log_acceptance_ratio < 0


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.9, 1.0])
def test_exact_transition_matrix_has_detailed_balance(monkeypatch, alpha):
    graph = logical_graph()
    states = list(product((0, 1), repeat=2))
    q = alpha * np.array([0.9, 0.04, 0.01, 0.05]) + (1 - alpha) / 4
    weights = np.array([edge.weight for edge in graph.edges])
    target = np.exp(-np.array(states) @ weights)
    target /= target.sum()
    transition = np.zeros((4, 4))
    for i, current in enumerate(states):
        for j, proposed in enumerate(states):
            draws = iter([current, proposed])
            monkeypatch.setattr(flow_mcmc, "sample_uniform_solution", lambda H, s, rng: np.array(next(draws), dtype=np.uint8))
            decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, ScriptedModel([proposed]), flow_probability=alpha)
            rng = Uniforms([0.0, 0.5] if 0 < alpha < 1 else [0.5])
            row = decoder.decode([], 1, rng=rng).trace[0]
            if i != j:
                transition[i, j] = q[j] * min(1, np.exp(row.log_acceptance_ratio))
        transition[i, i] = 1 - transition[i].sum()
    np.testing.assert_allclose(target[:, None] * transition, (target[:, None] * transition).T, atol=1e-15)
    np.testing.assert_allclose(target @ transition, target, atol=1e-15)


def test_uniform_initialization_replay_and_model_is_frozen():
    graph = build_decoding_graph(stim.DetectorErrorModel("error(0.2) D0\nerror(0.1) D0 L0\nerror(0.8) L0"))
    H = build_incidence_matrix(graph)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = ConditionalAutoregressiveBernoulli(1, 2, 8).train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, model)
    frozen_before = {key: value.clone() for key, value in decoder._model.state_dict().items()}
    results = [decoder.decode([1], 12, seed=33), decoder.decode([1], 12, seed=33),
               decoder.decode([1], 12, rng=np.random.default_rng(33))]
    expected_initial = sample_uniform_solution(H, [1], np.random.default_rng(33))
    for result in results:
        np.testing.assert_array_equal(result.initial_configuration, expected_initial)
        assert result.trace == results[0].trace
        for name in ("initial_configuration", "final_configuration", "best_configuration"):
            np.testing.assert_array_equal(H @ getattr(result, name) % 2, [1])
        assert result.best_weight == min([result.initial_weight] + [row.current_weight for row in result.trace])
        running_best = result.initial_weight
        for row in result.trace:
            running_best = min(running_best, row.current_weight)
            assert row.best_weight_so_far == running_best
    assert model.training
    assert all(p.requires_grad for p in model.parameters())
    assert not decoder._model.training
    assert all(not p.requires_grad and p.grad is None for p in decoder._model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(decoder._model.state_dict()[key], frozen_before[key], rtol=0, atol=0)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert decoder.decode([1], 12, seed=33).trace == results[0].trace


def test_zero_iterations_and_independent_snapshots():
    graph = logical_graph()
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, ScriptedModel([]))
    result = decoder.decode([], 0, seed=4)
    assert result.trace == () and result.acceptance_rate == 0
    assert result.accepted_moves == result.rejected_moves == 0
    assert decoder._model.sample_calls == 0
    assert result.initial_weight == result.final_weight == result.best_weight
    result.final_configuration[:] ^= 1
    np.testing.assert_array_equal(result.initial_configuration, result.best_configuration)


def test_surface_code_every_proposal_checked_and_best_logical_parity(monkeypatch):
    circuit = build_surface_code(3, 1, 0.005)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    H = build_incidence_matrix(graph)
    coordinates = AffineCoordinates(H)
    model = ConditionalAutoregressiveBernoulli(H.shape[0], coordinates.nullity, 8)
    original = AffineCoordinates.coordinates_to_error
    errors = []

    def checked(self, s, z):
        e = original(self, s, z)
        np.testing.assert_array_equal(H @ e % 2, s)
        errors.append(e.copy())
        return e

    monkeypatch.setattr(AffineCoordinates, "coordinates_to_error", checked)
    syndrome = sample_shots(circuit, 3, seed=22)[0][0]
    result = flow_mcmc.FlowProposalMCMCDecoder(graph, model).decode(syndrome, 5, seed=7)
    assert len(errors) == 5
    weights = np.array([edge.weight for edge in graph.edges])
    visited = [result.initial_configuration] + [e for e, row in zip(errors, result.trace) if row.accepted]
    assert result.best_weight == pytest.approx(min(float(e @ weights) for e in visited))
    assert result.prediction == sum(edge.logical_flip for active, edge in zip(result.best_configuration, graph.edges) if active) % 2


def test_checkpoint_identity_and_dimension_validation(tmp_path):
    graph = logical_graph()
    model = ConditionalAutoregressiveBernoulli(0, 2, 8)
    path = tmp_path / "model.pt"
    model.save_checkpoint(path, fingerprints=flow_mcmc._graph_fingerprints(graph))
    decoder = flow_mcmc.FlowProposalMCMCDecoder.from_checkpoint(graph, path)
    assert len(decoder.decode([], 2, seed=7).trace) == 2
    changed = build_decoding_graph(stim.DetectorErrorModel("error(0.3) L0\nerror(0.1) L0"))
    with pytest.raises(ValueError, match="fingerprints"):
        flow_mcmc.FlowProposalMCMCDecoder.from_checkpoint(changed, path)
    with pytest.raises(ValueError, match="dimensions"):
        flow_mcmc.FlowProposalMCMCDecoder(graph, ConditionalAutoregressiveBernoulli(1, 2, 8))


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_invalid_iterations(iterations):
    decoder = flow_mcmc.FlowProposalMCMCDecoder(logical_graph(), ScriptedModel([]))
    with pytest.raises(ValueError, match="iterations"):
        decoder.decode([], iterations)


def test_rng_seed_exclusive_and_nonfinite_densities():
    decoder = flow_mcmc.FlowProposalMCMCDecoder(logical_graph(), ScriptedModel([]), flow_probability=1)
    with pytest.raises(ValueError, match="either rng or seed"):
        decoder.decode([], 1, seed=1, rng=np.random.default_rng(1))
    decoder._model.log_prob = lambda z, s: torch.tensor(float("-inf"))
    with pytest.raises(ValueError, match="full support"):
        decoder.decode([], 1, seed=1)


@pytest.mark.parametrize("p", [0, 1])
def test_nonfinite_edge_weights(p):
    graph = build_decoding_graph(stim.DetectorErrorModel(f"error({p}) L0"))
    with pytest.raises(ValueError, match="finite edge weights"):
        flow_mcmc.FlowProposalMCMCDecoder(graph, ConditionalAutoregressiveBernoulli(0, 1))


def test_invalid_syndrome_and_proposal_guard(monkeypatch):
    graph = build_decoding_graph(stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.2) L0"))
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, ConditionalAutoregressiveBernoulli(2, 1, 8))
    with pytest.raises(ValueError, match="inconsistent"):
        decoder.decode([0, 1], 1, seed=7)
    original = AffineCoordinates.coordinates_to_error

    def bad(self, s, z):
        e = original(self, s, z)
        e[0] ^= 1
        return e

    monkeypatch.setattr(AffineCoordinates, "coordinates_to_error", bad)
    with pytest.raises(RuntimeError, match="violates"):
        decoder.decode([1, 1], 1, seed=7)


def test_import_independence():
    script = '''
import builtins
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if 'mwpm' in name.lower() or name.endswith('random_mcmc'):
        raise AssertionError('Unexpected decoder dependency')
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
import stim
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_mcmc import FlowProposalMCMCDecoder
graph = build_decoding_graph(stim.DetectorErrorModel('error(0.1) D0'))
decoder = FlowProposalMCMCDecoder(graph, ConditionalAutoregressiveBernoulli(1, 0))
assert decoder.decode([1], 1, seed=1).best_configuration.tolist() == [1]
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("alpha", [0, 0.3, 0.9, 1])
def test_mixture_probabilities_normalize(alpha):
    q = np.array([0.9, 0.04, 0.01, 0.05])
    actual = np.array([flow_mcmc.log_mixture_probability(np.log(p), 2, alpha) for p in q])
    np.testing.assert_allclose(actual, np.log(alpha * q + (1 - alpha) / 4), atol=1e-15)
    assert np.exp(actual).sum() == pytest.approx(1)


def test_log_mixture_is_stable_for_tiny_probabilities_and_empty_coordinates():
    floor = np.log1p(-0.9) - 100000 * np.log(2)
    assert flow_mcmc.log_mixture_probability(-1e6, 100000, 0.9) == pytest.approx(floor)
    assert flow_mcmc.log_mixture_probability(-np.inf, 100000, 0.9) == pytest.approx(floor)
    assert flow_mcmc.log_mixture_probability(-1e6, 100000, 1) == -1e6
    assert flow_mcmc.log_mixture_probability(-np.inf, 100000, 0) == -100000 * np.log(2)
    assert flow_mcmc.log_mixture_probability(0, 0, 0.9) == pytest.approx(0, abs=1e-15)


def test_mixed_components_use_total_density_and_separate_counts(monkeypatch):
    graph = logical_graph()
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, ScriptedModel([[0, 0], [1, 0]]))
    assert decoder.flow_probability == 0.9
    uniform_states = iter([[1, 0], [1, 1], [0, 1]])
    uniform_calls = []

    def uniform(H, s, rng):
        uniform_calls.append(1)
        return np.array(next(uniform_states), dtype=np.uint8)

    monkeypatch.setattr(flow_mcmc, "sample_uniform_solution", uniform)
    # Selection and acceptance draws alternate; scripted samplers consume none.
    rng = Uniforms([0.1, 0.99, 0.95, 0.001, 0.1, 0.9, 0.95, 0.8])
    result = decoder.decode([], 4, rng=rng)
    assert len(uniform_calls) == 3  # Initial state and two uniform proposals.
    assert decoder._model.sample_calls == 2 and decoder._model.log_calls == 5
    assert [row.proposal_source for row in result.trace] == ["flow", "uniform", "flow", "uniform"]
    assert [row.accepted for row in result.trace] == [False, True, True, False]
    assert result.flow_proposals == result.uniform_proposals == 2
    assert result.flow_accepted_moves == result.uniform_accepted_moves == 1
    assert result.flow_acceptance_rate == result.uniform_acceptance_rate == 0.5
    assert result.accepted_moves == result.flow_accepted_moves + result.uniform_accepted_moves
    assert result.flow_probability == 0.9
    previous_weight, previous_q = result.initial_weight, 0.9 * 0.01 + 0.1 / 4
    for row, q_flow in zip(result.trace, [0.9, 0.05, 0.01, 0.04]):
        q_proposed = 0.9 * q_flow + 0.1 / 4
        assert row.log_q_mix_current == pytest.approx(np.log(previous_q))
        assert row.log_q_mix_proposed == pytest.approx(np.log(q_proposed))
        assert row.log_acceptance_ratio == pytest.approx(previous_weight - row.proposed_weight + np.log(previous_q / q_proposed))
        if row.accepted:
            previous_q = q_proposed
        previous_weight = row.current_weight


@pytest.mark.parametrize("source,choice,proposal,u,expected", [
    ("flow", 0.1, [0, 0], 0.1, True),  # Component-only flow density would reject.
    ("uniform", 0.95, [1, 1], 0.08, False),  # Component-only uniform density would accept.
])
def test_acceptance_differs_from_component_only_ratio(monkeypatch, source, choice, proposal, u, expected):
    graph = logical_graph()
    states = iter([[1, 0], proposal])
    monkeypatch.setattr(flow_mcmc, "sample_uniform_solution", lambda H, s, rng: np.array(next(states), dtype=np.uint8))
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, ScriptedModel([proposal]))
    result = decoder.decode([], 1, rng=Uniforms([choice, u]))
    row = result.trace[0]
    assert row.proposal_source == source and row.accepted is expected
    component_ratio = result.initial_weight - row.proposed_weight
    if source == "flow":
        component_ratio += np.log(0.01) - np.log(0.9)
    assert flow_mcmc.independent_metropolis_accept(component_ratio, Uniforms([u])) is not expected


def test_pure_uniform_matches_original_baseline_without_model_calls():
    from surface_code.random_mcmc import decode_random_mcmc

    graph = build_decoding_graph(stim.DetectorErrorModel("error(0.2) D0\nerror(0.1) D0 L0\nerror(0.8) L0"))
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, ConditionalAutoregressiveBernoulli(1, 2, 8), flow_probability=0)

    def forbidden(*args):
        raise AssertionError("Pure uniform decoding must not call the learned model")

    decoder._model.sample = decoder._model.log_prob = forbidden
    actual = decoder.decode([1], 100, seed=19)
    expected = decode_random_mcmc([1], graph, 100, seed=19)
    for name in ("initial_configuration", "final_configuration", "best_configuration"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.accepted_moves == expected.accepted_moves
    assert actual.best_weight == expected.best_weight
    assert actual.flow_proposals == actual.flow_accepted_moves == 0
    assert actual.flow_acceptance_rate == 0
    assert actual.uniform_proposals == 100 and actual.uniform_accepted_moves == expected.accepted_moves
    for a, b in zip(actual.trace, expected.trace):
        assert a.proposal_source == "uniform"
        assert a.log_acceptance_ratio == -b.delta_weight
        assert a.accepted == b.accepted
        assert a.current_weight == b.current_weight
        assert a.log_q_mix_current == a.log_q_mix_proposed == -2 * np.log(2)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, -np.inf, None])
def test_invalid_mixture_probability(alpha):
    with pytest.raises(ValueError, match="flow_probability"):
        flow_mcmc.FlowProposalMCMCDecoder(logical_graph(), ScriptedModel([]), flow_probability=alpha)


def test_zero_flow_density_still_has_uniform_support(monkeypatch):
    model = ScriptedModel([[0, 0]])
    model.probabilities.copy_(torch.tensor([0.9, 0.05, 0.0, 0.05]))
    monkeypatch.setattr(flow_mcmc, "sample_uniform_solution", lambda H, s, rng: np.array([1, 0], dtype=np.uint8))
    decoder = flow_mcmc.FlowProposalMCMCDecoder(logical_graph(), model)
    row = decoder.decode([], 1, rng=Uniforms([0.1, 0.1])).trace[0]
    assert row.log_q_mix_current == pytest.approx(np.log(0.1 / 4))
    assert np.isfinite(row.log_acceptance_ratio)


@pytest.mark.parametrize("flags,expected", [
    (["L0", "L0", "L1"], 0),
    (["L0", "L0", "L0 L1"], 1),
    (["L1", "L1", "L1"], 0),
    (["L0", "L1", "L1"], 1),
])
@pytest.mark.parametrize("iterations", [0, 3])
def test_logical_prediction_matches_random_baseline_convention(flags, expected, iterations):
    from surface_code.random_mcmc import decode_random_mcmc

    graph = build_decoding_graph(stim.DetectorErrorModel("\n".join(
        [f"error(0.1) D{i} {flag}" for i, flag in enumerate(flags)]
        + ["error(0.1) D3 L0"]
    )))
    # First three edges are forced active. The fourth L0 edge must be ignored.
    syndrome = [1, 1, 1, 0]
    model = ConditionalAutoregressiveBernoulli(4, 0, 8)
    result = flow_mcmc.FlowProposalMCMCDecoder(graph, model).decode(syndrome, iterations, seed=42)
    baseline = decode_random_mcmc(syndrome, graph, iterations, seed=42)
    np.testing.assert_array_equal(result.best_configuration, syndrome)
    assert result.prediction == baseline.prediction == expected
    assert result.best_weight == baseline.best_weight


def test_prediction_uses_best_not_final_or_majority_and_returns_requested_fields(monkeypatch):
    graph = logical_graph()
    # Best state 10 flips L0. Initial/final states and most visited states are
    # 11, whose two active L0 flags cancel. The model's modal state is 00.
    model = ScriptedModel([[1, 0], [1, 1], [1, 1], [1, 1]])
    decoder = flow_mcmc.FlowProposalMCMCDecoder(graph, model, flow_probability=1)
    monkeypatch.setattr(flow_mcmc, "sample_uniform_solution", lambda H, s, rng: np.array([1, 1], dtype=np.uint8))
    result = decoder.decode([], 4, rng=Uniforms([0.9, 0.001, 0.9, 0.9]))
    assert all(row.accepted for row in result.trace)
    np.testing.assert_array_equal(result.final_configuration, [1, 1])
    np.testing.assert_array_equal(result.best_configuration, [1, 0])
    expected = {
        "prediction": 1,
        "best_configuration": np.array([1, 0], dtype=np.uint8),
        "best_weight": graph.edges[0].weight,
        "initial_weight": sum(edge.weight for edge in graph.edges),
        "final_weight": sum(edge.weight for edge in graph.edges),
        "acceptance_rate": 1.0,
        "flow_acceptance_rate": 1.0,
        "uniform_acceptance_rate": 0.0,
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(result, name), value)
