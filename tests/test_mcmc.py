"""Exact Metropolis decisions and independent uniform-proposal chain behavior."""

import numpy as np
import pytest
import stim

from surface_code import random_mcmc
from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.gf2 import build_incidence_matrix
from surface_code.random_mcmc import decode_random_mcmc, metropolis_accept


class FixedUniforms:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


@pytest.mark.parametrize("delta,u,expected", [
    (-1000, 0.999, True),
    (-1, 0.999, True),
    (0, 0.999, True),
    (np.log(2), 0.49, True),
    (np.log(2), 0.5, False),  # Strict inequality at the exact threshold.
    (np.log(2), 0.51, False),
    (1000, 0.01, False),
    (700, np.exp(-701), True),
    (700, np.exp(-699), False),
])
def test_exact_log_acceptance_rule(delta, u, expected):
    rng = FixedUniforms([u])
    assert metropolis_accept(delta, rng) is expected
    assert rng.calls == 1


def test_zero_uniform_is_redrawn():
    rng = FixedUniforms([0.0, 0.0, 0.25])
    assert metropolis_accept(np.log(2), rng)
    assert rng.calls == 3


def test_nan_delta_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        metropolis_accept(np.nan, FixedUniforms([]))


def test_controlled_chain_tracks_initial_final_best_and_rejections(monkeypatch):
    graph = build_decoding_graph(stim.DetectorErrorModel("""
        error(0.2) L0
        error(0.1) L0
    """))
    states = [[1, 1], [0, 1], [1, 1], [0, 0], [1, 0], [1, 0]]
    proposals = iter(states)
    calls = []
    rng = FixedUniforms([0.5, 0.9, 0.5, 0.01, 0.9])

    def sample(H, s, supplied_rng):
        assert H.shape == (0, 2)
        assert list(s) == []
        assert supplied_rng is rng
        calls.append(1)
        return np.array(next(proposals), dtype=np.uint8)

    monkeypatch.setattr(random_mcmc, "sample_uniform_solution", sample)
    result = decode_random_mcmc([], graph, 5, rng=rng)
    a, b = [edge.weight for edge in graph.edges]
    assert len(calls) == 6  # Initialization plus one independent draw per iteration.
    assert rng.calls == 5
    np.testing.assert_array_equal(result.initial_configuration, [1, 1])
    np.testing.assert_array_equal(result.final_configuration, [1, 0])
    np.testing.assert_array_equal(result.best_configuration, [0, 0])
    assert result.initial_weight == a + b
    assert result.final_weight == a
    assert result.best_weight == 0
    assert result.prediction == 0  # Best is empty; final state flips L0.
    assert result.accepted_moves == 4
    assert result.rejected_moves == 1
    assert result.acceptance_rate == 0.8
    assert [row.iteration for row in result.trace] == [1, 2, 3, 4, 5]
    assert [row.accepted for row in result.trace] == [True, False, True, True, True]
    np.testing.assert_allclose(
        [row.proposed_weight for row in result.trace], [b, a + b, 0, a, a]
    )
    np.testing.assert_allclose(
        [row.current_weight for row in result.trace], [b, b, 0, a, a]
    )
    np.testing.assert_allclose(
        [row.delta_weight for row in result.trace], [-a, a, -b, a, 0]
    )
    np.testing.assert_allclose(
        [row.best_weight_so_far for row in result.trace], [b, b, 0, 0, 0]
    )


def test_zero_iterations_and_independent_result_snapshots():
    graph = build_decoding_graph(stim.DetectorErrorModel("error(0.1) D0"))
    result = decode_random_mcmc([1], graph, 0, seed=42)
    assert result.trace == ()
    assert result.accepted_moves == result.rejected_moves == 0
    assert result.acceptance_rate == 0
    assert result.initial_weight == result.final_weight == result.best_weight
    for vector in (result.initial_configuration, result.final_configuration, result.best_configuration):
        np.testing.assert_array_equal(vector, [1])
    result.final_configuration[0] = 0
    assert result.initial_configuration[0] == result.best_configuration[0] == 1


@pytest.mark.parametrize("flags,expected", [
    (["L0", "L0", "L1"], 0),
    (["L0", "L0", "L0 L1"], 1),
])
def test_best_configuration_logical_parity_by_hand(flags, expected):
    # Three isolated detector-boundary edges are forced active by s=(1,1,1).
    # Two L0 flags cancel; a third L0 flag makes the parity odd. L1 is ignored.
    model = stim.DetectorErrorModel("\n".join(
        f"error(0.1) D{i} {flag}" for i, flag in enumerate(flags)
    ))
    graph = build_decoding_graph(model)
    result = decode_random_mcmc([1, 1, 1], graph, 3, seed=42)
    np.testing.assert_array_equal(result.best_configuration, [1, 1, 1])
    assert result.prediction == expected
    assert result.best_weight == sum(edge.weight for edge in graph.edges)


def test_seed_and_external_rng_reproducibility():
    graph = build_decoding_graph(stim.DetectorErrorModel("""
        error(0.2) D0
        error(0.1) D0 L0
        error(0.8) L0
    """))
    results = [
        decode_random_mcmc([1], graph, 30, seed=123),
        decode_random_mcmc([1], graph, 30, seed=123),
        decode_random_mcmc([1], graph, 30, rng=np.random.default_rng(123)),
    ]
    for result in results[1:]:
        assert result.trace == results[0].trace
        for field in ("initial_configuration", "final_configuration", "best_configuration"):
            np.testing.assert_array_equal(getattr(result, field), getattr(results[0], field))


def test_surface_code_chain_invariants():
    circuit = build_surface_code(3, 3, 0.005)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    H = build_incidence_matrix(graph)
    syndrome = sample_shots(circuit, 10, seed=42)[0][0]
    original = syndrome.copy()
    result = decode_random_mcmc(syndrome, graph, 20, seed=7)
    weights = np.array([edge.weight for edge in graph.edges])
    for config, weight in (
        (result.initial_configuration, result.initial_weight),
        (result.final_configuration, result.final_weight),
        (result.best_configuration, result.best_weight),
    ):
        np.testing.assert_array_equal((H @ config) % 2, syndrome)
        assert float(np.sum(config * weights)) == weight
    np.testing.assert_array_equal(syndrome, original)
    previous = best = result.initial_weight
    for row in result.trace:
        assert row.delta_weight == row.proposed_weight - previous
        assert row.current_weight == (row.proposed_weight if row.accepted else previous)
        best = min(best, row.current_weight)
        assert row.best_weight_so_far == best
        previous = row.current_weight
    assert result.final_weight == previous
    assert result.best_weight == best
    assert result.accepted_moves == sum(row.accepted for row in result.trace)
    assert result.accepted_moves + result.rejected_moves == 20
    assert result.acceptance_rate == result.accepted_moves / 20


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_invalid_iterations(iterations):
    graph = build_decoding_graph(stim.DetectorErrorModel())
    with pytest.raises(ValueError, match="iterations"):
        decode_random_mcmc([], graph, iterations, seed=1)


def test_rng_and_seed_are_mutually_exclusive():
    graph = build_decoding_graph(stim.DetectorErrorModel())
    with pytest.raises(ValueError, match="either rng or seed"):
        decode_random_mcmc([], graph, 1, rng=np.random.default_rng(1), seed=1)


@pytest.mark.parametrize("p", [0, 1])
def test_infinite_energies_are_rejected(p):
    graph = build_decoding_graph(stim.DetectorErrorModel(f"error({p}) D0"))
    with pytest.raises(ValueError, match="finite edge weights"):
        decode_random_mcmc([0], graph, 1, seed=1)


def test_inconsistent_syndrome():
    graph = build_decoding_graph(stim.DetectorErrorModel("error(0.1) D0 D1"))
    with pytest.raises(ValueError, match="inconsistent"):
        decode_random_mcmc([1, 0], graph, 1, seed=1)


def test_empty_configuration_equal_weight_moves_are_accepted():
    graph = build_decoding_graph(stim.DetectorErrorModel())
    result = decode_random_mcmc([], graph, 3, seed=1)
    assert result.best_configuration.shape == (0,)
    assert result.best_weight == 0
    assert result.acceptance_rate == 1
    assert all(row.delta_weight == 0 and row.accepted for row in result.trace)
