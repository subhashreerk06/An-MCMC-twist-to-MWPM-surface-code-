"""Posterior state counting, tie isolation, and unchanged MH trajectories."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import stim
import torch

from surface_code import posterior_mcmc as posterior
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_mcmc import FlowProposalMCMCDecoder
from surface_code.gf2 import build_incidence_matrix, sample_uniform_solution


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ScriptedModel(torch.nn.Module):
    def __init__(self, proposals):
        super().__init__()
        self.syndrome_dim, self.z_dim = 0, 2
        self.proposals = iter(proposals)

    def sample(self, syndrome, rng):
        return np.array(next(self.proposals), dtype=np.uint8)

    def log_prob(self, z, syndrome):
        return np.log([0.9, 0.04, 0.01, 0.05][2 * int(z[0]) + int(z[1])])


class Uniforms:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def graph():
    return build_decoding_graph(stim.DetectorErrorModel("error(0.2) L0\nerror(0.1) L0"))


def forbidden(*args, **kwargs):
    raise AssertionError("MWPM must only be called for a retained-count tie")


def test_rejections_self_proposals_burn_in_and_saved_trace(monkeypatch, tmp_path):
    g = graph()
    monkeypatch.setattr(posterior, "sample_uniform_solution", lambda *args: np.array([1, 0], dtype=np.uint8))
    monkeypatch.setattr(posterior, "decode_mwpm", forbidden)
    decoder = posterior.PosteriorMCMCDecoder(
        g, ScriptedModel([[0, 0], [1, 1], [1, 0], [0, 1], [1, 0]]), flow_probability=1,
    )
    path = tmp_path / "logical.npy"
    result = decoder.decode([], 5, burn_in=1, rng=Uniforms([0.5, 0.001, 0.9, 0.9, 0.9]), logical_trace_path=path)
    assert [r.accepted for r in result.trace] == [False, True, True, False, True]
    assert result.logical_parity_trace == (0, 1, 1, 1)
    assert result.retained_samples == 4 and result.burn_in == 1
    assert result.logical_0_samples == 1 and result.logical_1_samples == 3
    assert result.posterior_p1 == 0.75 and result.posterior_p0 == 0.25
    assert result.prediction == 1 and result.logical_class_transitions == 1
    diagnostics = posterior.logical_trace_diagnostics(result.logical_parity_trace)
    assert result.logical_diagnostics == diagnostics
    assert result.logical_transitions == result.logical_class_transitions
    assert result.logical_ess == diagnostics.logical_ess
    assert result.logical_autocorrelation == diagnostics.autocorrelation
    assert result.logical_diagnostic_status == diagnostics.status
    assert result.posterior_p1 == diagnostics.posterior_p1
    assert result.acceptance_rate == 0.6
    assert result.flow_proposals == 5 and result.flow_accepted == 3
    assert result.uniform_proposals == result.uniform_accepted == 0
    assert result.minimum_weight_seen == result.initial_weight == g.edges[0].weight
    assert result.mean_retained_weight == pytest.approx(g.edges[0].weight + g.edges[1].weight / 4)
    np.testing.assert_array_equal(np.load(path), result.logical_parity_trace)


@pytest.mark.parametrize("tie_prediction", [0, 1])
def test_mwpm_only_after_chain_and_only_exact_tie(monkeypatch, tie_prediction):
    events = []
    def initial(*args):
        events.append("initial")
        return np.array([1, 1], dtype=np.uint8)
    def tie(*args):
        events.append("tie")
        assert events == ["initial", "decision", "decision", "tie"]
        return SimpleNamespace(prediction=tie_prediction)
    class Decisions(Uniforms):
        def random(self):
            events.append("decision")
            return super().random()
    monkeypatch.setattr(posterior, "sample_uniform_solution", initial)
    monkeypatch.setattr(posterior, "decode_mwpm", tie)
    result = posterior.PosteriorMCMCDecoder(graph(), ScriptedModel([[1, 0], [1, 1]]), flow_probability=1).decode(
        [], 2, rng=Decisions([0.9, 0.001]),
    )
    assert result.logical_parity_trace == (1, 0)
    assert result.posterior_p1 == result.posterior_p0 == 0.5
    assert result.prediction == tie_prediction


def test_majority_differs_from_minimum_weight_without_reweighting(monkeypatch):
    monkeypatch.setattr(posterior, "sample_uniform_solution", lambda *args: np.array([1, 1], dtype=np.uint8))
    monkeypatch.setattr(posterior, "decode_mwpm", forbidden)
    result = posterior.PosteriorMCMCDecoder(
        graph(), ScriptedModel([[1, 0], [1, 1], [1, 1], [1, 1]]), flow_probability=1,
    ).decode([], 4, rng=Uniforms([0.9, 0.001, 0.9, 0.9]))
    assert result.logical_parity_trace == (1, 0, 0, 0)
    assert result.prediction == 0 and result.posterior_p1 == 0.25
    # The minimum-weight visited state is 10, whose L0 parity is 1.
    assert result.minimum_weight_seen == graph().edges[0].weight


@pytest.mark.parametrize("alpha", [0, 0.3, 0.9, 1])
def test_seeded_chain_matches_existing_sampler_and_random_start(alpha):
    g = build_decoding_graph(stim.DetectorErrorModel("error(0.2) D0\nerror(0.1) D0 L0\nerror(0.8) L1"))
    model = ConditionalAutoregressiveBernoulli(1, 2, 8)
    expected = FlowProposalMCMCDecoder(g, model, flow_probability=alpha).decode([1], 40, seed=19)
    result = posterior.PosteriorMCMCDecoder(g, model, flow_probability=alpha).decode([1], 40, burn_in=7, seed=19)
    assert result.trace == expected.trace
    np.testing.assert_array_equal(result.initial_configuration, sample_uniform_solution(build_incidence_matrix(g), [1], np.random.default_rng(19)))
    np.testing.assert_array_equal(result.final_configuration, expected.final_configuration)
    assert result.flow_accepted == expected.flow_accepted_moves
    assert result.uniform_accepted == expected.uniform_accepted_moves
    assert result.minimum_weight_seen == expected.best_weight
    assert result.retained_samples == 33


def test_frequencies_match_enumerated_posterior():
    g = graph()
    states = np.array(list(product((0, 1), repeat=2)))
    target = np.exp(-states @ np.array([e.weight for e in g.edges]))
    target /= target.sum()
    exact_p1 = target[states.sum(axis=1) % 2 == 1].sum()
    result = posterior.posterior_mcmc_decode([], g, ConditionalAutoregressiveBernoulli(0, 2, 8), 12000, burn_in=1000, flow_probability=0, seed=42)
    assert result.posterior_p1 == pytest.approx(exact_p1, abs=0.025)


@pytest.mark.parametrize("iterations,burn_in", [(0, 0), (2, 2), (2, 3), (2, -1), (True, 0), (2, True), (2, 0.5)])
def test_invalid_sample_budget(iterations, burn_in):
    decoder = posterior.PosteriorMCMCDecoder(graph(), ScriptedModel([]))
    with pytest.raises(ValueError):
        decoder.decode([], iterations, burn_in=burn_in)


def test_absolute_l0_convention_ignores_other_observables():
    g = build_decoding_graph(stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L1"))
    result = posterior.PosteriorMCMCDecoder(g, ConditionalAutoregressiveBernoulli(2, 0, 8)).decode([1, 1], 3, burn_in=2, seed=1)
    assert result.logical_parity_trace == (1,)
    assert result.prediction == 1 and result.logical_class_transitions == 0
    assert result.logical_ess is None
    assert result.logical_diagnostic_status == "constant_trace"


@pytest.mark.parametrize("bit,n", [(0, 1), (1, 1), (0, 20), (1, 20)])
def test_constant_trace_diagnostics_are_explicit_and_nan_free(bit, n):
    result = posterior.logical_trace_diagnostics([bit] * n)
    assert result.logical_transitions == 0
    assert result.transitions_0_to_1 == result.transitions_1_to_0 == 0
    assert result.posterior_p1 == bit and result.posterior_p0 == 1 - bit
    assert result.logical_ess is None and result.status == "constant_trace"
    assert result.autocorrelation == (None,) * n
    assert result.ess_truncation_lag == 0


def test_alternating_trace_stops_at_first_negative_correlation():
    result = posterior.logical_trace_diagnostics([0, 1] * 4)
    assert result.logical_transitions == 7
    assert result.transitions_0_to_1 == 4 and result.transitions_1_to_0 == 3
    assert result.posterior_p0 == result.posterior_p1 == 0.5
    assert result.autocorrelation[:3] == (1.0, -0.875, 0.75)
    # The later positive correlation must not restart the sum.
    assert result.logical_ess == 8 and result.ess_truncation_lag == 0
    assert result.status == "ok"


def test_persistent_trace_has_small_ess_and_exact_autocorrelation():
    result = posterior.logical_trace_diagnostics([0] * 4 + [1] * 4)
    assert result.logical_transitions == result.transitions_0_to_1 == 1
    assert result.transitions_1_to_0 == 0
    assert result.autocorrelation[:4] == (1.0, 0.625, 0.25, -0.125)
    assert result.logical_ess == pytest.approx(8 / (1 + 2 * (0.625 + 0.25)))
    assert result.ess_truncation_lag == 2 and result.status == "ok"
    reverse = posterior.logical_trace_diagnostics([1] * 4 + [0] * 4)
    assert reverse.transitions_1_to_0 == 1 and reverse.transitions_0_to_1 == 0
    assert reverse.logical_ess == result.logical_ess


def test_zero_correlation_stops_sum():
    result = posterior.logical_trace_diagnostics([0, 0, 0, 1, 1, 1])
    assert result.autocorrelation[:3] == (1.0, 0.5, 0.0)
    assert result.logical_ess == 3 and result.ess_truncation_lag == 1
    assert result.status == "ok"


def test_lag_limit_is_bounded_and_reported():
    result = posterior.logical_trace_diagnostics([0] * 200 + [1] * 200)
    assert len(result.autocorrelation) == 101
    assert result.status == "lag_limit_reached" and result.ess_truncation_lag == 100
    limited = posterior.logical_trace_diagnostics([0] * 4 + [1] * 4, max_lag=1)
    assert limited.logical_ess == pytest.approx(8 / 2.25)
    assert limited.status == "lag_limit_reached"


def test_independent_binary_trace_has_large_ess():
    trace = np.random.default_rng(42).integers(0, 2, size=10000)
    result = posterior.logical_trace_diagnostics(trace)
    assert 0.9 * len(trace) < result.logical_ess <= len(trace)
    assert result.posterior_p1 == np.mean(trace)
    assert result.logical_transitions == np.count_nonzero(trace[1:] != trace[:-1])


@pytest.mark.parametrize("trace", [[], [[0, 1]], [0, 2], [0, np.nan], [0, 0.5]])
def test_invalid_diagnostic_trace(trace):
    with pytest.raises(ValueError, match="binary sequence"):
        posterior.logical_trace_diagnostics(trace)


@pytest.mark.parametrize("max_lag", [0, -1, True, 1.5])
def test_invalid_diagnostic_lag(max_lag):
    with pytest.raises(ValueError, match="max_lag"):
        posterior.logical_trace_diagnostics([0, 1], max_lag=max_lag)
