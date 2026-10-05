"""Fixed correction gates and independence from matching information."""
from types import SimpleNamespace

import numpy as np
import pytest
import stim

from surface_code import mwpm_posterior_corrector as corrector
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.gf2 import build_incidence_matrix, sample_uniform_solution
from surface_code.mwpm_decoder import MWPMResult
from surface_code.posterior_mcmc import PosteriorMCMCDecoder


@pytest.fixture
def inputs():
    graph = build_decoding_graph(stim.DetectorErrorModel("error(0.2) D0\nerror(0.1) D0 L0"))
    return graph, ConditionalAutoregressiveBernoulli(1, 1, 8)


@pytest.mark.parametrize("mwpm,q,margin,minimum_transitions,minimum_ess,transitions,ess,expected,passed", [
    (0, 0.25, 0.1, None, None, 0, None, 0, True),  # Agree.
    (1, 0.75, 0.1, 2, 5, 0, None, 1, False),  # Agreement despite failed gates.
    (0, 0.75, 0.25, None, None, 0, None, 1, True),  # Inclusive margin.
    (1, 0.25, 0.25, None, None, 0, None, 0, True),  # Correction in other direction.
    (0, 0.75, 0.3, None, None, 0, None, 0, True),  # Confidence alone fails.
    (0, 0.75, 0.25, 2, None, 1, 100, 0, False),
    (0, 0.75, 0.25, 2, 5, 2, 5, 1, True),  # Inclusive quality gates.
    (0, 0.75, 0.25, 2, 5, 2, 4.9, 0, False),
    (0, 1.0, 0.5, None, 0, 0, None, 0, False),  # Undefined ESS fails enabled gate.
    (0, 1.0, 0.5, None, None, 0, None, 1, True),  # Gates are optional.
    (0, 0.75, 0.25, None, 5, 2, np.nan, 0, False),
    (0, 0.75, 0.25, None, 5, 2, np.inf, 0, False),
    (1, 0.5, 0.1, None, None, 2, 5, 1, True),  # Positive margin blocks tie.
    (1, 0.5, 0.0, None, None, 2, 5, 0, True),  # Raw tie is 0 as specified.
    (0, 0.5, 0.0, None, None, 2, 5, 0, True),
])
def test_correction_policy(inputs, monkeypatch, mwpm, q, margin, minimum_transitions,
                           minimum_ess, transitions, ess, expected, passed):
    monkeypatch.setattr(corrector, "decode_mwpm", lambda *args: MWPMResult(mwpm, (), 123))
    posterior = SimpleNamespace(posterior_p1=q, logical_transitions=transitions, logical_ess=ess,
                                prediction=mwpm)  # Deliberately ignore this tie-broken prediction.
    monkeypatch.setattr(PosteriorMCMCDecoder, "decode", lambda *args, **kwargs: posterior)
    decoder = corrector.MWPMPPosteriorCorrector(
        *inputs, correction_margin=margin, min_logical_transitions=minimum_transitions,
        min_logical_ess=minimum_ess,
    )
    result = decoder.decode([1], 10, burn_in=2, seed=5)
    assert result.final_prediction == result.prediction == expected
    assert result.mwpm_prediction == mwpm and result.posterior_prediction == int(q > 0.5)
    assert result.posterior_p1 == q and result.confidence == abs(q - 0.5)
    assert result.correction_attempted is (int(q > 0.5) != mwpm)
    assert result.correction_applied is (expected != mwpm)
    assert result.diagnostic_pass is passed
    assert result.posterior_result is posterior
    assert decoder.correction_margin == margin
    assert decoder.min_logical_transitions == minimum_transitions
    assert decoder.min_logical_ess == minimum_ess


@pytest.mark.parametrize("settings", [
    {"correction_margin": -0.1}, {"correction_margin": 0.6},
    {"correction_margin": np.nan}, {"correction_margin": np.inf},
    {"correction_margin": True}, {"correction_margin": None},
    {"min_logical_transitions": -1}, {"min_logical_transitions": 0.5},
    {"min_logical_transitions": True}, {"min_logical_ess": -1},
    {"min_logical_ess": np.nan}, {"min_logical_ess": np.inf}, {"min_logical_ess": True},
])
def test_invalid_fixed_thresholds(inputs, settings):
    config = {"correction_margin": 0.1, **settings}
    with pytest.raises(ValueError):
        corrector.MWPMPPosteriorCorrector(*inputs, **config)


def test_matching_cannot_change_posterior_chain(inputs, monkeypatch, tmp_path):
    graph, model = inputs
    syndrome = np.array([1], dtype=np.uint8)
    expected = PosteriorMCMCDecoder(graph, model).decode(syndrome, 15, burn_in=2, seed=42)
    expected_initial = sample_uniform_solution(build_incidence_matrix(graph), syndrome, np.random.default_rng(42))
    def matching(values, supplied_graph):
        assert supplied_graph is graph
        np.testing.assert_array_equal(values, syndrome)
        values[:] = 0
        return MWPMResult(1, (999999,), -123456.0)
    monkeypatch.setattr(corrector, "decode_mwpm", matching)
    path = tmp_path / "trace.npy"
    for margin in (0, 0.5):
        decoder = corrector.MWPMPPosteriorCorrector(graph, model, correction_margin=margin)
        result = decoder.decode(syndrome, 15, burn_in=2, seed=42, logical_trace_path=path)
        assert result.posterior_result.trace == expected.trace
        assert result.posterior_result.logical_parity_trace == expected.logical_parity_trace
        np.testing.assert_array_equal(result.posterior_result.initial_configuration, expected_initial)
        np.testing.assert_array_equal(np.load(path), expected.logical_parity_trace)
        np.testing.assert_array_equal(syndrome, [1])
    assert corrector.MWPMPosteriorCorrector is corrector.MWPMPPosteriorCorrector
