"""Fixed-threshold posterior corrections to independent MWPM predictions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surface_code.decoding_graph import DecodingGraph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_mcmc import _integer
from surface_code.mwpm_decoder import MWPMResult, decode_mwpm
from surface_code.posterior_mcmc import PosteriorMCMCDecoder, PosteriorMCMCResult


def _nonnegative_finite(value, name):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative.") from exc
    return number


@dataclass(frozen=True)
class MWPMPosteriorCorrectionResult:
    """Decision and independent decoder results for auditing.

    correction_attempted means raw posterior and MWPM predictions disagree,
    even when a gate blocks the correction. diagnostic_pass describes only
    the optional transition/ESS gates, independently of confidence. With no
    quality gates configured it is True. posterior_prediction is int(q > 0.5),
    which can differ from posterior_result.prediction on an exact tie.
    """

    final_prediction: int
    mwpm_prediction: int
    posterior_prediction: int
    posterior_p1: float
    correction_attempted: bool
    correction_applied: bool
    confidence: float
    diagnostic_pass: bool
    mwpm_result: MWPMResult
    posterior_result: PosteriorMCMCResult

    @property
    def prediction(self) -> int:
        return self.final_prediction


@dataclass(frozen=True)
class PosteriorCorrectionPolicy:
    """Fixed gates applied to already computed independent decoder results.

    Reusing these results allows a paired benchmark to compare the standalone
    posterior and correction layer on exactly the same retained chain.
    No minimum-weight configuration participates in this decision.
    """

    correction_margin: float
    min_logical_transitions: int | None = None
    min_logical_ess: float | None = None

    def __post_init__(self):
        margin = _nonnegative_finite(self.correction_margin, "correction_margin")
        if margin > 0.5:
            raise ValueError("correction_margin must be in [0, 0.5].")
        object.__setattr__(self, "correction_margin", margin)
        if self.min_logical_transitions is not None:
            object.__setattr__(self, "min_logical_transitions",
                               _integer(self.min_logical_transitions, "min_logical_transitions"))
        if self.min_logical_ess is not None:
            object.__setattr__(self, "min_logical_ess",
                               _nonnegative_finite(self.min_logical_ess, "min_logical_ess"))

    def apply(self, mwpm: MWPMResult, posterior: PosteriorMCMCResult) -> MWPMPosteriorCorrectionResult:
        """Apply the same policy used by the standalone corrector, without sampling."""
        q = posterior.posterior_p1
        raw_prediction = int(q > 0.5)
        confidence = abs(q - 0.5)
        transitions_pass = (
            self.min_logical_transitions is None
            or posterior.logical_transitions >= self.min_logical_transitions
        )
        ess_pass = self.min_logical_ess is None or (
            posterior.logical_ess is not None
            and np.isfinite(posterior.logical_ess)
            and posterior.logical_ess >= self.min_logical_ess
        )
        diagnostic_pass = bool(transitions_pass and ess_pass)
        attempted = raw_prediction != mwpm.prediction
        applied = bool(attempted and confidence >= self.correction_margin and diagnostic_pass)
        return MWPMPosteriorCorrectionResult(
            final_prediction=raw_prediction if applied else mwpm.prediction,
            mwpm_prediction=mwpm.prediction, posterior_prediction=raw_prediction,
            posterior_p1=q, correction_attempted=attempted, correction_applied=applied,
            confidence=confidence, diagnostic_pass=diagnostic_pass,
            mwpm_result=mwpm, posterior_result=posterior,
        )


class MWPMPPosteriorCorrector:
    """Override MWPM only on disagreement with sufficient confidence/quality.

    Thresholds are fixed constructor inputs, never fitted or adapted to shots.
    Choose them independently of the final test dataset. correction_margin is
    required and lies in [0, 0.5]; all gates use inclusive >= comparisons.
    Undefined/nonfinite ESS fails an enabled ESS gate, even a threshold of zero.

    The posterior chain starts independently at a random compatible state and
    receives no matching correction, weight, prediction, or threshold. Its
    retained states and MH kernel are unchanged. The raw decision follows the
    requested int(q > 0.5) convention: at q=0.5 it is 0. With margin=0 this can
    override an MWPM prediction of 1 if quality gates pass; a positive margin
    prevents any exact-tie override.
    """

    def __init__(
        self, decoding_graph: DecodingGraph, model: ConditionalAutoregressiveBernoulli, *,
        correction_margin: float, min_logical_transitions: int | None = None,
        min_logical_ess: float | None = None, flow_probability: float = 0.9,
        threads: int = 1,
    ):
        self._policy = PosteriorCorrectionPolicy(
            correction_margin, min_logical_transitions, min_logical_ess,
        )
        self._graph = decoding_graph
        self._posterior_decoder = PosteriorMCMCDecoder(
            decoding_graph, model, threads=threads, flow_probability=flow_probability,
        )

    @property
    def correction_margin(self) -> float:
        return self._policy.correction_margin

    @property
    def min_logical_transitions(self) -> int | None:
        return self._policy.min_logical_transitions

    @property
    def min_logical_ess(self) -> float | None:
        return self._policy.min_logical_ess

    def decode(
        self, syndrome, iterations: int, *, burn_in: int = 0,
        rng: np.random.Generator | None = None, seed: int | None = None,
        logical_trace_path: str | Path | None = None,
    ) -> MWPMPosteriorCorrectionResult:
        """Decode independently, then apply the fixed gates to the raw posterior."""
        syndrome = np.asarray(syndrome).copy()
        mwpm = decode_mwpm(syndrome.copy(), self._graph)
        posterior = self._posterior_decoder.decode(
            syndrome.copy(), iterations, burn_in=burn_in, rng=rng, seed=seed,
            logical_trace_path=logical_trace_path,
        )
        return self._policy.apply(mwpm, posterior)


# Also expose the conventional spelling without the extra P.
MWPMPosteriorCorrector = MWPMPPosteriorCorrector
