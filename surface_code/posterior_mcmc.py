"""Posterior logical-class decoding using ordinary retained MH frequencies.

The flow/uniform proposal and target exp(-W(E)) match flow_mcmc. Samples
are never weighted by exp(-W) again. MWPM is consulted only after sampling,
only when the two retained logical-class counts are exactly equal.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from surface_code.decoding_graph import DecodingGraph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_mcmc import (
    FlowProposalMCMCDecoder, FlowTraceEntry, _integer,
    independent_metropolis_accept,
)
from surface_code.gf2 import sample_uniform_solution
from surface_code.mwpm_decoder import decode_mwpm


@dataclass(frozen=True)
class LogicalTraceDiagnostics:
    """Diagnostics of the full retained sequence, with repeats preserved.

    autocorrelation[k] is the centered, biased sample autocorrelation at lag k.
    For a constant trace it is undefined (None at every lag), as is logical_ess;
    status='constant_trace' does not imply that the chain has mixed.
    """

    logical_transitions: int
    transitions_0_to_1: int
    transitions_1_to_0: int
    posterior_p0: float
    posterior_p1: float
    autocorrelation: tuple[float | None, ...]
    logical_ess: float | None
    ess_truncation_lag: int
    status: str


def logical_trace_diagnostics(trace, *, max_lag: int = 100) -> LogicalTraceDiagnostics:
    """Estimate logical mixing without thinning or deduplicating the trace.

    Evaluate lags 0..min(max_lag, n-1), defaulting to at most 100 nonzero
    lags. rho[k] = sum((L[t]-mean)*(L[t+k]-mean)) / sum((L-mean)**2).
    Estimate tau = 1 + 2*sum(rho[1:m+1]), stopping BEFORE the first nonpositive
    correlation. ESS = n/tau is clipped to [1, n]; negative correlations never
    increase ESS above n. Later positive correlations are not restarted.
    This is an approximate single-chain diagnostic, not proof of convergence.
    If all evaluated correlations stay positive, status='lag_limit_reached'
    flags that unmeasured positive tails could make ESS too optimistic.
    Constant traces return None for ESS/ACF, avoiding a misleading ESS=n.
    """
    max_lag = _integer(max_lag, "max_lag", 1)
    values = np.asarray(trace)
    if values.ndim != 1 or values.size == 0 or not np.all((values == 0) | (values == 1)):
        raise ValueError("trace must be a nonempty one-dimensional binary sequence.")
    n = len(values)
    lag_limit = min(max_lag, n - 1)
    up = int(np.count_nonzero((values[:-1] == 0) & (values[1:] == 1)))
    down = int(np.count_nonzero((values[:-1] == 1) & (values[1:] == 0)))
    p1 = float(np.mean(values))
    centered = values.astype(float) - p1
    denominator = float(np.sum(centered * centered))
    if denominator == 0:
        return LogicalTraceDiagnostics(
            up + down, up, down, 1 - p1, p1,
            (None,) * (lag_limit + 1), None, 0, "constant_trace",
        )
    correlations = (1.0,) + tuple(
        float(np.sum(centered[:-lag] * centered[lag:]) / denominator)
        for lag in range(1, lag_limit + 1)
    )
    tau, truncation_lag, status = 1.0, 0, "lag_limit_reached"
    for lag, correlation in enumerate(correlations[1:], 1):
        if correlation <= 0:
            status = "ok"
            break
        tau += 2 * correlation
        truncation_lag = lag
    return LogicalTraceDiagnostics(
        up + down, up, down, 1 - p1, p1, correlations,
        float(np.clip(n / tau, 1, n)), truncation_lag, status,
    )


@dataclass(frozen=True)
class PosteriorMCMCResult:
    """Posterior counts and chain diagnostics.

    logical_parity_trace contains every post-decision parity at iterations
    burn_in+1 through iterations, including repeats from rejection and self
    proposals. Initialization is not retained. logical_class_transitions counts
    changes between adjacent entries of this retained trace (no burn-in boundary).
    Acceptance/component counts cover all iterations, including burn-in.
    minimum_weight_seen covers initialization and visited chain states only,
    excluding rejected proposals. It does not determine the prediction.
    """

    prediction: int
    posterior_p0: float
    posterior_p1: float
    retained_samples: int
    burn_in: int
    acceptance_rate: float
    flow_proposals: int
    flow_accepted: int
    uniform_proposals: int
    uniform_accepted: int
    logical_0_samples: int
    logical_1_samples: int
    logical_class_transitions: int
    initial_weight: float
    minimum_weight_seen: float
    mean_retained_weight: float
    logical_parity_trace: tuple[int, ...]
    initial_configuration: np.ndarray
    final_configuration: np.ndarray
    final_weight: float
    accepted_moves: int
    rejected_moves: int
    flow_probability: float
    trace: tuple[FlowTraceEntry, ...]
    logical_diagnostics: LogicalTraceDiagnostics

    @property
    def logical_transitions(self) -> int:
        """Total 0->1 plus 1->0 transitions; see diagnostics for each direction."""
        return self.logical_diagnostics.logical_transitions

    @property
    def logical_ess(self) -> float | None:
        return self.logical_diagnostics.logical_ess

    @property
    def logical_autocorrelation(self) -> tuple[float | None, ...]:
        return self.logical_diagnostics.autocorrelation

    @property
    def logical_diagnostic_status(self) -> str:
        return self.logical_diagnostics.status

    def save_logical_trace(self, path: str | Path) -> None:
        """Save the retained parity trace as a NumPy .npy array of uint8 bits."""
        with Path(path).open("wb") as output:
            np.save(output, np.asarray(self.logical_parity_trace, dtype=np.uint8), allow_pickle=False)


class PosteriorMCMCDecoder(FlowProposalMCMCDecoder):
    """Use posterior majority instead of the minimum-weight visited state.

    Inherits frozen model setup, graph/checkpoint validation, full mixture
    density evaluation, and from_checkpoint from FlowProposalMCMCDecoder.
    The sampling loop uses the same random draws and Hastings decisions as
    that decoder; counting retained states consumes no random draws.
    """

    @torch.no_grad()
    def decode(
        self, syndrome, iterations: int, *, burn_in: int = 0,
        rng: np.random.Generator | None = None, seed: int | None = None,
        logical_trace_path: str | Path | None = None,
    ) -> PosteriorMCMCResult:
        """Sample iterations total steps, discarding the first burn_in steps.

        Requires 0 <= burn_in < iterations so posterior frequencies exist.
        Supply either rng or seed. The trace is always saved in the result;
        logical_trace_path optionally persists it as a NumPy .npy file.
        """
        iterations = _integer(iterations, "iterations", 1)
        burn_in = _integer(burn_in, "burn_in")
        if burn_in >= iterations:
            raise ValueError("burn_in must be less than iterations (retain at least one sample).")
        if rng is not None and seed is not None:
            raise ValueError("Supply either rng or seed, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)
        # No model call or other random draw precedes uniform initialization.
        current = sample_uniform_solution(self._H, syndrome, rng)
        syndrome = np.asarray(syndrome, dtype=np.uint8).copy()
        z_current = self._coordinates.error_to_coordinates(syndrome, current)
        current_weight = self._weight(current)
        initial, initial_weight = current.copy(), current_weight
        minimum_weight = current_weight
        logical_trace, retained_weights = [], []
        accepted_moves, trace = 0, []
        proposals_by_source = {"flow": 0, "uniform": 0}
        accepted_by_source = {"flow": 0, "uniform": 0}
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._threads)
            log_q_current = self._log_probability(z_current, syndrome)
            for iteration in range(1, iterations + 1):
                use_flow = self.flow_probability == 1 or (
                    self.flow_probability > 0 and rng.random() < self.flow_probability
                )
                proposal_source = "flow" if use_flow else "uniform"
                proposals_by_source[proposal_source] += 1
                if use_flow:
                    z_proposed = self._model.sample(syndrome, rng)
                    if isinstance(z_proposed, torch.Tensor):
                        z_proposed = z_proposed.detach().cpu().numpy()
                else:
                    uniform_error = sample_uniform_solution(self._H, syndrome, rng)
                    z_proposed = self._coordinates.error_to_coordinates(syndrome, uniform_error)
                proposed = self._coordinates.coordinates_to_error(syndrome, z_proposed)
                if not np.array_equal((self._H @ proposed) % 2, syndrome):
                    raise RuntimeError("Proposed configuration violates H e = s mod 2.")
                z_proposed = np.asarray(z_proposed, dtype=np.uint8).copy()
                proposed_weight = self._weight(proposed)
                log_q_proposed = self._log_probability(z_proposed, syndrome)
                before_log_q = log_q_current
                log_ratio = current_weight - proposed_weight + (log_q_current - log_q_proposed)
                accepted = independent_metropolis_accept(log_ratio, rng)
                if accepted:
                    z_current, current = z_proposed, proposed
                    current_weight, log_q_current = proposed_weight, log_q_proposed
                    accepted_moves += 1
                    accepted_by_source[proposal_source] += 1
                minimum_weight = min(minimum_weight, current_weight)
                if iteration > burn_in:
                    logical_trace.append(sum(
                        int(edge.logical_flip)
                        for active, edge in zip(current, self._graph.edges) if active
                    ) % 2)
                    retained_weights.append(current_weight)
                trace.append(FlowTraceEntry(
                    iteration=iteration, current_weight=current_weight,
                    proposed_weight=proposed_weight, log_q_mix_current=before_log_q,
                    log_q_mix_proposed=log_q_proposed, log_acceptance_ratio=log_ratio,
                    accepted=accepted, best_weight_so_far=minimum_weight,
                    proposal_source=proposal_source,
                ))
        finally:
            torch.set_num_threads(previous_threads)
        retained_samples = len(logical_trace)
        logical_1_samples = sum(logical_trace)
        logical_0_samples = retained_samples - logical_1_samples
        diagnostics = logical_trace_diagnostics(logical_trace)
        posterior_p1 = diagnostics.posterior_p1
        if logical_1_samples == logical_0_samples:
            prediction = decode_mwpm(syndrome, self._graph).prediction
        else:
            prediction = int(logical_1_samples > logical_0_samples)
        result = PosteriorMCMCResult(
            prediction=prediction, posterior_p0=1.0 - posterior_p1,
            posterior_p1=posterior_p1, retained_samples=retained_samples,
            burn_in=burn_in, acceptance_rate=accepted_moves / iterations,
            flow_proposals=proposals_by_source["flow"],
            flow_accepted=accepted_by_source["flow"],
            uniform_proposals=proposals_by_source["uniform"],
            uniform_accepted=accepted_by_source["uniform"],
            logical_0_samples=logical_0_samples, logical_1_samples=logical_1_samples,
            logical_class_transitions=diagnostics.logical_transitions,
            initial_weight=initial_weight, minimum_weight_seen=minimum_weight,
            mean_retained_weight=float(np.mean(retained_weights)),
            logical_parity_trace=tuple(logical_trace),
            initial_configuration=initial, final_configuration=current.copy(),
            final_weight=current_weight, accepted_moves=accepted_moves,
            rejected_moves=iterations - accepted_moves,
            flow_probability=self.flow_probability, trace=tuple(trace),
            logical_diagnostics=diagnostics,
        )
        if logical_trace_path is not None:
            result.save_logical_trace(logical_trace_path)
        return result


def posterior_mcmc_decode(
    syndrome, decoding_graph: DecodingGraph,
    model: ConditionalAutoregressiveBernoulli, iterations: int, *,
    burn_in: int = 0, flow_probability: float = 0.9, threads: int = 1,
    rng: np.random.Generator | None = None, seed: int | None = None,
    logical_trace_path: str | Path | None = None,
) -> PosteriorMCMCResult:
    """Convenience wrapper; reuse a decoder instance when decoding many shots."""
    decoder = PosteriorMCMCDecoder(
        decoding_graph, model, threads=threads, flow_probability=flow_probability,
    )
    return decoder.decode(
        syndrome, iterations, burn_in=burn_in, rng=rng, seed=seed,
        logical_trace_path=logical_trace_path,
    )
