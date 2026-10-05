"""Uniform-start independent Metropolis-Hastings with a flow/uniform mixture.

The chain targets exp(-W(E)) on the affine syndrome solution space. The map
between z and E is bijective, so no Jacobian or multiplicity factor is needed.
The proposal correction uses the FULL mixture density for both states:
q_mix(z | s) = alpha q_theta(z | s) + (1-alpha) 2**(-nullity).
It never uses just the density of the component that generated a proposal.
Initialization uses only the existing uniform GF(2) sampler.
"""

from copy import deepcopy
from dataclasses import dataclass
from operator import index
from pathlib import Path

import numpy as np
import torch

from surface_code.decoding_graph import DecodingGraph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_dataset import _json_fingerprint, _matrix_fingerprint
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix, sample_uniform_solution


@dataclass(frozen=True)
class FlowTraceEntry:
    """One decision, numbered from 1; initialization is not a trace entry.

    log_q_mix_current is the PRE-decision density used in log_acceptance_ratio.
    current_weight and best_weight_so_far are POST-decision, as in the uniform
    baseline. The previous row's current_weight (or initial_weight) supplies
    the pre-decision energy. Best tracks only initial and accepted chain states.
    """

    iteration: int
    current_weight: float
    proposed_weight: float
    log_q_mix_current: float
    log_q_mix_proposed: float
    log_acceptance_ratio: float
    accepted: bool
    best_weight_so_far: float
    proposal_source: str

    @property
    def log_q_current(self) -> float:
        """Compatibility alias: now the full mixture density, before decision."""
        return self.log_q_mix_current

    @property
    def log_q_proposed(self) -> float:
        """Compatibility alias: now the full mixture density of the proposal."""
        return self.log_q_mix_proposed


@dataclass(frozen=True)
class FlowMCMCResult:
    """Lowest-weight visited error and independent initial/final snapshots.

    prediction = sum(edge.logical_flip for active edges in best_configuration)
    modulo 2, exactly the uniform baseline's L0 convention. Other observable
    bits are ignored. The final chain state can have higher weight. No voting
    over the chain is performed.

    Returned attributes include prediction, best_configuration, best_weight,
    initial_weight, final_weight, acceptance_rate, flow_acceptance_rate, and
    uniform_acceptance_rate. Component rates are accepted/proposed for that
    component, or zero when the component made no proposals.
    """

    prediction: int
    initial_configuration: np.ndarray
    initial_weight: float
    final_configuration: np.ndarray
    final_weight: float
    best_configuration: np.ndarray
    best_weight: float
    accepted_moves: int
    rejected_moves: int
    acceptance_rate: float
    flow_probability: float
    flow_proposals: int
    uniform_proposals: int
    flow_accepted_moves: int
    uniform_accepted_moves: int
    trace: tuple[FlowTraceEntry, ...]

    @property
    def flow_acceptance_rate(self) -> float:
        return self.flow_accepted_moves / self.flow_proposals if self.flow_proposals else 0.0

    @property
    def uniform_acceptance_rate(self) -> float:
        return self.uniform_accepted_moves / self.uniform_proposals if self.uniform_proposals else 0.0


def independent_metropolis_accept(log_ratio: float, rng: np.random.Generator) -> bool:
    """Use strict log(U) < min(0, log_ratio), redrawing exact zero uniforms."""
    if np.isnan(log_ratio):
        raise ValueError("log_ratio must not be NaN.")
    uniform = rng.random()
    while uniform == 0:
        uniform = rng.random()
    return bool(np.log(uniform) < min(0.0, log_ratio))


def _integer(value, name, minimum=0):
    try:
        if isinstance(value, (bool, np.bool_)) or index(value) < minimum:
            raise ValueError
        return int(index(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc


def _probability(value):
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("flow_probability must be finite and in [0, 1].") from exc
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("flow_probability must be finite and in [0, 1].")
    return probability


def log_mixture_probability(log_q_flow: float, z_dim: int, flow_probability: float) -> float:
    """Stable log(alpha * q_flow + (1-alpha) * 2**(-z_dim)).

    Handles alpha=0/1 without taking log(0). No exponentiation of log_q_flow
    or 2**(-z_dim) is needed, so very small probabilities do not underflow.
    A zero flow density (-inf) is allowed; any alpha < 1 still yields positive
    mixture mass for every configuration. Pure-flow decoding requires finite
    log densities, as before.
    """
    alpha = _probability(flow_probability)
    log_uniform = -_integer(z_dim, "z_dim") * np.log(2.0)
    if alpha == 0:
        return float(log_uniform)
    if np.isnan(log_q_flow) or log_q_flow > 0:
        raise ValueError("Flow log probability must be nonpositive and not NaN.")
    if alpha == 1:
        return float(log_q_flow)
    return float(np.logaddexp(np.log(alpha) + log_q_flow, np.log1p(-alpha) + log_uniform))


def _graph_fingerprints(graph):
    H = build_incidence_matrix(graph)
    N = AffineCoordinates(H).nullspace_basis()
    ordered_edges = [
        {"edge_id": edge.edge_id, "detector_endpoints": edge.detector_endpoints,
         "p_error": float(edge.p_error).hex(), "logical_mask": edge.logical_mask,
         "mechanism_id": edge.mechanism_id, "component_index": edge.component_index}
        for edge in graph.edges
    ]
    return {
        "edge_ordering_fingerprint": _json_fingerprint(ordered_edges),
        "h_nullspace_fingerprint": _json_fingerprint([_matrix_fingerprint(H), _matrix_fingerprint(N)]),
    }


class FlowProposalMCMCDecoder:
    """Independent flow/uniform mixture with the exact Hastings correction.

    Construct from an already compatible model or use from_checkpoint to verify
    graph/basis fingerprints automatically. A private CPU copy of the model is
    frozen (eval mode, requires_grad=False) and all decoding runs without
    gradients. The caller's model, mode, and parameters are not modified.
    CPU thread count is scoped to decode and restored afterward.

    Initialization is exactly sample_uniform_solution(H, syndrome, rng); the
    same NumPy stream then supplies component selection, sampling and acceptance
    draws. flow_probability defaults to 0.9; 1 selects pure flow and 0 selects
    pure uniform proposals. At these endpoints no component-selection draw is
    consumed. Pure uniform mode never calls the model's sample or log_prob.
    The component label is diagnostic only: all ratios use total mixture mass.
    Best-state tracking includes the initial state and accepted chain states,
    exactly as in the uniform baseline. A rejected proposal cannot become best,
    even if its weight is lower: it was evaluated but not visited by the chain.
    Equal weights retain the earlier best configuration.
    """

    def __init__(
        self, decoding_graph: DecodingGraph,
        model: ConditionalAutoregressiveBernoulli, *, threads: int = 1,
        flow_probability: float = 0.9,
    ):
        self._threads = _integer(threads, "threads", 1)
        self._flow_probability = _probability(flow_probability)
        self._graph = decoding_graph
        self._H = build_incidence_matrix(decoding_graph)
        self._coordinates = AffineCoordinates(self._H)
        self._weights = np.array([edge.weight for edge in decoding_graph.edges], dtype=float)
        if not np.isfinite(self._weights).all():
            raise ValueError("Decoding requires finite edge weights (0 < p_error < 1).")
        if (model.syndrome_dim, model.z_dim) != (self._H.shape[0], self._coordinates.nullity):
            raise ValueError("Model dimensions do not match the graph/basis.")
        self._model = deepcopy(model).to("cpu").eval().requires_grad_(False)
        for parameter in self._model.parameters():
            parameter.grad = None

    @classmethod
    def from_checkpoint(
        cls, decoding_graph: DecodingGraph, checkpoint: str | Path, *, threads: int = 1,
        flow_probability: float = 0.9,
    ):
        """Verify checkpoint identity against the actual graph and affine basis."""
        with torch.random.fork_rng(devices=[]):
            model = ConditionalAutoregressiveBernoulli.load_checkpoint(
                checkpoint, expected_fingerprints=_graph_fingerprints(decoding_graph), device="cpu",
            )
        return cls(decoding_graph, model, threads=threads, flow_probability=flow_probability)

    @property
    def flow_probability(self) -> float:
        return self._flow_probability

    def _weight(self, error):
        value = float(np.sum(error * self._weights))
        if not np.isfinite(value):
            raise ValueError("Configuration weight must be finite.")
        return value

    def _log_probability(self, z, syndrome):
        if self.flow_probability == 0:
            return log_mixture_probability(0.0, self._coordinates.nullity, 0.0)
        value = self._model.log_prob(z, syndrome)
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError("Model log_prob must return a scalar for one configuration.")
            value = value.item()
        value = float(value)
        if np.isnan(value) or value > 0 or (self.flow_probability == 1 and not np.isfinite(value)):
            raise ValueError("Model log_prob must be nonpositive; pure flow requires finite values (full support).")
        return log_mixture_probability(value, self._coordinates.nullity, self.flow_probability)

    @torch.no_grad()
    def decode(
        self, syndrome, iterations: int, *, rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> FlowMCMCResult:
        """Return best visited error, chain snapshots, and an auditable trace.

        Supply either rng or seed. Zero iterations returns the uniform initial
        state with no proposals and acceptance rate zero. Invalid/inconsistent
        syndromes raise ValueError through the existing GF(2) utilities.
        """
        iterations = _integer(iterations, "iterations")
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
        best, best_weight = current.copy(), current_weight
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
                if current_weight < best_weight:
                    best, best_weight = current.copy(), current_weight
                trace.append(FlowTraceEntry(
                    iteration=iteration, current_weight=current_weight,
                    proposed_weight=proposed_weight, log_q_mix_current=before_log_q,
                    log_q_mix_proposed=log_q_proposed, log_acceptance_ratio=log_ratio,
                    accepted=accepted, best_weight_so_far=best_weight,
                    proposal_source=proposal_source,
                ))
        finally:
            torch.set_num_threads(previous_threads)
        prediction = sum(
            int(edge.logical_flip) for active, edge in zip(best, self._graph.edges) if active
        ) % 2
        return FlowMCMCResult(
            prediction=prediction, initial_configuration=initial, initial_weight=initial_weight,
            final_configuration=current.copy(), final_weight=current_weight,
            best_configuration=best, best_weight=best_weight, accepted_moves=accepted_moves,
            rejected_moves=iterations - accepted_moves,
            acceptance_rate=accepted_moves / iterations if iterations else 0.0, trace=tuple(trace),
            flow_probability=self.flow_probability,
            flow_proposals=proposals_by_source["flow"], uniform_proposals=proposals_by_source["uniform"],
            flow_accepted_moves=accepted_by_source["flow"], uniform_accepted_moves=accepted_by_source["uniform"],
        )
