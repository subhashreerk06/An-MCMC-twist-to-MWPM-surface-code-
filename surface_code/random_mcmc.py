"""Metropolis sampling with independent uniform affine-solution proposals.

The target on syndrome-compatible edge vectors is proportional to exp(-W(E)),
where W is the sum of the graph's stored log-odds weights. Uniform proposals
have equal forward and reverse probabilities, leaving the acceptance ratio
exp(-delta_W). DEM component correlations are not enforced by this experiment.
"""

from dataclasses import dataclass
from operator import index

import numpy as np

from surface_code.decoding_graph import DecodingGraph
from surface_code.gf2 import build_incidence_matrix, sample_uniform_solution


@dataclass(frozen=True)
class TraceEntry:
    """One proposal: delta uses the pre-decision current weight.

    Iterations start at 1. current_weight and best_weight_so_far describe the
    state after the accept/reject decision; initialization is not a trace row.
    """

    iteration: int
    proposed_weight: float
    current_weight: float
    delta_weight: float
    accepted: bool
    best_weight_so_far: float


@dataclass(frozen=True)
class MCMCResult:
    """State snapshots and the L0 prediction from best_configuration alone."""

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
    trace: tuple[TraceEntry, ...]


def metropolis_accept(delta_weight: float, rng: np.random.Generator) -> bool:
    """Test log(U) < min(0, -delta_weight) without exponentiating.

    NumPy draws on [0, 1); redraw an exact zero to obtain U in (0, 1).
    A draw is consumed even for downhill and equal-weight proposals.
    """
    if np.isnan(delta_weight):
        raise ValueError("delta_weight must not be NaN.")
    uniform = rng.random()
    while uniform == 0:
        uniform = rng.random()
    return bool(np.log(uniform) < min(0.0, -delta_weight))


def decode_random_mcmc(
    syndrome,
    decoding_graph: DecodingGraph,
    iterations: int,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> MCMCResult:
    """Run the uniform-proposal experiment and return the best vector plus trace.

    Initialization and every proposal come directly from
    sample_uniform_solution(H, syndrome, rng). No decoder reference state or
    local moves are used. prediction is the parity of active edges' L0 flags
    in best_configuration, without voting over samples.
    The best_configuration field is the decoder's output;
    final_configuration may have higher weight due to accepted uphill moves.

    Supply either an external NumPy Generator or a seed, not both. Zero
    iterations returns the initial configuration and an acceptance rate of 0.
    Requires finite graph weights (0 < p_error < 1), since infinite energies
    make the specified delta-weight subtraction undefined. Negative finite
    weights are supported. Invalid or inconsistent syndromes raise ValueError.
    """
    if isinstance(iterations, (bool, np.bool_)):
        raise ValueError("iterations must be a nonnegative integer.")
    try:
        iterations = index(iterations)
    except TypeError as exc:
        raise ValueError("iterations must be a nonnegative integer.") from exc
    if iterations < 0:
        raise ValueError("iterations must be a nonnegative integer.")
    if rng is not None and seed is not None:
        raise ValueError("Supply either rng or seed, not both.")
    if rng is None:
        rng = np.random.default_rng(seed)

    weights = np.array([edge.weight for edge in decoding_graph.edges], dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("MCMC requires finite edge weights (0 < p_error < 1).")
    H = build_incidence_matrix(decoding_graph)
    current = sample_uniform_solution(H, syndrome, rng)
    current_weight = float(np.sum(current * weights))
    initial = current.copy()
    initial_weight = current_weight
    best = current.copy()
    best_weight = current_weight
    accepted_moves = 0
    trace = []

    for iteration in range(1, iterations + 1):
        proposed = sample_uniform_solution(H, syndrome, rng)
        proposed_weight = float(np.sum(proposed * weights))
        delta_weight = proposed_weight - current_weight
        accepted = metropolis_accept(delta_weight, rng)
        if accepted:
            current = proposed
            current_weight = proposed_weight
            accepted_moves += 1
        if current_weight < best_weight:
            best = current.copy()
            best_weight = current_weight
        trace.append(TraceEntry(
            iteration=iteration, proposed_weight=proposed_weight,
            current_weight=current_weight, delta_weight=delta_weight,
            accepted=accepted, best_weight_so_far=best_weight,
        ))

    prediction = sum(
        int(edge.logical_flip)
        for active, edge in zip(best, decoding_graph.edges) if active
    ) % 2
    return MCMCResult(
        prediction=prediction,
        initial_configuration=initial, initial_weight=initial_weight,
        final_configuration=current.copy(), final_weight=current_weight,
        best_configuration=best, best_weight=best_weight,
        accepted_moves=accepted_moves, rejected_moves=iterations - accepted_moves,
        acceptance_rate=accepted_moves / iterations if iterations else 0.0,
        trace=tuple(trace),
    )
