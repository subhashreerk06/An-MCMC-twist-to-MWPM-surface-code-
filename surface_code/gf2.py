"""Detector incidence matrices and direct affine linear algebra over GF(2).

All inputs must contain binary entries. Outputs use uint8 and inputs are never
modified. Sampling is uniform over syndrome-compatible edge vectors, not over
the probability-weighted DEM distribution. Correlated DEM component constraints
are not part of H e = s.
"""

from dataclasses import dataclass

import numpy as np

from surface_code.decoding_graph import DecodingGraph


def build_incidence_matrix(decoding_graph: DecodingGraph) -> np.ndarray:
    """Return H with detector-only rows and columns in ``edges`` order.

    Parallel edges have separate columns. Boundary edges have a single 1 and
    pure logical errors have zero columns. The virtual boundary has no row.
    """
    rows = {node: i for i, node in enumerate(decoding_graph.detector_nodes)}
    matrix = np.zeros(
        (len(rows), len(decoding_graph.edges)), dtype=np.uint8
    )
    for column, edge in enumerate(decoding_graph.edges):
        for detector in edge.detector_endpoints:
            matrix[rows[detector], column] ^= 1
    return matrix


def _binary_array(value, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary entries (0 or 1).")
    return array.astype(np.uint8, copy=True)


def gf2_rref(matrix) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return reduced row echelon form and pivot-column indices over GF(2).

    Scan columns left to right and use the first available pivot row. Row
    addition is XOR; no floating-point arithmetic or division is used.
    """
    reduced = _binary_array(matrix, 2, "matrix")
    pivots = []
    pivot_row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if not candidates.size:
            continue
        source = pivot_row + int(candidates[0])
        reduced[[pivot_row, source]] = reduced[[source, pivot_row]]
        other_rows = np.flatnonzero(reduced[:, column])
        other_rows = other_rows[other_rows != pivot_row]
        reduced[other_rows] ^= reduced[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == reduced.shape[0]:
            break
    return reduced, tuple(pivots)


@dataclass(frozen=True)
class AffineSystemSolution:
    """Reduced equations and a particular solution with all free bits zero.

    particular_solution is None when inconsistent. Pivot/free variables always
    refer to columns of H, never the augmented right-hand-side column. Nullity
    describes H even for an inconsistent system; a consistent system has exactly
    2**nullity solutions.
    """

    consistent: bool
    particular_solution: np.ndarray | None
    pivot_variables: tuple[int, ...]
    free_variables: tuple[int, ...]
    reduced_matrix: np.ndarray
    reduced_rhs: np.ndarray

    @property
    def nullity(self) -> int:
        return len(self.free_variables)

    @property
    def rank(self) -> int:
        return len(self.pivot_variables)


def solve_affine_system(H, s) -> AffineSystemSolution:
    """Solve H e = s mod 2 by reducing the augmented matrix [H | s].

    Report inconsistency as ``consistent=False`` rather than throwing; malformed
    inputs raise ValueError. The reduced equations can be reused to assign any
    free bits and determine the corresponding pivot bits.
    """
    matrix = _binary_array(H, 2, "H")
    syndrome = _binary_array(s, 1, "s")
    if syndrome.shape != (matrix.shape[0],):
        raise ValueError("s must have one entry per row of H.")
    num_variables = matrix.shape[1]
    augmented = np.column_stack((matrix, syndrome))
    reduced, augmented_pivots = gf2_rref(augmented)
    pivots = tuple(p for p in augmented_pivots if p < num_variables)
    pivot_set = set(pivots)
    free = tuple(i for i in range(num_variables) if i not in pivot_set)
    consistent = num_variables not in augmented_pivots
    particular = None
    if consistent:
        particular = np.zeros(num_variables, dtype=np.uint8)
        particular[list(pivots)] = reduced[:len(pivots), -1]
    return AffineSystemSolution(
        consistent=consistent, particular_solution=particular,
        pivot_variables=pivots, free_variables=free,
        reduced_matrix=reduced[:, :-1].copy(), reduced_rhs=reduced[:, -1].copy(),
    )


class AffineCoordinates:
    """Reusable binary coordinates for a fixed H: e = e_p(s) + N z mod 2.

    Elimination is performed once, without probability or logical information.
    Coordinate j is the error bit at free_columns[j], in ascending H-column
    order. N has shape (number of columns of H, nullity), with an identity
    submatrix on the free rows. The particular solution has all free bits zero.
    Inconsistent syndromes and invalid binary vectors raise ValueError.

    Inputs are copied; returned arrays can be modified without changing this
    representation. This utility does not sample or propose configurations.
    """

    def __init__(self, H):
        self._matrix = _binary_array(H, 2, "H")
        rows, columns = self._matrix.shape
        # Carry the identity through row reduction to retain a transform T.
        # Pivots in the identity block only operate after all H pivots;
        # the resulting blocks still satisfy R = T H over GF(2).
        reduced, pivots = gf2_rref(np.column_stack((
            self._matrix, np.eye(rows, dtype=np.uint8),
        )))
        self._pivot_columns = tuple(p for p in pivots if p < columns)
        pivot_set = set(self._pivot_columns)
        self._free_columns = tuple(i for i in range(columns) if i not in pivot_set)
        self._transform = reduced[:, columns:].copy()
        self._basis = np.zeros((columns, self.nullity), dtype=np.uint8)
        self._basis[list(self.free_columns)] = np.eye(self.nullity, dtype=np.uint8)
        self._basis[list(self.pivot_columns)] = reduced[
            :self.rank, list(self.free_columns)
        ]

    @property
    def pivot_columns(self) -> tuple[int, ...]:
        return self._pivot_columns

    @property
    def free_columns(self) -> tuple[int, ...]:
        return self._free_columns

    @property
    def rank(self) -> int:
        return len(self.pivot_columns)

    @property
    def nullity(self) -> int:
        return len(self.free_columns)

    def _syndrome(self, s) -> np.ndarray:
        syndrome = _binary_array(s, 1, "s")
        if syndrome.shape != (self._matrix.shape[0],):
            raise ValueError("s must have one entry per row of H.")
        return syndrome

    def particular_solution(self, s) -> np.ndarray:
        """Return the deterministic elimination solution with free bits zero."""
        rhs = (self._transform @ self._syndrome(s)) % 2
        if np.any(rhs[self.rank:]):
            raise ValueError("The GF(2) system is inconsistent; no solution exists.")
        particular = np.zeros(self._matrix.shape[1], dtype=np.uint8)
        particular[list(self.pivot_columns)] = rhs[:self.rank]
        return particular

    def nullspace_basis(self) -> np.ndarray:
        """Return a copy of the fixed basis N, with H N = 0 mod 2."""
        return self._basis.copy()

    def coordinates_to_error(self, s, z) -> np.ndarray:
        """Map binary coordinates z to a syndrome-compatible error vector."""
        coordinates = _binary_array(z, 1, "z")
        if coordinates.shape != (self.nullity,):
            raise ValueError("z must have one entry per free column of H.")
        particular = self.particular_solution(s)
        # uint8 wraparound preserves parity, even for wide matrices.
        return particular ^ ((self._basis @ coordinates) % 2)

    def error_to_coordinates(self, s, e) -> np.ndarray:
        """Invert the representation, rejecting errors that do not satisfy H e = s."""
        syndrome = self._syndrome(s)
        error = _binary_array(e, 1, "e")
        if error.shape != (self._matrix.shape[1],):
            raise ValueError("e must have one entry per column of H.")
        if not np.array_equal((self._matrix @ error) % 2, syndrome):
            raise ValueError("e does not satisfy H e = s mod 2.")
        return error[list(self.free_columns)].copy()


def sample_uniform_solution(H, s, rng: np.random.Generator) -> np.ndarray:
    """Sample directly from the affine solution set using independent free bits.

    Every free-bit assignment determines exactly one full solution, so uniform
    free bits give uniform solutions. Uses only the supplied generator; passing
    identically seeded generators with the same calls reproduces the results.
    Raises ValueError for an inconsistent system. Verifies H e = s exactly
    before returning, including when the system has no rows or no columns.
    """
    matrix = _binary_array(H, 2, "H")
    syndrome = _binary_array(s, 1, "s")
    solution = solve_affine_system(matrix, syndrome)
    if not solution.consistent:
        raise ValueError("The GF(2) system is inconsistent; no solution exists.")
    vector = np.zeros(matrix.shape[1], dtype=np.uint8)
    free = list(solution.free_variables)
    vector[free] = rng.integers(0, 2, size=solution.nullity, dtype=np.uint8)
    for row, pivot in enumerate(solution.pivot_variables):
        vector[pivot] = (
            int(solution.reduced_rhs[row])
            ^ (int(solution.reduced_matrix[row, free] @ vector[free]) % 2)
        )
    # uint8 accumulation can wrap modulo 256, which preserves parity exactly.
    if not np.array_equal((matrix @ vector) % 2, syndrome):
        raise RuntimeError("Internal GF(2) sampling error: H e does not equal s.")
    return vector


if __name__ == "__main__":
    from textwrap import fill

    from surface_code.circuit import (
        build_surface_code,
        get_detector_error_model,
        sample_shots,
    )
    from surface_code.decoding_graph import build_decoding_graph

    circuit = build_surface_code(distance=3, rounds=3, p=0.005)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    H = build_incidence_matrix(graph)
    measured, _ = sample_shots(circuit, shots=100, seed=42)
    # Prefer a nonzero measured syndrome to make the constraint visible.
    nonzero_shots = np.flatnonzero(measured.any(axis=1))
    shot_index = int(nonzero_shots[0]) if nonzero_shots.size else 0
    syndrome = measured[shot_index].astype(np.uint8)
    weights = np.array([edge.weight for edge in graph.edges])

    print("Surface code: distance=3, rounds=3, p=0.005")
    print(f"Measured syndrome: shot {shot_index} of 100, sampler seed=42")
    print(f"H shape: {H.shape}; boundary excluded from syndrome rows")
    print("Uniform affine samples; W(E) is the additive component-edge weight.")
    configurations = set()
    summary = []
    for seed in range(10):
        e = sample_uniform_solution(H, syndrome, np.random.default_rng(seed))
        active_ids = [graph.edges[i].edge_id for i in np.flatnonzero(e)]
        actual = (H @ e) % 2
        valid = bool(np.array_equal(actual, syndrome))
        weight = float(np.sum(e * weights))
        configurations.add(e.tobytes())
        summary.append((seed, len(active_ids), weight, valid))
        print(f"\nseed = {seed}")
        print(f"number of active edges = {len(active_ids)}")
        print(f"total edge IDs = {len(graph.edges)} (0..{len(graph.edges) - 1})")
        print(fill(f"active edge IDs = {active_ids}", width=100, subsequent_indent="  "))
        print(f"H e mod 2          = {actual.tolist()}")
        print(f"requested syndrome = {syndrome.tolist()}")
        print(f"valid = {valid}")
        print(f"W(E) = {weight:.9f}")

    print("\nComparison (all configurations target the same measured syndrome):")
    print("seed  active edges          W(E)  valid")
    for seed, count, weight, valid in summary:
        print(f"{seed:4d}  {count:12d}  {weight:12.6f}  {valid}")
    print(f"Distinct configurations: {len(configurations)}/10")
    print(f"Distinct total weights: {len({row[2] for row in summary})}/10")
    print(f"All syndrome checks valid: {all(row[3] for row in summary)}")
