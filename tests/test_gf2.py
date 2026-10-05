"""Exact elimination, incidence semantics, and uniform affine sampling."""

from itertools import product

import numpy as np
import pytest
import stim

from surface_code.circuit import (
    build_surface_code,
    get_detector_error_model,
    sample_shots,
)
from surface_code.decoding_graph import build_decoding_graph
from surface_code.gf2 import (
    build_incidence_matrix,
    gf2_rref,
    sample_uniform_solution,
    solve_affine_system,
)


def enumerate_solutions(H, s):
    return {
        bits for bits in product((0, 1), repeat=H.shape[1])
        if np.array_equal((H @ np.array(bits, dtype=np.uint8)) % 2, s)
    }


def test_incidence_boundary_parallel_and_logical_edges():
    graph = build_decoding_graph(stim.DetectorErrorModel("""
        error(0.1) D1 D2
        error(0.2) D0 L0
        error(0.3) D2 D1 L0
        error(0.1) L0
        error(0.1) D0 D0
        detector D3
    """))
    H = build_incidence_matrix(graph)
    assert H.dtype == np.uint8
    assert H.shape == (4, 5)
    np.testing.assert_array_equal(H, [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    # NetworkX groups edges by node; H must instead follow the explicit edge IDs.
    assert list(graph.graph.edges(keys=True))[0][2] != 0
    assert graph.boundary_node not in graph.detector_nodes


def test_empty_incidence_matrix():
    graph = build_decoding_graph(stim.DetectorErrorModel())
    assert build_incidence_matrix(graph).shape == (0, 0)


def test_rref_requires_row_swap_and_cancels_dependent_rows():
    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    original = matrix.copy()
    reduced, pivots = gf2_rref(matrix)
    np.testing.assert_array_equal(reduced, [[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    assert pivots == (0, 1)
    np.testing.assert_array_equal(matrix, original)
    again, again_pivots = gf2_rref(reduced)
    np.testing.assert_array_equal(again, reduced)
    assert again_pivots == pivots


def test_affine_solution_with_nonleading_pivots():
    H = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 1, 1]])
    result = solve_affine_system(H, [1, 1, 0])
    assert result.consistent
    assert result.pivot_variables == (1, 3)
    assert result.free_variables == (0, 2)
    assert result.rank == result.nullity == 2
    np.testing.assert_array_equal(result.particular_solution, [0, 1, 0, 1])


def test_inconsistent_system():
    H = [[1, 1], [1, 1]]
    result = solve_affine_system(H, [0, 1])
    assert not result.consistent
    assert result.particular_solution is None
    assert result.pivot_variables == (0,)
    assert result.free_variables == (1,)
    assert result.nullity == 1
    with pytest.raises(ValueError, match="inconsistent"):
        sample_uniform_solution(H, [0, 1], np.random.default_rng(0))


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (3, 4)])
def test_zero_and_empty_systems(shape):
    H = np.zeros(shape, dtype=np.uint8)
    s = np.zeros(shape[0], dtype=np.uint8)
    reduced, pivots = gf2_rref(H)
    assert reduced.shape == shape
    assert pivots == ()
    solution = solve_affine_system(H, s)
    assert solution.consistent
    assert solution.nullity == shape[1]
    e = sample_uniform_solution(H, s, np.random.default_rng(0))
    assert e.shape == (shape[1],)
    assert e.dtype == np.uint8
    np.testing.assert_array_equal((H @ e) % 2, s)


def test_no_variables_with_nonzero_rhs():
    result = solve_affine_system(np.zeros((2, 0)), [0, 1])
    assert not result.consistent
    assert result.rank == result.nullity == 0


def test_unique_solution():
    H = np.eye(4, dtype=np.uint8)
    s = [1, 0, 1, 1]
    result = solve_affine_system(H, s)
    assert result.nullity == 0
    for seed in range(4):
        np.testing.assert_array_equal(
            sample_uniform_solution(H, s, np.random.default_rng(seed)), s
        )


def test_tiny_matrix_sampling_matches_enumerated_solutions():
    H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    valid = enumerate_solutions(H, s)
    assert len(valid) == 4
    rng = np.random.default_rng(123)
    counts = {bits: 0 for bits in valid}
    for _ in range(2000):
        e = sample_uniform_solution(H, s, rng)
        assert tuple(e) in valid
        counts[tuple(e)] += 1
    assert all(400 < count < 600 for count in counts.values())


def test_every_free_assignment_maps_to_exactly_one_solution():
    H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    free = list(solve_affine_system(H, s).free_variables)

    class FixedBits:
        def __init__(self, bits):
            self.bits = bits

        def integers(self, low, high, size, dtype):
            assert (low, high, size, dtype) == (0, 2, 2, np.uint8)
            return np.array(self.bits, dtype=dtype)

    sampled = set()
    for bits in product((0, 1), repeat=2):
        e = sample_uniform_solution(H, s, FixedBits(bits))
        np.testing.assert_array_equal(e[free], bits)
        sampled.add(tuple(e))
    assert sampled == enumerate_solutions(H, s)


def test_all_two_by_three_binary_systems_against_enumeration():
    rng = np.random.default_rng(11)
    for entries in product((0, 1), repeat=6):
        H = np.array(entries, dtype=np.uint8).reshape(2, 3)
        for s in product((0, 1), repeat=2):
            valid = enumerate_solutions(H, s)
            result = solve_affine_system(H, s)
            assert result.consistent == bool(valid)
            assert result.rank + result.nullity == 3
            if valid:
                assert tuple(result.particular_solution) in valid
                assert len(valid) == 2 ** result.nullity
                assert tuple(sample_uniform_solution(H, s, rng)) in valid
            else:
                assert result.particular_solution is None


def test_seed_reproducibility_and_inputs_unchanged():
    H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
    s = np.array([1, 0], dtype=bool)
    original_H, original_s = H.copy(), s.copy()
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(10):
        np.testing.assert_array_equal(
            sample_uniform_solution(H, s, a), sample_uniform_solution(H, s, b)
        )
    np.testing.assert_array_equal(H, original_H)
    np.testing.assert_array_equal(s, original_s)


@pytest.mark.parametrize("bad", [[[2]], [[-1]], [[0.5]], [[np.nan]], [[np.inf]], [["1"]]])
def test_reject_nonbinary_matrices(bad):
    with pytest.raises(ValueError, match="binary"):
        gf2_rref(bad)
    with pytest.raises(ValueError, match="binary"):
        solve_affine_system(bad, [0])


@pytest.mark.parametrize("H,s", [
    ([1, 0], [0]), ([[1, 0]], [[0]]), ([[1, 0]], [0, 1]),
    ([[1, 0]], [2]), ([[1, 0]], [np.nan]),
])
def test_invalid_system_inputs(H, s):
    with pytest.raises(ValueError):
        solve_affine_system(H, s)
    with pytest.raises(ValueError):
        sample_uniform_solution(H, s, np.random.default_rng(0))


def test_wide_matrix_parity_is_exact_past_uint8_range():
    H = np.ones((1, 1025), dtype=np.uint8)
    e = sample_uniform_solution(H, [1], np.random.default_rng(17))
    assert int(e.astype(np.int64).sum()) % 2 == 1


def test_surface_code_samples_have_affine_solutions():
    circuit = build_surface_code(3, 3, 0.005)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    H = build_incidence_matrix(graph)
    assert H.shape == (24, len(graph.edges))
    detectors, _ = sample_shots(circuit, 12, seed=42)
    rng = np.random.default_rng(73)
    for syndrome in detectors:
        e = sample_uniform_solution(H, syndrome, rng)
        np.testing.assert_array_equal((H @ e) % 2, syndrome)
