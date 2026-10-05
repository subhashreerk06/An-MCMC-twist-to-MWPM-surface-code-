"""Fixed null-space coordinates, exact inverses, and input validation."""

from itertools import product

import numpy as np
import pytest

from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix, solve_affine_system


def test_known_basis_and_coordinate_order():
    H = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 1, 1]])
    space = AffineCoordinates(H)
    assert space.pivot_columns == (1, 3)
    assert space.free_columns == (0, 2)
    assert space.rank == space.nullity == 2
    np.testing.assert_array_equal(space.nullspace_basis(), [[1, 0], [0, 1], [0, 1], [0, 0]])
    np.testing.assert_array_equal(space.particular_solution([1, 1, 0]), [0, 1, 0, 1])
    np.testing.assert_array_equal(space.coordinates_to_error([1, 1, 0], [1, 1]), [1, 0, 1, 1])


def test_exhaustive_small_systems():
    for entries in product((0, 1), repeat=6):
        H = np.array(entries, dtype=np.uint8).reshape(2, 3)
        space = AffineCoordinates(H)
        N = space.nullspace_basis()
        np.testing.assert_array_equal(H @ N % 2, np.zeros((2, space.nullity)))
        for s in product((0, 1), repeat=2):
            expected = {
                e for e in product((0, 1), repeat=3)
                if np.array_equal(H @ np.array(e) % 2, s)
            }
            if not expected:
                with pytest.raises(ValueError, match="inconsistent"):
                    space.particular_solution(s)
                with pytest.raises(ValueError, match="inconsistent"):
                    space.coordinates_to_error(s, np.zeros(space.nullity))
                continue
            np.testing.assert_array_equal(
                space.particular_solution(s), solve_affine_system(H, s).particular_solution
            )
            actual = set()
            for z in product((0, 1), repeat=space.nullity):
                e = space.coordinates_to_error(s, z)
                np.testing.assert_array_equal(space.error_to_coordinates(s, e), z)
                actual.add(tuple(e))
            assert actual == expected


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (5, 0), (8, 12), (12, 8), (24, 60), (1, 1025)])
def test_many_random_syndromes_and_coordinates(shape):
    rng = np.random.default_rng(912)
    for _ in range(4):
        H = rng.integers(0, 2, size=shape, dtype=np.uint8)
        space = AffineCoordinates(H)
        N = space.nullspace_basis()
        assert N.dtype == np.uint8
        assert N.shape == (shape[1], space.nullity)
        np.testing.assert_array_equal(H @ N % 2, np.zeros((shape[0], space.nullity)))
        np.testing.assert_array_equal(N[list(space.free_columns)], np.eye(space.nullity))
        np.testing.assert_array_equal(AffineCoordinates(H).nullspace_basis(), N)
        for _ in range(20):
            original_error = rng.integers(0, 2, size=shape[1], dtype=np.uint8)
            s = H @ original_error % 2
            particular = space.particular_solution(s)
            np.testing.assert_array_equal(particular, solve_affine_system(H, s).particular_solution)
            np.testing.assert_array_equal(
                space.coordinates_to_error(s, space.error_to_coordinates(s, original_error)),
                original_error,
            )
            for _ in range(5):
                z = rng.integers(0, 2, size=space.nullity, dtype=np.uint8)
                e = space.coordinates_to_error(s, z)
                assert e.dtype == np.uint8
                np.testing.assert_array_equal(e, (particular + N @ z) % 2)
                np.testing.assert_array_equal(H @ e % 2, s)
                np.testing.assert_array_equal(space.error_to_coordinates(s, e), z)
        np.testing.assert_array_equal(space.nullspace_basis(), N)


def test_measured_surface_code_syndromes():
    circuit = build_surface_code(3, 3, 0.005)
    H = build_incidence_matrix(build_decoding_graph(get_detector_error_model(circuit)))
    space = AffineCoordinates(H)
    rng = np.random.default_rng(71)
    for s in sample_shots(circuit, 64, seed=42)[0]:
        for _ in range(10):
            z = rng.integers(0, 2, size=space.nullity, dtype=np.uint8)
            e = space.coordinates_to_error(s, z)
            np.testing.assert_array_equal(H @ e % 2, s)
            np.testing.assert_array_equal(space.error_to_coordinates(s, e), z)


@pytest.mark.parametrize("H", [np.zeros((3, 4)), np.eye(4), np.ones((1, 1025))])
def test_zero_unique_and_wide_systems(H):
    space = AffineCoordinates(H)
    e = np.ones(H.shape[1], dtype=np.uint8)
    s = H.astype(np.int64) @ e % 2
    z = space.error_to_coordinates(s, e)
    np.testing.assert_array_equal(space.coordinates_to_error(s, z), e)


def test_inputs_and_returned_arrays_do_not_mutate_representation():
    H = np.array([[1, 1]], dtype=np.uint8)
    space = AffineCoordinates(H)
    H[:] = 0
    s, z, e = np.array([1]), np.array([1]), np.array([0, 1])
    space.nullspace_basis()[:] = 0
    space.particular_solution(s)[:] = 0
    space.coordinates_to_error(s, z)[:] = 0
    space.error_to_coordinates(s, e)[:] = 0
    np.testing.assert_array_equal(space.nullspace_basis(), [[1], [1]])
    np.testing.assert_array_equal(space.particular_solution(s), [1, 0])
    np.testing.assert_array_equal(space.coordinates_to_error(s, z), e)
    np.testing.assert_array_equal(s, [1])
    np.testing.assert_array_equal(z, [1])
    np.testing.assert_array_equal(e, [0, 1])


@pytest.mark.parametrize("bad", [[2], [-1], [0.5], [np.nan], [np.inf], ["1"], [[1]], [0, 1]])
def test_invalid_vectors(bad):
    space = AffineCoordinates([[1, 1]])
    with pytest.raises(ValueError):
        space.particular_solution(bad)
    with pytest.raises(ValueError):
        space.coordinates_to_error([0], bad)
    with pytest.raises(ValueError):
        space.error_to_coordinates(bad, [0, 0])


@pytest.mark.parametrize("bad", [[2, 0], [0.5, 0], [np.nan, 0], [[0, 0]], [0], [0, 0, 0], [1, 0]])
def test_invalid_or_incompatible_errors(bad):
    with pytest.raises(ValueError):
        AffineCoordinates([[1, 1]]).error_to_coordinates([0], bad)


@pytest.mark.parametrize("bad", [[1, 0], [[2]], [[np.nan]], [[0.5]], [["1"]]])
def test_invalid_matrices(bad):
    with pytest.raises(ValueError):
        AffineCoordinates(bad)
