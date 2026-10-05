"""Tests for Stim circuit construction and raw detector sampling."""

import numpy as np
import pytest
import stim

from surface_code.circuit import (
    build_surface_code,
    compile_detector_sampler,
    get_detector_error_model,
    sample_shots,
)


@pytest.mark.parametrize("distance,rounds,p", [(3, 3, 0.005), (5, 2, 0.01), (3, 1, 0)])
def test_build_and_sample(distance, rounds, p):
    circuit = build_surface_code(distance, rounds, p)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors == (distance**2 - 1) * rounds
    assert circuit.num_observables == 1
    detectors, observables = sample_shots(circuit, 100, seed=42)
    assert detectors.shape == (100, circuit.num_detectors)
    assert observables.shape == (100, 1)
    assert detectors.dtype == observables.dtype == np.bool_


def test_detector_error_model():
    circuit = build_surface_code(3, 3, 0.005)
    model = get_detector_error_model(circuit)
    assert isinstance(model, stim.DetectorErrorModel)
    assert model.num_detectors == 24
    assert model.num_observables == 1
    assert model.num_errors > 0
    assert model == circuit.detector_error_model(decompose_errors=True)


def test_seeded_shots_are_reproducible():
    circuit = build_surface_code(3, 3, 0.005)
    first = sample_shots(circuit, 1000, seed=1234)
    second = sample_shots(circuit, 1000, seed=1234)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_compiled_sampler_retains_observables():
    circuit = build_surface_code(3, 3, 0.005)
    sampler = compile_detector_sampler(circuit, seed=7)
    detectors, observables = sampler.sample(100, separate_observables=True)
    expected = sample_shots(circuit, 100, seed=7)
    np.testing.assert_array_equal(detectors, expected[0])
    np.testing.assert_array_equal(observables, expected[1])


def test_noiseless_shots_have_no_flips():
    circuit = build_surface_code(3, 3, 0)
    detectors, observables = sample_shots(circuit, 100, seed=42)
    assert not detectors.any()
    assert not observables.any()


def test_zero_shots():
    detectors, observables = sample_shots(build_surface_code(3, 3, 0.005), 0, seed=0)
    assert detectors.shape == (0, 24)
    assert observables.shape == (0, 1)
