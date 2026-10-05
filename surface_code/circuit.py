"""Rotated memory-X circuits, detector error models, and raw Stim samples."""

import numpy as np
import stim


def build_surface_code(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build a memory-X experiment with depolarization after Clifford gates.

    Parameter validation is delegated to Stim. No other noise is enabled.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
    )


def get_detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Extract the detector error model with errors decomposed by Stim."""
    return circuit.detector_error_model(decompose_errors=True)


def compile_detector_sampler(
    circuit: stim.Circuit, *, seed: int | None = None
) -> stim.CompiledDetectorSampler:
    """Compile a sampler for detector events and logical observable flips.

    Call ``sampler.sample(shots, separate_observables=True)`` to retain both.
    A sampler advances its random state with each sampling call.
    """
    return circuit.compile_detector_sampler(seed=seed)


def sample_shots(
    circuit: stim.Circuit, shots: int, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (detector_events, observable_flips) arrays.

    Shapes are (shots, circuit.num_detectors) and
    (shots, circuit.num_observables). These are raw flips, not decoded errors.
    A fresh sampler makes identical calls with a supplied seed reproducible
    on the same machine and Stim version. Changing the shot count or batching
    can change the seeded results.
    """
    sampler = compile_detector_sampler(circuit, seed=seed)
    return sampler.sample(shots=shots, separate_observables=True)


if __name__ == "__main__":
    circuit = build_surface_code(distance=3, rounds=3, p=0.005)
    model = get_detector_error_model(circuit)
    print(f"Number of qubits (Stim index span): {circuit.num_qubits}")
    print(f"Number of detectors: {circuit.num_detectors}")
    print(f"Number of observables: {circuit.num_observables}")
    print("First 12 lines of the Stim circuit:")
    print("\n".join(str(circuit).splitlines()[:12]))
    print("Detector error model summary:")
    print(
        f"  {model.num_detectors} detectors, "
        f"{model.num_observables} observables, {model.num_errors} error mechanisms"
    )
