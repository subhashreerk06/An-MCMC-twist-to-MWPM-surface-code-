"""Saved coordinates reconstruct the sampled independent-edge errors exactly."""

import json

import numpy as np
import pytest
import stim

from surface_code import flow_dataset as dataset
from surface_code.circuit import build_surface_code, get_detector_error_model
from surface_code.decoding_graph import build_decoding_graph
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix


def test_saved_pairs_reconstruct_every_original_error(tmp_path):
    output = dataset.generate_dataset(
        train_samples=256, validation_samples=64, seed=314159, output=tmp_path / "data",
    )
    metadata = json.loads((output / "metadata.json").read_text())
    graph = build_decoding_graph(get_detector_error_model(build_surface_code(3, 3, 0.005)))
    H = build_incidence_matrix(graph)
    coordinates = AffineCoordinates(H)
    probabilities = np.array([edge.p_error for edge in graph.edges])
    rng = np.random.Generator(np.random.PCG64(314159))
    with np.load(output / "samples.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"train_s", "train_z", "validation_s", "validation_z"}
        for split, count in (("train", 256), ("validation", 64)):
            syndromes, latent = saved[f"{split}_s"], saved[f"{split}_z"]
            assert syndromes.shape == (count, H.shape[0])
            assert latent.shape == (count, coordinates.nullity)
            assert syndromes.dtype == latent.dtype == np.uint8
            for s, z in zip(syndromes, latent):
                original = (rng.random(len(graph.edges)) < probabilities).astype(np.uint8)
                reconstructed = coordinates.coordinates_to_error(s, z)
                np.testing.assert_array_equal(reconstructed, original)
                np.testing.assert_array_equal(H @ reconstructed % 2, s)
                np.testing.assert_array_equal(coordinates.error_to_coordinates(s, reconstructed), z)
    assert (metadata["distance"], metadata["rounds"], metadata["p"]) == (3, 3, 0.005)
    assert metadata["num_graph_edges"] == H.shape[1]
    assert metadata["num_syndrome_bits"] == H.shape[0]
    assert metadata["z_dimension"] == coordinates.nullity
    assert metadata["seed"] == 314159
    assert metadata["free_columns"] == list(coordinates.free_columns)
    assert metadata["h_fingerprint"] == dataset._matrix_fingerprint(H)
    assert metadata["nullspace_fingerprint"] == dataset._matrix_fingerprint(coordinates.nullspace_basis())
    for name in ("edge_ordering_fingerprint", "h_nullspace_fingerprint"):
        assert len(metadata[name]) == 64


def test_reproducible_cli_and_fingerprints(tmp_path, capsys):
    paths = [tmp_path / "a", tmp_path / "b"]
    for path in paths:
        dataset.main([
            "--distance", "3", "--rounds", "3", "--p", "0.005",
            "--train-samples", "12", "--validation-samples", "5",
            "--seed", "19", "--output", str(path),
        ])
    assert "Dataset saved" in capsys.readouterr().out
    assert (paths[0] / "metadata.json").read_bytes() == (paths[1] / "metadata.json").read_bytes()
    with np.load(paths[0] / "samples.npz") as a, np.load(paths[1] / "samples.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert dataset._matrix_fingerprint(np.zeros((1, 2))) != dataset._matrix_fingerprint(np.zeros((2, 1)))
    assert dataset._json_fingerprint([0, 1]) != dataset._json_fingerprint([1, 0])


def test_bernoulli_probabilities_and_independent_components():
    graph = build_decoding_graph(stim.DetectorErrorModel("""
        error(0) L0
        error(1) L0
        error(0.25) L0 ^ L0
        error(0.75) L0
    """))
    s, z = dataset.sample_pairs(graph, 20000, np.random.default_rng(12))
    assert s.shape == (20000, 0)
    # H has no rows, so all error bits are directly observable as coordinates.
    assert np.all(z[:, 0] == 0)
    assert np.all(z[:, 1] == 1)
    np.testing.assert_allclose(z[:, 2:].mean(axis=0), [0.25, 0.25, 0.75], atol=0.015)
    assert abs(np.mean(z[:, 2] * z[:, 3]) - 0.25**2) < 0.01
    assert np.any(z[:, 2] != z[:, 3])


def test_empty_splits_and_empty_graph(tmp_path):
    output = dataset.generate_dataset(
        p=0, train_samples=0, validation_samples=3, output=tmp_path / "empty",
    )
    with np.load(output / "samples.npz") as saved:
        assert saved["train_s"].shape == (0, 24)
        assert saved["train_z"].shape == (0, 0)
        assert saved["validation_z"].shape == (3, 0)
        assert not saved["validation_s"].any()


@pytest.mark.parametrize("kwargs", [
    {"train_samples": -1}, {"validation_samples": 1.5}, {"seed": -1},
    {"train_samples": True}, {"seed": "123"},
])
def test_invalid_settings_do_not_create_output(tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        dataset.generate_dataset(output=output, **kwargs)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path / "samples.npz"
    marker.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        dataset.generate_dataset(output=tmp_path, train_samples=1, validation_samples=1)
    assert marker.read_bytes() == b"existing data"
