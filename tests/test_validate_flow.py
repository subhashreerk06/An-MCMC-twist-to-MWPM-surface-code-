"""Paired proposal weights, exact syndrome checks, and identity validation."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from surface_code import validate_flow
from surface_code.circuit import build_surface_code, get_detector_error_model
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_dataset import generate_dataset
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix


@pytest.fixture
def inputs(tmp_path):
    dataset = generate_dataset(
        rounds=1, train_samples=8, validation_samples=9, output=tmp_path / "data",
    )
    metadata = json.loads((dataset / "metadata.json").read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = ConditionalAutoregressiveBernoulli(metadata["num_syndrome_bits"], metadata["z_dimension"], 8)
    checkpoint = tmp_path / "model.pt"
    model.save_checkpoint(checkpoint, fingerprints=metadata)
    return dataset, checkpoint


def read_rows(output):
    with (output / "validation.csv").open() as stream:
        return list(csv.DictReader(stream))


def test_every_saved_proposal_and_weight_is_exact(inputs, tmp_path, monkeypatch, capsys):
    dataset, checkpoint = inputs
    original = validate_flow.sample_uniform_solution
    calls = []

    def capture(H, s, rng):
        e = original(H, s, rng)
        calls.append(e.copy())
        return e

    monkeypatch.setattr(validate_flow, "sample_uniform_solution", capture)
    output = validate_flow.validate(
        dataset=dataset, checkpoint=checkpoint, output=tmp_path / "validation",
        num_syndromes=7, proposals_per_syndrome=3, batch_size=8,
    )
    assert "5th percentile" in capsys.readouterr().out
    graph = build_decoding_graph(get_detector_error_model(build_surface_code(3, 1, 0.005)))
    H = build_incidence_matrix(graph)
    coordinates = AffineCoordinates(H)
    weights = np.array([edge.weight for edge in graph.edges])
    rows = read_rows(output)
    assert len(rows) == len(calls) == 21
    with np.load(dataset / "samples.npz") as data:
        for row, original_error in zip(rows, calls):
            s = np.array(list(row["syndrome"]), dtype=np.uint8)
            np.testing.assert_array_equal(s, data["validation_s"][int(row["syndrome_index"])])
            for method in ("uniform", "flow"):
                z = np.array(list(row[f"z_{method}"]), dtype=np.uint8)
                e = coordinates.coordinates_to_error(s, z)
                np.testing.assert_array_equal(H @ e % 2, s)
                assert float(row[f"{method}_weight"]) == float(np.sum(e * weights))
                assert row[f"{method}_syndrome_valid"] == "True"
                if method == "uniform":
                    np.testing.assert_array_equal(e, original_error)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["all_syndromes_valid"]
    for method in ("uniform", "flow"):
        values = np.array([float(row[f"{method}_weight"]) for row in rows])
        assert summary[method] == {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "minimum": float(values.min()), "percentile_5": float(np.percentile(values, 5)),
            "percentile_95": float(np.percentile(values, 95)),
        }
    assert (output / "proposal_weights.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_reproducibility_and_preserving_output(inputs, tmp_path):
    dataset, checkpoint = inputs
    outputs = [tmp_path / "a", tmp_path / "b"]
    for output in outputs:
        validate_flow.main([
            "--dataset", str(dataset), "--checkpoint", str(checkpoint), "--output", str(output),
            "--num-syndromes", "4", "--proposals-per-syndrome", "2", "--batch-size", "3", "--seed", "99",
        ])
    for filename in ("validation.csv", "summary.json"):
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    with pytest.raises(FileExistsError):
        validate_flow.validate(dataset=dataset, checkpoint=checkpoint, output=outputs[0], num_syndromes=1)


def test_no_decoder_imports(inputs, tmp_path):
    dataset, checkpoint = inputs
    script = '''
import builtins
original_import = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if any(word in name.lower() for word in ('mwpm', 'mcmc')):
        raise AssertionError('Decoder import attempted: ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
from surface_code.validate_flow import main
main()
'''
    result = subprocess.run(
        [sys.executable, "-c", script, "--dataset", str(dataset), "--checkpoint", str(checkpoint),
         "--output", str(tmp_path / "isolated"), "--num-syndromes", "2", "--proposals-per-syndrome", "1"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("damage", ["graph", "basis", "checkpoint", "dimensions"])
def test_mismatched_identity_rejected(inputs, tmp_path, damage):
    dataset, checkpoint = inputs
    path = dataset / "metadata.json"
    metadata = json.loads(path.read_text())
    if damage == "checkpoint":
        model = ConditionalAutoregressiveBernoulli(metadata["num_syndrome_bits"], metadata["z_dimension"], 8)
        model.save_checkpoint(checkpoint, fingerprints={**metadata, "h_nullspace_fingerprint": "a" * 64})
    elif damage == "graph":
        metadata["p"] = 0.01
    elif damage == "basis":
        metadata["h_nullspace_fingerprint"] = "a" * 64
    else:
        metadata["z_dimension"] += 1
    path.write_text(json.dumps(metadata))
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        validate_flow.validate(dataset=dataset, checkpoint=checkpoint, output=output, num_syndromes=1)
    assert not output.exists()


def test_syndrome_verification_detects_invalid_reconstruction(inputs, tmp_path, monkeypatch):
    dataset, checkpoint = inputs
    original = AffineCoordinates.coordinates_to_error

    def corrupt(self, s, z):
        e = original(self, s, z)
        e[0] ^= 1
        return e

    monkeypatch.setattr(AffineCoordinates, "coordinates_to_error", corrupt)
    with pytest.raises(RuntimeError, match="violates"):
        validate_flow.validate(dataset=dataset, checkpoint=checkpoint, output=tmp_path / "invalid", num_syndromes=1)


@pytest.mark.parametrize("kwargs", [{"num_syndromes": 0}, {"num_syndromes": 10}, {"proposals_per_syndrome": 0}, {"batch_size": 0}, {"seed": -1}])
def test_invalid_parameters(inputs, tmp_path, kwargs):
    dataset, checkpoint = inputs
    with pytest.raises(ValueError):
        validate_flow.validate(dataset=dataset, checkpoint=checkpoint, output=tmp_path / "bad", **kwargs)


def test_statistics_known_values():
    assert validate_flow.summarize_weights([0, 10, 20]) == {
        "mean": 10, "median": 10, "minimum": 0, "percentile_5": 1, "percentile_95": 19,
    }
