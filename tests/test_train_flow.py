"""Likelihood training, reproducible splits, early stopping, and saved metrics."""

import csv
import json

import numpy as np
import pytest
import torch

from surface_code import train_flow
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_dataset import generate_dataset


@pytest.fixture
def dataset(tmp_path):
    return generate_dataset(
        distance=3, rounds=1, p=0.02, train_samples=29, validation_samples=11,
        seed=13, output=tmp_path / "data",
    )


def history(path):
    with (path / "training_history.csv").open() as stream:
        return list(csv.DictReader(stream))


def test_training_saves_reproducible_history_and_exact_checkpoint_metrics(dataset, tmp_path, capsys):
    paths = [tmp_path / "first", tmp_path / "replay"]
    before_rng = torch.random.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    for path in paths:
        train_flow.main([
            "--dataset", str(dataset), "--output", str(path), "--epochs", "3",
            "--batch-size", "8", "--learning-rate", "0.01", "--seed", "31",
            "--hidden-dim", "12",
        ])
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    printed = capsys.readouterr().out
    assert "train NLL" in printed and "validation NLL" in printed
    assert "validation mean log probability" in printed
    assert history(paths[0]) == history(paths[1])
    metadata = json.loads((paths[0] / "metadata.json").read_text())
    assert metadata == json.loads((paths[1] / "metadata.json").read_text())
    rows = history(paths[0])
    assert len(rows) == 3
    assert float(rows[-1]["train_nll"]) < float(rows[0]["train_nll"])
    assert metadata["best_validation_nll"] == min(float(r["validation_nll"]) for r in rows)
    assert metadata["validation_mean_log_probability"] == -metadata["best_validation_nll"]
    with np.load(dataset / "samples.npz") as arrays:
        s = torch.from_numpy(arrays["validation_s"])
        z = torch.from_numpy(arrays["validation_z"])
    for filename, metric in (("best.pt", "best_validation_nll"), ("last.pt", "last_validation_nll")):
        model = ConditionalAutoregressiveBernoulli.load_checkpoint(
            paths[0] / filename, expected_fingerprints=metadata["fingerprints"],
        )
        actual = -train_flow.mean_log_probability(model, s, z, 8)
        assert actual == pytest.approx(metadata[metric], abs=1e-7)
        replay = ConditionalAutoregressiveBernoulli.load_checkpoint(
            paths[1] / filename, expected_fingerprints=metadata["fingerprints"],
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, replay.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("scores,min_delta,best_epoch", [([4, 3, 3.5, 3.6], 0, 2), ([10, 9.5, 9.4], 1, 3)])
def test_early_stopping_and_best_selection(dataset, tmp_path, monkeypatch, scores, min_delta, best_epoch):
    sequence = iter(scores + [min(scores)])
    monkeypatch.setattr(train_flow, "mean_log_probability", lambda *args: -next(sequence))
    saved = {}
    original = ConditionalAutoregressiveBernoulli.save_checkpoint

    def capture(self, path, **kwargs):
        saved.setdefault(path.name, []).append({key: value.clone() for key, value in self.state_dict().items()})
        return original(self, path, **kwargs)

    monkeypatch.setattr(ConditionalAutoregressiveBernoulli, "save_checkpoint", capture)
    output = train_flow.train(
        dataset=dataset, output=tmp_path / "stopped", epochs=20, batch_size=8,
        patience=2, min_delta=min_delta, hidden_dim=8,
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["stopped_early"]
    assert metadata["epochs_completed"] == len(scores)
    assert metadata["best_epoch"] == best_epoch
    assert metadata["best_validation_nll"] == min(scores)
    assert len(saved["last.pt"]) == len(scores)
    for filename, epoch in (("best.pt", best_epoch), ("last.pt", len(scores))):
        model = ConditionalAutoregressiveBernoulli.load_checkpoint(
            output / filename, expected_fingerprints=metadata["fingerprints"],
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved["last.pt"][epoch - 1][key])


def test_partial_validation_batch_is_sample_weighted():
    class Fixed:
        def eval(self):
            pass

        def log_prob(self, z, s):
            return -s[:, 0].double()

    s = torch.tensor([[0], [0], [1]])
    assert train_flow.mean_log_probability(Fixed(), s, s, 2) == -1 / 3


def test_validation_never_enters_optimizer(dataset, tmp_path, monkeypatch):
    from collections import Counter

    original = ConditionalAutoregressiveBernoulli.log_prob
    training_pairs = []

    def record(self, z, s):
        if self.training:
            training_pairs.extend(tuple(row) for row in torch.cat((s, z), dim=1).tolist())
        return original(self, z, s)

    monkeypatch.setattr(ConditionalAutoregressiveBernoulli, "log_prob", record)
    train_flow.train(dataset=dataset, output=tmp_path / "audit", epochs=2, batch_size=8, hidden_dim=8)
    with np.load(dataset / "samples.npz") as data:
        expected = Counter(tuple(row) for row in np.concatenate((data["train_s"], data["train_z"]), axis=1))
    assert Counter(training_pairs) == Counter({key: value * 2 for key, value in expected.items()})


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0},
    {"learning_rate": float("nan")}, {"seed": -1}, {"patience": 0},
    {"min_delta": -1}, {"hidden_dim": 0}, {"eps": 0},
])
def test_invalid_settings_create_no_output(dataset, tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        train_flow.train(dataset=dataset, output=output, **kwargs)
    assert not output.exists()


def test_existing_output_preserved(dataset, tmp_path):
    marker = tmp_path / "best.pt"
    marker.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        train_flow.train(dataset=dataset, output=tmp_path)
    assert marker.read_bytes() == b"keep"


@pytest.mark.parametrize("damage", ["nonbinary", "dtype", "empty", "width", "fingerprint"])
def test_invalid_dataset_rejected(dataset, tmp_path, damage):
    with np.load(dataset / "samples.npz") as archive:
        arrays = dict(archive)
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if damage == "nonbinary":
        arrays["train_z"][0, 0] = 2
    elif damage == "dtype":
        arrays["validation_z"] = arrays["validation_z"].astype(float)
    elif damage == "empty":
        metadata["validation_samples"] = 0
        arrays["validation_s"] = arrays["validation_s"][:0]
        arrays["validation_z"] = arrays["validation_z"][:0]
    elif damage == "width":
        metadata["z_dimension"] += 1
    else:
        metadata["h_nullspace_fingerprint"] = "bad"
    np.savez(dataset / "samples.npz", **arrays)
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        train_flow.train(dataset=dataset, output=output)
    assert not output.exists()
