"""Maximum-likelihood training on saved independent-edge (syndrome, z) pairs.

The existing train_s/train_z and validation_s/validation_z arrays define the
split; validation samples never enter optimization. Each mini-batch minimizes
-mean(log q_theta(z | s)) with Adam. No labels, error energies, or selected
configurations are added. Epoch training NLL is the sample-weighted mean of
pre-update mini-batch losses; validation NLL evaluates the end-of-epoch model.
All reported log probabilities use natural logarithms, per complete z vector.

CPU training uses a fixed thread count, seeded initialization, and a separate
seeded shuffle generator. Reproducibility applies with the same data, software,
and hardware. Checkpoints contain inference parameters and graph/basis identity,
not optimizer state for resuming interrupted training.
"""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from operator import index
from pathlib import Path

import numpy as np
import torch

from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli, _fingerprints


def _integer(value, name: str, minimum: int = 1) -> int:
    try:
        if isinstance(value, (bool, np.bool_)) or index(value) < minimum:
            raise ValueError
        return int(index(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc


def _load_dataset(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported dataset format.")
    if metadata.get("distribution") != "independent Bernoulli(graph.edges[i].p_error)":
        raise ValueError("Expected an independent-edge dataset from flow_dataset.py.")
    if metadata.get("dem_component_correlations_enforced") is not False:
        raise ValueError("Expected independent component edges.")
    _fingerprints(metadata)
    syndrome_dim = _integer(metadata["num_syndrome_bits"], "num_syndrome_bits", 0)
    z_dim = _integer(metadata["z_dimension"], "z_dimension", 0)
    arrays = {}
    with np.load(path / "samples.npz", allow_pickle=False) as archive:
        for split in ("train", "validation"):
            count = _integer(metadata[f"{split}_samples"], f"{split}_samples")
            for suffix, width in (("s", syndrome_dim), ("z", z_dim)):
                key = f"{split}_{suffix}"
                value = archive[key]
                if value.dtype != np.uint8 or value.shape != (count, width):
                    raise ValueError(f"{key} must be uint8 with shape {(count, width)}.")
                if not np.all((value == 0) | (value == 1)):
                    raise ValueError(f"{key} must be binary.")
                arrays[key] = torch.from_numpy(value.copy())
    return metadata, arrays


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _seeded_cpu(seed: int, threads: int):
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


@torch.no_grad()
def mean_log_probability(model, syndromes, coordinates, batch_size: int) -> float:
    """Evaluate all samples with sample weighting, including the final short batch."""
    batch_size = _integer(batch_size, "batch_size")
    if len(syndromes) != len(coordinates) or not len(syndromes):
        raise ValueError("Evaluation requires nonempty paired arrays.")
    model.eval()
    total = 0.0
    for start in range(0, len(syndromes), batch_size):
        total += model.log_prob(
            coordinates[start:start + batch_size], syndromes[start:start + batch_size],
        ).sum().item()
    result = total / len(syndromes)
    if not math.isfinite(result):
        raise ValueError("Evaluation log probability is not finite.")
    return result


def train(
    *, dataset: str | Path, output: str | Path, epochs: int = 50,
    batch_size: int = 256, learning_rate: float = 1e-3, seed: int = 271828,
    patience: int = 5, min_delta: float = 0.0, hidden_dim: int = 128,
    eps: float = 1e-6, threads: int = 1,
) -> Path:
    """Train and save best.pt, last.pt, training_history.csv, and metadata.json.

    Best always means the lowest observed validation NLL. Early stopping uses
    a separate reference: improvement must exceed min_delta to reset patience.
    Stop after patience consecutive epochs without such an improvement.
    The final reported validation mean log probability is for best.pt.
    Existing output directories are refused.
    """
    epochs = _integer(epochs, "epochs")
    batch_size = _integer(batch_size, "batch_size")
    seed = _integer(seed, "seed", 0)
    patience = _integer(patience, "patience")
    threads = _integer(threads, "threads")
    if seed >= 2**63:
        raise ValueError("seed must be less than 2**63.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative.")
    dataset, output = Path(dataset), Path(output)
    source, arrays = _load_dataset(dataset)
    fingerprints = _fingerprints(source)
    metadata = {
        "format_version": 1, "dataset": str(dataset.resolve()),
        "dataset_metadata": source,
        "dataset_sha256": _file_hash(dataset / "samples.npz"),
        "dataset_metadata_sha256": _file_hash(dataset / "metadata.json"),
        "fingerprints": fingerprints,
        "objective": "-mean(log q_theta(z | s))", "optimizer": "Adam",
        "epochs_requested": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "seed": seed, "patience": patience, "min_delta": min_delta,
        "hidden_dim": hidden_dim, "eps": eps, "device": "cpu", "threads": threads,
        "split": "original dataset train/validation arrays, no resplitting",
        "train_nll_definition": "sample-weighted pre-update mini-batch NLL",
        "validation_nll_definition": "end-of-epoch mean NLL per complete z vector (nats)",
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    with _seeded_cpu(seed, threads):
        model = ConditionalAutoregressiveBernoulli(
            source["num_syndrome_bits"], source["z_dimension"], hidden_dim, eps,
        ).to(device="cpu", dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        shuffle = torch.Generator(device="cpu").manual_seed(seed)
        output.mkdir(parents=True, exist_ok=False)
        best_nll, stopping_reference = math.inf, math.inf
        stale_epochs = 0
        fields = ("epoch", "train_nll", "validation_nll", "validation_mean_log_probability", "is_best")
        with (output / "training_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                model.train()
                permutation = torch.randperm(len(arrays["train_s"]), generator=shuffle)
                total_train_nll = 0.0
                for indices in permutation.split(batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = -model.log_prob(arrays["train_z"][indices], arrays["train_s"][indices]).mean()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Training loss is not finite.")
                    loss.backward()
                    optimizer.step()
                    total_train_nll += loss.item() * len(indices)
                train_nll = total_train_nll / len(permutation)
                validation_nll = -mean_log_probability(
                    model, arrays["validation_s"], arrays["validation_z"], batch_size,
                )
                is_best = validation_nll < best_nll
                if is_best:
                    best_nll, best_epoch = validation_nll, epoch
                    model.save_checkpoint(output / "best.pt", fingerprints=fingerprints)
                model.save_checkpoint(output / "last.pt", fingerprints=fingerprints)
                writer.writerow({
                    "epoch": epoch, "train_nll": train_nll, "validation_nll": validation_nll,
                    "validation_mean_log_probability": -validation_nll, "is_best": is_best,
                })
                stream.flush()
                print(f"epoch {epoch:3d}  train NLL {train_nll:.8f}  validation NLL {validation_nll:.8f}", flush=True)
                if stopping_reference - validation_nll > min_delta:
                    stopping_reference, stale_epochs = validation_nll, 0
                else:
                    stale_epochs += 1
                if stale_epochs >= patience:
                    break
        best_model = ConditionalAutoregressiveBernoulli.load_checkpoint(
            output / "best.pt", expected_fingerprints=fingerprints,
        )
        best_mean_log_probability = mean_log_probability(
            best_model, arrays["validation_s"], arrays["validation_z"], batch_size,
        )
    metadata.update({
        "epochs_completed": epoch, "best_epoch": best_epoch,
        "best_validation_nll": best_nll, "last_validation_nll": validation_nll,
        "validation_mean_log_probability": best_mean_log_probability,
        "validation_report_checkpoint": "best.pt",
        "stopped_early": epoch < epochs,
    })
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Best checkpoint epoch {best_epoch}; validation mean log probability {best_mean_log_probability:.8f} nats", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=1)
    try:
        output = train(**vars(parser.parse_args(argv)))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Training outputs saved to {output}")


if __name__ == "__main__":
    main()
