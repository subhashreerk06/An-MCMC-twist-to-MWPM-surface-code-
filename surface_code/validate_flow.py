"""Compare raw uniform and learned proposals on the same validation syndromes.

The uniform baseline calls sample_uniform_solution(H, s, rng), the existing
GF(2) free-variable sampler, then extracts z with error_to_coordinates.
Both methods reconstruct errors through e_p(s) + N z and use the graph's stored
weights to evaluate W(E). Every reconstructed proposal is checked against its
syndrome before it is recorded. There is no acceptance step or state search.

Syndromes are selected without replacement from validation_s; repeated syndrome
values are retained, preserving the dataset's empirical syndrome distribution.
These are the validation data used for model selection, not a fresh test set.
The comparison assesses proposal weights only, not logical decoding accuracy.
"""

import argparse
import csv
import hashlib
import json
from operator import index
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch

from surface_code.circuit import build_surface_code, get_detector_error_model
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli, _fingerprints
from surface_code.flow_dataset import _json_fingerprint, _matrix_fingerprint
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix, sample_uniform_solution


def _integer(value, name, minimum=1):
    try:
        if isinstance(value, (bool, np.bool_)) or index(value) < minimum:
            raise ValueError
        return int(index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc


def summarize_weights(weights) -> dict[str, float]:
    """Describe all proposals with NumPy's default linear percentiles."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Weights must be a nonempty finite vector.")
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "minimum": float(values.min()), "percentile_5": float(np.percentile(values, 5)),
        "percentile_95": float(np.percentile(values, 95)),
    }


def _checked_error(coordinates, H, s, z):
    error = coordinates.coordinates_to_error(s, z)
    if not np.array_equal((H @ error) % 2, s):
        raise RuntimeError("Proposal violates H e = s mod 2.")
    return error


def _bit_string(bits):
    return "".join(str(int(bit)) for bit in bits)


def _plot(uniform, learned, path):
    figure = Figure(figsize=(9, 5), layout="constrained")
    axis = figure.subplots()
    bins = np.histogram_bin_edges(np.concatenate((uniform, learned)), bins=60)
    axis.hist(uniform, bins=bins, alpha=0.65, color="tab:blue", label="Uniform GF(2) proposals")
    axis.hist(learned, bins=bins, alpha=0.65, color="tab:orange", label="Flow proposals")
    axis.axvline(np.mean(uniform), color="tab:blue", linestyle="--", label="Uniform mean")
    axis.axvline(np.mean(learned), color="tab:orange", linestyle="--", label="Flow mean")
    axis.set(xlabel="Proposal weight W(E)", ylabel="Proposal count", title="Proposal weights on the same validation syndromes")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    FigureCanvasAgg(figure).print_figure(path, dpi=160, bbox_inches="tight")


def validate(
    *, dataset: str | Path, checkpoint: str | Path, output: str | Path,
    num_syndromes: int = 256, proposals_per_syndrome: int = 4,
    seed: int = 161803, batch_size: int = 128, threads: int = 1,
) -> Path:
    """Save auditable paired weights, a histogram, and summary.json.

    Separate seeded NumPy streams select validation rows, generate uniform
    proposals, and sample the model. Seed, batch size, and thread count are
    recorded for replay. Outputs go into a new directory, never overwriting
    a previous run. The graph, H, and N are rebuilt and fingerprints verified
    against both the dataset and checkpoint before sampling.
    """
    num_syndromes = _integer(num_syndromes, "num_syndromes")
    proposals_per_syndrome = _integer(proposals_per_syndrome, "proposals_per_syndrome")
    seed = _integer(seed, "seed", 0)
    batch_size = _integer(batch_size, "batch_size")
    threads = _integer(threads, "threads")
    dataset, checkpoint, output = Path(dataset), Path(checkpoint), Path(output)
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    expected = _fingerprints(metadata)
    graph = build_decoding_graph(get_detector_error_model(build_surface_code(
        metadata["distance"], metadata["rounds"], metadata["p"],
    )))
    H = build_incidence_matrix(graph)
    coordinates = AffineCoordinates(H)
    ordered_edges = [
        {"edge_id": edge.edge_id, "detector_endpoints": edge.detector_endpoints,
         "p_error": float(edge.p_error).hex(), "logical_mask": edge.logical_mask,
         "mechanism_id": edge.mechanism_id, "component_index": edge.component_index}
        for edge in graph.edges
    ]
    actual = {
        "edge_ordering_fingerprint": _json_fingerprint(ordered_edges),
        "h_nullspace_fingerprint": _json_fingerprint([
            _matrix_fingerprint(H), _matrix_fingerprint(coordinates.nullspace_basis()),
        ]),
    }
    if actual != expected:
        raise ValueError("Rebuilt graph/basis fingerprints do not match the dataset.")
    if (metadata["num_graph_edges"], metadata["num_syndrome_bits"], metadata["z_dimension"]) != (H.shape[1], H.shape[0], coordinates.nullity):
        raise ValueError("Dataset dimensions do not match the rebuilt graph/basis.")
    weights = np.array([edge.weight for edge in graph.edges], dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Proposal weight comparison requires finite graph weights.")
    with np.load(dataset / "samples.npz", allow_pickle=False) as archive:
        syndromes = archive["validation_s"]
    if (syndromes.dtype != np.uint8 or syndromes.ndim != 2
            or syndromes.shape != (metadata["validation_samples"], H.shape[0])
            or not np.all((syndromes == 0) | (syndromes == 1))):
        raise ValueError("validation_s must be a binary uint8 array matching dataset metadata.")
    if num_syndromes > len(syndromes):
        raise ValueError("num_syndromes exceeds available validation samples.")
    # Loading initializes parameters before replacing them; preserve caller RNG.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalAutoregressiveBernoulli.load_checkpoint(
            checkpoint, expected_fingerprints=actual, device="cpu",
        )
    if (model.syndrome_dim, model.z_dim) != (H.shape[0], coordinates.nullity):
        raise ValueError("Checkpoint dimensions do not match the graph/basis.")
    selection_rng, uniform_rng, proposal_rng = [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)
    ]
    selected = selection_rng.choice(len(syndromes), size=num_syndromes, replace=False)
    row_indices = np.repeat(selected, proposals_per_syndrome)
    proposal_indices = np.tile(np.arange(proposals_per_syndrome), num_syndromes)
    uniform_weights, learned_weights = [], []
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with (output / "validation.csv").open("w", newline="") as stream:
            fields = ("syndrome_index", "proposal_index", "syndrome", "z_uniform", "z_flow",
                      "uniform_weight", "flow_weight", "uniform_syndrome_valid", "flow_syndrome_valid")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for start in range(0, len(row_indices), batch_size):
                batch_s = syndromes[row_indices[start:start + batch_size]]
                batch_z = model.sample(batch_s, proposal_rng).cpu().numpy()
                for offset, (s, z_learned) in enumerate(zip(batch_s, batch_z)):
                    position = start + offset
                    uniform_error = sample_uniform_solution(H, s, uniform_rng)
                    z_uniform = coordinates.error_to_coordinates(s, uniform_error)
                    uniform_error = _checked_error(coordinates, H, s, z_uniform)
                    learned_error = _checked_error(coordinates, H, s, z_learned)
                    uniform_weight = float(np.sum(uniform_error * weights))
                    learned_weight = float(np.sum(learned_error * weights))
                    uniform_weights.append(uniform_weight)
                    learned_weights.append(learned_weight)
                    writer.writerow({
                        "syndrome_index": int(row_indices[position]),
                        "proposal_index": int(proposal_indices[position]),
                        "syndrome": _bit_string(s), "z_uniform": _bit_string(z_uniform),
                        "z_flow": _bit_string(z_learned), "uniform_weight": uniform_weight,
                        "flow_weight": learned_weight, "uniform_syndrome_valid": True,
                        "flow_syndrome_valid": True,
                    })
                stream.flush()
    finally:
        torch.set_num_threads(previous_threads)
    uniform_weights, learned_weights = np.array(uniform_weights), np.array(learned_weights)
    summary = {
        "uniform": summarize_weights(uniform_weights), "flow": summarize_weights(learned_weights),
        "num_syndromes": num_syndromes, "proposals_per_syndrome": proposals_per_syndrome,
        "proposals_per_method": len(row_indices), "all_syndromes_valid": True,
        "flow_lower_weight_fraction": float(np.mean(learned_weights < uniform_weights)),
        "seed": seed, "batch_size": batch_size, "threads": threads,
        "selected_validation_indices": selected.tolist(),
        "dataset": str(dataset.resolve()), "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "fingerprints": actual,
        "scope": "proposal weights on model-selection validation syndromes; no decoding accuracy claim",
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _plot(uniform_weights, learned_weights, output / "proposal_weights.png")
    print(f"Checked {len(row_indices)} proposals per method: all syndromes valid.")
    print(f"{'method':<10} {'mean':>14} {'median':>14} {'minimum':>14} {'5th percentile':>16} {'95th percentile':>16}")
    for name in ("uniform", "flow"):
        stats = summary[name]
        print(f"{name:<10} {stats['mean']:14.6f} {stats['median']:14.6f} {stats['minimum']:14.6f} "
              f"{stats['percentile_5']:16.6f} {stats['percentile_95']:16.6f}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-syndromes", type=int, default=256)
    parser.add_argument("--proposals-per-syndrome", type=int, default=4)
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    try:
        output = validate(**vars(parser.parse_args(argv)))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Validation outputs saved to {output}")


if __name__ == "__main__":
    main()
