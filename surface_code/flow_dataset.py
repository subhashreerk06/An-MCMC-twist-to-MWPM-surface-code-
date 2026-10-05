"""Generate conditional-coordinate data from the independent-edge error model.

For each edge i, draw e_i independently with probability graph.edges[i].p_error.
Thus P(e) = product_i p_i**e_i (1-p_i)**(1-e_i). For interior probabilities,
this equals a constant times exp(-sum_i w_i e_i), where
w_i = log((1-p_i)/p_i). Computing s = H e and the bijective affine coordinates
z therefore samples the joint P(s, z); conditioning on s gives exactly the
desired independent-edge conditional distribution for training q_theta(z | s).
Syndromes follow their physical marginal, rather than a uniform distribution.

Decomposed DEM components are intentionally sampled independently, including
components sharing a mechanism. This matches the additive edge-weight model,
not the correlated circuit distribution. No observable labels are sampled or
used for supervision. Coordinates are an encoding, not a proposal operation.

Run with ``python -m surface_code.flow_dataset --output flow_data/my_run``.
Existing output directories are never overwritten.
"""

import argparse
import hashlib
import json
from operator import index
from pathlib import Path

import numpy as np
import stim

from surface_code.circuit import build_surface_code, get_detector_error_model
from surface_code.decoding_graph import DecodingGraph, build_decoding_graph
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix


def _nonnegative_integer(value, name: str) -> int:
    try:
        if isinstance(value, (bool, np.bool_)) or index(value) < 0:
            raise ValueError
        return int(index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc


def sample_pairs(
    graph: DecodingGraph, samples: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 (syndromes, coordinates) from independent Bernoulli errors.

    Columns of e use graph.edges order. One uniform draw is consumed per edge,
    even when its probability is zero or one. No rejection or reweighting is
    applied. Zero samples and zero-dimensional coordinate spaces are supported.
    """
    samples = _nonnegative_integer(samples, "samples")
    probabilities = np.array([edge.p_error for edge in graph.edges], dtype=float)
    if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Edge probabilities must be finite and in [0, 1].")
    H = build_incidence_matrix(graph)
    coordinates = AffineCoordinates(H)
    syndromes = np.empty((samples, H.shape[0]), dtype=np.uint8)
    latent = np.empty((samples, coordinates.nullity), dtype=np.uint8)
    for row in range(samples):
        error = (rng.random(len(probabilities)) < probabilities).astype(np.uint8)
        syndrome = (H @ error) % 2
        syndromes[row] = syndrome
        latent[row] = coordinates.error_to_coordinates(syndrome, error)
    return syndromes, latent


def _json_fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    """Hash a JSON shape header, newline, then C-order uint8 matrix bytes."""
    header = json.dumps(list(matrix.shape), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + matrix.astype(np.uint8).tobytes(order="C")).hexdigest()


def generate_dataset(
    *, distance: int = 3, rounds: int = 3, p: float = 0.005,
    train_samples: int = 50000, validation_samples: int = 10000,
    seed: int = 314159, output: str | Path,
) -> Path:
    """Save samples.npz and reproducibility metadata in a new directory.

    Training is drawn first, then validation from the same advancing PCG64
    stream. Independent draws can legitimately produce duplicate pairs.
    Fingerprints describe ordered edge metadata and the exact H/N matrices;
    library versions are recorded because DEM generation can vary by version.
    """
    train_samples = _nonnegative_integer(train_samples, "train_samples")
    validation_samples = _nonnegative_integer(validation_samples, "validation_samples")
    seed = _nonnegative_integer(seed, "seed")
    graph = build_decoding_graph(get_detector_error_model(build_surface_code(distance, rounds, p)))
    H = build_incidence_matrix(graph)
    coordinates = AffineCoordinates(H)
    N = coordinates.nullspace_basis()
    ordered_edges = [
        {
            "edge_id": edge.edge_id, "detector_endpoints": edge.detector_endpoints,
            "p_error": float(edge.p_error).hex(), "logical_mask": edge.logical_mask,
            "mechanism_id": edge.mechanism_id, "component_index": edge.component_index,
        }
        for edge in graph.edges
    ]
    h_fingerprint = _matrix_fingerprint(H)
    n_fingerprint = _matrix_fingerprint(N)
    metadata = {
        "format_version": 1,
        "distance": distance, "rounds": rounds, "p": p,
        "train_samples": train_samples, "validation_samples": validation_samples,
        "num_graph_edges": len(graph.edges), "num_syndrome_bits": H.shape[0],
        "z_dimension": coordinates.nullity, "seed": seed,
        "edge_ordering": "graph.edges in expanded DEM traversal/component order",
        "edge_ordering_fingerprint": _json_fingerprint(ordered_edges),
        "h_fingerprint": h_fingerprint, "nullspace_fingerprint": n_fingerprint,
        "h_nullspace_fingerprint": _json_fingerprint([h_fingerprint, n_fingerprint]),
        "fingerprint_algorithm": "SHA-256; canonical JSON for edges; shape header + LF + uint8 C-order bytes for matrices",
        "free_columns": list(coordinates.free_columns),
        "coordinate_ordering": "ascending free column of H; zero-free-bit particular solution",
        "distribution": "independent Bernoulli(graph.edges[i].p_error)",
        "dem_component_correlations_enforced": False,
        "rng": "NumPy Generator(PCG64); training then validation",
        "versions": {"numpy": np.__version__, "stim": stim.__version__},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    train_s, train_z = sample_pairs(graph, train_samples, rng)
    validation_s, validation_z = sample_pairs(graph, validation_samples, rng)
    np.savez_compressed(
        output / "samples.npz", train_s=train_s, train_z=train_z,
        validation_s=validation_s, validation_z=validation_z,
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--validation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = generate_dataset(**vars(args))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Dataset saved to {output}")


if __name__ == "__main__":
    main()
