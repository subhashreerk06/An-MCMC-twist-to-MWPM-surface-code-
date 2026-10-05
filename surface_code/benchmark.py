"""Benchmark independent decoders on the same freshly sampled Stim shots.

Run with ``python -m surface_code.benchmark --output results/my_run``.
The output is a new run directory; existing directories are never overwritten.
Add --flow-checkpoint to compare all three methods. Without it, the original
two-decoder output format and behavior are preserved.
"""

import argparse
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from operator import index
from pathlib import Path
import platform
import secrets
import time

import networkx as nx
import numpy as np
import stim

from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.mwpm_decoder import decode_mwpm
from surface_code.random_mcmc import TraceEntry, decode_random_mcmc


SHOT_FIELDS = (
    "shot", "syndrome_weight", "actual_logical",
    "mwpm_prediction", "mwpm_mismatch", "mwpm_weight", "mwpm_seconds",
    "mcmc_prediction", "mcmc_mismatch", "mcmc_initial_weight",
    "mcmc_best_weight", "mcmc_final_weight", "mcmc_acceptance_rate",
    "mcmc_accepted_moves", "mcmc_seconds", "mcmc_best_minus_mwpm_weight",
)

# Keep legacy mcmc_* fields for existing readers and plotting commands.
THREE_METHOD_SHOT_FIELDS = SHOT_FIELDS + (
    "random_mcmc_prediction", "random_mcmc_initial_weight", "random_mcmc_best_weight",
    "random_mcmc_acceptance_rate", "random_mcmc_mismatch", "random_mcmc_seconds",
    "flow_mcmc_prediction", "flow_mcmc_initial_weight", "flow_mcmc_best_weight",
    "flow_mcmc_acceptance_rate", "flow_mcmc_flow_acceptance_rate",
    "flow_mcmc_uniform_acceptance_rate", "flow_mcmc_mismatch", "flow_mcmc_seconds",
    "flow_mcmc_final_weight", "flow_mcmc_accepted_moves",
    "flow_mcmc_flow_proposals", "flow_mcmc_uniform_proposals",
    "flow_mcmc_flow_accepted_moves", "flow_mcmc_uniform_accepted_moves",
    "random_best_minus_mwpm", "flow_best_minus_mwpm", "flow_best_minus_random_best",
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_benchmark(
    *, distance: int = 3, rounds: int = 3, p: float = 0.005,
    shots: int = 100, iterations: int = 5000,
    stim_seed: int | None = None, mcmc_seed: int | None = None,
    output: str | Path | None = None, progress: bool = False,
    flow_checkpoint: str | Path | None = None, flow_probability: float = 0.9,
    flow_seed: int | None = None,
) -> Path:
    """Save shot metrics, decoder traces, settings, and summary in a new run.

    MCMC receives only the measured syndrome, graph, iteration budget, and its
    own RNG. Its RNG is seeded once and advances across shots. The Stim stream
    is separate. Unspecified seeds are generated and recorded for replay; timing
    measurements are not reproducible. Decoder timings include per-shot random
    initialization and exclude circuit setup, checkpoint loading, reusable
    flow-decoder construction, shot sampling, and file output.

    With a flow checkpoint, each MCMC starts from its own uniform GF(2) draw.
    The flow RNG is independent and advances across shots; its seed is derived
    from mcmc_seed without drawing from the baseline stream, unless flow_seed is
    supplied. Neither MCMC receives matching output. Only benchmark reporting
    compares their weights afterward. Legacy mcmc_* columns and traces/ remain;
    explicit random_mcmc_*/flow_mcmc_* columns and flow_traces/ are added.

    All decoders use the additive component-edge objective, which does not
    impose the DEM's correlated-component groups. Logical mismatch compares
    each prediction directly to the shot's measured L0 observable flip.
    """
    for name, value, minimum in (("shots", shots, 1), ("iterations", iterations, 0)):
        try:
            if isinstance(value, (bool, np.bool_)) or index(value) < minimum:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    shots, iterations = int(shots), int(iterations)
    if not np.isfinite(flow_probability) or not 0 <= flow_probability <= 1:
        raise ValueError("flow_probability must be finite and in [0, 1].")
    stim_seed = secrets.randbits(64) if stim_seed is None else index(stim_seed)
    mcmc_seed = secrets.randbits(64) if mcmc_seed is None else index(mcmc_seed)
    if not 0 <= stim_seed < 2**64:
        raise ValueError("stim_seed must be in [0, 2**64).")
    rng = np.random.default_rng(mcmc_seed)
    circuit = build_surface_code(distance, rounds, p)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    if graph.num_observables != 1:
        raise ValueError("This benchmark requires exactly one logical observable, L0.")
    if any(not np.isfinite(edge.weight) for edge in graph.edges):
        raise ValueError("The MCMC experiment requires finite edge weights.")
    flow_decoder = None
    if flow_checkpoint is not None:
        # Optional import keeps the original benchmark usable without PyTorch.
        import torch
        from surface_code.flow_mcmc import FlowProposalMCMCDecoder, FlowTraceEntry

        if flow_seed is None:
            flow_seed = int(np.random.SeedSequence(mcmc_seed, spawn_key=(1,)).generate_state(1, dtype=np.uint64)[0])
        else:
            flow_seed = index(flow_seed)
        if not 0 <= flow_seed < 2**64:
            raise ValueError("flow_seed must be in [0, 2**64).")
        flow_rng = np.random.default_rng(flow_seed)
        flow_checkpoint = Path(flow_checkpoint)
        flow_decoder = FlowProposalMCMCDecoder.from_checkpoint(
            graph, flow_checkpoint, flow_probability=flow_probability,
        )
    if output is None:
        run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        output = Path("results") / run_name
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    traces = output / "traces"
    traces.mkdir()
    if flow_decoder is not None:
        flow_traces = output / "flow_traces"
        flow_traces.mkdir()
    settings = {
        "distance": distance, "rounds": rounds, "p": p,
        "shots": shots, "iterations": iterations,
        "stim_seed": stim_seed, "mcmc_seed": mcmc_seed,
        "output": str(output.resolve()),
        "algorithms": ["mwpm", "random_start_probability_weighted_mcmc"],
        "mcmc_rng": "one NumPy Generator stream shared across shots",
        "weight_objective": "sum of selected component-edge log((1-p_e)/p_e)",
        "dem_component_correlations_enforced": False,
        "num_detectors": graph.num_detectors, "num_edges": len(graph.edges),
        "num_observables": graph.num_observables,
        "versions": {
            "python": platform.python_version(), "stim": stim.__version__,
            "numpy": np.__version__, "networkx": nx.__version__,
        },
    }
    if flow_decoder is not None:
        settings.update({
            "flow_checkpoint": str(flow_checkpoint.resolve()),
            "flow_checkpoint_sha256": hashlib.sha256(flow_checkpoint.read_bytes()).hexdigest(),
            "flow_probability": flow_probability, "flow_seed": flow_seed,
            "flow_rng": "separate NumPy Generator stream shared across flow shots",
            "mcmc_initialization": "independent uniform GF(2) draws for both MCMC methods",
            "flow_best_configuration_policy": "initial_and_accepted_states",
            "timing_excludes": "circuit/graph setup, checkpoint loading, reusable flow-decoder construction, shot sampling, file output",
        })
        settings["algorithms"].append("random_start_flow_assisted_mcmc")
        settings["versions"]["torch"] = str(torch.__version__)
    _write_json(output / "settings.json", settings)
    syndromes, observables = sample_shots(circuit, shots, seed=stim_seed)
    rows = []
    with (output / "shots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=THREE_METHOD_SHOT_FIELDS if flow_decoder is not None else SHOT_FIELDS)
        writer.writeheader()
        for shot, syndrome in enumerate(syndromes):
            actual = int(observables[shot, 0])
            start = time.perf_counter()
            mwpm = decode_mwpm(syndrome.copy(), graph)
            mwpm_seconds = time.perf_counter() - start

            start = time.perf_counter()
            mcmc = decode_random_mcmc(syndrome.copy(), graph, iterations, rng=rng)
            mcmc_seconds = time.perf_counter() - start

            if flow_decoder is not None:
                start = time.perf_counter()
                flow = flow_decoder.decode(syndrome.copy(), iterations, rng=flow_rng)
                flow_seconds = time.perf_counter() - start

            row = {
                "shot": shot, "syndrome_weight": int(syndrome.sum()),
                "actual_logical": actual,
                "mwpm_prediction": mwpm.prediction,
                "mwpm_mismatch": int(mwpm.prediction != actual),
                "mwpm_weight": mwpm.total_weight, "mwpm_seconds": mwpm_seconds,
                "mcmc_prediction": mcmc.prediction,
                "mcmc_mismatch": int(mcmc.prediction != actual),
                "mcmc_initial_weight": mcmc.initial_weight,
                "mcmc_best_weight": mcmc.best_weight,
                "mcmc_final_weight": mcmc.final_weight,
                "mcmc_acceptance_rate": mcmc.acceptance_rate,
                "mcmc_accepted_moves": mcmc.accepted_moves,
                "mcmc_seconds": mcmc_seconds,
                "mcmc_best_minus_mwpm_weight": mcmc.best_weight - mwpm.total_weight,
            }
            if flow_decoder is not None:
                row.update({
                    "random_mcmc_prediction": mcmc.prediction,
                    "random_mcmc_initial_weight": mcmc.initial_weight,
                    "random_mcmc_best_weight": mcmc.best_weight,
                    "random_mcmc_acceptance_rate": mcmc.acceptance_rate,
                    "random_mcmc_mismatch": int(mcmc.prediction != actual),
                    "random_mcmc_seconds": mcmc_seconds,
                    "flow_mcmc_prediction": flow.prediction,
                    "flow_mcmc_initial_weight": flow.initial_weight,
                    "flow_mcmc_best_weight": flow.best_weight,
                    "flow_mcmc_acceptance_rate": flow.acceptance_rate,
                    "flow_mcmc_flow_acceptance_rate": flow.flow_acceptance_rate,
                    "flow_mcmc_uniform_acceptance_rate": flow.uniform_acceptance_rate,
                    "flow_mcmc_mismatch": int(flow.prediction != actual),
                    "flow_mcmc_seconds": flow_seconds,
                    "flow_mcmc_final_weight": flow.final_weight,
                    "flow_mcmc_accepted_moves": flow.accepted_moves,
                    "flow_mcmc_flow_proposals": flow.flow_proposals,
                    "flow_mcmc_uniform_proposals": flow.uniform_proposals,
                    "flow_mcmc_flow_accepted_moves": flow.flow_accepted_moves,
                    "flow_mcmc_uniform_accepted_moves": flow.uniform_accepted_moves,
                    "random_best_minus_mwpm": mcmc.best_weight - mwpm.total_weight,
                    "flow_best_minus_mwpm": flow.best_weight - mwpm.total_weight,
                    "flow_best_minus_random_best": flow.best_weight - mcmc.best_weight,
                })
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            with (traces / f"shot_{shot:06d}.csv").open("w", newline="") as trace_stream:
                trace_writer = csv.DictWriter(
                    trace_stream, fieldnames=[field.name for field in fields(TraceEntry)]
                )
                trace_writer.writeheader()
                trace_writer.writerows(asdict(entry) for entry in mcmc.trace)
            if flow_decoder is not None:
                with (flow_traces / f"shot_{shot:06d}.csv").open("w", newline="") as trace_stream:
                    trace_writer = csv.DictWriter(
                        trace_stream, fieldnames=[field.name for field in fields(FlowTraceEntry)],
                    )
                    trace_writer.writeheader()
                    trace_writer.writerows(asdict(entry) for entry in flow.trace)
            if progress:
                print(f"Completed shot {shot + 1}/{shots}", flush=True)

    def mean(field: str) -> float:
        return float(np.mean([row[field] for row in rows]))

    summary = {"shots": shots}
    for decoder in ("mwpm", "mcmc"):
        summary[decoder] = {
            "logical_mismatches": sum(row[f"{decoder}_mismatch"] for row in rows),
            "logical_error_rate": mean(f"{decoder}_mismatch"),
            "mean_seconds": mean(f"{decoder}_seconds"),
            "total_seconds": sum(row[f"{decoder}_seconds"] for row in rows),
        }
    summary["mwpm"]["mean_weight"] = mean("mwpm_weight")
    summary["mcmc"].update({
        "mean_initial_weight": mean("mcmc_initial_weight"),
        "mean_best_weight": mean("mcmc_best_weight"),
        "mean_final_weight": mean("mcmc_final_weight"),
        "mean_acceptance_rate": mean("mcmc_acceptance_rate"),
        "accepted_moves": sum(row["mcmc_accepted_moves"] for row in rows),
    })
    summary["mean_mcmc_best_minus_mwpm_weight"] = mean("mcmc_best_minus_mwpm_weight")
    if flow_decoder is not None:
        summary["random_mcmc"] = summary["mcmc"].copy()
        summary["flow_mcmc"] = {
            "logical_mismatches": sum(row["flow_mcmc_mismatch"] for row in rows),
            "logical_error_rate": mean("flow_mcmc_mismatch"),
            "mean_seconds": mean("flow_mcmc_seconds"),
            "total_seconds": sum(row["flow_mcmc_seconds"] for row in rows),
            "mean_initial_weight": mean("flow_mcmc_initial_weight"),
            "mean_best_weight": mean("flow_mcmc_best_weight"),
            "mean_final_weight": mean("flow_mcmc_final_weight"),
            "mean_acceptance_rate": mean("flow_mcmc_acceptance_rate"),
            "mean_flow_acceptance_rate": mean("flow_mcmc_flow_acceptance_rate"),
            "mean_uniform_acceptance_rate": mean("flow_mcmc_uniform_acceptance_rate"),
            "accepted_moves": sum(row["flow_mcmc_accepted_moves"] for row in rows),
        }
        for component in ("flow", "uniform"):
            proposals = sum(row[f"flow_mcmc_{component}_proposals"] for row in rows)
            accepted = sum(row[f"flow_mcmc_{component}_accepted_moves"] for row in rows)
            summary["flow_mcmc"][f"{component}_proposals"] = proposals
            summary["flow_mcmc"][f"{component}_accepted_moves"] = accepted
            summary["flow_mcmc"][f"{component}_acceptance_rate"] = accepted / proposals if proposals else 0.0
        for comparison in ("random_best_minus_mwpm", "flow_best_minus_mwpm", "flow_best_minus_random_best"):
            summary[f"mean_{comparison}"] = mean(comparison)
    _write_json(output / "summary.json", summary)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--stim-seed", type=int)
    parser.add_argument("--mcmc-seed", type=int)
    parser.add_argument("--flow-checkpoint", type=Path, help="Enable the third decoder using this trained checkpoint")
    parser.add_argument("--flow-probability", type=float, default=0.9)
    parser.add_argument("--flow-seed", type=int, help="Flow RNG seed (default: derived independently from --mcmc-seed)")
    parser.add_argument("--output", type=Path, help="New run directory (default: results/run_<UTC timestamp>)")
    args = parser.parse_args(argv)
    try:
        output = run_benchmark(**vars(args), progress=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Results saved to {output}")
    print((output / "summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
