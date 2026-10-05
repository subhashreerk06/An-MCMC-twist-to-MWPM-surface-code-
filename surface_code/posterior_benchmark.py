"""Compare MWPM, posterior sampling, and posterior corrections on fresh shots.

Run with ``python -m surface_code.posterior_benchmark --flow-checkpoint MODEL
--output results/posterior_run``. Existing output paths are never overwritten.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import secrets
import time

import networkx as nx
import numpy as np
import scipy
from scipy.stats import binomtest
import stim
import torch

from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.flow_mcmc import _integer, _probability
from surface_code.mwpm_decoder import decode_mwpm
from surface_code.mwpm_posterior_corrector import PosteriorCorrectionPolicy
from surface_code.posterior_mcmc import PosteriorMCMCDecoder


SHOT_FIELDS = (
    "shot", "actual_logical", "syndrome_weight",
    "mwpm_prediction", "mwpm_mismatch", "mwpm_weight", "mwpm_seconds",
    "posterior_prediction", "posterior_mismatch", "posterior_p0", "posterior_p1",
    "posterior_confidence", "posterior_initial_weight", "posterior_minimum_weight_seen",
    "posterior_mean_retained_weight", "posterior_acceptance_rate",
    "posterior_flow_acceptance_rate", "posterior_uniform_acceptance_rate",
    "posterior_retained_samples", "posterior_logical_transitions", "posterior_logical_ess",
    "posterior_seconds", "posterior_diagnostic_status",
    "posterior_flow_proposals", "posterior_flow_accepted",
    "posterior_uniform_proposals", "posterior_uniform_accepted",
    "paired_outcome",
    "corrector_prediction", "corrector_mismatch", "corrector_seconds",
    "correction_layer_seconds", "correction_attempted", "correction_applied",
    "correction_diagnostic_pass", "corrector_raw_posterior_prediction",
)

CHANGED_DECISION_FIELDS = (
    "shot", "actual_logical", "mwpm_prediction", "posterior_prediction",
    "posterior_p1", "posterior_confidence", "posterior_logical_transitions",
    "posterior_logical_ess", "mwpm_weight", "posterior_minimum_weight_seen",
    "paired_outcome", "posterior_diagnostic_status",
)


def _paired_outcome(row):
    return {
        (True, True): "A", (False, True): "B",
        (True, False): "C", (False, False): "D",
    }[(row["mwpm_prediction"] == row["actual_logical"],
       row["posterior_prediction"] == row["actual_logical"])]


def _paired_analysis(rows, *, bootstrap_seed=0, bootstrap_resamples=10000):
    """Signed empirical changes; fractions use all shots, not just disagreements.

    A/B/C/D are mutually exclusive and exhaustive. Relative reduction is null
    when MWPM has no failures. Helping means strictly B > C on this sample,
    without implying statistical significance or out-of-sample improvement.

    The exact two-sided McNemar test conditions on B+C discordant shots,
    testing B ~ Binomial(B+C, 0.5). With no discordance its p-value is 1.
    Marginal 95% Wilson intervals are not used to test the paired difference.
    Bootstrap resampling preserves whole shot pairs: their failure differences
    are -1 (B), +1 (C), or 0 (A/D). Multinomial resampling of these counts is
    distributionally identical to resampling shot indices, without allocating
    a resamples-by-shots array. The interval uses 2.5/97.5 percentile quantiles.
    A constant observed difference gives a degenerate empirical bootstrap;
    this is flagged and is not evidence of zero population uncertainty.
    """
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    bootstrap_resamples = _integer(bootstrap_resamples, "bootstrap_resamples", 1)
    shots = len(rows)
    if not shots:
        raise ValueError("Paired analysis requires at least one shot.")
    counts = dict.fromkeys("ABCD", 0)
    for row in rows:
        counts[_paired_outcome(row)] += 1
    names = {"A": "both_correct", "B": "mwpm_failures_fixed",
             "C": "mwpm_correct_broken", "D": "both_wrong"}
    result = {}
    for category, name in names.items():
        result[name] = counts[category]
        result[f"{name}_fraction"] = counts[category] / shots
    mwpm_ler = (counts["B"] + counts["D"]) / shots
    posterior_ler = (counts["C"] + counts["D"]) / shots
    result.update({
        "paired_outcomes": {
            category: {"name": name, "count": counts[category], "fraction": counts[category] / shots}
            for category, name in names.items()
        },
        "mwpm_logical_error_rate": mwpm_ler,
        "posterior_logical_error_rate": posterior_ler,
        "absolute_ler_change": posterior_ler - mwpm_ler,
        "absolute_ler_reduction": mwpm_ler - posterior_ler,
        "relative_ler_reduction": (mwpm_ler - posterior_ler) / mwpm_ler if mwpm_ler > 0 else None,
        "net_failures_fixed": counts["B"] - counts["C"],
        "helping_on_tested_sample": counts["B"] > counts["C"],
    })
    fixed, broken = counts["B"], counts["C"]
    discordant = fixed + broken
    result.update({
        "discordant_fixed": fixed,
        "discordant_broken": broken,
        "mcnemar_exact_pvalue": float(binomtest(fixed, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant else 1.0,
    })
    for name, failures in (("mwpm", fixed + counts["D"]), ("posterior", broken + counts["D"])):
        interval = binomtest(failures, shots).proportion_ci(confidence_level=0.95, method="wilson")
        result[f"{name}_ler_wilson_ci95"] = {"low": float(interval.low), "high": float(interval.high)}
    # A dedicated stream never consumes Stim or MCMC random draws.
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    resampled = bootstrap_rng.multinomial(
        shots, [fixed / shots, broken / shots, (shots - discordant) / shots],
        size=bootstrap_resamples,
    )
    differences = (resampled[:, 1] - resampled[:, 0]) / shots
    low, high = np.quantile(differences, [0.025, 0.975], method="linear")
    constant_difference = max(fixed, broken, shots - discordant) == shots
    result["paired_bootstrap_ler_change_ci95"] = {
        "low": float(low), "high": float(high),
        "estimand": "posterior_LER - MWPM_LER",
        "method": "paired percentile bootstrap via multinomial shot-difference counts",
        "seed": bootstrap_seed, "resamples": bootstrap_resamples,
        "status": "constant_observed_difference" if constant_difference else "ok",
    }
    result["statistical_interpretation"] = (
        "helping_on_tested_sample describes only the observed B > C count; it is not a significance claim. "
        "The exact two-sided test assesses equal discordant probabilities. Negative LER differences favor "
        "posterior; use the paired interval and p-value to assess uncertainty. A degenerate bootstrap "
        "interval reflects the observed sample, not certainty about unseen shots."
    )
    return result


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_benchmark(
    *, flow_checkpoint: str | Path,
    distance: int = 3, rounds: int = 3, p: float = 0.005,
    shots: int = 100, iterations: int = 5000, burn_in: int = 1000,
    flow_probability: float = 0.9,
    stim_seed: int | None = None, mcmc_seed: int | None = None,
    bootstrap_seed: int = 0, bootstrap_resamples: int = 10000,
    correction_margin: float = 0.0, min_logical_transitions: int | None = 0,
    min_logical_ess: float | None = 0,
    output: str | Path | None = None, progress: bool = False,
) -> Path:
    """Write shot data, changed_decisions.csv, summary and logical traces.

    shots.npz records syndromes and actual_logical, indexed by the CSV shot ID.
    Each retained trace is stored as shot_NNNNNN.npy with diagnostics in a
    companion JSON. Undefined ESS is blank in CSV and null in diagnostic JSON.
    Component acceptance rates are zero if no proposals used that component.
    changed_decisions.csv contains only prediction disagreements, with the
    same column names as shots.csv, including shot ID and paired outcome.

    A separate NumPy RNG advances across independently initialized chains;
    neither MWPM corrections nor weights are passed to the posterior decoder.
    Exact ties invoke MWPM inside the decoder only after sampling. Posterior
    timing includes this possible tie-break and diagnostics, but excludes model
    loading, graph setup, Stim sampling and file output. Seeds are recorded.
    The correction layer reuses these independent results and the identical
    retained posterior trace. corrector_seconds is MWPM time + posterior time
    + correction_layer_seconds, representing its full decoding cost.
    Fixed gates are never tuned on these shots. With the default ESS gate of
    zero, undefined ESS still blocks corrections, as in the standalone decoder.

    Logical failure compares to Stim's observed L0, not to MWPM's prediction.
    The posterior targets the additive independent-component graph model; DEM
    component correlations are not enforced. Summary rate differences are
    empirical paired comparisons, not claims of statistical significance.
    """
    if output is None:
        output = Path("results") / datetime.now(timezone.utc).strftime("posterior_%Y%m%dT%H%M%S_%fZ")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    shots = _integer(shots, "shots", 1)
    correction_policy = PosteriorCorrectionPolicy(
        correction_margin, min_logical_transitions, min_logical_ess,
    )
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    bootstrap_resamples = _integer(bootstrap_resamples, "bootstrap_resamples", 1)
    iterations = _integer(iterations, "iterations", 1)
    burn_in = _integer(burn_in, "burn_in")
    if burn_in >= iterations:
        raise ValueError("burn_in must be less than iterations.")
    flow_probability = _probability(flow_probability)
    stim_seed = secrets.randbits(64) if stim_seed is None else _integer(stim_seed, "stim_seed")
    mcmc_seed = secrets.randbits(64) if mcmc_seed is None else _integer(mcmc_seed, "mcmc_seed")
    if stim_seed >= 2**64:
        raise ValueError("stim_seed must be in [0, 2**64).")
    rng = np.random.default_rng(mcmc_seed)
    circuit = build_surface_code(distance, rounds, p)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    if graph.num_observables != 1:
        raise ValueError("This benchmark requires exactly one logical observable, L0.")
    flow_checkpoint = Path(flow_checkpoint)
    decoder = PosteriorMCMCDecoder.from_checkpoint(
        graph, flow_checkpoint, flow_probability=flow_probability,
    )
    settings = {
        "distance": distance, "rounds": rounds, "p": p,
        "shots": shots, "iterations": iterations, "burn_in": burn_in,
        "flow_probability": flow_probability,
        "flow_checkpoint": str(flow_checkpoint.resolve()),
        "flow_checkpoint_sha256": hashlib.sha256(flow_checkpoint.read_bytes()).hexdigest(),
        "stim_seed": stim_seed, "mcmc_seed": mcmc_seed,
        "bootstrap_seed": bootstrap_seed, "bootstrap_resamples": bootstrap_resamples,
        "confidence_level": 0.95,
        "output": str(output.resolve()),
        "algorithms": ["mwpm", "posterior_mcmc", "mwpm_posterior_corrector"],
        **asdict(correction_policy),
        "correction_sampling": "reuse the same independently computed MWPM and posterior results",
        "correction_thresholds": "fixed inputs, never tuned on benchmark shots",
        "corrector_timing": "mwpm_seconds + posterior_seconds + correction_layer_seconds",
        "corrector_raw_prediction": "int(posterior_p1 > 0.5); exact tie is 0",
        "corrector_ess_gate": "undefined ESS fails an enabled gate, including threshold 0",
        "initialization": "independent uniform syndrome-compatible GF(2) draw per shot",
        "mcmc_rng": "one NumPy Generator stream across shots, separate from Stim",
        "retention": "every post-decision state after burn-in, including repeats; no reweighting",
        "tie_break": "MWPM prediction only for exact equal retained class counts",
        "ess": "at most 100 lags; stop before first nonpositive autocorrelation; constant trace ESS undefined",
        "timing_excludes": "circuit/graph setup, checkpoint loading, decoder construction, Stim sampling, file output",
        "posterior_timing_includes": "sampling, diagnostics and MWPM tie-break if needed",
        "dem_component_correlations_enforced": False,
        "num_detectors": graph.num_detectors, "num_edges": len(graph.edges),
        "num_observables": graph.num_observables,
        "versions": {"python": platform.python_version(), "stim": stim.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "networkx": nx.__version__, "torch": str(torch.__version__)},
    }
    # Atomic creation also protects against a path appearing during setup.
    output.mkdir(parents=True, exist_ok=False)
    traces = output / "logical_traces"
    traces.mkdir()
    _write_json(output / "settings.json", settings)
    syndromes, observables = sample_shots(circuit, shots, seed=stim_seed)
    np.savez_compressed(output / "shots.npz", syndromes=syndromes, actual_logical=observables[:, 0])
    rows = []
    with (output / "shots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SHOT_FIELDS)
        writer.writeheader()
        for shot, syndrome in enumerate(syndromes):
            actual = int(observables[shot, 0])
            start = time.perf_counter()
            mwpm = decode_mwpm(syndrome.copy(), graph)
            mwpm_seconds = time.perf_counter() - start
            start = time.perf_counter()
            posterior = decoder.decode(syndrome.copy(), iterations, burn_in=burn_in, rng=rng)
            posterior_seconds = time.perf_counter() - start
            start = time.perf_counter()
            corrected = correction_policy.apply(mwpm, posterior)
            correction_layer_seconds = time.perf_counter() - start
            row = {
                "shot": shot, "actual_logical": actual, "syndrome_weight": int(syndrome.sum()),
                "mwpm_prediction": mwpm.prediction, "mwpm_mismatch": int(mwpm.prediction != actual),
                "mwpm_weight": mwpm.total_weight, "mwpm_seconds": mwpm_seconds,
                "posterior_prediction": posterior.prediction,
                "posterior_mismatch": int(posterior.prediction != actual),
                "posterior_p0": posterior.posterior_p0, "posterior_p1": posterior.posterior_p1,
                "posterior_confidence": abs(posterior.posterior_p1 - 0.5),
                "posterior_initial_weight": posterior.initial_weight,
                "posterior_minimum_weight_seen": posterior.minimum_weight_seen,
                "posterior_mean_retained_weight": posterior.mean_retained_weight,
                "posterior_acceptance_rate": posterior.acceptance_rate,
                "posterior_retained_samples": posterior.retained_samples,
                "posterior_logical_transitions": posterior.logical_transitions,
                "posterior_logical_ess": posterior.logical_ess,
                "posterior_diagnostic_status": posterior.logical_diagnostic_status,
                "posterior_seconds": posterior_seconds,
                "corrector_prediction": corrected.final_prediction,
                "corrector_mismatch": int(corrected.final_prediction != actual),
                "corrector_seconds": mwpm_seconds + posterior_seconds + correction_layer_seconds,
                "correction_layer_seconds": correction_layer_seconds,
                "correction_attempted": int(corrected.correction_attempted),
                "correction_applied": int(corrected.correction_applied),
                "correction_diagnostic_pass": int(corrected.diagnostic_pass),
                "corrector_raw_posterior_prediction": corrected.posterior_prediction,
            }
            for component in ("flow", "uniform"):
                proposals = getattr(posterior, f"{component}_proposals")
                accepted = getattr(posterior, f"{component}_accepted")
                row[f"posterior_{component}_proposals"] = proposals
                row[f"posterior_{component}_accepted"] = accepted
                row[f"posterior_{component}_acceptance_rate"] = accepted / proposals if proposals else 0.0
            row["paired_outcome"] = _paired_outcome(row)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            posterior.save_logical_trace(traces / f"shot_{shot:06d}.npy")
            _write_json(traces / f"shot_{shot:06d}.json", asdict(posterior.logical_diagnostics))
            if progress:
                print(f"Completed shot {shot + 1}/{shots}", flush=True)
    with (output / "changed_decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CHANGED_DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in CHANGED_DECISION_FIELDS}
            for row in rows if row["mwpm_prediction"] != row["posterior_prediction"]
        )
    summary = {"shots": shots, **_paired_analysis(
        rows, bootstrap_seed=bootstrap_seed, bootstrap_resamples=bootstrap_resamples,
    )}
    for name in ("mwpm", "posterior", "corrector"):
        failures = sum(row[f"{name}_mismatch"] for row in rows)
        seconds = sum(row[f"{name}_seconds"] for row in rows)
        summary[name] = {"logical_mismatches": failures, "logical_error_rate": failures / shots,
                         "total_seconds": seconds, "mean_seconds": seconds / shots}
    rescued = summary["mwpm_failures_fixed"]
    harmed = summary["mwpm_correct_broken"]
    summary["paired_comparison"] = {
        "mwpm_wrong_posterior_correct": rescued,
        "mwpm_correct_posterior_wrong": harmed,
        "both_wrong": summary["both_wrong"],
        "both_correct": summary["both_correct"],
        "net_failures_reduced": rescued - harmed,
        "logical_error_rate_reduction": (rescued - harmed) / shots,
        "interpretation": "positive reduction favors posterior; empirical comparison only",
    }
    fixed = sum(row["mwpm_mismatch"] == 1 and row["corrector_mismatch"] == 0 for row in rows)
    broken = sum(row["mwpm_mismatch"] == 0 and row["corrector_mismatch"] == 1 for row in rows)
    net = fixed - broken
    outcome = "reduced_failures" if net > 0 else "increased_failures" if net < 0 else "no_net_difference"
    summary["corrector"].update({
        "correction_attempts": sum(row["correction_attempted"] for row in rows),
        "corrections_applied": sum(row["correction_applied"] for row in rows),
        "corrections_that_fixed_MWPM": fixed,
        "corrections_that_broke_MWPM": broken,
        "net_corrections": net,
        "outcome_on_tested_sample": outcome,
    })
    summary["correction_outcome"] = outcome
    summary["correction_summary"] = (
        f"On {shots} tested shots: MWPM had {summary['mwpm']['logical_mismatches']} failures; "
        f"MWPM + posterior correction had {summary['corrector']['logical_mismatches']}. "
        f"Corrections fixed {fixed} and broke {broken}; net corrections = {net}. "
        f"Outcome: {outcome}. This is an observed sample comparison, not a significance claim."
    )
    corrected_analysis = _paired_analysis(
        [{"actual_logical": row["actual_logical"], "mwpm_prediction": row["mwpm_prediction"],
          "posterior_prediction": row["corrector_prediction"]} for row in rows],
        bootstrap_seed=bootstrap_seed, bootstrap_resamples=bootstrap_resamples,
    )
    summary["corrector_vs_mwpm"] = {
        key: corrected_analysis[key] for key in (
            "discordant_fixed", "discordant_broken", "mcnemar_exact_pvalue",
            "absolute_ler_change", "absolute_ler_reduction", "relative_ler_reduction",
        )
    }
    summary["corrector_vs_mwpm"]["corrector_ler_wilson_ci95"] = corrected_analysis["posterior_ler_wilson_ci95"]
    summary["corrector_vs_mwpm"]["paired_bootstrap_ler_change_ci95"] = {
        **corrected_analysis["paired_bootstrap_ler_change_ci95"],
        "estimand": "corrector_LER - MWPM_LER",
    }
    # Put the sample-level outcome first for readers of summary.json.
    summary = {"correction_outcome": outcome, "correction_summary": summary["correction_summary"], **summary}
    _write_json(output / "summary.json", summary)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-probability", type=float, default=0.9)
    parser.add_argument("--correction-margin", type=float, default=0.0)
    parser.add_argument("--min-logical-transitions", type=int, default=0)
    parser.add_argument("--min-logical-ess", type=float, default=0.0)
    parser.add_argument("--stim-seed", type=int)
    parser.add_argument("--mcmc-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="Independent paired-bootstrap RNG seed (default: 0)")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000,
                        help="Number of paired bootstrap replicates (default: 10000)")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        output = run_benchmark(**vars(args), progress=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Results saved to {output}")
    print(json.loads((output / "summary.json").read_text())["correction_summary"])
    print((output / "summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
