"""Plot completed benchmark results; no decoder is invoked by this module.

Example: python -m surface_code.plotting results/my_run --shot 0 --mwpm-reference
Three-method runs automatically produce the five comparison plots; their
convergence plot always includes the saved MWPM reference. Posterior runs
automatically produce six accuracy and sampling-diagnostic plots.
"""

import argparse
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def _frame(data, required: tuple[str, ...]) -> pd.DataFrame:
    """Accept a CSV path, DataFrame, or sequence of trace records."""
    frame = pd.read_csv(data) if isinstance(data, (str, Path)) else pd.DataFrame(data)
    if frame.empty and not len(frame.columns):
        frame = pd.DataFrame(columns=required)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing plot columns: {', '.join(sorted(missing))}")
    return frame


def _save(figure: Figure, output_dir: str | Path, filename: str) -> Path:
    output = Path(output_dir) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure).print_figure(output, dpi=160, bbox_inches="tight")
    return output


def plot_convergence(
    trace, shot: int, *, output_dir: str | Path = ".",
    mwpm_weight: float | None = None,
) -> Path:
    """Save current and best weights from one completed MCMC trace.

    trace may be a CSV path, DataFrame, or sequence of TraceEntry records.
    mwpm_weight is an optional, independently calculated plotting reference.
    It only draws a horizontal line and never affects sampling or decoding.
    """
    frame = _frame(trace, ("iteration", "current_weight", "best_weight_so_far"))
    frame = frame.sort_values("iteration")
    figure = Figure(figsize=(8, 5), layout="constrained")
    ax = figure.subplots()
    ax.plot(frame["iteration"], frame["current_weight"], label="MCMC current weight", color="tab:blue")
    ax.plot(
        frame["iteration"], frame["best_weight_so_far"],
        label="MCMC best weight so far", color="tab:orange", linestyle="--",
    )
    if mwpm_weight is not None:
        ax.axhline(mwpm_weight, label="MWPM weight (reference)", color="tab:green", linestyle=":")
    if frame.empty:
        ax.text(0.5, 0.5, "No MCMC iterations recorded", transform=ax.transAxes, ha="center")
    ax.set(xlabel="MCMC iteration", ylabel="Configuration weight W(E)", title=f"MCMC convergence — shot {shot}")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(figure, output_dir, f"convergence_shot_{shot}.png")


def plot_weight_comparison(shots, *, output_dir: str | Path = ".") -> Path:
    """Save MWPM and MCMC best weights by shot from completed shot records."""
    frame = _frame(shots, ("shot", "mwpm_weight", "mcmc_best_weight"))
    frame = frame.sort_values("shot")
    figure = Figure(figsize=(8, 5), layout="constrained")
    ax = figure.subplots()
    ax.plot(frame["shot"], frame["mwpm_weight"], label="MWPM weight", marker="o", color="tab:green")
    ax.plot(frame["shot"], frame["mcmc_best_weight"], label="MCMC best weight", marker="o", color="tab:orange")
    ax.set(xlabel="Shot number", ylabel="Weight W(E)", title="Independent decoder weight comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(figure, output_dir, "weight_comparison.png")


def plot_convergence_comparison(
    random_trace, flow_trace, shot: int, *, mwpm_weight: float,
    random_initial_weight: float | None = None, flow_initial_weight: float | None = None,
    output_dir: str | Path = ".",
) -> Path:
    """Compare recorded best weights, with optional initialization at step 0.

    New flow runs track only initial and accepted chain states. Historical
    runs may include rejected proposals; saved traces are plotted unchanged.
    These curves are not current-state energy or evidence of chain mixing.
    """
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    for data, initial, label, color in (
        (random_trace, random_initial_weight, "Uniform-proposal MCMC", "tab:blue"),
        (flow_trace, flow_initial_weight, "Flow-assisted MCMC", "tab:orange"),
    ):
        frame = _frame(data, ("iteration", "best_weight_so_far")).sort_values("iteration")
        x, y = frame["iteration"].tolist(), frame["best_weight_so_far"].tolist()
        if initial is not None:
            x, y = [0, *x], [initial, *y]
        ax.plot(x, y, label=label, color=color, drawstyle="steps-post", marker="o", markersize=3)
    ax.axhline(mwpm_weight, color="tab:green", linestyle="--", label="MWPM weight (reference)")
    ax.set(xlabel="MCMC iteration", ylabel="Best weight so far", title=f"Best recorded weight — shot {shot}")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(figure, output_dir, f"convergence_comparison_shot_{shot}.png")


def plot_proposal_weight_histogram(random_trace, flow_trace, *, output_dir: str | Path = ".") -> Path:
    """Compare all uniform-baseline proposals against flow-source proposals.

    Include accepted AND rejected proposals and exclude initialization. Uniform
    components of the flow mixture are not labeled as flow proposals. Use
    shared bins and density normalization because sample counts can differ.
    Inputs may pool traces across all shots from the same benchmark.
    """
    uniform = _frame(random_trace, ("proposed_weight",))
    mixture = _frame(flow_trace, ("proposed_weight", "proposal_source"))
    if not mixture["proposal_source"].isin(("flow", "uniform")).all():
        raise ValueError("proposal_source must be flow or uniform.")
    uniform_weights = uniform["proposed_weight"].to_numpy(dtype=float)
    flow_weights = mixture.loc[mixture["proposal_source"] == "flow", "proposed_weight"].to_numpy(dtype=float)
    combined = np.concatenate((uniform_weights, flow_weights))
    if not np.isfinite(combined).all():
        raise ValueError("Proposal weights must be finite.")
    bins = np.histogram_bin_edges(combined, bins=50) if combined.size else np.linspace(0, 1, 51)
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    for values, label, color in ((uniform_weights, "Uniform proposals", "tab:blue"),
                                 (flow_weights, "Flow proposals", "tab:orange")):
        if values.size:
            ax.hist(values, bins=bins, density=True, alpha=0.65, label=f"{label} (n={values.size})", color=color)
    if not combined.size:
        ax.text(0.5, 0.5, "No proposals recorded", transform=ax.transAxes, ha="center")
    else:
        ax.legend()
        if not flow_weights.size:
            ax.text(0.5, 0.9, "No flow-source proposals recorded", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Proposed configuration weight W(E)", ylabel="Probability density",
           title="Proposal weights across all shots (accepted and rejected)")
    ax.grid(axis="y", alpha=0.25)
    return _save(figure, output_dir, "proposal_weight_histogram.png")


def plot_acceptance_comparison(shots, *, output_dir: str | Path = ".") -> Path:
    """Compare mean per-shot chain acceptance, including both mixture sources."""
    frame = _frame(shots, ("random_mcmc_acceptance_rate", "flow_mcmc_acceptance_rate"))
    values = frame[["random_mcmc_acceptance_rate", "flow_mcmc_acceptance_rate"]].to_numpy(dtype=float)
    if not len(frame) or not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
        raise ValueError("Acceptance rates must be nonempty, finite, and in [0, 1].")
    rates = values.mean(axis=0)
    figure = Figure(figsize=(7, 5), layout="constrained")
    ax = figure.subplots()
    bars = ax.bar(["Uniform MCMC", "Flow-MCMC"], rates, color=["tab:blue", "tab:orange"])
    ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=4)
    ax.set(ylabel="Mean per-shot acceptance rate", ylim=(0, 1.1), title=f"Chain acceptance — {len(frame)} shots")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return _save(figure, output_dir, "acceptance_rate_comparison.png")


def plot_three_method_weights(shots, *, output_dir: str | Path = ".") -> Path:
    """Compare saved MWPM and both MCMC best weights on each shot."""
    frame = _frame(shots, ("shot", "mwpm_weight", "random_mcmc_best_weight", "flow_mcmc_best_weight"))
    frame = frame.sort_values("shot")
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    for column, label, color, marker in (
        ("mwpm_weight", "MWPM", "tab:green", "o"),
        ("random_mcmc_best_weight", "Random MCMC", "tab:blue", "s"),
        ("flow_mcmc_best_weight", "Flow-MCMC", "tab:orange", "x"),
    ):
        ax.plot(frame["shot"], frame[column], label=label, color=color, marker=marker)
    ax.set(xlabel="Shot", ylabel="Best weight", title="Best configuration weight per shot")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(figure, output_dir, "per_shot_best_weight.png")


def plot_logical_error_summary(shots, *, output_dir: str | Path = ".") -> Path:
    """Plot empirical logical mismatch fractions, labeled with counts and N."""
    columns = ("mwpm_mismatch", "random_mcmc_mismatch", "flow_mcmc_mismatch")
    frame = _frame(shots, columns)
    values = frame[list(columns)].to_numpy(dtype=float)
    if not len(frame) or not np.all((values == 0) | (values == 1)):
        raise ValueError("Logical mismatches must be nonempty binary values.")
    counts = values.sum(axis=0).astype(int)
    rates = counts / len(frame)
    figure = Figure(figsize=(8, 5), layout="constrained")
    ax = figure.subplots()
    bars = ax.bar(["MWPM", "Random MCMC", "Flow-MCMC"], rates,
                  color=["tab:green", "tab:blue", "tab:orange"])
    ax.bar_label(bars, labels=[f"{count}/{len(frame)} ({rate:.1%})" for count, rate in zip(counts, rates)], padding=4)
    ax.set(ylabel="Logical error rate", ylim=(0, 1.1), title=f"Empirical logical error rates — {len(frame)} shots")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return _save(figure, output_dir, "logical_error_rate_summary.png")


def plot_three_method_benchmark(run_dir: str | Path, *, shot: int = 0, output_dir=None) -> tuple[Path, ...]:
    """Create five comparisons using saved files only; never invoke decoders.

    Convergence uses the selected shot. All other plots aggregate every saved
    shot (histogram: every recorded proposal). Raw CSVs and summaries are read
    only. By default plots are saved under <run_dir>/plots/.
    """
    run_dir = Path(run_dir)
    output_dir = run_dir / "plots" if output_dir is None else Path(output_dir)
    shots = _frame(run_dir / "shots.csv", (
        "shot", "mwpm_weight", "random_mcmc_initial_weight", "flow_mcmc_initial_weight",
        "random_mcmc_best_weight", "flow_mcmc_best_weight",
        "random_mcmc_acceptance_rate", "flow_mcmc_acceptance_rate",
        "mwpm_mismatch", "random_mcmc_mismatch", "flow_mcmc_mismatch",
    ))
    selected = shots.loc[shots["shot"] == shot]
    if len(selected) != 1 or shots["shot"].duplicated().any():
        raise ValueError("Expected unique shot IDs and exactly one selected shot.")
    random_traces, flow_traces = {}, {}
    for shot_id in shots["shot"]:
        name = f"shot_{int(shot_id):06d}.csv"
        random_traces[shot_id] = _frame(run_dir / "traces" / name, ("iteration", "best_weight_so_far", "proposed_weight"))
        flow_traces[shot_id] = _frame(run_dir / "flow_traces" / name, ("iteration", "best_weight_so_far", "proposed_weight", "proposal_source"))
    chosen = selected.iloc[0]
    return (
        plot_convergence_comparison(
            random_traces[shot], flow_traces[shot], shot, output_dir=output_dir,
            mwpm_weight=float(chosen["mwpm_weight"]),
            random_initial_weight=float(chosen["random_mcmc_initial_weight"]),
            flow_initial_weight=float(chosen["flow_mcmc_initial_weight"]),
        ),
        plot_proposal_weight_histogram(pd.concat(random_traces.values()), pd.concat(flow_traces.values()), output_dir=output_dir),
        plot_acceptance_comparison(shots, output_dir=output_dir),
        plot_three_method_weights(shots, output_dir=output_dir),
        plot_logical_error_summary(shots, output_dir=output_dir),
    )


POSTERIOR_METHODS = ("mwpm", "posterior", "corrector")
POSTERIOR_LABELS = ("MWPM", "Posterior MCMC", "MWPM + posterior\ncorrection")
POSTERIOR_COLORS = ("tab:blue", "tab:orange", "tab:purple")


def _posterior_binary_frame(shots, columns):
    frame = _frame(shots, columns)
    values = frame[list(columns)].to_numpy(dtype=float)
    if not len(frame) or not np.all((values == 0) | (values == 1)):
        raise ValueError("Posterior plot outcomes must be nonempty binary values.")
    return frame


def plot_posterior_logical_error_rates(shots, *, output_dir=".") -> Path:
    """Primary outcome: empirical failure rates with marginal 95% Wilson CIs.

    Intervals are recomputed from saved shot counts using the benchmark's
    Wilson method. Their overlap is not a paired significance test; see the
    paired tests and difference intervals in summary.json.
    """
    from scipy.stats import binomtest

    columns = tuple(f"{method}_mismatch" for method in POSTERIOR_METHODS)
    frame = _posterior_binary_frame(shots, columns)
    counts = frame[list(columns)].sum().to_numpy(dtype=int)
    rates = counts / len(frame)
    intervals = [binomtest(int(count), len(frame)).proportion_ci(0.95, method="wilson") for count in counts]
    errors = np.maximum(0, np.array([
        rates - [ci.low for ci in intervals], [ci.high for ci in intervals] - rates,
    ]))
    figure = Figure(figsize=(9, 6), layout="constrained")
    ax = figure.subplots()
    bars = ax.bar(POSTERIOR_LABELS, rates, color=POSTERIOR_COLORS, yerr=errors, capsize=6)
    for bar, count, ci in zip(bars, counts, intervals):
        ax.annotate(f"{count}/{len(frame)}", (bar.get_x() + bar.get_width()/2, ci.high),
                    xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set(ylabel="Logical error rate", ylim=(0, min(1.12, max(ci.high for ci in intervals) * 1.2 + 0.01)),
           title=f"Logical decoding accuracy — {len(frame)} shots\n95% Wilson confidence intervals")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.supxlabel("Paired comparison uncertainty is reported in summary.json", fontsize=9)
    return _save(figure, output_dir, "posterior_logical_error_rates.png")


def plot_posterior_fixed_vs_broken(shots, *, output_dir=".") -> Path:
    """Show both standalone posterior changes and gated corrector changes."""
    frame = _posterior_binary_frame(shots, tuple(f"{m}_mismatch" for m in POSTERIOR_METHODS))
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    positions = np.arange(2)
    for offset, wrong, label, color in (
        (-0.2, 1, "MWPM failures fixed", "tab:green"),
        (0.2, 0, "MWPM correct decisions broken", "tab:red"),
    ):
        counts = [int(((frame["mwpm_mismatch"] == wrong) & (frame[f"{m}_mismatch"] == 1-wrong)).sum())
                  for m in ("posterior", "corrector")]
        bars = ax.bar(positions + offset, counts, width=0.4, label=label, color=color)
        ax.bar_label(bars, padding=3)
    ax.set(xticks=positions, xticklabels=POSTERIOR_LABELS[1:], ylabel="Number of shots",
           title="Changes to MWPM decisions: fixed versus broken")
    ax.set_ylim(0, max(1, max(patch.get_height() for patch in ax.patches)) * 1.25)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return _save(figure, output_dir, "posterior_fixed_vs_broken.png")


def _posterior_probabilities(frame):
    values = frame["posterior_p1"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
        raise ValueError("posterior_p1 must be finite and in [0, 1].")
    return values


def plot_posterior_changed_confidence(shots, *, output_dir=".") -> Path:
    """All standalone posterior disagreements with MWPM, before quality gates.

    Beneficial/harmful uses the actual logical outcome, not energy or confidence.
    A posterior confidence is distance from 0.5, not a statistical confidence level.
    """
    frame = _posterior_binary_frame(shots, ("actual_logical", "mwpm_prediction", "posterior_prediction"))
    frame = _frame(frame, ("posterior_p1",))
    confidence = np.abs(_posterior_probabilities(frame) - 0.5)
    changed = (frame["mwpm_prediction"] != frame["posterior_prediction"]).to_numpy()
    correct = (frame["posterior_prediction"] == frame["actual_logical"]).to_numpy()
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    for mask, label, color in ((changed & correct, "Beneficial correction", "tab:green"),
                                (changed & ~correct, "Harmful correction", "tab:red")):
        ax.hist(confidence[mask], bins=np.linspace(0, 0.5, 21), alpha=0.6,
                label=f"{label} (n={int(mask.sum())})", color=color)
    if not changed.any():
        ax.text(0.5, 0.5, "No changed decisions", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Posterior confidence = abs(P(L=1 | s) - 0.5)", ylabel="Number of shots",
           xlim=(0, 0.5), title="MWPM → posterior changed decisions (before correction gates)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(figure, output_dir, "posterior_changed_confidence.png")


def plot_posterior_probability_vs_ess(shots, *, output_dir=".") -> Path:
    frame = _posterior_binary_frame(shots, ("posterior_mismatch",))
    frame = _frame(frame, ("posterior_p1", "posterior_logical_ess"))
    probabilities = _posterior_probabilities(frame)
    ess = frame["posterior_logical_ess"].to_numpy(dtype=float)
    if np.any(np.isinf(ess) | (ess < 0)):
        raise ValueError("Logical ESS must be nonnegative and finite, or missing when undefined.")
    defined = np.isfinite(ess)
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    for mismatch, label, color, marker in ((0, "Posterior correct", "tab:blue", "o"),
                                           (1, "Posterior wrong", "tab:red", "x")):
        mask = defined & (frame["posterior_mismatch"].to_numpy() == mismatch)
        ax.scatter(probabilities[mask], ess[mask], label=f"{label} (n={int(mask.sum())})",
                   color=color, marker=marker, alpha=0.7)
    ax.text(0.02, 0.98, f"Undefined ESS omitted: {int((~defined).sum())}/{len(frame)} shots",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set(xlabel="posterior_p1 = P(L=1 | s)", ylabel="Logical effective sample size (ESS)",
           xlim=(-0.02, 1.02), title="Logical posterior probability and sampling ESS")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center")
    ax.grid(alpha=0.25)
    return _save(figure, output_dir, "posterior_probability_vs_ess.png")


def plot_posterior_logical_transitions(shots, *, output_dir=".") -> Path:
    frame = _frame(shots, ("posterior_logical_transitions",))
    transitions = frame["posterior_logical_transitions"].to_numpy(dtype=float)
    if not len(frame) or not np.all(np.isfinite(transitions) & (transitions >= 0) & (transitions == np.floor(transitions))):
        raise ValueError("Logical transitions must be nonempty nonnegative integers.")
    # Integer-aligned bins, capped at 50 even for long chains.
    width = max(1, int(np.ceil((transitions.max() + 1) / 50)))
    bins = np.arange(-0.5, transitions.max() + width + 0.5, width)
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    ax.hist(transitions, bins=bins, color="tab:blue", edgecolor="white")
    ax.set(xlabel="Logical-class transitions in retained trace", ylabel="Number of shots",
           title="Logical-class transitions per shot (repeated states retained)")
    ax.grid(axis="y", alpha=0.25)
    return _save(figure, output_dir, "posterior_logical_transitions.png")


def plot_posterior_runtime(shots, *, output_dir=".") -> Path:
    columns = tuple(f"{m}_seconds" for m in POSTERIOR_METHODS)
    frame = _frame(shots, columns)
    values = frame[list(columns)].to_numpy(dtype=float)
    if not len(frame) or not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("Runtimes must be nonempty, finite, and nonnegative.")
    means = values.mean(axis=0)
    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    bars = ax.bar(POSTERIOR_LABELS, means, color=POSTERIOR_COLORS)
    ax.bar_label(bars, labels=[f"{value:.4g} s" for value in means], padding=4)
    ax.set(ylabel="Mean decoding runtime per shot (seconds)",
           ylim=(0, max(1e-9, means.max()) * 1.2), title="Decoder runtime comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.supxlabel("Correction runtime includes MWPM + posterior sampling + correction overhead", fontsize=9)
    return _save(figure, output_dir, "posterior_runtime_comparison.png")


def plot_posterior_benchmark(run_dir: str | Path, *, output_dir=None) -> tuple[Path, ...]:
    """Create six accuracy-first plots from shots.csv, without running decoders."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "plots" if output_dir is None else Path(output_dir)
    shots = _frame(run_dir / "shots.csv", ("shot",))
    if shots["shot"].duplicated().any():
        raise ValueError("Expected unique shot IDs.")
    return tuple(plot(shots, output_dir=output_dir) for plot in (
        plot_posterior_logical_error_rates, plot_posterior_fixed_vs_broken,
        plot_posterior_changed_confidence, plot_posterior_probability_vs_ess,
        plot_posterior_logical_transitions, plot_posterior_runtime,
    ))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed benchmark directory")
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--mwpm-reference", action="store_true", help="Overlay the saved MWPM weight on convergence")
    parser.add_argument("--output", type=Path, help="Plot directory (default: <run_dir>/plots)")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else args.run_dir / "plots"
    try:
        shots = _frame(args.run_dir / "shots.csv", ("shot",))
        if "posterior_p1" in shots.columns:
            paths = plot_posterior_benchmark(args.run_dir, output_dir=output)
            for path in paths:
                print(f"Saved {path}")
            return
        shots = _frame(shots, ("mwpm_weight",))
        if "flow_mcmc_best_weight" in shots.columns:
            paths = plot_three_method_benchmark(args.run_dir, shot=args.shot, output_dir=output)
            for path in paths:
                print(f"Saved {path}")
            return
        selected = shots.loc[shots["shot"] == args.shot]
        if len(selected) != 1:
            raise ValueError(f"Expected one result for shot {args.shot}; found {len(selected)}.")
        reference = float(selected.iloc[0]["mwpm_weight"]) if args.mwpm_reference else None
        convergence = plot_convergence(
            args.run_dir / "traces" / f"shot_{args.shot:06d}.csv",
            args.shot, output_dir=output, mwpm_weight=reference,
        )
        comparison = plot_weight_comparison(shots, output_dir=output)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Saved {convergence}")
    print(f"Saved {comparison}")


if __name__ == "__main__":
    main()
