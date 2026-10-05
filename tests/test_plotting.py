"""Saved plot data and optional, display-only MWPM references."""

import numpy as np
import pandas as pd
import pytest

from surface_code import plotting
from surface_code.random_mcmc import TraceEntry


@pytest.mark.parametrize("reference", [None, 1.5])
def test_convergence_plots_trace_and_optional_reference(tmp_path, monkeypatch, reference):
    trace = (
        TraceEntry(1, 8, 8, -2, True, 8),
        TraceEntry(2, 9, 9, 1, True, 8),
        TraceEntry(3, 6, 6, -3, True, 6),
    )
    original_save = plotting._save
    figures = []

    def capture(figure, output_dir, filename):
        figures.append(figure)
        return original_save(figure, output_dir, filename)

    monkeypatch.setattr(plotting, "_save", capture)
    output = plotting.plot_convergence(trace, 7, output_dir=tmp_path, mwpm_weight=reference)
    assert output.name == "convergence_shot_7.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    lines = figures[0].axes[0].lines
    assert len(lines) == (2 if reference is None else 3)
    np.testing.assert_array_equal(lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(lines[0].get_ydata(), [8, 9, 6])
    np.testing.assert_array_equal(lines[1].get_ydata(), [8, 8, 6])
    if reference is not None:
        np.testing.assert_array_equal(lines[2].get_ydata(), [reference, reference])


def test_comparison_orders_shots_without_mutating_input(tmp_path, monkeypatch):
    shots = pd.DataFrame({"shot": [2, 0], "mwpm_weight": [3, 1], "mcmc_best_weight": [9, 7]})
    original = shots.copy(deep=True)
    figures = []
    original_save = plotting._save

    def capture(figure, output_dir, filename):
        figures.append(figure)
        return original_save(figure, output_dir, filename)

    monkeypatch.setattr(plotting, "_save", capture)
    output = plotting.plot_weight_comparison(shots, output_dir=tmp_path)
    assert output.name == "weight_comparison.png"
    assert output.is_file()
    lines = figures[0].axes[0].lines
    np.testing.assert_array_equal(lines[0].get_xdata(), [0, 2])
    np.testing.assert_array_equal(lines[0].get_ydata(), [1, 3])
    np.testing.assert_array_equal(lines[1].get_ydata(), [7, 9])
    pd.testing.assert_frame_equal(shots, original)


def test_empty_trace_and_missing_columns(tmp_path):
    assert plotting.plot_convergence((), 0, output_dir=tmp_path).is_file()
    with pytest.raises(ValueError, match="Missing plot columns"):
        plotting.plot_convergence([{"iteration": 1}], 0, output_dir=tmp_path)


def test_cli_reads_completed_csv_files(tmp_path, capsys):
    (tmp_path / "traces").mkdir()
    (tmp_path / "shots.csv").write_text("shot,mwpm_weight,mcmc_best_weight\n0,2,8\n")
    (tmp_path / "traces" / "shot_000000.csv").write_text(
        "iteration,current_weight,best_weight_so_far\n1,8,8\n"
    )
    plotting.main([str(tmp_path), "--mwpm-reference"])
    assert (tmp_path / "plots" / "convergence_shot_0.png").is_file()
    assert (tmp_path / "plots" / "weight_comparison.png").is_file()
    assert "Saved" in capsys.readouterr().out
