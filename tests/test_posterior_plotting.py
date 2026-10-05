"""Accuracy-first posterior plots and unchanged legacy plotting behavior."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from surface_code import plotting


@pytest.fixture
def shots():
    return pd.DataFrame({
        "shot": [0, 1, 2, 3], "actual_logical": [0, 1, 0, 1],
        "mwpm_prediction": [0, 0, 0, 0], "posterior_prediction": [0, 1, 1, 0],
        "mwpm_mismatch": [0, 1, 0, 1], "posterior_mismatch": [0, 0, 1, 1],
        "corrector_mismatch": [0, 0, 0, 1], "posterior_p1": [0.1, 0.9, 0.6, 0.2],
        "posterior_logical_ess": [10, np.nan, 5, 2], "posterior_logical_transitions": [0, 2, 1, 3],
        "mwpm_seconds": [0.01, 0.02, 0.01, 0.02], "posterior_seconds": [1., 2., 1., 2.],
        "corrector_seconds": [1.02, 2.03, 1.02, 2.03],
    })


@pytest.fixture
def figures(monkeypatch):
    captured = {}
    def save(figure, output_dir, filename):
        captured[filename] = figure
        return Path(output_dir) / filename
    monkeypatch.setattr(plotting, "_save", save)
    return captured


def test_logical_rates_include_wilson_intervals(shots, figures):
    path = plotting.plot_posterior_logical_error_rates(shots)
    ax = figures[path.name].axes[0]
    np.testing.assert_allclose([p.get_height() for p in ax.patches], [0.5, 0.5, 0.25])
    segments = ax.containers[0].lines[2][0].get_segments()
    for segment, count in zip(segments, [2, 2, 1]):
        ci = binomtest(count, 4).proportion_ci(0.95, method="wilson")
        np.testing.assert_allclose(segment[:, 1], [ci.low, ci.high])
    assert "95% Wilson" in ax.get_title()


@pytest.mark.parametrize("value", [0, 1])
def test_wilson_boundary_rates_remain_finite(shots, figures, value):
    for method in plotting.POSTERIOR_METHODS:
        shots[f"{method}_mismatch"] = value
    path = plotting.plot_posterior_logical_error_rates(shots)
    segments = figures[path.name].axes[0].containers[0].lines[2][0].get_segments()
    assert np.isfinite(segments).all()
    assert all(segment[1, 1] > segment[0, 1] for segment in segments)


def test_fixed_and_broken_counts_show_raw_and_gated_results(shots, figures):
    path = plotting.plot_posterior_fixed_vs_broken(shots)
    ax = figures[path.name].axes[0]
    assert [p.get_height() for p in ax.patches] == [1, 1, 1, 0]


def test_confidence_histogram_only_changed_decisions(shots, figures):
    path = plotting.plot_posterior_changed_confidence(shots)
    ax = figures[path.name].axes[0]
    for container, confidence in zip(ax.containers, [abs(0.9 - 0.5), abs(0.6 - 0.5)]):
        expected, _ = np.histogram([confidence], bins=np.linspace(0, 0.5, 21))
        np.testing.assert_array_equal([p.get_height() for p in container], expected)
    assert "before correction gates" in ax.get_title()


def test_scatter_preserves_correctness_and_omits_undefined_ess(shots, figures):
    path = plotting.plot_posterior_probability_vs_ess(shots)
    ax = figures[path.name].axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[0.1, 10]])
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[0.6, 5], [0.2, 2]])
    assert "Undefined ESS omitted: 1/4 shots" in [t.get_text() for t in ax.texts]


def test_transition_counts_and_runtime_means(shots, figures):
    path = plotting.plot_posterior_logical_transitions(shots)
    assert [p.get_height() for p in figures[path.name].axes[0].patches] == [1, 1, 1, 1]
    path = plotting.plot_posterior_runtime(shots)
    np.testing.assert_allclose([p.get_height() for p in figures[path.name].axes[0].patches], [0.015, 1.5, 1.525])


def test_no_changes_and_all_undefined_ess(shots, figures):
    shots["posterior_prediction"] = shots["mwpm_prediction"]
    shots["posterior_logical_ess"] = np.nan
    path = plotting.plot_posterior_changed_confidence(shots)
    assert "No changed decisions" in [t.get_text() for t in figures[path.name].axes[0].texts]
    path = plotting.plot_posterior_probability_vs_ess(shots)
    assert all(len(c.get_offsets()) == 0 for c in figures[path.name].axes[0].collections)
    assert "Undefined ESS omitted: 4/4 shots" in [t.get_text() for t in figures[path.name].axes[0].texts]


def test_long_transition_trace_uses_bounded_bins(shots, figures):
    shots["posterior_logical_transitions"] = [0, 0, 100000, 1]
    path = plotting.plot_posterior_logical_transitions(shots)
    bars = figures[path.name].axes[0].patches
    assert len(bars) <= 50
    assert sum(p.get_height() for p in bars) == 4


@pytest.mark.parametrize("column,value,plot", [
    ("posterior_p1", 1.1, plotting.plot_posterior_probability_vs_ess),
    ("posterior_logical_ess", -1, plotting.plot_posterior_probability_vs_ess),
    ("posterior_logical_transitions", 0.5, plotting.plot_posterior_logical_transitions),
    ("posterior_seconds", np.nan, plotting.plot_posterior_runtime),
    ("posterior_mismatch", 2, plotting.plot_posterior_logical_error_rates),
])
def test_invalid_plot_values_rejected(shots, column, value, plot):
    shots[column] = value
    with pytest.raises(ValueError):
        plot(shots)


def test_cli_writes_six_pngs_without_weight_or_trace_inputs(shots, tmp_path, capsys):
    shots.to_csv(tmp_path / "shots.csv", index=False)
    (tmp_path / "summary.json").write_text('{"unchanged": true}')
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    plotting.main([str(tmp_path)])
    files = list((tmp_path / "plots").glob("*.png"))
    assert {p.name for p in files} == {
        "posterior_logical_error_rates.png", "posterior_fixed_vs_broken.png",
        "posterior_changed_confidence.png", "posterior_probability_vs_ess.png",
        "posterior_logical_transitions.png", "posterior_runtime_comparison.png",
    }
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in files)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert capsys.readouterr().out.count("Saved") == 6
