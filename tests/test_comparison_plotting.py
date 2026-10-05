"""Three-method plot values, proposal-source filtering, and saved artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surface_code import plotting


@pytest.fixture
def figures(monkeypatch):
    captured = {}

    def save(figure, output_dir, filename):
        captured[filename] = figure
        return Path(output_dir) / filename

    monkeypatch.setattr(plotting, "_save", save)
    return captured


def test_convergence_uses_best_weights_and_initial_states(figures):
    random = pd.DataFrame({"iteration": [2, 1], "current_weight": [10, 9], "best_weight_so_far": [6, 8]})
    learned = pd.DataFrame({"iteration": [1, 2], "best_weight_so_far": [3, 2]})
    original = random.copy(deep=True)
    path = plotting.plot_convergence_comparison(
        random, learned, 4, mwpm_weight=1.5, random_initial_weight=12, flow_initial_weight=13,
    )
    ax = figures[path.name].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [12, 8, 6])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [13, 3, 2])
    np.testing.assert_array_equal(ax.lines[2].get_ydata(), [1.5, 1.5])
    assert ax.get_xlabel() == "MCMC iteration"
    assert ax.get_ylabel() == "Best weight so far"
    pd.testing.assert_frame_equal(random, original)


def test_histogram_includes_rejections_and_excludes_mixture_uniform_draws(figures):
    random = pd.DataFrame({"proposed_weight": [1, 2, 3], "accepted": [True, False, False]})
    learned = pd.DataFrame({
        "proposed_weight": [1, 100, 4], "proposal_source": ["flow", "uniform", "flow"],
        "accepted": [False, True, True],
    })
    path = plotting.plot_proposal_weight_histogram(random, learned)
    ax = figures[path.name].axes[0]
    bins = np.histogram_bin_edges([1, 2, 3, 1, 4], bins=50)
    for container, values in zip(ax.containers, ([1, 2, 3], [1, 4])):
        expected, _ = np.histogram(values, bins=bins, density=True)
        actual = np.array([patch.get_height() for patch in container])
        np.testing.assert_allclose(actual, expected)
        assert np.sum(actual * np.diff(bins)) == pytest.approx(1)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["Uniform proposals (n=3)", "Flow proposals (n=2)"]


def test_acceptance_uses_overall_chain_rates(figures):
    path = plotting.plot_acceptance_comparison(pd.DataFrame({
        "random_mcmc_acceptance_rate": [0.2, 0.4], "flow_mcmc_acceptance_rate": [0.8, 0.4],
        "flow_mcmc_flow_acceptance_rate": [1, 1],
    }))
    ax = figures[path.name].axes[0]
    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], [0.3, 0.6])
    assert ax.get_ylabel() == "Mean per-shot acceptance rate"


def test_per_shot_best_weights_sorted_and_all_three_methods(figures):
    frame = pd.DataFrame({
        "shot": [2, 0], "mwpm_weight": [3, 1], "random_mcmc_best_weight": [9, 7],
        "flow_mcmc_best_weight": [4, 2],
    })
    original = frame.copy(deep=True)
    path = plotting.plot_three_method_weights(frame)
    ax = figures[path.name].axes[0]
    for line, values in zip(ax.lines, ([1, 3], [7, 9], [2, 4])):
        np.testing.assert_array_equal(line.get_xdata(), [0, 2])
        np.testing.assert_array_equal(line.get_ydata(), values)
    assert [line.get_label() for line in ax.lines] == ["MWPM", "Random MCMC", "Flow-MCMC"]
    pd.testing.assert_frame_equal(frame, original)


def test_logical_rates_and_sample_counts(figures):
    path = plotting.plot_logical_error_summary(pd.DataFrame({
        "mwpm_mismatch": [0, 1, 1], "random_mcmc_mismatch": [1, 0, 0], "flow_mcmc_mismatch": [0, 0, 0],
    }))
    ax = figures[path.name].axes[0]
    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], [2 / 3, 1 / 3, 0])
    assert [t.get_text() for t in ax.texts] == ["2/3 (66.7%)", "1/3 (33.3%)", "0/3 (0.0%)"]


def test_empty_proposals_and_uniform_only_mixture(figures):
    path = plotting.plot_proposal_weight_histogram([], [])
    assert "No proposals recorded" in [t.get_text() for t in figures[path.name].axes[0].texts]
    path = plotting.plot_proposal_weight_histogram(
        [{"proposed_weight": 5}], [{"proposed_weight": 5, "proposal_source": "uniform"}],
    )
    assert "No flow-source proposals recorded" in [t.get_text() for t in figures[path.name].axes[0].texts]
    path = plotting.plot_convergence_comparison([], [], 0, mwpm_weight=2,
                                                random_initial_weight=5, flow_initial_weight=4)
    assert len(figures[path.name].axes[0].lines) == 3


@pytest.mark.parametrize("invalid", [np.nan, -0.1, 1.1])
def test_invalid_acceptance_rates_rejected(invalid):
    with pytest.raises(ValueError, match="Acceptance rates"):
        plotting.plot_acceptance_comparison([{"random_mcmc_acceptance_rate": invalid, "flow_mcmc_acceptance_rate": 0.5}])


def test_invalid_histogram_and_logical_values_rejected():
    with pytest.raises(ValueError, match="proposal_source"):
        plotting.plot_proposal_weight_histogram([], [{"proposed_weight": 0, "proposal_source": "unknown"}])
    with pytest.raises(ValueError, match="finite"):
        plotting.plot_proposal_weight_histogram([{"proposed_weight": np.inf}], [])
    with pytest.raises(ValueError, match="binary"):
        plotting.plot_logical_error_summary([{"mwpm_mismatch": 2, "random_mcmc_mismatch": 0, "flow_mcmc_mismatch": 0}])


@pytest.mark.parametrize("iterations", [0, 2])
def test_cli_writes_all_five_plots_and_preserves_inputs(tmp_path, capsys, iterations):
    shots = pd.DataFrame({
        "shot": [0, 1], "mwpm_weight": [1, 2], "random_mcmc_best_weight": [8, 9],
        "flow_mcmc_best_weight": [2, 3], "random_mcmc_initial_weight": [10, 11],
        "flow_mcmc_initial_weight": [11, 12], "random_mcmc_acceptance_rate": [0.2, 0.4],
        "flow_mcmc_acceptance_rate": [0.8, 0.4], "mwpm_mismatch": [0, 0],
        "random_mcmc_mismatch": [1, 0], "flow_mcmc_mismatch": [0, 1],
    })
    shots.to_csv(tmp_path / "shots.csv", index=False)
    for folder in ("traces", "flow_traces"):
        (tmp_path / folder).mkdir()
        for shot in (0, 1):
            trace = pd.DataFrame({"iteration": [1, 2], "proposed_weight": [8, 9], "best_weight_so_far": [8, 8]})
            if folder == "flow_traces":
                trace["proposal_source"] = ["flow", "uniform"]
            trace.iloc[:iterations].to_csv(tmp_path / folder / f"shot_{shot:06d}.csv", index=False)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.csv")}
    plotting.main([str(tmp_path), "--shot", "1"])
    files = {path.name for path in (tmp_path / "plots").iterdir()}
    assert files == {
        "convergence_comparison_shot_1.png", "proposal_weight_histogram.png", "acceptance_rate_comparison.png",
        "per_shot_best_weight.png", "logical_error_rate_summary.png",
    }
    for path in (tmp_path / "plots").iterdir():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all(path.read_bytes() == value for path, value in before.items())
    assert capsys.readouterr().out.count("Saved") == 5
