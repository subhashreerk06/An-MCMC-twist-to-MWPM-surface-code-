"""Fresh paired shots, isolated decoder inputs, reproducible saved diagnostics."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import torch

from surface_code import posterior_benchmark as benchmark
from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
from surface_code.flow_mcmc import _graph_fingerprints
from surface_code.gf2 import AffineCoordinates, build_incidence_matrix, sample_uniform_solution
from surface_code.mwpm_decoder import MWPMResult


@pytest.fixture
def checkpoint(tmp_path):
    graph = build_decoding_graph(get_detector_error_model(build_surface_code(3, 1, 0.005)))
    H = build_incidence_matrix(graph)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = ConditionalAutoregressiveBernoulli(H.shape[0], AffineCoordinates(H).nullity, 8)
    path = tmp_path / "model.pt"
    model.save_checkpoint(path, fingerprints=_graph_fingerprints(graph))
    return path


def read_rows(output):
    with (output / "shots.csv").open() as stream:
        return list(csv.DictReader(stream))


def options(checkpoint):
    return dict(rounds=1, shots=3, iterations=7, burn_in=2,
                stim_seed=42, mcmc_seed=7, flow_checkpoint=checkpoint)


def test_outputs_metrics_and_replay(checkpoint, tmp_path):
    first = benchmark.run_benchmark(**options(checkpoint), output=tmp_path / "first")
    second = benchmark.run_benchmark(**options(checkpoint), output=tmp_path / "second")
    records = read_rows(first)
    assert tuple(records[0]) == benchmark.SHOT_FIELDS
    syndromes, observables = sample_shots(build_surface_code(3, 1, 0.005), 3, seed=42)
    with np.load(first / "shots.npz") as data:
        np.testing.assert_array_equal(data["syndromes"], syndromes)
        np.testing.assert_array_equal(data["actual_logical"], observables[:, 0])
    for shot, (row, replay) in enumerate(zip(records, read_rows(second))):
        assert {k: v for k, v in row.items() if not k.endswith("seconds")} == {
            k: v for k, v in replay.items() if not k.endswith("seconds")}
        assert int(row["shot"]) == shot
        assert int(row["actual_logical"]) == observables[shot, 0]
        assert int(row["syndrome_weight"]) == syndromes[shot].sum()
        for decoder in ("mwpm", "posterior", "corrector"):
            assert int(row[f"{decoder}_mismatch"]) == (int(row[f"{decoder}_prediction"]) != observables[shot, 0])
            assert float(row[f"{decoder}_seconds"]) >= 0
        assert float(row["corrector_seconds"]) == pytest.approx(
            float(row["mwpm_seconds"]) + float(row["posterior_seconds"]) + float(row["correction_layer_seconds"])
        )
        trace_path = f"logical_traces/shot_{shot:06d}.npy"
        trace = np.load(first / trace_path)
        np.testing.assert_array_equal(trace, np.load(second / trace_path))
        assert len(trace) == int(row["posterior_retained_samples"]) == 5
        assert float(row["posterior_p1"]) == np.mean(trace)
        assert float(row["posterior_p0"]) == 1 - np.mean(trace)
        assert float(row["posterior_confidence"]) == abs(np.mean(trace) - 0.5)
        assert int(row["posterior_logical_transitions"]) == np.count_nonzero(trace[1:] != trace[:-1])
        diagnostic = json.loads((first / f"logical_traces/shot_{shot:06d}.json").read_text())
        assert diagnostic["status"] == row["posterior_diagnostic_status"]
        if diagnostic["logical_ess"] is None:
            assert row["posterior_logical_ess"] == ""
        else:
            assert float(row["posterior_logical_ess"]) == diagnostic["logical_ess"]
        for component in ("flow", "uniform"):
            proposed = int(row[f"posterior_{component}_proposals"])
            accepted = int(row[f"posterior_{component}_accepted"])
            assert float(row[f"posterior_{component}_acceptance_rate"]) == (accepted / proposed if proposed else 0)
    summary = json.loads((first / "summary.json").read_text())
    for decoder in ("mwpm", "posterior", "corrector"):
        failures = sum(int(r[f"{decoder}_mismatch"]) for r in records)
        assert summary[decoder]["logical_mismatches"] == failures
        assert summary[decoder]["logical_error_rate"] == failures / 3
    paired = summary["paired_comparison"]
    assert sum(paired[k] for k in ("both_correct", "both_wrong", "mwpm_wrong_posterior_correct", "mwpm_correct_posterior_wrong")) == 3
    assert paired["net_failures_reduced"] == summary["mwpm"]["logical_mismatches"] - summary["posterior"]["logical_mismatches"]
    settings = json.loads((first / "settings.json").read_text())
    assert settings["burn_in"] == 2 and settings["mcmc_seed"] == 7
    assert len(settings["flow_checkpoint_sha256"]) == 64
    assert settings["bootstrap_seed"] == 0 and settings["bootstrap_resamples"] == 10000
    assert "scipy" in settings["versions"]
    assert settings["correction_margin"] == settings["min_logical_transitions"] == settings["min_logical_ess"] == 0
    assert len(settings["algorithms"]) == 3
    replay_summary = json.loads((second / "summary.json").read_text())
    for key in ("mcnemar_exact_pvalue", "mwpm_ler_wilson_ci95", "posterior_ler_wilson_ci95",
                "paired_bootstrap_ler_change_ci95"):
        assert summary[key] == replay_summary[key]


def test_no_matching_information_enters_chain(checkpoint, tmp_path, monkeypatch):
    syndromes = sample_shots(build_surface_code(3, 1, 0.005), 3, seed=42)[0]
    calls = {"mwpm": 0, "posterior": 0}
    original = benchmark.PosteriorMCMCDecoder.decode
    def mwpm(syndrome, graph):
        np.testing.assert_array_equal(syndrome, syndromes[calls["mwpm"]])
        calls["mwpm"] += 1
        syndrome[:] ^= True
        return MWPMResult(1, (999999,), -123456.0)
    def decode(self, syndrome, iterations, *, burn_in, rng):
        np.testing.assert_array_equal(syndrome, syndromes[calls["posterior"]])
        initial = sample_uniform_solution(self._H, syndrome, deepcopy(rng))
        result = original(self, syndrome, iterations, burn_in=burn_in, rng=rng)
        np.testing.assert_array_equal(result.initial_configuration, initial)
        calls["posterior"] += 1
        return result
    monkeypatch.setattr(benchmark, "decode_mwpm", mwpm)
    monkeypatch.setattr(benchmark.PosteriorMCMCDecoder, "decode", decode)
    benchmark.run_benchmark(**options(checkpoint), output=tmp_path / "isolated")
    assert calls == {"mwpm": 3, "posterior": 3}


def test_cli_uniform_mode_and_constant_trace(checkpoint, tmp_path, capsys):
    output = tmp_path / "cli"
    benchmark.main([
        "--distance", "3", "--rounds", "1", "--p", "0.005", "--shots", "1",
        "--iterations", "2", "--burn-in", "1", "--flow-checkpoint", str(checkpoint),
        "--flow-probability", "0", "--stim-seed", "42", "--mcmc-seed", "7",
        "--bootstrap-seed", "91", "--bootstrap-resamples", "2000",
        "--correction-margin", "0.2", "--min-logical-transitions", "3", "--min-logical-ess", "5",
        "--output", str(output),
    ])
    assert "Results saved" in capsys.readouterr().out
    row = read_rows(output)[0]
    assert float(row["posterior_flow_acceptance_rate"]) == 0
    assert row["posterior_logical_ess"] == ""
    assert row["posterior_diagnostic_status"] == "constant_trace"
    settings = json.loads((output / "settings.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert settings["bootstrap_seed"] == 91 and settings["bootstrap_resamples"] == 2000
    assert (settings["correction_margin"], settings["min_logical_transitions"], settings["min_logical_ess"]) == (0.2, 3, 5)
    assert summary["paired_bootstrap_ler_change_ci95"]["seed"] == 91
    assert summary["paired_bootstrap_ler_change_ci95"]["resamples"] == 2000


@pytest.mark.parametrize("changes", [
    {"shots": 0}, {"iterations": 0}, {"burn_in": 7}, {"burn_in": -1},
    {"flow_probability": 1.1}, {"stim_seed": 2**64}, {"mcmc_seed": -1}, {"rounds": 3},
])
def test_invalid_settings_create_no_output(checkpoint, tmp_path, changes):
    config = options(checkpoint)
    config.update(changes)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(**config, output=output)
    assert not output.exists()


def test_existing_directory_never_overwritten(checkpoint, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "shots.csv"
    sentinel.write_text("preserve existing results")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(**options(checkpoint), output=output)
    assert sentinel.read_text() == "preserve existing results"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("counts", [
    (2, 3, 1, 2),  # Improvement.
    (2, 1, 3, 2),  # Regression: negative reductions must remain visible.
    (2, 2, 2, 2),  # Tied failure rates do not count as helping.
    (3, 0, 2, 0),  # Zero MWPM LER with posterior failures.
    (3, 0, 0, 0),  # Neither decoder fails.
    (0, 0, 0, 3),  # Both always fail.
])
def test_paired_analysis_all_outcomes_and_signed_changes(counts):
    rows = []
    for (mwpm_wrong, posterior_wrong), count in zip(((0, 0), (1, 0), (0, 1), (1, 1)), counts):
        for i in range(count):
            actual = i % 2
            rows.append({"actual_logical": actual, "mwpm_prediction": actual ^ mwpm_wrong,
                         "posterior_prediction": actual ^ posterior_wrong})
    result = benchmark._paired_analysis(rows)
    a, b, c, d = counts
    n = sum(counts)
    assert [result["paired_outcomes"][k]["count"] for k in "ABCD"] == list(counts)
    assert sum(item["fraction"] for item in result["paired_outcomes"].values()) == pytest.approx(1)
    for category in result["paired_outcomes"].values():
        assert result[category["name"]] == category["count"]
        assert result[f'{category["name"]}_fraction'] == category["count"] / n
    assert result["mwpm_logical_error_rate"] == (b + d) / n
    assert result["posterior_logical_error_rate"] == (c + d) / n
    assert result["absolute_ler_change"] == pytest.approx((c - b) / n)
    assert result["absolute_ler_reduction"] == pytest.approx((b - c) / n)
    if b + d:
        assert result["relative_ler_reduction"] == pytest.approx((b - c) / (b + d))
    else:
        assert result["relative_ler_reduction"] is None
    assert result["net_failures_fixed"] == b - c
    assert result["helping_on_tested_sample"] is (b > c)


@pytest.mark.parametrize("categories", ["ABBCCCD", "AADD"])
def test_saved_paired_analysis_and_changed_decisions(checkpoint, tmp_path, monkeypatch, categories):
    from dataclasses import replace

    config = options(checkpoint)
    config["shots"] = len(categories)
    _, observables = sample_shots(build_surface_code(3, 1, 0.005), len(categories), seed=42)
    actuals = observables[:, 0].astype(int)
    mwpm_predictions = iter(int(actual ^ (category in "BD")) for actual, category in zip(actuals, categories))
    posterior_predictions = iter(int(actual ^ (category in "CD")) for actual, category in zip(actuals, categories))
    original = benchmark.PosteriorMCMCDecoder.decode

    def matching(syndrome, graph):
        return MWPMResult(next(mwpm_predictions), (), 123.0)

    def decode(self, syndrome, iterations, *, burn_in, rng):
        result = original(self, syndrome, iterations, burn_in=burn_in, rng=rng)
        return replace(result, prediction=next(posterior_predictions))

    monkeypatch.setattr(benchmark, "decode_mwpm", matching)
    monkeypatch.setattr(benchmark.PosteriorMCMCDecoder, "decode", decode)
    output = benchmark.run_benchmark(**config, output=tmp_path / "paired")
    records = read_rows(output)
    assert "".join(row["paired_outcome"] for row in records) == categories
    with (output / "changed_decisions.csv").open() as stream:
        reader = csv.DictReader(stream)
        changed = list(reader)
        assert tuple(reader.fieldnames) == benchmark.CHANGED_DECISION_FIELDS
    expected = [{key: row[key] for key in benchmark.CHANGED_DECISION_FIELDS}
                for row in records if row["mwpm_prediction"] != row["posterior_prediction"]]
    assert changed == expected
    assert len(changed) == categories.count("B") + categories.count("C")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["mwpm_failures_fixed"] == categories.count("B")
    assert summary["mwpm_correct_broken"] == categories.count("C")
    assert summary["net_failures_fixed"] == categories.count("B") - categories.count("C")
    assert summary["helping_on_tested_sample"] is False
    if "C" in categories:
        assert summary["absolute_ler_reduction"] < 0
        assert summary["relative_ler_reduction"] < 0
    for name in ("mwpm", "posterior"):
        assert summary[f"{name}_logical_error_rate"] == summary[name]["logical_error_rate"]


def statistical_rows(a=0, b=0, c=0, d=0):
    return [{"actual_logical": 0, "mwpm_prediction": mwpm, "posterior_prediction": posterior}
            for mwpm, posterior, count in ((0, 0, a), (1, 0, b), (0, 1, c), (1, 1, d))
            for _ in range(count)]


@pytest.mark.parametrize("b,c,expected", [(0, 0, 1.0), (1, 0, 1.0), (3, 3, 1.0),
                                           (10, 0, 0.001953125), (0, 10, 0.001953125)])
def test_exact_mcnemar_known_probabilities(b, c, expected):
    result = benchmark._paired_analysis(statistical_rows(a=5, b=b, c=c))
    assert result["discordant_fixed"] == b and result["discordant_broken"] == c
    assert result["mcnemar_exact_pvalue"] == pytest.approx(expected)
    # Concordant shots do not enter the conditional binomial test.
    more = benchmark._paired_analysis(statistical_rows(a=50, b=b, c=c, d=20))
    assert more["mcnemar_exact_pvalue"] == result["mcnemar_exact_pvalue"]


@pytest.mark.parametrize("failures,expected", [
    (0, (0, 0.2775327998628892)),
    (5, (0.236593090512564, 0.763406909487436)),
    (10, (0.7224672001371108, 1)),
])
def test_wilson_known_intervals(failures, expected):
    result = benchmark._paired_analysis(statistical_rows(a=10-failures, d=failures))
    for method in ("mwpm", "posterior"):
        interval = result[f"{method}_ler_wilson_ci95"]
        assert (interval["low"], interval["high"]) == pytest.approx(expected)
    # Identical paired outcomes give zero difference in every resample,
    # even though each marginal rate can have considerable uncertainty.
    interval = result["paired_bootstrap_ler_change_ci95"]
    assert interval["low"] == interval["high"] == 0
    assert interval["status"] == "constant_observed_difference"


def test_paired_bootstrap_replay_and_shot_resampling_reference():
    rows = statistical_rows(a=12, b=18, c=5, d=15)
    result = benchmark._paired_analysis(rows, bootstrap_seed=123, bootstrap_resamples=20000)
    assert result == benchmark._paired_analysis(rows, bootstrap_seed=123, bootstrap_resamples=20000)
    interval = result["paired_bootstrap_ler_change_ci95"]
    assert interval["seed"] == 123 and interval["resamples"] == 20000
    assert interval["estimand"] == "posterior_LER - MWPM_LER"
    # Independently implement literal resampling of whole shot pairs.
    differences = np.array([row["posterior_prediction"] - row["mwpm_prediction"] for row in rows])
    rng = np.random.default_rng(91)
    bootstrap = differences[rng.integers(len(rows), size=(20000, len(rows)))].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    assert interval["low"] == pytest.approx(low, abs=0.025)
    assert interval["high"] == pytest.approx(high, abs=0.025)
    assert interval["high"] < 0  # This synthetic sample favors posterior.


@pytest.mark.parametrize("b,c,expected", [(10, 0, -1), (0, 10, 1)])
def test_bootstrap_keeps_difference_sign(b, c, expected):
    result = benchmark._paired_analysis(statistical_rows(b=b, c=c))
    interval = result["paired_bootstrap_ler_change_ci95"]
    assert interval["low"] == interval["high"] == expected
    assert interval["status"] == "constant_observed_difference"


@pytest.mark.parametrize("kwargs", [{"bootstrap_seed": -1}, {"bootstrap_resamples": 0},
                                    {"bootstrap_seed": True}, {"bootstrap_resamples": 1.5}])
def test_invalid_bootstrap_settings(checkpoint, tmp_path, kwargs):
    output = tmp_path / "invalid_bootstrap"
    with pytest.raises(ValueError, match="bootstrap"):
        benchmark.run_benchmark(**options(checkpoint), **kwargs, output=output)
    assert not output.exists()


@pytest.mark.parametrize("categories,expected_outcome", [
    ("ABBBCCD", "reduced_failures"),
    ("ABCCCDD", "increased_failures"),
    ("AABBCCD", "no_net_difference"),
])
def test_three_predictions_and_correction_outcome(checkpoint, tmp_path, monkeypatch, categories, expected_outcome):
    from dataclasses import replace
    from surface_code.posterior_mcmc import logical_trace_diagnostics

    config = {**options(checkpoint), "shots": len(categories)}
    _, observables = sample_shots(build_surface_code(3, 1, 0.005), len(categories), seed=42)
    actuals = observables[:, 0].astype(int)
    mwpm_predictions = iter(int(a ^ (c in "BD")) for a, c in zip(actuals, categories))
    posterior_predictions = iter(int(a ^ (c in "CD")) for a, c in zip(actuals, categories))
    original = benchmark.PosteriorMCMCDecoder.decode
    calls = []

    def decode(self, syndrome, iterations, *, burn_in, rng):
        calls.append(1)
        result = original(self, syndrome, iterations, burn_in=burn_in, rng=rng)
        pred = next(posterior_predictions)
        trace = (0, 1, 1, 1, 1) if pred else (1, 0, 0, 0, 0)
        diagnostics = logical_trace_diagnostics(trace)
        return replace(result, prediction=pred, posterior_p1=diagnostics.posterior_p1,
                       posterior_p0=diagnostics.posterior_p0, logical_parity_trace=trace,
                       logical_diagnostics=diagnostics, minimum_weight_seen=-123456.)

    monkeypatch.setattr(benchmark, "decode_mwpm", lambda *args: MWPMResult(next(mwpm_predictions), (), 456.))
    monkeypatch.setattr(benchmark.PosteriorMCMCDecoder, "decode", decode)
    output = benchmark.run_benchmark(**config, output=tmp_path / "three_methods")
    assert len(calls) == len(categories)  # One chain shared by posterior and corrector.
    records = read_rows(output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["correction_outcome"] == expected_outcome
    assert expected_outcome in summary["correction_summary"]
    fixed, broken = categories.count("B"), categories.count("C")
    corrected = summary["corrector"]
    assert corrected["correction_attempts"] == corrected["corrections_applied"] == fixed + broken
    assert corrected["corrections_that_fixed_MWPM"] == fixed
    assert corrected["corrections_that_broke_MWPM"] == broken
    assert corrected["net_corrections"] == fixed - broken
    assert corrected["logical_mismatches"] == summary["mwpm"]["logical_mismatches"] - (fixed - broken)
    assert summary["corrector_vs_mwpm"]["discordant_fixed"] == fixed
    assert summary["corrector_vs_mwpm"]["discordant_broken"] == broken
    assert summary["corrector_vs_mwpm"]["paired_bootstrap_ler_change_ci95"]["estimand"] == "corrector_LER - MWPM_LER"
    for row, category in zip(records, categories):
        assert row["corrector_prediction"] == row["posterior_prediction"]
        assert int(row["correction_attempted"]) == int(row["correction_applied"]) == (category in "BC")
        for method in ("mwpm", "posterior", "corrector"):
            assert int(row[f"{method}_mismatch"]) == (row[f"{method}_prediction"] != row["actual_logical"])
    for method in ("mwpm", "posterior", "corrector"):
        assert summary[method]["total_seconds"] == pytest.approx(sum(float(r[f"{method}_seconds"]) for r in records))


@pytest.mark.parametrize("gates,constant", [
    ({"correction_margin": 0.4}, False),
    ({"min_logical_transitions": 2}, False),
    ({"min_logical_ess": 100}, False),
    ({}, True),  # Default ESS=0 rejects undefined ESS.
])
def test_benchmark_quality_gates_block_proposed_corrections(checkpoint, tmp_path, monkeypatch, gates, constant):
    from dataclasses import replace
    from surface_code.posterior_mcmc import logical_trace_diagnostics

    original = benchmark.PosteriorMCMCDecoder.decode
    def decode(self, syndrome, iterations, *, burn_in, rng):
        result = original(self, syndrome, iterations, burn_in=burn_in, rng=rng)
        trace = (1,) * 5 if constant else (0, 1, 1, 1, 1)
        diagnostics = logical_trace_diagnostics(trace)
        return replace(result, prediction=1, posterior_p1=diagnostics.posterior_p1,
                       posterior_p0=diagnostics.posterior_p0, logical_diagnostics=diagnostics,
                       logical_parity_trace=trace)
    monkeypatch.setattr(benchmark, "decode_mwpm", lambda *args: MWPMResult(0, (), 0.))
    monkeypatch.setattr(benchmark.PosteriorMCMCDecoder, "decode", decode)
    output = benchmark.run_benchmark(**options(checkpoint), **gates, output=tmp_path / "gated")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["corrector"]["correction_attempts"] == 3
    assert summary["corrector"]["corrections_applied"] == 0
    assert summary["corrector"]["net_corrections"] == 0
    assert summary["correction_outcome"] == "no_net_difference"
    assert all(row["corrector_prediction"] == row["mwpm_prediction"] for row in read_rows(output))


@pytest.mark.parametrize("gates", [{"correction_margin": 0.6}, {"min_logical_transitions": -1},
                                   {"min_logical_ess": float("nan")}])
def test_invalid_correction_gates_create_no_output(checkpoint, tmp_path, gates):
    output = tmp_path / "bad_gates"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(**options(checkpoint), **gates, output=output)
    assert not output.exists()
