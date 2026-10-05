"""Benchmark outputs, seed replay, and separation of the two decoders."""

import csv
import json

import numpy as np
import pytest

from surface_code import benchmark
from surface_code.circuit import build_surface_code, sample_shots
from surface_code.mwpm_decoder import MWPMResult


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_benchmark_outputs_and_summary(tmp_path):
    output = benchmark.run_benchmark(
        shots=3, iterations=4, stim_seed=42, mcmc_seed=17,
        output=tmp_path / "run",
    )
    rows = read_rows(output / "shots.csv")
    assert len(rows) == 3
    assert tuple(rows[0]) == benchmark.SHOT_FIELDS
    syndromes, observables = sample_shots(build_surface_code(3, 3, 0.005), 3, seed=42)
    for shot, row in enumerate(rows):
        assert int(row["shot"]) == shot
        assert int(row["syndrome_weight"]) == syndromes[shot].sum()
        assert int(row["actual_logical"]) == observables[shot, 0]
        for decoder in ("mwpm", "mcmc"):
            assert int(row[f"{decoder}_mismatch"]) == (
                int(row[f"{decoder}_prediction"]) != int(row["actual_logical"])
            )
            assert float(row[f"{decoder}_seconds"]) >= 0
        assert float(row["mcmc_best_minus_mwpm_weight"]) == pytest.approx(
            float(row["mcmc_best_weight"]) - float(row["mwpm_weight"])
        )
        trace = read_rows(output / "traces" / f"shot_{shot:06d}.csv")
        assert len(trace) == 4
        assert float(trace[-1]["best_weight_so_far"]) == float(row["mcmc_best_weight"])
        assert float(trace[-1]["current_weight"]) == float(row["mcmc_final_weight"])
        accepted = sum(entry["accepted"] == "True" for entry in trace)
        assert accepted == int(row["mcmc_accepted_moves"])
        assert float(row["mcmc_acceptance_rate"]) == accepted / 4
    summary = json.loads((output / "summary.json").read_text())
    assert summary["shots"] == 3
    for decoder in ("mwpm", "mcmc"):
        mismatches = sum(int(row[f"{decoder}_mismatch"]) for row in rows)
        assert summary[decoder]["logical_mismatches"] == mismatches
        assert summary[decoder]["logical_error_rate"] == mismatches / 3
    settings = json.loads((output / "settings.json").read_text())
    assert settings["stim_seed"] == 42
    assert settings["mcmc_seed"] == 17
    assert settings["iterations"] == 4
    assert len(settings["algorithms"]) == 2


def test_generated_seeds_can_replay_results(tmp_path):
    first = benchmark.run_benchmark(shots=2, iterations=3, output=tmp_path / "first")
    settings = json.loads((first / "settings.json").read_text())
    second = benchmark.run_benchmark(
        shots=2, iterations=3, output=tmp_path / "second",
        stim_seed=settings["stim_seed"], mcmc_seed=settings["mcmc_seed"],
    )
    a, b = read_rows(first / "shots.csv"), read_rows(second / "shots.csv")
    for rows in (a, b):
        for row in rows:
            row.pop("mwpm_seconds")
            row.pop("mcmc_seconds")
    assert a == b
    for shot in range(2):
        name = f"shot_{shot:06d}.csv"
        assert (first / "traces" / name).read_text() == (second / "traces" / name).read_text()


def test_mcmc_receives_no_mwpm_information(tmp_path, monkeypatch):
    actual_mcmc = benchmark.decode_random_mcmc
    calls = []

    def mwpm(syndrome, graph):
        calls.append(syndrome.copy())
        # Deliberately unrelated metadata must not reach the MCMC decoder.
        return MWPMResult(1, (999999,), -123456.0)

    def mcmc(syndrome, graph, iterations, *, rng):
        np.testing.assert_array_equal(syndrome, calls[-1])
        assert isinstance(rng, np.random.Generator)
        return actual_mcmc(syndrome, graph, iterations, rng=rng)

    monkeypatch.setattr(benchmark, "decode_mwpm", mwpm)
    monkeypatch.setattr(benchmark, "decode_random_mcmc", mcmc)
    output = benchmark.run_benchmark(
        shots=2, iterations=2, stim_seed=42, mcmc_seed=7, output=tmp_path / "run",
    )
    syndromes, _ = sample_shots(build_surface_code(3, 3, 0.005), 2, seed=42)
    from surface_code.decoding_graph import build_decoding_graph
    from surface_code.circuit import get_detector_error_model

    graph = build_decoding_graph(get_detector_error_model(build_surface_code(3, 3, 0.005)))
    rng = np.random.default_rng(7)
    for row, syndrome in zip(read_rows(output / "shots.csv"), syndromes):
        independent = actual_mcmc(syndrome, graph, 2, rng=rng)
        assert float(row["mcmc_best_weight"]) == independent.best_weight
        assert int(row["mcmc_prediction"]) == independent.prediction
    assert len(calls) == 2


def test_cli_options_and_zero_iteration_trace(tmp_path, capsys):
    output = tmp_path / "cli"
    benchmark.main([
        "--distance", "3", "--rounds", "1", "--p", "0.01",
        "--shots", "1", "--iterations", "0", "--stim-seed", "5",
        "--mcmc-seed", "6", "--output", str(output),
    ])
    assert "Results saved" in capsys.readouterr().out
    assert read_rows(output / "traces" / "shot_000000.csv") == []
    assert (output / "traces" / "shot_000000.csv").read_text().startswith("iteration,")
    settings = json.loads((output / "settings.json").read_text())
    assert (settings["rounds"], settings["p"], settings["iterations"]) == (1, 0.01, 0)


@pytest.mark.parametrize("kwargs", [
    {"shots": 0}, {"iterations": -1}, {"shots": 1.5}, {"stim_seed": -1},
])
def test_invalid_settings_create_no_output(tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(output=output, **kwargs)
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    marker = tmp_path / "settings.json"
    marker.write_text("keep me")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(output=tmp_path, shots=1, iterations=0)
    assert marker.read_text() == "keep me"
