"""Three independent decoders, preserved baseline outputs, and flow traces."""

from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import torch

from surface_code import benchmark, flow_mcmc
from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph
from surface_code.discrete_flow import ConditionalAutoregressiveBernoulli
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
    model.save_checkpoint(path, fingerprints=flow_mcmc._graph_fingerprints(graph))
    return path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_three_method_metrics_traces_and_summary(checkpoint, tmp_path):
    output = benchmark.run_benchmark(
        rounds=1, shots=3, iterations=6, stim_seed=42, mcmc_seed=7,
        flow_checkpoint=checkpoint, flow_probability=0.6, output=tmp_path / "run",
    )
    records = rows(output / "shots.csv")
    assert tuple(records[0]) == benchmark.THREE_METHOD_SHOT_FIELDS
    settings = json.loads((output / "settings.json").read_text())
    assert len(settings["algorithms"]) == 3
    assert settings["flow_probability"] == 0.6
    assert settings["flow_checkpoint"] == str(checkpoint.resolve())
    assert len(settings["flow_checkpoint_sha256"]) == 64
    actual = sample_shots(build_surface_code(3, 1, 0.005), 3, seed=42)[1][:, 0]
    for shot, row in enumerate(records):
        assert int(row["actual_logical"]) == int(actual[shot])
        for decoder in ("mwpm", "random_mcmc", "flow_mcmc"):
            assert int(row[f"{decoder}_mismatch"]) == (int(row[f"{decoder}_prediction"]) != actual[shot])
            assert float(row[f"{decoder}_seconds"]) >= 0
        for suffix in ("prediction", "initial_weight", "best_weight", "acceptance_rate", "mismatch", "seconds"):
            assert row[f"random_mcmc_{suffix}"] == row[f"mcmc_{suffix}"]
        a, b, c = (float(row[field]) for field in ("mwpm_weight", "random_mcmc_best_weight", "flow_mcmc_best_weight"))
        assert float(row["random_best_minus_mwpm"]) == pytest.approx(b - a)
        assert float(row["flow_best_minus_mwpm"]) == pytest.approx(c - a)
        assert float(row["flow_best_minus_random_best"]) == pytest.approx(c - b)
        trace = rows(output / "flow_traces" / f"shot_{shot:06d}.csv")
        assert len(trace) == 6
        assert float(trace[-1]["best_weight_so_far"]) == c
        assert float(trace[-1]["current_weight"]) == float(row["flow_mcmc_final_weight"])
        assert sum(t["accepted"] == "True" for t in trace) / 6 == float(row["flow_mcmc_acceptance_rate"])
        for component in ("flow", "uniform"):
            selected = [t for t in trace if t["proposal_source"] == component]
            accepted = sum(t["accepted"] == "True" for t in selected)
            assert int(row[f"flow_mcmc_{component}_proposals"]) == len(selected)
            assert int(row[f"flow_mcmc_{component}_accepted_moves"]) == accepted
            assert float(row[f"flow_mcmc_{component}_acceptance_rate"]) == (accepted / len(selected) if selected else 0)
        assert len(rows(output / "traces" / f"shot_{shot:06d}.csv")) == 6
    summary = json.loads((output / "summary.json").read_text())
    assert summary["random_mcmc"] == summary["mcmc"]
    for decoder in ("mwpm", "random_mcmc", "flow_mcmc"):
        assert summary[decoder]["logical_mismatches"] == sum(int(r[f"{decoder}_mismatch"]) for r in records)
    for component in ("flow", "uniform"):
        count = sum(int(r[f"flow_mcmc_{component}_proposals"]) for r in records)
        accepted = sum(int(r[f"flow_mcmc_{component}_accepted_moves"]) for r in records)
        assert summary["flow_mcmc"][f"{component}_acceptance_rate"] == (accepted / count if count else 0)
    for comparison in ("random_best_minus_mwpm", "flow_best_minus_mwpm", "flow_best_minus_random_best"):
        assert summary[f"mean_{comparison}"] == np.mean([float(r[comparison]) for r in records])


def test_same_syndromes_random_starts_and_no_matching_information(checkpoint, tmp_path, monkeypatch):
    syndromes = sample_shots(build_surface_code(3, 1, 0.005), 2, seed=42)[0]
    calls = {"mwpm": 0, "random": 0, "flow": 0}
    streams = {}
    actual_random = benchmark.decode_random_mcmc
    actual_flow = flow_mcmc.FlowProposalMCMCDecoder.decode

    def matching(syndrome, graph):
        np.testing.assert_array_equal(syndrome, syndromes[calls["mwpm"]])
        calls["mwpm"] += 1
        syndrome[:] ^= True  # Each branch must receive its own original syndrome.
        return MWPMResult(1, (999999,), -123456.0)

    def random_decoder(syndrome, graph, iterations, *, rng):
        np.testing.assert_array_equal(syndrome, syndromes[calls["random"]])
        expected_initial = sample_uniform_solution(build_incidence_matrix(graph), syndrome, deepcopy(rng))
        streams["random"] = rng
        result = actual_random(syndrome, graph, iterations, rng=rng)
        np.testing.assert_array_equal(result.initial_configuration, expected_initial)
        calls["random"] += 1
        return result

    def learned_decoder(self, syndrome, iterations, *, rng):
        np.testing.assert_array_equal(syndrome, syndromes[calls["flow"]])
        assert rng is not streams["random"]
        expected_initial = sample_uniform_solution(self._H, syndrome, deepcopy(rng))
        result = actual_flow(self, syndrome, iterations, rng=rng)
        np.testing.assert_array_equal(result.initial_configuration, expected_initial)
        calls["flow"] += 1
        return result

    monkeypatch.setattr(benchmark, "decode_mwpm", matching)
    monkeypatch.setattr(benchmark, "decode_random_mcmc", random_decoder)
    monkeypatch.setattr(flow_mcmc.FlowProposalMCMCDecoder, "decode", learned_decoder)
    benchmark.run_benchmark(rounds=1, shots=2, iterations=2, stim_seed=42, mcmc_seed=7,
                            flow_checkpoint=checkpoint, output=tmp_path / "independent")
    assert calls == {"mwpm": 2, "random": 2, "flow": 2}


def test_baseline_stream_unchanged_and_flow_replay(checkpoint, tmp_path):
    settings = dict(rounds=1, shots=2, iterations=3, stim_seed=42, mcmc_seed=7)
    baseline = benchmark.run_benchmark(**settings, output=tmp_path / "baseline")
    first = benchmark.run_benchmark(**settings, flow_checkpoint=checkpoint, output=tmp_path / "first")
    seeds = json.loads((first / "settings.json").read_text())
    replay = benchmark.run_benchmark(**settings, flow_checkpoint=checkpoint, flow_seed=seeds["flow_seed"], output=tmp_path / "replay")
    uniform = benchmark.run_benchmark(**settings, flow_checkpoint=checkpoint, flow_probability=0, output=tmp_path / "uniform")
    baseline_rows = rows(baseline / "shots.csv")
    for path in (first, replay, uniform):
        for old, new in zip(baseline_rows, rows(path / "shots.csv")):
            for key in benchmark.SHOT_FIELDS:
                if not key.endswith("seconds"):
                    assert old[key] == new[key]
        for shot in range(2):
            name = f"shot_{shot:06d}.csv"
            assert (baseline / "traces" / name).read_bytes() == (path / "traces" / name).read_bytes()
    for a, b in zip(rows(first / "shots.csv"), rows(replay / "shots.csv")):
        assert {k: v for k, v in a.items() if not k.endswith("seconds")} == {k: v for k, v in b.items() if not k.endswith("seconds")}
    for shot in range(2):
        name = f"shot_{shot:06d}.csv"
        assert (first / "flow_traces" / name).read_bytes() == (replay / "flow_traces" / name).read_bytes()


def test_three_method_cli_zero_iterations(checkpoint, tmp_path, capsys):
    output = tmp_path / "cli"
    benchmark.main([
        "--rounds", "1", "--shots", "1", "--iterations", "0", "--stim-seed", "42",
        "--mcmc-seed", "7", "--flow-seed", "11", "--flow-checkpoint", str(checkpoint),
        "--flow-probability", "0.7", "--output", str(output),
    ])
    assert "Results saved" in capsys.readouterr().out
    row = rows(output / "shots.csv")[0]
    for key in ("flow_mcmc_acceptance_rate", "flow_mcmc_flow_acceptance_rate", "flow_mcmc_uniform_acceptance_rate"):
        assert float(row[key]) == 0
    for folder in ("traces", "flow_traces"):
        assert rows(output / folder / "shot_000000.csv") == []
    assert "proposal_source" in (output / "flow_traces" / "shot_000000.csv").read_text()
    config = json.loads((output / "settings.json").read_text())
    assert config["flow_seed"] == 11 and config["flow_probability"] == 0.7


@pytest.mark.parametrize("kwargs", [{"rounds": 3}, {"flow_probability": -0.1}, {"flow_probability": float("nan")}, {"flow_seed": -1}])
def test_invalid_configuration_creates_no_output(checkpoint, tmp_path, kwargs):
    options = dict(rounds=1, shots=1, iterations=0, flow_checkpoint=checkpoint)
    options.update(kwargs)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(**options, output=output)
    assert not output.exists()


def test_existing_directory_is_preserved(checkpoint, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "shots.csv"
    marker.write_text("keep these results")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(rounds=1, shots=1, iterations=0, flow_checkpoint=checkpoint, output=output)
    assert marker.read_text() == "keep these results"
