"""Regression checks for the intended independent affine-proposal experiment."""

import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

from surface_code import random_mcmc
from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
from surface_code.decoding_graph import build_decoding_graph


def test_random_mcmc_matches_pre_extension_source_checksum():
    # Recorded from the original source before any affine/flow extensions.
    expected = "6269a08ceb763ba3ffd0416d26a9e855305a9a6d441686c8199f27a41325636c"
    assert hashlib.sha256(Path(random_mcmc.__file__).read_bytes()).hexdigest() == expected


def test_dataset_and_training_run_without_decoders_or_outcome_supervision(tmp_path):
    script = textwrap.dedent("""
        import builtins
        import sys
        from pathlib import Path

        original_import = builtins.__import__
        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            names = [name, *(fromlist or ())]
            if any('mwpm' in item.lower() or 'mcmc' in item.lower() for item in names):
                raise AssertionError('Training attempted to import a decoder')
            return original_import(name, globals, locals, fromlist, level)
        builtins.__import__ = blocked_import

        import networkx as nx
        import numpy as np
        from surface_code import circuit
        from surface_code.flow_dataset import generate_dataset
        from surface_code.train_flow import train

        def forbidden(*args, **kwargs):
            raise AssertionError('Training requested circuit outcomes or a graph decoder')
        circuit.sample_shots = forbidden
        circuit.compile_detector_sampler = forbidden
        nx.cycle_basis = forbidden
        nx.shortest_path = forbidden
        nx.single_source_dijkstra = forbidden
        nx.min_weight_matching = forbidden

        root = Path(sys.argv[1])
        dataset = generate_dataset(rounds=1, train_samples=13, validation_samples=5,
                                   output=root / 'data', seed=12)
        with np.load(dataset / 'samples.npz') as archive:
            arrays = dict(archive)
        arrays.update(actual_logical=np.ones(13), mwpm_solutions=np.ones((13, 9)),
                      mcmc_best_states=np.ones((13, 9)))
        np.savez(dataset / 'samples.npz', **arrays)

        # Any attempt to read supplemental labels or selected states now fails.
        original_load = np.load
        read_keys = []
        allowed = {'train_s', 'train_z', 'validation_s', 'validation_z'}
        class GuardedArchive:
            def __init__(self, archive):
                self.archive = archive
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.archive.close()
            def __getitem__(self, key):
                assert key in allowed, f'Unexpected training supervision: {key}'
                read_keys.append(key)
                return self.archive[key]
        np.load = lambda *args, **kwargs: GuardedArchive(original_load(*args, **kwargs))
        output = train(dataset=dataset, output=root / 'trained', epochs=2,
                       batch_size=4, hidden_dim=8, seed=9)
        assert set(read_keys) == allowed
        assert (output / 'best.pt').is_file()
        assert not any('mwpm' in name.lower() or 'mcmc' in name.lower() for name in sys.modules)
        print('Training completed using only physical (s, z) pairs')
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Training completed using only physical (s, z) pairs" in completed.stdout


def test_flow_decoder_has_no_matching_or_incremental_edge_flip_operations():
    path = Path(__file__).resolve().parents[1] / "surface_code" / "flow_mcmc.py"
    tree = ast.parse(path.read_text())
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree) if isinstance(node, (ast.Name, ast.Attribute))
    }
    assert not names.intersection({"cycle_basis", "symmetric_difference_update", "decode_mwpm"})
    assert not any("cycle" in name.lower() or "mwpm" in name.lower() for name in names)
    assert not any(isinstance(node, ast.BitXor) for node in ast.walk(tree))


def test_mcmc_runs_in_fresh_process_with_mwpm_imports_blocked():
    script = textwrap.dedent("""
        import builtins
        import sys

        original_import = builtins.__import__
        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if 'mwpm' in name.lower() or any('mwpm' in item.lower() for item in (fromlist or ())):
                raise AssertionError('MCMC attempted to import MWPM')
            return original_import(name, globals, locals, fromlist, level)
        builtins.__import__ = blocked_import

        import numpy as np
        from surface_code.circuit import build_surface_code, get_detector_error_model, sample_shots
        from surface_code.decoding_graph import build_decoding_graph
        from surface_code.gf2 import build_incidence_matrix
        from surface_code.random_mcmc import decode_random_mcmc

        circuit = build_surface_code(3, 3, 0.005)
        graph = build_decoding_graph(get_detector_error_model(circuit))
        syndrome = sample_shots(circuit, 10, seed=42)[0][0]
        result = decode_random_mcmc(syndrome, graph, 10, seed=7)
        H = build_incidence_matrix(graph)
        for vector in (result.initial_configuration, result.final_configuration, result.best_configuration):
            np.testing.assert_array_equal((H @ vector) % 2, syndrome)
        assert len(result.trace) == 10
        assert not any('mwpm' in name.lower() for name in sys.modules)
        print('MCMC completed with MWPM imports blocked')
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "MCMC completed with MWPM imports blocked" in completed.stdout


def test_every_proposal_is_valid_and_best_is_minimum_encountered(monkeypatch):
    circuit = build_surface_code(3, 3, 0.005)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    syndrome = sample_shots(circuit, 10, seed=42)[0][0]
    sampler = random_mcmc.sample_uniform_solution
    configurations = []

    def checked_sampler(H, s, rng):
        np.testing.assert_array_equal(s, syndrome)
        vector = sampler(H, s, rng)
        np.testing.assert_array_equal((H @ vector) % 2, s)
        configurations.append(vector.copy())
        return vector

    monkeypatch.setattr(random_mcmc, "sample_uniform_solution", checked_sampler)
    result = random_mcmc.decode_random_mcmc(syndrome, graph, 30, seed=7)
    assert len(configurations) == 31
    weights = np.array([edge.weight for edge in graph.edges])
    encountered = [float(np.sum(vector * weights)) for vector in configurations]
    assert result.initial_weight == encountered[0]
    assert [entry.proposed_weight for entry in result.trace] == encountered[1:]
    assert result.best_weight == min(encountered)
    for i, entry in enumerate(result.trace, start=1):
        assert entry.best_weight_so_far == min(encountered[:i + 1])
    np.testing.assert_array_equal(result.best_configuration, configurations[np.argmin(encountered)])


def test_no_cycle_implementation_and_random_mcmc_remains_independent():
    package = Path(__file__).resolve().parents[1] / "surface_code"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.alias):
                names.append(node.name)
        assert not any("cycle" in name.lower() for name in names), path
        if path.name == "random_mcmc.py":
            assert "symmetric_difference_update" not in names
            assert not any("mwpm" in name.lower() for name in names)
            assert not any("flow" in name.lower() for name in names)
            assert not any(isinstance(node, ast.BitXor) for node in ast.walk(tree))
