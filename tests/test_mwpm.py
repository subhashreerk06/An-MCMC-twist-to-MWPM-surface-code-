"""Known corrections and exhaustive checks of the independent MWPM baseline."""

from itertools import product

import numpy as np
import pytest
import stim

from surface_code.circuit import (
    build_surface_code,
    get_detector_error_model,
    sample_shots,
)
from surface_code.decoding_graph import build_decoding_graph
from surface_code.mwpm_decoder import decode_mwpm


def graph_from(text):
    return build_decoding_graph(stim.DetectorErrorModel(text))


def correction_syndrome(graph, edge_ids):
    syndrome = np.zeros(graph.num_detectors, dtype=np.uint8)
    for edge_id in edge_ids:
        for detector in graph.edges[edge_id].detector_endpoints:
            syndrome[detector] ^= 1
    return syndrome


def test_shortest_path_beats_direct_edge():
    graph = graph_from("""
        error(0.2) D0 D1 L0
        error(0.2) D1 D2
        error(0.01) D0 D2
    """)
    prediction, selected, weight = decode_mwpm([1, 0, 1], graph)
    assert selected == (0, 1)
    assert prediction == 1
    assert weight == graph.edges[0].weight + graph.edges[1].weight


def test_multiple_boundary_terminations_and_logical_parity():
    graph = graph_from("""
        error(0.2) D0 L0
        error(0.2) D1 L0
        error(0.01) D0 D1
    """)
    result = decode_mwpm([1, 1], graph)
    assert result.selected_edge_ids == (0, 1)
    assert result.prediction == 0
    single = decode_mwpm([1, 0], graph)
    assert single.selected_edge_ids == (0,)
    assert single.prediction == 1


def test_four_defects_choose_minimum_pairing():
    graph = graph_from("""
        error(0.3) D0 D1
        error(0.3) D2 D3 L0
        error(0.01) D0 D2
        error(0.01) D1 D3
    """)
    result = decode_mwpm([1, 1, 1, 1], graph)
    assert result.selected_edge_ids == (0, 1)
    assert result.prediction == 1


def test_parallel_edges_preserve_selected_identity():
    graph = graph_from("""
        error(0.01) D0 D1
        error(0.2) D0 D1 L0
        error(0.2) D0 D1
    """)
    result = decode_mwpm([1, 1], graph)
    assert result.selected_edge_ids == (1,)
    assert result.prediction == 1
    assert result.total_weight == graph.edges[1].weight


def test_zero_syndrome_and_empty_graph():
    graph = graph_from("error(0.1) D0 L0")
    assert decode_mwpm([0], graph) == (0, (), 0.0)
    assert decode_mwpm([], graph_from("")) == (0, (), 0.0)


@pytest.mark.parametrize("syndrome", [[1], [0, 0, 0], [[0, 1]], [0, 2], [0, np.nan]])
def test_invalid_syndrome(syndrome):
    with pytest.raises(ValueError, match="Syndrome"):
        decode_mwpm(syndrome, graph_from("error(0.1) D0 D1"))


def test_unreachable_syndrome():
    graph = graph_from("error(0.1) D0 D1\ndetector D2")
    for syndrome in ([1, 0, 0], [0, 0, 1]):
        with pytest.raises(ValueError, match="no correction"):
            decode_mwpm(syndrome, graph)


def test_probability_zero_and_one_are_fixed():
    graph = graph_from("error(0) D0\nerror(1) D1 L0")
    assert decode_mwpm([0, 1], graph) == (1, (1,), -np.inf)
    for syndrome in ([1, 1], [0, 0]):
        with pytest.raises(ValueError, match="no correction"):
            decode_mwpm(syndrome, graph)


@pytest.mark.parametrize("p", [0.15, 0.5, 0.8])
def test_matches_exhaustive_minimum_for_every_syndrome(p):
    # Includes parallel edges, a cycle, two boundary edges, and a logical loop.
    graph = graph_from(f"""
        error({p}) D0 D1 L0
        error(0.2) D0 D1
        error(0.3) D1 D2
        error(0.1) D0 D2
        error(0.25) D2 D3 L0
        error(0.2) D0
        error(0.1) D3
        error({p}) L0
    """)
    minima = {}
    for bits in product((0, 1), repeat=len(graph.edges)):
        ids = tuple(i for i, bit in enumerate(bits) if bit)
        syndrome = tuple(correction_syndrome(graph, ids))
        weight = sum(graph.edges[i].weight for i in ids)
        minima[syndrome] = min(minima.get(syndrome, np.inf), weight)
    for syndrome in product((0, 1), repeat=4):
        result = decode_mwpm(syndrome, graph)
        np.testing.assert_array_equal(
            correction_syndrome(graph, result.selected_edge_ids), syndrome
        )
        assert result.total_weight == pytest.approx(minima[syndrome])
        assert result.total_weight == sum(
            graph.edges[i].weight for i in result.selected_edge_ids
        )
        assert result.prediction == sum(
            graph.edges[i].logical_flip for i in result.selected_edge_ids
        ) % 2


def test_sampled_surface_code_syndromes():
    circuit = build_surface_code(3, 3, 0.005)
    graph = build_decoding_graph(get_detector_error_model(circuit))
    detectors, _ = sample_shots(circuit, 20, seed=7)
    for syndrome in detectors:
        result = decode_mwpm(syndrome, graph)
        np.testing.assert_array_equal(
            correction_syndrome(graph, result.selected_edge_ids), syndrome
        )
