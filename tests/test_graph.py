"""Graph identities, probability weights, and DEM parsing semantics."""

import networkx as nx
import numpy as np
import pytest
import stim

from surface_code.circuit import build_surface_code, get_detector_error_model
from surface_code.decoding_graph import build_decoding_graph


def test_surface_code_graph():
    model = get_detector_error_model(build_surface_code(3, 3, 0.005))
    result = build_decoding_graph(model)
    assert isinstance(result.graph, nx.MultiGraph)
    assert result.num_detectors == 24
    assert result.detector_nodes == tuple(range(24))
    assert result.boundary_node == 24
    assert result.boundary_node not in result.detector_nodes
    assert result.graph.nodes[24]["is_boundary"]
    assert result.graph.number_of_nodes() == 25
    assert result.graph.number_of_edges() == len(result.edges)
    assert len(result.mechanism_edges) == model.num_errors
    assert [e.edge_id for e in result.edges] == list(range(len(result.edges)))
    assert result.edges == build_decoding_graph(model).edges
    assert any(e.logical_flip for e in result.edges)
    for edge in result.edges:
        assert edge.weight == np.log((1 - edge.p_error) / edge.p_error)
        data = result.graph[edge.u][edge.v][edge.edge_id]
        assert data["p_error"] == edge.p_error
        assert data["weight"] == edge.weight
        assert data["logical_mask"] == edge.logical_mask


@pytest.mark.parametrize("p", [0.001, 0.005, 0.1, 0.5, 0.9])
def test_probability_weight(p):
    edge = build_decoding_graph(stim.DetectorErrorModel(f"error({p}) D0")).edges[0]
    assert edge.p_error == p
    assert edge.weight == np.log((1 - p) / p)


def test_parallel_mechanisms_and_boundary():
    result = build_decoding_graph(stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.2) D1 D0 L0
        error(0.1) D0 D1
        error(0.3) D1 L0
        detector D3
    """))
    assert result.graph.number_of_edges(0, 1) == 3
    assert [e.p_error for e in result.edges] == [0.1, 0.2, 0.1, 0.3]
    assert [e.logical_flip for e in result.edges] == [False, True, False, True]
    assert result.mechanism_edges == ((0,), (1,), (2,), (3,))
    assert result.edges[3].detector_endpoints == (1,)
    assert (result.edges[3].u, result.edges[3].v) == (1, 4)
    assert result.graph.degree[3] == 0


def test_separator_components_preserve_correlation_and_logicals():
    result = build_decoding_graph(stim.DetectorErrorModel(
        "error(0.125) D0 D1 L0 ^ D2 L0 L1"
    ))
    a, b = result.edges
    assert result.mechanism_edges == ((0, 1),)
    assert a.mechanism_id == b.mechanism_id == 0
    assert (a.component_index, b.component_index) == (0, 1)
    assert a.p_error == b.p_error == 0.125
    assert a.detector_endpoints == (0, 1)
    assert b.detector_endpoints == (2,)
    assert (a.logical_mask, b.logical_mask) == (1, 3)
    assert a.logical_mask ^ b.logical_mask == 2


def test_nested_repeats_and_shifts():
    result = build_decoding_graph(stim.DetectorErrorModel("""
        shift_detectors(0, 1) 2
        repeat 2 {
            repeat 2 {
                error(0.1) D0 D1 L0
                shift_detectors 1
            }
            detector(1, 2) D0
        }
        error(0.2) D0
        logical_observable L1
    """))
    assert [e.detector_endpoints for e in result.edges] == [
        (2, 3), (3, 4), (4, 5), (5, 6), (6,),
    ]
    assert result.num_detectors == 7
    assert result.num_observables == 2
    assert [e.mechanism_id for e in result.edges] == list(range(5))


def test_zero_detector_errors_and_target_parity():
    result = build_decoding_graph(stim.DetectorErrorModel("""
        error(0.1) L0
        error(0.2) D0 D0 L0 L0
        error(0.3)
    """))
    assert [e.logical_mask for e in result.edges] == [1, 0, 0]
    for edge in result.edges:
        assert edge.detector_endpoints == ()
        assert edge.u == edge.v == result.boundary_node


def test_hyperedge_is_rejected():
    with pytest.raises(ValueError, match="decompose_errors=True"):
        build_decoding_graph(stim.DetectorErrorModel("error(0.1) D0 D1 D2"))


def test_probability_endpoints():
    result = build_decoding_graph(stim.DetectorErrorModel("""
        error(0) D0
        error(1) D1
    """))
    assert result.edges[0].weight == np.inf
    assert result.edges[1].weight == -np.inf


def test_empty_model():
    result = build_decoding_graph(stim.DetectorErrorModel())
    assert result.edges == result.detector_nodes == result.mechanism_edges == ()
    assert list(result.graph.nodes) == [result.boundary_node]
