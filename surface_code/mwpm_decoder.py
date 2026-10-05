"""Independent MWPM baseline for the graph's additive edge-weight objective.

DEM components are treated as independent edges here. Correlations recorded in
mechanism_edges are not enforced by this baseline.
"""

from typing import NamedTuple, Sequence

import networkx as nx
import numpy as np

from surface_code.decoding_graph import DecodingGraph


class MWPMResult(NamedTuple):
    """L0 prediction, sorted correction edge IDs, and their original weight sum."""

    prediction: int
    selected_edge_ids: tuple[int, ...]
    total_weight: float


def decode_mwpm(
    syndrome: Sequence[int] | np.ndarray, decoding_graph: DecodingGraph
) -> MWPMResult:
    """Find a minimum-weight edge subset with the requested detector parity.

    Syndrome entries follow decoding_graph.detector_nodes, without a boundary
    entry. Weighted shortest paths form a metric graph of syndrome defects.
    Each defect has a private virtual partner for a path to the boundary;
    zero-cost links between partners allow any number of boundary terminations.
    NetworkX blossom matching pairs every metric-graph vertex. Matched paths
    are XORed, so a shared error edge cancels instead of being counted twice.

    Parallel edges retain their IDs: shortest paths use the cheapest edge for
    each endpoint pair (lowest ID for an exact tie). Other matching ties may
    yield any optimal correction. Stored weights are never replaced by hop
    counts or recomputed from probabilities.

    Negative-weight edges are initially selected and their detector parity is
    removed from the syndrome. Matching then finds toggles with costs abs(w),
    an equivalent nonnegative optimization. Infinite weights represent fixed
    edges: +inf is excluded and -inf is selected, corresponding to p=0 and p=1.
    Raises ValueError for invalid syndromes or an infeasible correction.
    """
    values = np.asarray(syndrome)
    if values.shape != (decoding_graph.num_detectors,):
        raise ValueError("Syndrome must have one entry per detector, without boundary.")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Syndrome entries must be binary (0 or 1).")
    residual = {
        node for node, bit in zip(decoding_graph.detector_nodes, values) if bit
    }
    selected: set[int] = set()
    paths_graph = nx.Graph()
    paths_graph.add_nodes_from(decoding_graph.detector_nodes)
    paths_graph.add_node(decoding_graph.boundary_node)
    edges_by_id = {edge.edge_id: edge for edge in decoding_graph.edges}
    for edge in sorted(decoding_graph.edges, key=lambda edge: edge.edge_id):
        if np.isnan(edge.weight):
            raise ValueError(f"Edge {edge.edge_id} has a NaN weight.")
        if edge.weight < 0:
            selected.add(edge.edge_id)
            residual.symmetric_difference_update(edge.detector_endpoints)
        if not np.isfinite(edge.weight) or edge.u == edge.v:
            continue
        cost = abs(edge.weight)
        existing = paths_graph.get_edge_data(edge.u, edge.v)
        if existing is None or cost < existing["weight"]:
            paths_graph.add_edge(
                edge.u, edge.v, weight=cost, edge_id=edge.edge_id
            )

    defects = sorted(residual)
    n = len(defects)
    metric = nx.Graph()
    metric.add_nodes_from(range(2 * n))
    shortest_paths = {}
    for i, source in enumerate(defects):
        distances, paths = nx.single_source_dijkstra(paths_graph, source)
        shortest_paths[i] = paths
        for j in range(i + 1, n):
            target = defects[j]
            if target in distances:
                metric.add_edge(i, j, weight=distances[target])
        boundary = decoding_graph.boundary_node
        if boundary in distances:
            metric.add_edge(i, n + i, weight=distances[boundary])
        for j in range(i + 1, n):
            metric.add_edge(n + i, n + j, weight=0.0)

    matching = nx.min_weight_matching(metric, weight="weight")
    if len(matching) != n:
        raise ValueError("Syndrome has no correction using the allowed graph edges.")
    for a, b in matching:
        a, b = sorted((a, b))
        if a >= n:
            continue  # A pair of unused boundary partners has no physical path.
        target = defects[b] if b < n else decoding_graph.boundary_node
        path = shortest_paths[a][target]
        for u, v in zip(path, path[1:]):
            selected.symmetric_difference_update((paths_graph[u][v]["edge_id"],))

    edge_ids = tuple(sorted(selected))
    prediction = 0
    for edge_id in edge_ids:
        prediction ^= int(edges_by_id[edge_id].logical_flip)
    total_weight = float(sum(edges_by_id[i].weight for i in edge_ids))
    return MWPMResult(prediction, edge_ids, total_weight)
