"""Parse Stim DEMs into a multigraph without merging error identities.

Each graphlike component of a DEM error instruction becomes an edge. Components
separated by ``^`` share a mechanism_id and probability: they are correlated,
not independent Bernoulli edges. An independent-edge decoder is an approximation.
An exact DEM likelihood must select a group's components together and count its
probability-derived weight once, not once per component. Stim may already have combined circuit faults
when constructing the DEM; identities here refer to DEM error instructions.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import stim


@dataclass(frozen=True)
class ErrorEdge:
    """One component in deterministic DEM traversal order.

    detector_endpoints excludes the boundary. Zero-detector components (including
    pure logical errors) use a boundary self-loop, contributing no syndrome.
    logical_mask packs observable flips into bits; bit zero corresponds to L0.
    """

    edge_id: int
    u: int
    v: int
    detector_endpoints: tuple[int, ...]
    p_error: float
    weight: float
    logical_mask: int
    mechanism_id: int
    component_index: int

    @property
    def logical_flip(self) -> bool:
        return bool(self.logical_mask & 1)


@dataclass(frozen=True)
class DecodingGraph:
    """Graph plus the authoritative ordering for future binary edge vectors.

    Use ``edges[i]`` for edge-vector index i, not NetworkX iteration order.
    Syndrome rows correspond only to ``detector_nodes``. The boundary node is
    num_detectors and must never be assigned a syndrome row. mechanism_edges[m]
    lists the edge IDs produced by the m-th expanded DEM error instruction.
    """

    graph: nx.MultiGraph
    edges: tuple[ErrorEdge, ...]
    detector_nodes: tuple[int, ...]
    boundary_node: int
    num_observables: int
    mechanism_edges: tuple[tuple[int, ...], ...]

    @property
    def num_detectors(self) -> int:
        return len(self.detector_nodes)


def build_decoding_graph(model: stim.DetectorErrorModel) -> DecodingGraph:
    """Parse errors, separators, nested repeats, and detector-index shifts.

    Annotation instructions have no error edges. Repeated targets cancel modulo
    two. Undecomposed components with more than two detectors raise ValueError;
    obtain a graphlike DEM with circuit.detector_error_model(decompose_errors=True).
    Probabilities are kept unchanged, including 0 and 1 (weights +inf and -inf).
    """
    boundary = model.num_detectors
    detector_nodes = tuple(range(boundary))
    graph = nx.MultiGraph()
    graph.add_nodes_from(detector_nodes, is_boundary=False)
    graph.add_node(boundary, is_boundary=True)
    edges: list[ErrorEdge] = []
    groups: list[tuple[int, ...]] = []

    def add_error(instruction: stim.DemInstruction, offset: int) -> None:
        p_error = instruction.args_copy()[0]
        with np.errstate(divide="ignore"):
            weight = float(np.log((1 - np.float64(p_error)) / np.float64(p_error)))
        components: list[list[stim.DemTarget]] = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                components.append([])
            else:
                components[-1].append(target)
        mechanism_id = len(groups)
        edge_ids = []
        for component_index, targets in enumerate(components):
            detectors: set[int] = set()
            logical_mask = 0
            for target in targets:
                if target.is_relative_detector_id():
                    detectors.symmetric_difference_update((offset + target.val,))
                elif target.is_logical_observable_id():
                    logical_mask ^= 1 << target.val
                else:
                    raise ValueError(f"Unsupported error target: {target}")
            endpoints = tuple(sorted(detectors))
            if len(endpoints) > 2:
                raise ValueError(
                    f"Error mechanism {mechanism_id}, component {component_index} "
                    f"has {len(endpoints)} detectors; use decompose_errors=True."
                )
            u = endpoints[0] if endpoints else boundary
            v = endpoints[1] if len(endpoints) == 2 else boundary
            edge = ErrorEdge(
                edge_id=len(edges), u=u, v=v, detector_endpoints=endpoints,
                p_error=p_error, weight=weight, logical_mask=logical_mask,
                mechanism_id=mechanism_id, component_index=component_index,
            )
            edges.append(edge)
            edge_ids.append(edge.edge_id)
            graph.add_edge(
                u, v, key=edge.edge_id, edge_id=edge.edge_id,
                detector_endpoints=endpoints, p_error=p_error, weight=weight,
                logical_mask=logical_mask, logical_flip=edge.logical_flip,
                mechanism_id=mechanism_id, component_index=component_index,
            )
        groups.append(tuple(edge_ids))

    def walk(block: stim.DetectorErrorModel, offset: int) -> int:
        for instruction in block:
            if isinstance(instruction, stim.DemRepeatBlock):
                body = instruction.body_copy()
                for _ in range(instruction.repeat_count):
                    offset = walk(body, offset)
            elif instruction.type == "shift_detectors":
                offset += instruction.targets_copy()[0]
            elif instruction.type == "error":
                add_error(instruction, offset)
            elif instruction.type not in ("detector", "logical_observable"):
                raise ValueError(f"Unsupported DEM instruction: {instruction.type}")
        return offset

    walk(model, 0)
    return DecodingGraph(
        graph=graph, edges=tuple(edges), detector_nodes=detector_nodes,
        boundary_node=boundary, num_observables=model.num_observables,
        mechanism_edges=tuple(groups),
    )


if __name__ == "__main__":
    from surface_code.circuit import build_surface_code, get_detector_error_model

    model = get_detector_error_model(build_surface_code(3, 3, 0.005))
    result = build_decoding_graph(model)
    print(f"Detector count: {result.num_detectors}")
    print(f"Edge count: {len(result.edges)}")
    print(f"Boundary node: {result.boundary_node} (excluded from syndrome rows)")
    print(f"DEM mechanisms: {len(result.mechanism_edges)}")
    print("First 10 edges:")
    for edge in result.edges[:10]:
        print(
            f"  edge_id={edge.edge_id} endpoints=({edge.u}, {edge.v}) "
            f"p_e={edge.p_error:.12g} w_e={edge.weight:.12g} "
            f"logical_L0={edge.logical_flip} mechanism_id={edge.mechanism_id}"
        )
