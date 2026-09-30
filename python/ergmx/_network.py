"""Conversion between igraph/networkx graphs and the arrays of the Rust core."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Network:
    """A binary network: vertex count, edges (0-based) and vertex attributes."""

    n: int
    directed: bool
    edges: np.ndarray  # (m, 2) uint32
    attributes: dict[str, list]
    source: Any = None  # the graph it came from, to build graphs of the same kind

    def attribute(self, name: str) -> list:
        try:
            return self.attributes[name]
        except KeyError:
            available = ", ".join(sorted(self.attributes)) or "none"
            raise KeyError(
                f"the network has no vertex attribute {name!r} (available: {available})"
            ) from None


def _edge_array(edges) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(edges, dtype=np.uint32).reshape(-1, 2))


def as_network(x: Any) -> Network:
    """Read an igraph or networkx graph."""
    if isinstance(x, Network):
        return x
    ig = sys.modules.get("igraph")
    nx = sys.modules.get("networkx")
    if ig is not None and isinstance(x, ig.Graph):
        if x.has_multiple():
            raise ValueError("ergmx models binary networks: the graph has multiple edges")
        if any(x.is_loop()):
            raise ValueError("ergmx models networks without self-loops")
        attributes = {a: x.vs[a] for a in x.vs.attributes()}
        return Network(x.vcount(), x.is_directed(), _edge_array(x.get_edgelist()), attributes, x)
    if nx is not None and isinstance(x, nx.Graph):
        if x.is_multigraph():
            raise ValueError("ergmx models binary networks: use nx.Graph or nx.DiGraph")
        if nx.number_of_selfloops(x):
            raise ValueError("ergmx models networks without self-loops")
        nodes = list(x)
        index = {node: i for i, node in enumerate(nodes)}
        edges = _edge_array([(index[u], index[v]) for u, v in x.edges()])
        names = {k for _, data in x.nodes(data=True) for k in data}
        attributes = {a: [x.nodes[v].get(a) for v in nodes] for a in names}
        return Network(len(nodes), x.is_directed(), edges, attributes, x)
    raise TypeError(f"expected an igraph or networkx graph, got {type(x).__name__}")


def to_graph(network: Network, edges: np.ndarray) -> Any:
    """A graph of the same kind as `network.source`, with the given edges."""
    source = network.source
    ig = sys.modules.get("igraph")
    if ig is not None and isinstance(source, ig.Graph):
        g = ig.Graph(n=network.n, edges=edges.tolist(), directed=network.directed)
        for name, values in network.attributes.items():
            g.vs[name] = values
        return g
    g = source.__class__()
    nodes = list(source)
    g.add_nodes_from(source.nodes(data=True))
    g.add_edges_from((nodes[i], nodes[j]) for i, j in edges.tolist())
    return g
