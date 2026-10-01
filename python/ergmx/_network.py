"""Conversion between igraph/networkx graphs and the arrays of the Rust core."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Network:
    """A binary network: vertex count, edges (0-based), vertex attributes, and
    the dyads whose value is unknown (edges marked ``na``)."""

    n: int
    directed: bool
    edges: np.ndarray  # (m, 2) uint32
    attributes: dict[str, list]
    source: Any = None  # the graph it came from, to build graphs of the same kind
    graph_attributes: dict[str, Any] = field(default_factory=dict)
    missing: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.uint32))

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """Out- and in-degrees (degrees twice, if undirected)."""
        out = np.bincount(self.edges[:, 0], minlength=self.n)
        into = np.bincount(self.edges[:, 1], minlength=self.n)
        if not self.directed:
            out = into = out + into
        return out, into

    def dyad_mask(self, dyads: np.ndarray) -> np.ndarray:
        """n x n boolean mask of the given dyads (both orientations if undirected)."""
        mask = np.zeros((self.n, self.n), dtype=bool)
        if len(dyads):
            mask[dyads[:, 0], dyads[:, 1]] = True
            if not self.directed:
                mask[dyads[:, 1], dyads[:, 0]] = True
        return mask

    def attribute(self, name: str) -> list:
        try:
            return self.attributes[name]
        except KeyError:
            available = ", ".join(sorted(self.attributes)) or "none"
            raise KeyError(
                f"the network has no vertex attribute {name!r} (available: {available})"
            ) from None

    def graph_attribute(self, name: str) -> Any:
        try:
            return self.graph_attributes[name]
        except KeyError:
            available = ", ".join(sorted(self.graph_attributes)) or "none"
            raise KeyError(
                f"the network has no graph attribute {name!r} (available: {available})"
            ) from None


def _edge_array(edges) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(edges, dtype=np.uint32).reshape(-1, 2))


def _split_missing(pairs, flags, directed: bool):
    """Ties and missing dyads (marked by a true `na` edge attribute)."""
    ties, missing = [], []
    for pair, na in zip(pairs, flags):
        (missing if na else ties).append(pair)
    key = (lambda p: p) if directed else (lambda p: tuple(sorted(p)))
    both = {key(p) for p in ties} & {key(p) for p in missing}
    if both:
        raise ValueError(f"dyads {sorted(both)[:5]} are both ties and missing (na=True)")
    if not directed:
        missing = sorted({key(p) for p in missing})
    return _edge_array(ties), _edge_array(missing)


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
        graph_attributes = {a: x[a] for a in x.attributes()}
        flags = [bool(v) for v in x.es["na"]] if "na" in x.es.attributes() else [False] * x.ecount()
        edges, missing = _split_missing(x.get_edgelist(), flags, x.is_directed())
        return Network(x.vcount(), x.is_directed(), edges, attributes, x, graph_attributes, missing)
    if nx is not None and isinstance(x, nx.Graph):
        if x.is_multigraph():
            raise ValueError("ergmx models binary networks: use nx.Graph or nx.DiGraph")
        if nx.number_of_selfloops(x):
            raise ValueError("ergmx models networks without self-loops")
        nodes = list(x)
        index = {node: i for i, node in enumerate(nodes)}
        pairs = [(index[u], index[v]) for u, v in x.edges()]
        flags = [bool(data.get("na", False)) for _, _, data in x.edges(data=True)]
        edges, missing = _split_missing(pairs, flags, x.is_directed())
        names = {k for _, data in x.nodes(data=True) for k in data}
        attributes = {a: [x.nodes[v].get(a) for v in nodes] for a in names}
        return Network(len(nodes), x.is_directed(), edges, attributes, x, dict(x.graph), missing)
    raise TypeError(f"expected an igraph or networkx graph, got {type(x).__name__}")


def to_graph(network: Network, edges: np.ndarray) -> Any:
    """A graph of the same kind as `network.source`, with the given edges."""
    source = network.source
    ig = sys.modules.get("igraph")
    if ig is not None and isinstance(source, ig.Graph):
        g = ig.Graph(n=network.n, edges=edges.tolist(), directed=network.directed)
        for name, values in network.attributes.items():
            g.vs[name] = values
        for name, value in network.graph_attributes.items():
            g[name] = value
        return g
    g = source.__class__()
    g.graph.update(source.graph)
    nodes = list(source)
    g.add_nodes_from(source.nodes(data=True))
    g.add_edges_from((nodes[i], nodes[j]) for i, j in edges.tolist())
    return g
