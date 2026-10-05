"""Conversion between igraph/networkx graphs and the arrays of the Rust core."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, repr=False)
class Block:
    """One network of a combined network (:func:`ergmx.Networks`,
    :func:`ergmx.NetSeries`): its vertices are ``start`` to ``start + network.n
    - 1`` of the combined network."""

    start: int
    network: Network
    #: Network-level attributes, for the linear models of N() and tergm's
    #: operators: the graph attributes, n, and .NetworkID, .NetworkName (or,
    #: in a series, .Time, .TimeID and .TimeDelta).
    attributes: dict[str, Any]
    #: In a series, the network at the previous time.
    prev: Network | None = None

    @property
    def stop(self) -> int:
        return self.start + self.network.n


@dataclass(frozen=True, repr=False)
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
    #: For bipartite networks, each vertex's mode: 1 (the first mode, ergm's
    #: "b1") or 2. None if the network is not bipartite.
    mode: np.ndarray | None = None
    #: For networks combined from several, each of them.
    blocks: tuple[Block, ...] | None = None

    @property
    def bipartite(self) -> bool:
        return self.mode is not None

    @property
    def combined(self) -> bool:
        """Whether the network combines several (Networks() or NetSeries())."""
        return self.blocks is not None

    @property
    def series(self) -> bool:
        """Whether the network is a series of transitions (NetSeries())."""
        return self.blocks is not None and self.blocks[0].prev is not None

    def block_ids(self) -> np.ndarray | None:
        """Each vertex's network in a combined network (None if not combined)."""
        if self.blocks is None:
            return None
        ids = np.full(self.n, -1, dtype=np.int64)
        for k, b in enumerate(self.blocks):
            ids[b.start:b.stop] = k
        return ids

    def between_blocks(self) -> np.ndarray:
        """n x n mask of the dyads between different networks of a combined network."""
        mask = np.zeros((self.n, self.n), dtype=bool)
        if self.blocks is not None:
            mask[:] = True
            for b in self.blocks:
                mask[b.start:b.stop, b.start:b.stop] = False
        return mask

    def split(self, edges: np.ndarray) -> list[np.ndarray]:
        """The edges of each block of a combined network, numbered within the block."""
        out = []
        for b in self.blocks:
            inside = (edges[:, 0] >= b.start) & (edges[:, 0] < b.stop)
            out.append(_edge_array(edges[inside] - b.start))
        return out

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        if self.blocks is None:
            return f"<Network: {self.n} vertices, {len(self.edges)} edges, {kind}>"
        layers = self.graph_attributes.get("_layers")
        if layers is not None:
            return f"<Layer: {', '.join(layers)} of {self.blocks[0].network.n} vertices, {kind}>"
        what = "transitions" if self.series else "networks"
        sizes = sorted({b.network.n for b in self.blocks})
        size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
        return f"<{'NetSeries' if self.series else 'Networks'}: {len(self.blocks)} {what} of {size} vertices, {kind}>"

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


def _with_modes(network: Network, bipartite) -> Network:
    """The network with the modes of a bipartite network, read from the vertex
    attribute `bipartite` (True: igraph's "type" or networkx's "bipartite")."""
    if bipartite is None or bipartite is False:
        return network
    if network.directed:
        raise ValueError("bipartite networks must be undirected")
    if bipartite is True:
        ig = sys.modules.get("igraph")
        bipartite = "type" if ig is not None and isinstance(network.source, ig.Graph) else "bipartite"
    values = network.attributes.get(bipartite)
    if values is None:
        raise KeyError(f"no vertex attribute {bipartite!r} with the vertices' modes; pass "
                       "bipartite='<attribute>', whose false (or 0) vertices are the first mode")
    if any(v is None for v in values):
        raise ValueError(f"every vertex needs a mode: {bipartite!r} is missing for some")
    mode = np.where([bool(v) for v in values], 2, 1)
    for kind, pairs in (("ties", network.edges), ("missing dyads", network.missing)):
        if len(pairs) and np.any(mode[pairs[:, 0]] == mode[pairs[:, 1]]):
            raise ValueError(f"a bipartite network has no {kind} within a mode, but this one has")
    return Network(network.n, network.directed, network.edges, network.attributes, network.source,
                   network.graph_attributes, network.missing, mode, network.blocks)


def as_network(x: Any, bipartite=None) -> Network:
    """Read an igraph or networkx graph; `bipartite` names the vertex attribute
    with the modes of a bipartite network (True for the default name)."""
    if isinstance(x, Network):
        if x.combined and bipartite not in (None, False) and x.mode is None:
            raise ValueError("for combined bipartite networks, give bipartite= to Networks() or "
                             "NetSeries()")
        return _with_modes(x, bipartite) if bipartite is not None and x.mode is None else x
    return _with_modes(_read(x), bipartite)


def _read(x: Any) -> Network:
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


def to_graphs(network: Network, edges: np.ndarray) -> Any:
    """A graph like `network`'s with the given edges, or, for a combined
    network, a list of graphs: one per network (for Layer(), a dict of them
    by layer name, which Layer() takes back)."""
    if network.blocks is None:
        return to_graph(network, edges)
    graphs = [to_graph(b.network, e) for b, e in zip(network.blocks, network.split(edges))]
    layers = network.graph_attributes.get("_layers")
    return dict(zip(layers, graphs)) if layers is not None else graphs


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
