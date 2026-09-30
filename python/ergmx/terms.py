"""Model terms, with the same names and statistics as the R package ergm.

Terms can be combined with ``+`` or written as a formula string::

    edges() + nodematch("Grade") + gwesp(0.5, fixed=True)
    "edges + nodematch('Grade') + gwesp(0.5, fixed=True)"
"""

from __future__ import annotations

import numpy as np

from ._network import Network


class Term:
    """A model term: one or more network statistics."""

    #: Whether the term's change statistic only depends on the toggled dyad.
    dyad_independent = True
    #: Whether the term counts triangles, so that triadic proposals help the MCMC mix.
    triadic = False
    #: True if only defined for directed networks, False if only for undirected.
    directed: bool | None = None

    def names(self, network: Network) -> list[str]:
        return [self.label]

    def spec(self, network: Network) -> tuple[str, list[float], list[int]]:
        """The term as the Rust core expects it: (name, real params, integer params)."""
        return (self.rust_name, [], [])

    @property
    def rust_name(self) -> str:
        return type(self).__name__.lower()

    @property
    def label(self) -> str:
        return self.rust_name

    def check(self, network: Network) -> None:
        if self.directed is not None and self.directed != network.directed:
            kind = "directed" if self.directed else "undirected"
            raise ValueError(f"{self!r} is only implemented for {kind} networks")

    def __add__(self, other) -> Formula:
        return Formula([self]) + other

    def __radd__(self, other) -> Formula:
        return as_formula(other) + self

    def __repr__(self) -> str:
        return f"{self.rust_name}()"


class Formula:
    """A sum of model terms."""

    def __init__(self, terms):
        self.terms = list(terms)

    def __add__(self, other) -> Formula:
        return Formula(self.terms + as_formula(other).terms)

    def __iter__(self):
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return " + ".join(map(repr, self.terms))


def as_formula(x) -> Formula:
    if isinstance(x, Formula):
        return x
    if isinstance(x, Term):
        return Formula([x])
    if isinstance(x, str):
        from .formula import parse_formula

        return parse_formula(x)
    raise TypeError(f"expected a formula string or terms, got {type(x).__name__}")


def _require_fixed(name: str, fixed: bool) -> None:
    if not fixed:
        raise NotImplementedError(
            f"{name} with an estimated decay (a curved ERGM) is not supported yet; "
            f"use {name}(decay, fixed=True)"
        )


# -- Dyadic and reciprocity terms --------------------------------------------------


class Edges(Term):
    pass


class Mutual(Term):
    dyad_independent = False
    directed = True


# -- Degree terms ----------------------------------------------------------------------


class _Stars(Term):
    dyad_independent = False

    def __init__(self, k):
        ks = [k] if isinstance(k, int) else list(k)
        if not ks or not all(isinstance(v, int) and v >= 1 for v in ks):
            raise ValueError(f"{self.rust_name}: k must be one or more integers >= 1, not {k!r}")
        self.ks = ks

    def names(self, network):
        return [f"{self.rust_name}{k}" for k in self.ks]

    def spec(self, network):
        return (self.rust_name, [], self.ks)

    def __repr__(self) -> str:
        return f"{self.rust_name}({self.ks[0] if len(self.ks) == 1 else self.ks})"


class KStar(_Stars):
    directed = False


class IStar(_Stars):
    directed = True


class OStar(_Stars):
    directed = True


class _Decay(Term):
    """A geometrically weighted term with a fixed decay."""

    dyad_independent = False
    #: Name of the statistic in ergm, before ".fixed.<decay>".
    stat_name = ""

    def __init__(self, decay: float):
        self.decay = float(decay)

    @property
    def label(self) -> str:
        return f"{self.stat_name}.fixed.{self.decay:g}"

    def spec(self, network):
        return (self.rust_name, [self.decay], [])

    def __repr__(self) -> str:
        return f"{self.rust_name}({self.decay:g}, fixed=True)"


class GwDegree(_Decay):
    directed = False
    stat_name = "gwdeg"


class GwIDegree(_Decay):
    directed = True
    stat_name = "gwideg"


class GwODegree(_Decay):
    directed = True
    stat_name = "gwodeg"


# -- Triad terms ---------------------------------------------------------------------------


class Triangle(Term):
    """Triangles; in directed networks, transitive plus cyclic triples."""

    dyad_independent = False
    triadic = True


class TTriple(Term):
    dyad_independent = False
    triadic = True
    directed = True


class CTriple(Term):
    dyad_independent = False
    triadic = True
    directed = True


class Gwesp(_Decay):
    """gwesp; directed networks use ergm's default outgoing two-paths (OTP)."""

    triadic = True

    def names(self, network):
        kind = ".OTP" if network.directed else ""
        return [f"gwesp{kind}.fixed.{self.decay:g}"]


class Gwdsp(_Decay):
    triadic = True
    directed = False
    stat_name = "gwdsp"


# -- Vertex and dyad attribute terms -------------------------------------------------------


class _AttributeTerm(Term):
    def __init__(self, attr: str):
        self.attr = attr

    @property
    def label(self) -> str:
        return f"{self.rust_name}.{self.attr}"

    def __repr__(self) -> str:
        return f"{self.rust_name}({self.attr!r})"


def _codes(values: list) -> tuple[list, list[int]]:
    """Sorted distinct values and each value's position among them."""
    levels = sorted(set(values))
    position = {v: i for i, v in enumerate(levels)}
    return levels, [position[v] for v in values]


def _level_name(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # GraphML stores R integers as doubles
    return str(value)


class NodeMatch(_AttributeTerm):
    def __init__(self, attr: str, diff: bool = False):
        super().__init__(attr)
        self.diff = diff

    def names(self, network):
        if not self.diff:
            return [self.label]
        levels, _ = _codes(network.attribute(self.attr))
        return [f"{self.label}.{_level_name(v)}" for v in levels]

    def spec(self, network):
        name = "nodematchdiff" if self.diff else "nodematch"
        return (name, [], _codes(network.attribute(self.attr))[1])

    def __repr__(self) -> str:
        return f"nodematch({self.attr!r}, diff=True)" if self.diff else super().__repr__()


class NodeFactor(_AttributeTerm):
    """One statistic per attribute level except the first (sorted) one, as in
    ergm's default ``levels = -1``."""

    def _levels(self, network):
        levels, codes = _codes(network.attribute(self.attr))
        return levels[1:], [c - 1 for c in codes]

    def names(self, network):
        return [f"{self.label}.{_level_name(v)}" for v in self._levels(network)[0]]

    def spec(self, network):
        return (self.rust_name, [], self._levels(network)[1])


class NodeIFactor(NodeFactor):
    directed = True


class NodeOFactor(NodeFactor):
    directed = True


class NodeCov(_AttributeTerm):
    def spec(self, network):
        return (self.rust_name, [float(v) for v in network.attribute(self.attr)], [])


class NodeICov(NodeCov):
    directed = True


class NodeOCov(NodeCov):
    directed = True


class AbsDiff(_AttributeTerm):
    def spec(self, network):
        return ("absdiff", [float(v) for v in network.attribute(self.attr)], [])


class EdgeCov(Term):
    """A dyadic covariate: a graph attribute holding an n x n matrix, or the matrix."""

    def __init__(self, x):
        self.x = x

    @property
    def label(self) -> str:
        return f"edgecov.{self.x}" if isinstance(self.x, str) else "edgecov"

    def matrix(self, network: Network) -> np.ndarray:
        x = network.graph_attribute(self.x) if isinstance(self.x, str) else self.x
        if hasattr(x, "get_adjacency"):  # an igraph graph, as ergm accepts a network
            x = x.get_adjacency().data
        x = np.asarray(x, dtype=float)
        if x.shape != (network.n, network.n):
            raise ValueError(f"edgecov: expected a {network.n} x {network.n} matrix, got shape {x.shape}")
        return x

    def spec(self, network):
        return ("edgecov", self.matrix(network).ravel().tolist(), [])

    def __repr__(self) -> str:
        return f"edgecov({self.x!r})" if isinstance(self.x, str) else "edgecov(<matrix>)"


# -- The functions users call, named as in ergm -------------------------------------------------


def edges() -> Term:
    """Number of edges."""
    return Edges()


def mutual() -> Term:
    """Number of reciprocated pairs of ties (directed networks)."""
    return Mutual()


def kstar(k) -> Term:
    """Number of k-stars, for one or more k (undirected networks)."""
    return KStar(k)


def istar(k) -> Term:
    """Number of in-k-stars: sets of k ties to the same vertex (directed networks)."""
    return IStar(k)


def ostar(k) -> Term:
    """Number of out-k-stars: sets of k ties from the same vertex (directed networks)."""
    return OStar(k)


def gwdegree(decay: float, fixed: bool = False) -> Term:
    """Geometrically weighted degree distribution (undirected networks).

    Only a fixed decay is supported (``fixed=True``).
    """
    _require_fixed("gwdegree", fixed)
    return GwDegree(decay)


def gwidegree(decay: float, fixed: bool = False) -> Term:
    """Geometrically weighted in-degree distribution (directed networks)."""
    _require_fixed("gwidegree", fixed)
    return GwIDegree(decay)


def gwodegree(decay: float, fixed: bool = False) -> Term:
    """Geometrically weighted out-degree distribution (directed networks)."""
    _require_fixed("gwodegree", fixed)
    return GwODegree(decay)


def triangle() -> Term:
    """Number of triangles; in directed networks, transitive plus cyclic triples."""
    return Triangle()


def ttriple() -> Term:
    """Number of transitive triples i -> j -> k with i -> k (directed networks)."""
    return TTriple()


def ctriple() -> Term:
    """Number of cyclic triples i -> j -> k -> i (directed networks)."""
    return CTriple()


def gwesp(decay: float, fixed: bool = False) -> Term:
    """Geometrically weighted edgewise shared partners.

    In directed networks, shared partners are outgoing two-paths (ergm's
    default ``type = "OTP"``): k is a shared partner of i -> j if i -> k -> j.
    Only a fixed decay is supported (``fixed=True``). As in ergm,
    ``fixed=False`` would estimate the decay (a curved ERGM).
    """
    _require_fixed("gwesp", fixed)
    return Gwesp(decay)


def gwdsp(decay: float, fixed: bool = False) -> Term:
    """Geometrically weighted dyadwise shared partners (undirected networks)."""
    _require_fixed("gwdsp", fixed)
    return Gwdsp(decay)


def nodematch(attr: str, diff: bool = False) -> Term:
    """Number of ties between vertices with the same value of ``attr``; with
    ``diff=True``, one statistic per value."""
    return NodeMatch(attr, diff)


def nodefactor(attr: str) -> Term:
    """Number of tie endpoints at each level of ``attr`` but the first."""
    return NodeFactor(attr)


def nodeifactor(attr: str) -> Term:
    """Number of ties received by vertices at each level of ``attr`` but the first."""
    return NodeIFactor(attr)


def nodeofactor(attr: str) -> Term:
    """Number of ties sent by vertices at each level of ``attr`` but the first."""
    return NodeOFactor(attr)


def nodecov(attr: str) -> Term:
    """Sum over ties of the endpoints' values of the numeric ``attr``."""
    return NodeCov(attr)


def nodeicov(attr: str) -> Term:
    """Sum over ties of the receiver's value of the numeric ``attr`` (directed networks)."""
    return NodeICov(attr)


def nodeocov(attr: str) -> Term:
    """Sum over ties of the sender's value of the numeric ``attr`` (directed networks)."""
    return NodeOCov(attr)


def absdiff(attr: str) -> Term:
    """Sum over ties of the absolute difference in the numeric ``attr``."""
    return AbsDiff(attr)


def edgecov(x) -> Term:
    """Sum over ties of a dyadic covariate.

    ``x`` is the name of a graph attribute holding an n x n matrix (in a
    formula string: ``"edgecov('trade')"``), an n x n array, or an igraph
    graph on the same vertices. Undirected networks use the upper triangle.
    """
    return EdgeCov(x)


TERMS = {
    f.__name__: f
    for f in (
        edges, mutual, kstar, istar, ostar, gwdegree, gwidegree, gwodegree, triangle, ttriple,
        ctriple, gwesp, gwdsp, nodematch, nodefactor, nodeifactor, nodeofactor, nodecov,
        nodeicov, nodeocov, absdiff, edgecov,
    )
}
