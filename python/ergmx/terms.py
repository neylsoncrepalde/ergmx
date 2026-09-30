"""Model terms, with the same names and statistics as the R package ergm.

Terms can be combined with ``+`` or written as a formula string::

    edges() + nodematch("Grade") + gwesp(0.5, fixed=True)
    "edges + nodematch('Grade') + gwesp(0.5, fixed=True)"
"""

from __future__ import annotations

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


class Edges(Term):
    pass


class Mutual(Term):
    dyad_independent = False
    directed = True


class Triangle(Term):
    dyad_independent = False
    triadic = True
    directed = False


class Gwesp(Term):
    dyad_independent = False
    triadic = True
    directed = False

    def __init__(self, decay: float):
        self.decay = float(decay)

    @property
    def label(self) -> str:
        return f"gwesp.fixed.{self.decay:g}"

    def spec(self, network):
        return ("gwesp", [self.decay], [])

    def __repr__(self) -> str:
        return f"gwesp({self.decay:g}, fixed=True)"


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
    def spec(self, network):
        return ("nodematch", [], _codes(network.attribute(self.attr))[1])


class NodeFactor(_AttributeTerm):
    """One statistic per attribute level except the first (sorted) one, as in
    ergm's default ``levels = -1``."""

    def _levels(self, network):
        levels, codes = _codes(network.attribute(self.attr))
        return levels[1:], [c - 1 for c in codes]

    def names(self, network):
        return [f"{self.label}.{_level_name(v)}" for v in self._levels(network)[0]]

    def spec(self, network):
        return ("nodefactor", [], self._levels(network)[1])


class NodeCov(_AttributeTerm):
    def spec(self, network):
        return ("nodecov", [float(v) for v in network.attribute(self.attr)], [])


class AbsDiff(_AttributeTerm):
    def spec(self, network):
        return ("absdiff", [float(v) for v in network.attribute(self.attr)], [])


# -- The functions users call, named as in ergm ---------------------------------


def edges() -> Term:
    """Number of edges."""
    return Edges()


def mutual() -> Term:
    """Number of reciprocated pairs of ties (directed networks)."""
    return Mutual()


def triangle() -> Term:
    """Number of triangles (undirected networks)."""
    return Triangle()


def gwesp(decay: float, fixed: bool = False) -> Term:
    """Geometrically weighted edgewise shared partners (undirected networks).

    Only a fixed decay is supported (``fixed=True``). As in ergm,
    ``fixed=False`` would estimate the decay (a curved ERGM).
    """
    if not fixed:
        raise NotImplementedError(
            "gwesp with an estimated decay (a curved ERGM) is not supported yet; "
            "use gwesp(decay, fixed=True)"
        )
    return Gwesp(decay)


def nodematch(attr: str, diff: bool = False) -> Term:
    """Number of ties between vertices with the same value of ``attr``."""
    if diff:
        raise NotImplementedError("nodematch(diff=True) is not supported yet")
    return NodeMatch(attr)


def nodefactor(attr: str) -> Term:
    """Number of tie endpoints at each level of ``attr`` but the first."""
    return NodeFactor(attr)


def nodecov(attr: str) -> Term:
    """Sum over ties of the endpoints' values of the numeric ``attr``."""
    return NodeCov(attr)


def absdiff(attr: str) -> Term:
    """Sum over ties of the absolute difference in the numeric ``attr``."""
    return AbsDiff(attr)


TERMS = {
    f.__name__: f
    for f in (edges, mutual, triangle, gwesp, nodematch, nodefactor, nodecov, absdiff)
}
