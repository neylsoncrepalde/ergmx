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
    #: The degrees the statistic is a function of ("in", "out"; both for
    #: degrees of undirected networks), so that it is constant when they are
    #: preserved; None if it is not a function of degrees.
    degree_dependence: frozenset | None = None
    #: Whether the coefficient is fixed by offset() rather than estimated.
    is_offset = False

    def names(self, network: Network) -> list[str]:
        return [self.label]

    def spec(self, network: Network) -> tuple[str, list[float], list[int]]:
        """The term as the Rust core expects it: (name, real params, integer params)."""
        return (self.rust_name, [], [])

    def full_spec(self, network: Network) -> tuple:
        """The spec with the terms an operator applies to (none for plain terms)."""
        return (*self.spec(network), [])

    def r_call(self) -> str:
        """The term as R deparses it, for the names of F() terms."""
        call = getattr(self, "_call", None)
        if call is None:
            return self.rust_name
        name, args, kwargs = call
        parts = [_r_literal(a) for a in args] + [f"{k} = {_r_literal(v)}" for k, v in kwargs.items()]
        return f"{name}({', '.join(parts)})" if parts or name not in _NO_PARENS else name

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


_DEGREES = frozenset({"in", "out"})
_IN, _OUT = frozenset({"in"}), frozenset({"out"})
# Terms R deparses without parentheses when they have no arguments (~edges).
_NO_PARENS = {"edges", "mutual", "triangle", "ttriple", "ctriple", "isolates"}


def _r_literal(value) -> str:
    """A Python value as R deparses it."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None:
        return "NA"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return format(value, ".15g")
    if isinstance(value, (list, tuple)):
        return "c(" + ", ".join(_r_literal(v) for v in value) + ")"
    return repr(value)


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
    degree_dependence = frozenset()  # constant whenever any degrees are preserved


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
    degree_dependence = _DEGREES


class IStar(_Stars):
    directed = True
    degree_dependence = _IN


class OStar(_Stars):
    directed = True
    degree_dependence = _OUT


class _DegreeCount(_Stars):
    """Number of vertices with each given degree."""

    def __init__(self, k):
        ks = [k] if isinstance(k, int) else list(k)
        if not ks or not all(isinstance(v, int) and v >= 0 for v in ks):
            raise ValueError(f"{self.rust_name}: d must be one or more integers >= 0, not {k!r}")
        self.ks = ks


class Degree(_DegreeCount):
    directed = False
    degree_dependence = _DEGREES


class IDegree(_DegreeCount):
    directed = True
    degree_dependence = _IN


class ODegree(_DegreeCount):
    directed = True
    degree_dependence = _OUT


class Isolates(Term):
    dyad_independent = False
    degree_dependence = _DEGREES


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
    degree_dependence = _DEGREES


class GwIDegree(_Decay):
    directed = True
    stat_name = "gwideg"
    degree_dependence = _IN


class GwODegree(_Decay):
    directed = True
    stat_name = "gwodeg"
    degree_dependence = _OUT


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
    """gwdsp; directed networks use outgoing two-paths (OTP) over ordered pairs."""

    triadic = True

    def names(self, network):
        kind = ".OTP" if network.directed else ""
        return [f"gwdsp{kind}.fixed.{self.decay:g}"]


class _SharedPartners(_Stars):
    """Number of ties (esp) or pairs (dsp) with each given number of shared
    partners; OTP shared partners if directed."""

    triadic = True

    def __init__(self, k):
        ks = [k] if isinstance(k, int) else list(k)
        if not ks or not all(isinstance(v, int) and v >= 0 for v in ks):
            raise ValueError(f"{self.rust_name}: d must be one or more integers >= 0, not {k!r}")
        self.ks = ks

    def names(self, network):
        kind = ".OTP" if network.directed else ""
        return [f"{self.rust_name}{kind}{k}" for k in self.ks]


class Esp(_SharedPartners):
    pass


class Dsp(_SharedPartners):
    pass


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
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"  # as R prints logical levels
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

    degree_dependence = _DEGREES

    def _levels(self, network):
        levels, codes = _codes(network.attribute(self.attr))
        return levels[1:], [c - 1 for c in codes]

    def names(self, network):
        return [f"{self.label}.{_level_name(v)}" for v in self._levels(network)[0]]

    def spec(self, network):
        return (self.rust_name, [], self._levels(network)[1])


class NodeIFactor(NodeFactor):
    directed = True
    degree_dependence = _IN


class NodeOFactor(NodeFactor):
    directed = True
    degree_dependence = _OUT


class NodeCov(_AttributeTerm):
    degree_dependence = _DEGREES

    def spec(self, network):
        return (self.rust_name, [float(v) for v in network.attribute(self.attr)], [])


class NodeICov(NodeCov):
    directed = True
    degree_dependence = _IN


class NodeOCov(NodeCov):
    directed = True
    degree_dependence = _OUT


def _select(spec, size: int, what: str) -> list[int]:
    """Positions (0-based) selected by an R-style index specification: TRUE
    (all), FALSE (none), 1-based indices, or negative indices to leave out."""
    if spec is True:
        return list(range(size))
    if spec is False:
        return []
    indices = [spec] if isinstance(spec, int) else list(spec)
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
        raise ValueError(f"{what} must be TRUE, FALSE or integer indices, not {spec!r}")
    if all(i < 0 for i in indices):
        dropped = {-i - 1 for i in indices}
        return [i for i in range(size) if i not in dropped]
    if any(i < 1 or i > size for i in indices):
        raise ValueError(f"{what}: indices must be between 1 and {size} (or all negative), not {spec!r}")
    return [i - 1 for i in indices]


def mixing_types(network: Network, attr: str, levels=None, levels2=-1):
    """The levels of `attr` and the mixing types (pairs of levels) that
    nodemix(attr, levels, levels2) counts, as in ergm.

    Returns the selected levels, each vertex's position among them (-1 if
    its level is left out), and the selected types as (row, column) pairs of
    level positions, in ergm's order: pairs with row <= column column by column
    if undirected, all pairs column by column (row = sender) if directed.
    """
    values = network.attribute(attr)
    every = sorted(set(values))
    if levels is None:
        chosen = every
    elif isinstance(levels, (list, tuple)) and levels and all(isinstance(v, str) for v in levels):
        missing = [v for v in levels if v not in every]
        if missing:
            raise ValueError(f"nodemix: {attr!r} has no levels {missing}; it has {every}")
        chosen = list(levels)
    elif isinstance(levels, str):
        chosen = [levels]
    else:
        chosen = [every[i] for i in _select(levels, len(every), "levels")]
    position = {v: k for k, v in enumerate(chosen)}
    codes = [position.get(v, -1) for v in values]
    size = len(chosen)
    if network.directed:
        types = [(row, col) for col in range(size) for row in range(size)]
    else:
        types = [(row, col) for col in range(size) for row in range(col + 1)]
    if isinstance(levels2, (list, tuple)) and levels2 and isinstance(levels2[0], (list, tuple)):
        matrix = np.asarray(levels2, dtype=bool)
        if matrix.shape != (size, size):
            raise ValueError(f"levels2: a mixing matrix must be {size} x {size}, not {matrix.shape}")
        selected = [t for t in types if matrix[t]]
    else:
        selected = [types[i] for i in _select(levels2, len(types), "levels2")]
    return chosen, codes, selected


class NodeMix(_AttributeTerm):
    """Ties by mixing type, as in ergm's nodemix."""

    def __init__(self, attr: str, levels=None, levels2=-1):
        super().__init__(attr)
        self.levels, self.levels2 = levels, levels2

    def names(self, network):
        chosen, _, types = mixing_types(network, self.attr, self.levels, self.levels2)
        return [f"mix.{self.attr}.{_level_name(chosen[r])}.{_level_name(chosen[c])}" for r, c in types]

    def spec(self, network):
        chosen, codes, types = mixing_types(network, self.attr, self.levels, self.levels2)
        size = len(chosen)
        mapping = np.full((size, size), -1, dtype=np.int64)
        for stat, (row, col) in enumerate(types):
            mapping[row, col] = stat
            if not network.directed:
                mapping[col, row] = stat
        return ("nodemix", [], [*codes, size, *mapping.ravel().tolist()])

    def __repr__(self) -> str:
        args = [repr(self.attr)]
        if self.levels is not None:
            args.append(f"levels={self.levels!r}")
        if self.levels2 != -1:
            args.append(f"levels2={self.levels2!r}")
        return f"nodemix({', '.join(args)})"


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


# -- Operators ------------------------------------------------------------------------------------


class Offset(Term):
    """A term whose coefficient is fixed rather than estimated, as ergm's offset()."""

    is_offset = True

    def __init__(self, term: Term):
        if not isinstance(term, Term) or isinstance(term, Offset):
            raise ValueError(f"offset() takes one term, not {term!r}")
        self.term = term

    dyad_independent = property(lambda self: self.term.dyad_independent)
    triadic = property(lambda self: self.term.triadic)
    directed = property(lambda self: self.term.directed)
    degree_dependence = property(lambda self: self.term.degree_dependence)

    def names(self, network):
        return [f"offset({name})" for name in self.term.names(network)]

    def spec(self, network):
        return self.term.spec(network)

    def full_spec(self, network):
        return self.term.full_spec(network)

    def check(self, network):
        self.term.check(network)

    def __repr__(self) -> str:
        return f"offset({self.term!r})"


class Filtered(Term):
    """Terms evaluated on the network of the ties that pass a filter, as ergm's F()."""

    degree_dependence = None

    def __init__(self, formula, filter, negate: bool = False):
        self.formula = as_formula(formula)
        filters = as_formula(filter)
        if len(filters) != 1:
            raise ValueError("F(): the filter must be a single term")
        self.filter, self.negate = filters.terms[0], bool(negate)
        for term in self.formula:
            if isinstance(term, (Filtered, Offset)):
                raise ValueError(f"F(): {term!r} can't be filtered; put F() inside offset() instead")
        if not self.filter.dyad_independent or self.filter.is_offset:
            raise ValueError(f"F(): the filter {self.filter!r} must be a dyad-independent term")

    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))
    triadic = property(lambda self: any(t.triadic for t in self.formula))

    def _filter_label(self) -> str:
        return ("!" if self.negate else "") + self.filter.r_call()

    def names(self, network):
        label = self._filter_label()
        return [f"F({label})~{name}" for term in self.formula for name in term.names(network)]

    def check(self, network):
        for term in [*self.formula, self.filter]:
            term.check(network)
        if len(self.filter.names(network)) != 1:
            raise ValueError(f"F(): the filter {self.filter!r} must have exactly one statistic")

    def full_spec(self, network):
        children = [t.full_spec(network) for t in self.formula] + [self.filter.full_spec(network)]
        return ("F", [], [int(self.negate)], children)

    def spec(self, network):
        raise TypeError("F() has no flat spec; use full_spec")

    def __repr__(self) -> str:
        negate = ", negate=True" if self.negate else ""
        return f"F({self.formula!r}, {self.filter!r}{negate})"


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


def degree(d, by=None) -> Term:
    """Number of vertices with degree exactly d, for one or more d (undirected networks)."""
    if by is not None:
        raise NotImplementedError("degree(d, by=...) is not supported yet")
    return Degree(d)


def idegree(d) -> Term:
    """Number of vertices with in-degree exactly d, for one or more d (directed networks)."""
    return IDegree(d)


def odegree(d) -> Term:
    """Number of vertices with out-degree exactly d, for one or more d (directed networks)."""
    return ODegree(d)


def isolates() -> Term:
    """Number of vertices without ties (in either direction, if directed)."""
    return Isolates()


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
    """Geometrically weighted dyadwise shared partners: as gwesp, over all pairs
    of vertices, tied or not (outgoing two-paths and ordered pairs if directed)."""
    _require_fixed("gwdsp", fixed)
    return Gwdsp(decay)


def esp(d) -> Term:
    """Number of ties with exactly d edgewise shared partners, for one or more d
    (outgoing two-paths if directed)."""
    return Esp(d)


def dsp(d) -> Term:
    """Number of pairs of vertices with exactly d shared partners, for one or more
    d (outgoing two-paths and ordered pairs if directed)."""
    return Dsp(d)


def nodematch(attr: str, diff: bool = False) -> Term:
    """Number of ties between vertices with the same value of ``attr``; with
    ``diff=True``, one statistic per value."""
    return NodeMatch(attr, diff)


def nodemix(attr: str, levels=None, levels2=-1) -> Term:
    """Number of ties for each mixing type: each pair of levels of ``attr``
    (from sender to receiver, if directed).

    ``levels`` selects levels: names, or R-style 1-based indices into the
    sorted levels (negative to leave out). ``levels2`` selects mixing types, in
    ergm's order: ``-1`` (the default) all but the first, ``TRUE`` all, indices,
    or a levels x levels logical matrix (rows: senders).
    """
    return NodeMix(attr, levels, levels2)


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


def offset(term) -> Term:
    """Fix a term's coefficients, as ergm's offset(). Give their values with
    ``ergm(..., offset_coef=[...])``; ``-inf`` forbids the ties the term counts."""
    if isinstance(term, str):
        term = as_formula(term)
    if isinstance(term, Formula):
        if len(term) != 1:
            raise ValueError("offset() takes a single term")
        term = term.terms[0]
    return Offset(term)


def F(formula, filter, negate: bool = False) -> Term:
    """Evaluate ``formula`` on the network of the ties that pass ``filter``, as ergm's F().

    ``filter`` is a dyad-independent term with one statistic: a tie passes if
    adding it changes that statistic (does not, with ``negate=True``). In a
    formula string: ``"F(~gwesp(0.5, fixed=TRUE), ~nodematch('level'))"``, with
    ``~!nodematch('level')`` to negate.
    """
    return Filtered(formula, filter, negate)


def edgecov(x) -> Term:
    """Sum over ties of a dyadic covariate.

    ``x`` is the name of a graph attribute holding an n x n matrix (in a
    formula string: ``"edgecov('trade')"``), an n x n array, or an igraph
    graph on the same vertices. Undirected networks use the upper triangle.
    """
    return EdgeCov(x)


def _recording(factory):
    """Wrap a term function so its terms remember how they were called."""

    def make(*args, **kwargs):
        term = factory(*args, **kwargs)
        term._call = (factory.__name__, args, kwargs)
        return term

    make.__name__, make.__doc__, make.__wrapped__ = factory.__name__, factory.__doc__, factory
    return make


_PLAIN = (
    edges, mutual, kstar, istar, ostar, degree, idegree, odegree, isolates, gwdegree, gwidegree,
    gwodegree, triangle, ttriple, ctriple, gwesp, gwdsp, esp, dsp, nodematch, nodemix, nodefactor,
    nodeifactor, nodeofactor, nodecov, nodeicov, nodeocov, absdiff, edgecov,
)
for _f in _PLAIN:
    globals()[_f.__name__] = _recording(_f)
del _f

#: Every term function by name; F and offset are operators.
TERMS = {name: globals()[name] for name in [f.__name__ for f in _PLAIN] + ["F", "offset"]}
