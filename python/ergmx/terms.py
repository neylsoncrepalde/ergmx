"""Model terms, with the same names and statistics as the R package ergm.

Terms can be combined with ``+`` or written as a formula string::

    edges() + nodematch("Grade") + gwesp(0.5, fixed=True)
    "edges + nodematch('Grade') + gwesp(0.5, fixed=True)"
"""

from __future__ import annotations

import inspect
import warnings

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
    #: Whether the term is curved: its statistics' coefficients are a nonlinear
    #: function of fewer parameters (gwesp with an estimated decay).
    curved = False

    def names(self, network: Network) -> list[str]:
        """Names of the term's statistics."""
        return [self.label]

    def param_names(self, network: Network) -> list[str]:
        """Names of the term's parameters: its statistics', unless curved."""
        return self.names(network)

    def eta(self, params: np.ndarray, network: Network) -> np.ndarray:
        """Coefficients of the statistics from the parameters (the same, unless curved)."""
        return np.asarray(params, dtype=float)

    def jacobian(self, params: np.ndarray, network: Network) -> np.ndarray:
        """Derivatives of eta with respect to the parameters (statistics x parameters)."""
        return np.eye(len(params))

    def starts(self, network: Network) -> list[tuple[int, float]]:
        """Starting values of the parameters a fit can't start at 0 (the decays
        of curved terms), as (position among the term's parameters, value)."""
        return []

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


def _chunks(*parts) -> list[int]:
    """Integer parameters for the Rust core: each part as its length, then its values."""
    out: list[int] = []
    for part in parts:
        values = [int(v) for v in part]
        out += [len(values), *values]
    return out


#: Codes of the types of shared partner of directed networks, as in ergm.
SP_TYPES = {"OTP": 1, "ITP": 2, "RTP": 3, "OSP": 4, "ISP": 5}


def _sp_type(type: str) -> str:
    kind = str(type).upper()
    if kind not in SP_TYPES:
        raise ValueError(f"type must be one of {', '.join(SP_TYPES)}, not {type!r}")
    return kind


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


def _curved_or_fixed(term, fixed: bool, cutoff: int):
    """The fixed-decay term, or its curved version if the decay is estimated."""
    return term if fixed else Curved(term, cutoff)


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

    def mask(self, network) -> list[int]:
        """Vertices counted: all (empty), or those of one bipartite mode."""
        return []

    def spec(self, network):
        return (self.rust_name, [], _chunks(self.ks, self.mask(network)))

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

    def spec(self, network):
        return (self.rust_name, [], _chunks(self.ks, [0], self.mask(network)))


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


class Concurrent(Term):
    """Number of vertices with degree 2 or more."""

    dyad_independent = False
    directed = False
    degree_dependence = _DEGREES

    def mask(self, network) -> list[int]:
        return []

    def spec(self, network):
        return ("concurrent", [], _chunks(self.mask(network)))


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

    def mask(self, network) -> list[int]:
        return []

    def spec(self, network):
        return (self.rust_name, [self.decay], _chunks(self.mask(network)))

    def __repr__(self) -> str:
        return f"{self.rust_name}({self.decay:g}, fixed=True)"


class _GwDegreeBase(_Decay):
    """What a geometrically weighted degree term needs to be curved."""

    #: The Rust degree histogram and ergm's name of the curved term.
    histogram_rust = "degree"
    curved_label = "gwdegree"

    def largest_count(self, network) -> int:
        return network.n - 1

    def histogram_name(self, network) -> str:
        return self.curved_label

    def curved_name(self, network) -> str:
        return self.curved_label

    def histogram_spec(self, network, ks, overflow):
        return (self.histogram_rust, [], _chunks(ks, [int(overflow)], self.mask(network)))


class GwDegree(_GwDegreeBase):
    directed = False
    stat_name = "gwdeg"
    degree_dependence = _DEGREES


class GwIDegree(_GwDegreeBase):
    directed = True
    stat_name = "gwideg"
    degree_dependence = _IN
    histogram_rust = "idegree"
    curved_label = "gwidegree"


class GwODegree(_GwDegreeBase):
    directed = True
    stat_name = "gwodeg"
    degree_dependence = _OUT
    histogram_rust = "odegree"
    curved_label = "gwodegree"


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


class _GwSharedPartners(_Decay):
    """gwesp, gwdsp and gwnsp; directed networks use the shared partners of
    `type`, ergm's outgoing two-paths (OTP) by default."""

    triadic = True

    def __init__(self, decay: float, type: str = "OTP"):
        super().__init__(decay)
        self.type = _sp_type(type)

    def names(self, network):
        kind = f".{self.type}" if network.directed else ""
        return [f"{self.rust_name}{kind}.fixed.{self.decay:g}"]

    def spec(self, network):
        return (self.rust_name, [self.decay], _chunks([SP_TYPES[self.type]], self.mask(network)))

    # Curved versions: the histogram of esp, dsp or nsp.
    def largest_count(self, network) -> int:
        return network.n - 2

    def histogram_name(self, network) -> str:
        base = self.rust_name[2:]  # esp, dsp, nsp
        return f"{base}.{self.type}" if network.directed else base

    def curved_name(self, network) -> str:
        return f"{self.rust_name}.{self.type}" if network.directed else self.rust_name

    def histogram_spec(self, network, ks, overflow):
        return (self.rust_name[2:], [], _chunks([SP_TYPES[self.type]], ks, [int(overflow)],
                                                self.mask(network)))

    def __repr__(self) -> str:
        kind = f", type={self.type!r}" if self.type != "OTP" else ""
        return f"{self.rust_name}({self.decay:g}, fixed=True{kind})"


class Gwesp(_GwSharedPartners):
    pass


class Gwdsp(_GwSharedPartners):
    pass


class Gwnsp(_GwSharedPartners):
    pass


class _SharedPartners(_Stars):
    """Number of ties (esp), pairs (dsp) or pairs without a tie (nsp) with
    each given number of shared partners, of `type` if directed."""

    triadic = True

    def __init__(self, k, type: str = "OTP"):
        ks = [k] if isinstance(k, int) else list(k)
        if not ks or not all(isinstance(v, int) and v >= 0 for v in ks):
            raise ValueError(f"{self.rust_name}: d must be one or more integers >= 0, not {k!r}")
        self.ks, self.type = ks, _sp_type(type)

    def names(self, network):
        kind = f".{self.type}" if network.directed else ""
        return [f"{self.rust_name}{kind}{k}" for k in self.ks]

    def spec(self, network):
        return (self.rust_name, [], _chunks([SP_TYPES[self.type]], self.ks, [0], self.mask(network)))


class Esp(_SharedPartners):
    pass


class Dsp(_SharedPartners):
    pass


class Nsp(_SharedPartners):
    pass


class Cycle(_Stars):
    """Number of cycles of each length k (k >= 3; k >= 2 if directed)."""

    triadic = True

    def __init__(self, k, semi: bool = False):
        if semi:
            raise NotImplementedError("cycle(k, semi=TRUE) is not supported yet")
        ks = [k] if isinstance(k, int) else list(k)
        if not ks or not all(isinstance(v, int) and v >= 2 for v in ks):
            raise ValueError(f"cycle: k must be one or more integers >= 2, not {k!r}")
        self.ks = ks

    def check(self, network):
        if not network.directed and min(self.ks) < 3:
            raise ValueError("cycle: k must be 3 or more in undirected networks")

    def spec(self, network):
        return ("cycle", [], self.ks)


class ErgmDifferenceWarning(UserWarning):
    """A term whose statistic differs from what R's ergm computes for it,
    because ergmx follows ergm's documented definition."""


class Transitive(Term):
    """Transitive triads: Davis and Leinhardt's types 030T, 120D, 120U and 300."""

    dyad_independent = False
    triadic = True
    directed = True

    def __init__(self):
        warnings.warn(
            "transitive counts transitive triads (types 030T, 120D, 120U and 300), as R's "
            "ergm documents; ergm 4.12 computes transitive triples instead, the same as "
            "ttriple. To reproduce ergm's results, use ttriple.",
            ErgmDifferenceWarning, stacklevel=4,
        )


class TwoPath(Term):
    dyad_independent = False

    @property
    def degree_dependence(self):
        return None  # a function of the degrees only if undirected, where it is kstar(2)

    def spec(self, network):
        if network.directed:
            return ("twopath", [], [])
        return ("kstar", [], _chunks([2], []))


class Asymmetric(Term):
    dyad_independent = False
    directed = True


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


class _VertexEffects(Term):
    """One statistic per vertex but the first: its out-degree (sender),
    in-degree (receiver) or degree (sociality), as in ergm."""

    #: The Rust term counting the ends of ties by level.
    rust_term = ""

    def names(self, network):
        return [f"{self.rust_name}{v + 1}" for v in range(1, network.n)]

    def spec(self, network):
        return (self.rust_term, [], [v - 1 for v in range(network.n)])  # vertex 1 is the base


class Sender(_VertexEffects):
    directed = True
    degree_dependence = _OUT
    rust_term = "nodeofactor"


class Receiver(_VertexEffects):
    directed = True
    degree_dependence = _IN
    rust_term = "nodeifactor"


class Sociality(_VertexEffects):
    directed = False
    degree_dependence = _DEGREES
    rust_term = "nodefactor"


class AbsDiffCat(_AttributeTerm):
    """For each distinct nonzero absolute difference of a numeric attribute,
    the number of ties with that difference."""

    rust_name = "absdiffcat"

    def _differences(self, network) -> list[float]:
        x = np.array([float(v) for v in network.attribute(self.attr)])
        return sorted({float(d) for d in np.abs(x[:, None] - x[None, :]).ravel() if d != 0})

    def names(self, network):
        return [f"absdiff.{self.attr}.{_level_name(d)}" for d in self._differences(network)]

    def spec(self, network):
        x = [float(v) for v in network.attribute(self.attr)]
        return ("absdiffcat", x + self._differences(network), [])


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
        if network.bipartite and x.shape != (network.n, network.n):
            # ergm's bipartite edgecov: first-mode vertices by second-mode vertices.
            first, second = np.flatnonzero(network.mode == 1), np.flatnonzero(network.mode == 2)
            if x.shape == (len(first), len(second)):
                full = np.zeros((network.n, network.n))
                full[np.ix_(first, second)] = x
                x = full + full.T
        if x.shape != (network.n, network.n):
            raise ValueError(f"edgecov: expected a {network.n} x {network.n} matrix, got shape {x.shape}")
        return x

    def spec(self, network):
        return ("edgecov", self.matrix(network).ravel().tolist(), [])

    def __repr__(self) -> str:
        return f"edgecov({self.x!r})" if isinstance(self.x, str) else "edgecov(<matrix>)"


# -- Bipartite terms ---------------------------------------------------------------------------


class _Mode:
    """A term about one mode of a bipartite network: 1 (ergm's b1) or 2."""

    mode_number = 1

    @property
    def b(self) -> str:
        return f"b{self.mode_number}"

    def check(self, network):
        if not network.bipartite:
            raise ValueError(f"{self!r} needs a bipartite network: pass bipartite=<attribute> "
                             "with each vertex's mode")
        super().check(network)

    def mask(self, network) -> list[int]:
        return (network.mode == self.mode_number).astype(int).tolist()

    def __repr__(self) -> str:
        call = getattr(self, "_call", None)
        if call is None:
            return f"{type(self).__name__.lower()}(...)"
        name, args, kwargs = call
        parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        return f"{name}({', '.join(parts)})"


class _BStar(_Mode, KStar):
    rust_name = "kstar"

    def names(self, network):
        return [f"{self.b}star{k}" for k in self.ks]


class _BDegree(_Mode, Degree):
    rust_name = "degree"

    def names(self, network):
        return [f"{self.b}degree{k}" for k in self.ks]


class _GwBDegree(_Mode, GwDegree):
    rust_name = "gwdegree"

    def names(self, network):
        return [f"gw{self.b}deg.fixed.{self.decay:g}"]

    def largest_count(self, network) -> int:
        return int(np.sum(network.mode != self.mode_number))

    def histogram_name(self, network) -> str:
        return f"gw{self.b}degree"

    curved_name = histogram_name


class _BConcurrent(_Mode, Concurrent):
    def names(self, network):
        return [f"{self.b}concurrent"]


class _BDsp(_Mode, Dsp):
    rust_name = "dsp"

    def names(self, network):
        return [f"{self.b}dsp{k}" for k in self.ks]

    def spec(self, network):
        return ("dsp", [], _chunks([0], self.ks, [0], self.mask(network)))


class _GwBDsp(_Mode, Gwdsp):
    rust_name = "gwdsp"

    def names(self, network):
        return [f"gw{self.b}dsp.fixed.{self.decay:g}"]

    def spec(self, network):
        return ("gwdsp", [self.decay], _chunks([0], self.mask(network)))

    def largest_count(self, network) -> int:
        return int(np.sum(network.mode != self.mode_number))

    def histogram_name(self, network) -> str:
        return f"{self.b}dsp"

    def curved_name(self, network) -> str:
        return f"gw{self.b}dsp"

    def histogram_spec(self, network, ks, overflow):
        return ("dsp", [], _chunks([0], ks, [int(overflow)], self.mask(network)))


class _BFactor(_Mode, NodeFactor):
    """nodefactor counting only the endpoints in one mode."""

    def _levels(self, network):
        values = network.attribute(self.attr)
        mine = [v for v, m in zip(values, network.mode) if m == self.mode_number]
        levels = sorted(set(mine))[1:]
        position = {v: k for k, v in enumerate(levels)}
        codes = [position.get(v, -1) if m == self.mode_number else -1
                 for v, m in zip(values, network.mode)]
        return levels, codes

    def names(self, network):
        return [f"{self.b}factor.{self.attr}.{_level_name(v)}" for v in self._levels(network)[0]]

    def spec(self, network):
        return ("nodefactor", [], self._levels(network)[1])


class _BCov(_Mode, NodeCov):
    """nodecov counting only the endpoints in one mode."""

    @property
    def label(self) -> str:
        return f"{self.b}cov.{self.attr}"

    def spec(self, network):
        x = [float(v) if m == self.mode_number else 0.0
             for v, m in zip(network.attribute(self.attr), network.mode)]
        return ("nodecov", x, [])


class _BNodeMatch(_Mode, _AttributeTerm):
    """Number of 2-stars centred on the other mode whose two ends, in this
    mode, have the same value of `attr` (ergm's default alpha = beta = 1)."""

    dyad_independent = False

    @property
    def label(self) -> str:
        return f"{self.b}nodematch.{self.attr}"

    def spec(self, network):
        values = network.attribute(self.attr)
        levels, _ = _codes([v for v, m in zip(values, network.mode) if m == self.mode_number])
        position = {v: k for k, v in enumerate(levels)}
        return ("bipartitematch", [], [position[v] if m == self.mode_number else -1
                                       for v, m in zip(values, network.mode)])


def _in_mode(base, number: int, name: str):
    """The class of `base` for bipartite mode `number`, named as ergm's term."""
    return type(f"{name}", (base,), {"mode_number": number, "__doc__": base.__doc__})


B1Star, B2Star = _in_mode(_BStar, 1, "B1Star"), _in_mode(_BStar, 2, "B2Star")
B1Degree, B2Degree = _in_mode(_BDegree, 1, "B1Degree"), _in_mode(_BDegree, 2, "B2Degree")
GwB1Degree, GwB2Degree = _in_mode(_GwBDegree, 1, "GwB1Degree"), _in_mode(_GwBDegree, 2, "GwB2Degree")
B1Concurrent = _in_mode(_BConcurrent, 1, "B1Concurrent")
B2Concurrent = _in_mode(_BConcurrent, 2, "B2Concurrent")
B1Dsp, B2Dsp = _in_mode(_BDsp, 1, "B1Dsp"), _in_mode(_BDsp, 2, "B2Dsp")
GwB1Dsp, GwB2Dsp = _in_mode(_GwBDsp, 1, "GwB1Dsp"), _in_mode(_GwBDsp, 2, "GwB2Dsp")
B1Factor, B2Factor = _in_mode(_BFactor, 1, "B1Factor"), _in_mode(_BFactor, 2, "B2Factor")
B1Cov, B2Cov = _in_mode(_BCov, 1, "B1Cov"), _in_mode(_BCov, 2, "B2Cov")
B1NodeMatch, B2NodeMatch = _in_mode(_BNodeMatch, 1, "B1NodeMatch"), _in_mode(_BNodeMatch, 2, "B2NodeMatch")


# -- Curved terms -------------------------------------------------------------------------------


class Curved(Term):
    """A geometrically weighted term whose decay is estimated, as ergm's
    ``fixed=FALSE``: a curved exponential family term.

    Its statistics are the counts of the histogram the term weights: ties
    with exactly 1, 2, ... K edgewise shared partners for gwesp, vertices with
    degree 1, 2, ... K for gwdegree, with K the cutoff (30 by default, as in
    ergm, or the largest possible count if smaller). With parameters theta and
    decay alpha, the coefficient of count k is

        eta_k = theta * exp(alpha) * (1 - (1 - exp(-alpha))^k),

    so that eta . counts is theta times the term with a fixed decay alpha. If
    the cutoff is below the largest possible count, a last statistic counts
    everything above it, with coefficient theta * exp(alpha), the limit of
    eta_k; ergm instead stops with an error when the cutoff is exceeded.
    """

    curved = True
    dyad_independent = False

    def __init__(self, term: Term, cutoff: int = 30):
        if int(cutoff) < 1:
            raise ValueError(f"cutoff must be 1 or more, not {cutoff!r}")
        self.term, self.cutoff = term, int(cutoff)

    triadic = property(lambda self: self.term.triadic)
    directed = property(lambda self: self.term.directed)
    degree_dependence = property(lambda self: self.term.degree_dependence)

    def check(self, network):
        self.term.check(network)

    def _bins(self, network) -> tuple[int, bool]:
        """The number of histogram counts and whether there is an overflow."""
        largest = self.term.largest_count(network)
        return min(self.cutoff, largest), self.cutoff < largest

    def names(self, network):
        k, overflow = self._bins(network)
        prefix = self.term.histogram_name(network)
        return [f"{prefix}#{i}" for i in range(1, k + 1)] + ([f"{prefix}#>{k}"] if overflow else [])

    def param_names(self, network):
        name = self.term.curved_name(network)
        return [name, f"{name}.decay"]

    def spec(self, network):
        k, overflow = self._bins(network)
        return self.term.histogram_spec(network, list(range(1, k + 1)), overflow)

    def initial(self) -> float:
        """The starting value of the decay: the decay argument."""
        return self.term.decay

    def starts(self, network):
        return [(1, self.initial())]

    def _weights(self, alpha: float, network):
        k, overflow = self._bins(network)
        counts = np.arange(1, k + 1)
        r = 1.0 - np.exp(-alpha)
        w = np.exp(alpha) * (1.0 - r**counts)
        dw = np.exp(alpha) * (1.0 - r**counts) - counts * r ** (counts - 1)  # d w / d alpha
        if overflow:
            w, dw = np.append(w, np.exp(alpha)), np.append(dw, np.exp(alpha))
        return w, dw

    def eta(self, params, network):
        theta, alpha = params
        return theta * self._weights(alpha, network)[0]

    def jacobian(self, params, network):
        theta, alpha = params
        w, dw = self._weights(alpha, network)
        return np.column_stack([w, theta * dw])

    def __repr__(self) -> str:
        return repr(self.term).replace("fixed=True", "fixed=False")


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

    curved = property(lambda self: self.term.curved)

    def names(self, network):
        return [f"offset({name})" for name in self.term.names(network)]

    def param_names(self, network):
        return [f"offset({name})" for name in self.term.param_names(network)]

    def eta(self, params, network):
        return self.term.eta(params, network)

    def jacobian(self, params, network):
        return self.term.jacobian(params, network)

    def starts(self, network):
        return self.term.starts(network)

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
            if isinstance(term, (Filtered, Offset, BlockOperator)):
                raise ValueError(f"F(): {term!r} can't be filtered; put F() inside offset() or "
                                 "N() instead")
        if not self.filter.dyad_independent or self.filter.is_offset:
            raise ValueError(f"F(): the filter {self.filter!r} must be a dyad-independent term")

    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))
    triadic = property(lambda self: any(t.triadic for t in self.formula))

    def _filter_label(self) -> str:
        return ("!" if self.negate else "") + self.filter.r_call()

    curved = property(lambda self: any(t.curved for t in self.formula))

    def names(self, network):
        label = self._filter_label()
        return [f"F({label})~{name}" for term in self.formula for name in term.names(network)]

    def param_names(self, network):
        label = self._filter_label()
        return [f"F({label})~{name}" for term in self.formula for name in term.param_names(network)]

    def _blocks(self, network):
        p = q = 0
        for term in self.formula:
            dp, dq = len(term.names(network)), len(term.param_names(network))
            yield term, slice(p, p + dp), slice(q, q + dq)
            p, q = p + dp, q + dq

    def eta(self, params, network):
        params = np.asarray(params, dtype=float)
        return np.concatenate([t.eta(params[q], network) for t, _, q in self._blocks(network)])

    def jacobian(self, params, network):
        params = np.asarray(params, dtype=float)
        blocks = list(self._blocks(network))
        out = np.zeros((blocks[-1][1].stop, blocks[-1][2].stop))
        for t, ps, qs in blocks:
            out[ps, qs] = t.jacobian(params[qs], network)
        return out

    def starts(self, network):
        return [(qs.start + i, v) for t, _, qs in self._blocks(network) for i, v in t.starts(network)]

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


def _formula_blocks(formula, network):
    """Each term of a formula with the slices of its statistics and parameters."""
    p = q = 0
    for term in formula:
        dp, dq = len(term.names(network)), len(term.param_names(network))
        yield term, slice(p, p + dp), slice(q, q + dq)
        p, q = p + dp, q + dq


#: How each block operator sees a block's network (the Rust core's codes):
#: the network, its union with the previous network, their intersection, or
#: the dyads that changed.
_VIEWS = {"N": 0, "Cross": 0, "Form": 1, "Persist": 2, "Diss": 2, "Change": 3}


class BlockOperator(Term):
    """Terms evaluated on each network of a combined network: ergm.multi's
    N() and tergm's Form(), Persist(), Diss(), Cross() and Change().

    Each network's statistics are combined through the operator's linear
    model, as R's ``lm`` argument: a design matrix X (networks x columns) of
    network-level attributes, by default an intercept, so that the
    statistics are summed. Each statistic g of the formula gives one
    statistic per column c, the sum over networks k of X[k, c] g_k, named
    ``N(c)~g`` as in ergm.multi: a term's coefficient in network k is the
    linear model's prediction for it. With curved terms, whose coefficients
    are nonlinear in their parameters, the statistics are each network's
    instead (named ``N#k~g``), and the parameters still those of the linear
    model.
    """

    degree_dependence = None

    def __init__(self, op: str, formula, lm=None, subset=None, weights=None, contrasts=None,
                 offset=None, label=None):
        if op not in _VIEWS:
            raise ValueError(f"unknown block operator {op!r}")
        for name, value in (("subset", subset), ("weights", weights), ("contrasts", contrasts),
                            ("offset", offset), ("label", label)):
            if value is not None and not (name == "subset" and value is True) \
                    and not (name == "weights" and value == 1):
                raise NotImplementedError(f"{op}(): the {name} argument is not supported yet")
        self.op, self.formula, self.lm = op, as_formula(formula), lm
        if not len(self.formula):
            raise ValueError(f"{op}() needs terms")
        for term in self.formula:
            if isinstance(term, (Offset, BlockOperator)):
                raise ValueError(f"{op}(): {term!r} can't be inside {op}(); put offset() outside "
                                 "instead, and don't nest the operators")

    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))
    triadic = property(lambda self: any(t.triadic for t in self.formula))
    curved = property(lambda self: any(t.curved for t in self.formula))

    @property
    def temporal(self) -> bool:
        """Whether the operator needs each network's previous network."""
        return self.op in ("Form", "Persist", "Diss", "Change")

    def _design(self, network):
        from ._lm import LmError, design

        try:
            return design(self.lm, [b.attributes for b in network.blocks])
        except LmError as e:
            raise ValueError(f"{self.op}(): {e}") from None

    def _inner_names(self, network, params=False) -> list[str]:
        return [n for t in self.formula for n in (t.param_names if params else t.names)(network)]

    def check(self, network):
        if not network.combined:
            raise ValueError(f"{self!r} needs networks combined with ergmx.Networks() or "
                             "ergmx.NetSeries()")
        if self.temporal and not network.series:
            raise ValueError(f"{self.op}() needs a series of networks: ergmx.NetSeries(), or "
                             "ergmx.tergm()")
        for block in network.blocks:
            for term in self.formula:
                term.check(block.network)
        self._design(network)
        counts = {len(self._inner_names(b.network, params=True)) for b in network.blocks}
        if len(counts) > 1:
            raise ValueError(f"{self!r}: the terms have different numbers of parameters in "
                             "different networks (for example, attribute levels that some "
                             "networks lack), which ergm.multi doesn't allow either")
        names = {tuple(self._inner_names(b.network, params=True)) for b in network.blocks}
        if len(names) > 1:
            warnings.warn(f"{self!r}: the terms' parameters have different names in different "
                          "networks, which may indicate specification problems", stacklevel=4)

    def names(self, network):
        if self.curved:
            return [f"N#{k + 1}~{n}" for k, b in enumerate(network.blocks)
                    for n in self._inner_names(b.network)]
        _, columns = self._design(network)
        return [f"{self.op}({c})~{n}" for n in self._inner_names(network.blocks[0].network)
                for c in columns]

    def param_names(self, network):
        _, columns = self._design(network)
        return [f"{self.op}({c})~{n}"
                for n in self._inner_names(network.blocks[0].network, params=True) for c in columns]

    def _thetas(self, params, network):
        """The design, and each network's parameters of the formula."""
        x, columns = self._design(network)
        coefficients = np.asarray(params, dtype=float).reshape(-1, len(columns))
        return x, [coefficients @ row for row in x]

    def eta(self, params, network):
        if not self.curved:
            return np.asarray(params, dtype=float)
        _, thetas = self._thetas(params, network)
        return np.concatenate([
            t.eta(theta[qs], b.network)
            for b, theta in zip(network.blocks, thetas)
            for t, _, qs in _formula_blocks(self.formula, b.network)
        ])

    def jacobian(self, params, network):
        if not self.curved:
            return np.eye(len(params))
        x, thetas = self._thetas(params, network)
        rows = []
        for b, row, theta in zip(network.blocks, x, thetas):
            blocks = list(_formula_blocks(self.formula, b.network))
            inner = np.zeros((blocks[-1][1].stop, blocks[-1][2].stop))
            for t, ps, qs in blocks:
                inner[ps, qs] = t.jacobian(theta[qs], b.network)
            # d eta_k / d coefficient (s, c) = d eta_k / d theta_s * x[k, c].
            rows.append(np.kron(inner, row[None, :]))
        return np.vstack(rows)

    def starts(self, network):
        # The coefficients that predict the formula's starting value in every network.
        x, columns = self._design(network)
        first = network.blocks[0].network
        out = []
        for t, _, qs in _formula_blocks(self.formula, first):
            for i, value in t.starts(first):
                b = np.linalg.lstsq(x, np.full(len(x), value), rcond=None)[0]
                out += [((qs.start + i) * len(columns) + c, float(b[c])) for c in range(len(columns))]
        return out

    def full_spec(self, network):
        x, columns = self._design(network)
        compact = not self.curved
        children = [("block", x[k].tolist() if compact else [], [],
                     [t.full_spec(b.network) for t in self.formula])
                    for k, b in enumerate(network.blocks)]
        ints = [_VIEWS[self.op], int(self.op == "Diss"), int(compact), len(columns) if compact else 0]
        return ("blocks", [], ints, children)

    def spec(self, network):
        raise TypeError(f"{self.op}() has no flat spec; use full_spec")

    def __repr__(self) -> str:
        lm = "" if self.lm is None else f", lm={self.lm!r}"
        return f"{self.op}({self.formula!r}{lm})"


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


def concurrent() -> Term:
    """Number of vertices with degree 2 or more (undirected networks)."""
    return Concurrent()


def sender() -> Term:
    """Each vertex's out-degree, one statistic per vertex but the first (directed networks)."""
    return Sender()


def receiver() -> Term:
    """Each vertex's in-degree, one statistic per vertex but the first (directed networks)."""
    return Receiver()


def sociality() -> Term:
    """Each vertex's degree, one statistic per vertex but the first (undirected networks)."""
    return Sociality()


def gwdegree(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted degree distribution (undirected networks).

    With ``fixed=False`` (the default, as in ergm) the decay is estimated, from
    ``decay``, and the term is curved (see :class:`~ergmx.terms.Curved`); ``fixed=True``
    fixes it. The other geometrically weighted terms work the same way.
    """
    return _curved_or_fixed(GwDegree(decay), fixed, cutoff)


def gwidegree(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted in-degree distribution (directed networks)."""
    return _curved_or_fixed(GwIDegree(decay), fixed, cutoff)


def gwodegree(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted out-degree distribution (directed networks)."""
    return _curved_or_fixed(GwODegree(decay), fixed, cutoff)


def cycle(k, semi: bool = False) -> Term:
    """Number of cycles of length k, for one or more k: 3 or more in undirected
    networks, 2 or more in directed ones (cycle(2) is mutual)."""
    return Cycle(k, semi)


def twopath() -> Term:
    """Number of 2-paths: i -> j -> k with i != k if directed, kstar(2) if undirected."""
    return TwoPath()


def asymmetric() -> Term:
    """Number of pairs with a tie in one direction only (directed networks)."""
    return Asymmetric()


def transitive() -> Term:
    """Number of transitive triads (directed networks): triads of types 030T,
    120D, 120U and 300 in Davis and Leinhardt's (1972) census, which have at
    least one transitive triple and no intransitive two-path.

    This is how R's ergm documents its ``transitive`` term, but ergm 4.12
    computes the number of transitive triples instead, the same as
    :func:`ttriple`. Use ``ttriple`` to reproduce ergm's results; using
    ``transitive`` warns with :class:`ErgmDifferenceWarning`.
    """
    return Transitive()


def triangle() -> Term:
    """Number of triangles; in directed networks, transitive plus cyclic triples."""
    return Triangle()


def ttriple() -> Term:
    """Number of transitive triples i -> j -> k with i -> k (directed networks)."""
    return TTriple()


def ctriple() -> Term:
    """Number of cyclic triples i -> j -> k -> i (directed networks)."""
    return CTriple()


_TYPE_DOC = """In directed networks, ``type`` sets what a shared partner k of the
    pair (i, j) is, as in ergm: ``"OTP"`` (default) i -> k -> j, ``"ITP"``
    j -> k -> i, ``"RTP"`` i <-> k <-> j, ``"OSP"`` i -> k <- j, ``"ISP"``
    i <- k -> j."""


def gwesp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30, type: str = "OTP") -> Term:
    """Geometrically weighted edgewise shared partners.

    With ``fixed=True`` the decay is fixed; with ``fixed=False`` (the
    default, as in ergm) it is estimated, and the term is curved (see
    :class:`~ergmx.terms.Curved`), starting from ``decay``. {type_doc}
    """
    return _curved_or_fixed(Gwesp(decay, type), fixed, cutoff)


def gwdsp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30, type: str = "OTP") -> Term:
    """Geometrically weighted dyadwise shared partners: as gwesp, over all pairs
    of vertices, tied or not (ordered pairs if directed). {type_doc}
    """
    return _curved_or_fixed(Gwdsp(decay, type), fixed, cutoff)


def gwnsp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30, type: str = "OTP") -> Term:
    """Geometrically weighted non-edgewise shared partners: as gwesp, over the
    pairs of vertices without a tie; gwdsp minus gwesp. {type_doc}
    """
    return _curved_or_fixed(Gwnsp(decay, type), fixed, cutoff)


def esp(d, type: str = "OTP") -> Term:
    """Number of ties with exactly d edgewise shared partners, for one or more d.
    {type_doc}
    """
    return Esp(d, type)


def dsp(d, type: str = "OTP") -> Term:
    """Number of pairs of vertices with exactly d shared partners, for one or more
    d (ordered pairs if directed). {type_doc}
    """
    return Dsp(d, type)


def nsp(d, type: str = "OTP") -> Term:
    """Number of pairs of vertices without a tie with exactly d shared partners,
    for one or more d. {type_doc}
    """
    return Nsp(d, type)


for _f in (gwesp, gwdsp, gwnsp, esp, dsp, nsp):
    # Python 3.13 dedents docstrings when compiling: dedent both before joining.
    _f.__doc__ = inspect.cleandoc(_f.__doc__).replace("{type_doc}", inspect.cleandoc(_TYPE_DOC))


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


def absdiffcat(attr: str) -> Term:
    """For each distinct nonzero absolute difference of the numeric ``attr``,
    the number of ties with that difference."""
    return AbsDiffCat(attr)


def absdiff(attr: str) -> Term:
    """Sum over ties of the absolute difference in the numeric ``attr``."""
    return AbsDiff(attr)


_B_DOC = "Bipartite networks only; b1 terms are about the first mode, b2 terms the second."


def b1star(k) -> Term:
    """Number of k-stars centred on first-mode vertices, for one or more k."""
    return B1Star(k)


def b2star(k) -> Term:
    """Number of k-stars centred on second-mode vertices, for one or more k."""
    return B2Star(k)


def b1degree(d) -> Term:
    """Number of first-mode vertices with degree exactly d, for one or more d."""
    return B1Degree(d)


def b2degree(d) -> Term:
    """Number of second-mode vertices with degree exactly d, for one or more d."""
    return B2Degree(d)


def gwb1degree(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted degree distribution of the first mode."""
    return _curved_or_fixed(GwB1Degree(decay), fixed, cutoff)


def gwb2degree(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted degree distribution of the second mode."""
    return _curved_or_fixed(GwB2Degree(decay), fixed, cutoff)


def b1concurrent() -> Term:
    """Number of first-mode vertices with degree 2 or more."""
    return B1Concurrent()


def b2concurrent() -> Term:
    """Number of second-mode vertices with degree 2 or more."""
    return B2Concurrent()


def b1factor(attr: str) -> Term:
    """For each level of ``attr`` among first-mode vertices but the first, their ties."""
    return B1Factor(attr)


def b2factor(attr: str) -> Term:
    """For each level of ``attr`` among second-mode vertices but the first, their ties."""
    return B2Factor(attr)


def b1cov(attr: str) -> Term:
    """Sum over ties of the first-mode endpoint's value of the numeric ``attr``."""
    return B1Cov(attr)


def b2cov(attr: str) -> Term:
    """Sum over ties of the second-mode endpoint's value of the numeric ``attr``."""
    return B2Cov(attr)


def _nodematch_defaults(name, diff, alpha, beta, byb2attr, levels):
    if diff or alpha != 1 or beta != 1 or byb2attr is not None or levels is not None:
        raise NotImplementedError(f"{name}: arguments other than attr are not supported yet")


def b1nodematch(attr: str, diff: bool = False, alpha: float = 1, beta: float = 1, byb2attr=None,
                levels=None) -> Term:
    """Number of 2-stars centred on second-mode vertices whose two first-mode ends
    have the same value of ``attr``."""
    _nodematch_defaults("b1nodematch", diff, alpha, beta, byb2attr, levels)
    return B1NodeMatch(attr)


def b2nodematch(attr: str, diff: bool = False, alpha: float = 1, beta: float = 1, byb1attr=None,
                levels=None) -> Term:
    """Number of 2-stars centred on first-mode vertices whose two second-mode ends
    have the same value of ``attr``."""
    _nodematch_defaults("b2nodematch", diff, alpha, beta, byb1attr, levels)
    return B2NodeMatch(attr)


def b1dsp(d) -> Term:
    """Number of pairs of first-mode vertices with exactly d shared partners, for one or more d."""
    return B1Dsp(d)


def b2dsp(d) -> Term:
    """Number of pairs of second-mode vertices with exactly d shared partners, for one or more d."""
    return B2Dsp(d)


def gwb1dsp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted shared partner distribution of pairs of first-mode vertices."""
    return _curved_or_fixed(GwB1Dsp(decay), fixed, cutoff)


def gwb2dsp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30) -> Term:
    """Geometrically weighted shared partner distribution of pairs of second-mode vertices."""
    return _curved_or_fixed(GwB2Dsp(decay), fixed, cutoff)


_BIPARTITE = (
    b1star, b2star, b1degree, b2degree, gwb1degree, gwb2degree, b1concurrent, b2concurrent,
    b1factor, b2factor, b1cov, b2cov, b1nodematch, b2nodematch, b1dsp, b2dsp, gwb1dsp, gwb2dsp,
)
for _f in _BIPARTITE:
    _f.__doc__ = inspect.cleandoc(_f.__doc__) + "\n\n" + _B_DOC
del _f


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


def N(formula, lm=None, subset=None, weights=None, contrasts=None, offset=None,  # noqa: N802
      label=None) -> Term:
    """Evaluate ``formula`` on each network of :func:`ergmx.Networks`, as
    ergm.multi's N(): with the default ``lm``, the statistics are sums over
    the networks.

    ``lm`` is a one-sided linear model over network-level attributes, in R
    syntax: ``"~log(n)"``, ``"~I(n <= 3) + weekday"``, ``"~0 + factor(.NetworkID)"``.
    Each term's coefficient in a network is the linear model's prediction:
    ``N(~edges, lm=~log(n))`` gives ``N(1)~edges`` and ``N(log(n))~edges``, the
    intercept and slope of the edges coefficient in the network size. The
    attributes are each network's graph attributes, ``n`` (its number of
    vertices), ``.NetworkID`` and ``.NetworkName``. In a formula string:
    ``"N(~edges + gwesp(0.5, fixed=TRUE), lm=~log(n))"``.
    """
    return BlockOperator("N", formula, lm, subset, weights, contrasts, offset, label)


_TEMPORAL_DOC = """``lm``, as in :func:`N`, makes the coefficients vary between
    transitions, with the attributes ``.Time``, ``.TimeID`` and ``.TimeDelta``
    as well as the networks' own (see :func:`ergmx.NetSeries`)."""


def Form(formula, lm=None, subset=None, weights=None, contrasts=None, offset=None,  # noqa: N802
         label=None) -> Term:
    """tergm's formation model: ``formula`` evaluated on the union of the
    previous and the current network, which only changes when ties form.
    {temporal_doc}
    """
    return BlockOperator("Form", formula, lm, subset, weights, contrasts, offset, label)


def Persist(formula, lm=None, subset=None, weights=None, contrasts=None, offset=None,  # noqa: N802
            label=None) -> Term:
    """tergm's persistence model: ``formula`` evaluated on the intersection of
    the previous and the current network, the ties that persisted, which only
    changes when ties dissolve. A positive coefficient means less dissolution.
    {temporal_doc}
    """
    return BlockOperator("Persist", formula, lm, subset, weights, contrasts, offset, label)


def Diss(formula, lm=None, subset=None, weights=None, contrasts=None, offset=None,  # noqa: N802
         label=None) -> Term:
    """tergm's dissolution model: :func:`Persist` with its statistics negated,
    so that a positive coefficient means more dissolution. {temporal_doc}
    """
    return BlockOperator("Diss", formula, lm, subset, weights, contrasts, offset, label)


def Cross(formula, lm=None, subset=None, weights=None, contrasts=None, offset=None,  # noqa: N802
          label=None) -> Term:
    """tergm's cross-sectional model: ``formula`` evaluated on the current
    network of each transition. {temporal_doc}
    """
    return BlockOperator("Cross", formula, lm, subset, weights, contrasts, offset, label)


def Change(formula, lm=None, subset=None, weights=None, contrasts=None, offset=None,  # noqa: N802
           label=None) -> Term:
    """tergm's change model: ``formula`` evaluated on the network of the dyads
    that changed between the previous and the current network. {temporal_doc}
    """
    return BlockOperator("Change", formula, lm, subset, weights, contrasts, offset, label)


for _f in (Form, Persist, Diss, Cross, Change):
    _f.__doc__ = inspect.cleandoc(_f.__doc__).replace("{temporal_doc}", inspect.cleandoc(_TEMPORAL_DOC))


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
    edges, mutual, asymmetric, kstar, istar, ostar, twopath, degree, idegree, odegree, isolates,
    concurrent, gwdegree, gwidegree, gwodegree, sender, receiver, sociality, triangle, ttriple,
    ctriple, transitive, cycle, gwesp, gwdsp, gwnsp, esp, dsp, nsp, nodematch, nodemix, nodefactor,
    nodeifactor, nodeofactor, nodecov, nodeicov, nodeocov, absdiff, absdiffcat, edgecov,
    *_BIPARTITE,
)
for _f in _PLAIN:
    globals()[_f.__name__] = _recording(_f)
del _f

#: The operators that evaluate terms on each network of a combined network.
BLOCK_OPERATORS = ("N", "Form", "Persist", "Diss", "Cross", "Change")

#: Every term function by name; F, offset and the block operators are operators.
TERMS = {name: globals()[name]
         for name in [f.__name__ for f in _PLAIN] + ["F", "offset", *BLOCK_OPERATORS]}
