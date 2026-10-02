"""Model terms, with the same names and statistics as the R package ergm.

Terms can be combined with ``+`` or written as a formula string::

    edges() + nodematch("Grade") + gwesp(0.5, fixed=True)
    "edges + nodematch('Grade') + gwesp(0.5, fixed=True)"
"""

from __future__ import annotations

import inspect
import re
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
    def __init__(self, attr: str, diff: bool = False, levels=None):
        super().__init__(attr)
        self.diff, self.levels = diff, levels

    def _chosen(self, network):
        values = _values(network, self.attr)
        return values, _select_levels(_sorted_levels(values), self.levels)

    def names(self, network):
        if not self.diff:
            return [self.label]
        return [f"{self.label}.{_level_name(v)}" for v in self._chosen(network)[1]]

    def spec(self, network):
        name = "nodematchdiff" if self.diff else "nodematch"
        values, chosen = self._chosen(network)
        return (name, [], _level_codes(values, chosen))

    def __repr__(self) -> str:
        return _call_repr(self, f"nodematch({self.attr!r}, diff=True)" if self.diff else super().__repr__())


class NodeFactor(_AttributeTerm):
    """One statistic per selected attribute level: by default all but the
    first (sorted) one, as in ergm's ``levels = -1``."""

    degree_dependence = _DEGREES

    def __init__(self, attr: str, levels=-1):
        super().__init__(attr)
        self.levels = levels

    def _levels(self, network):
        values = _values(network, self.attr)
        chosen = _select_levels(_sorted_levels(values), self.levels)
        return chosen, _level_codes(values, chosen)

    def __repr__(self) -> str:
        return _call_repr(self, super().__repr__())

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
        return (self.rust_name, [float(v) for v in _values(network, self.attr)], [])


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
    values = _values(network, attr)
    every = sorted(set(values))
    chosen = _select_levels(every, levels)
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
        return ("absdiff", [float(v) for v in _values(network, self.attr)], [])


class AbsDiffCat(_AttributeTerm):
    """For each distinct nonzero absolute difference of a numeric attribute,
    the number of ties with that difference."""

    rust_name = "absdiffcat"

    def _differences(self, network) -> list[float]:
        x = np.array([float(v) for v in _values(network, self.attr)])
        return sorted({float(d) for d in np.abs(x[:, None] - x[None, :]).ravel() if d != 0})

    def names(self, network):
        return [f"absdiff.{self.attr}.{_level_name(d)}" for d in self._differences(network)]

    def spec(self, network):
        x = [float(v) for v in _values(network, self.attr)]
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
    degree_dependence = property(lambda self: frozenset({self.b}))

    rust_name = "kstar"

    def names(self, network):
        return [f"{self.b}star{k}" for k in self.ks]


class _BDegree(_Mode, Degree):
    degree_dependence = property(lambda self: frozenset({self.b}))

    rust_name = "degree"

    def names(self, network):
        return [f"{self.b}degree{k}" for k in self.ks]


class _GwBDegree(_Mode, GwDegree):
    degree_dependence = property(lambda self: frozenset({self.b}))

    rust_name = "gwdegree"

    def names(self, network):
        return [f"gw{self.b}deg.fixed.{self.decay:g}"]

    def largest_count(self, network) -> int:
        return int(np.sum(network.mode != self.mode_number))

    def histogram_name(self, network) -> str:
        return f"gw{self.b}degree"

    curved_name = histogram_name


class _BConcurrent(_Mode, Concurrent):
    degree_dependence = property(lambda self: frozenset({self.b}))

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

    degree_dependence = property(lambda self: frozenset({self.b}))

    def _levels(self, network):
        values = _values(network, self.attr)
        mine = [v for v, m in zip(values, network.mode) if m == self.mode_number]
        levels = _select_levels(_sorted_levels(mine), self.levels)
        codes = [c if m == self.mode_number else -1 for c, m in zip(_level_codes(values, levels), network.mode)]
        return levels, codes

    def names(self, network):
        return [f"{self.b}factor.{self.attr}.{_level_name(v)}" for v in self._levels(network)[0]]

    def spec(self, network):
        return ("nodefactor", [], self._levels(network)[1])


class _BCov(_Mode, NodeCov):
    """nodecov counting only the endpoints in one mode."""

    degree_dependence = property(lambda self: frozenset({self.b}))

    @property
    def label(self) -> str:
        return f"{self.b}cov.{self.attr}"

    def spec(self, network):
        x = [float(v) if m == self.mode_number else 0.0
             for v, m in zip(_values(network, self.attr), network.mode)]
        return ("nodecov", x, [])


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


# -- Multilevel terms (MPNet's) --------------------------------------------------------------------


class _Multilevel(Term):
    """One of MPNet's configurations of two-level networks (`Wang, Robins,
    Pattison and Lazega 2013 <https://doi.org/10.1016/j.socnet.2013.01.004>`__), with the levels given by a vertex attribute."""

    dyad_independent = False
    directed = False
    #: MPNet's name, and whether the statistic is alternating (has a decay).
    mpnet = ""
    alternating = False

    def __init__(self, attr: str, levels=None, decay: float | None = None):
        self.attr, self.levels = attr, None if levels is None else tuple(levels)
        if self.levels is not None and len(self.levels) != 2:
            raise ValueError(f"{self.mpnet}: levels must be the two levels (A, B), not {levels!r}")
        self.decay = None if decay is None else float(decay)

    triadic = property(lambda self: self.mpnet.startswith(("TX", "ATX", "C4", "EXT")))

    def _levels(self, network):
        values = network.attribute(self.attr)
        if self.levels is not None:
            return self.levels
        present = sorted({v for v in values if v is not None}, key=lambda v: (str(type(v)), v))
        if len(present) != 2:
            raise ValueError(f"{self.mpnet}: {self.attr!r} has {len(present)} levels; give the two "
                             "levels of the network with levels=(A, B)")
        return tuple(present)

    def check(self, network):
        super().check(network)
        a, b = self._levels(network)
        values = network.attribute(self.attr)
        if a not in values or b not in values:
            raise ValueError(f"{self.mpnet}: no vertices with {self.attr} = {a!r} or {b!r}")
        if network.directed and len(network.edges):
            # Affiliations are the arcs from A to B; arcs from B to A would be lost.
            level = np.asarray(values, dtype=object)
            tails, heads = level[network.edges[:, 0]], level[network.edges[:, 1]]
            if np.any((tails == b) & (heads == a)):
                raise ValueError(
                    f"{self.mpnet}: in directed multilevel networks, affiliations are the arcs from level "
                    f"A ({a!r}) to level B ({b!r}), but the network has arcs from {b!r} to {a!r}. Reverse "
                    "them, or give levels=(B, A); then fix the dyads from B to A with blocks().")

    @property
    def label(self) -> str:
        suffix = f".{self.decay:g}" if self.alternating else ""
        return f"{self.mpnet}.{self.attr}{suffix}"

    def spec(self, network):
        a, b = self._levels(network)
        codes = [0 if v == a else 1 if v == b else -1 for v in network.attribute(self.attr)]
        rust = self.rust_name + ("arc" if network.directed and self.mpnet in ("L3XAX", "L3XBX") else "")
        return (rust, [self.decay] if self.alternating else [], codes)

    def __repr__(self) -> str:
        decay = f", decay={self.decay:g}" if self.alternating else ""
        levels = "" if self.levels is None else f", levels={self.levels!r}"
        return f"{self.rust_name}({self.attr!r}{decay}{levels})"

    # Estimated decays (curved; see Curved): the histogram the term weights.
    #: The Rust core's histogram kind, or None if the decay can't be estimated.
    histogram_kind: int | None = None

    def _sizes(self, network) -> tuple[int, int]:
        a, b = self._levels(network)
        values = network.attribute(self.attr)
        return sum(v == a for v in values), sum(v == b for v in values)

    def largest_count(self, network) -> int:
        size_a, size_b = self._sizes(network)
        mine, other = (size_a, size_b) if self.side == 0 else (size_b, size_a)
        # X-degrees and shared partners are at most the other level's size; degrees within S, its size - 1.
        return mine - 1 if self.histogram_kind in (3, 4, 5) else other

    def histogram_name(self, network) -> str:
        return f"{self.mpnet}.{self.attr}"

    curved_name = histogram_name

    def histogram_spec(self, network, ks, overflow):
        a, b = self._levels(network)
        codes = [0 if v == a else 1 if v == b else -1 for v in network.attribute(self.attr)]
        return ("multilevelhistogram", [], _chunks([self.histogram_kind, self.side], codes, ks, [int(overflow)]))


#: The side (0 for A, 1 for B) and histogram kind of the alternating
#: configurations whose decay can be estimated.
_HISTOGRAMS = {
    "AXS1A": (0, 0), "AXS1B": (1, 0), "AXS1Ain": (0, 1), "AXS1Bin": (1, 1), "AXS1Aout": (0, 2),
    "AXS1Bout": (1, 2), "AAS1X": (0, 3), "ABS1X": (1, 3), "AAinS1X": (0, 4), "ABinS1X": (1, 4),
    "AAoutS1X": (0, 5), "ABoutS1X": (1, 5), "ATXAX": (0, 6), "ATXBX": (1, 6), "ATXAXarc": (0, 7),
    "ATXBXarc": (1, 7), "ATXAXreciprocity": (0, 8), "ATXBXreciprocity": (1, 8),
}


def _multilevel(mpnet: str, alternating: bool = False, directed: bool | None = False):
    side, kind = _HISTOGRAMS.get(mpnet, (0, None))
    return type(mpnet, (_Multilevel,), {"mpnet": mpnet, "alternating": alternating, "side": side,
                                         "histogram_kind": kind, "rust_name": mpnet.lower(),
                                         "directed": directed})


def _alternating(term: _Multilevel, fixed: bool, cutoff: int) -> Term:
    """The term with its decay fixed, or estimated (curved) when it can be."""
    if fixed:
        return term
    if term.histogram_kind is None:
        raise NotImplementedError(f"{term.mpnet}: the decay of a term with two alternating parts can't be "
                                  "estimated yet; give fixed=TRUE")
    return Curved(term, cutoff)


(Star2AX, Star2BX, AXS1A, AXS1B, AAS1X, ABS1X, AAAXS, ABAXS, TXAX, TXBX, ATXAX, ATXBX, L3XAX, L3XBX,
 L3AXB, C4AXB) = (
    _multilevel("Star2AX"), _multilevel("Star2BX"), _multilevel("AXS1A", True), _multilevel("AXS1B", True),
    _multilevel("AAS1X", True), _multilevel("ABS1X", True), _multilevel("AAAXS", True),
    _multilevel("ABAXS", True), _multilevel("TXAX"), _multilevel("TXBX"), _multilevel("ATXAX", True),
    _multilevel("ATXBX", True), _multilevel("L3XAX", directed=None), _multilevel("L3XBX", directed=None),
    _multilevel("L3AXB"), _multilevel("C4AXB"),
)
EXTA, EXTB, ASAXASB = _multilevel("EXTA"), _multilevel("EXTB"), _multilevel("ASAXASB", True)
_DIRECTED_MULTILEVEL = {
    name: _multilevel(name, alternating, directed=True) for name, alternating in [
        ("In2StarAX", False), ("In2StarBX", False), ("Out2StarAX", False), ("Out2StarBX", False),
        ("AXS1Ain", True), ("AXS1Bin", True), ("AXS1Aout", True), ("AXS1Bout", True),
        ("AAinS1X", True), ("ABinS1X", True), ("AAoutS1X", True), ("ABoutS1X", True),
        ("TXAXarc", False), ("TXBXarc", False), ("TXAXreciprocity", False), ("TXBXreciprocity", False),
        ("ATXAXarc", True), ("ATXBXarc", True), ("ATXAXreciprocity", True), ("ATXBXreciprocity", True),
        ("L3XAXreciprocity", False), ("L3XBXreciprocity", False), ("L3AXBin", False), ("L3AXBout", False),
        ("L3AXBpath", False), ("L3BXApath", False), ("C4AXBentrainment", False), ("C4AXBexchange", False),
        ("C4AXBexchangeAreciprocity", False), ("C4AXBexchangeBreciprocity", False),
        ("C4AXBreciprocity", False), ("AinASXAinBS", True), ("AoutASXAoutBS", True), ("AinASXAoutBS", True),
        ("AoutASXAinBS", True),
    ]
}


# -- More of ergm's vocabulary ---------------------------------------------------------------------


def _call_repr(term, default: str) -> str:
    """A term as it was called (its factory and arguments), when recorded."""
    call = getattr(term, "_call", None)
    if call is None:
        return default
    name, args, kwargs = call
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def _values(network: Network, attr: str) -> list:
    """A vertex attribute's values, refusing missing ones, as ergm does."""
    values = network.attribute(attr)
    if any(v is None or (isinstance(v, float) and v != v) for v in values):
        raise ValueError(f"the vertex attribute {attr!r} has missing values (None or NaN), which "
                         "ergm doesn't support in terms either; recode them as a level of their own")
    return values


def _sorted_levels(values) -> list:
    return sorted(set(values), key=lambda v: (str(type(v)), v))


def _select_levels(levels: list, spec, what: str = "levels") -> list:
    """The levels selected by an ergm levels specification (see R's
    ?nodal_attributes): None or TRUE (all), FALSE (none), values (strings),
    or 1-based indices into the sorted levels (negative ones to leave out)."""
    if spec is None or spec is True:
        return list(levels)
    if spec is False:
        return []
    items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    if items and all(isinstance(v, int) and not isinstance(v, bool) for v in items):
        return [levels[i] for i in _select(items, len(levels), what)]
    missing = [v for v in items if v not in levels]
    if missing:
        raise ValueError(f"{what}: no levels {missing}; the levels are {levels}")
    return items


def _level_codes(values: list, chosen: list) -> list[int]:
    """Each vertex's position among the chosen levels, or -1."""
    position = {v: k for k, v in enumerate(chosen)}
    return [position.get(v, -1) for v in values]


def _vertex_selection(spec, numbers: list[int], what: str) -> list[int]:
    """Vertex numbers (1-based) selected by ergm's nodes= argument among `numbers`."""
    return [numbers[i] for i in _select(spec, len(numbers), what)]


def _mode_values(network: Network, mode: int | None) -> np.ndarray:
    """1/0 mask of a bipartite mode's vertices (all ones without a mode)."""
    if mode is None:
        return np.ones(network.n, dtype=int)
    return (network.mode == mode).astype(int)


class _DegreeRange(Term):
    """Vertices with degree in [from, to), for each range: degree(d, by=,
    homophily=) and degrange(), their in-, out- and bipartite versions,
    b1mindegree and concurrent(by=)."""

    dyad_independent = False

    def __init__(self, rust: str, prefix: str, ranges, *, by=None, homophily=False, levels=None,
                 mode=None, style="range", directed=None):
        self.rust, self.prefix, self.ranges = rust, prefix, list(ranges)
        self.by, self.homophily, self.levels, self.mode, self.style = by, bool(homophily), levels, mode, style
        self.directed = directed
        if self.homophily and by is None:
            raise ValueError(f"{prefix}: homophily=TRUE needs by=")

    @property
    def degree_dependence(self):
        if self.homophily:
            return None
        if self.mode is not None:
            return frozenset({f"b{self.mode}"})
        return {"degrange": _DEGREES, "idegrange": _IN, "odegrange": _OUT}[self.rust]

    def check(self, network):
        if self.mode is not None and not network.bipartite:
            raise ValueError(f"{self.prefix}: needs a bipartite network")
        super().check(network)

    def _chosen(self, network):
        values = _values(network, self.by)
        if self.mode is not None:
            values_in_mode = [v for v, m in zip(values, network.mode) if m == self.mode]
            return values, _select_levels(_sorted_levels(values_in_mode), self.levels)
        return values, _select_levels(_sorted_levels(values), self.levels)

    def _range_name(self, lo, hi) -> str:
        if self.style == "exact":
            return f"{self.prefix}{lo}"
        if self.style == "min":
            return f"{self.prefix}{lo}"
        if self.style == "concurrent":
            return self.prefix
        return f"{self.prefix}{lo}+" if hi is None else f"{self.prefix}{lo}to{hi}"

    def names(self, network):
        bases = [self._range_name(lo, hi) for lo, hi in self.ranges]
        if self.by is None:
            return bases
        if self.homophily:
            return [f"{b}.homophily.{self.by}" for b in bases]
        _, chosen = self._chosen(network)
        if self.style == "exact":
            return [f"{b}.{self.by}.{_level_name(v)}" for v in chosen for b in bases]
        return [f"{b}.{self.by}{_level_name(v)}" for v in chosen for b in bases]

    def spec(self, network):
        frm = [lo for lo, _ in self.ranges]
        to = [-1 if hi is None else hi for _, hi in self.ranges]
        mask = [] if self.mode is None else _mode_values(network, self.mode).tolist()
        codes = []
        if self.by is not None and self.homophily:
            # As ergm: every vertex counts, with its value over both modes, and
            # the levels left out are one more level.
            values = _values(network, self.by)
            chosen = _select_levels(_sorted_levels(values), self.levels)
            codes = [c if c >= 0 else len(chosen) for c in _level_codes(values, chosen)]
        elif self.by is not None:
            values, chosen = self._chosen(network)
            codes = _level_codes(values, chosen)
            if self.mode is not None:
                codes = [c if m == self.mode else -1 for c, m in zip(codes, network.mode)]
        return (self.rust, [], _chunks(frm, to, mask, codes, [int(self.homophily)]))

    def __repr__(self) -> str:
        return _call_repr(self, f"{self.prefix}(...)")


def _ranges(frm, to):
    """degrange's from and to, recycled as R does; to=None (or Inf) is unbounded."""
    def listed(x):
        return [x] if not isinstance(x, (list, tuple)) else list(x)

    frm, to = listed(frm), listed(to)
    if len(frm) == 1 and len(to) > 1:
        frm = frm * len(to)
    if len(to) == 1 and len(frm) > 1:
        to = to * len(frm)
    if len(frm) != len(to):
        raise ValueError("from and to must have the same length, or one of them length 1")
    out = []
    for lo, hi in zip(frm, to):
        hi = None if hi is None or (isinstance(hi, float) and np.isinf(hi)) else int(hi)
        if int(lo) < 0 or (hi is not None and hi <= int(lo)):
            raise ValueError(f"degree range [{lo}, {hi}) is empty or negative")
        out.append((int(lo), hi))
    return out


class _DegreePower(Term):
    """Sum over vertices of their degree to the power 1.5 (degree1.5)."""

    dyad_independent = False

    def __init__(self, rust: str, label: str, directed):
        self.rust, self._label, self.directed = rust, label, directed

    @property
    def label(self):
        return self._label

    @property
    def degree_dependence(self):
        return {"degreepower": _DEGREES, "idegreepower": _IN, "odegreepower": _OUT}[self.rust]

    def spec(self, network):
        return (self.rust, [1.5], [])

    def __repr__(self) -> str:
        return self._label.replace(".", "_")


class ConcurrentTies(Term):
    """Ties of each vertex beyond its first, by level of ``by``."""

    dyad_independent = False
    directed = False
    degree_dependence = _DEGREES

    def __init__(self, by=None, levels=None):
        self.by, self.levels = by, levels

    def names(self, network):
        if self.by is None:
            return ["concurrentties"]
        chosen = _select_levels(_sorted_levels(_values(network, self.by)), self.levels)
        return [f"concurrentties.{self.by}{_level_name(v)}" for v in chosen]

    def spec(self, network):
        codes = []
        if self.by is not None:
            values = _values(network, self.by)
            codes = _level_codes(values, _select_levels(_sorted_levels(values), self.levels))
        return ("concurrentties", [], _chunks([], codes))

    def __repr__(self) -> str:
        return "concurrentties" if self.by is None else f"concurrentties(by={self.by!r})"


class Density(Term):
    """The density: edges over the number of dyads."""

    degree_dependence = frozenset()

    def spec(self, network):
        if network.bipartite:
            n1 = int(np.sum(network.mode == 1))
            dyads = n1 * (network.n - n1)
        else:
            dyads = network.n * (network.n - 1) / (1 if network.directed else 2)
        return ("scalededges", [1.0 / dyads], [])


class MeanDeg(Term):
    """The mean degree: twice the edges over the vertices (the edges, if directed)."""

    degree_dependence = frozenset()

    def spec(self, network):
        return ("scalededges", [(1.0 if network.directed else 2.0) / network.n], [])


class IsolatedEdges(Term):
    """Ties whose two vertices have no other tie."""

    dyad_independent = False
    directed = False


def _matrix_argument(network: Network, x, what: str) -> tuple[np.ndarray, str | None]:
    """An n x n matrix given as a graph attribute name, an array or a graph,
    and the attribute name (for the statistic's name)."""
    name = x if isinstance(x, str) else None
    value = network.graph_attribute(x) if isinstance(x, str) else x
    if hasattr(value, "get_adjacency"):  # an igraph graph
        value = np.array(value.get_adjacency().data, dtype=float)
    elif hasattr(value, "nodes") and hasattr(value, "edges"):  # a networkx graph
        import networkx as nx

        value = nx.to_numpy_array(value, nodelist=list(value.nodes))
    m = np.asarray(value, dtype=float)
    if network.bipartite and m.shape != (network.n, network.n):
        first, second = np.flatnonzero(network.mode == 1), np.flatnonzero(network.mode == 2)
        if m.shape == (len(first), len(second)):
            full = np.zeros((network.n, network.n))
            full[np.ix_(first, second)] = m
            m = full + full.T
    if m.shape != (network.n, network.n):
        raise ValueError(f"{what}: expected a {network.n} x {network.n} matrix, got shape {m.shape}")
    return m, name


class DyadCov(Term):
    """A dyadic covariate by dyad state: summed over mutual dyads, and over the
    dyads with only the tie from the lower to the higher vertex, and the
    reverse (directed networks; edgecov if undirected)."""

    def __init__(self, x):
        self.x = x

    def _matrix(self, network):
        m, name = _matrix_argument(network, self.x, "dyadcov")
        if network.directed:
            upper = np.triu(m, 1)
            m = upper + upper.T  # ergm uses the upper triangle
        return m, name

    def names(self, network):
        _, name = self._matrix(network)
        base = f"dyadcov.{name}" if name else "dyadcov"
        return [f"{base}.{s}" for s in ("mutual", "utri", "ltri")] if network.directed else [base]

    def spec(self, network):
        m, _ = self._matrix(network)
        return ("dyadcov" if network.directed else "edgecov", m.ravel().tolist(), [])

    def check(self, network):
        super().check(network)
        if network.directed:
            warnings.warn(
                "dyadcov's utri statistic counts the dyads whose only tie is in the upper triangle of "
                "the adjacency matrix (from the lower- to the higher-numbered vertex), and ltri the "
                "reverse, as R's ergm documents; ergm 4.12 computes them the other way round.",
                ErgmDifferenceWarning, stacklevel=5,
            )

    def __repr__(self) -> str:
        return f"dyadcov({self.x!r})" if isinstance(self.x, str) else "dyadcov(<matrix>)"


class Hamming(Term):
    """The Hamming distance to a reference network (the observed one by
    default), weighted by the covariate ``cov`` if given."""

    def __init__(self, x=None, cov=None):
        self.x, self.cov = x, cov

    def names(self, network):
        if self.x is None:
            return ["hamming"]
        base = f"hamming.{self.x}" if isinstance(self.x, str) else "hamming"
        if self.cov is not None:
            base += f".wt.{self.cov}" if isinstance(self.cov, str) else ".wt"
        return [base]

    def spec(self, network):
        if self.x is None:
            reference = network.dyad_mask(network.edges).astype(float)
        else:
            reference, _ = _matrix_argument(network, self.x, "hamming")
        weights = [] if self.cov is None else _matrix_argument(network, self.cov, "hamming")[0].ravel().tolist()
        return ("hamming", (reference != 0).astype(float).ravel().tolist() + weights, [])

    def __repr__(self) -> str:
        return "hamming" if self.x is None else f"hamming({self.x!r})"


class AttrCov(_AttributeTerm):
    """Sum over ties of a covariate of the mixing type of their vertices' levels."""

    def __init__(self, attr: str, mat):
        super().__init__(attr)
        self.mat = mat

    def spec(self, network):
        values = _values(network, self.attr)
        levels = _sorted_levels(values)
        m = np.asarray(self.mat, dtype=float)
        if m.shape != (len(levels), len(levels)):
            raise ValueError(f"attrcov: mat must be {len(levels)} x {len(levels)} (the levels of "
                             f"{self.attr!r}), not {m.shape}")
        return ("attrcov", m.ravel().tolist(), _level_codes(values, levels))


class MixingMatrix(Term):
    """The cells (or margins) of the mixing matrix of a row and a column
    attribute, as ergm's mm()."""

    def __init__(self, attrs, levels=None, levels2=-1):
        text = str(attrs).strip()
        if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]  # mm("A"), an attribute name
        if "~" in text:
            left, right = (s.strip() for s in text.split("~", 1))
            left = left or right  # a one-sided formula is symmetrized
        else:
            left = right = text
        for side in (left, right):
            if side != "." and not side.replace("_", "").replace(".", "").isalnum():
                raise ValueError(f"mm: attributes must be vertex attribute names (or .), not {side!r}")
        if left == "." and right == ".":
            raise ValueError("mm: give an attribute on at least one side")
        self.row, self.col, self.levels, self.levels2 = left, right, levels, levels2

    def _layout(self, network):
        """The row and column levels (None for a margin), the vertices' codes
        and the selected cells as (row, column) positions."""
        def side(attr):
            if attr == ".":
                return None, [0] * network.n
            values = _values(network, attr)
            chosen = _select_levels(_sorted_levels(values), self.levels)
            return chosen, _level_codes(values, chosen)

        rows, row_codes = side(self.row)
        cols, col_codes = side(self.col)
        nr, nc = len(rows) if rows is not None else 1, len(cols) if cols is not None else 1
        symmetric = not network.directed and self.row == self.col
        cells = [(r, c) for c in range(nc) for r in range(nr) if not symmetric or r <= c]
        if isinstance(self.levels2, (list, tuple)) and self.levels2 and isinstance(self.levels2[0], (list, tuple)):
            matrix = np.asarray(self.levels2, dtype=bool)
            selected = [cell for cell in cells if matrix[cell]]
        else:
            selected = [cells[i] for i in _select(self.levels2, len(cells), "levels2")]
        return rows, cols, row_codes, col_codes, selected, symmetric

    def names(self, network):
        rows, cols, _, _, cells, _ = self._layout(network)

        def part(attr, levels, k):
            return "." if levels is None else f"{attr}={_level_name(levels[k])}"

        return [f"mm[{part(self.row, rows, r)},{part(self.col, cols, c)}]" for r, c in cells]

    def spec(self, network):
        rows, cols, row_codes, col_codes, cells, symmetric = self._layout(network)
        nr, nc = len(rows) if rows is not None else 1, len(cols) if cols is not None else 1
        mapping = np.full((nr, nc), -1, dtype=np.int64)
        for stat, (r, c) in enumerate(cells):
            mapping[r, c] = stat
            if symmetric:
                mapping[c, r] = stat
        both = not network.directed and not symmetric
        return ("mixmatrix", [], _chunks(row_codes, col_codes, [nc], [int(both)], mapping.ravel().tolist()))

    def __repr__(self) -> str:
        attrs = self.row if self.row == self.col else f"{self.row}~{self.col}"
        return f"mm({attrs!r})"


class _CovRange(_AttributeTerm):
    """Sum over vertices of the range of a covariate over their neighbours."""

    dyad_independent = False
    #: The sides of a directed tie counted: the tail's out-neighbours, the head's in-neighbours.
    sides = (1, 1)
    mode: int | None = None

    def check(self, network):
        if self.mode is not None and not network.bipartite:
            raise ValueError(f"{self.rust_name}: needs a bipartite network")
        super().check(network)

    def spec(self, network):
        x = [float(v) for v in _values(network, self.attr)]
        mask = [] if self.mode is None else _mode_values(network, self.mode).tolist()
        return ("covrange", x, _chunks(self.sides, mask))


class NodeCovRange(_CovRange):
    pass


class NodeICovRange(_CovRange):
    directed = True
    sides = (0, 1)


class NodeOCovRange(_CovRange):
    directed = True
    sides = (1, 0)


class B1CovRange(_CovRange):
    directed = False
    mode = 1


class B2CovRange(_CovRange):
    directed = False
    mode = 2


class _FactorDistinct(_AttributeTerm):
    """Sum over vertices of the number of distinct levels among their neighbours."""

    dyad_independent = False
    sides = (1, 1)
    mode: int | None = None

    def __init__(self, attr: str, levels=True):
        super().__init__(attr)
        self.levels = levels

    def check(self, network):
        if self.mode is not None and not network.bipartite:
            raise ValueError(f"{self.rust_name}: needs a bipartite network")
        super().check(network)

    def spec(self, network):
        values = _values(network, self.attr)
        if self.mode is not None:
            # The levels of the other mode's vertices, the neighbours.
            others = [v for v, m in zip(values, network.mode) if m != self.mode]
            chosen = _select_levels(_sorted_levels(others), self.levels)
            codes = [c if m != self.mode else -1 for c, m in zip(_level_codes(values, chosen), network.mode)]
        else:
            codes = _level_codes(values, _select_levels(_sorted_levels(values), self.levels))
        mask = [] if self.mode is None else _mode_values(network, self.mode).tolist()
        return ("factordistinct", [], _chunks(codes, self.sides, mask))


class NodeFactorDistinct(_FactorDistinct):
    pass


class NodeOFactorDistinct(_FactorDistinct):
    directed = True
    sides = (1, 0)


class NodeIFactorDistinct(_FactorDistinct):
    directed = True
    sides = (0, 1)


class B1FactorDistinct(_FactorDistinct):
    directed = False
    mode = 1


class B2FactorDistinct(_FactorDistinct):
    directed = False
    mode = 2


_DIRS = {"t-h": 0, "tail-head": 0, "b1-b2": 0, "h-t": 1, "head-tail": 1, "b2-b1": 1}
_SIGN_ACTIONS = ("identity", "abs", "posonly", "negonly")


class Diff(_AttributeTerm):
    """Sum over ties of a function of the difference of the vertices' values."""

    rust_name = "diff"

    def __init__(self, attr: str, pow: float = 1, dir: str = "t-h", sign_action: str = "identity"):
        super().__init__(attr)
        self.dir, self.action, self.pow = str(dir).lower(), str(sign_action).lower(), float(pow)
        if self.dir not in _DIRS:
            raise ValueError(f"diff: dir must be one of {', '.join(_DIRS)}, not {dir!r}")
        if self.action not in _SIGN_ACTIONS:
            raise ValueError(f"diff: sign.action must be one of {', '.join(_SIGN_ACTIONS)}, not {sign_action!r}")
        if self.action in ("identity", "negonly") and self.pow != round(self.pow):
            raise ValueError("diff: with negative differences, pow must be an integer")

    @property
    def label(self):
        pow_ = "" if self.pow == 1 else f"{self.pow:g}"
        action = "" if self.action == "identity" else f".{self.action}"
        direction = "" if self.action == "abs" else f".{self.dir}"
        return f"diff{pow_}{action}{direction}.{self.attr}"

    def spec(self, network):
        x = [float(v) for v in _values(network, self.attr)] + [self.pow]
        orient = 0 if network.directed else 2 if network.bipartite else 1
        ints = [_DIRS[self.dir], _SIGN_ACTIONS.index(self.action), orient]
        if orient == 2:
            ints += _mode_values(network, 1).tolist()
        return ("diff", x, ints)


class SmallDiff(_AttributeTerm):
    """Ties whose vertices' values differ by at most ``cutoff`` (as ergm's code)."""

    def __init__(self, attr: str, cutoff: float):
        super().__init__(attr)
        self.cutoff = float(cutoff)

    @property
    def label(self):
        return f"smalldiff.{self.attr}{self.cutoff:g}"

    def spec(self, network):
        return ("smalldiff", [float(v) for v in _values(network, self.attr)] + [self.cutoff], [])


class AltKStar(Term):
    """Alternating k-stars with a fixed lambda."""

    dyad_independent = False
    directed = False
    degree_dependence = _DEGREES

    def __init__(self, lam: float):
        self.lam = float(lam)
        if self.lam <= 1:
            raise ValueError("altkstar: lambda must be greater than 1")

    @property
    def label(self):
        return f"altkstar.{self.lam:g}"

    def spec(self, network):
        return ("altkstar", [self.lam], [])

    def __repr__(self) -> str:
        return f"altkstar({self.lam:g}, fixed=True)"


class _NodeEffects(Term):
    """One statistic per selected vertex: its out-degree (sender), in-degree
    (receiver) or degree (sociality, b1sociality, b2sociality), optionally
    counting only ties to vertices of the same level of ``attr``."""

    rust_term = ""
    mode: int | None = None

    def __init__(self, nodes=-1, attr=None, levels=None):
        self.nodes, self.attr, self.levels = nodes, attr, levels

    @property
    def degree_dependence(self):
        if self.attr is not None:
            return None
        if self.mode is not None:
            return frozenset({f"b{self.mode}"})
        return {"nodeofactor": _OUT, "nodeifactor": _IN, "nodefactor": _DEGREES}[self.rust_term]

    def check(self, network):
        if self.mode is not None and not network.bipartite:
            raise ValueError(f"{self.rust_name}: needs a bipartite network")
        super().check(network)

    def _vertices(self, network) -> list[int]:
        numbers = [v + 1 for v in range(network.n)
                   if self.mode is None or network.mode[v] == self.mode]
        return _vertex_selection(self.nodes, numbers, "nodes")

    def names(self, network):
        suffix = f".{self.attr}" if self.attr is not None else ""
        return [f"{self.rust_name}{v}{suffix}" for v in self._vertices(network)]

    def spec(self, network):
        stat = np.full(network.n, -1)
        for k, v in enumerate(self._vertices(network)):
            stat[v - 1] = k
        if self.attr is None:
            return (self.rust_term, [], stat.tolist())
        values = _values(network, self.attr)
        codes = _level_codes(values, _select_levels(_sorted_levels(values), self.levels))
        codes = [c if c >= 0 else -2 - k for k, c in enumerate(codes)]  # levels left out never match
        return ("factormatch", [], _chunks(stat.tolist(), codes))

    def __repr__(self) -> str:
        return f"{self.rust_name}()"


class Sender(_NodeEffects):
    directed = True
    rust_term = "nodeofactor"


class Receiver(_NodeEffects):
    directed = True
    rust_term = "nodeifactor"


class Sociality(_NodeEffects):
    directed = False
    rust_term = "nodefactor"


class B1Sociality(_NodeEffects):
    directed = False
    rust_term = "nodefactor"
    mode = 1


class B2Sociality(_NodeEffects):
    directed = False
    rust_term = "nodefactor"
    mode = 2


#: Davis and Leinhardt's (1972) triad types, in ergm's order (codes 0 to 15).
TRIAD_TYPES = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201",
               "120D", "120U", "120C", "210", "300")


class _Triads(Term):
    """Counts of triads by type (the triad census and the terms built on it)."""

    dyad_independent = False
    triadic = True
    #: Statistic of each type: {type code: statistic}, by directedness.
    groups: dict = {}

    def _map(self, network) -> dict[int, int]:
        return self.groups[network.directed]

    def spec(self, network):
        size = 16 if network.directed else 4
        mapping = self._map(network)
        return ("triadcensus", [], [mapping.get(t, -1) for t in range(size)])


class TriadCensus(_Triads):
    """The triad census: counts of the selected triad types (by default all
    but the empty one)."""

    def __init__(self, levels=None):
        self.levels = levels

    def _types(self, network) -> list[int]:
        size = 16 if network.directed else 4
        if self.levels is None:
            return list(range(1, size))
        items = list(self.levels) if isinstance(self.levels, (list, tuple)) else [self.levels]
        out = []
        for item in items:
            if isinstance(item, str) and network.directed and item in TRIAD_TYPES:
                out.append(TRIAD_TYPES.index(item))
            elif isinstance(item, int) and not isinstance(item, bool) and 0 <= item < size:
                out.append(item)
            else:
                raise ValueError(f"triadcensus: {item!r} is not a triad type; use the codes 0 to "
                                 f"{size - 1}" + (" or names such as '021D'" if network.directed else ""))
        return out

    def _map(self, network):
        return {t: k for k, t in enumerate(self._types(network))}

    def names(self, network):
        return [f"triadcensus.{TRIAD_TYPES[t] if network.directed else t}" for t in self._types(network)]

    def __repr__(self) -> str:
        return "triadcensus" if self.levels is None else f"triadcensus({self.levels!r})"


def _group(*types):
    return {TRIAD_TYPES.index(t) if isinstance(t, str) else t: 0 for t in types}


class Balance(_Triads):
    """Balanced triads: types 102 and 300 (undirected: 1 or 3 ties)."""

    groups = {True: _group("102", "300"), False: _group(1, 3)}


class Intransitive(_Triads):
    """Intransitive triads: types 111D, 201, 111U, 021C and 030C."""

    directed = True
    groups = {True: _group("111D", "201", "111U", "021C", "030C")}

    def __init__(self):
        warnings.warn(
            "intransitive counts intransitive triads (types 111D, 201, 111U, 021C and 030C), as "
            "R's ergm documents; ergm 4.12 computes intransitive triples (two-paths without a "
            "shortcut) instead, the same as twopath - ttriple. To reproduce ergm's results, use "
            "twopath and ttriple.",
            ErgmDifferenceWarning, stacklevel=4,
        )


class Simmelian(_Triads):
    """Simmelian triads (Krackhardt and Handcock 2007): complete triads, type 300."""

    directed = True
    groups = {True: _group("300")}


class NearSimmelian(_Triads):
    """Near-Simmelian triads: one tie short of complete, type 210."""

    directed = True
    groups = {True: _group("210")}


class SimmelianTies(Term):
    """Ties in at least one Simmelian triad."""

    dyad_independent = False
    triadic = True
    directed = True


class OpenTriad(Term):
    """2-stars minus three times the triangles: the open triads."""

    dyad_independent = False
    triadic = True
    directed = False


class _SupportedTies(Term):
    """Ties with a two-path between their ends (transitiveties, i -> k -> j for
    i -> j; cyclicalties, j -> k -> i); in undirected networks, ties with a
    shared partner. With ``attr``, ties and two-paths whose three vertices match."""

    dyad_independent = False
    triadic = True
    cyclical = False

    def __init__(self, attr=None, levels=None):
        self.attr, self.levels = attr, levels

    @property
    def label(self):
        return self.rust_name if self.attr is None else f"{self.rust_name}.{self.attr}"

    def spec(self, network):
        codes = []
        if self.attr is not None:
            values = _values(network, self.attr)
            codes = _level_codes(values, _select_levels(_sorted_levels(values), self.levels))
        return ("supportedties", [], _chunks([int(self.cyclical)], codes))


class TransitiveTies(_SupportedTies):
    pass


class CyclicalTies(_SupportedTies):
    cyclical = True


_TRAILS = ("RRR", "RRL", "LRR", "LRL")


class ThreeTrail(Term):
    """3-trails: in directed networks, by the directions of their outer steps."""

    dyad_independent = False
    triadic = True

    def __init__(self, keep=None, levels=None):
        self.levels = levels if levels is not None else keep

    def _types(self) -> list[int]:
        if self.levels is None:
            return [1, 2, 3, 4]
        items = list(self.levels) if isinstance(self.levels, (list, tuple)) else [self.levels]
        if all(isinstance(v, str) for v in items):
            unknown = [v for v in items if v not in _TRAILS]
            if unknown:
                raise ValueError(f"threetrail: unknown types {unknown}; they are {', '.join(_TRAILS)}")
            return [_TRAILS.index(v) + 1 for v in items]
        return [i + 1 for i in _select(items, 4, "levels")]

    def names(self, network):
        if not network.directed:
            return ["threetrail"]
        return [f"threetrail.{_TRAILS[t - 1]}" for t in self._types()]

    def spec(self, network):
        return ("threetrail", [], self._types() if network.directed else [])


class LocalTriangle(Term):
    """Triangles whose three pairs are all neighbours in ``x``."""

    dyad_independent = False
    triadic = True

    def __init__(self, x):
        self.x = x

    def names(self, network):
        _, name = _matrix_argument(network, self.x, "localtriangle")
        return [f"localtriangle.{name}" if name else "localtriangle"]

    def spec(self, network):
        m, _ = _matrix_argument(network, self.x, "localtriangle")
        m = ((m != 0) | (m.T != 0)).astype(float)
        return ("localtriangle", m.ravel().tolist(), [])


class _StarsMatch(_Stars):
    """k-stars whose vertices all have the same level of ``attr``."""

    rust = "kstarmatch"

    def __init__(self, k, attr: str, levels=None, mode=None, label=None):
        super().__init__(k)
        self.attr, self.levels, self.mode, self._prefix = attr, levels, mode, label

    degree_dependence = None

    def check(self, network):
        if self.mode is not None and not network.bipartite:
            raise ValueError(f"{self._prefix}: needs a bipartite network")
        super().check(network)

    def names(self, network):
        prefix = self._prefix or self.rust.replace("match", "")
        return [f"{prefix}{k}.{self.attr}" for k in self.ks]

    def spec(self, network):
        values = _values(network, self.attr)
        codes = _level_codes(values, _select_levels(_sorted_levels(values), self.levels))
        mask = [] if self.mode is None else _mode_values(network, self.mode).tolist()
        return (self.rust, [], _chunks(self.ks, codes, mask))

    def __repr__(self) -> str:
        return f"{self._prefix or self.rust.replace('match', '')}({self.ks!r}, attr={self.attr!r})"


class _TrianglesMatch(Term):
    """Triangles (transitive or cyclic triples) whose vertices all have the
    same level of ``attr``, in total or by level."""

    dyad_independent = False
    triadic = True
    kind = 0
    stat = "triangle"

    def __init__(self, attr: str, diff: bool = False, levels=None):
        self.attr, self.diff, self.levels = attr, bool(diff), levels

    def _chosen(self, network):
        values = _values(network, self.attr)
        return values, _select_levels(_sorted_levels(values), self.levels)

    def names(self, network):
        if not self.diff:
            return [f"{self.stat}.{self.attr}"]
        return [f"{self.stat}.{self.attr}.{_level_name(v)}" for v in self._chosen(network)[1]]

    def spec(self, network):
        values, chosen = self._chosen(network)
        return ("trianglesmatch", [], _chunks(_level_codes(values, chosen), [int(self.diff), self.kind]))

    def __repr__(self) -> str:
        return f"{self.stat}(attr={self.attr!r}{', diff=True' if self.diff else ''})"


class TriangleMatch(_TrianglesMatch):
    pass


class TTripleMatch(_TrianglesMatch):
    directed = True
    kind, stat = 1, "ttriple"


class CTripleMatch(_TrianglesMatch):
    directed = True
    kind, stat = 2, "ctriple"


class MutualMatch(Term):
    """Mutual dyads whose vertices match on an attribute (same=), in total or
    by level, or the vertices of each level in mutual dyads (by=)."""

    dyad_independent = False
    directed = True

    def __init__(self, same=None, by=None, diff=False, levels=None):
        if (same is None) == (by is None):
            raise ValueError("mutual: give same= or by=, not both")
        self.attr, self.by, self.diff, self.levels = same if same is not None else by, by is not None, bool(diff), levels

    def _chosen(self, network):
        values = _values(network, self.attr)
        return values, _select_levels(_sorted_levels(values), self.levels)

    def names(self, network):
        if self.by:
            return [f"mutual.by.{self.attr}.{_level_name(v)}" for v in self._chosen(network)[1]]
        if self.diff:
            return [f"mutual.same.{self.attr}.{_level_name(v)}" for v in self._chosen(network)[1]]
        return [f"mutual.{self.attr}"]

    def spec(self, network):
        values, chosen = self._chosen(network)
        return ("mutualmatch", [], _chunks(_level_codes(values, chosen), [int(self.diff), int(self.by)]))

    def __repr__(self) -> str:
        return f"mutual({'by' if self.by else 'same'}={self.attr!r})"


class AsymmetricMatch(_AttributeTerm):
    """Asymmetric dyads whose vertices match on an attribute, in total or by level."""

    dyad_independent = False
    directed = True

    def __init__(self, attr: str, diff: bool = False, levels=None):
        super().__init__(attr)
        self.diff, self.levels = bool(diff), levels

    def _chosen(self, network):
        values = _values(network, self.attr)
        return values, _select_levels(_sorted_levels(values), self.levels)

    def names(self, network):
        if not self.diff:
            return [f"asymmetric.{self.attr}"]
        return [f"asymmetric.{self.attr}.{_level_name(v)}" for v in self._chosen(network)[1]]

    def spec(self, network):
        values, chosen = self._chosen(network)
        return ("asymmetricmatch", [], _chunks(_level_codes(values, chosen), [int(self.diff)]))


class _GwDegreeMatch(Term):
    """Geometrically weighted degree by level of ``attr``, with a fixed decay."""

    dyad_independent = False
    rust = "gwdegreematch"

    def __init__(self, decay: float, attr: str, levels=None, prefix="gwdeg", mode=None, directed=None):
        self.decay, self.attr, self.levels, self.prefix, self.mode = float(decay), attr, levels, prefix, mode
        self.directed = directed

    degree_dependence = None

    def check(self, network):
        if self.mode is not None and not network.bipartite:
            raise ValueError(f"{self.prefix}: needs a bipartite network")
        super().check(network)

    def _chosen(self, network):
        values = _values(network, self.attr)
        pool = values if self.mode is None else [v for v, m in zip(values, network.mode) if m == self.mode]
        return values, _select_levels(_sorted_levels(pool), self.levels)

    def names(self, network):
        return [f"{self.prefix}{self.decay:g}.{self.attr}.{_level_name(v)}" for v in self._chosen(network)[1]]

    def spec(self, network):
        values, chosen = self._chosen(network)
        codes = _level_codes(values, chosen)
        mask = [] if self.mode is None else _mode_values(network, self.mode).tolist()
        if self.mode is not None:
            codes = [c if m == self.mode else -1 for c, m in zip(codes, network.mode)]
        return (self.rust, [self.decay], _chunks(codes, mask))

    def __repr__(self) -> str:
        return _call_repr(self, f"{self.prefix}({self.decay:g}, fixed=True, attr={self.attr!r})")


class _NodeMatchBipartite(_Mode, _AttributeTerm):
    """b1nodematch and b2nodematch with their options (Bomiriya et al. 2023)."""

    dyad_independent = False

    def __init__(self, attr: str, diff=False, alpha=1.0, beta=1.0, by=None, levels=None):
        _AttributeTerm.__init__(self, attr)
        self.diff, self.alpha, self.beta, self.by, self.levels = bool(diff), float(alpha), float(beta), by, levels
        if self.alpha < 1 and self.beta < 1:
            raise ValueError(f"{self.b}nodematch: give alpha or beta, not both")

    def _ends(self, network):
        values = _values(network, self.attr)
        mine = [v for v, m in zip(values, network.mode) if m == self.mode_number]
        chosen = _select_levels(_sorted_levels(mine), self.levels)
        codes = [c if m == self.mode_number else -1 for c, m in zip(_level_codes(values, chosen), network.mode)]
        return chosen, codes

    def _centres(self, network):
        if self.by is None:
            return None, []
        values = _values(network, self.by)
        theirs = [v for v, m in zip(values, network.mode) if m != self.mode_number]
        levels = _sorted_levels(theirs)
        codes = [c if m != self.mode_number else -1 for c, m in zip(_level_codes(values, levels), network.mode)]
        return levels, codes

    def names(self, network):
        base = f"{self.b}nodematch.{self.attr}"
        rows = [""] if not self.diff else [f".{_level_name(v)}" for v in self._ends(network)[0]]
        by, _ = self._centres(network)
        cols = [""] if by is None else [f".{_level_name(v)}" for v in by]
        return [f"{base}{r}{c}" for r in rows for c in cols]

    def spec(self, network):
        _, codes = self._ends(network)
        _, by = self._centres(network)
        mask = (network.mode == self.mode_number).astype(int).tolist()
        return ("nodematchbipartite", [self.alpha, self.beta], _chunks(mask, codes, [int(self.diff)], by))


class _StarMixBipartite(_Mode, _AttributeTerm):
    """k-stars centred on one mode whose ends share a level, by the levels of
    the centre and (``diff``) the ends (b1starmix, b2starmix)."""

    dyad_independent = False

    def __init__(self, k, attr: str, base=None, diff=True):
        _AttributeTerm.__init__(self, attr)
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"starmix: k must be one integer >= 1, not {k!r}")
        self.k, self.base, self.diff = k, base, bool(diff)

    def _layout(self, network):
        values = _values(network, self.attr)
        mine = _sorted_levels([v for v, m in zip(values, network.mode) if m == self.mode_number])
        theirs = _sorted_levels([v for v, m in zip(values, network.mode) if m != self.mode_number])
        if self.diff and self.mode_number == 1:
            cells = [(c, leaf) for leaf in range(len(theirs)) for c in range(len(mine))]  # b1 fastest
        elif self.diff:
            cells = [(c, leaf) for c in range(len(mine)) for leaf in range(len(theirs))]
        else:
            cells = [(c, None) for c in range(len(mine))]
        if self.base:
            dropped = {b - 1 for b in ([self.base] if isinstance(self.base, int) else self.base)}
            cells = [cell for k, cell in enumerate(cells) if k not in dropped]
        return values, mine, theirs, cells

    def names(self, network):
        _, mine, theirs, cells = self._layout(network)
        base = f"{self.b}starmix.{self.k}.{self.attr}"
        if not self.diff:
            return [f"{base}.{_level_name(mine[c])}" for c, _ in cells]
        return [f"{base}.{_level_name(mine[c])}.{_level_name(theirs[leaf])}" for c, leaf in cells]

    def spec(self, network):
        values, mine, theirs, cells = self._layout(network)
        centre = (network.mode == self.mode_number).astype(int).tolist()
        centre_codes = [c if m == self.mode_number else -1 for c, m in zip(_level_codes(values, mine), network.mode)]
        leaf_codes = [c if m != self.mode_number else -1 for c, m in zip(_level_codes(values, theirs), network.mode)]
        mapping = np.full((len(mine), len(theirs)), -1, dtype=np.int64)
        for stat, (c, leaf) in enumerate(cells):
            if leaf is None:
                mapping[c, :] = stat
            else:
                mapping[c, leaf] = stat
        return ("starmix", [], _chunks([self.k], centre, centre_codes, leaf_codes, [len(theirs)],
                                       mapping.ravel().tolist()))


class _TwoStarMixBipartite(_Mode, Term):
    """Two-stars centred on one mode by the level of the centre and the
    (unordered) levels of the two ends (b1twostar, b2twostar)."""

    dyad_independent = False

    def __init__(self, centre_attr: str, leaf_attr=None, base=None, centre_levels=None, leaf_levels=None,
                 levels2=None):
        self.centre_attr, self.leaf_attr = centre_attr, leaf_attr if leaf_attr is not None else centre_attr
        self.centre_levels, self.leaf_levels = centre_levels, leaf_levels
        self.levels2 = levels2 if levels2 is not None else (None if base is None else
                                                            [-b for b in ([base] if isinstance(base, int) else base)])

    def _layout(self, network):
        cv, lv = _values(network, self.centre_attr), _values(network, self.leaf_attr)
        mine = _select_levels(_sorted_levels([v for v, m in zip(cv, network.mode) if m == self.mode_number]),
                              self.centre_levels)
        theirs = _select_levels(_sorted_levels([v for v, m in zip(lv, network.mode) if m != self.mode_number]),
                                self.leaf_levels)
        pairs = [(lo, hi) for hi in range(len(theirs)) for lo in range(hi + 1)]
        cells = [(c, lo, hi) for lo, hi in pairs for c in range(len(mine))]
        if self.levels2 is not None:
            cells = [cells[i] for i in _select(self.levels2, len(cells), "levels2")]
        return cv, lv, mine, theirs, cells

    def names(self, network):
        _, _, mine, theirs, cells = self._layout(network)
        return [f"{self.b}twostar.{self.centre_attr}.{_level_name(mine[c])}.{self.leaf_attr}."
                f"{_level_name(theirs[lo])}.{_level_name(theirs[hi])}" for c, lo, hi in cells]

    def spec(self, network):
        cv, lv, mine, theirs, cells = self._layout(network)
        centre = (network.mode == self.mode_number).astype(int).tolist()
        centre_codes = [c if m == self.mode_number else -1 for c, m in zip(_level_codes(cv, mine), network.mode)]
        leaf_codes = [c if m != self.mode_number else -1 for c, m in zip(_level_codes(lv, theirs), network.mode)]
        n_leaf = len(theirs)
        mapping = np.full(len(mine) * n_leaf * n_leaf, -1, dtype=np.int64)
        for stat, (c, lo, hi) in enumerate(cells):
            mapping[(c * n_leaf + lo) * n_leaf + hi] = stat
        return ("twostarmix", [], _chunks(centre, centre_codes, leaf_codes, [n_leaf], mapping.tolist()))


B1NodeMatch = _in_mode(_NodeMatchBipartite, 1, "B1NodeMatch")
B2NodeMatch = _in_mode(_NodeMatchBipartite, 2, "B2NodeMatch")
B1StarMix, B2StarMix = _in_mode(_StarMixBipartite, 1, "B1StarMix"), _in_mode(_StarMixBipartite, 2, "B2StarMix")
B1TwoStar = _in_mode(_TwoStarMixBipartite, 1, "B1TwoStar")
B2TwoStar = _in_mode(_TwoStarMixBipartite, 2, "B2TwoStar")


class M2Star(TwoPath):
    """Mixed 2-stars: pairs of distinct ties i -> j, j -> k (directed networks), ergm's twopath."""

    directed = True

    @property
    def label(self):
        return "m2star"


class Interaction(Term):
    """Products of the change statistics of dyad-independent terms, as ergm's
    ``a:b``: one statistic per pair, the first term's varying fastest."""

    def __init__(self, left, right):
        self.left, self.right = (list(x) if isinstance(x, (list, tuple)) else as_formula(x).terms for x in (left, right))
        for term in [*self.left, *self.right]:
            if not term.dyad_independent or term.curved or term.is_offset:
                raise ValueError(f"interactions need dyad-independent terms, not {term!r} (as ergm by default)")

    def names(self, network):
        a = [n for t in self.left for n in t.names(network)]
        b = [n for t in self.right for n in t.names(network)]
        return [f"{x}:{y}" for y in b for x in a]

    def check(self, network):
        for term in [*self.left, *self.right]:
            term.check(network)

    def full_spec(self, network):
        children = [t.full_spec(network) for t in [*self.left, *self.right]]
        return ("interact", [], [len(self.left)], children)

    def spec(self, network):
        raise TypeError("an interaction has no flat spec; use full_spec")

    def __repr__(self) -> str:
        def side(terms):
            text = " + ".join(map(repr, terms))
            return f"({text})" if len(terms) > 1 else text

        return f"{side(self.left)}:{side(self.right)}"


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


class Subgraph(Term):
    """Terms evaluated on a subgraph, as ergm's S(): the subgraph induced by
    some vertices, or the bipartite subgraph of the ties between two disjoint
    sets of vertices."""

    degree_dependence = None

    def __init__(self, formula, attrs):
        self.formula = as_formula(formula)
        self.attrs = attrs
        for term in self.formula:
            if isinstance(term, (Offset, BlockOperator)):
                raise ValueError(f"S(): {term!r} can't be inside S(); put offset() outside instead")

    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))
    triadic = property(lambda self: any(t.triadic for t in self.formula))
    curved = property(lambda self: any(t.curved for t in self.formula))

    def _sets(self, network):
        from ._lm import LmError, vertex_sets

        try:
            return vertex_sets(self.attrs, network.attributes, network.n)
        except LmError as e:
            raise ValueError(f"S(): {e}") from None

    def _local(self, network) -> Network:
        """The subgraph, as a network of its own: the first set's vertices,
        then (if bipartite) the second's."""
        tails, heads, _ = self._sets(network)
        vertices = tails if heads is None else np.concatenate([tails, heads])
        position = np.full(network.n, -1)
        position[vertices] = np.arange(len(vertices))
        side = np.zeros(network.n, dtype=int)
        if heads is not None:
            side[heads] = 1

        def inside(pairs):
            if not len(pairs):
                return np.zeros((0, 2), dtype=np.uint32)
            a, b = position[pairs[:, 0].astype(int)], position[pairs[:, 1].astype(int)]
            keep = (a >= 0) & (b >= 0)
            if heads is not None:
                tail_side, head_side = side[pairs[:, 0].astype(int)], side[pairs[:, 1].astype(int)]
                # In a directed network, only the arcs from the first set to the second.
                keep &= (tail_side == 0) & (head_side == 1) if network.directed else tail_side != head_side
            return np.ascontiguousarray(np.column_stack([a[keep], b[keep]]).astype(np.uint32))

        attributes = {k: [v[i] for i in vertices] for k, v in network.attributes.items()}
        graph = {}
        for k, v in network.graph_attributes.items():
            x = np.asarray(v) if isinstance(v, (list, np.ndarray)) else None
            if x is not None and x.shape == (network.n, network.n):
                graph[k] = x[np.ix_(tails, tails)] if heads is None else x[np.ix_(tails, heads)]
            else:
                graph[k] = v
        mode = None
        if heads is not None:
            mode = np.array([1] * len(tails) + [2] * len(heads))
        elif network.mode is not None:
            mode = network.mode[vertices]
        directed = network.directed and heads is None
        return Network(len(vertices), directed, inside(network.edges), attributes, None, graph,
                       inside(network.missing), mode)

    def _label(self, network) -> str:
        return self._sets(network)[2]

    def names(self, network):
        local, label = self._local(network), self._label(network)
        return [f"S({label})~{name}" for t in self.formula for name in t.names(local)]

    def param_names(self, network):
        local, label = self._local(network), self._label(network)
        return [f"S({label})~{name}" for t in self.formula for name in t.param_names(local)]

    def eta(self, params, network):
        local, params = self._local(network), np.asarray(params, dtype=float)
        return np.concatenate([t.eta(params[q], local) for t, _, q in _formula_blocks(self.formula, local)])

    def jacobian(self, params, network):
        local, params = self._local(network), np.asarray(params, dtype=float)
        blocks = list(_formula_blocks(self.formula, local))
        out = np.zeros((blocks[-1][1].stop, blocks[-1][2].stop))
        for t, ps, qs in blocks:
            out[ps, qs] = t.jacobian(params[qs], local)
        return out

    def starts(self, network):
        local = self._local(network)
        return [(qs.start + i, v) for t, _, qs in _formula_blocks(self.formula, local) for i, v in t.starts(local)]

    def check(self, network):
        if network.combined:
            raise ValueError("S() on several networks combined is not supported yet; put it inside N()")
        tails, heads, _ = self._sets(network)
        if heads is not None:
            if np.intersect1d(tails, heads).size:
                raise ValueError("S(): the two sets of vertices must be disjoint")
            if network.bipartite:
                raise ValueError("S(): bipartite subgraphs of bipartite networks are not supported")
        size = len(tails) + (0 if heads is None else len(heads))
        if size < 2 or (heads is not None and (not len(tails) or not len(heads))):
            raise ValueError(f"S(): the subgraph of {self._label(network)} has too few vertices")
        local = self._local(network)
        for term in self.formula:
            term.check(local)

    def full_spec(self, network):
        tails, heads, _ = self._sets(network)
        local = self._local(network)
        vertices = list(tails) + ([] if heads is None else list(heads))
        ints = [int(heads is not None), len(tails), *map(int, vertices)]
        return ("subgraph", [], ints, [t.full_spec(local) for t in self.formula])

    def spec(self, network):
        raise TypeError("S() has no flat spec; use full_spec")

    def __repr__(self) -> str:
        return f"S({self.formula!r}, {self.attrs!r})"


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
        if weights is not None and not (np.isscalar(weights) and str(weights).lstrip("~").strip() in ("1", "1.0")):
            raise NotImplementedError(f"{op}(): network weights other than 1 are not supported, "
                                      "as in ergm.multi")
        if contrasts is not None:
            raise NotImplementedError(f"{op}(): the contrasts argument is not supported yet")
        if label is not None and not (isinstance(label, str) or callable(label)):
            raise TypeError(f"{op}(): label must be a string, or a function of a statistic's name "
                            "and a column of the linear model")
        self.op, self.formula, self.lm = op, as_formula(formula), lm
        self.subset = None if subset is True or (isinstance(subset, str) and
                                                 subset.strip().lstrip("~").strip() == "TRUE") else subset
        self.offset, self.name_label = offset, label
        if not len(self.formula):
            raise ValueError(f"{op}() needs terms")
        for term in self.formula:
            if isinstance(term, (Offset, BlockOperator)):
                raise ValueError(f"{op}(): {term!r} can't be inside {op}(); put offset() outside "
                                 "instead, and don't nest the operators")

    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))
    triadic = property(lambda self: any(t.triadic for t in self.formula))
    _inner_curved = property(lambda self: any(t.curved for t in self.formula))

    @property
    def _has_offset(self) -> bool:
        return self.offset is not None or (self.lm is not None and "offset(" in re.sub(r"\s", "", str(self.lm)))

    @property
    def curved(self) -> bool:
        # An offset adds statistics whose coefficients are fixed at 1.
        return self._inner_curved or self._has_offset

    @property
    def temporal(self) -> bool:
        """Whether the operator needs each network's previous network."""
        return self.op in ("Form", "Persist", "Diss", "Change")

    def _frame(self, attributes: list[dict]):
        """The networks kept by ``subset``, the design matrix (networks x
        columns: 0 for the others) and its columns, and the offsets (None
        without), for networks with these attributes."""
        from ._lm import LmError, evaluate, model_frame, network_subset

        try:
            kept = np.ones(len(attributes), dtype=bool) if self.subset is None \
                else network_subset(self.subset, attributes)
            if not kept.any():
                raise LmError("subset keeps no network")
            chosen = [a for a, k in zip(attributes, kept) if k]
            x, columns, offset = model_frame(self.lm, chosen)
            if self.offset is not None:
                value = evaluate(self.offset, chosen) if isinstance(self.offset, str) \
                    else np.broadcast_to(np.asarray(self.offset, dtype=float), (len(chosen),))
                value = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(value)):
                    raise LmError("the offset is not finite for some networks")
                offset = value if offset is None else offset + value
        except (LmError, ValueError) as e:
            raise ValueError(f"{self.op}(): {e}") from None
        full = np.zeros((len(attributes), x.shape[1]))
        full[kept] = x
        offsets = None
        if offset is not None:
            offsets = np.zeros(len(attributes))
            offsets[kept] = offset
        return kept, full, columns, offsets

    def _design(self, network):
        _, x, columns, _ = self._frame([b.attributes for b in network.blocks])
        return x, columns

    def _kept(self, network) -> np.ndarray:
        return self._frame([b.attributes for b in network.blocks])[0]

    def varies(self, network) -> bool:
        """Whether the terms' coefficients differ between networks."""
        return self.subset is not None or self._has_offset or self._design(network)[1] != ["1"]

    def _inner_names(self, network, params=False) -> list[str]:
        return [n for t in self.formula for n in (t.param_names if params else t.names)(network)]

    def _kept_blocks(self, network) -> list:
        return [b for b, k in zip(network.blocks, self._kept(network)) if k]

    def check(self, network):
        if not network.combined:
            raise ValueError(f"{self!r} needs networks combined with ergmx.Networks() or "
                             "ergmx.NetSeries()")
        if self.temporal and not network.series:
            raise ValueError(f"{self.op}() needs a series of networks: ergmx.NetSeries(), or "
                             "ergmx.tergm()")
        kept = self._kept_blocks(network)
        for block in kept:
            for term in self.formula:
                term.check(block.network)
        counts = {len(self._inner_names(b.network, params=True)) for b in kept}
        if len(counts) > 1:
            raise ValueError(f"{self!r}: the terms have different numbers of parameters in "
                             "different networks (for example, attribute levels that some "
                             "networks lack), which ergm.multi doesn't allow either")
        names = {tuple(self._inner_names(b.network, params=True)) for b in kept}
        if len(names) > 1:
            warnings.warn(f"{self!r}: the terms' parameters have different names in different "
                          "networks, which may indicate specification problems", stacklevel=4)

    def _name(self, name: str, column: str) -> str:
        if callable(self.name_label):
            return str(self.name_label(name, column))
        if self.name_label is not None:
            return f"{self.op}({self.name_label},{column})~{name}"
        return f"{self.op}({column})~{name}"

    def names(self, network):
        if self._inner_curved:
            # N#k whatever the operator, as ergm.multi and tergm name them.
            return [f"N#{k + 1}~{n}" for k, b in enumerate(self._kept_blocks(network))
                    for n in self._inner_names(b.network)]
        _, columns = self._design(network)
        out = []
        for s, n in enumerate(self._inner_names(self._kept_blocks(network)[0].network)):
            out += [self._name(n, c) for c in columns]
            if self._has_offset:
                out.append(f"offset{s + 1}")
        return out

    def param_names(self, network):
        _, columns = self._design(network)
        return [self._name(n, c)
                for n in self._inner_names(self._kept_blocks(network)[0].network, params=True)
                for c in columns]

    def _thetas(self, params, network):
        """The design of the kept networks, and their parameters of the formula."""
        kept, x, columns, offsets = self._frame([b.attributes for b in network.blocks])
        coefficients = np.asarray(params, dtype=float).reshape(-1, len(columns))
        shift = np.zeros(len(x)) if offsets is None else offsets
        return x[kept], [coefficients @ row + o for row, o in zip(x[kept], shift[kept])]

    def eta(self, params, network):
        params = np.asarray(params, dtype=float)
        if not self._inner_curved:
            if not self._has_offset:
                return params
            _, columns = self._design(network)
            per = params.reshape(-1, len(columns))
            return np.column_stack([per, np.ones(len(per))]).ravel()
        _, thetas = self._thetas(params, network)
        return np.concatenate([
            t.eta(theta[qs], b.network)
            for b, theta in zip(self._kept_blocks(network), thetas)
            for t, _, qs in _formula_blocks(self.formula, b.network)
        ])

    def jacobian(self, params, network):
        if not self._inner_curved:
            if not self._has_offset:
                return np.eye(len(params))
            _, columns = self._design(network)
            p = len(columns)
            return np.kron(np.eye(len(params) // p), np.vstack([np.eye(p), np.zeros((1, p))]))
        x, thetas = self._thetas(params, network)
        rows = []
        for b, row, theta in zip(self._kept_blocks(network), x, thetas):
            blocks = list(_formula_blocks(self.formula, b.network))
            inner = np.zeros((blocks[-1][1].stop, blocks[-1][2].stop))
            for t, ps, qs in blocks:
                inner[ps, qs] = t.jacobian(theta[qs], b.network)
            # d eta_k / d coefficient (s, c) = d eta_k / d theta_s * x[k, c].
            rows.append(np.kron(inner, row[None, :]))
        return np.vstack(rows)

    def starts(self, network):
        # The coefficients that predict the formula's starting value in every kept network.
        kept, x, columns, offsets = self._frame([b.attributes for b in network.blocks])
        shift = np.zeros(int(kept.sum())) if offsets is None else offsets[kept]
        first = self._kept_blocks(network)[0].network
        out = []
        for t, _, qs in _formula_blocks(self.formula, first):
            for i, value in t.starts(first):
                b = np.linalg.lstsq(x[kept], value - shift, rcond=None)[0]
                out += [((qs.start + i) * len(columns) + c, float(b[c])) for c in range(len(columns))]
        return out

    def full_spec(self, network):
        kept, x, columns, offsets = self._frame([b.attributes for b in network.blocks])
        compact = not self._inner_curved
        if compact and offsets is not None:
            x = np.column_stack([x, offsets])
        children = [("block", x[k].tolist() if compact else [], [],
                     [t.full_spec(b.network) for t in self.formula] if compact or kept[k] else [])
                    for k, b in enumerate(network.blocks)]
        ints = [_VIEWS[self.op], int(self.op == "Diss"), int(compact), x.shape[1] if compact else 0]
        return ("blocks", [], ints, children)

    def spec(self, network):
        raise TypeError(f"{self.op}() has no flat spec; use full_spec")

    def __repr__(self) -> str:
        extra = "" if self.lm is None else f", lm={self.lm!r}"
        for name, value in (("subset", self.subset), ("offset", self.offset), ("label", self.name_label)):
            if value is not None:
                extra += f", {name}={value!r}"
        return f"{self.op}({self.formula!r}{extra})"


# -- The functions users call, named as in ergm -------------------------------------------------


def edges() -> Term:
    """Number of edges."""
    return Edges()


def mutual(same=None, by=None, diff: bool = False, keep=None, levels=None) -> Term:
    """Number of reciprocated pairs of ties (directed networks). With ``same``,
    only those between vertices with the same value of that attribute (by
    value, with ``diff=TRUE``); with ``by``, the vertices of each value in
    reciprocated pairs."""
    if same is None and by is None:
        return Mutual()
    return MutualMatch(same, by, diff, levels if levels is not None else keep)


def kstar(k, attr=None, levels=None) -> Term:
    """Number of k-stars, for one or more k (undirected networks); with
    ``attr``, only those whose vertices all have the same value."""
    if attr is not None:
        return _with(_StarsMatch(k, attr, levels), directed=False)
    return KStar(k)


def istar(k, attr=None, levels=None) -> Term:
    """Number of in-k-stars: sets of k ties to the same vertex (directed
    networks); with ``attr``, only those whose vertices all have the same value."""
    if attr is not None:
        return _with(_StarsMatch(k, attr, levels), directed=True, rust="istarmatch")
    return IStar(k)


def ostar(k, attr=None, levels=None) -> Term:
    """Number of out-k-stars: sets of k ties from the same vertex (directed
    networks); with ``attr``, only those whose vertices all have the same value."""
    if attr is not None:
        return _with(_StarsMatch(k, attr, levels), directed=True, rust="ostarmatch")
    return OStar(k)


def degree(d, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of vertices with degree exactly d, for one or more d (undirected
    networks). With ``by``, one set of counts per value of that attribute (of
    ``levels``); with ``homophily=TRUE``, degrees count only the ties between
    vertices with the same value."""
    if by is None and not homophily:
        return Degree(d)
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("degrange", "deg", [(k, k + 1) for k in ds], by=by, homophily=homophily,
                        levels=levels, style="exact", directed=False)


def idegree(d, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of vertices with in-degree exactly d, for one or more d (directed
    networks); ``by``, ``homophily`` and ``levels`` as in :func:`degree`."""
    if by is None and not homophily:
        return IDegree(d)
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("idegrange", "ideg", [(k, k + 1) for k in ds], by=by, homophily=homophily,
                        levels=levels, style="exact", directed=True)


def odegree(d, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of vertices with out-degree exactly d, for one or more d (directed
    networks); ``by``, ``homophily`` and ``levels`` as in :func:`degree`."""
    if by is None and not homophily:
        return ODegree(d)
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("odegrange", "odeg", [(k, k + 1) for k in ds], by=by, homophily=homophily,
                        levels=levels, style="exact", directed=True)


def isolates() -> Term:
    """Number of vertices without ties (in either direction, if directed)."""
    return Isolates()


def concurrent(by=None, levels=None) -> Term:
    """Number of vertices with degree 2 or more (undirected networks), by value
    of ``by`` if given."""
    if by is None:
        return Concurrent()
    return _DegreeRange("degrange", "concurrent", [(2, None)], by=by, levels=levels, style="concurrent",
                        directed=False)


def sender(base=1, nodes=-1) -> Term:
    """Each vertex's out-degree, one statistic per vertex of ``nodes`` (by
    default all but the first; directed networks)."""
    return Sender(_nodes(base, nodes))


def receiver(base=1, nodes=-1) -> Term:
    """Each vertex's in-degree, one statistic per vertex of ``nodes`` (by
    default all but the first; directed networks)."""
    return Receiver(_nodes(base, nodes))


def sociality(attr=None, base=1, levels=None, nodes=-1) -> Term:
    """Each vertex's degree, one statistic per vertex of ``nodes`` (by default
    all but the first; undirected networks); with ``attr``, only the ties to
    vertices with the same value (of ``levels``)."""
    return Sociality(_nodes(base, nodes), attr, levels)


def gwdegree(decay: float = 0.5, fixed: bool = False, attr=None, cutoff: int = 30, levels=None) -> Term:
    """Geometrically weighted degree distribution (undirected networks).

    With ``fixed=False`` (the default, as in ergm) the decay is estimated, from
    ``decay``, and the term is curved (see :class:`~ergmx.terms.Curved`); ``fixed=True``
    fixes it. The other geometrically weighted terms work the same way. With
    ``attr`` (and a fixed decay, as in ergm), one statistic per value of ``attr``.
    """
    if attr is not None:
        _fixed_with_attr("gwdegree", fixed)
        return _GwDegreeMatch(decay, attr, levels, "gwdeg", directed=False)
    return _curved_or_fixed(GwDegree(decay), fixed, cutoff)


def gwidegree(decay: float = 0.5, fixed: bool = False, attr=None, cutoff: int = 30, levels=None) -> Term:
    """Geometrically weighted in-degree distribution (directed networks), by
    value of ``attr`` if given (with a fixed decay)."""
    if attr is not None:
        _fixed_with_attr("gwidegree", fixed)
        return _with(_GwDegreeMatch(decay, attr, levels, "gwideg", directed=True), rust="gwidegreematch")
    return _curved_or_fixed(GwIDegree(decay), fixed, cutoff)


def gwodegree(decay: float = 0.5, fixed: bool = False, attr=None, cutoff: int = 30, levels=None) -> Term:
    """Geometrically weighted out-degree distribution (directed networks), by
    value of ``attr`` if given (with a fixed decay)."""
    if attr is not None:
        _fixed_with_attr("gwodegree", fixed)
        return _with(_GwDegreeMatch(decay, attr, levels, "gwodeg", directed=True), rust="gwodegreematch")
    return _curved_or_fixed(GwODegree(decay), fixed, cutoff)


def cycle(k, semi: bool = False) -> Term:
    """Number of cycles of length k, for one or more k: 3 or more in undirected
    networks, 2 or more in directed ones (cycle(2) is mutual)."""
    return Cycle(k, semi)


def twopath() -> Term:
    """Number of 2-paths: i -> j -> k with i != k if directed, kstar(2) if undirected."""
    return TwoPath()


def asymmetric(attr=None, diff: bool = False, keep=None, levels=None) -> Term:
    """Number of pairs with a tie in one direction only (directed networks);
    with ``attr``, only pairs of vertices with the same value (by value, with
    ``diff=TRUE``)."""
    if attr is None:
        return Asymmetric()
    return AsymmetricMatch(attr, diff, levels if levels is not None else keep)


def transitive() -> Term:
    """Number of transitive triads (directed networks): those with at least one
    transitive triple and no intransitive two-path, the types 030T, 120D, 120U
    and 300 of `Davis and Leinhardt's (1972) <https://scholar.google.com/scholar?q=%22The+structure+of+positive+interpersonal+relations+in+small+groups%22+Davis+Leinhardt>`__ triad census.

    This is how R's ergm documents its ``transitive`` term, but ergm 4.12
    computes the number of transitive triples instead, the same as
    :func:`ttriple`. Use ``ttriple`` to reproduce ergm's results; using
    ``transitive`` warns with :class:`ErgmDifferenceWarning`.
    """
    return Transitive()


def triangle(attr=None, diff: bool = False, levels=None) -> Term:
    """Number of triangles; in directed networks, transitive plus cyclic
    triples. With ``attr``, only triangles whose vertices all have the same
    value, in total or (``diff=TRUE``) by value."""
    if attr is not None:
        return TriangleMatch(attr, diff, levels)
    return Triangle()


def ttriple(attr=None, diff: bool = False, levels=None) -> Term:
    """Number of transitive triples i -> j -> k with i -> k (directed networks);
    ``attr`` and ``diff`` as in :func:`triangle`."""
    if attr is not None:
        return TTripleMatch(attr, diff, levels)
    return TTriple()


def ctriple(attr=None, diff: bool = False, levels=None) -> Term:
    """Number of cyclic triples i -> j -> k -> i (directed networks); ``attr``
    and ``diff`` as in :func:`triangle`."""
    if attr is not None:
        return CTripleMatch(attr, diff, levels)
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


def nodematch(attr: str, diff: bool = False, keep=None, levels=None) -> Term:
    """Number of ties between vertices with the same value of ``attr``; with
    ``diff=True``, one statistic per value. ``levels`` (or the older ``keep``)
    selects the values counted."""
    return NodeMatch(attr, diff, levels if levels is not None else keep)


def nodemix(attr: str, levels=None, levels2=-1) -> Term:
    """Number of ties for each mixing type: each pair of levels of ``attr``
    (from sender to receiver, if directed).

    ``levels`` selects levels: names, or R-style 1-based indices into the
    sorted levels (negative to leave out). ``levels2`` selects mixing types, in
    ergm's order: ``-1`` (the default) all but the first, ``TRUE`` all, indices,
    or a levels x levels logical matrix (rows: senders).
    """
    return NodeMix(attr, levels, levels2)


def nodefactor(attr: str, base=1, levels=-1) -> Term:
    """Number of tie endpoints at each level of ``attr`` of ``levels`` (by
    default, all but the first; ``levels=TRUE`` for all)."""
    return NodeFactor(attr, _base_levels(base, levels))


def nodeifactor(attr: str, base=1, levels=-1) -> Term:
    """Number of ties received by vertices at each level of ``attr`` of ``levels``
    (by default, all but the first)."""
    return NodeIFactor(attr, _base_levels(base, levels))


def nodeofactor(attr: str, base=1, levels=-1) -> Term:
    """Number of ties sent by vertices at each level of ``attr`` of ``levels``
    (by default, all but the first)."""
    return NodeOFactor(attr, _base_levels(base, levels))


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


def b1star(k, attr=None, levels=None) -> Term:
    """Number of k-stars centred on first-mode vertices, for one or more k;
    with ``attr``, only those whose vertices all have the same value."""
    if attr is not None:
        return _bipartite_with(_StarsMatch(k, attr, levels, mode=1, label="b1star"))
    return B1Star(k)


def b2star(k, attr=None, levels=None) -> Term:
    """Number of k-stars centred on second-mode vertices, for one or more k;
    with ``attr``, only those whose vertices all have the same value."""
    if attr is not None:
        return _bipartite_with(_StarsMatch(k, attr, levels, mode=2, label="b2star"))
    return B2Star(k)


def b1degree(d, by=None, levels=None) -> Term:
    """Number of first-mode vertices with degree exactly d, for one or more d,
    by value of ``by`` if given."""
    if by is None:
        return B1Degree(d)
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("degrange", "b1deg", [(k, k + 1) for k in ds], by=by, levels=levels, mode=1,
                        style="exact", directed=False)


def b2degree(d, by=None, levels=None) -> Term:
    """Number of second-mode vertices with degree exactly d, for one or more d,
    by value of ``by`` if given."""
    if by is None:
        return B2Degree(d)
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("degrange", "b2deg", [(k, k + 1) for k in ds], by=by, levels=levels, mode=2,
                        style="exact", directed=False)


def gwb1degree(decay: float = 0.5, fixed: bool = False, attr=None, cutoff: int = 30, levels=None) -> Term:
    """Geometrically weighted degree distribution of the first mode, by value of
    ``attr`` if given (with a fixed decay)."""
    if attr is not None:
        _fixed_with_attr("gwb1degree", fixed)
        return _bipartite_with(_GwDegreeMatch(decay, attr, levels, "gwb1deg", mode=1, directed=False))
    return _curved_or_fixed(GwB1Degree(decay), fixed, cutoff)


def gwb2degree(decay: float = 0.5, fixed: bool = False, attr=None, cutoff: int = 30, levels=None) -> Term:
    """Geometrically weighted degree distribution of the second mode, by value of
    ``attr`` if given (with a fixed decay)."""
    if attr is not None:
        _fixed_with_attr("gwb2degree", fixed)
        return _bipartite_with(_GwDegreeMatch(decay, attr, levels, "gwb2deg", mode=2, directed=False))
    return _curved_or_fixed(GwB2Degree(decay), fixed, cutoff)


def b1concurrent(by=None, levels=None) -> Term:
    """Number of first-mode vertices with degree 2 or more, by value of ``by`` if given."""
    if by is None:
        return B1Concurrent()
    return _DegreeRange("degrange", "b1concurrent", [(2, None)], by=by, levels=levels, mode=1,
                        style="concurrent", directed=False)


def b2concurrent(by=None, levels=None) -> Term:
    """Number of second-mode vertices with degree 2 or more, by value of ``by`` if given."""
    if by is None:
        return B2Concurrent()
    return _DegreeRange("degrange", "b2concurrent", [(2, None)], by=by, levels=levels, mode=2,
                        style="concurrent", directed=False)


def b1factor(attr: str, base=1, levels=-1) -> Term:
    """For each level of ``attr`` among first-mode vertices of ``levels`` (by
    default, all but the first), their ties."""
    return B1Factor(attr, _base_levels(base, levels))


def b2factor(attr: str, base=1, levels=-1) -> Term:
    """For each level of ``attr`` among second-mode vertices of ``levels`` (by
    default, all but the first), their ties."""
    return B2Factor(attr, _base_levels(base, levels))


def b1cov(attr: str) -> Term:
    """Sum over ties of the first-mode endpoint's value of the numeric ``attr``."""
    return B1Cov(attr)


def b2cov(attr: str) -> Term:
    """Sum over ties of the second-mode endpoint's value of the numeric ``attr``."""
    return B2Cov(attr)


def b1nodematch(attr: str, diff: bool = False, keep=None, alpha: float = 1, beta: float = 1, byb2attr=None,
                levels=None) -> Term:
    """Number of 2-stars centred on second-mode vertices whose two first-mode
    ends have the same value of ``attr`` (Bomiriya et al. 2023), by value with
    ``diff=TRUE`` and by value of the centres' ``byb2attr``. ``beta`` < 1
    discounts each tie's two-stars, half their number to the power beta;
    ``alpha`` < 1 counts each pair of matching ends' shared partners to the
    power alpha."""
    return B1NodeMatch(attr, diff, alpha, beta, byb2attr, levels if levels is not None else keep)


def b2nodematch(attr: str, diff: bool = False, keep=None, alpha: float = 1, beta: float = 1, byb1attr=None,
                levels=None) -> Term:
    """Number of 2-stars centred on first-mode vertices whose two second-mode
    ends have the same value of ``attr``; the options as in :func:`b1nodematch`."""
    return B2NodeMatch(attr, diff, alpha, beta, byb1attr, levels if levels is not None else keep)


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

    ``subset`` keeps some networks only: an R expression of their attributes
    (``"~n >= 4"``), logical values (recycled) or 1-based indices; the
    others contribute nothing to the terms, and the linear model's factor
    levels are those of the kept networks. ``offset`` adds a known amount to
    every coefficient of the formula in each network (an R expression, such
    as ``"~log(n)"``, or numbers), as do ``offset()`` terms in ``lm``: each
    statistic then gets an extra statistic, ``offset1``, ``offset2``...,
    whose coefficient is fixed at 1, as in ergm.multi. ``label`` names the
    operator in the statistics' names (``N(label,1)~edges``), or, a function
    of a statistic's name and a column of the linear model, names them.
    ``weights`` other than 1 and ``contrasts`` are not supported, as in
    ergm.multi.
    """
    return BlockOperator("N", formula, lm, subset, weights, contrasts, offset, label)


_TEMPORAL_DOC = """``lm``, as in :func:`N`, makes the coefficients vary between
    transitions, with the attributes ``.Time``, ``.TimeID`` and ``.TimeDelta``
    as well as the networks' own (see :func:`ergmx.NetSeries`); ``subset``,
    ``offset`` and ``label`` are those of :func:`N`."""


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


def S(formula, attrs) -> Term:  # noqa: N802 (ergm's name)
    """Evaluate ``formula`` on a subgraph, as ergm's S().

    ``attrs`` is an R formula of the vertex attributes, as a string. One-sided,
    it picks the vertices of an induced subgraph: ``"~level == 'individual'"``
    gives the network among individuals (directed if the network is).
    Two-sided, it picks two disjoint sets, and the formula is evaluated on the
    undirected bipartite network of the ties between them (in a directed
    network, of the arcs from the first set to the second, as ergm), whose
    first mode (``b1`` terms) is the left-hand set: ``"(level == 'individual')
    ~ (level == 'organization')"``. Each side may also be a boolean array or 1-based
    vertex indices. In a formula string: ``"S(~edges + gwesp(0.5, fixed=TRUE),
    ~level == 'individual')"``. Names: ``S(level=="individual")~edges``, as in
    ergm.
    """
    return Subgraph(formula, attrs)


_LEVEL_DOC = """For two-level (multilevel) networks, as MPNet (`Wang et al. 2013 <https://doi.org/10.1016/j.socnet.2013.01.004>`__): the
    levels are the values of the vertex attribute ``attr``, A and B (``levels=(A,
    B)``, or the attribute's two values, sorted); A-ties are within A, B-ties
    within B, X-ties between them. Undirected networks."""
_DIRECTED_DOC = """For directed two-level networks, as MPNet: the levels are the values
    of the vertex attribute ``attr``, A and B (``levels=(A, B)``, or the
    attribute's two values, sorted); A-ties and B-ties are arcs within each
    level, and X-ties (affiliations) the arcs from an A vertex to a B vertex
    (fix the dyads from B to A with ``blocks()``). in(v) and out(v) are a
    vertex's in- and out-degrees within its level, x(v) its X-ties."""
_ALT_DOC = """``decay`` weights the alternating statistic geometrically, as
    gwesp's: g(d) = exp(decay) (1 - (1 - exp(-decay))^d), MPNet's lambda being
    exp(decay) (its default, 2, is ``decay=log(2)``). It is fixed by default,
    as in MPNet; with ``fixed=FALSE``, it is estimated, as ergm's curved terms
    (see :class:`~ergmx.terms.Curved`), for the terms with one alternating
    part."""
_LOG2 = float(np.log(2.0))


def star2ax(attr: str, levels=None) -> Term:
    """Star2AX: 2-stars of an A-tie and an X-tie, the sum over A vertices of
    their A-degree times their X-degree. {level_doc}"""
    return Star2AX(attr, levels)


def star2bx(attr: str, levels=None) -> Term:
    """Star2BX: 2-stars of a B-tie and an X-tie, the sum over B vertices of
    their B-degree times their X-degree. {level_doc}"""
    return Star2BX(attr, levels)


def axs1a(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """AXS1A: alternating X-stars with one A-tie, the sum over A vertices of
    their A-degree times g(X-degree). {level_doc} {alt_doc}"""
    return _alternating(AXS1A(attr, levels, decay), fixed, cutoff)


def axs1b(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """AXS1B: alternating X-stars with one B-tie, the sum over B vertices of
    their B-degree times g(X-degree). {level_doc} {alt_doc}"""
    return _alternating(AXS1B(attr, levels, decay), fixed, cutoff)


def aas1x(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """AAS1X: alternating A-stars with one X-tie, the sum over A vertices of
    g(A-degree) times their X-degree. {level_doc} {alt_doc}"""
    return _alternating(AAS1X(attr, levels, decay), fixed, cutoff)


def abs1x(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """ABS1X: alternating B-stars with one X-tie, the sum over B vertices of
    g(B-degree) times their X-degree. {level_doc} {alt_doc}"""
    return _alternating(ABS1X(attr, levels, decay), fixed, cutoff)


def aaaxs(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """AAAXS: alternating A-stars and alternating X-stars, the sum over A
    vertices of g(A-degree) g(X-degree). {level_doc} {alt_doc}"""
    return _alternating(AAAXS(attr, levels, decay), fixed, cutoff)


def abaxs(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """ABAXS: alternating B-stars and alternating X-stars, the sum over B
    vertices of g(B-degree) g(X-degree). {level_doc} {alt_doc}"""
    return _alternating(ABAXS(attr, levels, decay), fixed, cutoff)


def txax(attr: str, levels=None) -> Term:
    """TXAX: triangles of an A-tie and two X-ties to a common B vertex, the sum
    over A-ties of their endpoints' shared B partners. {level_doc}"""
    return TXAX(attr, levels)


def txbx(attr: str, levels=None) -> Term:
    """TXBX: triangles of a B-tie and two X-ties to a common A vertex. {level_doc}"""
    return TXBX(attr, levels)


def atxax(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """ATXAX: alternating TXAX triangles, the sum over A-ties of g(shared B
    partners), as gwesp with partners in B. {level_doc} {alt_doc}"""
    return _alternating(ATXAX(attr, levels, decay), fixed, cutoff)


def atxbx(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """ATXBX: alternating TXBX triangles, the sum over B-ties of g(shared A
    partners). {level_doc} {alt_doc}"""
    return _alternating(ATXBX(attr, levels, decay), fixed, cutoff)


def l3xax(attr: str, levels=None) -> Term:
    """L3XAX: three-paths of an X-tie, an A-tie and an X-tie, the sum over
    A-ties of the product of their endpoints' X-degrees (so closed paths, the
    TXAX triangles, count too, as Wang et al. say: "the TXAX configuration is
    also part of L3XAX"). In directed networks, over A-arcs (MPNet's directed
    L3XAX). {level_doc}"""
    return L3XAX(attr, levels)


def l3xbx(attr: str, levels=None) -> Term:
    """L3XBX: three-paths of an X-tie, a B-tie and an X-tie, the sum over
    B-ties of the product of their endpoints' X-degrees. {level_doc}"""
    return L3XBX(attr, levels)


def l3axb(attr: str, levels=None) -> Term:
    """L3AXB: cross-level three-paths of an A-tie, an X-tie and a B-tie, the
    sum over X-ties of the A-degree of their A end times the B-degree of their
    B end. {level_doc}"""
    return L3AXB(attr, levels)


def c4axb(attr: str, levels=None) -> Term:
    """C4AXB: cross-level 4-cycles of an A-tie, a B-tie and the two X-ties
    that join their ends (alignment between the levels). {level_doc}"""
    return C4AXB(attr, levels)


def exta(attr: str, levels=None) -> Term:
    """EXTA: an A-triangle with an X-tie at one of its vertices, the sum over A
    vertices of their A-triangles times their X-degree. {level_doc}"""
    return EXTA(attr, levels)


def extb(attr: str, levels=None) -> Term:
    """EXTB: a B-triangle with an X-tie at one of its vertices. {level_doc}"""
    return EXTB(attr, levels)


def asaxasb(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
    """ASAXASB: alternating A-stars and alternating B-stars joined by an X-tie,
    the sum over X-ties (a, b) of g(A-degree of a) g(B-degree of b). {level_doc} {alt_doc}"""
    return _alternating(ASAXASB(attr, levels, decay), fixed, cutoff)


def _directed_term(mpnet: str, doc: str):
    cls = _DIRECTED_MULTILEVEL[mpnet]
    if cls.alternating:
        def make(attr: str, decay: float = _LOG2, levels=None, fixed: bool = True, cutoff: int = 30) -> Term:
            return _alternating(cls(attr, levels, decay), fixed, cutoff)
    else:
        def make(attr: str, levels=None) -> Term:
            return cls(attr, levels)
    make.__name__ = mpnet.lower()
    make.__doc__ = f"{mpnet}: {doc} {{directed_doc}}" + (" {alt_doc}" if cls.alternating else "")
    return make


_DIRECTED_DOCS = {
    "In2StarAX": "sum over A vertices of in(v) x(v): an incoming A-tie with an X-tie.",
    "In2StarBX": "sum over B vertices of in(v) x(v).",
    "Out2StarAX": "sum over A vertices of out(v) x(v): an outgoing A-tie with an X-tie.",
    "Out2StarBX": "sum over B vertices of out(v) x(v).",
    "AXS1Ain": "alternating X-stars with one incoming A-tie, the sum over A vertices of in(v) g(x(v)).",
    "AXS1Bin": "the sum over B vertices of in(v) g(x(v)).",
    "AXS1Aout": "alternating X-stars with one outgoing A-tie, the sum over A vertices of out(v) g(x(v)).",
    "AXS1Bout": "the sum over B vertices of out(v) g(x(v)).",
    "AAinS1X": "alternating A-in-stars with one X-tie, the sum over A vertices of g(in(v)) x(v).",
    "ABinS1X": "the sum over B vertices of g(in(v)) x(v).",
    "AAoutS1X": "alternating A-out-stars with one X-tie, the sum over A vertices of g(out(v)) x(v).",
    "ABoutS1X": "the sum over B vertices of g(out(v)) x(v).",
    "TXAXarc": "over A-arcs, the number of B vertices X-tied to both ends.",
    "TXBXarc": "over B-arcs, the number of A vertices X-tied to both ends.",
    "TXAXreciprocity": "over reciprocated pairs of A-arcs, the number of B vertices X-tied to both.",
    "TXBXreciprocity": "over reciprocated pairs of B-arcs, the number of A vertices X-tied to both.",
    "ATXAXarc": "over A-arcs, g(the B vertices X-tied to both ends).",
    "ATXBXarc": "over B-arcs, g(the A vertices X-tied to both ends).",
    "ATXAXreciprocity": "over reciprocated pairs of A-arcs, g(the B vertices X-tied to both).",
    "ATXBXreciprocity": "over reciprocated pairs of B-arcs, g(the A vertices X-tied to both).",
    "L3XAXreciprocity": "over reciprocated pairs of A-arcs, the product of the ends' X-degrees.",
    "L3XBXreciprocity": "over reciprocated pairs of B-arcs, the product of the ends' X-degrees.",
    "L3AXBin": "over X-ties a -> b, in(a) in(b): both ends receive within their level.",
    "L3AXBout": "over X-ties a -> b, out(a) out(b): both ends send within their level.",
    "L3AXBpath": "over X-ties a -> b, in(a) out(b): a path from A through X into B.",
    "L3BXApath": "over X-ties a -> b, out(a) in(b): a path from B through X into A.",
    "C4AXBentrainment": "4-cycles of an A-arc u -> v, a B-arc w -> z and the X-ties u -> w and v -> z: the arcs aligned.",
    "C4AXBexchange": "4-cycles of an A-arc u -> v, a B-arc w -> z and the X-ties u -> z and v -> w: the arcs opposed.",
    "C4AXBexchangeAreciprocity": "4-cycles of a reciprocated pair of A-arcs, a B-arc and two X-ties.",
    "C4AXBexchangeBreciprocity": "4-cycles of an A-arc, a reciprocated pair of B-arcs and two X-ties.",
    "C4AXBreciprocity": "4-cycles of reciprocated pairs of A- and B-arcs and two X-ties.",
    "AinASXAinBS": "over X-ties a -> b, g(in(a)) g(in(b)): alternating in-stars at both ends.",
    "AoutASXAoutBS": "over X-ties a -> b, g(out(a)) g(out(b)).",
    "AinASXAoutBS": "over X-ties a -> b, g(in(a)) g(out(b)).",
    "AoutASXAinBS": "over X-ties a -> b, g(out(a)) g(in(b)).",
}
_DIRECTED_FUNCTIONS = tuple(_directed_term(name, doc) for name, doc in _DIRECTED_DOCS.items())
for _f in _DIRECTED_FUNCTIONS:
    globals()[_f.__name__] = _f


_MULTILEVEL = (star2ax, star2bx, axs1a, axs1b, aas1x, abs1x, aaaxs, abaxs, txax, txbx, atxax, atxbx,
               l3xax, l3xbx, l3axb, c4axb, exta, extb, asaxasb, *_DIRECTED_FUNCTIONS)
for _f in _MULTILEVEL:
    _f.__doc__ = inspect.cleandoc(_f.__doc__).replace("{level_doc}", inspect.cleandoc(_LEVEL_DOC)) \
        .replace("{alt_doc}", inspect.cleandoc(_ALT_DOC)).replace("{directed_doc}", inspect.cleandoc(_DIRECTED_DOC))


def edgecov(x) -> Term:
    """Sum over ties of a dyadic covariate.

    ``x`` is the name of a graph attribute holding an n x n matrix (in a
    formula string: ``"edgecov('trade')"``), an n x n array, or an igraph
    graph on the same vertices. Undirected networks use the upper triangle.
    """
    return EdgeCov(x)


def _with(term: Term, **attributes) -> Term:
    """The term with some attributes set (the variant of a shared class)."""
    for k, v in attributes.items():
        setattr(term, k, v)
    return term


def _bipartite_with(term: Term) -> Term:
    return term


def _nodes(base, nodes):
    """ergm's deprecated base= as nodes=: base=k leaves out vertex k (0: none)."""
    if base != 1 and nodes == -1:
        return True if base == 0 else -base
    return nodes


def _base_levels(base, levels):
    """ergm's deprecated base= as levels=: base=k leaves out level k (0: none)."""
    if base != 1 and levels == -1:
        return True if base == 0 else -base
    return levels


def _fixed_with_attr(name: str, fixed: bool) -> None:
    if not fixed:
        raise NotImplementedError(f"{name}: with attr, the decay must be fixed (fixed=TRUE), as in ergm")


_INF = float("inf")


def degrange(frm, to=_INF, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of vertices with degree in [from, to), for each pair (undirected
    networks); ``to`` defaults to infinity, and either can be recycled. ``by``,
    ``homophily`` and ``levels`` as in :func:`degree`. In R: ``degrange(from, to)``."""
    return _DegreeRange("degrange", "deg", _ranges(frm, to), by=by, homophily=homophily, levels=levels,
                        directed=False)


def idegrange(frm, to=_INF, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of vertices with in-degree in [from, to) (directed networks), as :func:`degrange`."""
    return _DegreeRange("idegrange", "ideg", _ranges(frm, to), by=by, homophily=homophily, levels=levels,
                        directed=True)


def odegrange(frm, to=_INF, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of vertices with out-degree in [from, to) (directed networks), as :func:`degrange`."""
    return _DegreeRange("odegrange", "odeg", _ranges(frm, to), by=by, homophily=homophily, levels=levels,
                        directed=True)


def b1degrange(frm, to=_INF, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of first-mode vertices with degree in [from, to), as :func:`degrange`."""
    return _DegreeRange("degrange", "b1deg", _ranges(frm, to), by=by, homophily=homophily, levels=levels,
                        mode=1, directed=False)


def b2degrange(frm, to=_INF, by=None, homophily: bool = False, levels=None) -> Term:
    """Number of second-mode vertices with degree in [from, to), as :func:`degrange`."""
    return _DegreeRange("degrange", "b2deg", _ranges(frm, to), by=by, homophily=homophily, levels=levels,
                        mode=2, directed=False)


def b1mindegree(d) -> Term:
    """Number of first-mode vertices with degree at least d, for one or more d."""
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("degrange", "b1mindeg", [(k, None) for k in ds], mode=1, style="min", directed=False)


def b2mindegree(d) -> Term:
    """Number of second-mode vertices with degree at least d, for one or more d."""
    ds = [d] if isinstance(d, int) else list(d)
    return _DegreeRange("degrange", "b2mindeg", [(k, None) for k in ds], mode=2, style="min", directed=False)


def degree1_5() -> Term:
    """Sum over vertices of their degree to the power 3/2 (undirected
    networks). In a formula string, R's name ``degree1.5`` works too."""
    return _DegreePower("degreepower", "degree1.5", False)


def idegree1_5() -> Term:
    """Sum over vertices of their in-degree to the power 3/2 (directed networks; ``idegree1.5``)."""
    return _DegreePower("idegreepower", "idegree1.5", True)


def odegree1_5() -> Term:
    """Sum over vertices of their out-degree to the power 3/2 (directed networks; ``odegree1.5``)."""
    return _DegreePower("odegreepower", "odegree1.5", True)


def concurrentties(by=None, levels=None) -> Term:
    """Sum over vertices of their ties beyond the first (undirected networks),
    by value of ``by`` if given."""
    return ConcurrentTies(by, levels)


def density() -> Term:
    """The density: the number of edges over the number of dyads."""
    return Density()


def meandeg() -> Term:
    """The mean degree: twice the number of edges over the number of vertices
    (the number of edges, if directed)."""
    return MeanDeg()


def isolatededges() -> Term:
    """Number of ties whose two vertices have no other tie (undirected networks)."""
    return IsolatedEdges()


def dyadcov(x) -> Term:
    """A dyadic covariate by dyad state, in directed networks: its sum over
    mutual dyads (``mutual``), over dyads with only the tie from the lower- to
    the higher-numbered vertex (``utri``, in the adjacency matrix's upper
    triangle), and the reverse (``ltri``); of ``x``, its upper triangle, as
    ergm. In undirected networks, the same as :func:`edgecov`.

    This is how R's ergm documents dyadcov, but ergm 4.12 swaps utri and
    ltri; using dyadcov on a directed network warns with
    :class:`ErgmDifferenceWarning`.
    """
    return DyadCov(x)


def hamming(x=None, cov=None) -> Term:
    """The Hamming distance to a reference network ``x``: the number of dyads
    whose tie differs. ``x`` is the observed network by default, or a graph
    attribute holding an adjacency matrix, the matrix or a graph; ``cov``
    weights the dyads."""
    return Hamming(x, cov)


def attrcov(attr: str, mat) -> Term:
    """Sum over ties of a covariate of the mixing type of their vertices: the
    entry of ``mat`` (levels x levels of ``attr``, sorted) of the pair of levels."""
    return AttrCov(attr, mat)


def mm(attrs, levels=None, levels2=-1) -> Term:
    """The cells of a mixing matrix, as ergm's mm(): ``"A"`` (or ``"~A"``) for
    attribute A with itself, ``"A~B"`` for rows of A and columns of B (from
    senders to receivers, if directed), and ``"A~."`` or ``".~B"`` for its
    margins. ``levels`` selects the levels of the attributes, ``levels2`` the
    cells (by default all but the first)."""
    return MixingMatrix(attrs, levels, levels2)


def nodecovrange(attr: str) -> Term:
    """Sum over vertices of the range of ``attr`` over their neighbours (in
    directed networks, over the out-neighbours plus over the in-neighbours;
    Hoffman, Block and Snijders 2023)."""
    return NodeCovRange(attr)


def nodeicovrange(attr: str) -> Term:
    """Sum over vertices of the range of ``attr`` over their in-neighbours (directed networks)."""
    return NodeICovRange(attr)


def nodeocovrange(attr: str) -> Term:
    """Sum over vertices of the range of ``attr`` over their out-neighbours (directed networks)."""
    return NodeOCovRange(attr)


def b1covrange(attr: str) -> Term:
    """Sum over first-mode vertices of the range of ``attr`` over their neighbours."""
    return B1CovRange(attr)


def b2covrange(attr: str) -> Term:
    """Sum over second-mode vertices of the range of ``attr`` over their neighbours."""
    return B2CovRange(attr)


def nodefactordistinct(attr: str, levels=True) -> Term:
    """Sum over vertices of the number of distinct values of ``attr`` among their
    neighbours (in either direction, if directed)."""
    return NodeFactorDistinct(attr, levels)


def nodeofactordistinct(attr: str, levels=True) -> Term:
    """Sum over vertices of the number of distinct values of ``attr`` among their out-neighbours."""
    return NodeOFactorDistinct(attr, levels)


def nodeifactordistinct(attr: str, levels=True) -> Term:
    """Sum over vertices of the number of distinct values of ``attr`` among their in-neighbours."""
    return NodeIFactorDistinct(attr, levels)


def b1factordistinct(attr: str, levels=True) -> Term:
    """Sum over first-mode vertices of the number of distinct values of ``attr`` among their neighbours."""
    return B1FactorDistinct(attr, levels)


def b2factordistinct(attr: str, levels=True) -> Term:
    """Sum over second-mode vertices of the number of distinct values of ``attr`` among their neighbours."""
    return B2FactorDistinct(attr, levels)


def diff(attr: str, pow: float = 1, dir: str = "t-h", sign_action: str = "identity") -> Term:
    """Sum over ties of a function of the difference of the vertices' values of
    ``attr``: tail minus head (``dir="t-h"``, also ``"b1-b2"``) or head minus
    tail (``"h-t"``, ``"b2-b1"``), transformed by ``sign_action`` (R's
    ``sign.action``: ``"identity"``, ``"abs"``, ``"posonly"``, ``"negonly"``)
    and raised to ``pow`` (the sign, for ``pow=0``). Undirected ties go from
    the lower- to the higher-numbered vertex, bipartite ties from the first mode."""
    return Diff(attr, pow, dir, sign_action)


def smalldiff(attr: str, cutoff: float) -> Term:
    """Number of ties whose vertices' values of ``attr`` differ by less than ``cutoff``."""
    return SmallDiff(attr, cutoff)


def altkstar(lambda_: float, fixed: bool = False) -> Term:
    """Alternating k-stars (Snijders et al. 2006) with weight ``lambda``
    (undirected networks): sum over vertices of lambda^2 ((1 - 1/lambda)^d - 1
    + d / lambda). Only with ``fixed=TRUE``: ergm's estimated version is not
    the same statistic, and ergm recommends :func:`gwdegree`, with edges the
    same model."""
    if not fixed:
        raise NotImplementedError("altkstar: only fixed=TRUE is supported; use gwdegree(decay) with an "
                                  "estimated decay, which ergm recommends (with edges, the same model)")
    return AltKStar(lambda_)


def b1sociality(nodes=-1) -> Term:
    """Each first-mode vertex's degree, one statistic per vertex of ``nodes``
    (indices among the first mode's vertices; by default all but the first)."""
    return B1Sociality(nodes)


def b2sociality(nodes=-1) -> Term:
    """Each second-mode vertex's degree, one statistic per vertex of ``nodes``
    (indices among the second mode's vertices; by default all but the first)."""
    return B2Sociality(nodes)


def triadcensus(levels=None) -> Term:
    """The triad census: the number of triads of each type of `Davis and Leinhardt (1972) <https://scholar.google.com/scholar?q=%22The+structure+of+positive+interpersonal+relations+in+small+groups%22+Davis+Leinhardt>`__,
    by default all but the empty one (directed networks: types 012 to 300,
    or their codes 1 to 15; undirected networks: triads with 1, 2 or 3 ties).
    ``levels`` selects types by code (0 to 15) or name (``"021D"``)."""
    return TriadCensus(levels)


def balance() -> Term:
    """Number of balanced triads: types 102 and 300 (undirected networks:
    triads with 1 or 3 ties)."""
    return Balance()


def intransitive() -> Term:
    """Number of intransitive triads (directed networks): types 111D, 201,
    111U, 021C and 030C of the triad census.

    This is how R's ergm documents its ``intransitive`` term, but ergm 4.12
    computes intransitive triples instead (two-paths i -> j -> k without i ->
    k), the same as ``twopath`` minus ``ttriple``. Using ``intransitive``
    warns with :class:`ErgmDifferenceWarning`.
    """
    return Intransitive()


def simmelian() -> Term:
    """Number of Simmelian triads (directed networks): complete triads, type 300."""
    return Simmelian()


def nearsimmelian() -> Term:
    """Number of near-Simmelian triads (directed networks): one tie short of complete, type 210."""
    return NearSimmelian()


def simmelianties() -> Term:
    """Number of ties in at least one Simmelian triad (directed networks)."""
    return SimmelianTies()


def transitiveties(attr=None, levels=None) -> Term:
    """Number of ties i -> j with a two-path i -> k -> j (in undirected networks,
    ties with a shared partner); with ``attr``, only ties and two-paths whose
    three vertices have the same value."""
    return TransitiveTies(attr, levels)


def cyclicalties(attr=None, levels=None) -> Term:
    """Number of ties i -> j with a two-path j -> k -> i (in undirected networks,
    ties with a shared partner); ``attr`` as in :func:`transitiveties`."""
    return CyclicalTies(attr, levels)


def threetrail(keep=None, levels=None) -> Term:
    """Number of 3-trails: walks of three distinct ties (a triangle counts as
    three). In directed networks, four statistics by the directions of the
    outer steps around the middle one, RRR, RRL, LRR and LRL; ``levels``
    selects some."""
    return ThreeTrail(keep, levels)


def opentriad() -> Term:
    """Number of 2-stars minus three times the number of triangles (undirected networks)."""
    return OpenTriad()


def localtriangle(x) -> Term:
    """Number of triangles whose three pairs of vertices are neighbours in ``x``
    (a graph attribute holding a symmetric adjacency matrix, the matrix or a graph)."""
    return LocalTriangle(x)


def m2star() -> Term:
    """Number of mixed 2-stars i -> j -> k, i != k (directed networks): twopath."""
    return M2Star()


def b1starmix(k: int, attr: str, base=None, diff: bool = True) -> Term:
    """Number of k-stars centred on first-mode vertices whose second-mode ends
    all have the same value of ``attr``, by value of the centre and (with
    ``diff=TRUE``) of the ends."""
    return B1StarMix(k, attr, base, diff)


def b2starmix(k: int, attr: str, base=None, diff: bool = True) -> Term:
    """Number of k-stars centred on second-mode vertices whose first-mode ends
    all have the same value of ``attr``, as :func:`b1starmix`."""
    return B2StarMix(k, attr, base, diff)


def b1twostar(b1attr: str, b2attr=None, base=None, b1levels=None, b2levels=None, levels2=None) -> Term:
    """Number of two-stars centred on first-mode vertices, by the value of
    ``b1attr`` of the centre and the (unordered) values of ``b2attr`` of the
    two ends."""
    return B1TwoStar(b1attr, b2attr, base, b1levels, b2levels, levels2)


def b2twostar(b1attr: str, b2attr=None, base=None, b1levels=None, b2levels=None, levels2=None) -> Term:
    """Number of two-stars centred on second-mode vertices, by the value of
    ``b2attr`` of the centre and the (unordered) values of ``b1attr`` of the
    two ends (``b2attr`` defaults to ``b1attr``)."""
    return B2TwoStar(b2attr if b2attr is not None else b1attr, b1attr, base, b2levels, b1levels, levels2)


def desp(d, type: str = "OTP") -> Term:
    """:func:`esp` for directed networks only, as ergm's desp."""
    return _with(Esp(d, type), directed=True)


def ddsp(d, type: str = "OTP") -> Term:
    """:func:`dsp` for directed networks only, as ergm's ddsp."""
    return _with(Dsp(d, type), directed=True)


def dnsp(d, type: str = "OTP") -> Term:
    """:func:`nsp` for directed networks only, as ergm's dnsp."""
    return _with(Nsp(d, type), directed=True)


def dgwesp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30, type: str = "OTP") -> Term:
    """:func:`gwesp` for directed networks only, as ergm's dgwesp."""
    return _with(gwesp(decay, fixed, cutoff, type), directed=True)


def dgwdsp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30, type: str = "OTP") -> Term:
    """:func:`gwdsp` for directed networks only, as ergm's dgwdsp."""
    return _with(gwdsp(decay, fixed, cutoff, type), directed=True)


def dgwnsp(decay: float = 0.5, fixed: bool = False, cutoff: int = 30, type: str = "OTP") -> Term:
    """:func:`gwnsp` for directed networks only, as ergm's dgwnsp."""
    return _with(gwnsp(decay, fixed, cutoff, type), directed=True)


def _recording(factory):
    """Wrap a term function so its terms remember how they were called."""

    def make(*args, **kwargs):
        term = factory(*args, **kwargs)
        term._call = (factory.__name__, args, kwargs)
        return term

    make.__name__, make.__doc__, make.__wrapped__ = factory.__name__, factory.__doc__, factory
    return make


_VOCABULARY = (
    degrange, idegrange, odegrange, degree1_5, idegree1_5, odegree1_5, concurrentties, density,
    meandeg, isolatededges, dyadcov, hamming, attrcov, mm, nodecovrange, nodeicovrange, nodeocovrange,
    nodefactordistinct, nodeofactordistinct, nodeifactordistinct, diff, smalldiff, altkstar,
    triadcensus, balance, intransitive, simmelian, nearsimmelian, simmelianties, transitiveties,
    cyclicalties, threetrail, opentriad, localtriangle, m2star, desp, ddsp, dnsp, dgwesp, dgwdsp, dgwnsp,
)
_MORE_BIPARTITE = (
    b1degrange, b2degrange, b1mindegree, b2mindegree, b1covrange, b2covrange, b1factordistinct,
    b2factordistinct, b1sociality, b2sociality, b1starmix, b2starmix, b1twostar, b2twostar,
)
for _f in _MORE_BIPARTITE:
    _f.__doc__ = inspect.cleandoc(_f.__doc__) + "\n\n" + _B_DOC

_PLAIN = (
    edges, mutual, asymmetric, kstar, istar, ostar, twopath, degree, idegree, odegree, isolates,
    concurrent, gwdegree, gwidegree, gwodegree, sender, receiver, sociality, triangle, ttriple,
    ctriple, transitive, cycle, gwesp, gwdsp, gwnsp, esp, dsp, nsp, nodematch, nodemix, nodefactor,
    nodeifactor, nodeofactor, nodecov, nodeicov, nodeocov, absdiff, absdiffcat, edgecov,
    *_BIPARTITE, *_MULTILEVEL, *_VOCABULARY, *_MORE_BIPARTITE,
)
for _f in _PLAIN:
    globals()[_f.__name__] = _recording(_f)
del _f

#: The operators that evaluate terms on each network of a combined network.
BLOCK_OPERATORS = ("N", "Form", "Persist", "Diss", "Cross", "Change")

#: Other names of terms, as in ergm: older or alternative names.
ALIASES = {"triangles": "triangle", "ttriad": "ttriple", "ctriad": "ctriple", "threepath": "threetrail",
           "nodemain": "nodecov", "degree1.5": "degree1_5", "idegree1.5": "idegree1_5",
           "odegree1.5": "odegree1_5"}

#: Every term function by name; F, offset and the block operators are operators.
TERMS = {name: globals()[name]
         for name in [f.__name__ for f in _PLAIN] + ["F", "S", "offset", *BLOCK_OPERATORS]}
TERMS.update({alias: TERMS[name] for alias, name in ALIASES.items() if "." not in alias})
TERMS.update({alias: TERMS[name] for alias, name in ALIASES.items() if "." in alias})


def _durational_terms():
    """tergm's statistics of tie ages, by their R names (mean.age...)."""
    from ._durational import DURATIONAL

    TERMS.update({name: _recording(factory) for name, factory in DURATIONAL.items()})


_durational_terms()
