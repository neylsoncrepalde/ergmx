"""Sample space constraints, with the same names and arguments as R's ergm.

A model's constraints restrict the networks it puts probability on::

    ergmx.ergm(g, "edges + mutual", constraints="bd(maxout=4)")
    ergmx.ergm(g, "edges + triangle", constraints="~degrees")

Constraints combine with ``+``, as terms do. Dyads given as edge lists use
R's vertex numbers, from 1.
"""

from __future__ import annotations

import ast

import numpy as np

from ._network import Network
from .terms import mixing_types

_NO_BOUND = 2**32 - 1


class Constraint:
    """A sample space constraint."""

    #: Whether the constraint makes dyads dependent: whether whether a dyad may
    #: change depends on other dyads. MPLE and exact MLEs need it False.
    dyad_dependent = False
    #: For degree-preserving constraints, the degrees they keep.
    preserves: frozenset = frozenset()
    #: The degree distributions they keep (beyond those of the degrees kept).
    preserves_distribution: frozenset = frozenset()
    #: The Rust core's name of the degree-preserving move, if any.
    preserve_kind = ""

    def fixed(self, network: Network) -> np.ndarray | None:
        """n x n mask of the dyads this constraint fixes at their observed
        value, for constraints that can't say it more compactly (below)."""
        return None

    def fixed_pairs(self, network: Network) -> np.ndarray | None:
        """The dyads it fixes, as pairs of vertices (k x 2)."""
        return None

    def free_pairs(self, network: Network) -> np.ndarray | None:
        """If it fixes all dyads but some, those, as pairs (k x 2)."""
        return None

    def groups(self, network: Network) -> np.ndarray | None:
        """If it fixes the dyads between groups of vertices, each vertex's group."""
        return None

    def bounds(self, network: Network):
        """(min_out, max_out, min_in, max_in) per vertex, or None."""
        return None

    def class_bounds(self, network: Network):
        """bd(attribs=)'s bounds by class of alters, or None."""
        return None

    def check(self, network: Network) -> None:
        pass

    def __repr__(self) -> str:
        return type(self).__name__.lower()


class Bd(Constraint):
    """Bounds on degrees, as ergm's bd(): at most ``maxout`` out-ties per vertex,
    and so on. Undirected networks use ``minout`` and ``maxout`` for degrees."""

    dyad_dependent = True

    def __init__(self, attribs=None, maxout=None, maxin=None, minout=None, minin=None):
        self.attribs = attribs
        self.maxout, self.maxin, self.minout, self.minin = maxout, maxin, minout, minin

    def _classes(self, network: Network) -> np.ndarray:
        x = network.graph_attribute(self.attribs) if isinstance(self.attribs, str) else self.attribs
        m = np.asarray(x, dtype=bool)
        if m.ndim == 1:
            m = m[:, None]
        if m.shape[0] != network.n:
            raise ValueError(f"bd(): attribs must have one row per vertex ({network.n}), not {m.shape[0]}")
        return m

    def class_bounds(self, network: Network):
        if self.attribs is None:
            return None
        classes = self._classes(network)
        shape = classes.shape

        def matrix(value, default, name):
            if value is None:
                return np.full(shape, default, dtype=np.int64)
            v = network.graph_attribute(value) if isinstance(value, str) else value
            m = np.asarray(v, dtype=float)
            m = np.broadcast_to(m if m.ndim else m.reshape(1, 1), shape) if m.size in (1, shape[1]) or \
                m.shape == shape else None
            if m is None:
                raise ValueError(f"bd(): {name} must be vertices x classes ({shape}) with attribs")
            return np.where(np.isnan(m), default, m).astype(np.int64)

        if not network.directed and (self.maxin is not None or self.minin is not None):
            raise ValueError("bd(): undirected networks use minout and maxout for degrees")
        return (shape[1], classes.ravel().tolist(), matrix(self.minout, 0, "minout").ravel().tolist(),
                matrix(self.maxout, _NO_BOUND, "maxout").ravel().tolist(),
                matrix(self.minin, 0, "minin").ravel().tolist(), matrix(self.maxin, _NO_BOUND, "maxin").ravel().tolist())

    def bounds(self, network: Network):
        if self.attribs is not None:
            return None
        def per_vertex(value, default, name):
            if value is None:
                return [default] * network.n
            values = [value] * network.n if np.isscalar(value) else list(value)
            if len(values) != network.n:
                raise ValueError(f"bd(): {name} needs one value or one per vertex ({network.n})")
            return [default if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)
                    for v in values]

        if not network.directed and (self.maxin is not None or self.minin is not None):
            raise ValueError("bd(): undirected networks use minout and maxout for degrees")
        return (per_vertex(self.minout, 0, "minout"), per_vertex(self.maxout, _NO_BOUND, "maxout"),
                per_vertex(self.minin, 0, "minin"), per_vertex(self.maxin, _NO_BOUND, "maxin"))

    def check(self, network: Network) -> None:
        if self.attribs is not None:
            self._check_classes(network)
            return
        min_out, max_out, min_in, max_in = self.bounds(network)
        out_degree, in_degree = network.degrees()
        for v in range(network.n):
            for kind, degree, low, high in (("out-degree" if network.directed else "degree",
                                             out_degree[v], min_out[v], max_out[v]),
                                            ("in-degree", in_degree[v], min_in[v], max_in[v])):
                if not network.directed and kind == "in-degree":
                    continue
                if not low <= degree <= high:
                    raise ValueError(
                        f"the observed network violates {self!r}: vertex {v} has {kind} {degree}"
                    )

    def _check_classes(self, network: Network) -> None:
        k, attribs, min_out, max_out, min_in, max_in = self.class_bounds(network)
        classes = np.asarray(attribs, dtype=bool).reshape(network.n, k)
        tails, heads = network.edges[:, 0].astype(np.int64), network.edges[:, 1].astype(np.int64)
        out_counts, in_counts = np.zeros((network.n, k), dtype=int), np.zeros((network.n, k), dtype=int)
        np.add.at(out_counts, tails, classes[heads])
        np.add.at(in_counts, heads, classes[tails])
        if not network.directed:
            np.add.at(out_counts, heads, classes[tails])
            in_counts = out_counts
        for name, counts, low, high in (("out", out_counts, min_out, max_out), ("in", in_counts, min_in, max_in)):
            if name == "in" and not network.directed:
                continue
            low, high = np.reshape(low, (network.n, k)), np.reshape(high, (network.n, k))
            bad = np.argwhere((counts < low) | (counts > high))
            if len(bad):
                v, c = bad[0]
                raise ValueError(f"the observed network violates {self!r}: vertex {v} has {counts[v, c]} "
                                 f"{name}-ties to alters of class {c + 1}")

    def __repr__(self) -> str:
        args = [f"{k}={v!r}" for k, v in (("attribs", self.attribs), ("maxout", self.maxout),
                                          ("maxin", self.maxin), ("minout", self.minout),
                                          ("minin", self.minin))
                if v is not None and not isinstance(v, np.ndarray)]
        return f"bd({', '.join(args)})"


class Blocks(Constraint):
    """Fix the dyads of some mixing types of a vertex attribute, as ergm's
    blocks(): those whose toggle would change ``nodemix(attr, levels, levels2)``."""

    def __init__(self, attr: str, levels=None, levels2=False):
        self.attr, self.levels, self.levels2 = attr, levels, levels2

    def fixed(self, network: Network) -> np.ndarray:
        _, codes, types = mixing_types(network, self.attr, self.levels, self.levels2)
        codes = np.asarray(codes)
        size = max(codes.max() + 1, 1)
        blocked = np.zeros((size, size), dtype=bool)
        for row, col in types:
            blocked[row, col] = True
            if not network.directed:
                blocked[col, row] = True
        mask = np.zeros((network.n, network.n), dtype=bool)
        valid = codes >= 0
        idx = np.flatnonzero(valid)
        mask[np.ix_(idx, idx)] = blocked[np.ix_(codes[idx], codes[idx])]
        np.fill_diagonal(mask, False)
        return mask

    def __repr__(self) -> str:
        args = [repr(self.attr)]
        if self.levels is not None:
            args.append(f"levels={self.levels!r}")
        if self.levels2 is not False:
            args.append(f"levels2={self.levels2!r}")
        return f"blocks({', '.join(args)})"


class Degrees(Constraint):
    """Preserve every vertex's degree (in- and out-degrees, if directed)."""

    dyad_dependent = True
    preserves = frozenset({"in", "out", "b1", "b2"})
    preserve_kind = "degrees"


class Edges(Constraint):
    """Preserve the number of edges, as ergm's edges constraint."""

    dyad_dependent = True
    preserves = frozenset({"edges"})
    preserve_kind = "edges"


class B1Degrees(Constraint):
    """Preserve the degrees of the first mode's vertices (bipartite networks)."""

    dyad_dependent = True
    preserves = frozenset({"b1"})
    preserve_kind = "b1degrees"
    mode = 1

    def check(self, network: Network) -> None:
        if not network.bipartite:
            raise ValueError(f"{self!r} needs a bipartite network")

    def first_mode(self, network: Network) -> list[bool]:
        return (network.mode == 1).tolist()


class B2Degrees(B1Degrees):
    """Preserve the degrees of the second mode's vertices (bipartite networks)."""

    preserves = frozenset({"b2"})
    preserve_kind = "b2degrees"
    mode = 2


def _dyad_pairs(network: Network, x, what: str) -> np.ndarray:
    """Dyads (k x 2, from 0; (low, high) if undirected) given as an edge list
    (R's vertex numbers, from 1), a boolean n x n matrix, a graph, or the name
    of a graph attribute holding one."""
    value = network.graph_attribute(x) if isinstance(x, str) else x
    if hasattr(value, "get_edgelist"):  # an igraph graph
        pairs = np.array(value.get_edgelist(), dtype=int).reshape(-1, 2)
    elif hasattr(value, "nodes") and hasattr(value, "edges"):  # a networkx graph
        index = {v: k for k, v in enumerate(value.nodes)}
        pairs = np.array([(index[a], index[b]) for a, b in value.edges], dtype=int).reshape(-1, 2)
    else:
        m = np.asarray(value)
        if m.shape == (network.n, network.n) and m.dtype == bool:
            pairs = np.argwhere(m)
            m = None
        elif m.ndim != 2 or m.shape[1] != 2:
            raise ValueError(f"{what}: expected an edge list (two columns of vertex numbers, from 1), "
                             "a logical n x n matrix or a graph")
        if m is not None:
            pairs = m.astype(int) - 1
            if len(pairs) and (pairs.min() < 0 or pairs.max() >= network.n):
                raise ValueError(f"{what}: vertex numbers must be between 1 and {network.n}")
    return _canonical(network, pairs)


def _canonical(network: Network, pairs) -> np.ndarray:
    """Distinct dyads without loops, as (low, high) if undirected."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not network.directed:
        pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype(np.uint32) if len(pairs) else np.zeros((0, 2), dtype=np.uint32)


def _within(pairs: np.ndarray, others: np.ndarray, network: Network) -> np.ndarray:
    """Whether each of `pairs` is one of `others` (both canonical)."""
    if not len(pairs) or not len(others):
        return np.zeros(len(pairs), dtype=bool)
    n = np.int64(network.n)
    return np.isin(pairs[:, 0].astype(np.int64) * n + pairs[:, 1], others[:, 0].astype(np.int64) * n + others[:, 1])


class Fixedas(Constraint):
    """Fix some dyads at their observed value, as ergm's fixedas():
    ``fixed_dyads`` (R's ``fixed.dyads``) as they are, ``present`` as ties and
    ``absent`` as non-ties (checked)."""

    def __init__(self, fixed_dyads=None, present=None, absent=None):
        self.fixed_dyads, self.present, self.absent = fixed_dyads, present, absent

    def fixed_pairs(self, network: Network) -> np.ndarray:
        given = [_dyad_pairs(network, x, "fixedas") for x in (self.fixed_dyads, self.present, self.absent)
                 if x is not None]
        return _canonical(network, np.vstack(given)) if given else np.zeros((0, 2), dtype=np.uint32)

    def check(self, network: Network) -> None:
        ties = _canonical(network, network.edges)
        if self.present is not None and not _within(_dyad_pairs(network, self.present, "fixedas"), ties, network).all():
            raise ValueError("fixedas(): some dyads of present= are not ties of the network")
        if self.absent is not None and _within(_dyad_pairs(network, self.absent, "fixedas"), ties, network).any():
            raise ValueError("fixedas(): some dyads of absent= are ties of the network")

    def __repr__(self) -> str:
        return "fixedas(...)"


class Fixallbut(Constraint):
    """Fix every dyad but ``free_dyads`` (R's ``free.dyads``), as ergm's fixallbut()."""

    def __init__(self, free_dyads):
        self.free_dyads = free_dyads

    def free_pairs(self, network: Network) -> np.ndarray:
        return _dyad_pairs(network, self.free_dyads, "fixallbut")

    def __repr__(self) -> str:
        return "fixallbut(...)"


class Observed(Constraint):
    """Fix the observed dyads: only those whose value is missing vary, as
    ergm's observed constraint (to simulate the missing ties)."""

    def free_pairs(self, network: Network) -> np.ndarray:
        return _canonical(network, network.missing)


class Blockdiag(Constraint):
    """Allow ties only between vertices with the same value of ``attr``, as
    ergm's blockdiag(): the dyads between blocks are fixed at no tie."""

    def __init__(self, attr: str, noncontig: str = "merge"):
        self.attr = attr

    def groups(self, network: Network) -> np.ndarray:
        values = network.attribute(self.attr)
        index: dict = {}
        return np.array([index.setdefault(v, len(index)) for v in values], dtype=np.int64)

    def check(self, network: Network) -> None:
        g = self.groups(network)
        if len(network.edges) and (g[network.edges[:, 0]] != g[network.edges[:, 1]]).any():
            raise ValueError(f"{self!r}: the network has ties between blocks")

    def __repr__(self) -> str:
        return f"blockdiag({self.attr!r})"


class DyadsConstraint(Constraint):
    """Fix or free the dyads that dyad-independent terms count, as ergm's
    Dyads(): with ``fix``, the dyads where any of its terms' statistics change
    are fixed; with ``vary``, only those where any of its terms' statistics
    change may vary; with both, the dyads that either lets vary."""

    def __init__(self, fix=None, vary=None):
        if fix is None and vary is None:
            raise ValueError("Dyads(): give fix= or vary=")
        self.fix, self.vary = fix, vary

    @staticmethod
    def _counted(network: Network, formula) -> np.ndarray:
        from ._model import bind

        model = bind(network, formula)
        if not model.dyad_independent:
            raise ValueError(f"Dyads(): the terms must be dyad-independent, not {model.formula!r}")
        x, _, pairs = model.core.mple_data(network.edges)
        mask = np.zeros((network.n, network.n), dtype=bool)
        hit = pairs[(x != 0).any(axis=1)].astype(np.int64)
        mask[hit[:, 0], hit[:, 1]] = True
        return mask if network.directed else mask | mask.T

    def fixed(self, network: Network) -> np.ndarray:
        free = np.zeros((network.n, network.n), dtype=bool)
        if self.fix is not None:
            free |= ~self._counted(network, self.fix)
        if self.vary is not None:
            free |= self._counted(network, self.vary)
        mask = ~free
        np.fill_diagonal(mask, False)
        return mask

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in (("fix", self.fix), ("vary", self.vary)) if v is not None]
        return f"Dyads({', '.join(parts)})"


class ODegrees(Constraint):
    """Preserve every vertex's out-degree (directed networks)."""

    dyad_dependent = True
    preserves = frozenset({"out"})
    preserve_kind = "odegrees"

    def check(self, network: Network) -> None:
        if not network.directed:
            raise ValueError(f"{self!r} needs a directed network")


class IDegrees(ODegrees):
    """Preserve every vertex's in-degree (directed networks)."""

    preserves = frozenset({"in"})
    preserve_kind = "idegrees"


class DegreeDist(Constraint):
    """Preserve the degree distribution (of in- and out-degrees, if
    directed), as R's ergm documents its degreedist constraint: the
    vertices' degrees may change, but not how many vertices have each
    degree. (ergm 4.12's proposals keep every vertex's out-degree and swap
    in-degrees, or the reverse, which reaches fewer networks; see
    ``odegreedist``.)"""

    dyad_dependent = True
    preserve_kind = "degreedist"
    preserves_distribution = frozenset({"in", "out", "b1", "b2"})


class ODegreeDist(Constraint):
    """Preserve the out-degree distribution (directed networks). R's ergm
    documents this; ergm 4.12 keeps every vertex's out-degree, as odegrees."""

    dyad_dependent = True
    preserve_kind = "odegreedist"
    preserves_distribution = frozenset({"out"})

    def check(self, network: Network) -> None:
        if not network.directed:
            raise ValueError(f"{self!r} needs a directed network")


class IDegreeDist(ODegreeDist):
    """Preserve the in-degree distribution (directed networks). R's ergm
    documents this; ergm 4.12 keeps every vertex's in-degree, as idegrees."""

    preserve_kind = "idegreedist"
    preserves_distribution = frozenset({"in"})


class Egocentric(Constraint):
    """Fix the dyads of some vertices, as ergm's egocentric(): of those whose
    ``attr`` is true (or, without ``attr``, whose ``na`` attribute is false).
    ``direction`` (directed networks): "both" fixes the dyads with such a
    vertex at either end, "out" those they send and "in" those they receive."""

    def __init__(self, attr=None, direction: str = "both"):
        if direction not in ("both", "out", "in"):
            raise ValueError(f"egocentric(): direction must be 'both', 'out' or 'in', not {direction!r}")
        self.attr, self.direction = attr, direction

    def _egos(self, network: Network) -> np.ndarray:
        if self.attr is None:
            if "na" not in network.attributes:
                raise ValueError("egocentric() without attr needs the vertex attribute 'na'")
            return ~np.asarray(network.attribute("na"), dtype=bool)
        values = network.attribute(self.attr) if isinstance(self.attr, str) else self.attr
        egos = np.asarray(values)
        if egos.dtype != bool or egos.shape != (network.n,):
            raise ValueError("egocentric(): attr must be a logical vertex attribute (or one logical value per vertex)")
        return egos

    def check(self, network: Network) -> None:
        if not network.directed and self.direction != "both":
            raise ValueError("egocentric(): direction applies to directed networks only")
        self._egos(network)

    def fixed(self, network: Network) -> np.ndarray:
        egos = self._egos(network)
        if self.direction == "out":
            mask = np.repeat(egos[:, None], network.n, axis=1)
        elif self.direction == "in":
            mask = np.repeat(egos[None, :], network.n, axis=0)
        else:
            mask = egos[:, None] | egos[None, :]
        np.fill_diagonal(mask, False)
        return mask

    def __repr__(self) -> str:
        args = ([repr(self.attr)] if isinstance(self.attr, str) else []) + \
            ([f"direction={self.direction!r}"] if self.direction != "both" else [])
        return f"egocentric({', '.join(args)})"


CONSTRAINTS = {
    "bd": Bd, "blocks": Blocks, "degrees": Degrees, "nodedegrees": Degrees,
    "odegrees": ODegrees, "idegrees": IDegrees, "edges": Edges, "b1degrees": B1Degrees,
    "b2degrees": B2Degrees, "fixedas": Fixedas, "fixallbut": Fixallbut, "observed": Observed,
    "blockdiag": Blockdiag, "Dyads": DyadsConstraint, "degreedist": DegreeDist,
    "odegreedist": ODegreeDist, "idegreedist": IDegreeDist, "egocentric": Egocentric,
}


class Constraints:
    """The constraints of a model: none, or a sum of constraints."""

    def __init__(self, items=()):
        self.items = list(items)
        preserving = [c for c in self.items if c.preserve_kind]
        if len(preserving) > 1:
            raise ValueError(f"only one degree-preserving constraint at a time, not {preserving}")

    @property
    def dyad_dependent(self) -> bool:
        return any(c.dyad_dependent for c in self.items)

    @property
    def preserves(self) -> frozenset | None:
        """The degrees a degree-preserving constraint keeps, or None."""
        kept = [c.preserves for c in self.items if c.preserve_kind]
        return kept[0] if kept else None

    @property
    def preserves_distribution(self) -> frozenset | None:
        """The degree distributions a degree-preserving constraint keeps
        (those of the degrees it keeps too), or None."""
        kept = [c.preserves | c.preserves_distribution for c in self.items if c.preserve_kind]
        return kept[0] if kept else None

    @property
    def preserve_kind(self) -> str:
        return next((c.preserve_kind for c in self.items if c.preserve_kind), "")

    def fixed(self, network: Network) -> np.ndarray | None:
        """n x n mask of the dyads fixed by the constraints that need one, or None."""
        masks = [m for m in (c.fixed(network) for c in self.items) if m is not None]
        if not masks:
            return None
        mask = masks[0].copy()
        for m in masks[1:]:
            mask |= m
        return mask

    def fixed_pairs(self, network: Network) -> np.ndarray:
        """The dyads fixed as pairs (k x 2)."""
        given = [p for p in (c.fixed_pairs(network) for c in self.items) if p is not None]
        return _canonical(network, np.vstack(given)) if given else np.zeros((0, 2), dtype=np.uint32)

    def free_pairs(self, network: Network) -> np.ndarray | None:
        """If the constraints fix all dyads but some, those (k x 2), or None."""
        free = None
        for p in (c.free_pairs(network) for c in self.items):
            if p is not None:
                free = p if free is None else free[_within(free, p, network)]
        return free

    def groups(self, network: Network) -> np.ndarray | None:
        """Each vertex's group, if the constraints fix the dyads between groups."""
        groups = None
        for g in (c.groups(network) for c in self.items):
            if g is not None:
                groups = g if groups is None else np.unique(np.column_stack([groups, g]), axis=0,
                                                            return_inverse=True)[1].ravel()
        return groups

    def first_mode(self, network: Network):
        """The first mode's vertices, for b1degrees and b2degrees."""
        return next((c.first_mode(network) for c in self.items if hasattr(c, "first_mode")), None)

    def class_bounds(self, network: Network):
        bounds = [b for b in (c.class_bounds(network) for c in self.items) if b is not None]
        if len(bounds) > 1:
            raise ValueError("only one bd(attribs=...) constraint at a time")
        return bounds[0] if bounds else None

    def bounds(self, network: Network):
        bounds = [b for b in (c.bounds(network) for c in self.items) if b is not None]
        if not bounds:
            return None
        # Several bd() constraints: the tightest bound of each kind.
        return tuple(
            list(np.max([b[k] for b in bounds], axis=0) if k in (0, 2) else np.min([b[k] for b in bounds], axis=0))
            for k in range(4)
        )

    def check(self, network: Network) -> None:
        for c in self.items:
            c.check(network)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __eq__(self, other) -> bool:
        return isinstance(other, Constraints) and repr(self) == repr(other)

    def __repr__(self) -> str:
        return " + ".join(map(repr, self.items)) if self.items else "none"


def _unparse_formula(text: str) -> str:
    """A formula argument of Dyads() back as a formula string ("~a" from ast's "~a")."""
    return text.lstrip("~").strip()


def parse_constraints(constraints) -> Constraints:
    """Constraints from a string in R syntax (``"~bd(maxout=4) + blocks('level')"``),
    a constraint, a list of them, or None."""
    if constraints is None:
        return Constraints()
    if isinstance(constraints, Constraints):
        return constraints
    if isinstance(constraints, Constraint):
        return Constraints([constraints])
    if isinstance(constraints, (list, tuple)):
        return Constraints([c for x in constraints for c in parse_constraints(x).items])
    if not isinstance(constraints, str):
        raise TypeError(f"constraints must be a string, not {type(constraints).__name__}")
    from .formula import FormulaError, _literal, _r_syntax, _strip_lhs

    text = _r_syntax(_strip_lhs(constraints))
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"can't parse the constraints {constraints!r}: {e.msg}") from None

    def items(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return items(node.left) + items(node.right)
        if isinstance(node, ast.Name):
            name, args, kwargs = node.id, [], {}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Dyads":
            # Its arguments are formulas of terms: kept as text.
            name = "Dyads"
            texts = [ast.unparse(a) for a in node.args]
            kwargs = {k.arg: ast.unparse(k.value) for k in node.keywords}
            args = []
            if texts:
                kwargs.setdefault("fix", texts[0])
            for key in ("fix", "vary"):
                if key in kwargs:
                    kwargs[key] = _unparse_formula(kwargs[key])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            args = [_literal(a) for a in node.args]
            kwargs = {k.arg: _literal(k.value) for k in node.keywords}
        else:
            raise FormulaError(f"can't parse {ast.unparse(node)!r} in the constraints {constraints!r}")
        if name not in CONSTRAINTS:
            raise FormulaError(f"unknown constraint {name!r}; available: {', '.join(CONSTRAINTS)}")
        try:
            return [CONSTRAINTS[name](*args, **kwargs)]
        except TypeError as e:
            raise FormulaError(f"{name}: {e}") from None

    return Constraints(items(tree))

