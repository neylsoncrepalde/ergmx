"""Sample space constraints, with the same names and arguments as R's ergm.

A model's constraints restrict the networks it puts probability on::

    ergmx.ergm(g, "edges + mutual", constraints="bd(maxout=4)")
    ergmx.ergm(g, "edges + triangle", constraints="~degrees")

Constraints combine with ``+``, as terms do.
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
    #: The Rust core's name of the degree-preserving move, if any.
    preserve_kind = ""

    def fixed(self, network: Network) -> np.ndarray | None:
        """n x n mask of the dyads this constraint fixes at their observed value."""
        return None

    def bounds(self, network: Network):
        """(min_out, max_out, min_in, max_in) per vertex, or None."""
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
        if attribs is not None:
            raise NotImplementedError("bd(attribs=...) bounds by attribute are not supported yet")
        self.maxout, self.maxin, self.minout, self.minin = maxout, maxin, minout, minin

    def bounds(self, network: Network):
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

    def __repr__(self) -> str:
        args = [f"{k}={v!r}" for k, v in (("maxout", self.maxout), ("maxin", self.maxin),
                                          ("minout", self.minout), ("minin", self.minin))
                if v is not None]
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
    preserves = frozenset({"in", "out"})
    preserve_kind = "degrees"


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


CONSTRAINTS = {
    "bd": Bd, "blocks": Blocks, "degrees": Degrees, "nodedegrees": Degrees,
    "odegrees": ODegrees, "idegrees": IDegrees,
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
    def preserve_kind(self) -> str:
        return next((c.preserve_kind for c in self.items if c.preserve_kind), "")

    def fixed(self, network: Network) -> np.ndarray:
        mask = np.zeros((network.n, network.n), dtype=bool)
        for c in self.items:
            m = c.fixed(network)
            if m is not None:
                mask |= m
        return mask

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

