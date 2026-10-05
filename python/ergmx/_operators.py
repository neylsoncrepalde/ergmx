"""ergm's operators on formulas: Sum(), Prod(), Log(), Exp(), Symmetrize(),
Label(), Passthrough(), I(), Offset() and Curve() (with For(), expanded
when the formula is parsed). Sum(), Log(), Exp() and Prod() transform a
submodel's statistics, Symmetrize() evaluates it on the undirected network
of a rule, Label() renames, Offset() fixes some of its parameters and
Curve() makes them functions of fewer."""

from __future__ import annotations

import dataclasses

import numpy as np

from ._network import Network
from .terms import Formula, Term, as_formula


class AsIs(str):
    """A label used as it is, as R's ``I("label")``."""


class _Submodel(Term):
    """A term made of a formula's terms: their statistics and parameters,
    passed through (or transformed, in subclasses)."""

    def __init__(self, formula):
        self.formula = as_formula(formula)
        if not len(self.formula):
            raise ValueError(f"{type(self).__name__}(): the formula has no terms")

    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))
    triadic = property(lambda self: any(t.triadic for t in self.formula))
    curved = property(lambda self: any(t.curved for t in self.formula))

    def check(self, network):
        for term in self.formula:
            term.check(network)

    def _inner_names(self, network) -> list[str]:
        return [n for t in self.formula for n in t.names(network)]

    def _inner_params(self, network) -> list[str]:
        return [n for t in self.formula for n in t.param_names(network)]

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

    def _children(self, network) -> list:
        return [t.full_spec(network) for t in self.formula]

    def full_spec(self, network):
        p = len(self._inner_names(network))
        return ("map", np.eye(p).ravel().tolist(), [0, p], self._children(network))

    def spec(self, network):
        return self.full_spec(network)[:3]

    def _linear_only(self):
        if self.curved:
            raise NotImplementedError(f"{type(self).__name__}() of curved terms is not supported: fix their decays")


class Passthrough(_Submodel):
    """ergm's Passthrough(formula): the formula's terms, as they are (names
    wrapped in ``Passthrough~`` with ``label=TRUE``)."""

    def __init__(self, formula, label: bool = False):
        super().__init__(formula)
        self.label_names = bool(label)

    def names(self, network):
        return [f"Passthrough~{n}" if self.label_names else n for n in self._inner_names(network)]

    def param_names(self, network):
        return [f"Passthrough~{n}" if self.label_names else n for n in self._inner_params(network)]

    def __repr__(self) -> str:
        return f"Passthrough({self.formula!r})"


class Label(_Submodel):
    """ergm's Label(formula, label, pos="("): the formula's terms, renamed:
    ``label(name)``, or with the label prepended, appended, or replacing the
    names (a list of names, None to keep one)."""

    POSITIONS = ("(", "prepend", "append", "replace")

    def __init__(self, formula, label, pos: str = "("):
        super().__init__(formula)
        if pos not in self.POSITIONS:
            raise ValueError(f"Label(pos=): one of {', '.join(map(repr, self.POSITIONS))}")
        self.rename_to, self.pos = label, pos

    def _rename(self, names: list[str], curved_part: int | None = None) -> list[str]:
        label = self.rename_to
        if callable(label):
            return [label(n) for n in names]
        if self.pos == "replace":
            labels = label[curved_part] if curved_part is not None and isinstance(label, (list, tuple)) \
                and len(label) == 2 and isinstance(label[0], (list, tuple)) else label
            labels = [labels] if isinstance(labels, str) else list(labels)
            if len(labels) != len(names):
                raise ValueError(f"Label(pos='replace'): {len(labels)} labels for {len(names)} names")
            return [n if new is None else str(new) for n, new in zip(names, labels)]
        if not isinstance(label, str):
            raise ValueError("Label(): the label is a string, unless pos='replace'")
        return {"(": [f"{label}({n})" for n in names], "prepend": [label + n for n in names],
                "append": [n + label for n in names]}[self.pos]

    def names(self, network):
        return self._rename(self._inner_names(network), 1)

    def param_names(self, network):
        return self._rename(self._inner_params(network), 0)

    def __repr__(self) -> str:
        return f"Label({self.formula!r}, {self.rename_to!r})"


def _weights(weight, p: int, what: str) -> np.ndarray:
    """A formula's weights in Sum() or Prod(): a matrix with p columns."""
    if weight is None:
        return np.eye(p)
    if isinstance(weight, str):
        if weight in ("sum", "prod"):
            return np.ones((1, p))
        if weight in ("mean", "geomean"):
            return np.full((1, p), 1 / p)
        raise ValueError(f"{what}(): weights 'sum', 'mean' (Sum) or 'prod', 'geomean' (Prod), not {weight!r}")
    w = np.asarray(weight, dtype=float)
    if w.ndim == 0:
        return w * np.eye(p)
    if w.ndim == 1:
        if len(w) != p:
            raise ValueError(f"{what}(): {len(w)} weights for {p} statistics")
        return np.diag(w)
    if w.shape[1] != p:
        raise ValueError(f"{what}(): a weight matrix with {p} columns, not {w.shape[1]}")
    return w


class _Combination(_Submodel):
    """Sum() and Prod(): formulas, each with weights (a number, a vector, a
    matrix, or "sum", "mean"; "prod", "geomean" for Prod), whose weighted
    statistics are added (multiplied, for Prod) elementwise."""

    op = "Sum"

    def __init__(self, formulas, label):
        if isinstance(formulas, (Formula, Term, str)):
            formulas = [(None, formulas)]
        pairs = [(None, f) if not isinstance(f, tuple) else f for f in formulas]
        self.parts = [(w, as_formula(f)) for w, f in pairs]
        super().__init__(Formula([t for _, f in self.parts for t in f]))
        self._linear_only()
        self.labels = label

    def matrix(self, network) -> np.ndarray:
        blocks = [_weights(w, sum(len(t.names(network)) for t in f), self.op) for w, f in self.parts]
        if len({b.shape[0] for b in blocks}) > 1:
            raise ValueError(f"{self.op}(): the formulas' weighted statistics differ in number")
        return np.hstack(blocks)

    def names(self, network):
        q = self.matrix(network).shape[0]
        label = self.labels
        labels = [label] if isinstance(label, str) else list(label)
        if len(labels) == 1 and q > 1:
            labels = [f"{labels[0]}{k}" for k in range(1, q + 1)]
        if len(labels) != q:
            raise ValueError(f"{self.op}(): {len(labels)} labels for {q} statistics")
        asis = isinstance(label, AsIs) or (not isinstance(label, str) and all(isinstance(x, AsIs) for x in label))
        return labels if asis else [f"{self.op}~{x}" for x in labels]

    param_names = names

    def eta(self, params, network):
        return np.asarray(params, dtype=float)

    def jacobian(self, params, network):
        return np.eye(len(params))

    def starts(self, network):
        return []

    def full_spec(self, network):
        m = self.matrix(network)
        return ("map", m.ravel().tolist(), [0, m.shape[0]], self._children(network))

    def __repr__(self) -> str:
        return f"{self.op}({[f for _, f in self.parts]!r}, {self.labels!r})"


class Sum(_Combination):
    """ergm's Sum(formulas, label): a linear combination of formulas' statistics."""

    op = "Sum"


class Prod(_Combination):
    """ergm's Prod(formulas, label): the product of formulas' statistics,
    each to the power of its weights, as ergm's Exp(Sum(Log())). The
    statistics must be nonnegative."""

    op = "Prod"

    def full_spec(self, network):
        m = self.matrix(network)
        logged = ("map", [_LOG0], [1], self._children(network))
        return ("map", [], [2], [("map", m.ravel().tolist(), [0, m.shape[0]], [logged])])


_LOG0 = -1 / np.sqrt(np.finfo(float).eps)


class Log(_Submodel):
    """ergm's Log(formula, log0): the logarithm of the formula's statistics,
    with ``log0`` for log(0)."""

    def __init__(self, formula, log0: float = _LOG0):
        super().__init__(formula)
        self._linear_only()
        self.log0 = float(log0)

    def names(self, network):
        return [f"Log~{n}" for n in self._inner_names(network)]

    param_names = names
    eta = _Combination.eta
    jacobian = _Combination.jacobian
    starts = _Combination.starts

    def full_spec(self, network):
        return ("map", [self.log0], [1], self._children(network))

    def __repr__(self) -> str:
        return f"Log({self.formula!r})"


class Exp(_Submodel):
    """ergm's Exp(formula): the exponential of the formula's statistics."""

    def __init__(self, formula):
        super().__init__(formula)
        self._linear_only()

    def names(self, network):
        return [f"Exp~{n}" for n in self._inner_names(network)]

    param_names = names
    eta = _Combination.eta
    jacobian = _Combination.jacobian
    starts = _Combination.starts

    def full_spec(self, network):
        return ("map", [], [2], self._children(network))

    def __repr__(self) -> str:
        return f"Exp({self.formula!r})"


class Symmetrize(_Submodel):
    """ergm's Symmetrize(formula, rule="weak"): the formula on the undirected
    network of the directed one: a tie {i, j} if either of i -> j and
    j -> i is ("weak", "max"), if both are ("strong", "min"), or if the one
    from the lower vertex ("upper") or the higher ("lower") is."""

    RULES = {"weak": 0, "max": 0, "strong": 1, "min": 1, "upper": 2, "lower": 3}

    def __init__(self, formula, rule: str = "weak"):
        super().__init__(formula)
        if rule not in self.RULES:
            raise ValueError(f"Symmetrize(rule=): one of {', '.join(map(repr, self.RULES))}")
        self.rule = {"max": "weak", "min": "strong"}.get(rule, rule)

    @staticmethod
    def _undirected(network):
        return dataclasses.replace(network, directed=False)

    def check(self, network):
        if not network.directed:
            raise ValueError("Symmetrize() is for directed networks")
        super().check(self._undirected(network))

    def names(self, network):
        return [f"Symmetrize({self.rule})~{n}" for n in self._inner_names(self._undirected(network))]

    def param_names(self, network):
        return [f"Symmetrize({self.rule})~{n}" for n in self._inner_params(self._undirected(network))]

    def eta(self, params, network):
        return super().eta(params, self._undirected(network))

    def jacobian(self, params, network):
        return super().jacobian(params, self._undirected(network))

    def starts(self, network):
        return super().starts(self._undirected(network))

    def full_spec(self, network):
        return ("symmetrize", [], [self.RULES[self.rule]], self._children(self._undirected(network)))

    def __repr__(self) -> str:
        return f"Symmetrize({self.formula!r}, rule={self.rule!r})"


class OffsetOp(_Submodel):
    """ergm's Offset(formula, coef, which): the formula with some parameters
    fixed at ``coef`` (``which``: logical, indices from 1, or names; all by
    default): the others are estimated."""

    def __init__(self, formula, coef=0.0, which=True):
        super().__init__(formula)
        self.coef, self.which = coef, which

    def _selection(self, network) -> np.ndarray:
        names = self._inner_params(network)
        which = self.which
        if isinstance(which, (bool, np.bool_)) or (isinstance(which, (list, tuple)) and which
                                                   and all(isinstance(w, (bool, np.bool_)) for w in which)):
            flags = np.resize(np.asarray(which, dtype=bool), len(names))
        elif isinstance(which, str) or (isinstance(which, (list, tuple)) and which and isinstance(which[0], str)):
            wanted = [which] if isinstance(which, str) else list(which)
            missing = [w for w in wanted if w not in names]
            if missing:
                raise ValueError(f"Offset(which=): no parameters {missing}; they are {names}")
            flags = np.isin(names, wanted)
        else:
            indices = np.atleast_1d(np.asarray(which, dtype=int))
            if np.any((indices < 1) | (indices > len(names))):
                raise ValueError(f"Offset(which=): indices from 1 to {len(names)}")
            flags = np.zeros(len(names), dtype=bool)
            flags[indices - 1] = True
        return flags

    def _fixed(self, network):
        flags = self._selection(network)
        return flags, np.resize(np.asarray(self.coef, dtype=float), int(flags.sum()))

    curved = True  # its parameters map to its statistics' coefficients, with some fixed

    def names(self, network):
        return self._inner_names(network)

    def param_names(self, network):
        flags = self._selection(network)
        return [n for n, f in zip(self._inner_params(network), flags) if not f]

    def _full(self, params, network):
        flags, values = self._fixed(network)
        full = np.zeros(len(flags))
        full[flags], full[~flags] = values, np.asarray(params, dtype=float)
        return full, flags

    def eta(self, params, network):
        full, _ = self._full(params, network)
        return super().eta(full, network)

    def jacobian(self, params, network):
        full, flags = self._full(params, network)
        return super().jacobian(full, network)[:, ~flags]

    def starts(self, network):
        flags = self._selection(network)
        position = np.cumsum(~flags) - 1
        return [(int(position[i]), v) for i, v in super().starts(network) if not flags[i]]

    def __repr__(self) -> str:
        return f"Offset({self.formula!r}, {self.coef!r})"


class Curve(_Submodel):
    """ergm's Curve(formula, params, map, gradient) (also Parametrise(),
    Parametrize()): the formula's coefficients as a function ``map`` of
    fewer parameters ``params`` (a dict of their starting values, or their
    names). ``map`` is a function of the parameters (and the number of
    coefficients), "rep" (the parameters recycled) or fixed numbers;
    ``gradient``, its derivatives (parameters x coefficients: a function,
    a matrix, or "linear", by finite differences), needed for functions."""

    curved = True

    def __init__(self, formula, params, map="rep", gradient=None, minpar=-np.inf, maxpar=np.inf):
        super().__init__(formula)
        if isinstance(params, dict):
            self.param_list = list(params)
            self.initial = [np.nan if v is None else float(v) for v in params.values()]
        else:
            self.param_list = [str(p) for p in ([params] if isinstance(params, str) else params)]
            self.initial = [np.nan] * len(self.param_list)
        self.map, self.gradient = map, gradient
        self.minpar, self.maxpar = minpar, maxpar
        if callable(map) and gradient is None:
            raise ValueError("Curve(): a map function needs its gradient (a function, a matrix or 'linear')")

    def names(self, network):
        return self._inner_names(network)

    def param_names(self, network):
        return list(self.param_list)

    def _n(self, network) -> int:
        return len(self._inner_params(network))

    def _map(self, params, network) -> np.ndarray:
        n, x = self._n(network), np.asarray(params, dtype=float)
        if callable(self.map):
            return np.asarray(self.map(x, n), dtype=float).reshape(n)
        if isinstance(self.map, str):
            if self.map != "rep":
                raise ValueError(f"Curve(map=): a function, numbers or 'rep', not {self.map!r}")
            return np.resize(x, n)
        return np.resize(np.asarray(self.map, dtype=float), n)

    def _gradient(self, params, network) -> np.ndarray:
        """Coefficients x parameters."""
        n, x = self._n(network), np.asarray(params, dtype=float)
        if isinstance(self.map, str) or not callable(self.map) and self.gradient is None:
            if not isinstance(self.map, str):  # fixed numbers
                return np.zeros((n, len(x)))
            return np.array([[1.0 if i % len(x) == k else 0.0 for k in range(len(x))] for i in range(n)])
        g = self.gradient
        if callable(g):
            return np.asarray(g(x, n), dtype=float).reshape(len(x), n).T
        if isinstance(g, str) and g == "linear":
            step = 1e-6
            base = self._map(x, network)
            return np.column_stack([(self._map(x + step * e, network) - base) / step for e in np.eye(len(x))])
        return np.asarray(g, dtype=float).reshape(len(x), n).T

    def eta(self, params, network):
        return super().eta(self._map(params, network), network)

    def jacobian(self, params, network):
        inner = self._map(params, network)
        return super().jacobian(inner, network) @ self._gradient(params, network)

    def starts(self, network):
        return [(k, v) for k, v in enumerate(self.initial) if np.isfinite(v)]

    def lower_bounds(self, network) -> np.ndarray:
        """ergm's minpar, recycled (-inf by default)."""
        return np.resize(np.asarray(self.minpar, dtype=float), len(self.param_list))

    def upper_bounds(self, network) -> np.ndarray:
        """ergm's maxpar, recycled (+inf by default)."""
        return np.resize(np.asarray(self.maxpar, dtype=float), len(self.param_list))

    def __repr__(self) -> str:
        return f"Curve({self.formula!r}, {self.param_list!r})"


class Taper(_Submodel):
    """ergm.tapered's Taper(formula, coef, m): the formula's statistics, and
    the penalty sum_k coef_k (g_k - m_k)^2 (``Taper_Penalty``), around the
    centers ``m`` (by default the network's statistics). A single ``coef``
    is scaled as ergm.tapered's: coef / (4 m). With the penalty's
    coefficient fixed at -1 (see :func:`ergmx.ergm_tapered`), a tapered ERGM."""

    def __init__(self, formula, coef=None, m=None):
        super().__init__(formula)
        self.coef, self.m = coef, m

    def _centers(self, network) -> np.ndarray:
        if self.m is not None:
            return np.atleast_1d(np.asarray(self.m, dtype=float))
        from ._simulate import summary_stats

        return np.array(list(summary_stats(network, self.formula).values()))

    def _tau(self, network, centers) -> np.ndarray:
        if self.coef is None:
            return 1 / (4 * centers)
        if np.ndim(self.coef) == 0:  # a multiplier, as ergm.tapered's
            return float(self.coef) / (4 * centers)
        coef = np.asarray(self.coef, dtype=float)
        if len(coef) != len(centers):
            raise ValueError(f"Taper(coef=): {len(coef)} coefficients for {len(centers)} statistics")
        return coef

    def names(self, network):
        return [*self._inner_names(network), "Taper_Penalty"]

    def param_names(self, network):
        return [*self._inner_params(network), "Taper_Penalty"]

    def eta(self, params, network):
        params = np.asarray(params, dtype=float)
        return np.concatenate([super().eta(params[:-1], network), params[-1:]])

    def jacobian(self, params, network):
        params = np.asarray(params, dtype=float)
        inner = super().jacobian(params[:-1], network)
        out = np.zeros((inner.shape[0] + 1, inner.shape[1] + 1))
        out[:-1, :-1], out[-1, -1] = inner, 1.0
        return out

    def full_spec(self, network):
        centers = self._centers(network)
        tau = self._tau(network, centers)
        if len(centers) != len(self._inner_names(network)):
            raise ValueError(f"Taper(m=): {len(centers)} centers for {len(self._inner_names(network))} statistics")
        return ("map", [*tau.tolist(), *centers.tolist()], [3], self._children(network))

    def __repr__(self) -> str:
        return f"Taper({self.formula!r})"


class Project(Term):
    """ergm's Project(formula, mode) (Proj1(), Proj2()): valued terms of the
    projection of a bipartite network onto a mode, the undirected network
    of that mode's vertices whose values are their numbers of shared
    partners (of the other mode)."""

    dyad_independent = False

    def __init__(self, formula, mode: int):
        from ._valued import parse_valued_formula

        if mode not in (1, 2):
            raise ValueError("Project(mode=): 1 or 2")
        self.formula, self.mode = parse_valued_formula(formula), int(mode)
        if not len(self.formula):
            raise ValueError("Project(): the formula has no terms")
        if any(t.is_offset for t in self.formula):
            raise ValueError("Project(): offset() terms aren't supported inside it")

    def _projected(self, network):
        """The mode's vertices (with their attributes), without values."""
        keep = np.flatnonzero(network.mode == self.mode)
        attributes = {a: [values[k] for k in keep] for a, values in network.attributes.items()}
        return Network(len(keep), False, np.zeros((0, 2), dtype=np.uint32), attributes)

    def check(self, network):
        if not network.bipartite or network.combined:
            raise ValueError(f"Proj{self.mode}() is for a bipartite network")
        projected = self._projected(network)
        for term in self.formula:
            term.check(projected)

    def names(self, network):
        return [f"Proj{self.mode}~{n}" for t in self.formula for n in t.names(self._projected(network))]

    def full_spec(self, network):
        local = np.cumsum(network.mode == self.mode) - 1
        local = np.where(network.mode == self.mode, local, -1).astype(int).tolist()
        projected = self._projected(network)
        return ("project", [], local, [t.full_spec(projected) for t in self.formula])

    def spec(self, network):
        return self.full_spec(network)[:3]

    def __repr__(self) -> str:
        return f"Proj{self.mode}({self.formula!r})"


def _proj(mode: int):
    def make(formula):
        return Project(formula, mode)

    return make


OPERATORS = {"Sum": Sum, "Prod": Prod, "Log": Log, "Exp": Exp, "Symmetrize": Symmetrize, "Label": Label,
             "Passthrough": Passthrough, "Offset": OffsetOp, "Curve": Curve, "Parametrise": Curve,
             "Parametrize": Curve, "Project": Project, "Proj1": _proj(1), "Proj2": _proj(2), "Taper": Taper}
