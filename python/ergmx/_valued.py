"""Valued networks, as R's ergm with ergm.count: each dyad has a count (an
edge attribute, ``response``), whose distribution is a reference measure
(Poisson, geometric, binomial, discrete uniform) tilted by the model's
statistics. Models are fitted by contrastive divergence and the Monte Carlo
MLE, as in ergm."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np

from . import _core
from ._network import Network, as_network
from .terms import TERMS, Formula, Term

#: The reference measures, as ergm.count's: name -> number of parameters.
REFERENCES = {"Poisson": 0, "Geometric": 0, "Binomial": 1, "DiscUnif": 2}


def parse_reference(reference) -> tuple[str, list[float]]:
    """A reference measure from R's syntax (``"~Poisson"``,
    ``"Binomial(trials=3)"``, ``"DiscUnif(0, 5)"``) or a (name, parameters) pair."""
    if isinstance(reference, tuple):
        name, params = reference
    else:
        text = str(reference).strip().lstrip("~").strip()
        m = re.fullmatch(r"([A-Za-z]+)\s*(?:\((.*)\))?", text)
        if m is None:
            raise ValueError(f"can't parse the reference {reference!r}")
        name = m.group(1)
        params = []
        for part in (m.group(2) or "").split(","):
            part = part.strip()
            if part:
                params.append(float(part.split("=")[-1]))
    if name not in REFERENCES:
        raise ValueError(f"unknown reference {name!r}; available: {', '.join(REFERENCES)} (and Bernoulli, "
                         "for binary networks)")
    if len(params) != REFERENCES[name]:
        raise ValueError(f"the {name} reference takes {REFERENCES[name]} parameter(s), not {len(params)}")
    if name == "Binomial" and (params[0] < 1 or params[0] != int(params[0])):
        raise ValueError("Binomial(trials): trials must be a positive integer")
    if name == "DiscUnif" and not (params[0] <= 0 <= params[1] and params[0] < params[1]
                                   and all(p == int(p) for p in params)):
        raise ValueError("DiscUnif(a, b): a and b must be integers, a <= 0 <= b, a < b")
    return name, [float(p) for p in params]


# -- Terms --------------------------------------------------------------------------------------


class ValuedTerm(Term):
    """A term of a valued network."""

    valued = True
    dyad_independent = True

    def __init__(self, label: str, rust: str, reals=(), ints=(), directed=None):
        self._label, self.rust, self.reals, self.ints = label, rust, list(reals), list(ints)
        self.directed = directed

    @property
    def label(self) -> str:
        return self._label

    def spec(self, network):
        return (self.rust, [float(x) for x in self.reals], [int(x) for x in self.ints])

    def full_spec(self, network):
        return (*self.spec(network), [])

    def __repr__(self) -> str:
        call = getattr(self, "_call", None)
        if call is None:
            return self._label
        name, args, kwargs = call
        parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        return f"{name}({', '.join(parts)})"


class _Dyadic(ValuedTerm):
    """A dyad-independent binary term of a valued network, as ergm's
    ``form=``: its change statistic of each dyad weighted by the dyad's value
    (``"sum"``) or by whether the value is nonzero (``"nonzero"``)."""

    def __init__(self, inner: Term, form: str, cn: str):
        if form not in ("sum", "nonzero"):
            raise ValueError(f"form must be 'sum' or 'nonzero', not {form!r}")
        if not inner.dyad_independent:
            raise ValueError(f"{inner!r} is not dyad-independent: only dyad-independent terms take form=")
        self.inner, self.form, self.cn = inner, form, cn
        self.directed = inner.directed

    def check(self, network):
        self.inner.check(network)

    def names(self, network):
        return [name.replace(self.cn, f"{self.cn}.{self.form}", 1) for name in self.inner.names(network)]

    def full_spec(self, network):
        return ("dyadic", [], [int(self.form == "nonzero")], [self.inner.full_spec(network)])

    def __repr__(self) -> str:
        return f"{self.inner!r}[{self.form}]"


def _wrapper(name: str, cn: str | None = None):
    factory = TERMS[name]

    def make(*args, form: str = "sum", **kwargs):
        return _Dyadic(factory(*args, **kwargs), form, cn or name)

    make.__name__ = name
    make.__doc__ = f"ergm's valued {name}(..., form='sum' or 'nonzero')."
    return make


def _sum(pow: float = 1):
    """The sum of the dyads' values (to the power ``pow``)."""
    return ValuedTerm("sum" if pow == 1 else f"sum{pow:g}", "sum", [pow])


class _NonZero(_Dyadic):
    def __init__(self):
        super().__init__(TERMS["edges"](), "nonzero", "edges")

    def names(self, network):
        return ["nonzero"]

    def __repr__(self) -> str:
        return "nonzero"


def _thresholds(kind: str):
    def make(threshold=0):
        values = [threshold] if np.isscalar(threshold) else list(threshold)
        term = ValuedTerm(kind, kind, values)
        term.names = lambda network: [f"{kind}.{_num(v)}" for v in values]
        return term

    make.__name__ = kind
    return make


def _num(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def _equalto(value=0, tolerance=0):
    """The number of dyads whose value is within ``tolerance`` of ``value``."""
    term = ValuedTerm(f"equalto.{_num(value)}.pm.{_num(tolerance)}", "ininterval",
                      [value - tolerance, value + tolerance], [0, 0])
    return term


def _ininterval(lower=-np.inf, upper=np.inf, open=(True, True)):
    """The number of dyads whose value is in the interval from ``lower`` to
    ``upper``, open at either end (``open``: two logicals, or "()", "(]"...)."""
    if isinstance(open, str):
        if open not in ("()", "(]", "[)", "[]"):
            raise ValueError("ininterval(): open must be '()', '(]', '[)' or '[]'")
        open = (open[0] == "(", open[1] == ")")
    elif isinstance(open, bool):
        open = (open, open)
    o1, o2 = bool(open[0]), bool(open[1])
    label = f"ininterval{'(' if o1 else '['}{_num(lower)},{_num(upper)}{')' if o2 else ']'}"
    big = sys.float_info.max
    return ValuedTerm(label, "ininterval", [max(lower, -big), min(upper, big)], [int(o1), int(o2)])


def _mutual(form: str = "min", threshold=0):
    """Reciprocity of values (directed networks): the sum over pairs of the
    minimum of y_ij and y_ji (``form="min"``), minus their absolute
    difference (``"nabsdiff"``), their product (``"product"``) or geometric
    mean (``"geometric"``)."""
    codes = {"min": (0, "mutual.min"), "nabsdiff": (1, "mutual.nabsdiff"), "product": (2, "mutual.product"),
             "geometric": (3, "mutual.geom.mean")}
    if form not in codes:
        raise ValueError(f"mutual(form=): one of {', '.join(codes)}")
    code, label = codes[form]
    return ValuedTerm(label, "mutual", [], [code], directed=True)


def _weights(cyclical: bool):
    name = "cyclicalweights" if cyclical else "transitiveweights"

    def make(twopath: str = "min", combine: str = "max", affect: str = "min"):
        for value, options, what in ((twopath, ("min", "geomean"), "twopath"), (combine, ("max", "sum"), "combine"),
                                     (affect, ("min", "geomean"), "affect")):
            if value not in options:
                raise ValueError(f"{name}({what}=): one of {', '.join(options)}")
        term = ValuedTerm(f"{name}.{twopath}.{combine}.{affect}", name,
                          [], [int(twopath == "geomean"), int(combine == "sum"), int(affect == "geomean")])
        term.dyad_independent = False
        term.triadic = True
        return term

    make.__name__ = name
    make.__doc__ = (f"ergm's {name}: for each dyad, its value compared ({{affect}}) with the strongest (or the "
                    "sum, combine) of its two-paths' values (each the minimum or geometric mean of its two).")
    return make


def _nodecovar(side: str):
    name = {"all": "nodecovar", "out": "nodeocovar", "in": "nodeicovar"}[side]

    def make(center: bool = False, transform: str = "identity"):
        if transform not in ("identity", "sqrt"):
            raise ValueError(f"{name}(transform=): 'identity' or 'sqrt'")
        term = ValuedTerm(name, name, [], [int(transform == "sqrt"), int(bool(center))],
                          directed=None if side == "all" else True)
        if side == "all":
            term.directed = False
        term.dyad_independent = False
        return term

    make.__name__ = name
    return make


def _cmp():
    """Conway-Maxwell-Poisson dispersion: the sum over dyads of log(y!)."""
    return ValuedTerm("CMP", "CMP")


class _TiesAbove(ValuedTerm):
    """Valued transitive (cyclical) ties: the dyads whose value is above the
    threshold with a two-path (h -> k -> t, for cyclical ties) whose two
    values are above it too."""

    dyad_independent = False
    triadic = True

    def check(self, network):
        super().check(network)
        if not network.directed:
            import warnings

            from .terms import ErgmDifferenceWarning

            warnings.warn(
                f"{self.rust}(threshold=) of an undirected valued network counts the ties above the "
                "threshold with a shared partner above it, as the binary term and directed networks "
                "do; ergm 4.12 counts more (all 78 ties of zach, where 67 have a shared partner).",
                ErgmDifferenceWarning, stacklevel=5)


def _ties_above(cyclical: bool):
    name = "cyclicalties" if cyclical else "transitiveties"

    def make(threshold=0):
        return _TiesAbove(name, name, [threshold])

    make.__name__ = name
    return make


#: Terms of valued networks, by name: ergm's valued terms, and its
#: dyad-independent binary terms with form= (as `nodematch(attr, form="sum")`).
VALUED_TERMS = {
    "sum": _sum, "nonzero": lambda: _NonZero(), "edges": lambda: _NonZero(),
    "atleast": _thresholds("atleast"), "atmost": _thresholds("atmost"),
    "greaterthan": _thresholds("greaterthan"), "smallerthan": _thresholds("smallerthan"),
    "equalto": _equalto, "ininterval": _ininterval, "mutual": _mutual,
    "transitiveweights": _weights(False), "cyclicalweights": _weights(True),
    "nodecovar": _nodecovar("all"), "nodeocovar": _nodecovar("out"), "nodeicovar": _nodecovar("in"),
    "nodesqrtcovar": lambda center=False: _nodecovar("all")(center, "sqrt"),
    "nodeosqrtcovar": lambda center=False: _nodecovar("out")(center, "sqrt"),
    "nodeisqrtcovar": lambda center=False: _nodecovar("in")(center, "sqrt"),
    "CMP": _cmp, "transitiveties": _ties_above(False), "cyclicalties": _ties_above(True),
}
for _name in ("nodecov", "nodeicov", "nodeocov", "nodefactor", "nodeifactor", "nodeofactor", "nodematch",
              "nodemix", "absdiff", "edgecov", "attrcov", "diff", "sociality", "sender", "receiver",
              "b1cov", "b2cov", "b1factor", "b2factor", "b1sociality", "b2sociality", "mm"):
    if _name in TERMS:
        VALUED_TERMS[_name] = _wrapper(_name)
VALUED_TERMS["absdiffcat"] = _wrapper("absdiffcat", "absdiff")
VALUED_TERMS["nodemain"] = _wrapper("nodemain", "nodecov")
VALUED_TERMS["match"] = VALUED_TERMS["nodematch"]
del _name


def parse_valued_formula(formula) -> Formula:
    """A valued model's terms, from a formula string (with ergm's valued
    term names) or terms."""
    from .formula import parse_formula, use_terms

    if isinstance(formula, Formula):
        terms = list(formula)
    elif isinstance(formula, Term):
        terms = [formula]
    else:
        with use_terms(VALUED_TERMS):
            terms = list(parse_formula(str(formula)))
    for t in terms:
        inner = t.term if t.is_offset else t
        if not getattr(inner, "valued", False):
            raise ValueError(f"{t!r} is a term of binary networks: valued networks take ergm's valued terms "
                             "(sum, nonzero, ... and dyad-independent terms with form=)")
    return Formula(terms)


# -- Networks and models ------------------------------------------------------------------------


def valued_network(network, response: str, bipartite=None) -> tuple[Network, np.ndarray]:
    """The network (its nonzero dyads as edges) and its dyads' values, from
    the edge attribute `response`."""
    ig = sys.modules.get("igraph")
    nx = sys.modules.get("networkx")
    net = as_network(network, bipartite)
    if net.combined:
        raise ValueError("valued models of several networks are not supported")
    if ig is not None and isinstance(network, ig.Graph):
        if response not in network.es.attributes():
            raise ValueError(f"the network has no edge attribute {response!r}")
        pairs = np.array(network.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        values = np.array(network.es[response], dtype=float)
    elif nx is not None and isinstance(network, nx.Graph):
        index = {v: k for k, v in enumerate(network)}
        rows = [(index[u], index[v], d.get(response)) for u, v, d in network.edges(data=True)]
        if any(r[2] is None for r in rows):
            raise ValueError(f"some edges have no attribute {response!r}")
        pairs = np.array([r[:2] for r in rows], dtype=np.int64).reshape(-1, 2)
        values = np.array([r[2] for r in rows], dtype=float)
    else:
        raise TypeError("valued networks are igraph or networkx graphs with an edge attribute")
    if len(net.missing):
        raise ValueError("valued networks with missing dyads are not supported")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError(f"the values of {response!r} must be finite and nonnegative")
    keep = values != 0
    pairs, values = pairs[keep], values[keep]
    if not net.directed:
        pairs = np.sort(pairs, axis=1)
    if len(pairs) != len({tuple(p) for p in pairs.tolist()}):
        raise ValueError("a dyad has several edges: give each dyad one edge, with its value")
    triples = np.column_stack([pairs, values]).astype(float) if len(pairs) else np.zeros((0, 3))
    binary = Network(net.n, net.directed, pairs.astype(np.uint32).reshape(-1, 2), net.attributes, net.source,
                     net.graph_attributes, mode=net.mode)
    return binary, triples


class _ValuedView:
    """The model's network as the estimation code sees it: its edges are
    the rows (i, j, value) of its nonzero dyads; the rest is the binary
    network's (vertices, attributes)."""

    def __init__(self, binary: Network, triples: np.ndarray):
        self._binary, self.edges = binary, triples
        self.missing = np.zeros((0, 2), dtype=np.uint32)
        self.combined, self.series, self.blocks = False, False, None

    def __getattr__(self, name):
        return getattr(self._binary, name)


@dataclass
class ValuedModel:
    """A valued formula bound to a valued network, with the parts of
    :class:`~ergmx._model.BoundModel` the estimation uses."""

    binary: Network
    triples: np.ndarray
    formula: Formula
    names: list[str]
    core: object
    reference: tuple[str, list[float]]
    constraints: object
    fixed: np.ndarray
    fixed_values: np.ndarray
    constant: np.ndarray
    p0: float | None = 0.2
    target: np.ndarray | None = None
    network: _ValuedView = field(init=False)

    def __post_init__(self):
        self.network = _ValuedView(self.binary, self.triples)

    stat_names = property(lambda self: self.names)
    n_stats = property(lambda self: len(self.names))
    n_params = property(lambda self: len(self.names))
    curved = False
    has_missing = False
    exact = False
    valued = True
    free = property(lambda self: ~self.fixed)
    dyad_independent = property(lambda self: all(t.dyad_independent for t in self.formula))

    @property
    def blocks(self):
        out, p = [], 0
        for term in self.formula:
            k = len(term.names(self.binary))
            out.append((term, slice(p, p + k), slice(p, p + k)))
            p += k
        return out

    def term_columns(self):
        return [(t, list(range(q.start, q.stop))) for t, _, q in self.blocks]

    def eta(self, theta):
        return np.asarray(theta, dtype=float).copy()

    def jacobian(self, theta):
        return np.eye(self.n_params)

    def initial(self):
        return np.where(self.fixed, self.fixed_values, 0.0)

    def theta(self, free_values):
        theta = self.fixed_values.copy()
        theta[self.free] = free_values
        return theta

    def triadic_weight(self, requested):
        return 0.0

    def null_loglik(self) -> float | None:
        """The log-likelihood at 0, where the dyads' values are independent
        draws from the reference measure (normalized): None for the
        geometric reference, which can't be normalized."""
        import math

        name, params = self.reference
        d = dyad_count(self.binary)
        values = self.triples[:, 2] if len(self.triples) else np.zeros(0)
        if name == "Poisson":
            return float(-sum(math.lgamma(v + 1) for v in values) - d)
        if name == "Binomial":
            n = params[0]
            return float(sum(math.lgamma(n + 1) - math.lgamma(v + 1) - math.lgamma(n - v + 1) for v in values)
                         - d * n * math.log(2))
        if name == "DiscUnif":
            return float(-d * math.log(params[1] - params[0] + 1))
        return None

    def observed(self):
        if self.target is not None:
            return self.target.copy()
        return np.array(self.core.summary(self.triples))

    @property
    def space(self):
        """The dyads that may change: within the modes of a bipartite network,
        but those the (dyad-independent) constraints fix."""
        from .constraints import _canonical

        net, c = self.binary, self.constraints
        groups = c.groups(net)
        mask = c.fixed(net)
        fixed = c.fixed_pairs(net)
        only = c.free_pairs(net)
        if groups is None and not net.bipartite and mask is None and not len(fixed) and only is None:
            return None
        as_pairs = lambda a: None if a is None else np.ascontiguousarray(a, dtype=np.uint32).reshape(-1, 2)  # noqa: E731
        return _core.Space(net.n, net.directed,
                           None if groups is None else np.ascontiguousarray(groups, dtype=np.int64),
                           None if not net.bipartite else np.ascontiguousarray(net.mode == 1, dtype=bool),
                           None if mask is None else np.ascontiguousarray(mask, dtype=bool).ravel(),
                           as_pairs(_canonical(net, fixed)), as_pairs(only))

    @property
    def n_observations(self) -> int:
        n = self.binary.n
        return n * (n - 1) // (1 if self.binary.directed else 2)

    def simulate(self, starts, theta, burnin, interval, samples, seed, *, conditional=False,
                 canonical=False, max_edges=None, triadic_weight=None, keep_networks=False,
                 chain_thetas=None):
        name, params = self.reference
        eta = [float(t) for t in np.asarray(theta, dtype=float)]
        return self.core.simulate([np.ascontiguousarray(s, dtype=float) for s in starts], eta, burnin, interval,
                                  samples, seed, (name, params), self.p0, keep_networks=keep_networks,
                                  max_nonzero=max_edges, chain_thetas=chain_thetas, space=self.space)


def dyad_count(network: Network) -> float:
    if network.bipartite:
        n1 = int(np.sum(network.mode == 1))
        return float(n1 * (network.n - n1))
    return network.n * (network.n - 1) / (1 if network.directed else 2)


def bind_valued(network, formula, response: str, reference="Poisson", *, constraints=None,
                offset_coef=None, bipartite=None, fitting=False, p0=0.2) -> ValuedModel:
    """A valued model: the formula's valued terms on the network's values."""
    from ._model import _make_unique
    from .constraints import parse_constraints

    binary, triples = valued_network(network, response, bipartite)
    formula = parse_valued_formula(formula)
    for term in formula:
        term.check(binary)
    constraints = parse_constraints(constraints)
    if constraints.dyad_dependent:
        raise ValueError(f"valued models take dyad-independent constraints only, not {constraints!r}")
    constraints.check(binary)
    names = _make_unique([n for t in formula for n in t.names(binary)])
    specs = [t.full_spec(binary) for t in formula]
    core = _core.WtModel(binary.n, binary.directed, specs, dyad_count(binary))
    if core.n_stats != len(names):
        raise RuntimeError(f"internal error: {core.n_stats} statistics, {len(names)} names")
    offsets = np.zeros(len(names), dtype=bool)
    start = 0
    for term in formula:
        k = len(term.names(binary))
        if term.is_offset:
            offsets[start:start + k] = True
        start += k
    values = np.zeros(len(names))
    if offsets.any() and (fitting or offset_coef is not None):
        if offset_coef is None:
            raise ValueError(f"give the values of the offset coefficients with offset_coef= ({int(offsets.sum())})")
        coef = np.atleast_1d(np.asarray(offset_coef, dtype=float))
        if coef.shape != (offsets.sum(),):
            raise ValueError(f"offset_coef must have {int(offsets.sum())} values")
        values[offsets] = coef
    return ValuedModel(binary, triples, formula, names, core, parse_reference(reference), constraints,
                       offsets, values, np.zeros(len(names), dtype=bool), p0)


# -- Fitting and simulating ---------------------------------------------------------------------


def to_valued_graphs(model: ValuedModel, triples: np.ndarray, response: str):
    """A graph like the model's, with the nonzero dyads as edges and their
    values in the edge attribute `response`."""
    from ._network import to_graph

    t = np.asarray(triples, dtype=float).reshape(-1, 3)
    g = to_graph(model.binary, t[:, :2].astype(np.uint32))
    values = t[:, 2].tolist()
    if hasattr(g, "es"):
        g.es[response] = values
    else:
        import networkx as nx

        nodes = list(g)
        nx.set_edge_attributes(g, {(nodes[int(i)], nodes[int(j)]): v for (i, j), v in zip(t[:, :2], values)},
                               response)
    return g


def simulate_valued(model: ValuedModel, coef, nsim: int, seed, burnin, interval, output: str, response: str):
    from ._estimation import Control

    defaults = Control()
    seed = int(np.random.default_rng(seed).integers(2**63))
    sample, _, networks = model.simulate(
        [model.triples], coef, defaults.burnin if burnin is None else burnin,
        defaults.interval if interval is None else interval, nsim, seed, keep_networks=output == "network")
    if output == "stats":
        return sample[0]
    return [to_valued_graphs(model, t, response) for t in networks[0]]


def fit_valued(model: ValuedModel, estimate: str, init, control, rng, seed, eval_loglik: bool):
    """Contrastive divergence and the Monte Carlo MLE of a valued model, as
    ergm's (whose valued models start from contrastive divergence)."""
    from . import _estimation
    from ._estimation import Estimate
    from ._fit import ErgmFit

    if estimate == "MPLE":
        raise ValueError("valued models have no MPLE: use estimate='MLE' or 'CD'")
    zeros = model.initial()
    nan = np.full((model.n_params, model.n_params), np.nan)
    pseudo = Estimate(zeros, nan, None, None, "zeros", 0, False, None)
    if init is None or (isinstance(init, str) and init == "CD"):
        cd = _estimation.contrastive_divergence(model, zeros, control, rng)
        if estimate == "CD":
            return ErgmFit(model, cd, pseudo, control, seed)
        start = cd.theta
    elif isinstance(init, str) and init == "zeros":
        start = zeros
    elif isinstance(init, str):
        raise ValueError(f"init must be 'CD', 'zeros' or coefficients, not {init!r}")
    else:
        start = np.asarray(init, dtype=float)
        if start.shape != (model.n_params,):
            raise ValueError(f"init must have {model.n_params} values, one per coefficient")
        start = np.where(model.fixed, model.fixed_values, start)
    result = _estimation.mcmle(model, start, control, rng)
    if eval_loglik and model.null_loglik() is not None:
        import dataclasses

        from ._loglik import bridge_loglik

        loglik, se, _ = bridge_loglik(model, result.theta, control, result.interval, rng)
        result = dataclasses.replace(result, loglik=loglik, loglik_se=se)
    return ErgmFit(model, result, pseudo, control, seed)
