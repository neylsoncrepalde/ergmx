"""Bootstrapped pseudo-likelihood TERGMs, as R's btergm (`Leifeld, Cranmer
and Desmarais 2018 <https://doi.org/10.18637/jss.v083.i06>`__): the
pseudo-likelihood of each network of a series, given the ones before it
through lagged covariates (``memory()``, ``delrecip()``) and time
(``timecov()``), pooled over the time steps, with confidence intervals from a
bootstrap of the time steps."""

from __future__ import annotations

import sys
import warnings

import numpy as np
from scipy.stats import norm

from ._estimation import logistic_regression
from ._model import _make_unique, bind
from ._network import Network, as_network
from .terms import EdgeCov, Formula, Term, as_formula


class _Temporal(Term):
    """A btergm term of the networks before (memory, delrecip) or of time
    (timecov): a dyadic covariate of each time step's network."""

    temporal = True
    lag = 0

    def covariate(self, step: _Step) -> np.ndarray:
        raise NotImplementedError

    def spec(self, network):
        raise ValueError(f"{self!r} is a term of btergm(): it needs the networks before")


class Memory(_Temporal):
    """btergm's memory(type, lag): the tie at time t - lag, as "autoregression"
    (1 if a tie, 0 if not), "stability" (1 or -1), "innovation" (1 if not a
    tie, 0 if one) or "loss" (-1 if a tie, 0 if not)."""

    TYPES = {"autoregression": (1.0, 0.0), "stability": (1.0, -1.0), "innovation": (0.0, 1.0), "loss": (-1.0, 0.0)}

    def __init__(self, type: str = "stability", lag: int = 1):
        if type not in self.TYPES:
            raise ValueError(f"memory(type=): one of {', '.join(self.TYPES)}, not {type!r}")
        if int(lag) != lag or lag < 1:
            raise ValueError("memory(lag=): a whole number, 1 or more")
        self.type, self.lag = type, int(lag)

    @property
    def label(self) -> str:
        return f"memory.{self.type}" + (f".lag{self.lag}" if self.lag != 1 else "")

    def covariate(self, step):
        tie, no_tie = self.TYPES[self.type]
        y = step.lagged(self.lag)
        return np.where(y == 1, tie, no_tie)

    def __repr__(self) -> str:
        return f"memory(type={self.type!r}, lag={self.lag})"


class DelRecip(_Temporal):
    """btergm's delrecip(mutuality, lag): delayed reciprocity, the tie j -> i
    at time t - lag for the dyad i -> j (with ``mutuality``, -1 rather than 0
    without it)."""

    def __init__(self, mutuality: bool = False, lag: int = 1):
        if int(lag) != lag or lag < 1:
            raise ValueError("delrecip(lag=): a whole number, 1 or more")
        self.mutuality, self.lag = bool(mutuality), int(lag)

    @property
    def label(self) -> str:
        return ("delrecip.mutuality" if self.mutuality else "delrecip") + (f".lag{self.lag}" if self.lag != 1 else "")

    def covariate(self, step):
        y = step.lagged(self.lag).T
        return np.where(y == 1, 1.0, -1.0 if self.mutuality else 0.0)

    def __repr__(self) -> str:
        return f"delrecip(mutuality={self.mutuality}, lag={self.lag})"


class TimeCov(_Temporal):
    """btergm's timecov(x, minimum, maximum, transform): the time step t (1 for
    the first network), transformed and 0 outside [minimum, maximum], for
    every dyad, or times the dyadic covariate ``x`` (each network's graph
    attribute)."""

    def __init__(self, x=None, minimum: int = 1, maximum: int | None = None, transform=None):
        self.x, self.minimum, self.maximum = x, int(minimum), None if maximum is None else int(maximum)
        if transform is None:
            transform = "t"
        self.transform = transform

    @property
    def label(self) -> str:
        return "timecov" if self.x is None else f"timecov.{self.x}"

    def value(self, t: int) -> float:
        """The transformed time step (1-based), 0 outside [minimum, maximum]."""
        if callable(self.transform):
            v = float(self.transform(t))
        else:
            namespace = {"t": float(t), "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs}
            v = float(eval(str(self.transform).replace("^", "**"), {"__builtins__": {}}, namespace))
        if not np.isfinite(v):
            raise ValueError(f"timecov(transform=) gives {v} at time step {t}")
        upper = self.maximum if self.maximum is not None else float("inf")
        return v if self.minimum <= t <= upper else 0.0

    def covariate(self, step):
        v = self.value(step.time)
        if self.x is None:
            return np.full((step.network.n, step.network.n), v)
        return v * step.matrix(self.x)

    def __repr__(self) -> str:
        return f"timecov({self.x!r})" if self.x is not None else "timecov()"


#: btergm's terms, by name, in formula strings (with ergm's terms).
BTERGM_TERMS = {"memory": Memory, "delrecip": DelRecip, "timecov": TimeCov}


def memory(type: str = "stability", lag: int = 1) -> Term:
    """btergm's memory term: the tie at time t - ``lag`` (see :func:`btergm`)."""
    return Memory(type, lag)


def delrecip(mutuality: bool = False, lag: int = 1) -> Term:
    """btergm's delayed reciprocity: the reverse tie at time t - ``lag`` (see :func:`btergm`)."""
    return DelRecip(mutuality, lag)


def timecov(x=None, minimum: int = 1, maximum: int | None = None, transform=None) -> Term:
    """btergm's time covariate: the (transformed) time step, alone or times a
    dyadic covariate (see :func:`btergm`). ``transform`` is a function of the
    time step, or an expression in ``t`` (``"t^2"``, ``"log(t)"``)."""
    return TimeCov(x, minimum, maximum, transform)


class _Covariate(EdgeCov):
    """A temporal term's covariate at one time step, with the term's name."""

    def __init__(self, matrix: np.ndarray, label: str):
        super().__init__(matrix)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def __repr__(self) -> str:
        return self._label


# -- The time steps, with their vertices ----------------------------------------------------------


def _names(network: Network) -> list | None:
    """The vertices' names: igraph's "name" attribute, or networkx's nodes."""
    if "name" in network.attributes and all(v is not None for v in network.attributes["name"]):
        return list(network.attributes["name"])
    nx = sys.modules.get("networkx")
    if nx is not None and isinstance(network.source, nx.Graph):
        return list(network.source)
    return None


def _adjacency(network: Network) -> np.ndarray:
    a = np.zeros((network.n, network.n))
    e = network.edges.astype(int)
    a[e[:, 0], e[:, 1]] = 1
    if not network.directed:
        a[e[:, 1], e[:, 0]] = 1
    return a


class _Step:
    """One time step: its network, on the vertices btergm keeps (those in all
    its objects, or, with ``offset``, in any, the others' dyads left out as
    structural zeros), and the networks before it on the same vertices."""

    def __init__(self, networks: list[Network], names: list[list] | None, k: int, lags: list[int], offset: bool):
        self.time = k + 1
        own = networks[k]
        if names is None:  # the same vertices, in the same order
            self.vertices = list(range(own.n))
            index = [list(range(own.n))] * len(networks)
            present = [np.ones(own.n, dtype=bool)] * len(networks)
        else:
            sets = [names[k]] + [names[k - lag] for lag in lags]
            if offset:
                vertices = list(dict.fromkeys(v for s in sets for v in s))
            else:
                common = set(sets[0]).intersection(*map(set, sets[1:]))
                vertices = [v for v in names[k] if v in common]
            self.vertices = vertices
            index, present = [], []
            for labels in names:
                position = {v: p for p, v in enumerate(labels)}
                index.append([position.get(v, -1) for v in vertices])
                present.append(np.array([v in position for v in vertices]))
        self._networks, self._index, self._present, self._k = networks, index, present, k
        n = len(self.vertices)
        # The structural zeros: vertices absent from the network or a lagged one.
        absent = ~present[k]
        for lag in lags:
            absent |= ~present[k - lag]
        self.absent = absent
        self.network = self._restrict(own, index[k], n, absent)

    def _restrict(self, net: Network, index: list[int], n: int, absent: np.ndarray) -> Network:
        """`net` on the step's vertices, with the dyads of absent ones missing."""
        new = {old: p for p, old in enumerate(index) if old >= 0}

        def mapped(pairs):
            rows = [(new[i], new[j]) for i, j in pairs.astype(int).tolist() if i in new and j in new]
            return np.array(rows, dtype=np.uint32).reshape(-1, 2)

        edges, missing = mapped(net.edges), mapped(net.missing)
        if absent.any():
            out = np.flatnonzero(absent)
            others = np.arange(n)
            pairs = {(min(a, b), max(a, b)) if not net.directed else (a, b)
                     for v in out for w in others if w != v for a, b in ((v, w), (w, v))}
            missing = np.vstack([missing, np.array(sorted(pairs), dtype=np.uint32).reshape(-1, 2)])
            if not net.directed:
                missing = np.unique(np.sort(missing, axis=1), axis=0).astype(np.uint32)
            else:
                missing = np.unique(missing, axis=0).astype(np.uint32)
        attributes = {}
        for name, values in net.attributes.items():
            attributes[name] = [values[old] if old >= 0 else self._attribute(name, p) for p, old in enumerate(index)]
        graph = {}
        for name, value in net.graph_attributes.items():
            m = np.asarray(value) if not np.isscalar(value) else None
            if m is not None and m.ndim == 2 and m.shape == (net.n, net.n):
                full = np.zeros((n, n))
                keep = np.array([old >= 0 for old in index])
                rows = np.array([old for old in index if old >= 0], dtype=int)
                full[np.ix_(keep, keep)] = m[np.ix_(rows, rows)]
                graph[name] = full
            else:
                graph[name] = value
        return Network(n, net.directed, edges, attributes, net.source, graph, missing)

    def _attribute(self, name: str, position: int):
        """A vertex attribute of a vertex absent from the step's network: from
        another network where it is present (its dyads are left out anyway)."""
        for net, index in zip(self._networks, self._index):
            old = index[position]
            if old >= 0 and name in net.attributes:
                return net.attributes[name][old]
        return None

    def lagged(self, lag: int) -> np.ndarray:
        """The network at time t - lag on the step's vertices (1 for absent vertices' dyads, as btergm's)."""
        k = self._k - lag
        if k < 0:
            raise ValueError(f"lag {lag} needs a network {lag} time steps before the first one modelled")
        net, index = self._networks[k], self._index[k]
        a = _adjacency(net)
        n = len(self.vertices)
        out = np.ones((n, n))
        keep = np.array([old >= 0 for old in index])
        rows = np.array([old for old in index if old >= 0], dtype=int)
        out[np.ix_(keep, keep)] = a[np.ix_(rows, rows)]
        return out

    def matrix(self, x) -> np.ndarray:
        """The step network's dyadic covariate `x` (a graph attribute, or a matrix)."""
        from .terms import _matrix_argument

        return _matrix_argument(self.network, x, "timecov")[0]


# -- Fitting ------------------------------------------------------------------------------------


def btergm(networks, formula, *, R: int = 500, offset: bool = False, seed=None) -> BtergmFit:
    """A temporal ERGM by bootstrapped pseudo-likelihood, as R's btergm.

    Each network of the series is modelled given the ones before it: the
    model's terms are evaluated on it, and btergm's temporal terms are dyadic
    covariates of the networks before and of time. The estimate maximizes the
    pseudo-likelihood pooled over the time steps (the networks after the
    largest lag); the confidence intervals come from ``R`` bootstrap samples
    of the time steps, with replacement.

    Parameters
    ----------
    networks : list of igraph.Graph or networkx.Graph
        The networks, in time order. Their vertices are matched by name
        (igraph's ``name`` attribute, networkx's nodes), so they may differ:
        each time step keeps the vertices of its network that are also in
        the lagged networks it needs (or, with ``offset``, all of them, the
        dyads of those absent from one of them left out). Without names, the
        networks must have the same vertices, in the same order.
    formula : str or terms
        ergm's terms (dyadic covariates, ``edgecov("x")``, are each network's
        graph attribute ``x``), and btergm's:

        - ``memory(type="stability", lag=1)``: the dyad's tie at time t - lag,
          as "autoregression" (1 if a tie, 0 if not), "stability" (1 or -1),
          "innovation" (1 if not a tie, 0 if one) or "loss" (-1 if a tie,
          0 if not);
        - ``delrecip(mutuality=False, lag=1)``: the reverse tie at time
          t - lag (1 if a tie, 0, or -1 with ``mutuality``, if not);
        - ``timecov(x=None, minimum=1, maximum=None, transform="t")``: the
          time step t (1 for the first network), transformed (a function, or
          an expression in ``t`` such as ``"t^2"``) and 0 outside [minimum,
          maximum], alone or times the dyadic covariate ``x``.
    R : int
        Bootstrap replications.
    offset : bool
        Keep the vertices absent from some network of a time step, as
        structural zeros (btergm's ``offset=TRUE``), rather than drop them.
    seed : int, optional
        For the bootstrap.

    Returns
    -------
    BtergmFit
    """
    from .formula import parse_formula, use_terms
    from .terms import TERMS

    graphs = list(networks)
    if len(graphs) < 2:
        raise ValueError("btergm() needs a series of at least 2 networks")
    nets = [as_network(g) for g in graphs]
    if len({n.directed for n in nets}) > 1 or any(n.combined or n.bipartite for n in nets):
        raise ValueError("btergm() takes networks that are all directed or all undirected, one-mode")
    if isinstance(formula, str):
        with use_terms({**TERMS, **BTERGM_TERMS}):
            formula = parse_formula(formula)
    formula = as_formula(formula)
    for term in formula:
        if term.is_offset:
            raise ValueError("btergm() takes no offset() terms")
    lags = sorted({t.lag for t in formula if getattr(t, "temporal", False) and t.lag})
    first = max(lags, default=0)
    if first >= len(nets):
        raise ValueError(f"a lag of {first} leaves no network to model, of {len(nets)}")
    names = [_names(n) for n in nets]
    if all(v is None for v in names) or len({tuple(v) for v in names if v is not None}) == 1 and None not in names:
        if len({n.n for n in nets}) > 1:
            raise ValueError("networks of different sizes need vertex names, to be matched")
        names = None
    elif any(v is None for v in names):
        raise ValueError("either every network has vertex names, or none does")
    xs, ys, ws, times, term_names, nobs = [], [], [], [], None, 0
    for k in range(first, len(nets)):
        step = _Step(nets, names, k, lags, offset)
        terms = [_Covariate(t.covariate(step), t.label) if getattr(t, "temporal", False) else t for t in formula]
        model = bind(step.network, Formula(terms))
        if model.curved:
            raise ValueError("btergm() takes no curved terms: fix their decays (fixed=TRUE)")
        x, y, w = model.mple_table()
        if term_names is None:
            term_names = list(model.names)
        xs.append(x), ys.append(y), ws.append(w), times.append(np.full(len(y), len(times)))
        nobs += int(w.sum())
    x, y, w, time = np.vstack(xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(times)
    coef, cov, loglik = logistic_regression(x, y, w)
    steps = len(xs)
    rng = np.random.default_rng(seed)
    boot = np.full((int(R), len(coef)), np.nan)
    for r in range(int(R)):
        counts = np.bincount(rng.integers(steps, size=steps), minlength=steps)
        weights = w * counts[time]
        used = weights > 0
        if np.linalg.matrix_rank(x[used] * np.sqrt(weights[used])[:, None]) < x.shape[1]:
            continue  # a coefficient can't be estimated from this resample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot[r] = logistic_regression(x[used], y[used], weights[used])[0]
    return BtergmFit(_make_unique(term_names), coef, cov, loglik, boot, steps, nobs, formula, offset,
                     (x, y, w, time))


def _norm_inter(t: np.ndarray, alpha: float) -> float:
    """boot's norm.inter: the alpha quantile of the replicates t, interpolated
    between order statistics on the normal quantile scale."""
    t = np.sort(t[np.isfinite(t)])
    n = len(t)
    rk = (n + 1) * alpha
    k = int(np.trunc(rk))
    if k == rk and 0 < k <= n:
        return float(t[k - 1])
    if k <= 0:
        return float(t[0])
    if k >= n:
        return float(t[-1])
    z, zk, zk1 = norm.ppf(alpha), norm.ppf(k / (n + 1)), norm.ppf((k + 1) / (n + 1))
    return float(t[k - 1] + (z - zk) / (zk1 - zk) * (t[k] - t[k - 1]))


class BtergmFit:
    """A fit of :func:`ergmx.btergm`: the pooled pseudo-likelihood estimates,
    and the bootstrap replicates (``boot``, replicates x coefficients; NaN
    rows where a coefficient couldn't be estimated)."""

    def __init__(self, names, coef, cov, loglik, boot, time_steps, nobs, formula, offset, data):
        self.names = list(names)
        self.params = np.asarray(coef, dtype=float)
        self.boot = boot
        #: The number of time steps modelled, and of dyads in the pseudo-likelihood.
        self.time_steps, self.nobs = time_steps, nobs
        self.formula, self.offset = formula, offset
        self.pseudo_loglik = loglik
        self._cov, self._data = cov, data

    @property
    def coef(self) -> dict[str, float]:
        return dict(zip(self.names, self.params.tolist()))

    @property
    def R(self) -> int:  # noqa: N802 (R's name)
        return len(self.boot)

    def _complete(self) -> np.ndarray:
        complete = self.boot[np.all(np.isfinite(self.boot), axis=1)]
        dropped = len(self.boot) - len(complete)
        if dropped:
            warnings.warn(f"{dropped} of {len(self.boot)} bootstrap replications ({100 * dropped / len(self.boot):.1f}%) "
                          "couldn't estimate every coefficient (too little variation in their time steps) "
                          "and are left out, as btergm does", stacklevel=3)
        return complete

    def confint(self, level: float = 0.95):
        """Bootstrap percentile intervals, as btergm's (boot's ``perc``), with
        the estimates and the bootstrap means."""
        from ._interpret import NumericTable

        complete = self._complete()
        alpha = (1 - level) / 2
        low = [_norm_inter(complete[:, k], alpha) for k in range(len(self.names))]
        high = [_norm_inter(complete[:, k], 1 - alpha) for k in range(len(self.names))]
        return NumericTable(f"Estimates and {100 * level:g}% bootstrap confidence intervals", self.names,
                            {"Estimate": self.params, "Boot mean": complete.mean(axis=0),
                             f"{100 * alpha:g}%": np.array(low), f"{100 * (1 - alpha):g}%": np.array(high)})

    def summary(self, level: float = 0.95) -> str:
        table = self.confint(level)
        return "\n".join([
            "Bootstrapped pseudo-likelihood TERGM (btergm)", "",
            f"Time steps: {self.time_steps}; dyads: {self.nobs}; bootstrap replications: {self.R}"
            + ("; absent vertices as structural zeros" if self.offset else ""), "", str(table)])

    def to_frame(self, level: float = 0.95):
        return self.confint(level).to_frame()

    def __repr__(self) -> str:
        return self.summary()
