"""Egocentric ERGMs, as R's ergm.ego: models of a population network
estimated from a sample of egos, each with its alters and the ties among
them (Krivitsky and Morris 2017, <https://doi.org/10.1214/16-AOAS1010>).

The data's statistics estimate the population's (each ego's contribution,
scaled to the population); the model is fitted to them on a
pseudo-population network of egos' replicates, with an offset that makes
the coefficients those of a network of the population's size; and the
standard errors come from the egos' sampling variance."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .terms import (
    AbsDiff,
    AbsDiffCat,
    Concurrent,
    ConcurrentTies,
    Curved,
    Degree,
    Edges,
    Esp,
    Formula,
    GwDegree,
    Gwesp,
    MeanDeg,
    NodeCov,
    NodeFactor,
    NodeMatch,
    NodeMix,
    Offset,
    Term,
    TransitiveTies,
    Triangle,
    _DegreePower,
    _DegreeRange,
    as_formula,
)


def _columns(table) -> dict[str, list]:
    """A table's columns: a pandas DataFrame, a dict of columns or a list of records."""
    if table is None:
        return {}
    if hasattr(table, "to_dict") and hasattr(table, "columns"):  # pandas
        return {str(c): list(table[c]) for c in table.columns}
    if isinstance(table, dict):
        return {str(k): list(v) for k, v in table.items()}
    rows = list(table)
    names = list(dict.fromkeys(k for r in rows for k in r))
    return {k: [r.get(k) for r in rows] for k in names}


class EgoData:
    """Egocentric data, as ergm.ego's (egor's): egos with their attributes;
    their alters, each with the ego who named it and its attributes; and,
    optionally, the ties among each ego's alters.

    Parameters
    ----------
    egos : pandas DataFrame, dict of columns or list of records
        One row per ego. ``ego_id`` names its identifier column (by default
        ``".egoID"``; without one, egos are numbered from 1).
    alters : same
        One row per alter, with the ``ego_id`` of the ego who named it. For
        the ties among alters, alters need identifiers, in ``alter_id``
        (``".altID"``), unique within each ego.
    aaties : same, optional
        One row per tie between two alters of an ego: its ``ego_id``, and the
        alters' identifiers in ``".srcID"`` and ``".tgtID"``.
    weights : str or array-like, optional
        The egos' sampling weights: a column of ``egos``, or one per ego.
    """

    def __init__(self, egos, alters, aaties=None, *, ego_id: str = ".egoID", alter_id: str = ".altID",
                 weights=None):
        egos, alters, aaties = _columns(egos), _columns(alters), _columns(aaties)
        n = len(next(iter(egos.values()))) if egos else 0
        ids = egos.pop(ego_id, list(range(1, n + 1)))
        position = {e: k for k, e in enumerate(ids)}
        if len(position) != n:
            raise ValueError("EgoData: the egos' identifiers must be unique")
        if ego_id not in alters:
            raise ValueError(f"EgoData: the alters need the column {ego_id!r} of the ego who named them")
        try:
            owner = np.array([position[e] for e in alters.pop(ego_id)], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"EgoData: an alter's ego {e.args[0]!r} is not among the egos") from None
        self.ego_ids = list(ids)
        self.alter_ids = alters.pop(alter_id, None)
        if isinstance(weights, str):
            weights = egos.pop(weights)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and (self.weights.shape != (n,) or np.any(self.weights <= 0)):
            raise ValueError("EgoData: weights must be one positive number per ego")
        self.egos, self.alters, self.owner = egos, alters, owner
        self.n = n
        self.degree = np.bincount(owner, minlength=n)
        # Ties among each ego's alters, as pairs of alter rows.
        self.aatie_owner = np.zeros(0, dtype=np.int64)
        self.aatie_pairs = np.zeros((0, 2), dtype=np.int64)
        self.has_aaties = bool(aaties)
        if aaties:
            if self.alter_ids is None:
                raise ValueError(f"EgoData: ties among alters need the alters' identifiers ({alter_id!r})")
            row = {(o, a): k for k, (o, a) in enumerate(zip(owner.tolist(), self.alter_ids))}
            pairs = set()
            for e, s, t in zip(aaties[ego_id], aaties[".srcID"], aaties[".tgtID"]):
                o = position[e]
                a, b = row[(o, s)], row[(o, t)]
                if a != b:
                    pairs.add((o, min(a, b), max(a, b)))
            pairs = sorted(pairs)
            self.aatie_owner = np.array([p[0] for p in pairs], dtype=np.int64)
            self.aatie_pairs = np.array([p[1:] for p in pairs], dtype=np.int64).reshape(-1, 2)

    @classmethod
    def from_network(cls, network) -> EgoData:
        """The egocentric census of a network, as egor's ``as.egor()``: every
        vertex an ego, its neighbours its alters, and the ties among them."""
        from ._network import as_network

        net = as_network(network)
        if net.directed:
            raise ValueError("EgoData.from_network: egocentric data are of undirected networks")
        names = [a for a in net.attributes if a not in ("name", "id", "na")]
        egos = {".egoID": list(range(1, net.n + 1)), **{a: list(net.attributes[a]) for a in names}}
        neighbours: list[list[int]] = [[] for _ in range(net.n)]
        for i, j in net.edges.tolist():
            neighbours[i].append(j)
            neighbours[j].append(i)
        alters: dict[str, list] = {".egoID": [], ".altID": [], **{a: [] for a in names}}
        aaties: dict[str, list] = {".egoID": [], ".srcID": [], ".tgtID": []}
        ties = {tuple(sorted(e)) for e in net.edges.tolist()}
        for i, nbrs in enumerate(neighbours):
            nbrs = sorted(nbrs)
            for j in nbrs:
                alters[".egoID"].append(i + 1)
                alters[".altID"].append(j + 1)
                for a in names:
                    alters[a].append(net.attributes[a][j])
            for x in range(len(nbrs)):
                for y in range(x + 1, len(nbrs)):
                    if (nbrs[x], nbrs[y]) in ties:
                        aaties[".egoID"].append(i + 1)
                        aaties[".srcID"].append(nbrs[x] + 1)
                        aaties[".tgtID"].append(nbrs[y] + 1)
        return cls(egos, alters, aaties)

    def sample(self, egos) -> EgoData:
        """The data of some egos (positions, from 0), with their alters and the ties among them."""
        keep = np.asarray(egos, dtype=np.int64)
        new = {old: k for k, old in enumerate(keep.tolist())}
        rows = [k for k, o in enumerate(self.owner.tolist()) if o in new]
        out = object.__new__(EgoData)
        out.egos = {a: [v[i] for i in keep] for a, v in self.egos.items()}
        out.alters = {a: [v[r] for r in rows] for a, v in self.alters.items()}
        out.ego_ids = [self.ego_ids[i] for i in keep]
        out.alter_ids = None if self.alter_ids is None else [self.alter_ids[r] for r in rows]
        out.weights = None if self.weights is None else self.weights[keep]
        out.owner = np.array([new[self.owner[r]] for r in rows], dtype=np.int64)
        out.n = len(keep)
        out.degree = np.bincount(out.owner, minlength=out.n)
        rowmap = {r: k for k, r in enumerate(rows)}
        kept = [k for k, o in enumerate(self.aatie_owner.tolist()) if o in new]
        out.aatie_owner = np.array([new[self.aatie_owner[k]] for k in kept], dtype=np.int64)
        out.aatie_pairs = np.array([[rowmap[a], rowmap[b]] for a, b in self.aatie_pairs[kept].tolist()],
                                   dtype=np.int64).reshape(-1, 2)
        out.has_aaties = self.has_aaties
        return out

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"<EgoData: {self.n} egos, {len(self.owner)} alters"
                + (f", {len(self.aatie_owner)} ties among alters" if self.has_aaties else "")
                + (", weighted" if self.weights is not None else "") + ">")

    # -- Helpers of the statistics ---------------------------------------------------------

    def shared_partners(self) -> np.ndarray:
        """For each alter (row), the number of the ego's other alters it is tied to."""
        sp = np.zeros(len(self.owner), dtype=np.int64)
        if len(self.aatie_pairs):
            np.add.at(sp, self.aatie_pairs[:, 0], 1)
            np.add.at(sp, self.aatie_pairs[:, 1], 1)
        return sp

    def per_ego(self, values: np.ndarray) -> np.ndarray:
        """Sums over each ego's alters (rows x columns of values)."""
        values = np.asarray(values, dtype=float)
        out = np.zeros((self.n,) + values.shape[1:])
        np.add.at(out, self.owner, values)
        return out


#: The orders of ergm.ego's statistics: 1 for dyadic ones (the size
#: adjustment's edges), 3 for triadic ones (its transitive ties), 0 for those
#: that don't scale with the population.
_DYADIC = (Edges, NodeCov, NodeFactor, NodeMatch, NodeMix, AbsDiff, AbsDiffCat)
_DEGREE = (Degree, _DegreeRange, Concurrent, ConcurrentTies, _DegreePower, GwDegree)
_TRIADIC = (Esp, Gwesp, TransitiveTies, Triangle)


def _supported(term: Term) -> bool:
    inner = term.term if isinstance(term, Curved) else term
    exact = type(inner) in (*_DYADIC, *_DEGREE, *_TRIADIC, MeanDeg)
    if type(inner) in (Esp, Gwesp) and getattr(inner, "type", "OTP") != "OTP":
        return False
    if type(inner) is _DegreeRange and inner.mode is not None:
        return False
    if type(inner) in (TransitiveTies,) and inner.attr is not None:
        return False
    return exact and (not isinstance(term, Curved) or type(inner) in (GwDegree, Gwesp))


def _pairs_network(data: EgoData, attributes: list[str]):
    """The egos and their alters as one network, with a tie from each ego to
    each of its alters: the dyadic statistics' change statistics are those of
    these ties."""
    from ._network import Network

    n_alters = len(data.owner)
    attrs = {a: list(data.egos[a]) + list(data.alters[a]) for a in attributes}
    edges = np.column_stack([data.owner, data.n + np.arange(n_alters)]).astype(np.uint32)
    return Network(data.n + n_alters, False, edges.reshape(-1, 2), attrs)


def _dyadic(term: Term, data: EgoData, population) -> tuple[list[str], np.ndarray]:
    """Each ego's contribution to a dyadic statistic: half the sum over its
    alters of the statistic of its tie to them (each tie is seen from both
    ends in the population). nodecov and nodefactor without the attribute on
    alters use the egos' only: their value times their degree."""
    from ._model import bind

    names = term.names(population)
    attr = getattr(term, "attr", None)
    if attr is not None and attr not in data.egos:
        raise ValueError(f"{term!r}: the egos have no attribute {attr!r}")
    if attr is not None and attr not in data.alters:
        if isinstance(term, NodeCov):
            x = np.array(data.egos[attr], dtype=float)
            return names, (x * data.degree)[:, None]
        if isinstance(term, NodeFactor):
            chosen, _ = term._levels(population)
            values = data.egos[attr]
            h = np.array([[float(v == level) for level in chosen] for v in values])
            return names, h * data.degree[:, None]
        raise ValueError(f"{term!r} needs the attribute {attr!r} on the alters too")
    attributes = [attr] if attr is not None else []
    pairs = _pairs_network(data, attributes)
    if attr is not None:
        ego_levels = set(data.egos[attr])
        stray = sorted({v for v in data.alters[attr] if v not in ego_levels}, key=str)
        if stray and not isinstance(term, (NodeCov, AbsDiff)):
            raise ValueError(f"{term!r}: alters have values of {attr!r} that no ego has ({stray}); "
                             "the egos' and alters' levels must agree, as in ergm.ego")
    if not len(pairs.edges):
        return names, np.zeros((data.n, len(names)))
    model = bind(pairs, Formula([term]))
    if model.stat_names != list(names):
        raise ValueError(f"{term!r}: the egos' and alters' levels must agree, as in ergm.ego")
    space = dataclasses.replace(model, constraints=_free(pairs.edges))._space()
    x, _, which = model.core.mple_data(np.zeros((0, 2), dtype=np.uint32), space)
    # Each row is a tie of an ego (the lower vertex) to an alter.
    alter_row = which[:, 1].astype(np.int64) - data.n
    values = np.zeros((len(data.owner), len(names)))
    values[alter_row] = x
    return names, data.per_ego(values) / 2


def _free(edges):
    from .constraints import Constraints, Fixallbut

    return Constraints([Fixallbut(edges + 1)])


def _degree_counts(term: Term, data: EgoData, population) -> tuple[list[str], np.ndarray]:
    """Each ego's contribution to a statistic of the degree distribution:
    its own vertex's."""
    names = term.names(population)
    d = data.degree.astype(float)
    if isinstance(term, Degree):
        return names, np.column_stack([d == k for k in term.ks]).astype(float)
    if isinstance(term, Concurrent):
        return names, (d >= 2).astype(float)[:, None]
    if isinstance(term, _DegreePower):
        return names, (d**1.5)[:, None]
    if isinstance(term, GwDegree):
        r = 1 - math.exp(-term.decay)
        return names, (math.exp(term.decay) * (1 - r**d))[:, None]
    if isinstance(term, ConcurrentTies) and term.by is None:
        return names, np.maximum(d - 1, 0)[:, None]
    if isinstance(term, _DegreeRange):
        return names, _degree_ranges(term, data, population)
    raise ValueError(f"{term!r} is not supported in egocentric models")


def _degree_ranges(term: _DegreeRange, data: EgoData, population) -> np.ndarray:
    """degree(d, by=, homophily=), degrange(), concurrent(by=): each ego's
    degree (counting alters of its own value, with homophily) in each range,
    by its value of `by`."""
    from .terms import _level_codes, _select_levels, _sorted_levels, _values

    degree = data.degree.astype(float)
    if term.by is None:
        levels, codes = [None], np.zeros(data.n, dtype=np.int64)
    else:
        values = _values(population, term.by)
        chosen = _select_levels(_sorted_levels(values), term.levels)
        ego_values = data.egos[term.by]
        codes = np.array(_level_codes(ego_values, chosen), dtype=np.int64)
        levels = chosen
        if term.homophily:
            if term.by not in data.alters:
                raise ValueError(f"{term!r}: homophily needs the attribute {term.by!r} on the alters")
            same = np.array([data.alters[term.by][k] == ego_values[o] for k, o in enumerate(data.owner)])
            degree = data.per_ego(same.astype(float))
    columns = []
    if term.homophily:
        for lo, hi in term.ranges:
            columns.append((degree >= lo) & (degree < (np.inf if hi is None else hi)))
    else:
        for level in range(len(levels)):
            for lo, hi in term.ranges:
                columns.append((codes == level) & (degree >= lo) & (degree < (np.inf if hi is None else hi)))
    out = np.column_stack(columns).astype(float)
    if out.shape[1] != len(term.names(population)):
        raise ValueError(f"{term!r} is not supported in egocentric models")
    return out


def _histogram(term: Curved, data: EgoData, population, counts: np.ndarray, per_alter: bool):
    """A curved term's histogram: each ego's vertex (degree) or half its
    ties (shared partners) with each count, and those above the cutoff."""
    names = term.names(population)
    k, overflow = term._bins(population)
    columns = [counts == j for j in range(1, k + 1)]
    if overflow:
        columns.append(counts > k)
    h = np.column_stack(columns).astype(float)
    return names, data.per_ego(h) / 2 if per_alter else h


def _triadic(term: Term, data: EgoData, population) -> tuple[list[str], np.ndarray]:
    """Each ego's contribution to a statistic of the shared partners of ties:
    half the sum over its ties to its alters, whose shared partners are the
    other alters tied to them."""
    if not data.has_aaties:
        raise ValueError(f"{term!r} needs the ties among the alters (aaties)")
    names = term.names(population)
    sp = data.shared_partners().astype(float)
    if isinstance(term, Esp):
        values = np.column_stack([sp == k for k in term.ks]).astype(float)
    elif isinstance(term, Gwesp):
        r = 1 - math.exp(-term.decay)
        values = (math.exp(term.decay) * (1 - r**sp))[:, None]
    elif isinstance(term, TransitiveTies):
        values = (sp >= 1).astype(float)[:, None]
    elif isinstance(term, Triangle):
        return names, (np.bincount(data.aatie_owner, minlength=data.n) / 3.0)[:, None]
    else:
        raise ValueError(f"{term!r} is not supported in egocentric models")
    return names, data.per_ego(values) / 2


def ego_contributions(formula, data: EgoData, population) -> tuple[list[str], np.ndarray, list[int]]:
    """Each ego's contribution to each statistic (egos x statistics), whose
    mean, times the population size, estimates the population's statistic;
    and each term's order (0: a mean that doesn't scale)."""
    names, blocks, orders = [], [], []
    for term in as_formula(formula):
        if isinstance(term, Offset):
            continue
        if not _supported(term):
            raise ValueError(f"{term!r} is not supported in egocentric models: ergmx estimates "
                             "edges, nodecov, nodefactor, nodematch, nodemix, absdiff, absdiffcat, "
                             "degree, degrange, concurrent, concurrentties, degree1.5, gwdegree, "
                             "meandeg, esp, gwesp, transitiveties and triangle from egocentric data")
        inner = term.term if isinstance(term, Curved) else term
        if isinstance(term, Curved):
            if isinstance(inner, GwDegree):
                n, h = _histogram(term, data, population, data.degree, per_alter=False)
                order = 1
            else:
                if not data.has_aaties:
                    raise ValueError(f"{term!r} needs the ties among the alters (aaties)")
                n, h = _histogram(term, data, population, data.shared_partners(), per_alter=True)
                order = 3
        elif isinstance(term, MeanDeg):
            n, h, order = term.names(population), data.degree.astype(float)[:, None], 0
        elif isinstance(term, _DYADIC):
            (n, h), order = _dyadic(term, data, population), 1
        elif isinstance(term, _DEGREE):
            (n, h), order = _degree_counts(term, data, population), 1
        else:
            (n, h), order = _triadic(term, data, population), 3
        names += n
        blocks.append(h)
        orders += [order] * len(n)
    return names, np.column_stack(blocks) if blocks else np.zeros((data.n, 0)), orders


def _pseudo_population(data: EgoData, size: int, scaling: str, rng):
    """The pseudo-population: `size` replicates of the egos, by their weights
    (each ego round(size w / sum(w)) times, or a weighted sample), with their
    attributes and no ties, as ergm.ego's template_network()."""
    import igraph as ig

    w = np.ones(data.n) if data.weights is None else data.weights
    if scaling == "round":
        index = np.repeat(np.arange(data.n), np.round(size * w / w.sum()).astype(np.int64))
    elif scaling == "sample":
        index = rng.choice(data.n, size=size, replace=True, p=w / w.sum())
    else:
        raise ValueError(f"ppop_wt must be 'round' or 'sample', not {scaling!r}")
    g = ig.Graph(n=len(index))
    for a, values in data.egos.items():
        g.vs[a] = [values[i] for i in index]
    g.vs[".ego.ind"] = (index + 1).tolist()
    return g, index


def _estimate(h: np.ndarray, w: np.ndarray, scale: np.ndarray, how: str, rng, boot_r: int):
    """The scaled weighted mean of the egos' contributions, and its variance."""
    n = len(w)
    s = h * scale

    def wmean(weights, x):
        weights = weights / weights.sum()
        return weights @ x

    m = wmean(w, s)
    if how == "survey":
        # Linearization of the weighted mean, as the survey package's svymean
        # for a design with these weights (with replacement).
        u = (w / w.sum())[:, None] * (s - m)
        v = n / (n - 1) * u.T @ u
    elif how == "asymptotic":
        wx = np.column_stack([w, s * w[:, None]])
        cov = np.atleast_2d(np.cov(wx, rowvar=False))
        a = np.column_stack([-m, np.eye(s.shape[1])]) / w.mean()
        v = a @ cov @ a.T / n
    elif how == "naive":
        wn = w / w.sum()
        d = (s - m) * np.sqrt(wn)[:, None]
        v = d.T @ d / (1 - np.sum(wn**2)) * np.sum(wn**2)
    elif how == "bootstrap":
        draws = np.array([wmean(w[i], s[i]) for i in (rng.integers(0, n, n) for _ in range(boot_r))])
        m = m - (draws.mean(axis=0) - m)
        v = np.atleast_2d(np.cov(draws, rowvar=False))
    elif how == "jackknife":
        keep = np.ones(n, dtype=bool)
        draws = []
        for i in range(n):
            keep[i] = False
            draws.append(wmean(w[keep], s[keep]))
            keep[i] = True
        draws = np.array(draws)
        m = n * m - (n - 1) * draws.mean(axis=0)
        d = draws - draws.mean(axis=0)
        v = (n - 1) / n * d.T @ d
    else:
        raise ValueError("stats_est must be 'survey', 'asymptotic', 'naive', 'bootstrap' or 'jackknife', "
                         f"not {how!r}")
    return m, np.atleast_2d(v)


def ego_stats(formula, data: EgoData, *, scaleto: float | None = None, stats_est: str = "survey",
              seed=None, boot_r: int = 10000):
    """The population statistics that egocentric data estimate, as R's
    ``summary(egor ~ formula, scaleto=)``: each ego's contribution averaged
    (weighted) and scaled to a population of ``scaleto`` (by default the
    number of egos); statistics that don't scale (``meandeg``) are means.

    Returns
    -------
    (dict, numpy.ndarray)
        The estimates by name, and their covariance (``stats_est``: the
        survey linearization's, ``"asymptotic"``, ``"naive"``,
        ``"bootstrap"`` or ``"jackknife"``, as ergm.ego's).
    """
    population, _ = _pseudo_population(data, data.n, "round", None)
    names, h, orders = ego_contributions(formula, data, _as_net(population))
    size = data.n if scaleto is None else scaleto
    scale = np.where(np.array(orders) != 0, size, 1.0)
    w = np.ones(data.n) if data.weights is None else data.weights
    m, v = _estimate(h, w, scale, stats_est, np.random.default_rng(seed), boot_r)
    return dict(zip(names, m.tolist())), v


def _as_net(graph):
    from ._network import as_network

    return as_network(graph)


class EgoFit:
    """An egocentric ERGM fitted with :func:`ergm_ego`: the ERGM's fit on the
    pseudo-population (``fit``), the estimated population statistics
    (``stats``, ``stats_cov``) and the coefficients' covariance from the
    egos' sampling variance. Most of :class:`ErgmFit`'s methods work on it."""

    def __init__(self, fit, stats: dict, stats_cov: np.ndarray, cov: np.ndarray, data: EgoData,
                 ppopsize: int, popsize, adjusted: bool, stats_est: str, scale: np.ndarray):
        self.fit, self.stats, self.stats_cov = fit, stats, stats_cov
        self._cov, self.data, self.ppopsize, self.popsize = cov, data, ppopsize, popsize
        self.adjusted, self.stats_est = adjusted, stats_est
        self._scale = scale  # of each statistic, from per capita to the pseudo-population

    names = property(lambda self: self.fit.names)
    params = property(lambda self: self.fit.params)
    coef = property(lambda self: self.fit.coef)
    method = property(lambda self: self.fit.method)
    converged = property(lambda self: self.fit.converged)
    iterations = property(lambda self: self.fit.iterations)
    formula = property(lambda self: self.fit.formula)
    sample = property(lambda self: self.fit.sample)
    _model = property(lambda self: self.fit._model)
    # Egocentric fits have no log-likelihood (nor do ergm.ego's).
    loglik = aic = bic = property(lambda self: None)

    @property
    def cov(self) -> np.ndarray:
        """The coefficients' covariance: the egos' sampling variance of the
        statistics, through the model (a sandwich), as ergm.ego's."""
        return self._cov.copy()

    @property
    def stderr(self) -> dict[str, float]:
        return dict(zip(self.names, np.sqrt(np.diag(self._cov)).tolist()))

    def simulate(self, nsim: int = 1, **options):
        """Simulate networks of the pseudo-population. See :meth:`ErgmFit.simulate`."""
        return self.fit.simulate(nsim, **options)

    def mcmc_diagnostics(self):
        return self.fit.mcmc_diagnostics()

    def gof(self, nsim: int = 100, *, stats=None, seed=None, interval: int | None = None,
            burnin: int | None = None):
        """Goodness of fit, as ergm.ego's ``gof()``: the degree and edgewise
        shared partner distributions and the model statistics that the egos
        estimate, per capita, against those of ``nsim`` networks of the
        pseudo-population simulated from the model.

        ``stats`` are among ``"degree"``, ``"espartners"`` (with the ties
        among alters) and ``"model"`` (by default all that apply): other
        distributions, such as distances, can't be estimated from egocentric
        data. As in ergm.ego, the degrees go up to twice the largest ego's
        degree (at least 6), the last value counting all higher ones, and the
        shared partners up to twice the largest degree, less 2.
        """
        from ._estimation import Control
        from ._gof import GofResult, GofTable, _many

        data = self.data
        if stats is None:
            stats = ["degree", "espartners", "model"] if data.has_aaties else ["degree", "model"]
        for stat in stats:
            if stat not in ("degree", "espartners", "model"):
                raise ValueError(f"unknown egocentric goodness-of-fit statistic {stat!r}: use 'degree', "
                                 "'espartners' and 'model' (other distributions, such as distances, can't "
                                 "be estimated from egocentric data)")
        if "espartners" in stats and not data.has_aaties:
            raise ValueError("'espartners' needs the ties among the alters (aaties)")
        model, network = self.fit._model, self.fit._model.network
        interval = interval or self.fit._estimate.interval or Control.interval
        burnin = 16 * interval if burnin is None else burnin
        chains = max(1, min(self.fit.control.n_chains, nsim))
        rng = np.random.default_rng(seed)
        sample, _, networks = model.simulate(
            [network.edges] * chains, self.params, burnin, interval, -(-nsim // chains),
            int(rng.integers(2**63)), keep_networks=True,
            triadic_weight=model.triadic_weight(self.fit.control.triadic_weight),
        )
        simulated = [e for chain in networks for e in chain][:nsim]
        sample = sample.reshape(-1, model.n_stats)[:nsim]
        n = network.n
        w = np.ones(data.n) if data.weights is None else data.weights
        w = w / w.sum()
        largest = max(int(data.degree.max(initial=0)), 3)
        tables = {}
        for stat in stats:
            if stat == "degree":
                top = min(2 * largest, n - 1)
                labels = [str(k) for k in range(top)] + [f"{top}+" if top < n - 1 else str(top)]
                observed = np.bincount(np.minimum(data.degree, top), weights=w, minlength=top + 1)
                sims = []
                for edges in simulated:
                    degree = np.bincount(np.asarray(edges, dtype=np.int64).ravel(), minlength=n)
                    sims.append(np.bincount(np.minimum(degree, top), minlength=top + 1) / n)
                tables[stat] = GofTable(stat, labels, observed, np.array(sims))
            elif stat == "espartners":
                top = 2 * (largest - 1)
                # Each ego counts half of each of its ties with k shared partners.
                sp = data.shared_partners()
                indicator = (sp[:, None] == np.arange(top + 1)).astype(float)
                observed = w @ data.per_ego(indicator) / 2
                sims = np.zeros((len(simulated), top + 1))
                counts = _many(n, False, simulated, stat)[:, :top + 1]
                sims[:, :counts.shape[1]] = counts / n
                tables[stat] = GofTable(stat, [str(k) for k in range(top + 1)], observed, sims)
            else:
                targeted = [k for term, ps, _ in model.blocks if not term.is_offset
                            for k in range(ps.start, ps.stop)]
                observed = np.array(list(self.stats.values())) / self._scale
                tables[stat] = GofTable(stat, list(self.stats), observed, sample[:, targeted] / self._scale)
        return GofResult(tables, nsim)

    def summary(self):
        from ._fit import _pvalue, _stars

        se = np.sqrt(np.diag(self._cov))
        model = self.fit._model
        with np.errstate(invalid="ignore", divide="ignore"):
            z = self.params / se
        width = max(map(len, self.names))
        lines = [f"Egocentric {'Monte Carlo ' if self.method == 'MCMLE' else ''}Maximum Likelihood Results:",
                 "", f"{'':<{width}}  {'Estimate':>9}  {'Std. Error':>10}  {'z value':>7}  {'Pr(>|z|)':>8}"]
        for i, name in enumerate(self.names):
            if model.fixed[i]:
                note = "constant under the constraints" if model.constant[i] else "offset"
                lines.append(f"{name:<{width}}  {self.params[i]:9.4f}  {'':10}  {'':7}  {'':8}  ({note})")
                continue
            p = _pvalue(z[i]) if np.isfinite(z[i]) else float("nan")
            pval = "<1e-04" if p < 1e-4 else f"{p:.5f}"
            lines.append(f"{name:<{width}}  {self.params[i]:9.4f}  {se[i]:10.4f}  {z[i]:7.3f}  {pval:>8} {_stars(p)}")
        lines += ["---", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", "",
                  f"Fitted to {self.data.n} egos, on a pseudo-population of {self.ppopsize}"
                  + (f"; coefficients for a population of {self.popsize:g}"
                     + (" (per capita)" if self.popsize == 1 else "") if self.adjusted else "") + ".",
                  f"Standard errors from the egos' sampling variance ({self.stats_est})."]
        if self.method == "MCMLE":
            lines.append(f"{'Converged' if self.converged else 'Did NOT converge'} after {self.iterations} "
                         "iterations.")
        return _Text("\n".join(lines))

    def __repr__(self) -> str:
        rows = "\n".join(f"  {n:<{max(map(len, self.names))}}  {v: .4f}" for n, v in self.coef.items())
        return f"EgoFit:\n{rows}"


class _Text(str):
    def __repr__(self) -> str:
        return str(self)


def ergm_ego(formula, data: EgoData, *, popsize: float = 1, adjust_size: bool = True,
             ppopsize: Any = "auto", ppopsize_mul: float = 1.0, ppop_wt: str = "round",
             stats_est: str = "survey", boot_r: int = 10000, constraints=None, offset_coef=None,
             seed=None, **ergm_args) -> EgoFit:
    """Fit an ERGM to egocentric data, as R's ``ergm.ego()``.

    The egos' contributions estimate the population's statistics (scaled to
    the pseudo-population); the model is fitted to them on a
    pseudo-population network, of ``ppopsize`` replicates of the egos (by
    default as many as the egos, or ``popsize``, times ``ppopsize_mul``),
    with :func:`ergm`'s ``target_stats``; with ``adjust_size``, an offset
    (on ``edges``, and on ``transitiveties`` for triadic terms, ergm.ego's
    ``netsize.adj``) of -log(ppopsize / popsize) makes the coefficients those
    of a network of ``popsize`` vertices (1, the default: per capita, as in
    ergm.ego). The standard errors come from the sampling variance of the
    egos' contributions (``stats_est``, as in :func:`ego_stats`), through the
    model's information (a sandwich), as ergm.ego's.

    Supported terms: edges, nodecov, nodefactor, nodematch, nodemix,
    absdiff, absdiffcat, degree, degrange, concurrent, concurrentties,
    degree1.5, gwdegree, meandeg, esp, gwesp, transitiveties and triangle
    (the last four need the ties among alters), and offset() terms.

    Returns
    -------
    EgoFit
    """
    from ._simulate import ergm

    terms = as_formula(formula)
    rng = np.random.default_rng(seed)
    n = data.n
    if isinstance(ppopsize, str):
        if ppopsize not in ("auto", "samp", "pop"):
            raise ValueError(f"ppopsize must be 'auto', 'samp', 'pop' or a number, not {ppopsize!r}")
        base = {"auto": n if popsize in (0, 1) else popsize, "samp": n, "pop": popsize}[ppopsize]
        ppopsize = int(round(base * ppopsize_mul))
    population, index = _pseudo_population(data, int(ppopsize), ppop_wt, rng)
    net = _as_net(population)
    ppopsize = net.n
    names, h, orders = ego_contributions(terms, data, net)
    w = np.ones(n) if data.weights is None else data.weights
    scale = np.where(np.array(orders) != 0, ppopsize, 1.0)
    m, v = _estimate(h, w, scale, stats_est, rng, boot_r)
    adjusted = adjust_size and popsize > 0
    extra, extra_coef = [], []
    if adjusted:
        shift = -math.log(ppopsize / popsize)
        if 1 in orders:
            extra.append(Offset(Edges()))
            extra_coef.append(shift)
        if 3 in orders:
            extra.append(Offset(TransitiveTies()))
            extra_coef.append(-shift / 3)
    user_offsets = list(np.atleast_1d(offset_coef)) if offset_coef is not None else []
    full = Formula(extra + list(terms))
    fit = ergm(population, full, target_stats=dict(zip(names, m.tolist())),
               offset_coef=(extra_coef + user_offsets) or None, constraints=constraints, seed=seed,
               eval_loglik=False, **ergm_args)
    # The sandwich: the statistics' sampling variance through the information.
    model, est = fit._model, fit._estimate
    free = model.free
    info_inverse = est.cov if est.mc_cov is None else est.cov - est.mc_cov
    targeted = [k for term, ps, _ in model.blocks if not term.is_offset for k in range(ps.start, ps.stop)]
    jac = model.jacobian(fit.params)[np.ix_(targeted, free)]
    bread = info_inverse[np.ix_(free, free)]
    cov = np.full((model.n_params, model.n_params), np.nan)
    cov[np.ix_(free, free)] = bread @ jac.T @ v @ jac @ bread
    return EgoFit(fit, dict(zip(names, m.tolist())), v, cov, data, ppopsize, popsize, adjusted, stats_est, scale)
