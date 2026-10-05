"""Goodness of fit: network statistics of the observed network compared with
networks simulated from a model, as in R's ``gof(ergm)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import _core
from ._estimation import Control
from ._model import bind

_TITLES = {
    "degree": "degree",
    "idegree": "in-degree",
    "odegree": "out-degree",
    "b1degree": "first-mode degree",
    "b2degree": "second-mode degree",
    "espartners": "edgewise shared partners",
    "dspartners": "dyadwise shared partners",
    "distance": "minimum geodesic distance",
    "affiliations": "affiliations",
    "cdf": "dyads with value at most x",
    "model": "model statistics",
}
_UNITS = {"degree": "nodes", "idegree": "nodes", "odegree": "nodes", "b1degree": "nodes",
          "b2degree": "nodes", "espartners": "edges", "dspartners": "dyads", "distance": "dyads",
          "affiliations": "nodes", "cdf": "dyads"}


def _adjacency(n: int, directed: bool, edges: np.ndarray) -> sparse.csr_matrix:
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    a = sparse.csr_matrix((np.ones(len(i), dtype=np.int64), (i, j)), shape=(n, n))
    return a if directed else a + a.T


#: The distributions computed by the Rust core, without n x n matrices.
_RUST = ("espartners", "dspartners", "distance")


def _distribution(n: int, directed: bool, edges: np.ndarray, stat: str, mode=None) -> np.ndarray:
    if stat in _RUST:
        return _many(n, directed, [edges], stat)[0]
    a = _adjacency(n, directed, edges)
    if stat in ("b1degree", "b2degree"):
        degrees = np.asarray(a.sum(axis=1)).ravel()[mode == (1 if stat == "b1degree" else 2)]
        return np.bincount(degrees, minlength=n)
    if stat in ("degree", "odegree"):
        return np.bincount(np.asarray(a.sum(axis=1)).ravel(), minlength=n)
    if stat == "idegree":
        return np.bincount(np.asarray(a.sum(axis=0)).ravel(), minlength=n)
    raise ValueError(f"unknown goodness-of-fit statistic {stat!r}")


def _many(n: int, directed: bool, edge_lists, stat: str) -> np.ndarray:
    """A distribution of each of several networks (networks x values)."""
    lists = [np.ascontiguousarray(e, dtype=np.uint32).reshape(-1, 2) for e in edge_lists]
    if stat in _RUST:
        if n < 2:
            return np.zeros((len(lists), n if stat == "distance" else 0), dtype=np.int64)
        return _core.gof_distribution(n, directed, lists, stat).astype(np.int64)
    return np.array([_distribution(n, directed, e, stat) for e in lists])


def _pooled_many(network, edge_lists, stat: str) -> np.ndarray:
    """A distribution of each of several networks, summed over the networks
    of a combined network (counting no pairs of vertices in different
    networks): networks x values."""
    if not network.combined:
        if stat in _RUST:
            return _many(network.n, network.directed, edge_lists, stat)
        return np.array([_pooled(network, e, stat) for e in edge_lists])
    size = max(b.network.n for b in network.blocks)
    total = np.zeros((len(edge_lists), size), dtype=np.int64)
    parts = [network.split(e) for e in edge_lists]
    for k, block in enumerate(network.blocks):
        if stat in _RUST:
            d = _many(block.network.n, network.directed, [p[k] for p in parts], stat)
        else:
            d = np.array([_distribution(block.network.n, network.directed, p[k], stat, block.network.mode)
                          for p in parts])
        if not d.shape[1]:
            continue
        if stat == "distance":  # distances 1..n-1, then unreachable pairs
            total[:, :d.shape[1] - 1] += d[:, :-1]
            total[:, -1] += d[:, -1]
        else:
            total[:, :d.shape[1]] += d
    return total[:, :size - 1] if stat in ("espartners", "dspartners") else total


def _pooled(network, edges: np.ndarray, stat: str) -> np.ndarray:
    """A statistic's distribution, summed over the networks of a combined
    network (counting no pairs of vertices in different networks)."""
    if not network.combined:
        return _distribution(network.n, network.directed, edges, stat, network.mode)
    size = max(b.network.n for b in network.blocks)
    total = np.zeros(size, dtype=np.int64)
    for block, part in zip(network.blocks, network.split(edges)):
        d = _distribution(block.network.n, network.directed, part, stat, block.network.mode)
        if stat == "distance":  # distances 1..n-1, then unreachable pairs
            total[:len(d) - 1] += d[:-1]
            total[-1] += d[-1]
        else:
            total[:len(d)] += d
    return total[:size - 1] if stat in ("espartners", "dspartners") else total


def _labels(n: int, stat: str) -> list[str]:
    if stat in ("degree", "idegree", "odegree", "b1degree", "b2degree"):
        return [str(k) for k in range(n)]
    if stat in ("espartners", "dspartners"):
        return [str(k) for k in range(n - 1)]
    return [str(k) for k in range(1, n)] + ["Inf"]


@dataclass
class GofTable:
    """One statistic's observed values and simulated distribution."""

    name: str
    labels: list[str]
    observed: np.ndarray
    simulated: np.ndarray  # nsim x len(labels)
    #: The statistic (degree, espartners...), for by-level tables named "degree.<level>".
    stat: str | None = None
    #: The level of a vertex attribute (gof's by=) whose network this is.
    level: str | None = None

    @property
    def kind(self) -> str:
        return self.stat or self.name

    @property
    def title(self) -> str:
        if self.level is None:
            return _TITLES[self.kind]
        if self.kind == "affiliations":
            return f"affiliations of {self.level}"
        return f"{_TITLES[self.kind]}, {self.level}"

    @property
    def min(self) -> np.ndarray:
        return self.simulated.min(axis=0)

    @property
    def mean(self) -> np.ndarray:
        return self.simulated.mean(axis=0)

    @property
    def max(self) -> np.ndarray:
        return self.simulated.max(axis=0)

    @property
    def pvalue(self) -> np.ndarray:
        """Monte Carlo p-values, as in ergm: twice the smaller tail, capped at 1."""
        below = (self.simulated <= self.observed).mean(axis=0)
        above = (self.simulated >= self.observed).mean(axis=0)
        return np.minimum(1.0, 2 * np.minimum(below, above))

    def _shown(self) -> np.ndarray:
        """Rows worth printing: up to the last one where anything is non-zero."""
        rows = np.arange(len(self.labels))
        if self.kind in ("model", "cdf"):
            return rows
        finite = len(rows) - 1 if self.kind == "distance" else len(rows)  # distance ends with Inf
        busy = np.flatnonzero(((self.observed > 0) | (self.max > 0))[:finite])
        keep = rows <= (busy.max() if busy.size else 0)
        if self.kind == "distance":
            keep[-1] = True  # always show unreachable pairs
        return rows[keep]

    def __str__(self) -> str:
        rows = self._shown()
        labels = [f"{float(x):.6g}" for x in self.labels] if self.kind == "cdf" else self.labels
        width = max(len(labels[r]) for r in rows)
        lines = [f"Goodness-of-fit for {self.title}", "",
                 f"{'':<{width}}  {'obs':>8}  {'min':>8}  {'mean':>9}  {'max':>8}  {'MC p-value':>10}"]
        for r in rows:
            lines.append(
                f"{labels[r]:<{width}}  {self.observed[r]:8.6g}  {self.min[r]:8.6g}  "
                f"{self.mean[r]:9.2f}  {self.max[r]:8.6g}  {self.pvalue[r]:10.2f}"
            )
        return "\n".join(lines)

    __repr__ = __str__


class GofResult:
    """Goodness-of-fit tables, by statistic. Print it, or call :meth:`plot`."""

    def __init__(self, tables: dict[str, GofTable], nsim: int):
        self.tables = tables
        self.nsim = nsim

    def __getitem__(self, name: str) -> GofTable:
        return self.tables[name]

    def __iter__(self):
        return iter(self.tables.values())

    def __str__(self) -> str:
        return "\n\n".join(map(str, self))

    __repr__ = __str__

    def plot(self, axes=None):
        """Boxplots of the simulated statistics with the observed ones as a line,
        like R's ``plot(gof(fit))`` (for the distribution of continuous
        values, at many points, the simulated 95% band). Needs matplotlib.
        Returns the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('plotting needs matplotlib: install "ergmx[plot]"') from None
        tables = list(self)
        if axes is None:
            ncols = len(tables) if len(tables) <= 5 else 4
            nrows = -(-len(tables) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
            for ax in axes.ravel()[len(tables):]:
                ax.set_visible(False)
            axes = axes.ravel()
        else:
            axes = np.ravel(axes)
            fig = axes[0].figure
        for ax, table in zip(axes, tables):
            rows = table._shown()
            if table.kind == "model":
                # Statistics have different scales: plot them standardized.
                sd = table.simulated.std(axis=0)
                sd[sd == 0] = 1.0
                sim, obs, ylabel = (table.simulated - table.observed) / sd, np.zeros(len(rows)), \
                    "simulated - observed (SD units)"
            elif table.kind == "cdf":
                # As ergm's: relative to all the dyads, the observed count at the last point.
                total = max(table.observed[-1], 1)
                sim, obs, ylabel = table.simulated / total, table.observed / total, "proportion of dyads"
            else:
                total = table.simulated.sum(axis=1, keepdims=True)
                total[total == 0] = 1
                sim = table.simulated / total
                obs = table.observed / max(table.observed.sum(), 1)
                ylabel = f"proportion of {_UNITS[table.kind]}"
            ax.set_title(table.title)
            ax.set_ylabel(ylabel)
            if table.kind == "cdf" and len(rows) > 30:
                # Many points (continuous values): the simulated 95% band and median, against x.
                x = np.array([float(v) for v in table.labels])
                low, mid, high = np.quantile(sim, [0.025, 0.5, 0.975], axis=0)
                ax.fill_between(x, low, high, color="lightgrey", label="simulated, 95%")
                ax.plot(x, mid, color="grey", linewidth=1)
                ax.plot(x, obs, color="black", linewidth=2)
                ax.set_xlabel("x")
                continue
            labels = [table.labels[r] for r in rows]
            if table.kind == "cdf":
                labels = [f"{float(x):.3g}" for x in labels]
            ax.boxplot(sim[:, rows], tick_labels=labels, showfliers=False, medianprops={"color": "grey"})
            ax.plot(np.arange(1, len(rows) + 1), obs[rows] if table.kind != "model" else obs,
                    color="black", linewidth=2, marker="o", markersize=3)
            if table.kind == "model" or len(rows) > 12:
                ax.tick_params(axis="x", labelrotation=90)
        fig.tight_layout()
        return fig


def _within(edges: np.ndarray, members: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """The ties among some vertices, renumbered among them."""
    position = np.full(n, -1)
    position[members] = np.arange(len(members))
    if not len(edges):
        return len(members), np.zeros((0, 2), dtype=np.uint32)
    a, b = position[edges[:, 0].astype(int)], position[edges[:, 1].astype(int)]
    keep = (a >= 0) & (b >= 0)
    return len(members), np.column_stack([a[keep], b[keep]]).astype(np.uint32)


def _affiliations(network, edges: np.ndarray, members: np.ndarray, others: np.ndarray, first: bool) -> np.ndarray:
    """The distribution of the members' ties to the other level (in directed
    networks, arcs from the first level to the second)."""
    inside = np.zeros(network.n, dtype=bool)
    inside[members] = True
    outside = np.zeros(network.n, dtype=bool)
    outside[others] = True
    count = np.zeros(network.n, dtype=np.int64)
    if len(edges):
        i, j = edges[:, 0].astype(int), edges[:, 1].astype(int)
        if network.directed:
            tail, head = (inside, outside) if first else (outside, inside)
            across = tail[i] & head[j]
            np.add.at(count, i[across] if first else j[across], 1)
        else:
            for a, b in ((i, j), (j, i)):
                across = inside[a] & outside[b]
                np.add.at(count, a[across], 1)
    return np.bincount(count[members], minlength=len(others) + 1)


def gof(x, formula=None, coef=None, *, constraints=None, nsim: int = 100, stats=None, seed=None,
        interval: int | None = None, burnin: int | None = None, n_chains: int | None = None,
        triadic_weight: float | None = None, by: str | None = None, response: str | None = None,
        reference="Bernoulli") -> GofResult:
    """Goodness of fit of an ERGM, like R's ``gof()``.

    Simulates ``nsim`` networks from the model and compares their degree,
    edgewise shared partner and geodesic distance distributions, and the model
    statistics, with the observed network's. For valued networks, as ergm's,
    the distribution of the dyads' values and the model statistics.

    Parameters
    ----------
    x : ErgmFit, or igraph.Graph / networkx.Graph
        A fitted model, or a network (then give ``formula`` and ``coef``).
    formula, coef
        The model and its coefficients, when ``x`` is a network.
    constraints : str, optional
        Sample space constraints, when ``x`` is a network; a fit's own are used
        otherwise.
    nsim : int
        Number of simulated networks.
    stats : list of str, optional
        Among ``"degree"`` (undirected), ``"idegree"``, ``"odegree"``
        (directed), ``"b1degree"``, ``"b2degree"`` (bipartite),
        ``"espartners"``, ``"dspartners"``, ``"distance"`` and ``"model"``,
        and for valued networks ``"cdf"``.
        The default is ergm's: degrees, edgewise shared partners, distances
        and the model statistics; for bipartite networks, the degrees of each
        mode, dyadwise shared partners, distances and the model statistics;
        for valued networks, the model statistics and ``"cdf"``, the number
        of dyads whose value is at most each of a range of values (ergm's
        ``cdf``: those of the nonzero values, widened by a tenth, in steps of
        their resolution, at most 100). The other statistics of a valued
        network are those of its nonzero dyads as ties.
    interval, burnin : int, optional
        MCMC proposals between and before the simulated networks. Default to
        the interval the fit ended with (1024 otherwise), and 16 times that.
    response, reference : str, optional
        For a valued network, as in :func:`ergmx.ergm`: the edge attribute
        with the values, and their reference measure.
    by : str, optional
        A vertex attribute, such as the level of a multilevel network: the
        distributions are then of the network within each of its values
        (tables ``"degree.<value>"``, ``"espartners.<value>"``...) and, with
        two values, of each value's number of ties to the other
        (``"affiliations.<value>"``; in directed networks, arcs from the first
        value to the second), with the model statistics.

    Notes
    -----
    With several networks (:func:`ergmx.Networks`, :func:`ergmx.NetSeries`),
    the distributions are summed over the networks, and only pairs of
    vertices in the same network count (ergm's gof also counts the pairs in
    different networks, as unreachable).

    With missing dyads, the "observed" distributions are averages over
    ``nsim`` networks drawn from the model conditional on the observed dyads,
    rather than those of the network with missing dyads as non-ties, which
    would make the model look like it overestimates every count of ties.

    Returns
    -------
    GofResult
    """
    from ._fit import ErgmFit

    if isinstance(x, ErgmFit):
        model, coef = x._model, x.params
        fitted_interval = x._estimate.interval
        n_chains = n_chains or x.control.n_chains
        triadic_weight = x.control.triadic_weight if triadic_weight is None else triadic_weight
    else:
        if formula is None or coef is None:
            raise TypeError("gof(network, formula, coef): give the formula and the coefficients")
        if response is not None:
            from ._valued import bind_valued

            if str(reference).lstrip("~").strip() == "Bernoulli":
                raise ValueError("valued networks need a reference measure, such as reference='Poisson'")
            model = bind_valued(x, formula, response, reference, constraints=constraints)
        else:
            model = bind(x, formula, constraints)
        if isinstance(coef, dict):
            coef = [coef[name] for name in model.names]
        fitted_interval = None
    network = model.network
    valued = getattr(model, "valued", False)
    if by is not None:
        if valued:
            raise ValueError("gof(by=) is not supported for valued networks")
        return _gof_by(model, coef, by, stats, nsim, seed, interval, burnin, n_chains, triadic_weight,
                       fitted_interval)
    stats = (["model", "cdf"] if valued else default_stats(network)) if stats is None else stats
    check_stats(network, stats, valued)

    interval = interval or fitted_interval or Control.interval
    burnin = 16 * interval if burnin is None else burnin
    chains = max(1, min(n_chains or Control().n_chains, nsim))
    per_chain = -(-nsim // chains)
    rng = np.random.default_rng(seed)

    def draw(conditional):
        sample, _, networks = model.simulate(
            [network.edges] * chains, coef, burnin, interval, per_chain,
            int(rng.integers(2**63)), conditional=conditional, keep_networks=True,
            triadic_weight=model.triadic_weight(triadic_weight),
        )
        return [e for chain in networks for e in chain][:nsim], sample.reshape(-1, model.n_stats)[:nsim]

    simulated_edges, model_stats = draw(conditional=False)
    if model.has_missing:
        imputed_edges, imputed_stats = draw(conditional=True)
    else:
        imputed_edges, imputed_stats = [network.edges], model.observed()[None, :]
    return GofResult(_tables(model, stats, simulated_edges, model_stats, imputed_edges, imputed_stats), nsim)


def check_stats(network, stats, valued: bool = False) -> None:
    for stat in stats:
        if stat not in _TITLES or stat == "affiliations":
            raise ValueError(f"unknown goodness-of-fit statistic {stat!r}; use "
                             f"{[s for s in _TITLES if s != 'affiliations']}")
        if stat == "cdf" and not valued:
            raise ValueError("'cdf' is the distribution of a valued network's values")
        if network.directed and stat == "degree" or not network.directed and stat in ("idegree", "odegree"):
            raise ValueError(f"{stat!r} does not apply to {'' if network.directed else 'un'}directed networks")
        if stat in ("b1degree", "b2degree") and not network.bipartite:
            raise ValueError(f"{stat!r} needs a bipartite network")


def default_stats(network) -> list[str]:
    """ergm's goodness-of-fit statistics for a network."""
    if network.bipartite:
        return ["b1degree", "b2degree", "dspartners", "distance", "model"]
    return (["idegree", "odegree"] if network.directed else ["degree"]) + ["espartners", "distance", "model"]


def _tables(model, stats, simulated_edges, model_stats, imputed_edges, imputed_stats) -> dict:
    """The goodness-of-fit tables of simulated networks against the observed
    (or imputed) ones."""
    network = model.network
    valued = getattr(model, "valued", False)

    def distributions(edge_lists, stat):
        if valued:  # the nonzero dyads as ties
            edge_lists = [np.asarray(e).reshape(-1, 3)[:, :2] for e in edge_lists]
        return _pooled_many(network, edge_lists, stat)

    tables = {}
    for stat in stats:
        if stat == "model":
            tables[stat] = GofTable(stat, list(model.stat_names), imputed_stats.mean(axis=0),
                                    model_stats)
            continue
        if stat == "cdf":
            from ._valued import dyad_count

            points = cdf_points(model.observed_values())
            dyads = dyad_count(model.binary)
            tables[stat] = GofTable(stat, [f"{x:.15g}" for x in points],
                                    np.mean([_cdf(points, e, dyads) for e in imputed_edges], axis=0),
                                    np.array([_cdf(points, e, dyads) for e in simulated_edges]))
            continue
        observed = distributions(imputed_edges, stat).mean(axis=0)
        size = max(b.network.n for b in network.blocks) if network.combined else network.n
        tables[stat] = GofTable(stat, _labels(size, stat), observed,
                                distributions(simulated_edges, stat))
    return tables


def cdf_points(values, margin: float = 0.1, nmax: int = 100) -> np.ndarray:
    """The points of ergm's ``cdf`` goodness-of-fit term: from the smallest
    to the largest nonzero value, widened by ``margin`` of their range, in
    steps of the values' resolution (their smallest difference), or a
    multiple of it, so that there are at most ``nmax``."""
    values = np.unique(np.asarray(values, dtype=float))
    values = values[values != 0]
    if not len(values):
        raise ValueError("the network has no nonzero values, so the points of their distribution can't be set")
    gaps = np.diff(values)
    gaps = gaps[gaps > np.sqrt(np.finfo(float).eps)]
    if not len(gaps):  # a single value
        return np.unique([0.0, values[0]])
    res = gaps.min()

    def ceiling(x):
        return np.ceil(x / res) * res

    widen = ceiling((values[-1] - values[0]) * margin)
    low, high = values[0] - widen, values[-1] + widen
    by = max(res, ceiling((high - low) / (nmax - 1)))
    return np.minimum(low + np.arange(int((high - low) / by + 1e-10) + 1) * by, high)  # as R's seq()


def _cdf(points: np.ndarray, triples, dyads: float) -> np.ndarray:
    """The number of dyads whose value is at most each point."""
    values = np.sort(np.asarray(triples, dtype=float).reshape(-1, 3)[:, 2])
    values = values[values != 0]
    return np.searchsorted(values, points, side="right") + (dyads - len(values)) * (points >= 0)


def _simulations(model, coef, nsim, seed, interval, burnin, n_chains, triadic_weight):
    """Networks simulated from the model, and the observed (or imputed) ones,
    with their model statistics."""
    network = model.network
    chains = max(1, min(n_chains or Control().n_chains, nsim))
    per_chain = -(-nsim // chains)
    rng = np.random.default_rng(seed)

    def draw(conditional):
        sample, _, networks = model.simulate(
            [network.edges] * chains, coef, burnin, interval, per_chain,
            int(rng.integers(2**63)), conditional=conditional, keep_networks=True,
            triadic_weight=model.triadic_weight(triadic_weight),
        )
        return [e for chain in networks for e in chain][:nsim], sample.reshape(-1, model.n_stats)[:nsim]

    simulated, stats = draw(conditional=False)
    if model.has_missing:
        observed, observed_stats = draw(conditional=True)
    else:
        observed, observed_stats = [network.edges], model.observed()[None, :]
    return simulated, stats, observed, observed_stats


def _gof_by(model, coef, by, stats, nsim, seed, interval, burnin, n_chains, triadic_weight, fitted_interval):
    network = model.network
    if network.combined:
        raise ValueError("gof(by=) is not supported for several networks combined")
    from .terms import _level_name

    values = np.asarray(network.attribute(by), dtype=object)
    levels = sorted({v for v in values if v is not None}, key=lambda v: (str(type(v)), v))
    within = ["idegree", "odegree"] if network.directed else ["degree"]
    stats = stats or [*within, "espartners", "distance", "affiliations", "model"]
    allowed = {*within, "espartners", "dspartners", "distance", "affiliations", "model"}
    unknown = [s for s in stats if s not in allowed]
    if unknown:
        raise ValueError(f"gof(by=): unknown statistics {unknown}; use {sorted(allowed)}")
    interval = interval or fitted_interval or Control.interval
    burnin = 16 * interval if burnin is None else burnin
    simulated, model_stats, observed, observed_stats = _simulations(
        model, coef, nsim, seed, interval, burnin, n_chains, triadic_weight)

    tables = {}
    for level in levels:
        members = np.flatnonzero(values == level)
        name = _level_name(level)
        if len(members) < 2:
            continue
        for stat in stats:
            if stat in ("model", "affiliations"):
                continue

            def dist(edges, stat=stat, members=members):
                size, part = _within(edges, members, network.n)
                return _distribution(size, network.directed, part, stat)

            obs = np.mean([dist(e) for e in observed], axis=0)
            tables[f"{stat}.{name}"] = GofTable(f"{stat}.{name}", _labels(len(members), stat), obs,
                                                np.array([dist(e) for e in simulated]), stat=stat, level=name)
    if "affiliations" in stats and len(levels) == 2:
        for k, level in enumerate(levels):
            members, others = np.flatnonzero(values == level), np.flatnonzero(values == levels[1 - k])

            def dist(edges, members=members, others=others, first=k == 0):
                return _affiliations(network, edges, members, others, first)

            name = _level_name(level)
            obs = np.mean([dist(e) for e in observed], axis=0)
            tables[f"affiliations.{name}"] = GofTable(
                f"affiliations.{name}", [str(d) for d in range(len(others) + 1)], obs,
                np.array([dist(e) for e in simulated]), stat="affiliations", level=name)
    if "model" in stats:
        tables["model"] = GofTable("model", list(model.stat_names), observed_stats.mean(axis=0), model_stats)
    return GofResult(tables, nsim)
