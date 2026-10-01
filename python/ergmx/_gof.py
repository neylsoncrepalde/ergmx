"""Goodness of fit: network statistics of the observed network compared with
networks simulated from a model, as in R's ``gof(ergm)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from ._estimation import Control
from ._model import bind

_TITLES = {
    "degree": "degree",
    "idegree": "in-degree",
    "odegree": "out-degree",
    "espartners": "edgewise shared partners",
    "distance": "minimum geodesic distance",
    "model": "model statistics",
}
_UNITS = {"degree": "nodes", "idegree": "nodes", "odegree": "nodes", "espartners": "edges",
          "distance": "dyads"}


def _adjacency(n: int, directed: bool, edges: np.ndarray) -> sparse.csr_matrix:
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    a = sparse.csr_matrix((np.ones(len(i), dtype=np.int64), (i, j)), shape=(n, n))
    return a if directed else a + a.T


def _distribution(n: int, directed: bool, edges: np.ndarray, stat: str) -> np.ndarray:
    a = _adjacency(n, directed, edges)
    if stat in ("degree", "odegree"):
        return np.bincount(np.asarray(a.sum(axis=1)).ravel(), minlength=n)
    if stat == "idegree":
        return np.bincount(np.asarray(a.sum(axis=0)).ravel(), minlength=n)
    if stat == "espartners":
        # Shared partners of each tie i -> j: two-paths i -> k -> j (OTP if directed).
        two_paths = (a @ a).tocsr()
        partners = np.asarray(two_paths[edges[:, 0], edges[:, 1]]).ravel().astype(np.int64)
        return np.bincount(partners, minlength=n - 1)
    if stat == "distance":
        d = shortest_path(a, directed=directed, unweighted=True)
        d = d[~np.eye(n, dtype=bool)] if directed else d[np.triu_indices(n, 1)]
        finite = d[np.isfinite(d)].astype(np.int64)
        return np.append(np.bincount(finite, minlength=n)[1:], np.sum(~np.isfinite(d)))
    raise ValueError(f"unknown goodness-of-fit statistic {stat!r}")


def _labels(n: int, stat: str) -> list[str]:
    if stat in ("degree", "idegree", "odegree"):
        return [str(k) for k in range(n)]
    if stat == "espartners":
        return [str(k) for k in range(n - 1)]
    return [str(k) for k in range(1, n)] + ["Inf"]


@dataclass
class GofTable:
    """One statistic's observed values and simulated distribution."""

    name: str
    labels: list[str]
    observed: np.ndarray
    simulated: np.ndarray  # nsim x len(labels)

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
        if self.name == "model":
            return rows
        finite = len(rows) - 1 if self.name == "distance" else len(rows)  # distance ends with Inf
        busy = np.flatnonzero(((self.observed > 0) | (self.max > 0))[:finite])
        keep = rows <= (busy.max() if busy.size else 0)
        if self.name == "distance":
            keep[-1] = True  # always show unreachable pairs
        return rows[keep]

    def __str__(self) -> str:
        rows = self._shown()
        width = max(len(self.labels[r]) for r in rows)
        lines = [f"Goodness-of-fit for {_TITLES[self.name]}", "",
                 f"{'':<{width}}  {'obs':>8}  {'min':>8}  {'mean':>9}  {'max':>8}  {'MC p-value':>10}"]
        for r in rows:
            lines.append(
                f"{self.labels[r]:<{width}}  {self.observed[r]:8.6g}  {self.min[r]:8.6g}  "
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
        like R's ``plot(gof(fit))``. Needs matplotlib. Returns the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('plotting needs matplotlib: install "ergmx[plot]"') from None
        tables = list(self)
        if axes is None:
            fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.6),
                                     squeeze=False)
            axes = axes[0]
        else:
            axes = np.ravel(axes)
            fig = axes[0].figure
        for ax, table in zip(axes, tables):
            rows = table._shown()
            if table.name == "model":
                # Statistics have different scales: plot them standardized.
                sd = table.simulated.std(axis=0)
                sd[sd == 0] = 1.0
                sim, obs, ylabel = (table.simulated - table.observed) / sd, np.zeros(len(rows)), \
                    "simulated - observed (SD units)"
            else:
                total = table.simulated.sum(axis=1, keepdims=True)
                total[total == 0] = 1
                sim = table.simulated / total
                obs = table.observed / max(table.observed.sum(), 1)
                ylabel = f"proportion of {_UNITS[table.name]}"
            ax.boxplot(sim[:, rows], tick_labels=[table.labels[r] for r in rows], showfliers=False,
                       medianprops={"color": "grey"})
            ax.plot(np.arange(1, len(rows) + 1), obs[rows] if table.name != "model" else obs,
                    color="black", linewidth=2, marker="o", markersize=3)
            ax.set_title(_TITLES[table.name])
            ax.set_ylabel(ylabel)
            if table.name == "model" or len(rows) > 12:
                ax.tick_params(axis="x", labelrotation=90)
        fig.tight_layout()
        return fig


def gof(x, formula=None, coef=None, *, constraints=None, nsim: int = 100, stats=None, seed=None,
        interval: int | None = None, burnin: int | None = None, n_chains: int | None = None,
        triadic_weight: float | None = None) -> GofResult:
    """Goodness of fit of an ERGM, like R's ``gof()``.

    Simulates ``nsim`` networks from the model and compares their degree,
    edgewise shared partner and geodesic distance distributions, and the model
    statistics, with the observed network's.

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
        (directed), ``"espartners"``, ``"distance"`` and ``"model"``. The
        default is all of them that apply, as in ergm.
    interval, burnin : int, optional
        MCMC proposals between and before the simulated networks. Default to
        the interval the fit ended with (1024 otherwise), and 16 times that.

    Notes
    -----
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
        model = bind(x, formula, constraints)
        if isinstance(coef, dict):
            coef = [coef[name] for name in model.names]
        fitted_interval = None
    network = model.network
    if stats is None:
        degrees = ["idegree", "odegree"] if network.directed else ["degree"]
        stats = degrees + ["espartners", "distance", "model"]
    for stat in stats:
        if stat not in _TITLES:
            raise ValueError(f"unknown goodness-of-fit statistic {stat!r}; use {list(_TITLES)}")
        if network.directed and stat == "degree" or not network.directed and stat in ("idegree", "odegree"):
            raise ValueError(f"{stat!r} does not apply to {'' if network.directed else 'un'}directed networks")

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

    def distributions(edge_lists, stat):
        return np.array([_distribution(network.n, network.directed, e, stat) for e in edge_lists])

    tables = {}
    for stat in stats:
        if stat == "model":
            tables[stat] = GofTable(stat, list(model.names), imputed_stats.mean(axis=0), model_stats)
            continue
        observed = distributions(imputed_edges, stat).mean(axis=0)
        tables[stat] = GofTable(stat, _labels(network.n, stat), observed,
                                distributions(simulated_edges, stat))
    return GofResult(tables, nsim)
