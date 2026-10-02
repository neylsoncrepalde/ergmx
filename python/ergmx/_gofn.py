"""Goodness of fit network by network, for several networks fitted jointly,
as ergm.multi's ``gofN()``: each network's observed statistics against the
mean and variance of those simulated from the fit, as Pearson residuals."""

from __future__ import annotations

import warnings

import numpy as np

from ._estimation import Control
from ._model import bind
from .terms import BlockOperator, ErgmDifferenceWarning, Formula, Term, _VIEWS, as_formula

_COLUMNS = ("observed", "fitted", "var", "var_obs", "pearson")


class _ByNetwork(Term):
    """An operator's statistics in each network separately: each network's
    row of the linear model, in columns of its own (ergm.multi's
    ByNetDStats). A term outside the operators is treated as N(~term)."""

    def __init__(self, term: BlockOperator, plain: bool):
        self.term, self.plain = term, plain

    dyad_independent = property(lambda self: self.term.dyad_independent)
    triadic = property(lambda self: self.term.triadic)
    curved = False

    def check(self, network):
        self.term.check(network)

    def _inner(self, network) -> list[list[str]]:
        """Each network's names of its statistics, without the operator's columns."""
        t = self.term
        return [[n if self.plain else f"{t.op}~{n}" for n in t._inner_names(b.network)] for b in network.blocks]

    def network_names(self, network) -> list[str]:
        """The statistics of each network: the operator's own, or for curved
        terms, whose statistics differ between networks, all of them."""
        t = self.term
        if t._inner_curved:
            seen: dict[str, None] = {}
            for names in self._inner(network):
                seen.update(dict.fromkeys(names))
            return list(seen)
        if self.plain:
            return t._inner_names(network.blocks[0].network)
        return t.names(network)

    def per_network(self, network, values: np.ndarray) -> np.ndarray:
        """The term's statistics (as summary() returns them) by network:
        networks x network_names()."""
        t, k = self.term, len(network.blocks)
        names = self.network_names(network)
        if t._inner_curved:
            out, start = np.zeros((k, len(names))), 0
            position = {n: i for i, n in enumerate(names)}
            for row, inner in enumerate(self._inner(network)):
                out[row, [position[n] for n in inner]] = values[start:start + len(inner)]
                start += len(inner)
            return out
        n_inner = len(t._inner_names(network.blocks[0].network))
        return values.reshape(n_inner, k, -1).transpose(1, 0, 2).reshape(k, -1)

    def names(self, network):
        return [f"{k + 1}:{n}" for k in range(len(network.blocks)) for n in self.network_names(network)]

    def full_spec(self, network):
        t = self.term
        k = len(network.blocks)
        if t._inner_curved:
            children = [("block", [], [], [s.full_spec(b.network) for s in t.formula]) for b in network.blocks]
            return ("blocks", [], [_VIEWS[t.op], int(t.op == "Diss"), 0, 0], children)
        _, x, _, offsets = t._frame([b.attributes for b in network.blocks])
        if offsets is not None:
            x = np.column_stack([x, offsets])
        q = x.shape[1]
        children = []
        for row, b in enumerate(network.blocks):
            design = np.zeros(k * q)
            design[row * q:(row + 1) * q] = x[row]
            children.append(("block", design.tolist(), [], [s.full_spec(b.network) for s in t.formula]))
        return ("blocks", [], [_VIEWS[t.op], int(t.op == "Diss"), 1, k * q], children)

    def spec(self, network):
        raise TypeError("no flat spec; use full_spec")

    def __repr__(self) -> str:
        return f"ByNetwork({self.term!r})"


def _by_network(formula, network) -> list[_ByNetwork]:
    out = []
    for term in as_formula(formula):
        term = term.term if term.is_offset else term
        if isinstance(term, BlockOperator):
            out.append(_ByNetwork(term, plain=False))
        else:
            out.append(_ByNetwork(BlockOperator("N", Formula([term])), plain=True))
    return out


def gofN(fit, GOF=None, *, subset=True, nsim: int = 100, seed=None,  # noqa: N802, N803 (R's names)
         interval: int | None = None, burnin: int | None = None, n_chains: int | None = None,
         triadic_weight: float | None = None) -> GofNResult:
    """Goodness of fit network by network, for a model of several networks
    (:func:`ergmx.Networks`, :func:`ergmx.NetSeries`), as ergm.multi's
    ``gofN()``.

    Simulates ``nsim`` combined networks from the fit, and compares each
    network's observed statistics with the mean (``fitted``) and variance
    (``var``) of its simulated ones, as Pearson residuals:
    (observed - fitted) / sqrt(var - var_obs). With missing dyads, the
    observed statistics are the mean of networks simulated conditional on
    the observed dyads, and ``var_obs`` their variance (0 otherwise).
    Statistics that don't vary in a network's simulations are NaN there.

    Parameters
    ----------
    fit : ErgmFit
        A model of several networks.
    GOF : str or terms, optional
        The statistics to check, as a formula, evaluated on each network: by
        default the model's, each network's share of the model's statistics
        (offset() terms included). A term outside N() is N(~term).
    subset : bool array, 1-based indices or str, optional
        The networks to report: as N()'s ``subset``, logical values, indices
        or an R expression of the networks' attributes (``"~n >= 4"``).
    nsim, seed, burnin, n_chains, triadic_weight
        As in :func:`ergmx.gof`.
    interval : int, optional
        MCMC proposals between simulated networks: by default three times
        the number of dyads (at least the fit's interval), so that each
        network's simulated statistics are nearly independent. (ergm.multi
        uses the fit's, 1024 by default, which leaves those of many small
        networks autocorrelated, and their fitted values and residuals
        noisier than ``nsim`` suggests.)

    Returns
    -------
    GofNResult
        Index it by statistic for a table of the networks; print its
        :meth:`~GofNResult.summary`, or :meth:`~GofNResult.plot` the residuals.
    """
    from ._fit import ErgmFit
    from ._gof import _simulations
    from ._lm import LmError, network_subset

    if not isinstance(fit, ErgmFit):
        raise TypeError("gofN() needs a fitted model (ergmx.ergm() or ergmx.tergm())")
    model, coef = fit._model, fit.params
    network = model.network
    if not network.combined:
        raise ValueError("gofN() is for models of several networks: ergmx.Networks() or ergmx.NetSeries()")
    attributes = [b.attributes for b in network.blocks]
    try:
        kept = np.ones(len(attributes), dtype=bool) if subset is True else network_subset(subset, attributes)
    except LmError as e:
        raise ValueError(f"gofN(): {e}") from None
    terms = _by_network(model.formula if GOF is None else GOF, network)
    gof_model = bind(network, Formula(terms))

    from ._multi import _dyads

    base = fit._estimate.interval or Control.interval
    interval = interval or int(max(base, min(3 * sum(_dyads(b.network) for b in network.blocks), 2**22)))
    burnin = 16 * interval if burnin is None else burnin
    simulated, _, observed, _ = _simulations(model, coef, nsim, seed, interval, burnin,
                                             n_chains or fit.control.n_chains,
                                             fit.control.triadic_weight if triadic_weight is None else triadic_weight)

    def per_network(edge_lists) -> np.ndarray:  # draws x networks x statistics
        draws = np.array([gof_model.core.summary(e) for e in edge_lists])
        parts = []
        for term, cols, _ in gof_model.blocks:
            parts.append(np.stack([term.per_network(network, d[cols]) for d in draws]))
        return np.concatenate(parts, axis=2)

    names = [n for term in terms for n in term.network_names(network)]
    sims, obs = per_network(simulated), per_network(observed)
    empty = per_network([np.zeros((0, 2), dtype=np.uint32)])[0]
    shifted = sorted({names[j] for j in np.flatnonzero(np.any(empty != 0, axis=0))})
    if shifted and not network.series:  # ergm.multi's gofN() is for Networks()
        warnings.warn(f"ergm.multi's gofN() reports {shifted} minus their value in the empty network "
                      "(it leaves out the empty network's statistics, which for degree0 and isolates "
                      "is the network size); ergmx reports the statistics themselves",
                      ErgmDifferenceWarning, stacklevel=2)
    fitted, var = sims.mean(axis=0), sims.var(axis=0, ddof=1)
    observed_mean = obs.mean(axis=0)
    var_obs = obs.var(axis=0, ddof=1) if len(obs) > 1 else np.zeros_like(var)
    varies = var > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.where(varies, (observed_mean - fitted) / np.sqrt(var - var_obs), np.nan)
    bad = varies & (var - var_obs <= 0)
    if bad.any():
        warnings.warn(f"{int(bad.sum())} statistics vary less in the simulations than given the "
                      "observed dyads: their residuals are NaN; simulate more networks (nsim=)",
                      stacklevel=2)
    nan = np.where(varies, 1.0, np.nan)
    table = {"observed": observed_mean * nan, "fitted": fitted * nan, "var": var * nan,
             "var_obs": var_obs * nan, "pearson": pearson}
    return GofNResult(names, {k: v[kept] for k, v in table.items()}, np.flatnonzero(kept) + 1,
                [a for a, k in zip(attributes, kept) if k], nsim)


class GofNTable:
    """One statistic of :func:`gofN`, by network: ``observed``, ``fitted``,
    ``var``, ``var_obs`` and ``pearson`` arrays (NaN where the statistic
    doesn't vary)."""

    def __init__(self, name: str, networks: np.ndarray, labels: list[str], values: dict[str, np.ndarray]):
        self.name, self.networks, self.labels = name, networks, labels
        for key in _COLUMNS:
            setattr(self, key, values[key])

    def to_frame(self):
        """As a pandas DataFrame, one row per network. Needs pandas."""
        import pandas as pd

        return pd.DataFrame({k: getattr(self, k) for k in _COLUMNS},
                            index=pd.Index(self.labels, name="network"))

    def __str__(self) -> str:
        width = max(len(s) for s in ["network", *self.labels])
        lines = [f"gofN: {self.name}", "",
                 f"{'network':<{width}}  {'observed':>10}  {'fitted':>10}  {'var':>10}  {'var_obs':>8}  {'pearson':>8}"]
        for k, label in enumerate(self.labels):
            lines.append(f"{label:<{width}}  {self.observed[k]:10.4g}  {self.fitted[k]:10.4g}  "
                         f"{self.var[k]:10.4g}  {self.var_obs[k]:8.4g}  {self.pearson[k]:8.3f}")
        return "\n".join(lines)

    __repr__ = __str__


def _describe(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [np.nan] * 6 + [len(values)]
    q = np.quantile(finite, [0, 0.25, 0.5, 0.75, 1])
    return [q[0], q[1], q[2], float(finite.mean()), q[3], q[4], int(len(values) - finite.size)]


class GofNSummary:
    """Summaries of the observed and fitted values and the Pearson residuals
    of each statistic over the networks (by group, with ``by``), as
    ergm.multi's ``summary(gofN)``: minimum, quartiles, mean, maximum and the
    number of networks where the statistic doesn't vary; and the variance and
    standard deviation of the residuals, which are near 1 for a model that
    fits."""

    _HEAD = ("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.", "NA's")

    def __init__(self, gof: GofNResult, groups: dict[str, np.ndarray]):
        self.groups = {}
        for group, rows in groups.items():
            self.groups[group] = {
                part: {name: _describe(getattr(gof[name], key)[rows]) for name in gof.names}
                for part, key in (("Observed/Imputed values", "observed"), ("Fitted values", "fitted"),
                                  ("Pearson residuals", "pearson"))}
            pearson = {name: gof[name].pearson[rows] for name in gof.names}
            self.groups[group]["residual variance"] = {
                n: float(np.nanvar(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan for n, v in pearson.items()}

    def to_frame(self):
        """As a pandas DataFrame: group, part and statistic by row. Needs pandas."""
        import pandas as pd

        rows = []
        for group, parts in self.groups.items():
            for part, stats in parts.items():
                if part == "residual variance":
                    continue
                for name, values in stats.items():
                    rows.append({"group": group, "part": part, "statistic": name, **dict(zip(self._HEAD, values))})
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        out = []
        for group, parts in self.groups.items():
            if group:
                out.append(f"== {group}")
            names = list(parts["Pearson residuals"])
            width = max(len(n) for n in names)
            for part in ("Observed/Imputed values", "Fitted values", "Pearson residuals"):
                out += [part, f"{'':<{width}}  " + "  ".join(f"{h:>8}" for h in self._HEAD)]
                for name in names:
                    v = parts[part][name]
                    out.append(f"{name:<{width}}  " + "  ".join(f"{x:8.4g}" for x in v[:6]) + f"  {v[6]:8d}")
                out.append("")
            out.append("Variance and std. dev. of the Pearson residuals")
            for name in names:
                v = parts["residual variance"][name]
                out.append(f"{name:<{width}}  {v:8.4g}  {np.sqrt(v):8.4g}")
            out.append("")
        return "\n".join(out).rstrip()

    __repr__ = __str__


def _smooth(x: np.ndarray, y: np.ndarray, w: np.ndarray, span: float = 0.75, points: int = 101):
    """A weighted local linear fit with tricube weights (as loess's)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 4 or np.ptp(x) == 0:
        return None
    grid = np.linspace(x.min(), x.max(), points)
    k = max(3, int(np.ceil(span * len(x))))
    fit = []
    for g in grid:
        d = np.abs(x - g)
        h = np.sort(d)[k - 1] or 1e-12
        weights = w * np.clip(1 - (d / h) ** 3, 0, None) ** 3
        if np.count_nonzero(weights) < 2:
            fit.append(np.nan)
            continue
        a = np.column_stack([np.ones_like(x), x - g]) * np.sqrt(weights)[:, None]
        fit.append(np.linalg.lstsq(a, y * np.sqrt(weights), rcond=None)[0][0])
    return grid, np.array(fit)


class GofNResult:
    """Goodness of fit by network (:func:`gofN`): a :class:`GofNTable` per
    statistic, by name. :meth:`summary` summarizes them over the networks,
    :meth:`plot` plots the residuals."""

    def __init__(self, names: list[str], table: dict[str, np.ndarray], networks: np.ndarray,
                 attributes: list[dict], nsim: int):
        self.names, self.networks, self.attributes, self.nsim = names, networks, attributes, nsim
        labels = [str(a.get(".NetworkName", k)) for a, k in zip(attributes, networks)]
        self.tables = {name: GofNTable(name, networks, labels, {k: v[:, j] for k, v in table.items()})
                       for j, name in enumerate(names)}

    def __getitem__(self, name: str) -> GofNTable:
        return self.tables[name]

    def __iter__(self):
        return iter(self.tables.values())

    def __len__(self) -> int:
        return len(self.tables)

    def to_frame(self):
        """All statistics, one row per network and statistic, as a pandas
        DataFrame. Needs pandas."""
        import pandas as pd

        return pd.concat([t.to_frame().assign(statistic=t.name).reset_index() for t in self],
                         ignore_index=True)[["network", "statistic", *_COLUMNS]]

    def _attribute(self, expression) -> np.ndarray:
        from ._lm import LmError, evaluate

        try:
            return evaluate(expression, self.attributes)
        except LmError as e:
            raise ValueError(f"gofN: {e}") from None

    def summary(self, by=None) -> GofNSummary:
        """Summaries over the networks, as ergm.multi's ``summary(gofN)``;
        with ``by``, an R expression of the networks' attributes (``"~n"``,
        ``"~I(n >= 4)"``), for each of its values."""
        if by is None:
            return GofNSummary(self, {"": np.arange(len(self.networks))})
        values = self._attribute(by)
        from ._lm import _level_label, _levels

        return GofNSummary(self, {f"{str(by).lstrip('~').strip()} = {_level_label(v)}": np.flatnonzero(values == v)
                                  for v in _levels(np.asarray(values, dtype=object))})

    def __str__(self) -> str:
        return (f"Goodness of fit by network: {len(self.names)} statistics in {len(self.networks)} "
                f"networks, {self.nsim} simulations\n\n{self.summary()}")

    __repr__ = __str__

    def plot(self, stats=None, against=None, which=(1, 2), id_n: int = 3, axes=None):
        """Each statistic's Pearson residuals against the fitted values (or
        ``against``, an R expression of the networks' attributes, where
        ``.fitted`` is the fitted values), as ergm.multi's ``plot(gofN)``:
        ``which`` chooses among 1, residuals; 2, the square root of their
        absolute values (scale-location); and 3, a normal Q-Q plot. A weighted
        local regression shows the trend, and the ``id_n`` most extreme
        residuals are labelled with their network. Needs matplotlib. Returns
        the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('plotting needs matplotlib: install "ergmx[plot]"') from None
        from scipy.stats import norm

        names = list(self.names if stats is None else [stats] if isinstance(stats, str) else stats)
        which = [which] if isinstance(which, int) else list(which)
        panels = [(name, w) for name in names for w in which]
        if axes is None:
            ncols = len(which)
            fig, axes = plt.subplots(len(names), ncols, figsize=(4.4 * ncols, 3.6 * len(names)), squeeze=False)
            axes = axes.ravel()
        else:
            axes = np.ravel(axes)
            fig = axes[0].figure
        titles = {1: "Residuals vs. fitted", 2: "Scale-location", 3: "Normal Q-Q"}
        for ax, (name, kind) in zip(axes, panels):
            t = self[name]
            r = t.pearson
            ok = np.isfinite(r)
            if kind == 3:
                order = np.argsort(r[ok])
                q = norm.ppf((np.arange(1, ok.sum() + 1) - 0.5) / ok.sum())
                ax.scatter(q, r[ok][order], s=12, facecolors="none", edgecolors="black")
                if ok.sum() > 1:
                    q1, q3 = np.quantile(r[ok], [0.25, 0.75])
                    slope = (q3 - q1) / (norm.ppf(0.75) - norm.ppf(0.25))
                    ax.axline((0, q1 - slope * norm.ppf(0.25)), slope=slope, color="grey", linewidth=1)
                ax.set_xlabel("theoretical quantiles")
                ax.set_ylabel("Pearson residuals")
            else:
                if against is None:
                    x, xlabel = t.fitted, "fitted values"
                else:
                    from ._lm import LmError, evaluate

                    try:
                        x = evaluate(against, [{**a, ".fitted": f} for a, f in zip(self.attributes, t.fitted)])
                    except LmError as e:
                        raise ValueError(f"gofN plot: {e}") from None
                    xlabel = str(against).lstrip("~").strip()
                y = r if kind == 1 else np.sqrt(np.abs(r))
                x = np.asarray(x, dtype=float)
                ax.scatter(x, y, s=12, facecolors="none", edgecolors="black")
                curve = _smooth(x, y, 1 / (t.var - t.var_obs))
                if curve is not None:
                    ax.plot(*curve, color="tab:red", linewidth=1)
                ax.axhline(0, linestyle=":", color="grey")
                n = ok.sum()
                cut = norm.ppf((n + 0.5) / (n + 1)) if n else np.inf
                ranked = np.argsort(-np.where(ok, np.abs(r), -np.inf))[:id_n]
                for k in ranked:
                    if ok[k] and abs(r[k]) > cut:
                        ax.annotate(t.labels[k], (x[k], y[k]), fontsize=7, xytext=(3, 3),
                                    textcoords="offset points")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Pearson residuals" if kind == 1 else "sqrt(|Pearson residuals|)")
            ax.set_title(f"{titles[kind]}: {name}", fontsize=10)
        fig.tight_layout()
        return fig
