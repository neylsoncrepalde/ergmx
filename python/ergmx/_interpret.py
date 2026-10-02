"""Interpreting fitted models: tie probabilities (as R's ``predict.ergm``),
average marginal effects (as R's ergMargins), odds ratios and confidence
intervals."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.special import expit

from . import _core
from ._estimation import Control, fixed_part


def _frame(columns: dict, index=None):
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        raise ImportError('to_frame() needs pandas: install "ergmx[pandas]"') from None
    return pd.DataFrame(columns, index=index)


class NumericTable:
    """A small table of numbers by name, printed like R's matrices; convert it
    with :meth:`to_frame` (needs pandas)."""

    def __init__(self, title: str, index: list[str], columns: dict[str, np.ndarray],
                 digits: int = 4, note: str = ""):
        self.title, self.index, self.columns = title, list(index), {k: np.asarray(v) for k, v in columns.items()}
        self.digits, self.note = digits, note

    def __getitem__(self, column: str) -> np.ndarray:
        return self.columns[column]

    def to_dict(self) -> dict[str, dict[str, float]]:
        """{row name: {column: value}}."""
        return {name: {c: float(v[i]) for c, v in self.columns.items()} for i, name in enumerate(self.index)}

    def to_frame(self):
        """The table as a pandas DataFrame, indexed by name."""
        return _frame(self.columns, self.index)

    def __str__(self) -> str:
        width = max(map(len, self.index), default=0)
        cells = {c: [f"{x:.{self.digits}g}" if c.lower() in ("p", "pr(>|z|)") else f"{x:.{self.digits}f}"
                     for x in v] for c, v in self.columns.items()}
        widths = {c: max(len(c), *map(len, cells[c])) for c in cells}
        lines = [f"{self.title}:", "", " " * width + "".join(f"  {c:>{widths[c]}}" for c in cells)]
        for i, name in enumerate(self.index):
            lines.append(f"{name:<{width}}" + "".join(f"  {cells[c][i]:>{widths[c]}}" for c in cells))
        if self.note:
            lines += ["", self.note]
        return "\n".join(lines)

    __repr__ = __str__


# -- Tie probabilities ------------------------------------------------------------------------


class TiePredictions:
    """Predicted tie probabilities of a model, one per dyad, as R's
    ``predict(fit)``: ``tail``, ``head`` (vertex numbers from 0, as igraph's)
    and ``p``. For several networks combined, ``network`` is each dyad's
    network (from 0), and the vertex numbers are those of the combined
    network."""

    def __init__(self, network, tail, head, p, conditional: bool, link: bool):
        self._network = network
        self.tail, self.head, self.p = tail, head, p
        self.conditional, self.link = conditional, link
        self.network = None
        if network.combined:
            starts = np.array([b.start for b in network.blocks])
            self.network = np.searchsorted(starts, tail, side="right") - 1

    def __len__(self) -> int:
        return len(self.p)

    def matrix(self):
        """An n x n array of the probabilities (symmetric if undirected), 0 on
        the diagonal and NaN for dyads that can't be ties (within a mode of a
        bipartite network); a list of them, one per network, for several
        networks combined."""
        net = self._network
        m = np.full((net.n, net.n), np.nan)
        np.fill_diagonal(m, 0.0)
        m[self.tail, self.head] = self.p
        if not net.directed:
            m[self.head, self.tail] = self.p
        if not net.combined:
            return m
        return [m[b.start:b.stop, b.start:b.stop] for b in net.blocks]

    def mean_by(self, attr: str) -> NumericTable:
        """The mean probability of the dyads between each pair of levels of a
        vertex attribute (from the tail's to the head's level, if directed)."""
        values = self._network.attribute(attr)
        levels = sorted({v for v in values if v is not None}, key=lambda v: (str(type(v)), v))
        code = {v: k for k, v in enumerate(levels)}
        a = np.array([code.get(values[t], -1) for t in self.tail])
        b = np.array([code.get(values[h], -1) for h in self.head])
        keep = (a >= 0) & (b >= 0)
        sums, counts = np.zeros((len(levels),) * 2), np.zeros((len(levels),) * 2)
        np.add.at(sums, (a[keep], b[keep]), self.p[keep])
        np.add.at(counts, (a[keep], b[keep]), 1)
        if not self._network.directed:
            sums, counts = sums + sums.T - np.diag(np.diag(sums)), counts + counts.T - np.diag(np.diag(counts))
        with np.errstate(invalid="ignore"):
            means = sums / counts
        from .terms import _level_name

        names = [_level_name(v) for v in levels]
        kind = "conditional" if self.conditional else "unconditional"
        note = "Rows: the tail's level, columns: the head's." if self._network.directed else ""
        return NumericTable(f"Mean {kind} tie probability by {attr}", names,
                     {name: means[:, k] for k, name in enumerate(names)}, note=note)

    def to_frame(self):
        """The predictions as a pandas DataFrame: tail, head (and network), p."""
        columns = {"tail": self.tail, "head": self.head, "p": self.p}
        if self.network is not None:
            columns = {"network": self.network, **columns}
        return _frame(columns)

    def __repr__(self) -> str:
        kind = "conditional" if self.conditional else "unconditional"
        what = "link (log-odds)" if self.link else "tie probabilities"
        return (f"<TiePredictions: {kind} {what} of {len(self)} dyads, mean "
                f"{np.nanmean(self.p):.4f}>")


def _dyads(model):
    """Every dyad that can be a tie (not within a mode of a bipartite network,
    nor between networks), with its change statistics, as ergm's ``predict``
    (which, like ergm, ignores the constraints and counts missing dyads)."""
    net = model.network
    space = None
    if net.combined or net.bipartite:
        space = _core.Space(net.n, net.directed, net.block_ids(),
                            np.ascontiguousarray(net.mode == 1) if net.bipartite else None)
    x, _, pairs = model.core.mple_data(net.edges, space)
    return x, pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)


def predict(model, theta, *, conditional: bool = True, type: str = "response", nsim: int = 100,
            seed=None, interval: int | None = None, burnin: int | None = None,
            n_chains: int | None = None, triadic_weight: float | None = None) -> TiePredictions:
    """Tie probabilities of a bound model at the parameters `theta`."""
    if type not in ("response", "link"):
        raise ValueError(f"type must be 'response' or 'link', not {type!r}")
    x, tail, head = _dyads(model)
    if conditional:
        link = fixed_part(x, np.arange(model.n_stats), model.eta(theta))
        return TiePredictions(model.network, tail, head, link if type == "link" else expit(link), True,
                              type == "link")
    if type == "link":
        raise ValueError("type='link' is only available for conditional probabilities, as in ergm")
    if int(nsim) < 2:
        raise ValueError("nsim must be 2 or more")
    interval = interval or Control.interval
    chains = max(1, min(n_chains or Control().n_chains, nsim))
    per_chain = -(-nsim // chains)
    rng = np.random.default_rng(seed)
    _, _, networks = model.simulate(
        [model.network.edges] * chains, theta, 16 * interval if burnin is None else burnin, interval,
        per_chain, int(rng.integers(2**63)), keep_networks=True,
        triadic_weight=model.triadic_weight(triadic_weight),
    )
    sims = [e for chain in networks for e in chain][:nsim]
    n = model.network.n
    counts = np.zeros((n, n))
    for edges in sims:
        counts[edges[:, 0], edges[:, 1]] += 1
        if not model.network.directed:
            counts[edges[:, 1], edges[:, 0]] += 1
    return TiePredictions(model.network, tail, head, counts[tail, head] / len(sims), False, False)


# -- Effects -----------------------------------------------------------------------------------


def _estimated(fit) -> np.ndarray:
    """Parameters with an effect: estimated, and not the decay of a curved term."""
    model = fit._model
    decay = np.zeros(model.n_params, dtype=bool)
    from ._model import _decays

    for position, _ in _decays(model.blocks, model.network):
        decay[position] = True
    return model.free & ~decay


def marginal_effects(fit) -> NumericTable:
    """Average marginal effects on the conditional tie probability."""
    model, theta = fit._model, fit.params
    if fit.method == "CD":
        raise ValueError("contrastive divergence estimates have no covariance: fit the MLE")
    x, _, _ = _dyads(model)
    eta = model.eta(theta)
    finite = np.isfinite(eta)
    jac = model.jacobian(theta)[finite]
    xf = x[:, finite]
    p = expit(fixed_part(x, np.arange(model.n_stats), eta))
    w = p * (1 - p)
    mean_w = w.mean()
    free = model.free
    # d mean(p(1 - p)) / d theta, through the linear predictor X eta(theta).
    dmean = ((w * (1 - 2 * p))[:, None] * (xf @ jac)).mean(axis=0)
    rows = np.flatnonzero(_estimated(fit))
    cov = np.zeros((model.n_params, model.n_params))
    cov[np.ix_(free, free)] = fit._estimate.cov[np.ix_(free, free)]
    ame, se = np.empty(len(rows)), np.empty(len(rows))
    for r, k in enumerate(rows):
        grad = theta[k] * dmean
        grad[k] += mean_w
        ame[r], se[r] = theta[k] * mean_w, math.sqrt(max(grad @ cov @ grad, 0.0))
    z = ame / se
    pvalue = 2 * stats.norm.sf(np.abs(z))
    names = [fit.names[k] for k in rows]
    note = (f"Averages over the {len(p)} dyads of the change in each dyad's conditional tie "
            "probability per unit of the term's statistic, theta * p (1 - p); delta-method "
            "standard errors.")
    return NumericTable("Average marginal effects on the tie probability", names,
                 {"AME": ame, "Delta SE": se, "Z": z, "P": pvalue}, note=note)


def confint(fit, level: float = 0.95) -> NumericTable:
    """Wald confidence intervals of the estimated coefficients, as R's
    ``confint()``."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    keep = fit._model.free
    z = stats.norm.ppf(0.5 + level / 2)
    coef, se = fit.params[keep], np.sqrt(np.diag(fit.cov))[keep]
    low, high = (100 * (1 - level) / 2), 100 * (1 + level) / 2
    return NumericTable(f"{100 * level:g}% confidence intervals", [n for n, k in zip(fit.names, keep) if k],
                 {f"{low:g} %": coef - z * se, f"{high:g} %": coef + z * se})


def odds_ratios(fit, level: float = 0.95) -> NumericTable:
    """Odds ratios, exp(coefficient), with Wald confidence intervals."""
    keep = _estimated(fit)
    intervals = confint(fit, level)
    names = [n for n, k in zip(fit.names, keep) if k]
    index = [intervals.index.index(n) for n in names]
    low, high = (np.exp(v[index]) for v in intervals.columns.values())
    columns = list(intervals.columns)
    note = ("The factor by which a unit increase in the term's statistic multiplies the "
            "conditional odds of a tie.")
    return NumericTable(f"Odds ratios, with {100 * level:g}% confidence intervals", names,
                 {"Odds ratio": np.exp(fit.params[keep]), columns[0]: low, columns[1]: high}, note=note)
