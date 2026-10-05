"""Bayesian ERGMs, as R's Bergm: the posterior distribution of the
coefficients by the approximate exchange algorithm (`Caimo and Friel 2011
<https://doi.org/10.1016/j.socnet.2010.09.004>`__), with adaptive direction
sampling across a population of chains.

The ERGM's likelihood has an intractable normalizing constant, z(theta), so
Metropolis-Hastings can't evaluate its ratio. The exchange algorithm draws,
for each proposed theta', an auxiliary network y' from the model at theta'
(by MCMC), and accepts theta' with probability

    min(1, exp((theta - theta') . (g(y') - g(y))) p(theta') / p(theta)),

in which the normalizing constants cancel. Proposals are those of adaptive
direction sampling: chain h moves to theta_h + gamma (theta_a - theta_b) + e,
with a and b two other chains and e a small normal step, so the population
of chains learns the posterior's scale and correlations."""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from . import _core
from ._estimation import (
    Control,
    _lower_bounds,
    _upper_bounds,
    _max_edges,
    autocorrelation_time,
    contrastive_divergence,
    mple,
    split_rhat,
)
from ._model import BoundModel, bind
from ._network import to_graphs

log = logging.getLogger("ergmx")


def _log_prior(mean: np.ndarray, sigma: np.ndarray):
    """The log density of the normal prior, up to a constant."""
    chol = np.linalg.cholesky(sigma)

    def value(x):
        z = np.linalg.solve(chol, x - mean)
        return -0.5 * float(z @ z)

    return value


class _Aux:
    """Auxiliary networks: one MCMC run per proposal, in parallel."""

    def __init__(self, model: BoundModel, aux_iters: int, triadic_weight, rng):
        self.model, self.aux_iters, self.rng = model, aux_iters, rng
        self.triadic = model.triadic_weight(triadic_weight)
        self.max_edges = _max_edges(model, Control())

    def _run(self, starts, etas, burnin, conditional=False, keep=False):
        return self.model.simulate(
            starts, [], burnin, 1, 1, int(self.rng.integers(2**63)), conditional=conditional,
            max_edges=self.max_edges, triadic_weight=self.triadic, keep_networks=keep,
            chain_thetas=[[float(v) for v in eta] for eta in etas])

    def draw(self, starts, etas, burnin=None, conditional=False, keep=False):
        """The statistics of a network drawn at each coefficient vector (None
        where the chain ran into the density guard), and the networks."""
        burnin = self.aux_iters - 1 if burnin is None else burnin
        try:
            sample, last, _ = self._run(starts, etas, burnin, conditional, keep)
            return [s[0] for s in sample], last
        except _core.DensityGuardError:
            stats, nets = [], []
            for start, eta in zip(starts, etas):
                try:
                    sample, last, _ = self._run([start], [eta], burnin, conditional, keep)
                    stats.append(sample[0][0])
                    nets.append(last[0])
                except _core.DensityGuardError:
                    stats.append(None)
                    nets.append(start)
            return stats, nets


def bergm(network, formula, *, prior_mean=None, prior_sigma=None, burn_in: int = 100,
          main_iters: int = 1000, aux_iters: int | None = None, nchains: int | None = None,
          gamma: float = 0.5, v_proposal: float = 0.0025, start=None, offset_coef=None,
          constraints=None, bipartite=None, triadic_weight: float | None = None,
          missing_update: int | None = None, seed=None) -> BergmFit:
    """The posterior distribution of an ERGM's coefficients, as R's Bergm's
    ``bergm()`` (and ``bergmM()``, with missing dyads).

    Parameters
    ----------
    network, formula, constraints, offset_coef, bipartite
        As in :func:`ergmx.ergm`. Offset coefficients stay at their values.
    prior_mean, prior_sigma
        The normal prior of the estimated coefficients: by default mean 0 and
        covariance 100 I, as Bergm's. ``prior_sigma`` may be a matrix, a
        vector (the variances) or a number. Decays of curved terms have the
        prior truncated at 0.
    burn_in, main_iters
        Iterations of every chain discarded, and kept.
    aux_iters
        MCMC proposals for each auxiliary network, from the observed one: by
        default at least 1000 (Bergm's default) and as many as the dyads
        that may change. Too few leave the auxiliary networks too like the
        observed one, which widens the posterior and shifts it: Bergm's
        1000 is too few for networks of more than about 50 vertices.
    nchains
        Chains of the population (by default twice the number of
        coefficients, at least 4).
    gamma, v_proposal
        Adaptive direction sampling's scale of the difference between two
        chains, and the variance of the normal step added to it.
    start
        Where the chains start, jittered by U(-0.1, 0.1): by default the MPLE
        (the contrastive divergence estimate, with dyad-dependent
        constraints).
    missing_update
        With missing dyads, the MCMC proposals that update each chain's
        imputation of them at every iteration (by default as many as missing
        dyads, as Bergm's).
    seed : int, optional

    Notes
    -----
    Bergm updates the chains one after the other, each proposal using the
    chains' current states. ergmx updates half of them at a time, with
    differences of the other half's, so that each half's auxiliary networks
    are drawn in parallel; both are valid samplers of the same posterior.

    Returns
    -------
    BergmFit
    """
    t0 = time.time()
    model = bind(network, formula, constraints, offset_coef, fitting=True, bipartite=bipartite)
    free = model.free
    dim = int(free.sum())
    if dim == 0:
        raise ValueError("the model has no coefficients to estimate")
    nchains = max(4, 2 * dim) if nchains is None else int(nchains)
    if nchains < 4:
        raise ValueError("bergm() needs at least 4 chains (the proposals of each half use two of the other's)")
    if aux_iters is None:
        n = model.network.n
        dyads = model.space.n_free if model.space is not None else n * (n - 1) // (1 if model.network.directed else 2)
        aux_iters = int(max(1000, dyads))
    if burn_in < 0 or main_iters < 1 or aux_iters < 1:
        raise ValueError("burn_in must be >= 0, and main_iters and aux_iters >= 1")
    mean = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(-1)
    if mean.shape != (dim,):
        raise ValueError(f"prior_mean must have {dim} values, one per estimated coefficient")
    if prior_sigma is None:
        sigma = 100 * np.eye(dim)
    else:
        sigma = np.asarray(prior_sigma, dtype=float)
        sigma = sigma * np.eye(dim) if sigma.ndim == 0 else np.diag(sigma) if sigma.ndim == 1 else sigma
    if sigma.shape != (dim, dim) or not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError(f"prior_sigma must be a {dim} x {dim} positive definite matrix (or its diagonal)")
    log_prior = _log_prior(mean, sigma)
    lower, upper = _lower_bounds(model)[free], _upper_bounds(model)[free]
    rng = np.random.default_rng(seed)

    if start is None:
        pseudo = mple(model)
        if model.constraints.dyad_dependent:
            pseudo = contrastive_divergence(model, pseudo.theta, Control(), rng)
        centre = pseudo.theta[free]
    else:
        centre = np.asarray(start, dtype=float).reshape(-1)
        if centre.shape == (model.n_params,):
            centre = centre[free]
        if centre.shape != (dim,):
            raise ValueError(f"start must have {dim} values (the estimated coefficients) or {model.n_params}")
    theta = centre[None, :] + rng.uniform(-0.1, 0.1, size=(nchains, dim))
    theta = np.minimum(np.maximum(theta, lower), upper)
    full = np.tile(np.where(model.fixed, model.fixed_values, 0.0), (nchains, 1))

    def eta(free_values):
        th = full[0].copy()
        th[free] = free_values
        return model.eta(th)

    aux = _Aux(model, aux_iters, triadic_weight, rng)
    observed = model.observed()
    edges = model.network.edges
    imputed = [edges] * nchains
    target = np.tile(observed, (nchains, 1))
    missing = model.has_missing
    update = (missing_update or max(1, len(model.network.missing))) if missing else 0
    if missing:
        stats, imputed = aux.draw(imputed, [eta(t) for t in theta], burnin=update, conditional=True)
        target = np.array([s if s is not None else observed for s in stats])

    groups = [np.arange(nchains // 2), np.arange(nchains // 2, nchains)]
    step = math.sqrt(v_proposal)
    draws = np.zeros((nchains, main_iters, dim))
    accepted = proposed = 0
    for k in range(burn_in + main_iters):
        for g, group in enumerate(groups):
            others = groups[1 - g]
            proposals = np.empty((len(group), dim))
            for row, h in enumerate(group):
                a, b = rng.choice(others, size=2, replace=False)
                proposals[row] = theta[h] + gamma * (theta[a] - theta[b]) + step * rng.standard_normal(dim)
            inside = np.all((proposals >= lower) & (proposals <= upper), axis=1)
            etas = [eta(p) for p in proposals]
            stats, _ = aux.draw([imputed[h] for h in group], etas)
            for row, h in enumerate(group):
                if not inside[row] or stats[row] is None:
                    continue
                now, then = eta(theta[h]), etas[row]
                same = now == then
                d = np.where(same, 0.0, now - then)
                delta = np.where(same, 0.0, np.asarray(stats[row]) - target[h])
                log_ratio = float(d @ delta) + log_prior(proposals[row]) - log_prior(theta[h])
                if k >= burn_in:
                    proposed += 1
                if np.isfinite(log_ratio) and log_ratio >= math.log(rng.uniform()):
                    theta[h] = proposals[row]
                    if k >= burn_in:
                        accepted += 1
            if missing:
                stats, nets = aux.draw([imputed[h] for h in group], [eta(theta[h]) for h in group],
                                       burnin=update, conditional=True)
                for row, h in enumerate(group):
                    if stats[row] is not None:
                        imputed[h], target[h] = nets[row], stats[row]
        if k >= burn_in:
            draws[:, k - burn_in] = theta
        if (k + 1) % 100 == 0:
            log.info("bergm: iteration %d of %d", k + 1, burn_in + main_iters)
    names = [n for n, f in zip(model.names, free) if f]
    settings = dict(burn_in=burn_in, main_iters=main_iters, aux_iters=aux_iters, nchains=nchains, gamma=gamma,
                    v_proposal=v_proposal, prior_mean=mean, prior_sigma=sigma)
    return BergmFit(model, names, draws, accepted / max(proposed, 1), settings, seed, time.time() - t0)


class BergmFit:
    """The posterior of a Bayesian ERGM (:func:`bergm`): the chains'
    draws, their summaries (``coef``, the posterior means; ``sd``,
    ``quantiles``), MCMC diagnostics (``acceptance_rate``, ``ess``,
    ``rhat``), :meth:`summary`, :meth:`plot`, and the posterior predictive
    :meth:`gof` and :meth:`simulate`."""

    def __init__(self, model: BoundModel, names: list[str], chains: np.ndarray, acceptance_rate: float,
                 settings: dict, seed, seconds: float):
        self._model, self.names, self.chains = model, names, chains
        self.acceptance_rate, self.settings, self._seed, self.seconds = acceptance_rate, settings, seed, seconds

    @property
    def draws(self) -> np.ndarray:
        """Every chain's draws, one after the other (draws x coefficients), as Bergm's ``Theta``."""
        return self.chains.reshape(-1, self.chains.shape[-1])

    @property
    def mean(self) -> np.ndarray:
        return self.draws.mean(axis=0)

    @property
    def coef(self) -> dict[str, float]:
        """The posterior means, by name."""
        return dict(zip(self.names, self.mean.tolist()))

    @property
    def sd(self) -> dict[str, float]:
        """The posterior standard deviations, by name."""
        return dict(zip(self.names, self.draws.std(axis=0, ddof=1).tolist()))

    @property
    def cov(self) -> np.ndarray:
        """The posterior covariance of the estimated coefficients."""
        return np.atleast_2d(np.cov(self.draws, rowvar=False))

    def quantiles(self, probs=(0.025, 0.25, 0.5, 0.75, 0.975)) -> np.ndarray:
        """Posterior quantiles (coefficients x probabilities)."""
        return np.quantile(self.draws, probs, axis=0).T

    @property
    def params(self) -> np.ndarray:
        """All the model's coefficients at the posterior mean (offsets at their values)."""
        theta = np.where(self._model.fixed, self._model.fixed_values, 0.0)
        theta[self._model.free] = self.mean
        return theta

    @property
    def tau(self) -> np.ndarray:
        """Integrated autocorrelation time of each coefficient, pooling the chains."""
        return autocorrelation_time(self.chains)

    @property
    def ess(self) -> dict[str, float]:
        """Effective sample size of each coefficient's draws."""
        return dict(zip(self.names, (self.draws.shape[0] / self.tau).tolist()))

    @property
    def rhat(self) -> dict[str, float]:
        """Split R-hat of each coefficient across the chains: about 1 when they agree."""
        return dict(zip(self.names, split_rhat(self.chains).tolist()))

    def summary(self) -> BergmSummary:
        """The posterior means, standard deviations, naive and time-series
        standard errors of the means, quantiles and diagnostics, as Bergm's
        ``summary()``."""
        return BergmSummary(self)

    def to_frame(self):
        """The draws as a pandas DataFrame (one column per coefficient, a
        column ``chain``). Needs pandas."""
        import pandas as pd

        frame = pd.DataFrame(self.draws, columns=self.names)
        frame.insert(0, "chain", np.repeat(np.arange(1, self.chains.shape[0] + 1), self.chains.shape[1]))
        return frame

    def _thetas(self, n: int, rng) -> np.ndarray:
        """n draws from the posterior (without replacement, if there are enough)."""
        draws = self.draws
        rows = rng.choice(len(draws), size=n, replace=n > len(draws))
        full = np.tile(np.where(self._model.fixed, self._model.fixed_values, 0.0), (n, 1))
        full[:, self._model.free] = draws[rows]
        return full

    def _simulate(self, n: int, seed, aux_iters: int, keep: bool, conditional=False):
        rng = np.random.default_rng(seed)
        thetas = self._thetas(n, rng)
        model = self._model
        aux = _Aux(model, aux_iters, None, rng)
        sample, _, networks = model.simulate(
            [model.network.edges] * n, [], aux_iters, 1, 1, int(rng.integers(2**63)), conditional=conditional,
            triadic_weight=aux.triadic, keep_networks=keep, chain_thetas=[model.eta(t).tolist() for t in thetas])
        return sample[:, 0], [nets[0] for nets in networks] if keep else None

    def simulate(self, nsim: int = 1, *, seed=None, output: str = "network", aux_iters: int = 10000):
        """Networks from the posterior predictive distribution: each from the
        model at a posterior draw, after ``aux_iters`` MCMC proposals from the
        observed network."""
        if output not in ("network", "stats"):
            raise ValueError(f"output must be 'network' or 'stats', not {output!r}")
        stats, networks = self._simulate(nsim, seed, aux_iters, keep=output == "network")
        if output == "stats":
            return stats
        return [to_graphs(self._model.network, e) for e in networks]

    def gof(self, sample_size: int = 100, *, aux_iters: int = 10000, stats=None, seed=None):
        """Bayesian goodness of fit, as Bergm's ``bgof()``: the observed
        network's degree, edgewise shared partner and distance distributions
        (and model statistics) against those of networks from the posterior
        predictive distribution (:meth:`simulate`). With missing dyads, the
        observed distributions average networks imputed at posterior draws.
        Returns a :class:`~ergmx.GofResult`; ``.plot()`` it."""
        from ._gof import GofResult, _tables, check_stats, default_stats

        network = self._model.network
        stats = default_stats(network) if stats is None else stats
        check_stats(network, stats)
        rng = np.random.default_rng(seed)
        model_stats, simulated = self._simulate(sample_size, rng.integers(2**63), aux_iters, keep=True)
        if self._model.has_missing:
            imputed_stats, imputed = self._simulate(sample_size, rng.integers(2**63), aux_iters, keep=True,
                                                    conditional=True)
        else:
            imputed, imputed_stats = [network.edges], self._model.observed()[None, :]
        return GofResult(_tables(self._model, stats, simulated, model_stats, imputed, imputed_stats), sample_size)

    def plot(self, names=None, lags: int = 30):
        """For each coefficient, its posterior density, the chains' traces
        and their autocorrelation, as Bergm's ``plot()``. Needs matplotlib.
        Returns the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('plotting needs matplotlib: install "ergmx[plot]"') from None
        from scipy.stats import gaussian_kde

        chosen = self.names if names is None else [names] if isinstance(names, str) else list(names)
        fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 2.8 * len(chosen)), squeeze=False)
        for row, name in zip(axes, chosen):
            k = self.names.index(name)
            values = self.draws[:, k]
            grid = np.linspace(values.min(), values.max(), 200)
            if np.ptp(values) > 0:
                row[0].plot(grid, gaussian_kde(values)(grid), color="black")
            row[0].set_title(f"posterior: {name}", fontsize=10)
            for chain in self.chains[:, :, k]:
                row[1].plot(chain, linewidth=0.6, alpha=0.8)
            row[1].set_title("traces", fontsize=10)
            row[1].set_xlabel("iteration")
            x = self.chains[:, :, k] - self.chains[:, :, k].mean(axis=1, keepdims=True)
            n = x.shape[1]
            acf = [1.0] + [float(np.mean([np.dot(c[:-lag], c[lag:]) / max(np.dot(c, c), 1e-300) * n / (n - lag)
                                          for c in x])) for lag in range(1, min(lags, n - 1) + 1)]
            row[2].bar(range(len(acf)), acf, width=0.5, color="grey")
            row[2].axhline(0, color="black", linewidth=0.8)
            row[2].set_title("autocorrelation", fontsize=10)
            row[2].set_xlabel("lag")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        width = max(map(len, self.names))
        rows = "\n".join(f"  {n:<{width}}  {m: .4f}  (sd {s:.4f})"
                         for n, m, s in zip(self.names, self.mean, self.sd.values()))
        return f"BergmFit (posterior means):\n{rows}"


class BergmSummary:
    """The posterior summary of a :class:`BergmFit`, printed like Bergm's ``summary()``."""

    def __init__(self, fit: BergmFit):
        self.fit = fit

    def to_frame(self):
        """As a pandas DataFrame. Needs pandas."""
        import pandas as pd

        f = self.fit
        n = f.draws.shape[0]
        sd = f.draws.std(axis=0, ddof=1)
        q = f.quantiles()
        columns = {"Mean": f.mean, "SD": sd, "Naive SE": sd / np.sqrt(n),
                   "Time-series SE": sd * np.sqrt(f.tau / n), "2.5%": q[:, 0], "25%": q[:, 1], "50%": q[:, 2],
                   "75%": q[:, 3], "97.5%": q[:, 4], "ESS": n / f.tau, "R-hat": np.array(list(f.rhat.values()))}
        return pd.DataFrame(columns, index=f.names)

    def __str__(self) -> str:
        f = self.fit
        n = f.draws.shape[0]
        sd = f.draws.std(axis=0, ddof=1)
        q = f.quantiles()
        width = max(len(name) for name in f.names)
        calibrated = f.settings.get("method") == "bergmC"
        kind = "Calibrated pseudo-posterior (bergmC)" if calibrated else "Posterior density estimate"
        lines = [f"{kind} for the model: {f._model.formula!r}", "",
                 f"{'':<{width}}  {'Mean':>9}  {'SD':>8}  {'Naive SE':>8}  {'Time-series SE':>14}  "
                 f"{'ESS':>6}  {'R-hat':>6}"]
        rhat = list(f.rhat.values())
        for k, name in enumerate(f.names):
            lines.append(f"{name:<{width}}  {f.mean[k]:9.4f}  {sd[k]:8.4f}  {sd[k] / math.sqrt(n):8.4f}  "
                         f"{sd[k] * math.sqrt(f.tau[k] / n):14.4f}  {n / f.tau[k]:6.0f}  {rhat[k]:6.3f}")
        lines += ["", f"{'':<{width}}  " + "  ".join(f"{h:>8}" for h in ("2.5%", "25%", "50%", "75%", "97.5%"))]
        for k, name in enumerate(f.names):
            lines.append(f"{name:<{width}}  " + "  ".join(f"{v:8.4f}" for v in q[k]))
        s = f.settings
        if calibrated:
            lines += ["", f"Acceptance rate: {f.acceptance_rate:.2f}.  The calibrated pseudo-posterior of "
                          f"{s['main_iters']} draws (after {s['burn_in']}), moved to the posterior mode found in "
                          f"{s['rm_iters']} Robbins-Monro steps."]
        else:
            lines += ["", f"Acceptance rate: {f.acceptance_rate:.2f}.  {s['nchains']} chains of {s['main_iters']} "
                          f"draws (after {s['burn_in']}), auxiliary networks of {s['aux_iters']} proposals."]
        if max(rhat) > 1.1:
            lines.append("Some R-hat are above 1.1: the chains disagree; run longer chains (main_iters).")
        return "\n".join(lines)

    __repr__ = __str__
