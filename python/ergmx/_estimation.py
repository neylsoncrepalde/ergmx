"""Estimation: MPLE, contrastive divergence, and Monte Carlo MLE with Hummel
et al. (2012) stepping."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.special import expit

from . import _core
from ._model import BoundModel

log = logging.getLogger("ergmx")


class DegeneracyError(RuntimeError):
    """The model could not be fitted: the simulated networks are very unlike the
    observed one, a sign that the model is degenerate or the starting values poor."""


def _default_chains() -> int:
    return max(1, min(4, os.cpu_count() or 1))


@dataclass
class Control:
    """Tuning parameters of the MCMC and of the Monte Carlo MLE.

    The defaults follow ergm: 1024 samples, 1024 proposals apart, after 16
    times as many burn-in proposals. When the samples are too autocorrelated
    to reach ``effective_size``, the interval (and the burn-in) grows.
    """

    #: Networks sampled per iteration, split across the chains.
    samplesize: int = 1024
    #: MCMC proposals between two sampled networks, at the start.
    interval: int = 1024
    #: MCMC proposals discarded at the start of every chain, at the start.
    burnin: int = 16384
    #: Parallel Markov chains (one thread each).
    n_chains: int = field(default_factory=_default_chains)
    #: Share of triadic MCMC proposals (they close or open a triangle). None
    #: uses 0.5 for models with triangle or shared partner terms, 0 otherwise.
    triadic_weight: float | None = None
    #: Minimum effective sample size per iteration; None keeps the interval fixed.
    effective_size: int | None = 64
    #: Largest interval the adaptation may reach.
    max_interval: int = 2**20
    #: Maximum number of Monte Carlo MLE iterations.
    max_iter: int = 60
    #: Margin by which the target must lie inside the sample's convex hull.
    steplength_margin: float = 0.05
    #: The final iteration samples this many times more networks.
    last_boost: int = 4
    #: Stop if a simulated network has more edges than this many times the
    #: observed ones (and more than `density_guard_min`), as ergm does.
    density_guard: float = float(np.exp(3))
    density_guard_min: int = 10000
    #: Stop after this many consecutive iterations with a step length below 0.1;
    #: None never stops early.
    stall_iterations: int | None = 10
    #: Contrastive divergence: MCMC proposals per sample, samples per iteration,
    #: maximum iterations, and the convergence p-value, as in ergm.
    cd_steps: int = 8
    cd_samplesize: int = 1024
    cd_max_iter: int = 60
    cd_conv_min_pval: float = 0.5
    #: Log-likelihood by path sampling: intervals along the path (points at
    #: both ends of each), chains per point, and samples per chain (at the
    #: MLE's MCMC interval).
    bridges: int = 32
    bridge_chains: int = 1
    bridge_samplesize: int = 256

    def per_chain(self) -> int:
        return max(1, self.samplesize // self.n_chains)


@dataclass
class Estimate:
    theta: np.ndarray
    cov: np.ndarray  # total covariance of the estimate
    mc_cov: np.ndarray | None  # part of it due to MCMC
    loglik: float | None
    method: str
    iterations: int
    converged: bool
    sample: np.ndarray | None  # last sample of statistics (chains x samples x stats)
    interval: int | None = None  # MCMC interval of the last iteration
    pvalue: float | None = None  # final test that the model reproduces the observed statistics
    loglik_se: float | None = None  # Monte Carlo standard error of the log-likelihood


# -- MPLE -------------------------------------------------------------------------


def logistic_regression(x, y, weights, max_iter=100, tol=1e-10):
    """Weighted logistic regression by Newton-Raphson.

    Returns the coefficients, their covariance and the log-likelihood.
    """

    def loglik(beta):
        eta = x @ beta
        return float(np.sum(weights * (y * eta - np.logaddexp(0, eta))))

    beta = np.zeros(x.shape[1])
    current = loglik(beta)
    for _ in range(max_iter):
        mu = expit(x @ beta)
        grad = x.T @ (weights * (y - mu))
        hess = (x * (weights * mu * (1 - mu))[:, None]).T @ x
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        new_beta, new = beta + step, loglik(beta + step)
        while new < current - 1e-12 and np.max(np.abs(step)) > tol:  # step halving
            step /= 2
            new_beta, new = beta + step, loglik(beta + step)
        beta, current = new_beta, new
        if np.max(np.abs(step)) < tol:
            break
    else:
        warnings.warn("the MPLE logistic regression did not converge", stacklevel=3)
    mu = expit(x @ beta)
    hess = (x * (weights * mu * (1 - mu))[:, None]).T @ x
    return beta, np.linalg.pinv(hess), current


def mple(model: BoundModel) -> Estimate:
    x, y = model.core.mple_data(model.network.edges)
    # Many dyads share the same change statistics: fit on the distinct rows.
    rows, counts = np.unique(np.column_stack([x, y]), axis=0, return_counts=True)
    theta, cov, loglik = logistic_regression(rows[:, :-1], rows[:, -1], counts.astype(float))
    return Estimate(theta, cov, None, loglik, "MPLE", 0, True, None, loglik_se=0.0)


# -- Monte Carlo MLE --------------------------------------------------------------


def _in_hull(sample: np.ndarray, point: np.ndarray) -> bool:
    """Whether `point` is a convex combination of the rows of `sample`."""
    k = sample.shape[0]
    a_eq = np.vstack([sample.T, np.ones(k)])
    b_eq = np.append(point, 1.0)
    result = optimize.linprog(np.zeros(k), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return result.status == 0


def hummel_steplength(sample, observed, margin, min_step=1e-4) -> float:
    """Largest step towards the observed statistics that stays inside the
    convex hull of the sample (with a margin), as in Hummel et al. (2012).

    Searched on a log scale down to `min_step`, which is returned if even that
    leaves the hull, so that the estimate keeps moving.
    """
    # Hull membership is unchanged by rescaling; standardizing helps the solver.
    mean, sd = sample.mean(axis=0), sample.std(axis=0)
    sd[sd == 0] = 1.0
    z, towards = (sample - mean) / sd, (1 + margin) * (observed - mean) / sd
    if _in_hull(z, towards):
        return 1.0
    if not _in_hull(z, min_step * towards):
        return min_step
    low, high = np.log(min_step), 0.0
    while high - low > 0.05:  # 5% relative precision
        mid = (low + high) / 2
        low, high = (mid, high) if _in_hull(z, np.exp(mid) * towards) else (low, mid)
    return float(np.exp(low))


def autocorrelation_time(sample: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation time of each statistic, pooling the chains.

    Uses Geyer's (1992) initial monotone sequence estimator on the
    autocovariances averaged over chains. Independent samples give 1.
    """
    chains, n, p = sample.shape
    x = sample - sample.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=size, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=size, axis=1)[:, :n].mean(axis=0) / n
    tau = np.ones(p)
    for k in range(p):
        g = acov[:, k]
        if g[0] <= 0:
            continue
        pairs = g[: n - n % 2].reshape(-1, 2).sum(axis=1)  # gamma(2m) + gamma(2m + 1)
        stop = np.flatnonzero(pairs <= 0)
        pairs = np.minimum.accumulate(pairs[: stop[0] if stop.size else pairs.size])
        tau[k] = max(1.0, (2 * pairs.sum() - g[0]) / g[0])
    return tau


def mean_covariance(sample: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Covariance of the mean of the sample, inflated by the autocorrelation times."""
    flat = sample.reshape(-1, sample.shape[-1])
    scale = np.sqrt(tau)
    return np.atleast_2d(np.cov(flat, rowvar=False)) * np.outer(scale, scale) / flat.shape[0]


def split_rhat(sample: np.ndarray) -> np.ndarray:
    """Split R-hat of each statistic (Gelman et al. 2013): about 1 when the chains
    agree; above 1.01 to 1.1 suggests they have not mixed."""
    chains, n, p = sample.shape
    half = n // 2
    if half < 2:
        return np.full(p, np.nan)
    parts = np.concatenate([sample[:, :half], sample[:, half:2 * half]])
    within = parts.var(axis=1, ddof=1).mean(axis=0)
    between = half * parts.mean(axis=1).var(axis=0, ddof=1)
    pooled = (half - 1) / half * within + between / half
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(within > 0, np.sqrt(pooled / within), np.nan)


def hotelling_pvalue(sample, observed, tau) -> float:
    """p-value of Hotelling's T^2 test that the model's mean statistics are the
    observed ones, with the effective sample size as the sample size."""
    p = sample.shape[-1]
    n_eff = sample.shape[0] * sample.shape[1] / tau.max()
    if n_eff <= p + 1:
        return 0.0
    diff = sample.reshape(-1, p).mean(axis=0) - observed
    t2 = float(diff @ np.linalg.pinv(mean_covariance(sample, tau)) @ diff)
    return float(stats.f.sf(t2 * (n_eff - p) / (p * (n_eff - 1)), p, n_eff - p))


def _step(sample, observed, margin):
    """Mean, covariance and the log-normal Newton step of a sample."""
    flat = sample.reshape(-1, sample.shape[-1])
    mean, cov = flat.mean(axis=0), np.atleast_2d(np.cov(flat, rowvar=False))
    gamma = hummel_steplength(flat, observed, margin)
    # Log-normal approximation: the log-likelihood ratio is quadratic in theta,
    # maximized at a Newton step towards the pseudo-observed target.
    step = np.linalg.lstsq(cov, gamma * (observed - mean), rcond=None)[0]
    return cov, gamma, step


def _max_edges(model: BoundModel, control: Control) -> int:
    observed_edges = len(model.network.edges)
    return int(max(control.density_guard_min, control.density_guard * observed_edges))


def _density_guard_error(model: BoundModel, max_edges: int) -> DegeneracyError:
    observed = len(model.network.edges)
    return DegeneracyError(
        f"a simulated network had more than {max_edges} edges, over "
        f"{max_edges / max(observed, 1):.0f} times the {observed} observed: a strong sign that "
        "the model is degenerate, or that the starting coefficients are poor. If you are "
        "confident neither is the case, raise Control.density_guard."
    )


def _simulate(model: BoundModel, starts, theta, burnin, interval, samples, rng, control,
              check_variance=True):
    """Sample statistics, stopping with a DegeneracyError if the density guard
    trips or (with `check_variance`) a statistic does not vary."""
    max_edges = _max_edges(model, control)
    try:
        sample, last, _ = model.core.simulate(
            starts, theta.tolist(), burnin, interval, samples, int(rng.integers(2**63)),
            triadic_weight=model.triadic_weight(control.triadic_weight), max_edges=max_edges,
        )
    except _core.DensityGuardError:
        raise _density_guard_error(model, max_edges) from None
    variances = sample.reshape(-1, model.n_stats).var(axis=0)
    if check_variance and np.any(variances <= 0):
        constant = [n for n, v in zip(model.names, variances) if v <= 0]
        raise DegeneracyError(
            f"the simulated statistics {constant} did not vary: the model may be degenerate "
            "at these coefficients, or the MCMC too short"
        )
    return sample, last


def _stall_message(model: BoundModel, sample: np.ndarray, iterations: int) -> str:
    observed = model.observed()
    mean = sample.reshape(-1, model.n_stats).mean(axis=0)
    width = max(map(len, model.names))
    rows = "\n".join(f"  {n:<{width}}  {m:12.2f}  {o:12.2f}"
                      for n, m, o in zip(model.names, mean, observed))
    lines = [
        f"the Monte Carlo MLE is not making progress: for {iterations} iterations the "
        "observed statistics were far outside the range of the simulated networks (step "
        "lengths below 0.1). The model may be degenerate, or the starting coefficients poor.",
        "",
        "At the current coefficients:",
        f"  {'':<{width}}  {'simulated':>12}  {'observed':>12}",
        rows,
    ]
    rhat = split_rhat(sample)
    if np.nanmax(rhat) > 1.2:
        lines += ["", f"The chains disagree (R-hat up to {np.nanmax(rhat):.1f}): the simulated "
                      "networks jump between very different regimes, a typical sign of degeneracy."]
    lines += ["", "Things to try: other terms (for example gwesp with a smaller decay instead of "
                  "triangle), adding gwdegree or attribute terms, init='CD', or a longer MCMC "
                  "(interval=...). Control(stall_iterations=None) keeps iterating."]
    return "\n".join(lines)


def contrastive_divergence(model: BoundModel, init: np.ndarray, control: Control,
                           rng: np.random.Generator) -> Estimate:
    """Contrastive divergence (Hinton 2002; Krivitsky 2017), as ergm's CD: each
    sample is `cd_steps` MCMC proposals away from the observed network, so the
    observed statistics stay in range, and the estimate moves by log-normal
    steps until the samples are centered on the observed statistics."""
    observed = model.observed()
    theta = np.asarray(init, dtype=float).copy()
    starts = [model.network.edges] * control.cd_samplesize
    converged = False
    for iteration in range(1, control.cd_max_iter + 1):
        # A few proposals may leave a rare statistic unchanged: no variance check.
        sample, _ = _simulate(model, starts, theta, control.cd_steps - 1, 1, 1, rng, control,
                              check_variance=False)
        sample = sample.reshape(1, -1, model.n_stats)  # independent samples
        _, gamma, step = _step(sample, observed, control.steplength_margin)
        pvalue = hotelling_pvalue(sample, observed, np.ones(model.n_stats))
        log.info("CD iteration %d: step length %.2f, p-value %.3f", iteration, gamma, pvalue)
        theta = theta + step
        if gamma == 1.0 and pvalue > control.cd_conv_min_pval:
            converged = True
            break
    nan = np.full((model.n_stats, model.n_stats), np.nan)
    return Estimate(theta, nan, None, None, "CD", iteration, converged, None)


def mcmle(model: BoundModel, init: np.ndarray, control: Control, rng: np.random.Generator) -> Estimate:
    """Monte Carlo MLE, stopping when the step length is 1 in two consecutive
    iterations (Hummel et al. 2012), then refining with a larger sample."""
    observed = model.observed()
    theta = np.asarray(init, dtype=float).copy()
    starts = [model.network.edges] * control.n_chains
    interval, burnin = control.interval, control.burnin

    def simulate(samples):
        nonlocal starts
        sample, starts = _simulate(model, starts, theta, burnin, interval, samples, rng, control)
        return sample

    converged, full_steps, stalled = False, 0, 0
    for iteration in range(1, control.max_iter + 1):
        sample = simulate(control.per_chain())
        tau = autocorrelation_time(sample)
        _, gamma, step = _step(sample, observed, control.steplength_margin)
        ess = sample.shape[0] * sample.shape[1] / tau.max()
        log.info("iteration %d: interval %d, effective size %.0f, step length %.2f",
                 iteration, interval, ess, gamma)
        stalled = stalled + 1 if gamma < 0.1 else 0
        if control.stall_iterations and stalled >= control.stall_iterations:
            raise DegeneracyError(_stall_message(model, sample, stalled))
        theta = theta + step
        # The convex hull test means little with few effective samples.
        enough = not control.effective_size or ess >= control.effective_size
        full_steps = full_steps + 1 if gamma == 1.0 and enough else 0
        if full_steps == 2:
            converged = True
            break
        if control.effective_size and ess < control.effective_size and interval < control.max_interval:
            factor = min(4, 1 << int(np.ceil(np.log2(control.effective_size / ess))))
            interval = min(control.max_interval, interval * factor)
            burnin = max(burnin, 16 * interval)
    else:
        warnings.warn(
            f"the Monte Carlo MLE did not converge in {control.max_iter} iterations",
            stacklevel=3,
        )

    sample = simulate(control.per_chain() * control.last_boost)
    tau = autocorrelation_time(sample)
    pvalue = hotelling_pvalue(sample, observed, tau)
    cov, gamma, step = _step(sample, observed, control.steplength_margin)
    log.info("final iteration: interval %d, step length %.2f, p-value %.3f", interval, gamma, pvalue)
    info = np.linalg.pinv(cov)  # inverse Fisher information
    mc_cov = info @ mean_covariance(sample, tau) @ info
    return Estimate(theta + step, info + mc_cov, mc_cov, None, "MCMLE", iteration + 1, converged,
                    sample, interval, pvalue)
