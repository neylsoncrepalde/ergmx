"""Estimation: MPLE, contrastive divergence, and Monte Carlo MLE with `Hummel
et al. (2012) <https://doi.org/10.1080/10618600.2012.679224>`__ stepping."""

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
    #: Stop if a simulated network has more than this many times the observed
    #: edges (and more than ``density_guard_min``), as ergm does.
    density_guard: float = float(np.exp(3))
    #: Edges a simulated network may always have, whatever the density guard.
    density_guard_min: int = 10000
    #: Stop after this many consecutive iterations with a step length below 0.1;
    #: None never stops early.
    stall_iterations: int | None = 10
    #: Contrastive divergence: MCMC proposals from the observed network per sample.
    cd_steps: int = 8
    #: Contrastive divergence: samples per iteration.
    cd_samplesize: int = 1024
    #: Contrastive divergence: maximum number of iterations.
    cd_max_iter: int = 60
    #: Contrastive divergence stops when Hotelling's test of the samples
    #: against the observed statistics has a larger p-value, as in ergm.
    cd_conv_min_pval: float = 0.5
    #: Log-likelihood by path sampling: intervals along the path, with
    #: samples at both ends of each.
    bridges: int = 32
    #: Log-likelihood by path sampling: chains per point of the path.
    bridge_chains: int = 1
    #: Log-likelihood by path sampling: samples per chain, spaced by the
    #: interval the Monte Carlo MLE ended with.
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
    sample_obs: np.ndarray | None = None  # last sample conditional on the observed dyads
    #: Whether the log-likelihood is relative to the null model (dyad-dependent constraints).
    loglik_relative: bool = False


# -- MPLE -------------------------------------------------------------------------


def logistic_regression(x, y, weights, offset=None, max_iter=100, tol=1e-10):
    """Weighted logistic regression by Newton-Raphson, with an optional fixed
    offset added to the linear predictor.

    Returns the coefficients, their covariance and the log-likelihood.
    """
    offset = np.zeros(len(y)) if offset is None else offset

    def loglik(beta):
        eta = x @ beta + offset
        return float(np.sum(weights * (y * eta - np.logaddexp(0, eta))))

    beta = np.zeros(x.shape[1])
    current = loglik(beta)
    for _ in range(max_iter):
        if x.shape[1] == 0:
            break
        mu = expit(x @ beta + offset)
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
    mu = expit(x @ beta + offset)
    hess = (x * (weights * mu * (1 - mu))[:, None]).T @ x
    return beta, np.linalg.pinv(hess), current


def fixed_part(x: np.ndarray, columns: np.ndarray, values: np.ndarray) -> np.ndarray:
    """x[:, columns] @ values, where 0 * -inf counts as 0."""
    if not len(values):
        return np.zeros(len(x))
    xs = x[:, columns]
    with np.errstate(invalid="ignore"):
        products = xs * values
    return np.where(xs != 0, products, 0.0).sum(axis=1)


def _pseudo_loglik(lin, y, weights) -> float:
    return float(np.sum(weights * (y * lin - np.logaddexp(0, lin))))


def regression(x, y, model: BoundModel, params=None, zero=None):
    """The logistic regression of the dyads on their change statistics: the
    MPLE, or the exact MLE of a dyad-independent model.

    The parameters in `params` (the free ones, by default) are estimated;
    the others keep their fixed values, or 0, and those in `zero` are 0
    whatever their fixed values. With curved terms, the
    statistics' coefficients are nonlinear in the parameters: the decays are
    first held at their starting values, which leaves a logistic regression,
    then all parameters are refined by Gauss-Newton steps. Returns the
    parameters, their covariance and the log-pseudo-likelihood.
    """
    params = model.free if params is None else params & model.free
    theta = np.where(model.fixed, model.fixed_values, model.initial())
    theta[~params & ~model.fixed] = 0.0
    if zero is not None:
        theta[zero] = 0.0
    # Many dyads share the same change statistics: fit on the distinct rows.
    rows, counts = np.unique(np.column_stack([x, y]), axis=0, return_counts=True)
    x, y, weights = rows[:, :-1], rows[:, -1], counts.astype(float)
    linear = params.copy()
    if model.curved:
        for position, _ in _decays_of(model):
            linear[position] = False  # held at their starting values first
    # The parameters in `linear` enter the linear predictor through the columns
    # of the Jacobian, which don't depend on them: a logistic regression.
    jac = model.jacobian(theta)
    held = theta.copy()
    held[linear] = 0.0
    offset = fixed_part(x, np.arange(model.n_stats), model.eta(held))
    beta, cov_linear, loglik = logistic_regression(x @ jac[:, linear], y, weights, offset)
    theta[linear] = beta
    cov = np.full((model.n_params, model.n_params), np.nan)
    cov[np.ix_(linear, linear)] = cov_linear
    if not model.curved or not (params & ~linear).any():
        return theta, cov, loglik
    theta, cov_params, loglik = _gauss_newton_mple(x, y, weights, model, theta, params)
    cov = np.full((model.n_params, model.n_params), np.nan)
    cov[np.ix_(params, params)] = cov_params
    return theta, cov, loglik


def _decays_of(model: BoundModel):
    from ._model import _decays

    return _decays(model.blocks, model.network)


def _maximize(value, grad_hess, theta, free, max_iter=500, tol=1e-9):
    """Maximizes `value` over the parameters in `free` by Levenberg-Marquardt
    steps: Gauss-Newton steps, damped when they don't improve, which keeps
    them sensible along nearly flat directions such as a poorly identified
    decay. `grad_hess` returns the gradient and a positive semi-definite
    approximation of minus the Hessian. Returns the parameters and whether
    they converged."""
    def safe(th):
        with np.errstate(over="ignore", invalid="ignore"):
            v = value(th)
        return v if np.isfinite(v) else -np.inf

    current, damping = safe(theta), 1e-4
    for _ in range(max_iter):
        with np.errstate(over="ignore", invalid="ignore"):
            grad, hess = grad_hess(theta)
        if np.max(np.abs(grad)) < tol * (1.0 + abs(current)):
            return theta, True
        scale = np.diag(np.maximum(np.diag(hess), 1e-12))
        while True:
            step = np.linalg.lstsq(hess + damping * scale, grad, rcond=None)[0]
            trial = theta.copy()
            trial[free] += step
            new = safe(trial)
            if new >= current:
                theta, current, damping = trial, new, max(damping / 10, 1e-12)
                break
            damping *= 10
            if damping > 1e12:
                return theta, True  # no step improves: at the optimum, to float precision
        if np.max(np.abs(step)) < tol:
            return theta, True
    return theta, False


def _gauss_newton_mple(x, y, weights, model, theta, params):
    """Maximizes the log-pseudo-likelihood of a curved model over `params`."""
    def lin_of(th):
        return fixed_part(x, np.arange(model.n_stats), model.eta(th))

    def grad_hess(th):
        mu = expit(lin_of(th))
        jx = x @ model.jacobian(th)[:, params]  # d lin / d params
        return jx.T @ (weights * (y - mu)), (jx * (weights * mu * (1 - mu))[:, None]).T @ jx

    theta, converged = _maximize(lambda th: _pseudo_loglik(lin_of(th), y, weights), grad_hess,
                                 theta, params)
    if not converged:
        warnings.warn("the MPLE of the curved model did not converge", stacklevel=4)
    _, hess = grad_hess(theta)
    return theta, np.linalg.pinv(hess), _pseudo_loglik(lin_of(theta), y, weights)


def mple(model: BoundModel) -> Estimate:
    x, y = model.mple_data()
    theta, cov, loglik = regression(x, y, model)
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
    convex hull of the sample (with a margin), as in `Hummel et al. (2012) <https://doi.org/10.1080/10618600.2012.679224>`__.

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

    Uses `Geyer's (1992) <https://doi.org/10.1214/ss/1177011137>`__ initial monotone sequence estimator on the
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
    """Split R-hat of each statistic (`Gelman et al. 2013 <https://doi.org/10.1201/b16018>`__): about 1 when the chains
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


def hotelling_pvalue(sample, observed, tau, sample_obs=None, tau_obs=None) -> float:
    """p-value of Hotelling's T^2 test that the model's mean statistics are the
    observed ones (or the mean of a conditional sample `sample_obs`), with the
    effective sample size as the sample size."""
    p = sample.shape[-1]
    n_eff = sample.shape[0] * sample.shape[1] / tau.max()
    cov = mean_covariance(sample, tau)
    target = observed
    if sample_obs is not None:
        target = sample_obs.reshape(-1, p).mean(axis=0)
        cov = cov + mean_covariance(sample_obs, tau_obs)
        n_eff = min(n_eff, sample_obs.shape[0] * sample_obs.shape[1] / tau_obs.max())
    if n_eff <= p + 1:
        return 0.0
    diff = sample.reshape(-1, p).mean(axis=0) - target
    t2 = float(diff @ np.linalg.pinv(cov) @ diff)
    return float(stats.f.sf(t2 * (n_eff - p) / (p * (n_eff - 1)), p, n_eff - p))


def _information(sample, sample_obs=None, jac=None):
    """Mean and covariance of a sample, minus the covariance of the conditional
    sample if there is one (the missing information principle, `Louis 1982 <https://doi.org/10.1111/j.2517-6161.1982.tb01203.x>`__).
    The difference must be positive definite in the parameters' space (through
    `jac`); with too few samples to tell the two apart, the plain covariance
    is used."""
    flat = sample.reshape(-1, sample.shape[-1])
    mean, cov = flat.mean(axis=0), np.atleast_2d(np.cov(flat, rowvar=False))
    if sample_obs is None:
        return mean, cov
    flat_obs = sample_obs.reshape(-1, sample_obs.shape[-1])
    info = cov - np.atleast_2d(np.cov(flat_obs, rowvar=False))
    projected = info if jac is None else jac.T @ info @ jac
    if np.linalg.eigvalsh(projected).min() <= 0:
        return mean, cov
    return mean, info


def _active(model: BoundModel, theta) -> np.ndarray:
    """Statistics with a finite coefficient: those -inf offsets don't fix."""
    return np.isfinite(model.eta(theta))


def project(model: BoundModel, theta, sample: np.ndarray) -> np.ndarray:
    """Statistics projected on the free parameters, through the Jacobian of
    eta: the estimating functions (the statistics themselves if no term is
    curved)."""
    active = _active(model, theta)
    jac = model.jacobian(theta)[np.ix_(active, model.free)]
    return sample[..., active] @ jac


def _step(model: BoundModel, theta, sample, target, margin, sample_obs=None):
    """Information, step length and new parameters from a sample, by the
    log-normal approximation of the log-likelihood ratio (`Hummel et al. 2012 <https://doi.org/10.1080/10618600.2012.679224>`__).

    The approximation, quadratic in the statistics' coefficients eta, is
    maximized towards the pseudo-observed target mean + gamma * (target -
    mean), with gamma the largest step that keeps it inside the sample's
    convex hull, in the space of the estimating functions. Without curved
    terms the maximum is a Newton step; with them, Gauss-Newton steps find it.
    """
    free, active = model.free, _active(model, theta)
    jac = model.jacobian(theta)[np.ix_(active, free)]
    sample, target = sample[..., active], target[active]
    sample_obs = None if sample_obs is None else sample_obs[..., active]
    flat = sample.reshape(-1, sample.shape[-1])
    mean, info = _information(sample, sample_obs, jac)
    gamma = hummel_steplength(flat @ jac, target @ jac, margin)
    pull = gamma * (target - mean)
    new = np.array(theta, dtype=float)
    new[free] += np.linalg.lstsq(jac.T @ info @ jac, jac.T @ pull, rcond=None)[0]
    if model.curved:
        new = _gauss_newton_lognormal(model, theta, new, pull, info, active)
    return jac.T @ info @ jac, gamma, new


def _gauss_newton_lognormal(model, theta0, theta, pull, info, active):
    """Maximizes (eta - eta0) . pull - (eta - eta0)' info (eta - eta0) / 2 over
    the free parameters, from `theta`."""
    free = model.free
    eta0 = model.eta(theta0)[active]

    def value(th):
        d = model.eta(th)[active] - eta0
        return d @ pull - 0.5 * d @ info @ d

    def grad_hess(th):
        d = model.eta(th)[active] - eta0
        jac = model.jacobian(th)[np.ix_(active, free)]
        return jac.T @ (pull - info @ d), jac.T @ info @ jac

    return _maximize(value, grad_hess, np.array(theta, dtype=float), free)[0]


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
              check_variance=True, conditional=False):
    """Sample statistics (conditional on the observed dyads with `conditional`),
    stopping with a DegeneracyError if the density guard trips or (with
    `check_variance`) a free statistic does not vary."""
    max_edges = _max_edges(model, control)
    try:
        sample, last, _ = model.simulate(
            starts, theta, burnin, interval, samples, int(rng.integers(2**63)),
            conditional=conditional, max_edges=max_edges,
            triadic_weight=model.triadic_weight(control.triadic_weight),
        )
    except _core.DensityGuardError:
        raise _density_guard_error(model, max_edges) from None
    variances = project(model, theta, sample).reshape(-1, int(model.free.sum())).var(axis=0)
    if check_variance and np.any(variances <= 0):
        constant = [n for n, v in zip(np.array(model.names)[model.free], variances) if v <= 0]
        raise DegeneracyError(
            f"the simulated statistics {constant} did not vary: the model may be degenerate "
            "at these coefficients, the MCMC too short, or the statistics constant under "
            "the constraints"
        )
    return sample, last


def _stall_message(model: BoundModel, sample: np.ndarray, iterations: int, observed=None,
                   theta=None) -> str:
    observed = model.observed() if observed is None else observed
    lines = [
        f"the Monte Carlo MLE is not making progress: for {iterations} iterations the "
        "observed statistics were far outside the range of the simulated networks (step "
        "lengths below 0.1). The model may be degenerate, or the starting coefficients poor.",
        "",
    ]
    if model.curved:
        # Many histogram counts: show the estimating functions instead.
        deviation = project(model, theta, sample).reshape(-1, int(model.free.sum())).mean(axis=0) \
            - project(model, theta, observed[None, None, :])[0, 0]
        names = list(np.array(model.names)[model.free])
        width = max(map(len, names))
        lines += ["At the current coefficients, simulated minus observed estimating functions:",
                  *(f"  {n:<{width}}  {d:12.2f}" for n, d in zip(names, deviation))]
    else:
        mean = sample.reshape(-1, model.n_stats).mean(axis=0)
        width = max(map(len, model.names))
        lines += ["At the current coefficients:",
                  f"  {'':<{width}}  {'simulated':>12}  {'observed':>12}",
                  *(f"  {n:<{width}}  {m:12.2f}  {o:12.2f}"
                    for n, m, o in zip(model.names, mean, observed))]
    rhat = split_rhat(project(model, theta, sample)) if model.curved else split_rhat(sample)
    if np.nanmax(rhat) > 1.2:
        lines += ["", f"The chains disagree (R-hat up to {np.nanmax(rhat):.1f}): the simulated "
                      "networks jump between very different regimes, a typical sign of degeneracy."]
    lines += ["", "Things to try: other terms (for example gwesp with a smaller decay instead of "
                  "triangle), adding gwdegree or attribute terms, init='CD', or a longer MCMC "
                  "(interval=...). Control(stall_iterations=None) keeps iterating."]
    return "\n".join(lines)


def contrastive_divergence(model: BoundModel, init: np.ndarray, control: Control,
                           rng: np.random.Generator) -> Estimate:
    """Contrastive divergence (`Hinton 2002 <https://doi.org/10.1162/089976602760128018>`__; `Krivitsky 2017 <https://doi.org/10.1016/j.csda.2016.10.015>`__), as ergm's CD: each
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
        projected = project(model, theta, sample)
        target = project(model, theta, observed[None, None, :])[0, 0]
        pvalue = hotelling_pvalue(projected, target, np.ones(projected.shape[-1]))
        _, gamma, theta = _step(model, theta, sample, observed, control.steplength_margin)
        log.info("CD iteration %d: step length %.2f, p-value %.3f", iteration, gamma, pvalue)
        if gamma == 1.0 and pvalue > control.cd_conv_min_pval:
            converged = True
            break
    nan = np.full((model.n_params, model.n_params), np.nan)
    return Estimate(theta, nan, None, None, "CD", iteration, converged, None)


def mcmle(model: BoundModel, init: np.ndarray, control: Control, rng: np.random.Generator) -> Estimate:
    """Monte Carlo MLE, stopping when the step length is 1 in two consecutive
    iterations (`Hummel et al. 2012 <https://doi.org/10.1080/10618600.2012.679224>`__), then refining with a larger sample.

    With missing dyads, each iteration also samples networks conditional on
    the observed dyads, whose mean statistics are the target (`Handcock and
    Gile 2010 <https://doi.org/10.1214/08-AOAS221>`__)."""
    observed = model.observed()
    free = model.free
    theta = np.asarray(init, dtype=float).copy()
    starts = [model.network.edges] * control.n_chains
    starts_obs = [model.network.edges] * control.n_chains
    interval, burnin = control.interval, control.burnin

    def simulate(samples):
        nonlocal starts, starts_obs
        sample, starts = _simulate(model, starts, theta, burnin, interval, samples, rng, control)
        if not model.has_missing:
            return sample, None
        sample_obs, starts_obs = _simulate(model, starts_obs, theta, burnin, interval, samples,
                                           rng, control, check_variance=False, conditional=True)
        return sample, sample_obs

    def effective_size(sample, sample_obs):
        projected = project(model, theta, sample)
        tau = autocorrelation_time(projected)
        ess = sample.shape[0] * sample.shape[1] / tau.max()
        if sample_obs is None:
            return projected, None, tau, None, ess
        projected_obs = project(model, theta, sample_obs)
        tau_obs = autocorrelation_time(projected_obs)
        ess = min(ess, sample_obs.shape[0] * sample_obs.shape[1] / tau_obs.max())
        return projected, projected_obs, tau, tau_obs, ess

    def target(sample_obs):
        return observed if sample_obs is None else sample_obs.reshape(-1, model.n_stats).mean(axis=0)

    converged, full_steps, stalled = False, 0, 0
    for iteration in range(1, control.max_iter + 1):
        sample, sample_obs = simulate(control.per_chain())
        *_, ess = effective_size(sample, sample_obs)
        previous = theta
        _, gamma, theta = _step(model, theta, sample, target(sample_obs), control.steplength_margin,
                                sample_obs)
        log.info("iteration %d: interval %d, effective size %.0f, step length %.2f",
                 iteration, interval, ess, gamma)
        stalled = stalled + 1 if gamma < 0.1 else 0
        if control.stall_iterations and stalled >= control.stall_iterations:
            raise DegeneracyError(_stall_message(model, sample, stalled, target(sample_obs), previous))
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

    sample, sample_obs = simulate(control.per_chain() * control.last_boost)
    projected, projected_obs, tau, tau_obs, _ = effective_size(sample, sample_obs)
    goal = target(sample_obs)
    goal_projected = project(model, theta, goal[None, None, :])[0, 0]
    pvalue = hotelling_pvalue(projected, goal_projected, tau, projected_obs, tau_obs)
    info, gamma, new_theta = _step(model, theta, sample, goal, control.steplength_margin, sample_obs)
    log.info("final iteration: interval %d, step length %.2f, p-value %.3f", interval, gamma, pvalue)
    inverse = np.linalg.pinv(info)  # inverse Fisher information of the free parameters
    error = mean_covariance(projected, tau)
    if projected_obs is not None:
        error = error + mean_covariance(projected_obs, tau_obs)
    mc_free = inverse @ error @ inverse
    cov, mc_cov = (np.full((model.n_params, model.n_params), np.nan) for _ in range(2))
    cov[np.ix_(free, free)] = inverse + mc_free
    mc_cov[np.ix_(free, free)] = mc_free
    return Estimate(new_theta, cov, mc_cov, None, "MCMLE", iteration + 1, converged, sample,
                    interval, pvalue, sample_obs=sample_obs)
