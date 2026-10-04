"""Estimation: MPLE, contrastive divergence, and Monte Carlo MLE with `Hummel
et al. (2012) <https://doi.org/10.1080/10618600.2012.679224>`__ stepping."""

from __future__ import annotations

import logging
import os
import warnings
import dataclasses
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
    #: When the Monte Carlo MLE stops, as ergm's MCMLE.termination:
    #: "confidence" (ergm's default) once an equivalence test shows, with
    #: probability ``confidence``, that the simulated mean statistics are
    #: within ``mcmc_precision`` of the observed, in units of their variance
    #: (the sample growing until it can tell); "Hummel" after two
    #: consecutive full steps (step length 1), with a last, larger sample.
    termination: str = "confidence"
    #: "confidence": the level of the equivalence test (ergm's MCMLE.confidence).
    confidence: float = 0.99
    #: "confidence": the tolerance region, as a share of the statistics'
    #: variance (ergm's MCMLE.MCMC.precision).
    mcmc_precision: float = 0.05
    #: "confidence": the most the sample grows by in an iteration (ergm's
    #: MCMLE.confidence.boost), when the test can't tell yet, or the
    #: estimating functions don't approach the tolerance region in more than
    #: ``confidence_boost_threshold`` of ``confidence_boost_lag`` iterations.
    confidence_boost: float = 2.0
    confidence_boost_threshold: int = 1
    confidence_boost_lag: int = 4
    #: Margin by which the target must lie inside the sample's convex hull.
    steplength_margin: float = 0.05
    #: "Hummel": the final iteration samples this many times more networks.
    last_boost: int = 4
    #: Stop if a simulated network has more than this many times the observed
    #: edges (and more than ``density_guard_min``), as ergm does.
    density_guard: float = float(np.exp(3))
    #: Edges a simulated network may always have, whatever the density guard.
    density_guard_min: int = 10000
    #: Stop after this many consecutive iterations with a step length below 0.1,
    #: or once longer intervals stop improving chains that barely mix (see
    #: MIXING_WAITS); None never stops early.
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


def logistic_regression(x, y, weights, offset=None, max_iter=100, tol=1e-10, shift=None):
    """Weighted logistic regression by Newton-Raphson, with an optional fixed
    offset added to the linear predictor. `shift` changes the sufficient
    statistics x' (weights y) by that much (to fit target statistics).

    Returns the coefficients, their covariance and the log-likelihood.
    """
    offset = np.zeros(len(y)) if offset is None else offset
    shift = np.zeros(x.shape[1]) if shift is None else shift

    def loglik(beta):
        eta = x @ beta + offset
        return float(np.sum(weights * (y * eta - np.logaddexp(0, eta))) + beta @ shift)

    beta = np.zeros(x.shape[1])
    current = loglik(beta)
    for _ in range(max_iter):
        if x.shape[1] == 0:
            break
        mu = expit(x @ beta + offset)
        grad = x.T @ (weights * (y - mu)) + shift
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


def regression(x, y, model: BoundModel, params=None, zero=None, weights=None):
    """The logistic regression of the dyads on their change statistics: the
    MPLE, or the exact MLE of a dyad-independent model.

    The parameters in `params` (the free ones, by default) are estimated;
    the others keep their fixed values, or 0, and those in `zero` are 0
    whatever their fixed values. ``weights`` counts the dyads of each row
    (as ``BoundModel.mple_table`` gives them); without, the rows are
    counted here. With curved terms, the
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
    if weights is None:
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
    shift = None
    if model.target is not None and model.exact:
        # Fitted to target statistics: the sufficient statistics are the
        # targets' (exactly, for dyad-independent models; otherwise the MPLE,
        # a start, is the network's, as ergm's).
        delta = model.target - np.array(model.core.summary(model.network.edges))
        shift = jac[:, linear].T @ delta
    beta, cov_linear, loglik = logistic_regression(x @ jac[:, linear], y, weights, offset, shift=shift)
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


def _lower_bounds(model: BoundModel) -> np.ndarray:
    """The parameters' lower bounds: 0 for the estimated decays of curved
    terms (as ergm's ``minpar``: a negative decay gives alternating weights),
    -inf for the others."""
    lower = np.full(model.n_params, -np.inf)
    for position, _ in _decays_of(model):
        if model.free[position]:
            lower[position] = 0.0
    return lower


def _maximize(value, grad_hess, theta, free, max_iter=500, tol=1e-9, lower=None, upper=None):
    """Maximizes `value` over the parameters in `free` by Levenberg-Marquardt
    steps: Gauss-Newton steps, damped when they don't improve, which keeps
    them sensible along nearly flat directions such as a poorly identified
    decay. `grad_hess` returns the gradient and a positive semi-definite
    approximation of minus the Hessian. Parameters stay between `lower` and
    `upper` (from a start moved there). Returns the parameters and whether
    they converged."""
    lower = np.full(len(theta), -np.inf) if lower is None else lower
    upper = np.full(len(theta), np.inf) if upper is None else upper
    theta = np.clip(theta, lower, upper)

    def safe(th):
        if np.any(th < lower) or np.any(th > upper):
            return -np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            v = value(th)
        return v if np.isfinite(v) else -np.inf

    current, damping = safe(theta), 1e-4
    for _ in range(max_iter):
        with np.errstate(over="ignore", invalid="ignore"):
            grad, hess = grad_hess(theta)
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
            # Derivatives overflow far out (a decay that grows without
            # bound): stay at this point, the best found.
            return theta, False
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
                                 theta, params, lower=_lower_bounds(model))
    if not converged:
        warnings.warn("the MPLE of the curved model did not converge", stacklevel=4)
    _, hess = grad_hess(theta)
    return theta, np.linalg.pinv(hess), _pseudo_loglik(lin_of(theta), y, weights)


def mple(model: BoundModel) -> Estimate:
    x, y, w = model.mple_table()
    theta, cov, loglik = regression(x, y, model, weights=w)
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


def _step(model: BoundModel, theta, sample, target, margin, sample_obs=None, radius=None, linear=False):
    """Information, step length and new parameters from a sample, by the
    log-normal approximation of the log-likelihood ratio (`Hummel et al. 2012 <https://doi.org/10.1080/10618600.2012.679224>`__).

    The approximation, quadratic in the statistics' coefficients eta, is
    maximized towards the pseudo-observed target mean + gamma * (target -
    mean), with gamma the largest step that keeps it inside the sample's
    convex hull, in the space of the estimating functions. Without curved
    terms the maximum is a Newton step; with them, Gauss-Newton steps find it
    (or, with `linear`, the first one: Fisher scoring's step, an ascent
    direction of the log-likelihood). `radius` bounds the step's length in
    the Fisher information's metric (a trust region).
    """
    theta = np.asarray(theta, dtype=float)
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
    fisher = jac.T @ info @ jac
    if model.curved and linear:
        new = np.maximum(new, _lower_bounds(model))
    elif model.curved:
        new = _gauss_newton_lognormal(model, theta, new, pull, info, active)
        # The approximation can be nearly flat along a decay, far from the
        # sample: decays move by at most MAX_DECAY_STEP, the whole step
        # shrunk to fit.
        moved = _decay_change(model, theta, new)
        if moved > MAX_DECAY_STEP:
            new = theta + (new - theta) * (MAX_DECAY_STEP / moved)
    if radius is not None:
        # The trust region: steps of at most `radius` in the metric of the
        # Fisher information (standard errors, roughly).
        length = _step_length(fisher, free, theta, new)
        if length > radius:
            new = theta + (new - theta) * (radius / length)
    return fisher, gamma, new


#: The largest change of a decay parameter in one Monte Carlo MLE iteration.
MAX_DECAY_STEP = 1.0
#: Curved models' first trust region, in the Fisher information's metric.
RADIUS = 1.0


def _step_length(fisher: np.ndarray, free: np.ndarray, before, after) -> float:
    """The length of a step in the metric of the Fisher information of the
    free parameters: sqrt(delta' I delta)."""
    delta = (np.asarray(after) - np.asarray(before))[free]
    return float(np.sqrt(max(delta @ fisher @ delta, 0.0)))


def _free_decays(model: BoundModel) -> list[int]:
    return [position for position, _ in _decays_of(model) if model.free[position]]


def _decay_change(model: BoundModel, before, after) -> float:
    """The largest change of an estimated decay between two parameter vectors."""
    decays = _free_decays(model)
    return float(np.max(np.abs(np.asarray(after)[decays] - np.asarray(before)[decays]), initial=0.0))


def _gauss_newton_lognormal(model, theta0, theta, pull, info, active, lower=None, upper=None):
    """Maximizes (eta - eta0) . pull - (eta - eta0)' info (eta - eta0) / 2 over
    the free parameters, from `theta`, between `lower` and `upper`."""
    free = model.free
    eta0 = model.eta(theta0)[active]

    def value(th):
        d = model.eta(th)[active] - eta0
        return d @ pull - 0.5 * d @ info @ d

    def grad_hess(th):
        d = model.eta(th)[active] - eta0
        jac = model.jacobian(th)[np.ix_(active, free)]
        return jac.T @ (pull - info @ d), jac.T @ info @ jac

    lower = _lower_bounds(model) if lower is None else lower
    return _maximize(value, grad_hess, np.array(theta, dtype=float), free, lower=lower, upper=upper)[0]


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
              check_variance=True, conditional=False, max_edges=None):
    """Sample statistics (conditional on the observed dyads with `conditional`),
    stopping with a DegeneracyError if the density guard trips (at
    `max_edges`, if lower than the guard's) or (with `check_variance`) a free
    statistic does not vary."""
    guard = _max_edges(model, control)
    max_edges = guard if max_edges is None else min(guard, max_edges)
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
                   theta=None, waits=None) -> str:
    observed = model.observed() if observed is None else observed
    if waits is None:
        reason = (f"for {iterations} iterations the observed statistics were far outside the range of "
                  "the simulated networks (step lengths below 0.1). The model may be degenerate, or the "
                  "starting coefficients poor.")
    else:
        sizes = ", ".join(f"{e:.0f}" for _, e in waits)
        reason = (f"the chains barely mix, and longer intervals don't help (effective sizes {sizes} with "
                  f"{waits[0][0]:,} to {waits[-1][0]:,} proposals between samples, at the same "
                  "coefficients): the simulated networks move between very different regimes. The model "
                  "is probably near-degenerate here.")
    lines = [f"the Monte Carlo MLE is not making progress: {reason}", ""]
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
    if getattr(model, "valued", False):
        lines += ["", "Things to try: fewer or other dyad-dependent terms (nodecovar, for one, can make "
                      "a valued model degenerate), CMP for the dispersion of the values, other starting "
                      "coefficients (init=...), or a longer MCMC (interval=...). "
                      "Control(stall_iterations=None) keeps iterating."]
    else:
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


# -- ergm's "confidence" termination ------------------------------------------------


def _quadratic(x: np.ndarray, a: np.ndarray, tol: float = np.sqrt(np.finfo(float).eps)) -> tuple[float, int]:
    """x' A^+ x, with A standardized by its diagonal and its eigenvalues below
    `tol` times the largest left out (statnet's xTAx_seigen), and A's rank."""
    diag = np.diag(a)
    d = np.where(diag > 0, 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)), 0.0)
    values, vectors = np.linalg.eigh(a * np.outer(d, d))
    keep = values > max(tol * values.max(initial=0.0), 0.0)
    h = vectors[:, keep].T @ (x * d)
    return float(np.sum(h * h / values[keep])), int(keep.sum())


def _weighted_variance(x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    return np.atleast_2d(np.cov(flat, rowvar=False, aweights=None if w is None else w.ravel()))


def _tolerance(model_precision: float, e, e_obs=None, w=None, w_obs=None) -> np.ndarray:
    """The tolerance region's matrix (ergm's target_prec): the estimating
    functions' variance (less the conditional sample's) times the precision,
    without the rows and columns of those that don't vary."""
    v = _weighted_variance(e, w)
    if e_obs is not None:
        v = v - _weighted_variance(e_obs, w_obs)
    v = model_precision * v
    flat = np.diag(v) <= 0
    v[flat, :] = 0.0
    v[:, flat] = 0.0
    return v


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The weighted mean of a sample (chains x samples x columns), the
    variance of that mean, inflated by the autocorrelation of each column's
    weighted deviations (by the delta method, as ergm's vcov_wmean_ar), and
    the effective sample size."""
    n = w.size
    mw = w.mean()
    m = (x * w[..., None]).sum(axis=(0, 1)) / w.sum()
    u = w[..., None] * (x - m) / mw
    tau = autocorrelation_time(u)
    flat = u.reshape(-1, x.shape[-1])
    scale = np.sqrt(tau)
    v = np.atleast_2d(np.cov(flat, rowvar=False, bias=True)) * np.outer(scale, scale)
    v0 = np.atleast_2d(np.cov(x.reshape(-1, x.shape[-1]), rowvar=False, bias=True))
    # The inflation of the variance by the weights and the autocorrelation:
    # the ratio of the determinants, per dimension, over the dimensions that vary.
    keep = (np.diag(v0) > 0) & (np.diag(v) > 0)
    if keep.any():
        with np.errstate(all="ignore"):  # nearly singular covariances: the fallback below
            sign0, log0 = np.linalg.slogdet(v0[np.ix_(keep, keep)])
            sign1, log1 = np.linalg.slogdet(v[np.ix_(keep, keep)])
        usable = sign0 > 0 and sign1 > 0 and np.isfinite(log1 - log0)
        inflation = np.exp((log1 - log0) / keep.sum()) if usable else float(tau.max())
    else:
        inflation = 1.0
    return m, v / n, n / max(inflation, 1e-12)


def _hotelling_df(n, v=None) -> float:
    """Degrees of freedom of Hotelling's T^2 (ergm's hotelling_t2_df): one
    sample, or two with unequal variances (Krishnamoorthy and Yu 2004)."""
    if len(n) == 1:
        return n[0] - 1
    d1 = np.linalg.pinv(v[0] + v[1]) @ v[0]
    d2 = np.eye(len(d1)) - d1
    p = np.linalg.matrix_rank(v[0] + v[1])
    return (p + p**2) / ((np.trace(d1 @ d1) + np.trace(d1) ** 2) / n[0]
                         + (np.trace(d2 @ d2) + np.trace(d2) ** 2) / n[1])


def _ellipsoid_distance(y: np.ndarray, w: np.ndarray, u: np.ndarray) -> float:
    """The shortest squared Mahalanobis distance (covariance `w`) from `y`,
    inside the ellipsoid x' U^+ x = 1, to it (ergm's ellipsoid_mahalanobis)."""
    from scipy.optimize import brentq

    q = len(y)
    wu = (np.linalg.pinv(u) @ w).T  # W U^+
    e1 = float(np.max(np.real(np.linalg.eigvals(wu))))
    if e1 <= 0:
        return np.inf

    def x(lam):
        return np.linalg.lstsq(np.eye(q) + lam * wu, y, rcond=None)[0]

    def f(lam):
        return _quadratic(x(lam), u)[0] - 1.0

    lower = -(1 - 1e-9) / e1
    if f(lower) <= 0:  # y negligible: the distance along the largest eigenvector
        return 1.0 / e1
    lam = brentq(f, lower, 0.0, xtol=1e-14)
    return _quadratic(y - x(lam), w)[0]


def _confidence_test(model: BoundModel, before, after, sample, sample_obs, observed, control: Control):
    """ergm's confidence test, at the new coefficients `after` from a sample
    at `before`: the sample reweighted to `after` (importance sampling), its
    estimating functions there, and an equivalence test that their mean is
    within the tolerance region. Returns (converged, boost, distance): by how
    much to grow the sample if not converged, and the squared distance of
    the estimating functions from the origin on the tolerance region's scale."""
    from scipy import stats as st

    active = np.isfinite(model.eta(before)) & np.isfinite(model.eta(after))
    deta = model.eta(after)[active] - model.eta(before)[active]
    jac = model.jacobian(after)[np.ix_(active, model.free)]

    def reweighted(x):
        xs = x[..., active]
        varies = xs.reshape(-1, xs.shape[-1]).std(axis=0) > 0
        lw = xs[..., varies] @ deta[varies]
        w = np.exp(lw - lw.max())
        return (xs - observed[active]) @ jac, w

    e, w = reweighted(sample)
    m, v, neff = _weighted_mean(e, w)
    e_obs = w_obs = None
    if sample_obs is not None:
        e_obs, w_obs = reweighted(sample_obs)
        if np.all(e_obs.reshape(-1, e_obs.shape[-1]).std(axis=0) == 0):
            m_obs, v_obs, neff_obs, e_obs, w_obs = e_obs.reshape(-1, e_obs.shape[-1])[0], None, None, None, None
        else:
            m_obs, v_obs, neff_obs = _weighted_mean(e_obs, w_obs)
    d = (m_obs if sample_obs is not None else 0.0) - m
    tolerance = _tolerance(control.mcmc_precision, e, e_obs, w, w_obs)
    d2, _ = _quadratic(d, tolerance)
    if d2 >= 1:
        return False, 1.0, d2
    two = sample_obs is not None and v_obs is not None
    try:
        t2 = _ellipsoid_distance(d, v + (v_obs if two else 0.0), tolerance)
    except (np.linalg.LinAlgError, ValueError):
        return False, control.confidence_boost, d2
    df = _hotelling_df([neff, neff_obs], [v, v_obs]) if two else _hotelling_df([neff])
    if not np.isfinite(t2) or df <= 0:
        return False, control.confidence_boost, d2
    pvalue = st.f.sf(t2, 1, df)
    log.info("convergence test p-value: %.4f", pvalue)
    if pvalue < 1 - control.confidence:
        return True, 0.0, d2
    critical = st.f.ppf(control.confidence, 1, df)
    return False, min(critical / t2, control.confidence_boost), d2


@dataclass
class _Iterate:
    """A Monte Carlo MLE iteration's parameters, sample(s), and the states its
    chains started from."""

    theta: np.ndarray
    sample: np.ndarray
    sample_obs: np.ndarray | None
    starts: tuple
    #: The Fisher information of the free parameters its sample gave, once it stepped.
    fisher: np.ndarray | None
    #: Its sample's effective size.
    ess: float = np.inf


#: A step is undone if it loses more than this in log-likelihood (or three
#: Monte Carlo standard errors of the loss, if more).
MAX_LOSS = 1.0
#: After this many steps from a point are undone, the point itself is: back
#: to the one before (of the last HISTORY).
MAX_REJECTIONS = 5
HISTORY = 4
#: At most this many consecutive steps are undone because their new sample
#: mixed poorly (fewer than half the effective draws needed, where the last
#: had enough); then the interval grows instead.
POOR_STEPS = 3
#: The chains of a near-degenerate model barely mix however long the
#: interval: they move between very different regimes. Stop once this many
#: consecutive samples at the same coefficients, each with too few effective
#: draws to step from, gained less than MIXING_GAIN times the last one's
#: effective size although the interval grew, and the interval is at least
#: MIXING_INTERVAL times the starting one.
MIXING_WAITS = 3
MIXING_GAIN = 1.5
MIXING_INTERVAL = 16


def _loss(model: BoundModel, before: _Iterate, theta, sample, sample_obs, observed) -> float | None:
    """How much log-likelihood the step from `before` to `theta` lost, if
    significantly more than MAX_LOSS; else None.

    The change, l(theta) - l(before), is the integral of the score along the
    straight path between the two coefficient vectors eta: by the trapezoid
    rule, the difference of the coefficients times the observed statistics
    (the mean of the conditional samples, with missing dyads) minus the mean
    of the two samples' means. (The corrected trapezoid rule, with the
    samples' variances, is worse: a step into a degenerate region, whose
    sample's variance is huge, would look like a gain.) Its Monte Carlo
    standard error comes from each sample's autocorrelation. The log-normal approximation steps assume the sample
    informs the new coefficients; when it doesn't (a curved term's decay,
    far from where its sample put its ties), that check catches the step."""
    eta0, eta1 = model.eta(before.theta), model.eta(theta)
    active = np.isfinite(eta0) & np.isfinite(eta1)
    d = eta1[active] - eta0[active]

    def mean_and_variance(x):
        series = x[..., active] @ d
        tau = autocorrelation_time(series[..., None])[0]
        return series.mean(), series.var() * tau / series.size

    m0, v0 = mean_and_variance(before.sample)
    m1, v1 = mean_and_variance(sample)
    if sample_obs is None:
        t, vt = float(observed[active] @ d), 0.0
    else:
        (a, va), (b, vb) = mean_and_variance(before.sample_obs), mean_and_variance(sample_obs)
        t, vt = (a + b) / 2, (va + vb) / 4
    gain = t - (m0 + m1) / 2
    se = np.sqrt((v0 + v1) / 4 + vt)
    return -gain if gain < -max(MAX_LOSS, 3 * se) else None


def mcmle(model: BoundModel, init: np.ndarray, control: Control, rng: np.random.Generator) -> Estimate:
    """Monte Carlo MLE, as ergm's: log-normal steps of `Hummel et al. (2012)
    <https://doi.org/10.1080/10618600.2012.679224>`__ lengths, until ergm's
    confidence test (`_confidence_test`) shows the estimate is within the
    tolerance region (or, with ``termination="Hummel"``, until two
    consecutive full steps, then refined with a larger sample).

    With missing dyads, each iteration also samples networks conditional on
    the observed dyads, whose mean statistics are the target (`Handcock and
    Gile 2010 <https://doi.org/10.1214/08-AOAS221>`__).

    Curved models are fitted in two stages: first with the decays held at
    their starting values, then with the decays too, from there. Their
    approximation misleads most far from the observed statistics, where
    the first steps are.

    A trust region keeps the steps where they improve the fit: each new
    sample also estimates the log-likelihood the last step gained (`_loss`),
    and a step that lost some is replaced by one half as long, from the
    previous chains' states (in curved models, Fisher scoring's), until one
    gains; the next steps are then at most that long, in the metric of the
    Fisher information (the trust region), which grows back when steps
    reach its edge and gain. Curved models start with a trust region of
    RADIUS."""
    if control.termination not in ("confidence", "Hummel"):
        raise ValueError(f"termination must be 'confidence' or 'Hummel', not {control.termination!r}")
    decays, held_iterations = _free_decays(model), 0
    if decays:
        # Curved models' first steps, far from the observed statistics, are
        # where their approximation misleads most: first fit the other
        # parameters with the decays held at their starting values (a model
        # that isn't curved in them), then free the decays from there.
        held = np.zeros(model.n_params, dtype=bool)
        held[decays] = True
        start = np.asarray(init, dtype=float)
        stage = dataclasses.replace(model, fixed=model.fixed | held,
                                    fixed_values=np.where(held, start, model.fixed_values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = mcmle(stage, start, control, rng)
        log.info("decays held at %s: %d iterations; now estimating them too", np.round(start[decays], 3),
                 first.iterations)
        init = np.where(held, start, first.theta)
        held_iterations = first.iterations
        # The chains' interval, as long as the first stage found it needed.
        interval = first.interval or control.interval
        control = dataclasses.replace(control, interval=interval, burnin=max(control.burnin, 16 * interval))
    observed = model.observed()
    free = model.free
    theta = np.asarray(init, dtype=float).copy()
    starts = [model.network.edges] * control.n_chains
    starts_obs = [model.network.edges] * control.n_chains
    interval, burnin = control.interval, control.burnin
    samplesize, ess_target = control.samplesize, control.effective_size

    def simulate(samples, max_edges=None):
        nonlocal starts, starts_obs
        sample, starts = _simulate(model, starts, theta, burnin, interval, samples, rng, control,
                                   max_edges=max_edges)
        if not model.has_missing:
            return sample, None
        sample_obs, starts_obs = _simulate(model, starts_obs, theta, burnin, interval, samples,
                                           rng, control, check_variance=False, conditional=True,
                                           max_edges=max_edges)
        return sample, sample_obs

    observed_edges = len(model.network.edges)

    def step_guard():
        """After a step, chains stop early once they have far more ties than
        the observed network and the last sample's networks: a step into
        degeneracy, which is then undone, rather than sampled at length."""
        recent = max(len(s) for s in starts)
        return int(max(3 * observed_edges, 1.5 * recent, observed_edges + 100))

    def per_chain():
        return max(1, samplesize // control.n_chains)

    def boost(factor):
        """Grows the sample as ergm does: the target effective size by
        `factor`, the sample size by its square root."""
        nonlocal samplesize, ess_target
        if factor > 1:
            samplesize = int(np.ceil(samplesize * np.sqrt(factor)))
            if ess_target:
                ess_target = int(np.ceil(ess_target * factor))
            log.info("sample size %d, target effective size %s", samplesize, ess_target)

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

    confidence = control.termination == "confidence"
    converged, full_steps, stalled = False, 0, 0
    accepted = None  # the last iteration whose step was kept: _Iterate
    # The trust region: the longest step, in the Fisher information's metric.
    radius = RADIUS if model.curved else None
    distance, not_closer = None, []  # "confidence": the distance from the tolerance region
    poor = 0  # consecutive steps undone because their samples barely mixed
    waits = []  # (interval, effective size, coefficients) of the samples too poor to step from
    rejections, history = 0, []  # steps undone from the last point; the points before it
    for iteration in range(1, control.max_iter + 1):
        begun = (starts, starts_obs)
        stepped = accepted is not None and accepted.fisher is not None
        try:
            sample, sample_obs = simulate(per_chain(), step_guard() if stepped else None)
        except DegeneracyError as error:
            if not stepped:
                raise
            # A step into degeneracy (the density guard tripped, or the
            # statistics froze): it lost, by far.
            sample, failure = None, str(error).split(":")[0]
        if stepped:
            loss = np.inf if sample is None else _loss(model, accepted, theta, sample, sample_obs, observed)
            length = _step_length(accepted.fisher, free, accepted.theta, theta)
            if loss is None and sample is not None and control.effective_size and poor < POOR_STEPS:
                # A step from a sample that mixed into coefficients where the
                # chains barely move (near degeneracy) is not trusted either.
                ess_now = effective_size(sample, sample_obs)[-1]
                if ess_now < control.effective_size / 2 <= accepted.ess / 2:
                    loss, poor = 0.0, poor + 1
            if loss is not None:
                rejections += 1
                waits = []
                if rejections > MAX_REJECTIONS and history:
                    # Even short steps from here fail: this point, though it
                    # passed, is itself where the trouble starts. Back to the
                    # one before, and a quarter of the step that led here.
                    bad, accepted = accepted, history.pop()
                    radius = _step_length(accepted.fisher, free, accepted.theta, bad.theta) / 4
                    theta = accepted.theta + (bad.theta - accepted.theta) / 4
                    starts, starts_obs = accepted.starts
                    rejections, full_steps = 0, 0
                    log.info("iteration %d: steps from the last point keep failing; back to the one "
                             "before it", iteration)
                    continue
                # The step lost likelihood: one half as long instead, from where
                # the chains were, and the next steps at most as long. A curved
                # model's step to the maximum of its approximation may not even
                # start uphill: Fisher scoring's does.
                radius = length / 2
                if model.curved:
                    theta = _step(model, accepted.theta, accepted.sample, target(accepted.sample_obs),
                                  control.steplength_margin, accepted.sample_obs, radius, linear=True)[2]
                else:
                    theta = accepted.theta + (theta - accepted.theta) / 2
                starts, starts_obs = accepted.starts
                full_steps = 0
                if sample is None:
                    log.info("iteration %d: the step led to degeneracy (%s); a shorter one instead",
                             iteration, failure)
                elif loss == 0.0:
                    log.info("iteration %d: the step led where the chains barely mix; a shorter one instead",
                             iteration)
                else:
                    log.info("iteration %d: the step lost %.1f in log-likelihood; a shorter one instead",
                             iteration, loss)
                continue
            if radius is not None and length >= 0.9 * radius:
                radius *= 2  # a step to the trust region's edge gained: a larger region
        projected, projected_obs, *_, ess = effective_size(sample, sample_obs)
        if accepted is not None and accepted.fisher is not None and accepted.theta is not theta:
            history = (history + [accepted])[-HISTORY:]
        accepted = _Iterate(theta, sample, sample_obs, begun, None, ess)
        rejections = 0
        previous = theta
        info, gamma, new_theta = _step(model, theta, sample, target(sample_obs), control.steplength_margin,
                                       sample_obs, radius)
        stalled = stalled + 1 if gamma < 0.1 else 0
        if control.stall_iterations and stalled >= control.stall_iterations:
            raise DegeneracyError(_stall_message(model, sample, stalled, target(sample_obs), previous))
        if (gamma >= 0.1 and control.effective_size and ess < control.effective_size
                and interval < control.max_interval):
            # Too few effective samples to step from, unless the observed
            # statistics are far outside them (ergm samples until it has
            # enough): sample again, at the same coefficients, with a longer interval.
            waits = [w for w in waits if np.array_equal(w[2], theta)] + [(interval, ess, theta.copy())]
            recent = waits[-MIXING_WAITS:]
            if (control.stall_iterations and len(recent) == MIXING_WAITS
                    and interval >= MIXING_INTERVAL * control.interval
                    and all(b[1] < MIXING_GAIN * a[1] for a, b in zip(recent, recent[1:]))):
                raise DegeneracyError(_stall_message(model, sample, iteration, target(sample_obs), theta,
                                                     waits=[(i, e) for i, e, _ in recent]))
            interval = min(control.max_interval,
                           interval * min(4, 1 << int(np.ceil(np.log2(control.effective_size / ess)))))
            burnin = max(burnin, 16 * interval)
            log.info("iteration %d: effective size %.0f, too small to step from; interval %d",
                     iteration, ess, interval)
            continue
        theta = new_theta
        accepted.fisher = info
        poor = 0
        log.info("iteration %d: interval %d, effective size %.0f, step length %.2f",
                 iteration, interval, ess, gamma)
        if confidence:
            # How far the estimating functions are from 0, on the tolerance region's scale.
            goal = project(model, previous, target(sample_obs)[None, None, :])[0, 0]
            d = (goal if projected_obs is None else projected_obs.reshape(-1, goal.size).mean(axis=0)) \
                - projected.reshape(-1, goal.size).mean(axis=0)
            d2, _ = _quadratic(d, _tolerance(control.mcmc_precision, projected, projected_obs))
            if distance is not None:
                not_closer = (not_closer + [d2 >= distance])[-control.confidence_boost_lag:]
            distance = d2
            after = np.inf
            if d2 < 2:
                done, factor, after = _confidence_test(model, previous, theta, sample, sample_obs,
                                                       observed, control)
                if done:
                    converged = True
                    break
                boost(factor)
            if (d2 >= 2 or after > 1) and sum(not_closer) > control.confidence_boost_threshold:
                boost(control.confidence_boost)
                not_closer = []
        else:
            # The convex hull test means little with few effective samples.
            enough = not ess_target or ess >= ess_target
            full_steps = full_steps + 1 if gamma == 1.0 and enough else 0
            if full_steps == 2:
                converged = True
                break
        if ess_target and ess < ess_target and interval < control.max_interval:
            factor = min(4, 1 << int(np.ceil(np.log2(ess_target / ess))))
            interval = min(control.max_interval, interval * factor)
            burnin = max(burnin, 16 * interval)
        elif ess_target and ess > 8 * ess_target and interval > control.interval:
            # Far more effective samples than needed (the chains mix again,
            # away from a hard region): a shorter interval.
            interval = max(control.interval, interval // 2)
            burnin = max(control.burnin, 16 * interval)
    else:
        warnings.warn(
            f"the Monte Carlo MLE did not converge in {control.max_iter} iterations",
            stacklevel=3,
        )

    if converged and confidence:
        # The estimate is the converged step's, from its sample.
        theta_sample, new_theta, iterations = previous, theta, iteration
        projected = project(model, theta_sample, sample)
        tau = autocorrelation_time(projected)
        projected_obs = tau_obs = None
        if sample_obs is not None:
            projected_obs = project(model, theta_sample, sample_obs)
            tau_obs = autocorrelation_time(projected_obs)
    else:
        sample, sample_obs = simulate(per_chain() * control.last_boost)
        projected, projected_obs, tau, tau_obs, _ = effective_size(sample, sample_obs)
        theta_sample = theta
        info, gamma, new_theta = _step(model, theta, sample, target(sample_obs), control.steplength_margin,
                                       sample_obs, radius)
        iterations = iteration + 1
        log.info("final iteration: interval %d, step length %.2f", interval, gamma)
    goal = target(sample_obs)
    goal_projected = project(model, theta_sample, goal[None, None, :])[0, 0]
    pvalue = hotelling_pvalue(projected, goal_projected, tau, projected_obs, tau_obs)
    log.info("p-value of the observed statistics against the last sample's: %.3f", pvalue)
    inverse = np.linalg.pinv(info)  # inverse Fisher information of the free parameters
    error = mean_covariance(projected, tau)
    if projected_obs is not None:
        error = error + mean_covariance(projected_obs, tau_obs)
    mc_free = inverse @ error @ inverse
    cov, mc_cov = (np.full((model.n_params, model.n_params), np.nan) for _ in range(2))
    cov[np.ix_(free, free)] = inverse + mc_free
    mc_cov[np.ix_(free, free)] = mc_free
    return Estimate(new_theta, cov, mc_cov, None, "MCMLE", held_iterations + iterations, converged, sample,
                    interval, pvalue, sample_obs=sample_obs)
