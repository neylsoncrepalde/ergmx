"""Tapered ERGMs, as R's ergm.tapered (Fellows and Handcock 2017; Blackburn
and Handcock 2023): the model's log-probabilities less a penalty
sum_k tau_k (g_k(y) - m_k)^2 that keeps the statistics near the network's
(or the target statistics), which stops the degeneracy that makes many
models unusable. Fitted by the Monte Carlo MLE of the model with ergm's
Taper() operator and the penalty's coefficient fixed at -1; the standard
errors are those of the tapered likelihood, as ergm.tapered computes them.
With ``fixed=False``, the coefficient's strength is estimated, by
ergm.tapered's kurtosis-penalized profile iterations."""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings

import numpy as np

from ._fit import ErgmFit, FitSummary
from .terms import Formula, as_formula

log = logging.getLogger("ergmx")
_EPS = float(np.finfo(float).eps)


@dataclasses.dataclass
class TaperingControl:
    """The estimated tapering's settings (:func:`ergmx.ergm_tapered` with
    ``fixed=False``), as ergm.tapered's ``control.ergm.tapered()``: the
    strength's start, the interval it is estimated in, the profile
    iterations, and the objective's kurtosis penalty."""

    #: The strength the first fit uses (MCMLE.tapering.init).
    init: float = 1.0
    #: The interval the strength is estimated in (MCMLE.tapering.interval).
    interval: tuple[float, float] = (1 / 3 + 1e-4, 3 - 1e-4)
    #: Most profile iterations, each a fit (MCMLE.tapering.maxit).
    maxit: int = 5
    #: The iterations stop once the strength the last fit's sample proposes
    #: is within tol (1 + strength) of its own (MCMLE.tapering.tol).
    tol: float = 0.01
    #: Each iteration moves this share of the way to the proposed strength
    #: (MCMLE.tapering.steplength).
    steplength: float = 1.0
    #: The objective's penalty of the statistics' kurtosis: its target and
    #: scale (MCMLE.kurtosis.location, MCMLE.kurtosis.scale).
    kurtosis_location: float = 3.0
    kurtosis_scale: float = 0.3
    #: The objective's term in the strength (MCMLE.kurtosis.penalty).
    kurtosis_penalty: float = 2.0
    #: The weight of the log-likelihood's variance term (MCMLE.varweight).
    varweight: float = 0.5

    def __post_init__(self):
        lower, upper = self.interval
        if not (0 < lower < upper < math.inf):
            raise ValueError("TaperingControl(interval=): two positive, finite, increasing values")
        if not lower <= self.init <= upper:
            raise ValueError("TaperingControl(init=) must be in the interval")
        if self.maxit < 1 or self.tol < 0 or not 0 < self.steplength <= 1 or self.varweight < 0:
            raise ValueError("TaperingControl: maxit >= 1, tol >= 0, 0 < steplength <= 1, varweight >= 0")


def _brent_min(f, ax: float, bx: float, tol: float = _EPS ** 0.25) -> float:
    """The minimum of f on [ax, bx] by Brent's method, as R's optimize()
    (its Brent_fmin), so that the strength is ergm.tapered's from the same
    sample."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(_EPS)
    a, b = ax, bx
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fv = fw = fx = f(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            return x
        p = q = r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r, e = e, d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):  # a golden-section step
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:  # a parabolic step, not too close to the ends
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0.0 else x - tol1)
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu


def _kurtosis_penalty(x: np.ndarray, weights, control: TaperingControl) -> float:
    """ergm.tapered's penalty of the statistics' kurtosis (their moments
    about the network's), its mean over the statistics; with importance
    weights, or equal ones."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if weights is None:
            m2, m4 = np.mean(x**2, axis=0), np.mean(x**4, axis=0)
        else:
            m2, m4 = weights @ x**2, weights @ x**4
        logm2, logm4 = np.log(m2), np.log(m4)
        logm2[~np.isfinite(logm2)] = 0.0
        logm4[~np.isfinite(logm4)] = 0.0
        kurt = np.exp(logm4 - 2 * logm2)
    kurt[np.isnan(kurt)] = 3.0
    return float(np.mean(-0.5 * ((kurt - control.kurtosis_location) / control.kurtosis_scale) ** 2))


def strength_objective(x: np.ndarray, strength: float, strength0: float, control: TaperingControl) -> float:
    """ergm.tapered's estimated-tapering objective (its llik.fun.Kpenalty)
    at the strength `strength`, from a sample at `strength0`: `x` the sampled
    statistics relative to the network's, the last column the penalty's
    (ergm.tapered's sign: minus the sum of the tapering terms). The
    log-likelihood ratio's log-normal approximation, plus the change in the
    kurtosis penalty (unless both strengths are 3 or more), plus
    kurtosis_penalty times the change in strength."""
    base = x[:, -1] * (strength - strength0)
    mean = base.mean()
    llr = -mean - control.varweight * np.mean((base - mean) ** 2)
    if not np.isfinite(llr) or llr < -200:
        llr = -200.0
    if strength < 3 - 0.001 or strength0 < 3 - 0.001:
        rest = x[:, :-1]
        top = base.max()
        weights = np.exp(base - (top + np.log(np.sum(np.exp(base - top)))))
        llr += _kurtosis_penalty(rest, weights, control) - _kurtosis_penalty(rest, None, control)
    llr += control.kurtosis_penalty * (strength - strength0)
    return float(llr) if np.isfinite(llr) else -800.0


def propose_strength(x: np.ndarray, strength0: float, control: TaperingControl) -> tuple[float, float, float]:
    """The strength maximizing the objective on the sample `x` (see
    :func:`strength_objective`), the objective there, and the effective size
    of the sample's importance weights at it (ergm.tapered's
    .estimate.tapering.strength)."""
    strength = _brent_min(lambda s: -strength_objective(x, s, strength0, control), *control.interval)
    logw = x[:, -1] * (strength - strength0)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return strength, strength_objective(x, strength, strength0, control), float(1 / np.sum(w**2))


def _term_key(term) -> str:
    return repr(term)


def ergm_tapered(network, formula, *, r: float = 2.0, beta=None, tau=None, tapering_centers=None,
                 target_stats=None, taper_terms="all", fixed: bool = True,
                 tapering_control: TaperingControl | None = None, **options) -> TaperedFit:
    """Fit a tapered ERGM, as R's ``ergm.tapered()``.

    Parameters
    ----------
    network, formula
        As :func:`ergmx.ergm`; ``options`` (``constraints``, ``offset_coef``,
        ``estimate="MLE"`` or ``"MPLE"``, ``seed``, ``eval_loglik``, the
        MCMC settings...) are passed to it.
    r : float
        The tapering's scale: each statistic's coefficient is
        ``1 / (r**2 * max(1, |g_k|))``, so that the penalty is about 1 when
        a statistic is ``r`` standard deviations (of a Poisson count) from its
        center. Smaller is stronger.
    beta, tau
        The coefficients directly: ``tau`` (one per tapered statistic, or one
        for all), or ``beta`` with ``tau = 1 / beta**2``.
    tapering_centers : dict, optional
        The centers, by statistic name (the network's statistics, or the
        target statistics, by default). With other centers, the standard
        errors are the Monte Carlo MLE's, as in ergm.tapered.
    target_stats : list, optional
        Fit to these statistics (from a network simulated towards them), which
        are also the centers.
    taper_terms : {"all", "dependent"} or formula
        The terms to taper: all of them, the dyad-dependent ones, or those of
        a formula (terms of ``formula``).
    fixed : bool
        Whether the tapering is the one given (by ``r``, ``beta`` or
        ``tau``). With False, its strength, a multiplier of the
        coefficients, is estimated as ergm.tapered's: each fit's sample
        proposes the strength that maximizes a kurtosis-penalized
        log-likelihood ratio, for the next fit, until the proposed strength
        is the fit's own. The coefficients are then the given ones times the
        strength, and ``r`` the given one over its square root.
    tapering_control : TaperingControl, optional
        With ``fixed=False``, the estimation's settings.
    """
    from . import ergm
    from ._model import bind
    from ._operators import OffsetOp, Taper
    from ._simulate import summary_stats

    if options.get("estimate", "MLE") not in ("MLE", "MPLE"):
        raise ValueError("ergm_tapered(estimate=): 'MLE' or 'MPLE'")
    if not fixed and options.get("estimate", "MLE") != "MLE":
        raise ValueError("estimated tapering (fixed=False) needs estimate='MLE', as in ergm.tapered")
    if fixed and tapering_control is not None:
        raise ValueError("tapering_control= is for estimated tapering (fixed=False)")
    terms = list(as_formula(formula))
    bound = bind(network, Formula(terms), bipartite=options.get("bipartite"))
    if isinstance(taper_terms, str) and taper_terms == "all":
        tapered = list(range(len(terms)))
    elif isinstance(taper_terms, str) and taper_terms == "dependent":
        tapered = [k for k, t in enumerate(terms) if not t.dyad_independent]
    else:
        wanted = {_term_key(t) for t in as_formula(taper_terms)}
        tapered = [k for k, t in enumerate(terms) if _term_key(t) in wanted]
        if len(tapered) != len(wanted):
            raise ValueError(f"taper_terms: {sorted(wanted - {_term_key(terms[k]) for k in tapered})} are not terms "
                             "of the formula")
    if not tapered:
        raise ValueError("ergm_tapered(): no terms to taper")
    trimmed = [t for k, t in enumerate(terms) if k not in tapered]
    taper = [terms[k] for k in tapered]
    # The statistics of the model, the untapered terms' first, and their centers.
    names = [n for t in [*trimmed, *taper] for n in t.names(bound.network)]
    n_taper = sum(len(t.names(bound.network)) for t in taper)
    if target_stats is not None:
        if len(np.atleast_1d(target_stats)) != len(bound.stat_names):
            raise ValueError(f"target_stats: {len(bound.stat_names)} values, one per statistic")
        by_name = dict(zip(bound.stat_names, np.atleast_1d(np.asarray(target_stats, dtype=float))))
        target = np.array([by_name[n] for n in names])
        centers = target[len(names) - n_taper:]
        options["target_stats"] = target.tolist() + [np.nan]  # the penalty's has no target
    else:
        target = None
        centers = np.array(list(summary_stats(bound.network, Formula(taper)).values()))
    if tapering_centers is not None:
        missing = [n for n in names[len(names) - n_taper:] if n not in tapering_centers]
        if missing:
            raise ValueError(f"tapering_centers: a center for each tapered statistic; missing {missing}")
        centers = np.array([float(tapering_centers[n]) for n in names[len(names) - n_taper:]])
    if tau is not None:
        tau = np.resize(np.asarray(tau, dtype=float), n_taper)
    elif beta is not None:
        tau = np.resize(1 / np.asarray(beta, dtype=float) ** 2, n_taper)
    else:
        tau = 1 / (r**2 * np.maximum(1.0, np.abs(centers)))
    def fit_at(strength, **more):
        penalty = OffsetOp(Formula([Taper(Formula(taper), coef=tau.tolist(), m=centers.tolist())]),
                           coef=-strength, which="Taper_Penalty")
        try:
            return ergm(network, Formula([*trimmed, penalty]), **{**options, **more})
        except DegeneracyError as error:
            raise DegeneracyError(f"{error}\n\nWith a tapered model, a stronger tapering (a smaller r, such as "
                                  "r=1, or larger tau) usually removes the degeneracy.") from None

    from ._estimation import DegeneracyError

    if fixed:
        fit, strength, estimated = fit_at(1.0), 1.0, None
    else:
        fit, strength, estimated = _estimate_strength(fit_at, tapering_control or TaperingControl(), options)
    coefficients = np.zeros(len(names))
    coefficients[len(names) - n_taper:] = tau * strength
    out = TaperedFit._from(fit, dict(zip(names, coefficients.tolist())), dict(zip(names[len(names) - n_taper:],
                           centers.tolist())), r / math.sqrt(strength), tapering_centers is None)
    if estimated is not None:
        out.tapering_strength, (out.tapering_history, out.tapering_converged) = strength, estimated
    return out


def _estimate_strength(fit_at, control: TaperingControl, options: dict):
    """ergm.tapered's estimated tapering: fits at a strength, each from the
    last one's estimate, the next strength proposed by the last fit's sample
    (moving `steplength` of the way to it, within the interval), until the
    proposal is the fit's own strength. A fit that stops for degeneracy
    (which ergm.tapered's don't detect, giving a sample from chains that
    hadn't mixed) moves the strength halfway to the interval's top, the
    strongest tapering. The log-likelihood is that of the last fit."""
    from ._estimation import DegeneracyError

    eval_loglik = options.get("eval_loglik", True)
    lower, upper = control.interval
    strength, init, history, converged, fit, failure = control.init, options.get("init"), [], False, None, None
    fitted = strength
    for outer in range(1, control.maxit + 1):
        try:
            fit = fit_at(strength, init=init, eval_loglik=False)
        except DegeneracyError as error:
            if strength >= upper:
                raise
            failure = error
            history.append({"iteration": outer, "strength": strength, "proposed": math.nan, "objective": math.nan,
                            "effective_size": math.nan})
            log.info("estimated tapering, iteration %d: the fit at strength %.4g is degenerate; %.4g instead",
                     outer, strength, (strength + upper) / 2)
            strength = (strength + upper) / 2
            continue
        fitted, model, est = strength, fit._model, fit._estimate
        # The sample relative to the network's statistics (to the fitted
        # network's, simulated towards target statistics), as ergm's.
        observed = np.array(model.core.summary(model.network.edges))
        x = np.asarray(est.sample, dtype=float).reshape(-1, model.n_stats) - observed[None, :]
        x[:, -1] = -x[:, -1]  # ergm.tapered's penalty statistic: minus the tapering terms' sum
        proposed, objective, size = propose_strength(x, strength, control)
        history.append({"iteration": outer, "strength": strength, "proposed": proposed, "objective": objective,
                        "effective_size": size})
        log.info("estimated tapering, iteration %d: strength %.4g, proposed %.4g", outer, strength, proposed)
        if abs(proposed - strength) <= control.tol * (1 + abs(strength)):
            converged = True
            break
        init = est.theta
        strength = min(max(strength + control.steplength * (proposed - strength), lower), upper)
    if fit is None:
        raise failure
    strength = fitted
    if not converged:
        warnings.warn(f"the estimated tapering did not converge after {control.maxit} iterations", stacklevel=3)
    if eval_loglik and fit.method == "MCMLE":
        from ._loglik import bridge_loglik

        model, est = fit._model, fit._estimate
        loglik, se, relative = bridge_loglik(model, est.theta, fit.control, est.interval,
                                             np.random.default_rng(options.get("seed")))
        fit._estimate = dataclasses.replace(est, loglik=loglik, loglik_se=se, loglik_relative=relative)
    return fit, strength, (history, converged)


class TaperedFit(ErgmFit):
    """A tapered ERGM's fit (:func:`ergmx.ergm_tapered`): an
    :class:`~ergmx.ErgmFit` whose standard errors are those of the tapered
    likelihood, with the tapering coefficients and centers."""

    #: With estimated tapering (fixed=False): the strength, a multiplier of
    #: the given coefficients (None if fixed).
    tapering_strength: float | None = None
    #: With estimated tapering: each iteration's strength, the strength its
    #: sample proposed, the objective there and the effective size of the
    #: importance weights.
    tapering_history: list | None = None
    #: With estimated tapering: whether the proposed strength reached the
    #: fit's own.
    tapering_converged: bool | None = None

    @classmethod
    def _from(cls, fit: ErgmFit, tau: dict, centers: dict, r: float, correct: bool) -> TaperedFit:
        out = cls(fit._model, fit._estimate, fit._mple, fit.control, fit._seed)
        out.tapering_coef, out.tapering_centers, out.r = tau, centers, r
        if correct and fit.method == "MCMLE" and fit._estimate.sample is not None:
            out._estimate = dataclasses.replace(fit._estimate, cov=out._tapered_cov())
        return out

    def _tapered_cov(self) -> np.ndarray:
        """ergm.tapered's covariance: of the mean-value parameters' derivative
        (I - 2 Sigma diag(tau))^-1 Sigma, with Sigma the sampled statistics'
        covariance, mapped to the parameters (all of them, decays too, where
        ergm.tapered maps only the curved terms' linear parameters), plus the
        MCMC error."""
        model, est = self._model, self._estimate
        p = len(self.tapering_coef)
        sample = np.asarray(est.sample).reshape(-1, est.sample.shape[-1])[:, :p]
        sigma = np.cov(sample, rowvar=False).reshape(p, p)
        tau = np.array(list(self.tapering_coef.values()))
        dmu = np.linalg.pinv(np.eye(p) - sigma * (2 * tau)[None, :]) @ sigma
        hess = -dmu - dmu.T @ np.diag(2 * tau) @ dmu
        free = model.free
        jac = model.jacobian(est.theta)[:p][:, free]
        cov_free = -np.linalg.pinv(jac.T @ hess @ jac)
        if np.mean(np.diag(cov_free) < 0) > 0.5:  # as ergm.tapered
            cov_free = -cov_free
        cov = np.full_like(est.cov, np.nan)
        cov[np.ix_(free, free)] = cov_free + (est.mc_cov[np.ix_(free, free)] if est.mc_cov is not None else 0)
        return cov

    @property
    def df(self) -> int:
        """Number of estimated coefficients, the tapering's strength included
        when estimated (as ergm.tapered's)."""
        return super().df + (self.tapering_strength is not None)

    def summary(self) -> FitSummary:
        return TaperedSummary(self)


class TaperedSummary(FitSummary):
    """A tapered fit's table, with its tapering."""

    def __str__(self) -> str:
        fit = self.fit
        tapered = [n for n, t in fit.tapering_coef.items() if t > 0]
        width = max(map(len, tapered))
        if fit.tapering_strength is None:
            title = f"Tapered (r = {fit.r:g}):"
        else:
            title = (f"Tapered, the strength estimated ({fit.tapering_strength:.5g}"
                     f"{'' if fit.tapering_converged else ', not converged'}; r = {fit.r:.4g}):")
        lines = [super().__str__(), "", title, f"{'':<{width}}  {'Center':>10}  {'Tau':>10}"]
        lines += [f"{n:<{width}}  {fit.tapering_centers[n]:10.4g}  {fit.tapering_coef[n]:10.4g}" for n in tapered]
        return "\n".join(lines)
