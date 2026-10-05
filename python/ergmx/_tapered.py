"""Tapered ERGMs, as R's ergm.tapered (Fellows and Handcock 2017; Blackburn
and Handcock 2023): the model's log-probabilities less a penalty
sum_k tau_k (g_k(y) - m_k)^2 that keeps the statistics near the network's
(or the target statistics), which stops the degeneracy that makes many
models unusable. Fitted by the Monte Carlo MLE of the model with ergm's
Taper() operator and the penalty's coefficient fixed at -1; the standard
errors are those of the tapered likelihood, as ergm.tapered computes them."""

from __future__ import annotations

import dataclasses

import numpy as np

from ._fit import ErgmFit, FitSummary
from .terms import Formula, as_formula


def _term_key(term) -> str:
    return repr(term)


def ergm_tapered(network, formula, *, r: float = 2.0, beta=None, tau=None, tapering_centers=None,
                 target_stats=None, taper_terms="all", **options) -> TaperedFit:
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
    """
    from . import ergm
    from ._model import bind
    from ._operators import OffsetOp, Taper
    from ._simulate import summary_stats

    if options.get("estimate", "MLE") not in ("MLE", "MPLE"):
        raise ValueError("ergm_tapered(estimate=): 'MLE' or 'MPLE'")
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
    penalty = OffsetOp(Formula([Taper(Formula(taper), coef=tau.tolist(), m=centers.tolist())]), coef=-1.0,
                       which="Taper_Penalty")
    fit = ergm(network, Formula([*trimmed, penalty]), **options)
    coefficients = np.zeros(len(names))
    coefficients[len(names) - n_taper:] = tau
    return TaperedFit._from(fit, dict(zip(names, coefficients.tolist())), dict(zip(names[len(names) - n_taper:],
                            centers.tolist())), r, tapering_centers is None)


class TaperedFit(ErgmFit):
    """A tapered ERGM's fit (:func:`ergmx.ergm_tapered`): an
    :class:`~ergmx.ErgmFit` whose standard errors are those of the tapered
    likelihood, with the tapering coefficients and centers."""

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

    def summary(self) -> FitSummary:
        return TaperedSummary(self)


class TaperedSummary(FitSummary):
    """A tapered fit's table, with its tapering."""

    def __str__(self) -> str:
        fit = self.fit
        tapered = [n for n, t in fit.tapering_coef.items() if t > 0]
        width = max(map(len, tapered))
        lines = [super().__str__(), "", f"Tapered (r = {fit.r:g}):", f"{'':<{width}}  {'Center':>10}  {'Tau':>10}"]
        lines += [f"{n:<{width}}  {fit.tapering_centers[n]:10.4g}  {fit.tapering_coef[n]:10.4g}" for n in tapered]
        return "\n".join(lines)
