"""Bergm's tools for model choice and fast posteriors (Bouranis, Friel and
Maire 2017, 2018): the adjusted pseudo-likelihood (``ergm_apl``, Bergm's
``ergmAPL()``), the model evidence from it (``evidence``, Bergm's
``evidence()``, by Chib and Jeliazkov's method or power posteriors), and the
calibrated pseudo-posterior (``bergmC``)."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import multivariate_normal

from ._estimation import Control, logistic_regression
from ._model import bind


def _prior(prior_mean, prior_sigma, p: int) -> tuple[np.ndarray, np.ndarray]:
    """The normal prior, Bergm's default N(0, 100 I)."""
    mean = np.zeros(p) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    if prior_sigma is None:
        sigma = 100 * np.eye(p)
    else:
        sigma = np.asarray(prior_sigma, dtype=float)
        sigma = sigma * np.eye(p) if sigma.ndim == 0 else np.diag(sigma) if sigma.ndim == 1 else sigma
    if mean.shape != (p,) or sigma.shape != (p, p):
        raise ValueError(f"the prior needs {p} means and a {p} x {p} covariance")
    return mean, sigma


def _upper_cholesky(a: np.ndarray) -> np.ndarray:
    """R's chol(): the upper triangular U with U'U = a, pivoted (with the
    pivots undone) if a is only positive semidefinite, as Bergm does."""
    try:
        return cholesky(a, lower=False)
    except np.linalg.LinAlgError:
        from scipy.linalg.lapack import dpstrf

        u, pivots, rank, _ = dpstrf(a, lower=0)
        u = np.triu(u)
        u[rank:, rank:] = 0.0
        order = np.argsort(pivots - 1)
        return u[:, order]


class _Pseudo:
    """The log pseudo-likelihood of the model's dyads, from the MPLE's table."""

    def __init__(self, model):
        self.x, self.y, self.w = model.mple_table()

    def loglik(self, eta: np.ndarray) -> np.ndarray:
        """At each row of `eta` (coefficients x ...), vectorized."""
        lin = np.atleast_2d(eta) @ self.x.T
        return (self.w * (self.y * lin - np.logaddexp(0, lin))).sum(axis=-1)

    def score(self, eta: np.ndarray) -> np.ndarray:
        return self.x.T @ (self.w * (self.y - expit(self.x @ eta)))

    def hessian(self, eta: np.ndarray) -> np.ndarray:
        p = expit(self.x @ eta)
        return -(self.x * (self.w * p * (1 - p))[:, None]).T @ self.x


@dataclass
class AdjustedPL:
    """The adjusted pseudo-likelihood of a model (Bergm's ``ergmAPL()``): the
    pseudo-likelihood, moved to the MLE and stretched to the likelihood's
    curvature there, and scaled to its value. ``loglik(theta)`` is
    log C + log PL(W (theta - theta_mle) + theta_pl)."""

    names: list[str]
    theta_mle: np.ndarray
    theta_pl: np.ndarray
    W: np.ndarray  # noqa: N815 (Bergm's name)
    #: The log-likelihood at the MLE, and log C, its difference from the
    #: pseudo-likelihood's maximum.
    loglik_mle: float
    log_c: float
    _pseudo: _Pseudo

    def transform(self, theta: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(theta) - self.theta_mle) @ self.W.T + self.theta_pl

    def loglik(self, theta, scaled: bool = True) -> np.ndarray | float:
        """The adjusted log pseudo-likelihood at `theta` (rows of coefficients);
        without the log C of `scaled`, the pseudo-likelihood's own scale."""
        value = self._pseudo.loglik(self.transform(theta)) + (self.log_c if scaled else 0.0)
        return value if np.ndim(theta) > 1 else float(value[0])

    def score(self, theta: np.ndarray) -> np.ndarray:
        """The gradient of the log adjusted pseudo-likelihood at each row of `theta`."""
        out = []
        for chunk in np.array_split(np.atleast_2d(theta), max(1, len(np.atleast_2d(theta)) // 2000)):
            lin = self.transform(chunk) @ self._pseudo.x.T
            residual = self._pseudo.w * (self._pseudo.y - expit(lin))
            out.append(residual @ self._pseudo.x @ self.W)
        return np.vstack(out)

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        return self.W.T @ self._pseudo.hessian(self.transform(theta)[0]) @ self.W


def _bind_plain(network, formula, constraints, bipartite):
    model = bind(network, formula, constraints, bipartite=bipartite)
    if model.curved or model.fixed.any() or model.constraints.dyad_dependent or model.has_missing:
        raise ValueError("the adjusted pseudo-likelihood needs a model without curved or offset terms, "
                         "dyad-dependent constraints or missing dyads")
    return model


def ergm_apl(network, formula, *, constraints=None, bipartite=None, seed=None, fit=None,
             control: Control | None = None) -> AdjustedPL:
    """The adjusted pseudo-likelihood of a model, as Bergm's ``ergmAPL()``
    (`Bouranis, Friel and Maire 2018 <https://doi.org/10.1080/10618600.2018.1448832>`__).

    The pseudo-likelihood is moved from its maximum, the MPLE, to the MLE,
    rescaled by W so that its curvature there is the likelihood's (the
    covariance of the model's statistics at the MLE, W = U_PL^-1 U, from
    the Cholesky factors of the two curvatures), and shifted by log C to
    the likelihood's value at the MLE. The MLE, the covariance and the
    log-likelihood come from :func:`ergmx.ergm` (or ``fit``), whose
    Monte Carlo MLE ends with a large sample at the MLE and estimates the
    log-likelihood by bridge sampling (Bergm estimates the covariance from
    50 networks and the log-likelihood by a ladder of importance samples).
    """
    from ._simulate import ergm

    model = _bind_plain(network, formula, constraints, bipartite)
    if fit is None:
        fit = ergm(network, formula, constraints=constraints, bipartite=bipartite, seed=seed,
                   control=control or Control())
    if fit.loglik is None or getattr(fit, "_estimate", None) is None:
        raise ValueError("the adjusted pseudo-likelihood needs a fit with its log-likelihood")
    pseudo = _Pseudo(model)
    theta_pl = logistic_regression(pseudo.x, pseudo.y, pseudo.w)[0]
    theta_mle = np.asarray(fit.params, dtype=float)
    if not (np.all(np.isfinite(theta_mle)) and np.all(np.isfinite(theta_pl))):
        raise ValueError("the MLE or the MPLE is infinite: change the terms with infinite coefficients")
    if fit.method == "MLE":  # exact (dyad-independent): the likelihood is the pseudo-likelihood
        information = -pseudo.hessian(theta_mle)
    else:
        sample = fit._estimate.sample.reshape(-1, model.n_stats)
        information = np.atleast_2d(np.cov(sample, rowvar=False))
    w = solve_triangular(_upper_cholesky(-pseudo.hessian(theta_pl)), _upper_cholesky(information))
    log_c = float(fit.loglik - pseudo.loglik(theta_pl)[0])
    return AdjustedPL(list(model.names), theta_mle, theta_pl, w, float(fit.loglik), log_c, pseudo)


# -- Random-walk Metropolis ---------------------------------------------------------------------


def _metropolis(log_density, start, cov, iters: int, burn_in: int, rng) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis with normal steps of covariance `cov`, as
    MCMCpack's MCMCmetrop1R: `iters` draws after `burn_in`."""
    p = len(start)
    steps = rng.multivariate_normal(np.zeros(p), cov, size=burn_in + iters)
    uniforms = np.log(rng.uniform(size=burn_in + iters))
    theta, current = np.asarray(start, dtype=float).copy(), log_density(start)
    draws, accepted = np.empty((iters, p)), 0
    for k in range(burn_in + iters):
        proposal = theta + steps[k]
        value = log_density(proposal)
        if uniforms[k] < value - current:
            theta, current = proposal, value
            accepted += k >= burn_in
        if k >= burn_in:
            draws[k - burn_in] = theta
    return draws, accepted / iters


# -- Model evidence -----------------------------------------------------------------------------


@dataclass
class ModelEvidence:
    """The model evidence (marginal likelihood) of :func:`ergmx.evidence`,
    with the posterior draws of the adjusted pseudo-likelihood."""

    method: str
    log_evidence: float
    names: list[str]
    draws: np.ndarray
    acceptance_rate: float
    apl: AdjustedPL
    seconds: float

    @property
    def coef(self) -> dict[str, float]:
        """The posterior means."""
        return dict(zip(self.names, self.draws.mean(axis=0).tolist()))

    def __repr__(self) -> str:
        return (f"Model evidence ({'power posteriors' if self.method == 'PP' else 'Chib and Jeliazkov'}, from "
                f"the adjusted pseudo-likelihood): log evidence {self.log_evidence:.4f}")


def evidence(network, formula, *, method: str = "CJ", prior_mean=None, prior_sigma=None, main_iters: int | None = None,
             burn_in: int = 5000, v_proposal: float = 1.5, num_samples: int = 25000, temps=None,
             constraints=None, bipartite=None, seed=None, apl: AdjustedPL | None = None) -> ModelEvidence:
    """The model evidence, log p(y), as Bergm's ``evidence()``
    (`Bouranis, Friel and Maire 2018 <https://doi.org/10.1080/10618600.2018.1448832>`__): of the
    adjusted pseudo-likelihood (:func:`ergm_apl`) with a normal prior, by
    Chib and Jeliazkov's method (``"CJ"``) or power posteriors (``"PP"``).
    Compare models by their evidence: their Bayes factor is exp of the
    difference.

    Parameters
    ----------
    method : {"CJ", "PP"}
        ``"CJ"``: the posterior's ordinate at its mean, from a random-walk
        Metropolis sample of the adjusted posterior (``main_iters`` draws,
        30000 by default, of which the first ``burn_in`` are dropped, as
        Bergm's) and ``num_samples`` draws of the proposal. ``"PP"``: the
        integral over temperatures (``temps``, by default 50 from 0 to 1,
        to the fifth power) of the tempered posteriors' expected log
        adjusted pseudo-likelihood, with control variates (main_iters,
        20000 by default, each).
    prior_mean, prior_sigma
        The normal prior, N(0, 100 I) by default.
    v_proposal
        The proposals' scale: their covariance is v_proposal^2 times the
        inverse of the adjusted posterior's curvature.
    """
    started = time.time()
    if method not in ("CJ", "PP"):
        raise ValueError(f"method must be 'CJ' or 'PP', not {method!r}")
    if apl is None:
        apl = ergm_apl(network, formula, constraints=constraints, bipartite=bipartite, seed=seed)
    p = len(apl.names)
    if p < 2:
        raise ValueError("the model needs at least 2 coefficients, as Bergm's")
    mean, sigma = _prior(prior_mean, prior_sigma, p)
    prior = multivariate_normal(mean, sigma)
    precision = np.linalg.inv(sigma)
    rng = np.random.default_rng(seed)
    hessian = apl.hessian(apl.theta_pl)
    s_prop = v_proposal**2 * np.linalg.inv(precision - hessian)
    if method == "CJ":
        iters = 30000 if main_iters is None else int(main_iters)

        def log_post(t):
            return apl.loglik(t) + prior.logpdf(t)

        draws, rate = _metropolis(log_post, apl.theta_pl, s_prop, iters, burn_in, rng)
        draws = draws[burn_in:]  # Bergm drops the burn-in twice
        star = draws.mean(axis=0)
        post_star = log_post(star)
        # Chib and Jeliazkov: the posterior ordinate at star, from the
        # acceptance probabilities of moves to it (from posterior draws) and
        # from it (proposals), whose ratio includes the prior.
        g = draws[rng.integers(len(draws), size=num_samples)]
        log_alpha_g = np.minimum(0.0, post_star - (apl.loglik(g) + prior.logpdf(g)))
        q_g = multivariate_normal(star, s_prop).pdf(g)
        j = rng.multivariate_normal(star, s_prop, size=num_samples)
        log_alpha_j = np.minimum(0.0, apl.loglik(j) + prior.logpdf(j) - post_star)
        ordinate = np.mean(np.exp(log_alpha_g) * q_g) / np.mean(np.exp(log_alpha_j))
        log_evidence = post_star - np.log(ordinate)
        from .terms import ErgmDifferenceWarning

        warnings.warn("Bergm's evidence(evidence.method = 'CJ') leaves the prior out of Chib and Jeliazkov's "
                      "acceptance probabilities; ergmx includes it, as the method has it (with a diffuse prior, the "
                      "estimates barely differ)", ErgmDifferenceWarning, stacklevel=2)
        return ModelEvidence("CJ", float(log_evidence), apl.names, draws, rate, apl, time.time() - started)
    iters = 20000 if main_iters is None else int(main_iters)
    temps = np.linspace(0, 1, 50) ** 5 if temps is None else np.asarray(temps, dtype=float)
    if temps[0] != 0 or temps[-1] != 1 or np.any(np.diff(temps) <= 0):
        raise ValueError("temps must increase from 0 to 1")
    # Each temperature's proposal: v_proposal^2 times the inverse curvature
    # of its tempered posterior, t (-H) + the prior's precision. (Bergm widens
    # the posterior's by a power of the temperature, too little at low ones,
    # where the tempered posterior is nearly the prior: those chains don't
    # mix, and their mean log-likelihoods, and the evidence, are off.)
    means, variances = np.zeros(len(temps)), np.zeros(len(temps))
    start, rate, posterior = apl.theta_pl, None, None
    for k in range(len(temps) - 1, -1, -1):
        t = temps[k]
        if t == 0:
            draws = rng.multivariate_normal(mean, sigma, size=iters)
        else:
            cov = v_proposal**2 * np.linalg.inv(precision - t * hessian)

            def log_tempered(theta, t=t):
                return t * apl.loglik(theta, scaled=False) + prior.logpdf(theta)

            draws, accepted = _metropolis(log_tempered, start, cov, iters, burn_in, rng)
            if k == len(temps) - 1:
                rate, posterior = accepted, draws
            start = draws[-1]
        # E_t[log adjusted PL], with second-degree zero-variance control variates (Mira et al. 2013).
        values = apl.loglik(draws, scaled=False)
        scores = t * apl.score(draws) - (draws - mean) @ precision
        columns = [scores, draws * scores + 1.0]
        for i in range(p - 1):
            for k2 in range(i + 1, p):
                columns.append((draws[:, i] * scores[:, k2] + draws[:, k2] * scores[:, i])[:, None])
        z = np.column_stack(columns)
        phi = -np.linalg.solve(np.atleast_2d(np.cov(z, rowvar=False)),
                               np.array([np.cov(z[:, c], values)[0, 1] for c in range(z.shape[1])]))
        corrected = values + z @ phi
        means[k], variances[k] = corrected.mean(), corrected.var(ddof=1)
    dt = np.diff(temps)
    integral = np.sum(dt * (means[1:] + means[:-1]) / 2 - dt**2 / 12 * (variances[1:] - variances[:-1]))
    return ModelEvidence("PP", float(apl.log_c + integral), apl.names, posterior, rate, apl, time.time() - started)


# -- Calibrated pseudo-posterior ----------------------------------------------------------------


def bergmC(network, formula, *, prior_mean=None, prior_sigma=None, burn_in: int = 10000,  # noqa: N802 (R's name)
           main_iters: int = 40000, aux_iters: int = 3000, v_proposal: float = 1.5, rm_iters: int = 50,
           n_aux_draws: int = 400, aux_thin: int = 50, constraints=None, bipartite=None, seed=None):
    """A posterior from the pseudo-likelihood, calibrated to the likelihood's,
    as Bergm's ``bergmC()`` (`Bouranis, Friel and Maire 2017 <https://doi.org/10.1016/j.socnet.2017.03.013>`__):
    much faster than :func:`ergmx.bergm` for large networks.

    A random-walk Metropolis sample of the pseudo-posterior (the
    pseudo-likelihood with the normal prior: ``main_iters`` draws after
    ``burn_in``) is moved and stretched to the true posterior: from its
    mode to the posterior's mode, found by Robbins-Monro stochastic
    approximation from the MLE (``rm_iters`` steps, each from
    ``n_aux_draws`` networks simulated ``aux_thin`` proposals apart after
    ``aux_iters``), and from its curvature to the posterior's (the
    covariance of the statistics at the mode, plus the prior's precision).
    The steps are Newton's, of gain 1/i: the gradient of the log posterior
    times the inverse of its curvature. (Bergm's, of size rm.a / i times the
    gradient, ignore the statistics' scales: with ``nodecov('wealth')``, the
    first moves the coefficient by several times its size.)

    Returns
    -------
    BergmFit, with its draws, summaries and posterior predictive checks.
    """
    from ._bayes import BergmFit
    from ._simulate import ergm
    from .terms import ErgmDifferenceWarning

    started = time.time()
    model = _bind_plain(network, formula, constraints, bipartite)
    p = model.n_params
    mean, sigma = _prior(prior_mean, prior_sigma, p)
    prior = multivariate_normal(mean, sigma)
    precision = np.linalg.inv(sigma)
    pseudo = _Pseudo(model)
    rng = np.random.default_rng(seed)
    theta_pl = logistic_regression(pseudo.x, pseudo.y, pseudo.w)[0]
    s_prop = v_proposal**2 * np.linalg.inv(precision - pseudo.hessian(theta_pl))

    def log_pseudo_posterior(t):
        return float(pseudo.loglik(t)[0]) + prior.logpdf(t)

    unadjusted, rate = _metropolis(log_pseudo_posterior, theta_pl, s_prop, int(main_iters), int(burn_in), rng)
    # The posterior's mode, by Robbins-Monro from the MLE.
    fit = ergm(network, formula, constraints=constraints, bipartite=bipartite, seed=seed, eval_loglik=False)
    observed = model.observed()
    theta = np.asarray(fit.params, dtype=float).copy()

    def statistics(at):
        sample, _, _ = model.simulate([model.network.edges], at, int(aux_iters), int(aux_thin), int(n_aux_draws),
                                      int(rng.integers(2**63)))
        return sample.reshape(-1, model.n_stats)

    for i in range(1, int(rm_iters) + 1):
        z = statistics(theta)
        gradient = -(z.mean(axis=0) - observed) - precision @ (theta - mean)
        curvature = np.atleast_2d(np.cov(z, rowvar=False)) + precision
        theta = theta + np.linalg.lstsq(curvature, gradient, rcond=None)[0] / i
    # The pseudo-posterior's mode and curvature.
    mode = minimize(lambda t: -log_pseudo_posterior(t), unadjusted.mean(axis=0),
                    jac=lambda t: -(pseudo.score(t) - precision @ (t - mean)), method="BFGS").x
    curvature_pl = -pseudo.hessian(mode) + precision
    curvature = np.atleast_2d(np.cov(statistics(theta), rowvar=False)) + precision
    w = solve_triangular(_upper_cholesky(curvature_pl), _upper_cholesky(curvature))
    corrected = (unadjusted - mode) @ np.linalg.inv(w).T + theta
    warnings.warn("Bergm's bergmC() gives optim() the gradient of the log pseudo-posterior with the wrong sign "
                  "for the function it minimizes, so that the optimizer stops at its start, the pseudo-posterior "
                  "sample's mean, rather than the mode the method calibrates from; ergmx finds the mode (the "
                  "posteriors differ little when the pseudo-posterior is nearly normal)",
                  ErgmDifferenceWarning, stacklevel=2)
    settings = {"method": "bergmC", "nchains": 1, "burn_in": burn_in, "main_iters": main_iters, "aux_iters": aux_iters,
                "rm_iters": rm_iters, "n_aux_draws": n_aux_draws, "aux_thin": aux_thin, "mode": theta}
    return BergmFit(model, list(model.names), corrected[None, :, :], rate, settings, seed, time.time() - started)
