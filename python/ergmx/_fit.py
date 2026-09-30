"""The result of fitting an ERGM."""

from __future__ import annotations

import math

import numpy as np

from ._estimation import Control, Estimate
from ._model import BoundModel


def _pvalue(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def _stars(p: float) -> str:
    for cut, mark in ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, ".")):
        if p < cut:
            return mark
    return ""


class ErgmFit:
    """A fitted ERGM. Use :meth:`summary` for the table of coefficients."""

    def __init__(self, model: BoundModel, estimate: Estimate, mple: Estimate, control: Control, seed):
        self._model = model
        self._estimate = estimate
        self._mple = mple
        self.control = control
        self._seed = seed

    # -- Coefficients -------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of the coefficients, as in ergm."""
        return list(self._model.names)

    @property
    def params(self) -> np.ndarray:
        """The estimated coefficients."""
        return self._estimate.theta.copy()

    @property
    def coef(self) -> dict[str, float]:
        """The estimated coefficients by name."""
        return dict(zip(self.names, self._estimate.theta.tolist()))

    @property
    def cov(self) -> np.ndarray:
        """Covariance matrix of the estimates, including MCMC error."""
        return self._estimate.cov.copy()

    @property
    def stderr(self) -> dict[str, float]:
        """Standard errors by name, including MCMC error."""
        return dict(zip(self.names, np.sqrt(np.diag(self._estimate.cov)).tolist()))

    @property
    def mple(self) -> dict[str, float]:
        """The maximum pseudo-likelihood estimate, the starting point of the MCMC MLE."""
        return dict(zip(self.names, self._mple.theta.tolist()))

    # -- Fit information ------------------------------------------------------------

    @property
    def method(self) -> str:
        """``"MLE"`` (exact, for dyad-independent models), ``"MCMLE"`` or ``"MPLE"``."""
        return self._estimate.method

    @property
    def iterations(self) -> int:
        return self._estimate.iterations

    @property
    def converged(self) -> bool:
        return self._estimate.converged

    @property
    def loglik(self) -> float | None:
        """Log-likelihood. Only computed for dyad-independent models for now."""
        return self._estimate.loglik

    @property
    def aic(self) -> float | None:
        return None if self.loglik is None else -2 * self.loglik + 2 * len(self.names)

    @property
    def bic(self) -> float | None:
        if self.loglik is None:
            return None
        n_dyads = self._model.network.n * (self._model.network.n - 1)
        if not self._model.network.directed:
            n_dyads //= 2
        return -2 * self.loglik + math.log(n_dyads) * len(self.names)

    @property
    def sample(self) -> np.ndarray | None:
        """Statistics sampled in the last iteration (chains x samples x statistics)."""
        return self._estimate.sample

    @property
    def observed(self) -> dict[str, float]:
        """Statistics of the observed network."""
        return dict(zip(self.names, self._model.observed().tolist()))

    # -- Using the model --------------------------------------------------------------

    def simulate(self, nsim: int = 1, *, seed=None, output: str = "network", **options):
        """Simulate networks from the fitted model, starting from the observed one.

        See :func:`ergmx.simulate`.
        """
        from ._simulate import simulate

        return simulate(
            self._model.network, self._model.formula, self.params, nsim, seed=seed,
            output=output, **options,
        )

    def gof(self, nsim: int = 100, **options):
        """Goodness of fit of the model. See :func:`ergmx.gof`."""
        from ._gof import gof

        return gof(self, nsim=nsim, **options)

    def summary(self) -> FitSummary:
        return FitSummary(self)

    def __repr__(self) -> str:
        title = {"MLE": "Maximum Likelihood", "MCMLE": "Monte Carlo MLE", "MPLE": "MPLE"}
        width = max(map(len, self.names))
        rows = "\n".join(f"  {n:<{width}}  {v: .4f}" for n, v in self.coef.items())
        return f"ErgmFit ({title[self.method]} coefficients):\n{rows}"


class FitSummary:
    """The table of coefficients of a fit, printed like R's ``summary(ergm)``."""

    def __init__(self, fit: ErgmFit):
        self.fit = fit

    def __str__(self) -> str:
        fit = self.fit
        est = fit._estimate
        se = np.sqrt(np.diag(est.cov))
        mc = np.zeros_like(se) if est.mc_cov is None else np.diag(est.mc_cov)
        pct = np.where(se > 0, 100 * mc / se**2, 0.0)
        z = fit.params / se
        p = [_pvalue(v) for v in z]
        header = {
            "MLE": "Maximum Likelihood Results",
            "MCMLE": "Monte Carlo Maximum Likelihood Results",
            "MPLE": "Maximum Pseudolikelihood Results",
        }[fit.method]
        width = max(map(len, fit.names))
        lines = [
            f"{header}:",
            "",
            f"{'':<{width}}  {'Estimate':>9}  {'Std. Error':>10}  {'MCMC %':>6}  "
            f"{'z value':>7}  {'Pr(>|z|)':>8}",
        ]
        for i, name in enumerate(fit.names):
            pval = "<1e-04" if p[i] < 1e-4 else f"{p[i]:.5f}"
            lines.append(
                f"{name:<{width}}  {fit.params[i]:9.4f}  {se[i]:10.4f}  {pct[i]:6.0f}  "
                f"{z[i]:7.3f}  {pval:>8} {_stars(p[i])}"
            )
        lines += ["---", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", ""]
        if fit.loglik is not None:
            lines.append(f"Log-likelihood: {fit.loglik:.4f}   AIC: {fit.aic:.4f}   BIC: {fit.bic:.4f}")
        else:
            lines.append("Log-likelihood: not computed for dyad-dependent models yet.")
        if fit.method == "MCMLE":
            status = "Converged" if fit.converged else "Did NOT converge"
            lines.append(f"{status} after {fit.iterations} iterations "
                         f"({fit.control.n_chains} chains, {fit.control.samplesize} samples).")
        elif fit.method == "MPLE":
            lines.append("Standard errors of the MPLE of a dyad-dependent model are unreliable.")
        return "\n".join(lines)

    __repr__ = __str__
