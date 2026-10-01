"""Comparing fitted models by log-likelihood, AIC and BIC."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _same_network(a, b) -> bool:
    x, y = a._model.network, b._model.network
    return (x.n == y.n and x.directed == y.directed and np.array_equal(x.edges, y.edges)
            and np.array_equal(x.missing, y.missing))


class ModelComparison:
    """AIC, BIC and likelihood-ratio tests of fitted models. Print it."""

    def __init__(self, fits):
        self.fits = list(fits)

    @property
    def aic(self) -> np.ndarray:
        return np.array([f.aic for f in self.fits])

    @property
    def bic(self) -> np.ndarray:
        return np.array([f.bic for f in self.fits])

    def lr_tests(self) -> list:
        """For each model after the first: (chi2, df, p-value) of the likelihood-ratio
        test against the previous model, if they are nested; None otherwise."""
        tests = [None]
        for previous, fit in zip(self.fits, self.fits[1:]):
            nested = set(previous.names) < set(fit.names) and _same_network(previous, fit)
            if nested and fit.df > previous.df:
                chi2 = 2 * (fit.loglik - previous.loglik)
                df = fit.df - previous.df
                tests.append((chi2, df, float(stats.chi2.sf(chi2, df))))
            else:
                tests.append(None)
        return tests

    def __str__(self) -> str:
        best = np.nanmin(self.aic)
        lines = ["Model comparison:", "",
                 f"{'':>3}  {'df':>3}  {'log-likelihood':>20}  {'AIC':>10}  {'BIC':>10}  {'dAIC':>7}  "
                 f"{'LR chi2':>8}  {'df':>3}  {'Pr(>chi2)':>9}"]
        for i, (fit, test) in enumerate(zip(self.fits, self.lr_tests()), 1):
            se = fit.loglik_se
            loglik = f"{fit.loglik:.3f}" + (f" ({se:.3f})" if se else "")
            row = (f"{i:>3}  {fit.df:>3}  {loglik:>20}  {fit.aic:10.2f}  {fit.bic:10.2f}  "
                   f"{fit.aic - best:7.2f}")
            if test is not None:
                row += f"  {test[0]:8.2f}  {test[1]:>3}  {test[2]:9.4g}"
            lines.append(row)
        lines.append("")
        lines += [f"{i:>3}: {fit.formula}" for i, fit in enumerate(self.fits, 1)]
        if any(f.loglik_se for f in self.fits):
            lines += ["", "Log-likelihoods of dyad-dependent models are Monte Carlo estimates "
                          "(standard errors in parentheses)."]
        if self.fits[0].loglik_relative:
            lines += [f"Log-likelihoods are relative to the null model of the constraints "
                      f"({self.fits[0].constraints!r})."]
        return "\n".join(lines)

    __repr__ = __str__


def compare(*fits) -> ModelComparison:
    """Compare fitted models of the same network, like R's ``AIC()`` and ``anova()``.

    Consecutive nested models (each with all the terms of the previous one)
    also get a likelihood-ratio test.
    """
    if len(fits) < 2:
        raise ValueError("compare needs at least two fitted models")
    missing = [i for i, f in enumerate(fits, 1) if f.loglik is None]
    if missing:
        raise ValueError(f"models {missing} have no log-likelihood: fit them with eval_loglik=True")
    if any(f.constraints != fits[0].constraints for f in fits):
        raise ValueError("the models have different constraints, so their log-likelihoods are "
                         "not comparable: " + ", ".join(repr(f.constraints) for f in fits))
    if any(not _same_network(f, fits[0]) for f in fits):
        raise ValueError("the models are fitted to different networks (or missing dyads)")
    return ModelComparison(fits)
