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
    """A fitted ERGM, as returned by :func:`ergmx.ergm`.

    Print :meth:`summary` for the table of coefficients, like R's
    ``summary(fit)``; check the fit with :meth:`mcmc_diagnostics` and
    :meth:`gof`; compare it with others with :func:`ergmx.compare`.
    """

    def __init__(self, model: BoundModel, estimate: Estimate, mple: Estimate, control: Control, seed):
        self._model = model
        self._estimate = estimate
        self._mple = mple
        self.control = control
        self._seed = seed

    def save(self, path) -> None:
        """Save the fit to a file, to reload with :func:`ergmx.load_fit`: its
        model and network, estimates, MCMC sample and settings, with Python's
        pickle (so only load files you trust)."""
        _save(self, path)

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
        """Standard errors by name, including MCMC error (NaN for fixed coefficients)."""
        return dict(zip(self.names, np.sqrt(np.diag(self._estimate.cov)).tolist()))

    @property
    def offset(self) -> dict[str, bool]:
        """Whether each coefficient is fixed by offset() rather than estimated."""
        return dict(zip(self.names, (self._model.fixed & ~self._model.constant).tolist()))

    @property
    def constraints(self):
        """The model's sample space constraints."""
        return self._model.constraints

    @property
    def mple(self) -> dict[str, float]:
        """The maximum pseudo-likelihood estimate, the starting point of the MCMC MLE."""
        return dict(zip(self.names, self._mple.theta.tolist()))

    # -- Fit information ------------------------------------------------------------

    @property
    def method(self) -> str:
        """``"MLE"`` (exact, for dyad-independent models), ``"MCMLE"``, ``"MPLE"`` or ``"CD"``."""
        return self._estimate.method

    @property
    def iterations(self) -> int:
        """Iterations of the Monte Carlo MLE (or of contrastive divergence)."""
        return self._estimate.iterations

    @property
    def converged(self) -> bool:
        """Whether the estimation met its convergence criterion."""
        return self._estimate.converged

    @property
    def loglik(self) -> float | None:
        """Log-likelihood: exact for dyad-independent models, estimated by path
        sampling for the others (None if fitted with ``eval_loglik=False``)."""
        return self._estimate.loglik

    @property
    def loglik_se(self) -> float | None:
        """Monte Carlo standard error of the log-likelihood (0 if exact)."""
        return self._estimate.loglik_se

    @property
    def loglik_relative(self) -> bool:
        """Whether the log-likelihood is relative to the null model (the uniform
        distribution on the networks the constraints allow), as with
        dyad-dependent constraints in ergm. Relative log-likelihoods compare
        models with the same constraints."""
        return self._estimate.loglik_relative

    @property
    def df(self) -> int:
        """Number of estimated coefficients (offsets and constant statistics excluded)."""
        return int(self._model.free.sum())

    @property
    def formula(self):
        """The model's terms."""
        return self._model.formula

    @property
    def aic(self) -> float | None:
        """Akaike's information criterion, -2 loglik + 2 p."""
        return None if self.loglik is None else -2 * self.loglik + 2 * self.df

    @property
    def bic(self) -> float | None:
        """Bayesian information criterion, -2 loglik + p log(number of free
        observed dyads), as in ergm."""
        if self.loglik is None:
            return None
        return -2 * self.loglik + math.log(self._model.n_observations) * self.df

    @property
    def sample(self) -> np.ndarray | None:
        """Statistics sampled in the last iteration (chains x samples x statistics)."""
        return self._estimate.sample

    @property
    def observed(self) -> dict[str, float]:
        """Statistics of the observed network (for curved terms, the counts
        their parameters weight)."""
        return dict(zip(self._model.stat_names, self._model.observed().tolist()))

    # -- Using the model --------------------------------------------------------------

    def simulate(self, nsim: int = 1, *, seed=None, output: str = "network", time_slices=None,
                 nw_start="last", **options):
        """Simulate networks from the fitted model, starting from the observed one.

        See :func:`ergmx.simulate`. For a model fitted to a series of networks
        (:func:`ergmx.tergm`), ``time_slices=`` simulates the process forward
        in time from the network ``nw_start`` instead, as tergm's
        ``simulate(fit, nw.start=, time.slices=)``: ``"last"`` (the default)
        or ``"first"`` network of the series, its 1-based position, or a
        network. See :func:`ergmx.simulate_dynamic`, whose options apply.
        """
        if time_slices is not None:
            from ._temporal import simulate_dynamic

            network = self._model.network
            if getattr(self._model, "valued", False):
                raise ValueError("time_slices= is for binary series: a valued series' simulate() draws each "
                                 "transition given the network before it")
            if not network.series:
                raise ValueError("time_slices= needs a model fitted to a series of networks "
                                 "(ergmx.tergm() or a NetSeries())")
            series = [network.blocks[0].prev, *(b.network for b in network.blocks)]
            if isinstance(nw_start, str):
                if nw_start not in ("first", "last"):
                    raise ValueError(f"nw_start must be 'first', 'last', a position or a network, "
                                     f"not {nw_start!r}")
                start = series[0 if nw_start == "first" else -1]
            elif isinstance(nw_start, (int, np.integer)):
                if not 1 <= nw_start <= len(series):
                    raise ValueError(f"nw_start must be between 1 and {len(series)}")
                start = series[nw_start - 1]
            else:
                start = nw_start
            if output != "network":
                raise ValueError("dynamic simulation returns DynamicSimulation objects; their "
                                 ".stats and .monitor hold the statistics")
            from ._temporal import _varying, simulate_varying

            if _varying(self._model):
                # Coefficients that change over time continue the series from its end.
                if not (isinstance(nw_start, str) and nw_start == "last"):
                    raise ValueError("a model whose coefficients vary over time (lm=) continues the "
                                     "series from its last network: use nw_start='last'")
                return simulate_varying(self._model, self.params, time_slices, nsim=nsim, seed=seed,
                                        **options)
            return simulate_dynamic(start, self._model.formula, self.params, time_slices, nsim=nsim,
                                    constraints=self._model.constraints, seed=seed, **options)
        if getattr(self._model, "valued", False):
            from ._valued import simulate_valued

            return simulate_valued(self._model, self.params, nsim, seed, options.get("burnin"),
                                   options.get("interval", self._estimate.interval), output,
                                   options.get("response", self._model.response or "weight"))
        from ._simulate import simulate

        return simulate(
            self._model.network, self._model.formula, self.params, nsim,
            constraints=self._model.constraints, seed=seed, output=output, **options,
        )

    def mcmc_diagnostics(self):
        """Diagnostics of the MCMC sample of the last iteration, like R's
        ``mcmc.diagnostics()``. Print the result, or call its ``plot()``."""
        from ._diagnostics import McmcDiagnostics

        if self._estimate.sample is None:
            raise ValueError(f"this model was fitted by {self.method}, without MCMC")
        from ._estimation import project

        model, est = self._model, self._estimate
        target = model.observed()
        if est.sample_obs is not None:  # missing dyads: their conditional mean
            target = est.sample_obs.reshape(-1, model.n_stats).mean(axis=0)
        # Curved terms have many statistics: diagnose the estimating functions,
        # one per parameter (the statistics themselves for other terms).
        theta = self.params
        names = [n for n, f in zip(self.names, model.free) if f]
        return McmcDiagnostics(names, project(model, theta, est.sample),
                               project(model, theta, target[None, None, :])[0, 0], est.interval)

    def gof(self, nsim: int = 100, **options):
        """Goodness of fit of the model. See :func:`ergmx.gof`."""
        from ._gof import gof

        return gof(self, nsim=nsim, **options)

    def summary(self) -> FitSummary:
        """The table of coefficients, standard errors and tests, with the
        log-likelihood, AIC and BIC. Print it (it prints itself in notebooks)."""
        return FitSummary(self)

    def to_frame(self):
        """The table of coefficients as a pandas DataFrame (needs pandas), with
        R's columns: Estimate, Std. Error, MCMC %, z value and Pr(>|z|)."""
        from ._interpret import _frame

        se = np.sqrt(np.diag(self._estimate.cov))
        mc = np.zeros_like(se) if self._estimate.mc_cov is None else np.diag(self._estimate.mc_cov)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(se > 0, 100 * mc / se**2, 0.0)
            z = self.params / se
        p = [_pvalue(v) if np.isfinite(v) else float("nan") for v in z]
        return _frame({"Estimate": self.params, "Std. Error": se, "MCMC %": pct, "z value": z,
                       "Pr(>|z|)": p}, index=self.names)

    # -- Interpreting the model ---------------------------------------------------------

    def predict(self, conditional: bool = True, type: str = "response", nsim: int = 100, *,
                seed=None, **options):
        """Tie probabilities of every dyad, as R's ``predict(fit)``.

        With ``conditional=True``, each dyad's probability of a tie given the
        rest of the observed network, computed exactly from its change
        statistics (``type="link"`` for the log-odds); with
        ``conditional=False``, the share of ``nsim`` networks simulated from
        the model in which the dyad is a tie. As in ergm, conditional
        probabilities ignore the sample space constraints, and dyads whose
        value is missing are predicted too; unconditional ones are simulated
        with the model's constraints (ergm ignores them).

        Returns
        -------
        TiePredictions
            ``tail``, ``head`` and ``p`` arrays, with ``.matrix()``,
            ``.mean_by(attribute)`` and ``.to_frame()``.
        """
        from ._interpret import predict

        if not conditional:
            options.setdefault("interval", self._estimate.interval)
            options.setdefault("n_chains", self.control.n_chains)
            options.setdefault("triadic_weight", self.control.triadic_weight)
        return predict(self._model, self.params, conditional=conditional, type=type, nsim=nsim,
                       seed=seed, **options)

    def marginal_effects(self):
        """Average marginal effects on the tie probability, as R's ergMargins
        (``ergm.AME()``): for each estimated parameter, the change in a dyad's
        conditional tie probability per unit of the term's statistic,
        theta * p (1 - p), averaged over the dyads, with its delta-method
        standard error. (ergMargins holds the probabilities fixed in the
        delta method; ergmx also counts how they change with the
        parameters.) Curved terms' decays have no marginal effect.

        Returns
        -------
        NumericTable
            Columns AME, Delta SE, Z and P; ``.to_frame()`` for pandas.
        """
        from ._interpret import marginal_effects

        return marginal_effects(self)

    def odds_ratios(self, level: float = 0.95):
        """Odds ratios, exp(coefficient), with Wald confidence intervals: the
        factor by which a unit increase in a term's statistic multiplies the
        conditional odds of a tie (of valued models, the conditional
        probability of a dyad's value relative to the value one lower)."""
        from ._interpret import odds_ratios

        return odds_ratios(self, level)

    def confint(self, level: float = 0.95):
        """Wald confidence intervals of the estimated coefficients, as R's
        ``confint(fit)``."""
        from ._interpret import confint

        return confint(self, level)

    def __repr__(self) -> str:
        title = {"MLE": "Maximum Likelihood", "MCMLE": "Monte Carlo MLE", "MPLE": "MPLE",
                 "CD": "Contrastive Divergence"}
        width = max(map(len, self.names))
        rows = "\n".join(f"  {n:<{width}}  {v: .4f}" for n, v in self.coef.items())
        return f"ErgmFit ({title[self.method]} coefficients):\n{rows}"


class FitSummary:
    """The table of coefficients of a fit, printed like R's ``summary(ergm)``."""

    def __init__(self, fit: ErgmFit):
        self.fit = fit

    def __str__(self) -> str:
        fit = self.fit
        model, est = fit._model, fit._estimate
        se = np.sqrt(np.diag(est.cov))
        mc = np.zeros_like(se) if est.mc_cov is None else np.diag(est.mc_cov)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(se > 0, 100 * mc / se**2, 0.0)
            z = fit.params / se
        conditional = "Conditional " if model.network.series else ""
        header = {
            "MLE": f"{conditional}Maximum Likelihood Results",
            "MCMLE": f"Monte Carlo {conditional}Maximum Likelihood Results",
            "MPLE": f"{conditional}Maximum Pseudolikelihood Results",
            "CD": "Contrastive Divergence Results",
        }[fit.method]
        width = max(map(len, fit.names))
        lines = [
            f"{header}:",
            "",
            f"{'':<{width}}  {'Estimate':>9}  {'Std. Error':>10}  {'MCMC %':>6}  "
            f"{'z value':>7}  {'Pr(>|z|)':>8}",
        ]
        for i, name in enumerate(fit.names):
            if model.fixed[i]:
                note = "constant under the constraints" if model.constant[i] else "offset"
                lines.append(f"{name:<{width}}  {fit.params[i]:9.4f}  {'':10}  {'':6}  {'':7}  "
                             f"{'':8}  ({note})")
                continue
            p = _pvalue(z[i]) if np.isfinite(z[i]) else float("nan")
            pval = "<1e-04" if p < 1e-4 else f"{p:.5f}"
            lines.append(
                f"{name:<{width}}  {fit.params[i]:9.4f}  {se[i]:10.4f}  {pct[i]:6.0f}  "
                f"{z[i]:7.3f}  {pval:>8} {_stars(p)}"
            )
        lines += ["---", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", ""]
        if fit.loglik is not None:
            se_text = f" (MC SE {fit.loglik_se:.3f})" if fit.loglik_se else ""
            kind = "Log-likelihood, relative to the null model" if fit.loglik_relative else "Log-likelihood"
            lines.append(f"{kind}: {fit.loglik:.4f}{se_text}   AIC: {fit.aic:.4f}   BIC: {fit.bic:.4f}")
        elif getattr(model, "valued", False) and model.loglik_unavailable:
            lines.append(f"Log-likelihood: not computed ({model.loglik_unavailable}).")
        else:
            lines.append("Log-likelihood: not computed (eval_loglik=False).")
        if model.constraints:
            lines.append(f"Constraints: {model.constraints!r}.")
        if model.has_missing:
            lines.append(f"Missing dyads: {len(model.network.missing)}, assumed missing at random.")
        if model.target is not None:
            lines.append("Fitted to target statistics, from a network simulated towards them (san()).")
        if model.network.combined:
            what = "transitions, each conditional on the network before it" if model.network.series \
                else "layers of a network" if "_layers" in model.network.graph_attributes else "networks"
            lines.append(f"Fitted to {len(model.network.blocks)} {what}.")
        if fit.method == "MCMLE":
            status = "Converged" if fit.converged else "Did NOT converge"
            lines.append(f"{status} after {fit.iterations} iterations "
                         f"({fit.control.n_chains} chains, {fit.control.samplesize} samples).")
        elif fit.method == "MPLE":
            lines.append("Standard errors of the MPLE of a dyad-dependent model are unreliable.")
        elif fit.method == "CD":
            lines.append("Contrastive divergence estimates have no standard errors; they are "
                         "starting values for the MLE.")
        return "\n".join(lines)

    __repr__ = __str__


def _save(fit, path) -> None:
    import pickle

    from . import __version__

    with open(path, "wb") as f:
        pickle.dump({"ergmx": __version__, "fit": fit}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_fit(path):
    """A fit saved with ``fit.save(path)`` (:meth:`ErgmFit.save`), ready to
    summarize, simulate, check and compare as before. It is read with
    Python's pickle, which can run code: only load files you trust."""
    import pickle
    import warnings

    from . import __version__

    with open(path, "rb") as f:
        saved = pickle.load(f)
    if not isinstance(saved, dict) or "fit" not in saved:
        raise ValueError(f"{path} is not a fit saved by ergmx")
    if saved.get("ergmx") != __version__:
        warnings.warn(f"the fit was saved with ergmx {saved.get('ergmx')}; this is ergmx {__version__}",
                      stacklevel=2)
    return saved["fit"]
