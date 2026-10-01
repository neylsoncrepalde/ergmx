"""Fitting, simulating and summarizing: the functions users call."""

from __future__ import annotations

import dataclasses

import numpy as np

from . import _estimation
from ._estimation import Control
from ._fit import ErgmFit
from ._loglik import bridge_loglik
from ._model import bind
from ._network import to_graph


def summary_stats(network, formula) -> dict[str, float]:
    """Statistics of a network, like R's ``summary(net ~ formula)``.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
    formula : str or terms
        For example ``"edges + triangle"`` or ``edges() + triangle()``.
    """
    model = bind(network, formula)
    return dict(zip(model.names, model.observed().tolist()))


def ergm(network, formula, *, estimate: str = "MLE", init=None, seed=None, eval_loglik: bool = True,
         control: Control | None = None, **control_args) -> ErgmFit:
    """Fit an exponential-family random graph model.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
        The observed network. Vertex attributes are available to the terms.
    formula : str or terms
        The model, in R syntax: ``"edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)"``,
        or terms combined with ``+``.
    estimate : {"MLE", "MPLE", "CD"}
        ``"MLE"`` (default) gives the exact MLE of dyad-independent models
        and the Monte Carlo MLE of the others. ``"MPLE"`` stops at the
        maximum pseudo-likelihood estimate, and ``"CD"`` at the contrastive
        divergence estimate (as in ergm; no standard errors).
    init : {"MPLE", "CD"} or array-like, optional
        Starting coefficients for the Monte Carlo MLE: the MPLE (default),
        the contrastive divergence estimate (started from the MPLE), or given
        values.
    seed : int, optional
        Seed for reproducible results.
    eval_loglik : bool
        Estimate the log-likelihood of dyad-dependent models, for AIC and BIC,
        by path sampling (as ergm does by default). Dyad-independent models
        always get their exact log-likelihood.
    control : Control, optional
        MCMC and estimation settings. Keyword arguments (``samplesize=...``,
        ``interval=...``, ``n_chains=...``) override single settings.

    Returns
    -------
    ErgmFit
    """
    if estimate not in ("MLE", "MPLE", "CD"):
        raise ValueError(f"estimate must be 'MLE', 'MPLE' or 'CD', not {estimate!r}")
    control = dataclasses.replace(control or Control(), **control_args)
    model = bind(network, formula)
    pseudo = _estimation.mple(model)
    rng = np.random.default_rng(seed)
    if model.dyad_independent:
        return ErgmFit(model, dataclasses.replace(pseudo, method="MLE"), pseudo, control, seed)
    if estimate == "MPLE":
        return ErgmFit(model, pseudo, pseudo, control, seed)
    if estimate == "CD" or (isinstance(init, str) and init == "CD"):
        cd = _estimation.contrastive_divergence(model, pseudo.theta, control, rng)
        if estimate == "CD":
            return ErgmFit(model, cd, pseudo, control, seed)
        start = cd.theta
    elif init is None or (isinstance(init, str) and init == "MPLE"):
        start = pseudo.theta
    elif isinstance(init, str):
        raise ValueError(f"init must be 'MPLE', 'CD' or coefficients, not {init!r}")
    else:
        start = np.asarray(init, dtype=float)
        if start.shape != (model.n_stats,):
            raise ValueError(f"init must have {model.n_stats} values, one per coefficient")
    result = _estimation.mcmle(model, start, control, rng)
    if eval_loglik:
        loglik, se = bridge_loglik(model, result.theta, control, result.interval, rng)
        result = dataclasses.replace(result, loglik=loglik, loglik_se=se)
    return ErgmFit(model, result, pseudo, control, seed)


def simulate(network, formula, coef, nsim: int = 1, *, seed=None, output: str = "network",
             burnin: int | None = None, interval: int | None = None,
             triadic_weight: float | None = None):
    """Simulate networks from an ERGM, starting from ``network``.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
        The starting network, which also provides the vertex attributes.
    formula : str or terms
    coef : array-like or dict
        Coefficients, in the order of the formula's statistics or by name.
    nsim : int
        Number of networks.
    output : {"network", "stats"}
        Return graphs of the same kind as ``network``, or an ``nsim x
        statistics`` array of their statistics.
    burnin, interval : int, optional
        MCMC proposals before the first network and between networks.
        Default to ergm's 16384 and 1024.
    triadic_weight : float, optional
        Share of MCMC proposals that close or open a triangle. Defaults to 0.5
        for models with triangle or shared partner terms, 0 otherwise.
    """
    if output not in ("network", "stats"):
        raise ValueError(f"output must be 'network' or 'stats', not {output!r}")
    model = bind(network, formula)
    if isinstance(coef, dict):
        coef = [coef[name] for name in model.names]
    defaults = Control()
    seed = int(np.random.default_rng(seed).integers(2**63))
    sample, _, networks = model.core.simulate(
        [model.network.edges], [float(c) for c in coef],
        defaults.burnin if burnin is None else burnin,
        defaults.interval if interval is None else interval,
        nsim, seed, keep_networks=output == "network",
        triadic_weight=model.triadic_weight(triadic_weight),
    )
    if output == "stats":
        return sample[0]
    return [to_graph(model.network, edges) for edges in networks[0]]
