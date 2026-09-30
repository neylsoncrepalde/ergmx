"""Fitting, simulating and summarizing: the functions users call."""

from __future__ import annotations

import dataclasses

import numpy as np

from . import _estimation
from ._estimation import Control
from ._fit import ErgmFit
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


def ergm(network, formula, *, estimate: str = "MLE", init=None, seed=None,
         control: Control | None = None, **control_args) -> ErgmFit:
    """Fit an exponential-family random graph model.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
        The observed network. Vertex attributes are available to the terms.
    formula : str or terms
        The model, in R syntax: ``"edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)"``,
        or terms combined with ``+``.
    estimate : {"MLE", "MPLE"}
        ``"MLE"`` (default) gives the exact MLE of dyad-independent models
        and the Monte Carlo MLE of the others. ``"MPLE"`` stops at the
        maximum pseudo-likelihood estimate.
    init : array-like, optional
        Starting coefficients for the Monte Carlo MLE. Defaults to the MPLE.
    seed : int, optional
        Seed for reproducible results.
    control : Control, optional
        MCMC and estimation settings. Keyword arguments (``samplesize=...``,
        ``interval=...``, ``n_chains=...``) override single settings.

    Returns
    -------
    ErgmFit
    """
    if estimate not in ("MLE", "MPLE"):
        raise ValueError(f"estimate must be 'MLE' or 'MPLE', not {estimate!r}")
    control = dataclasses.replace(control or Control(), **control_args)
    model = bind(network, formula)
    pseudo = _estimation.mple(model)
    if model.dyad_independent:
        result = dataclasses.replace(pseudo, method="MLE")  # the MPLE is the MLE
    elif estimate == "MPLE":
        result = pseudo
    else:
        start = pseudo.theta if init is None else np.asarray(init, dtype=float)
        if start.shape != (model.n_stats,):
            raise ValueError(f"init must have {model.n_stats} values, one per coefficient")
        rng = np.random.default_rng(seed)
        result = _estimation.mcmle(model, start, control, rng)
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
