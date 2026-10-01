"""Fitting, simulating and summarizing: the functions users call."""

from __future__ import annotations

import dataclasses

import numpy as np

from . import _estimation
from ._estimation import Control
from ._fit import ErgmFit
from ._loglik import bridge_loglik
from ._model import bind
from ._network import to_graphs


def summary_stats(network, formula, *, bipartite=None) -> dict[str, float]:
    """Statistics of a network, like R's ``summary(net ~ formula)``.

    Missing dyads (edges with ``na=True``) count as non-ties, as in ergm.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
    formula : str or terms
        For example ``"edges + triangle"`` or ``edges() + triangle()``.
    bipartite : str or bool, optional
        For a bipartite network, the vertex attribute with each vertex's mode,
        as in :func:`ergm`.
    """
    model = bind(network, formula, bipartite=bipartite)
    return dict(zip(model.stat_names, model.observed().tolist()))


def ergm(network, formula, *, constraints=None, offset_coef=None, bipartite=None,
         estimate: str = "MLE", init=None, seed=None, eval_loglik: bool = True,
         control: Control | None = None, **control_args) -> ErgmFit:
    """Fit an exponential-family random graph model.

    Parameters
    ----------
    network : igraph.Graph, networkx.Graph, Networks or NetSeries
        The observed network. Vertex attributes are available to the terms.
        Edges with a true ``na`` attribute mark dyads whose value is unknown:
        the fit is then conditional on the observed dyads (missing at random,
        as in ergm). Several networks combined with :func:`ergmx.Networks`
        are modeled jointly (with N() terms), and the transitions of a
        :func:`ergmx.NetSeries` conditionally on the network before each
        (with tergm's operators; see :func:`ergmx.tergm`).
    formula : str or terms
        The model, in R syntax: ``"edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)"``,
        or terms combined with ``+``.
    constraints : str, optional
        Sample space constraints, in R syntax: ``"bd(maxout=5)"``,
        ``"degrees"``, ``"blocks('level', levels2=TRUE)"``... See
        :mod:`ergmx.constraints`.
    offset_coef : array-like, optional
        The fixed coefficients of the ``offset()`` terms, in formula order;
        ``-inf`` forbids the ties they count.
    bipartite : str or bool, optional
        For a bipartite (two-mode) network, the vertex attribute giving each
        vertex's mode: false or 0 for the first mode (ergm's "b1"), true or 1
        for the second. ``True`` uses igraph's ``type`` or networkx's
        ``bipartite`` attribute. Only ties between the modes are modeled.
    estimate : {"MLE", "MPLE", "CD"}
        ``"MLE"`` (default) gives the exact MLE of dyad-independent models
        and the Monte Carlo MLE of the others. ``"MPLE"`` stops at the
        maximum pseudo-likelihood estimate, and ``"CD"`` at the contrastive
        divergence estimate (as in ergm; no standard errors).
    init : {"MPLE", "CD"} or array-like, optional
        Starting coefficients for the Monte Carlo MLE: the MPLE (the default,
        unless the constraints are dyad-dependent), the contrastive
        divergence estimate (started from the MPLE; the default with
        dyad-dependent constraints, as in ergm), or given values.
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
    model = bind(network, formula, constraints, offset_coef, fitting=True, bipartite=bipartite)
    pseudo = _estimation.mple(model)
    rng = np.random.default_rng(seed)
    if model.exact:
        return ErgmFit(model, dataclasses.replace(pseudo, method="MLE"), pseudo, control, seed)
    if estimate == "MPLE":
        if model.constraints.dyad_dependent:
            raise ValueError(f"the MPLE ignores dyad-dependent constraints ({model.constraints!r}); "
                             "use estimate='CD' or 'MLE'")
        return ErgmFit(model, pseudo, pseudo, control, seed)
    if init is None:
        init = "CD" if model.constraints.dyad_dependent else "MPLE"
    if estimate == "CD" or (isinstance(init, str) and init == "CD"):
        cd = _estimation.contrastive_divergence(model, pseudo.theta, control, rng)
        if estimate == "CD":
            return ErgmFit(model, cd, pseudo, control, seed)
        start = cd.theta
    elif isinstance(init, str) and init == "MPLE":
        start = pseudo.theta
    elif isinstance(init, str):
        raise ValueError(f"init must be 'MPLE', 'CD' or coefficients, not {init!r}")
    else:
        start = np.asarray(init, dtype=float)
        if start.shape != (model.n_params,):
            raise ValueError(f"init must have {model.n_params} values, one per coefficient")
        start = np.where(model.fixed, model.fixed_values, start)
    result = _estimation.mcmle(model, start, control, rng)
    if eval_loglik:
        loglik, se, relative = bridge_loglik(model, result.theta, control, result.interval, rng)
        result = dataclasses.replace(result, loglik=loglik, loglik_se=se, loglik_relative=relative)
    return ErgmFit(model, result, pseudo, control, seed)


def simulate(network, formula, coef, nsim: int = 1, *, constraints=None, bipartite=None,
             seed=None, output: str = "network", burnin: int | None = None,
             interval: int | None = None, triadic_weight: float | None = None):
    """Simulate networks from an ERGM, starting from ``network``.

    Parameters
    ----------
    network : igraph.Graph, networkx.Graph, Networks or NetSeries
        The starting network, which also provides the vertex attributes.
        Missing dyads (``na`` edges) start as non-ties and are simulated too.
        With several networks combined, each simulation is a list of
        networks, one per network (for a NetSeries, each transition's
        current network, given the previous one). For simulating a network
        over time, see :func:`ergmx.simulate_dynamic`.
    formula : str or terms
    coef : array-like or dict
        Coefficients, in the order of the formula's parameters or by name,
        including those of offset terms (and the decays of curved terms).
    nsim : int
        Number of networks.
    constraints : str, optional
        Sample space constraints, as in :func:`ergm`.
    bipartite : str or bool, optional
        For a bipartite network, the vertex attribute with each vertex's mode,
        as in :func:`ergm`.
    output : {"network", "stats"}
        Return graphs of the same kind as ``network`` (lists of them, with
        combined networks), or an ``nsim x statistics`` array of their
        statistics.
    burnin, interval : int, optional
        MCMC proposals before the first network and between networks.
        Default to ergm's 16384 and 1024.
    triadic_weight : float, optional
        Share of MCMC proposals that close or open a triangle. Defaults to 0.5
        for models with triangle or shared partner terms, 0 otherwise.
    """
    if output not in ("network", "stats"):
        raise ValueError(f"output must be 'network' or 'stats', not {output!r}")
    model = bind(network, formula, constraints, bipartite=bipartite)
    if isinstance(coef, dict):
        coef = [coef[name] for name in model.names]
    defaults = Control()
    seed = int(np.random.default_rng(seed).integers(2**63))
    sample, _, networks = model.simulate(
        [model.network.edges], coef,
        defaults.burnin if burnin is None else burnin,
        defaults.interval if interval is None else interval,
        nsim, seed, keep_networks=output == "network",
        triadic_weight=model.triadic_weight(triadic_weight),
    )
    if output == "stats":
        return sample[0]
    return [to_graphs(model.network, edges) for edges in networks[0]]
