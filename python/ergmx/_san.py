"""Networks with given statistics, by simulated annealing (ergm's ``san()``),
and models fitted to target statistics rather than to a network."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np

from ._model import BoundModel, bind
from ._network import to_graphs

log = logging.getLogger("ergmx")


@dataclass
class SanControl:
    """Settings of the simulated annealing, with ergm's defaults (``control.san()``).

    ``maxit`` runs, of ``nsteps`` proposals in all (shared out as
    ``nsteps_alloc(maxit)``, by default 2, 4, 8..., so later runs are longer),
    at temperatures falling linearly from ``tau`` to 0 at the last run. The
    weights of the statistics start at ``invcov`` (by default the identity
    over the number of statistics) and are then the inverse of the
    covariance of the changes proposed in the previous run (only its
    diagonal with ``invcov_diag``), scaled to trace 1. ``samplesize``
    samples per run estimate that covariance. Finite offsets are ignored
    unless ``ignore_finite_offsets`` is false; infinite ones forbid the
    changes they count.
    """

    maxit: int = 4
    tau: float = 1.0
    invcov: np.ndarray | None = None
    invcov_diag: bool = False
    nsteps_alloc: object = None
    nsteps: int = 2**19
    samplesize: int = 2**12
    ignore_finite_offsets: bool = True
    #: Share of triadic proposals; None: as the model's MCMC.
    triadic_weight: float | None = None


def _targeted(model: BoundModel) -> tuple[list[int], list[tuple[int, float]]]:
    """The statistics SAN targets (those not of offset terms), and the offset
    statistics with their coefficients."""
    targeted, offsets = [], []
    theta = np.where(model.fixed, model.fixed_values, 0.0)
    eta = model.eta(theta)
    for term, ps, _ in model.blocks:
        for k in range(ps.start, ps.stop):
            if term.is_offset:
                offsets.append((k, float(eta[k])))
            else:
                targeted.append(k)
    return targeted, offsets


def _target_vector(model: BoundModel, targeted: list[int], target_stats) -> np.ndarray:
    """The targets of the targeted statistics, from a list (in order) or a dict by name."""
    names = [model.stat_names[k] for k in targeted]
    if isinstance(target_stats, dict):
        unknown = [k for k in target_stats if k not in names]
        missing = [k for k in names if k not in target_stats]
        if unknown or missing:
            raise ValueError(f"target_stats: {'no statistics ' + str(unknown) if unknown else ''}"
                             f"{'; ' if unknown and missing else ''}"
                             f"{'missing ' + str(missing) if missing else ''}; the statistics are {names}")
        return np.array([float(target_stats[k]) for k in names])
    target = np.atleast_1d(np.asarray(target_stats, dtype=float))
    if target.shape != (len(names),):
        raise ValueError(f"target_stats must have {len(names)} values, one per statistic of the "
                         f"non-offset terms ({names}), not {target.size}")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_stats must be finite")
    return target


def anneal(model: BoundModel, target: np.ndarray, control: SanControl, rng: np.random.Generator,
           only_last: bool = True):
    """Simulated annealing from the model's network towards `target` (of the
    statistics `_targeted` gives). Returns the last network's edges (or each
    run's, without `only_last`) and its deviations from the targets."""
    targeted, offsets = _targeted(model)
    q = len(targeted)
    if control.maxit < 1:
        raise ValueError("SanControl.maxit must be at least 1")
    weights = np.eye(q) / q if control.invcov is None else np.asarray(control.invcov, dtype=float)
    if weights.shape != (q, q):
        raise ValueError(f"invcov must be {q} x {q}, one row and column per targeted statistic")
    alloc = (control.nsteps_alloc(control.maxit) if callable(control.nsteps_alloc)
             else control.nsteps_alloc if control.nsteps_alloc is not None
             else 2.0 ** np.arange(1, control.maxit + 1))
    alloc = np.resize(np.asarray(alloc, dtype=float), control.maxit)
    steps = np.round(alloc / alloc.sum() * control.nsteps).astype(np.int64)
    if control.ignore_finite_offsets:
        offsets = [(k, eta) for k, eta in offsets if not np.isfinite(eta)]
    triadic = model.triadic_weight(control.triadic_weight)
    edges, runs = model.network.edges, []
    deviations = model.observed()[targeted] - target
    for i in range(1, control.maxit + 1):
        tau = control.tau * ((1 / i - 1 / control.maxit) / (1 - 1 / control.maxit)) if control.maxit > 1 else 0.0
        # At zero temperature only infinite offsets act, as in ergm.
        run_offsets = offsets if tau > np.finfo(float).eps else [(k, e) for k, e in offsets if not np.isfinite(e)]
        edges, sample, proposed = model.core.san(
            edges, targeted, [float(t) for t in target], run_offsets,
            [float(w) for w in weights.ravel()], float(tau), int(steps[i - 1]), control.samplesize,
            int(rng.integers(2**63)), triadic, space=model.space)
        deviations = sample[-1]
        runs.append(edges)
        log.info("SAN run %d: temperature %.3g, %d proposals, largest deviation %.4g", i, tau, steps[i - 1],
                 np.max(np.abs(deviations), initial=0.0))
        if len(proposed) > 1:
            cov = np.atleast_2d(np.cov(proposed, rowvar=False))
            if control.invcov_diag:
                cov = np.diag(np.diag(cov))
            weights = np.linalg.pinv(cov, rcond=np.finfo(float).eps ** 0.75, hermitian=True)
            small = np.diag(weights) < np.finfo(float).eps
            weights[small, small] = min(max(np.diag(weights).max(initial=0.0), np.finfo(float).eps), 1.0)
            weights = weights / np.trace(weights)
        if only_last and i < control.maxit and np.all(deviations == 0):
            break
    return (edges if only_last else runs), deviations


def san(network, formula, target_stats, *, constraints=None, offset_coef=None, bipartite=None,
        seed=None, only_last: bool = True, control: SanControl | None = None, **control_args):
    """A network whose statistics are near ``target_stats``, by simulated
    annealing from ``network``, as ergm's ``san()``.

    Each proposal (the model's MCMC proposals, within the constraints) is
    accepted if it brings the statistics nearer the targets, in the
    weighted distance (s - target)' W (s - target), or, at a positive
    temperature, with a probability that falls with how much farther it
    takes them; the temperature falls to 0 over the runs, and W adapts to
    the statistics' covariance (see :class:`SanControl`). The result is no
    draw from any distribution: SAN only searches. ergm's :func:`ergm` uses
    it to fit models to target statistics (``ergm(target_stats=...)``).

    Parameters
    ----------
    network : igraph.Graph, networkx.Graph, Networks
        The starting network, which also gives the vertices and their
        attributes (an empty graph is fine).
    formula : str or terms
        The statistics. Terms in ``offset()`` are not targeted: their
        coefficients (``offset_coef``) bias the search, -inf ones forbidding
        the changes they count.
    target_stats : array-like or dict
        The targets of the statistics of the non-offset terms, in order or by name.
    constraints, offset_coef, bipartite
        As in :func:`ergm`.
    seed : int, optional
    only_last : bool
        Return the last network, or (False) each run's.
    control : SanControl, optional
        Settings; keyword arguments (``nsteps=...``, ``maxit=...``) override single ones.

    Returns
    -------
    A graph of the same kind as ``network`` (a list of them for combined
    networks), or a list of them, one per run, without ``only_last``.
    """
    control = dataclasses.replace(control or SanControl(), **control_args)
    model = bind(network, formula, constraints, offset_coef=offset_coef, bipartite=bipartite)
    if any(t.is_offset for t in model.formula) and offset_coef is None:
        raise ValueError("give the offset terms' coefficients with offset_coef=, as ergm's san() needs")
    targeted, _ = _targeted(model)
    target = _target_vector(model, targeted, target_stats)
    result, _ = anneal(model, target, control, np.random.default_rng(seed), only_last)
    if only_last:
        return to_graphs(model.network, result)
    return [to_graphs(model.network, edges) for edges in result]


def target_model(model: BoundModel, target_stats, control: SanControl, rng, formula, constraints,
                 offset_coef, bipartite):
    """The model fitted to target statistics, as ergm's ``target.stats``: on a
    network from simulated annealing towards them, with the targets as the
    observed statistics."""
    from ._model import bind as _bind

    targeted, _ = _targeted(model)
    target = _target_vector(model, targeted, target_stats)
    edges, deviations = anneal(model, target, control, rng)
    if np.any(deviations != 0):
        log.info("SAN did not reach the target statistics exactly (largest deviation %.4g)",
                 np.max(np.abs(deviations)))
    network = dataclasses.replace(model.network, edges=edges, missing=model.network.missing[:0])
    fitted = _bind(network, formula, constraints, offset_coef, fitting=True, bipartite=bipartite)
    full = fitted.observed()
    full[targeted] = target
    object.__setattr__(fitted, "target", full)
    return fitted
