"""Temporal ERGMs, as R's tergm: the conditional MLE of a series of networks,
and dynamic simulation, one time step after another."""

from __future__ import annotations

import numpy as np

from . import _core
from ._estimation import Control
from ._model import bind
from ._multi import NetSeries, series_from
from ._network import Network, as_network, to_graph
from .terms import BlockOperator


def tergm(networks, formula, *, estimate: str = "CMLE", times=None, constraints=None,
          offset_coef=None, bipartite=None, init=None, seed=None, eval_loglik: bool = True,
          control: Control | None = None, **control_args):
    """Fit a temporal ERGM to a series of networks by conditional maximum
    likelihood, as R's ``tergm(..., estimate="CMLE")``.

    Each transition, from one network of the series to the next, is modeled
    conditionally on the network before it, with tergm's operators:
    ``Form(~terms)`` models the ties that form (its terms are those of the
    union of the previous and the current network), ``Persist(~terms)`` those
    that persist (the intersection; ``Diss()`` is the same with the signs of
    the coefficients reversed), ``Cross(~terms)`` the current network and
    ``Change(~terms)`` the dyads that changed. A model with only Form() and
    Persist() (or Diss()) is *separable* (a STERGM): formation and
    persistence are independent given the previous network. Every transition
    has the same coefficients, unless their linear models (``lm=``) make them
    depend on the time (see :func:`ergmx.NetSeries`).

    Parameters
    ----------
    networks : list of igraph.Graph or networkx.Graph, or NetSeries
        The networks, in time order, on the same vertices.
    formula : str or terms
        For example ``"Form(~edges + mutual + gwesp(0.5, fixed=TRUE)) + Persist(~edges + mutual)"``.
    estimate : {"CMLE", "CMPLE"}
        The conditional MLE (Monte Carlo, or exact for dyad-independent
        models), or the conditional maximum pseudo-likelihood estimate.
        (tergm's EGMME, which fits a single network with durations, is not
        supported.)
    times : list of numbers, optional
        The times the networks were observed (0, 1, 2... by default), for
        ``.Time`` and ``.TimeDelta`` in linear models.
    constraints, offset_coef, bipartite, init, seed, eval_loglik, control
        As in :func:`ergmx.ergm`.

    Returns
    -------
    ErgmFit
        Its :meth:`~ergmx.ErgmFit.simulate` continues the series with
        ``time_slices=``.
    """
    from ._simulate import ergm

    if estimate not in ("CMLE", "CMPLE"):
        raise ValueError(f"estimate must be 'CMLE' or 'CMPLE', not {estimate!r} (EGMME is not "
                         "supported)")
    series = series_from(networks, times, bipartite)
    return ergm(series, formula, constraints=constraints, offset_coef=offset_coef,
                estimate="MLE" if estimate == "CMLE" else "MPLE", init=init, seed=seed,
                eval_loglik=eval_loglik, control=control, **control_args)


class DynamicSimulation:
    """A network simulated over time by :func:`ergmx.simulate_dynamic`: the
    starting network and the network after each time step."""

    def __init__(self, start: Network, edges: list[np.ndarray], stats: np.ndarray,
                 stat_names: list[str], steps: list[int], monitor: dict | None = None):
        self._start, self._edges = start, edges
        #: The model's statistics after each time step (time steps x
        #: statistics): each transition's, given the network before it.
        self.stats = stats
        self.stat_names = stat_names
        #: The MCMC proposals each time step took.
        self.steps = np.asarray(steps)
        #: The statistics of the ``monitor`` formula of each network, by name.
        self.monitor = monitor or {}

    def __len__(self) -> int:
        return len(self._edges)

    @property
    def start(self):
        """The starting network, as a graph."""
        return to_graph(self._start, self._start.edges)

    @property
    def networks(self) -> list:
        """The network after each time step, as graphs like the starting one."""
        return [to_graph(self._start, e) for e in self._edges]

    def _sets(self) -> list[set]:
        return [set(map(tuple, e.tolist())) for e in [self._start.edges, *self._edges]]

    @property
    def edges(self) -> np.ndarray:
        """The number of edges at the start and after each time step."""
        return np.array([len(e) for e in [self._start.edges, *self._edges]])

    @property
    def formed(self) -> np.ndarray:
        """The ties that formed at each time step."""
        sets = self._sets()
        return np.array([len(b - a) for a, b in zip(sets, sets[1:])])

    @property
    def dissolved(self) -> np.ndarray:
        """The ties that dissolved at each time step."""
        sets = self._sets()
        return np.array([len(a - b) for a, b in zip(sets, sets[1:])])

    def durations(self) -> tuple[np.ndarray, np.ndarray]:
        """The durations, in time steps, of the ties that dissolved during the
        simulation (formed during it, or present at the start), and of the
        ties still present at the end (censored: their duration so far).
        A tie present at the start counts from time 0."""
        sets = self._sets()
        since = dict.fromkeys(sets[0], 0)
        finished = []
        for t, current in enumerate(sets[1:], start=1):
            for tie in list(since):
                if tie not in current:
                    finished.append(t - since.pop(tie))
            for tie in current - since.keys():
                since[tie] = t
        end = len(sets) - 1
        return np.array(finished, dtype=int), np.array([end - s for s in since.values()], dtype=int)

    def __repr__(self) -> str:
        lines = [f"DynamicSimulation: {len(self)} time steps from a network of {self._start.n} "
                 "vertices", "", f"{'time':>4}  {'edges':>6}  {'formed':>6}  {'dissolved':>9}  "
                 f"{'MCMC steps':>10}"]
        edges, formed, dissolved = self.edges, self.formed, self.dissolved
        lines.append(f"{0:>4}  {edges[0]:>6}  {'':>6}  {'':>9}  {'':>10}")
        rows = range(len(self)) if len(self) <= 12 else [*range(5), None, *range(len(self) - 5, len(self))]
        for t in rows:
            if t is None:
                lines.append(f"{'...':>4}")
                continue
            lines.append(f"{t + 1:>4}  {edges[t + 1]:>6}  {formed[t]:>6}  {dissolved[t]:>9}  "
                         f"{self.steps[t]:>10}")
        return "\n".join(lines)


def _check_dynamic(model) -> None:
    for term in model.formula:
        inner = term.term if term.is_offset else term
        if isinstance(inner, BlockOperator) and inner._design(model.network)[1] != ["1"]:
            raise NotImplementedError(f"{inner!r}: dynamic simulation with time-varying linear "
                                      "models (lm=) is not supported")


def simulate_dynamic(network, formula, coef, time_slices: int = 1, *, nsim: int = 1,
                     constraints=None, bipartite=None, seed=None, monitor=None,
                     triadic_weight: float | None = None, min_steps: int = 1000,
                     max_steps: int = 100_000, pval: float = 0.5, add: float = 1.0):
    """Simulate a network forward in time from a temporal ERGM, as R's tergm
    does with ``simulate(..., dynamic=TRUE)``.

    At each of ``time_slices`` time steps, the next network is drawn from the
    model conditional on the current one: the model of each transition of
    :func:`ergmx.tergm`, with Form(), Persist(), Diss(), Cross() and Change()
    (terms outside them describe the current network, as Cross()). Each time
    step's Markov chain starts from the current network and runs, as in
    tergm, until the number of dyads that differ from it stops growing:
    after at least ``min_steps`` proposals, once a z-test no longer finds the
    (exponentially weighted) average change in that number positive, with
    p-value above ``pval``, the chain runs ``add`` times as many proposals
    again; at most ``max_steps`` in all.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
        The network at time 0.
    formula : str or terms
        A model with tergm's operators, for example
        ``"Form(~edges + gwesp(0.5, fixed=TRUE)) + Persist(~edges)"``.
    coef : array-like or dict
        Coefficients, in the order of the formula's parameters or by name
        (``"Form(1)~edges"``...), as :func:`ergmx.tergm` estimates them.
    time_slices : int
        Number of time steps.
    nsim : int
        Number of independent simulations, run in parallel.
    monitor : str or terms, optional
        A formula whose statistics are computed on each network, such as
        ``"edges + mutual"``.
    constraints, bipartite, seed, triadic_weight
        As in :func:`ergmx.simulate`.
    min_steps, max_steps, pval, add
        The length of each time step's chain, as tergm's MCMC.burnin.min,
        MCMC.burnin.max, MCMC.burnin.pval and MCMC.burnin.add. Set
        ``min_steps`` and ``max_steps`` equal for a fixed number.

    Returns
    -------
    DynamicSimulation, or a list of nsim of them
    """
    from ._simulate import summary_stats

    start = as_network(network, bipartite)
    if start.combined:
        raise ValueError("simulate_dynamic() starts from a single network")
    if int(time_slices) < 1 or int(nsim) < 1:
        raise ValueError("need time_slices >= 1 and nsim >= 1")
    model = bind(NetSeries(start, start), formula, constraints)
    _check_dynamic(model)
    if isinstance(coef, dict):
        coef = [coef[name] for name in model.names]
    coef = np.asarray(coef, dtype=float)
    if coef.shape != (model.n_params,):
        raise ValueError(f"coef must have {model.n_params} values, one per parameter: {model.names}")
    seed = int(np.random.default_rng(seed).integers(2**63))
    eta = model.eta(coef)
    try:
        runs = model.core.simulate_series(
            [start.edges] * int(nsim), [float(v) for v in eta], int(time_slices), seed,
            min_steps=int(min_steps), max_steps=int(max_steps), pval=float(pval), add=float(add),
            triadic_weight=model.triadic_weight(triadic_weight), space=model.space,
        )
    except _core.DensityGuardError as e:  # pragma: no cover - no limit is set
        raise RuntimeError(str(e)) from None
    results = []
    for edges, stats, steps in runs:
        tracked = None
        if monitor is not None:
            rows = [summary_stats(to_graph(start, e), monitor) for e in edges]
            tracked = {name: np.array([r[name] for r in rows]) for name in rows[0]}
        results.append(DynamicSimulation(start, list(edges), np.asarray(stats), list(model.stat_names),
                                         list(steps), tracked))
    return results[0] if nsim == 1 else results
