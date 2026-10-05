"""Temporal ERGMs, as R's tergm: the conditional MLE of a series of networks,
and dynamic simulation, one time step after another."""

from __future__ import annotations

import sys

import numpy as np

from . import _core
from ._estimation import Control
from ._model import bind
from ._multi import NetSeries, series_from
from ._network import Network, as_network, to_graph
from .terms import BlockOperator


def tergm(networks, formula, *, estimate: str = "CMLE", times=None, constraints=None,
          offset_coef=None, bipartite=None, init=None, seed=None, eval_loglik: bool = True,
          control: Control | None = None, targets=None, target_stats=None, egmme=None,
          na_impute=None, **control_args):
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
    estimate : {"CMLE", "CMPLE", "EGMME"}
        The conditional MLE (Monte Carlo, or exact for dyad-independent
        models), the conditional maximum pseudo-likelihood estimate, or
        tergm's equilibrium generalized method of moments estimate (EGMME),
        which fits the process to a single network and the durations of its
        ties: the coefficients whose process has, at equilibrium, the
        ``target_stats`` of the statistics of ``targets``.
    targets : str or terms
        For the EGMME, a formula of the statistics to match: ergm terms, and
        statistics of tie ages (``mean.age``, ``edge.ages``,
        ``edges.ageinterval``, ``edgecov.ages``, ``nodefactor.mean.age``).
    target_stats : array-like, optional
        Their values: by default, the network's (necessary for the ages).
    egmme : dict, optional
        Settings of the EGMME's stochastic approximation (see
        ``ergmx._temporal._EGMME_DEFAULTS``): its burn-in, gradient runs,
        gain, subphases and iterations, and the length of each time step's
        chain (``min_steps``...).
    times : list of numbers, optional
        The times the networks were observed (0, 1, 2... by default), for
        ``.Time`` and ``.TimeDelta`` in linear models.
    na_impute : str or list of str, optional
        How to impute the missing dyads of the networks transitioned from, as
        tergm's ``CMLE.NA.impute``: see :func:`ergmx.NetSeries`.
    constraints, offset_coef, bipartite, init, seed, eval_loglik, control
        As in :func:`ergmx.ergm`.

    Returns
    -------
    ErgmFit
        Its :meth:`~ergmx.ErgmFit.simulate` continues the series with
        ``time_slices=``.
    """
    from ._simulate import ergm

    if estimate == "EGMME":
        return _egmme(networks, formula, targets, target_stats, constraints=constraints, init=init,
                      seed=seed, control=egmme)
    if estimate not in ("CMLE", "CMPLE"):
        raise ValueError(f"estimate must be 'CMLE', 'CMPLE' or 'EGMME', not {estimate!r}")
    series = series_from(networks, times, bipartite, na_impute)
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


def _monitor(start, edges, monitor, summary_stats, ages=None) -> dict:
    """The monitor's statistics of each network of a simulation; statistics of
    tie ages start from the starting network's `ages` (Ages; by default its
    ties count as formed at time 0)."""
    from ._durational import Ages
    from .terms import Formula

    plain, durational = _split_targets(monitor)
    columns: dict[str, list] = {}
    if plain:
        for e in edges:
            for name, value in summary_stats(to_graph(start, e), Formula(plain)).items():
                columns.setdefault(name, []).append(value)
    if durational:
        ages = Ages(start.directed, start.edges) if ages is None else ages.copy()
        for e in edges:
            current = ages.step(e)
            for term in durational:
                for name, value in zip(term.names(start), term.value(start, e, current)):
                    columns.setdefault(name, []).append(value)
    return {name: np.array(values) for name, values in columns.items()}


def _check_dynamic(model) -> None:
    for term in model.formula:
        inner = term.term if term.is_offset else term
        if isinstance(inner, BlockOperator) and inner.varies(model.network):
            raise NotImplementedError(f"{inner!r}: simulate_dynamic() has no times to predict "
                                      "time-varying coefficients (lm=) at; continue a fitted series "
                                      "with fit.simulate(time_slices=...)")


def _start_ages(start, ages):
    """The ages of the starting network's ties: 1 (formed at time 0), or
    those of the edge attribute `ages`."""
    from ._durational import Ages

    if ages is None:
        return Ages(start.directed, start.edges)
    ig = sys.modules.get("igraph")
    source = start.source
    if ig is not None and isinstance(source, ig.Graph):
        if ages not in source.es.attributes():
            raise ValueError(f"the network has no edge attribute {ages!r} with its ties' ages")
        values = dict(zip(source.get_edgelist(), source.es[ages]))
    else:
        values = {(u, v): d.get(ages) for u, v, d in source.edges(data=True)}
        index = {node: k for k, node in enumerate(source)}
        values = {(index[u], index[v]): a for (u, v), a in values.items()}
    found = [values.get((int(i), int(j)), values.get((int(j), int(i)))) for i, j in start.edges]
    if any(a is None or not float(a).is_integer() or a < 1 for a in found):
        raise ValueError(f"the ages of the ties ({ages!r}) must be whole numbers, 1 or more")
    return Ages(start.directed, start.edges, np.array(found, dtype=int))


def simulate_dynamic(network, formula, coef, time_slices: int = 1, *, nsim: int = 1,
                     constraints=None, bipartite=None, seed=None, monitor=None,
                     triadic_weight: float | None = None, min_steps: int = 1000,
                     max_steps: int = 100_000, pval: float = 0.5, add: float = 1.0, ages: str | None = None):
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
        ``"edges + mutual"``, and of the ages of its ties (``mean.age``...).
    ages : str, optional
        An edge attribute with the ages of the starting network's ties, for
        the terms and monitors of tie ages. By default they are 1, as if they
        formed at time 0 (tergm counts them as formed long before, unless
        the network has its ``lasttoggle`` attribute).
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
    model = bind(NetSeries(start, start), formula, constraints, dynamic=True)
    _check_dynamic(model)
    if isinstance(coef, dict):
        coef = [coef[name] for name in model.names]
    coef = np.asarray(coef, dtype=float)
    if coef.shape != (model.n_params,):
        raise ValueError(f"coef must have {model.n_params} values, one per parameter: {model.names}")
    seed = int(np.random.default_rng(seed).integers(2**63))
    eta = model.eta(coef)
    start_ages = _start_ages(start, ages)
    try:
        runs = model.core.simulate_series(
            [start.edges] * int(nsim), [float(v) for v in eta], int(time_slices), seed,
            min_steps=int(min_steps), max_steps=int(max_steps), pval=float(pval), add=float(add),
            triadic_weight=model.triadic_weight(triadic_weight), space=model.space,
            ages=[start_ages.rows()] * int(nsim),
        )
    except _core.DensityGuardError as e:  # pragma: no cover - no limit is set
        raise RuntimeError(str(e)) from None
    results = []
    for edges, stats, steps, _ in runs:
        tracked = None
        if monitor is not None:
            tracked = _monitor(start, edges, monitor, summary_stats, start_ages)
        results.append(DynamicSimulation(start, list(edges), np.asarray(stats), list(model.stat_names),
                                         list(steps), tracked))
    return results[0] if nsim == 1 else results


# -- EGMME ---------------------------------------------------------------------------------------


class EgmmeFit:
    """A temporal ERGM fitted by tergm's equilibrium generalized method of
    moments (:func:`ergmx.tergm` with ``estimate="EGMME"``): coefficients
    whose dynamic process has, at equilibrium, the target statistics."""

    def __init__(self, network, formula, model, names, coef, cov, targets, target_stats, simulated,
                 history, converged, settings):
        self.network, self.formula, self._model = network, formula, model
        #: Parameter names, as tergm's (``Form~edges``, ``Persist~edges``).
        self.names = names
        self.params, self.cov = np.asarray(coef, dtype=float), cov
        #: The targets, their values, and their mean over the process at the estimate.
        self.target_names, self.target_stats, self.simulated_targets = targets, target_stats, simulated
        #: The coefficients after each iteration of the stochastic approximation.
        self.history = history
        self.converged = converged
        self._settings = settings

    @property
    def coef(self) -> dict:
        return dict(zip(self.names, self.params.tolist()))

    @property
    def stderr(self) -> dict:
        return dict(zip(self.names, np.sqrt(np.diag(self.cov)).tolist()))

    def save(self, path) -> None:
        """Save the fit to a file, to reload with :func:`ergmx.load_fit`."""
        from ._fit import _save

        _save(self, path)

    def summary(self) -> str:
        from scipy import stats

        se = np.sqrt(np.diag(self.cov))
        z = self.params / se
        p = 2 * stats.norm.sf(np.abs(z))
        width = max(map(len, self.names))
        lines = ["Equilibrium Generalized Method of Moments Results:", "",
                 f"{'':<{width}}  {'Estimate':>9}  {'Std. Error':>10}  {'z value':>8}  {'Pr(>|z|)':>9}"]
        for n, c, s_, zz, pp in zip(self.names, self.params, se, z, p):
            lines.append(f"{n:<{width}}  {c:9.4f}  {s_:10.4f}  {zz:8.3f}  {pp:9.3g}")
        tw = max(map(len, self.target_names))
        lines += ["", "Targets:", "", f"{'':<{tw}}  {'target':>9}  {'simulated':>9}"]
        for n, t, m in zip(self.target_names, self.target_stats, self.simulated_targets):
            lines.append(f"{n:<{tw}}  {t:9.3f}  {m:9.3f}")
        lines += ["", "Converged." if self.converged else
                  "Not converged: the simulated targets are far from the targets; try more iterations."]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def simulate(self, time_slices: int = 1, **options):
        """Simulate the process forward from the observed network: see
        :func:`ergmx.simulate_dynamic`."""
        return simulate_dynamic(self.network, self.formula, self.params, time_slices, **options)


def _split_targets(targets):
    """Ordinary terms and durational statistics of a targets formula."""
    from .terms import as_formula

    terms = as_formula(targets).terms
    durational = [t for t in terms if getattr(t, "durational", False)]
    plain = [t for t in terms if not getattr(t, "durational", False)]
    return plain, durational


class _Process:
    """A network evolving under a temporal model, with the ages of its ties,
    and the target statistics of each time step."""

    def __init__(self, model, start, plain, durational, settings):
        from .terms import Formula

        self.model, self.start = model, start
        self.target_model = bind(start, Formula(plain)) if plain else None
        self.durational = durational
        self.edges = start.edges.copy()
        from ._durational import Ages

        self.ages = Ages(start.directed, self.edges)
        self.settings = settings

    def targets(self, edges, ages) -> np.ndarray:
        parts = []
        if self.target_model is not None:
            parts.append(np.asarray(self.target_model.core.summary(edges), dtype=float))
        for term in self.durational:
            parts.append(term.value(self.start, edges, ages))
        return np.concatenate(parts)

    def run(self, theta, steps: int, seed: int, *, keep: bool = True, state=None):
        """Advance `steps` time steps at `theta` (from `state`, or the current
        one); returns the targets of each step and the final state."""
        edges, ages = state if state is not None else (self.edges, self.ages)
        ages = ages.copy()
        s = self.settings
        eta = self.model.eta(theta)
        runs = self.model.core.simulate_series(
            [edges], [float(v) for v in eta], int(steps), int(seed), min_steps=s["min_steps"],
            max_steps=s["max_steps"], pval=s["pval"], add=s["add"],
            triadic_weight=self.model.triadic_weight(s["triadic_weight"]), space=self.model.space,
            ages=[ages.rows()])
        networks = runs[0][0]
        values = np.array([self.targets(e, ages.step(e)) for e in networks])
        final = (networks[-1], ages)
        if keep:
            self.edges, self.ages = final
        return values, final


def _initial(model, start, plain, durational, target_stats, names) -> np.ndarray:
    """Starting values: the cross-sectional MPLE of Form()'s terms, less
    log(duration) for its edges, and Persist()'s (Diss()'s) edges for that mean
    duration, as EpiModel's approximation; 0 otherwise."""
    from ._durational import MeanAge
    from ._simulate import ergm
    from .terms import BlockOperator

    theta = np.zeros(model.n_params)
    offset = len(plain) and sum(len(t.names(start)) for t in plain)
    # The density the targets ask for, when they have edges (the network may be far from them).
    plain_names = [n for t in plain for n in t.names(start)]
    dyads = start.n * (start.n - 1) / (1 if start.directed else 2)
    density = float(target_stats[plain_names.index("edges")]) / dyads if "edges" in plain_names else None
    duration = None
    k = offset
    for term in durational:
        n = len(term.names(start))
        if isinstance(term, MeanAge) and not term.log:
            duration = float(target_stats[k])
        k += n
    for term, cols in model.term_columns():
        inner = term.term if term.is_offset else term
        if not isinstance(inner, BlockOperator):
            continue
        local = [n.split("~", 1)[1] for n in (names[c] for c in cols)]
        if inner.op == "Form":
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cross = ergm(start, inner.formula, estimate="MPLE")
                # Only moderate coefficients: the network may be far from the
                # targets (empty, for one), and its MPLE infinite.
                values = {n: v for n, v in zip(cross.names, cross.params) if np.isfinite(v) and abs(v) < 10}
            except Exception:  # pragma: no cover - fall back to 0
                values = {}
            for c, name in zip(cols, local):
                theta[c] = values.get(name, 0.0)
                if name == "edges" and density is not None and 0 < density < 1:
                    theta[c] = np.log(density / (1 - density))
                if name == "edges" and duration is not None and duration > 1:
                    theta[c] -= np.log(duration)
        elif inner.op in ("Persist", "Diss") and duration is not None and duration > 1:
            for c, name in zip(cols, local):
                if name == "edges":
                    persist = np.log(duration - 1.0)  # logit(1 - 1/duration)
                    theta[c] = persist if inner.op == "Persist" else -persist
    return theta


def _batch_cov(values: np.ndarray, batches: int = 10) -> np.ndarray:
    """Covariance of the mean of an autocorrelated series, by batch means."""
    n = len(values) // batches * batches
    means = values[:n].reshape(batches, -1, values.shape[1]).mean(axis=1)
    return np.atleast_2d(np.cov(means, rowvar=False)) / batches


def _egmme(network, formula, targets, target_stats, *, constraints, init, seed, control):
    from .terms import as_formula

    if isinstance(network, (list, tuple)):
        raise ValueError("the EGMME fits a single network: give one network, not a series")
    start = as_network(network)
    if start.combined:
        raise ValueError("the EGMME fits a single network: give one network, not a series")
    model = bind(NetSeries(start, start), formula, constraints, dynamic=True)
    _check_dynamic(model)
    if targets is None:
        raise ValueError("the EGMME needs targets=, a formula of the statistics to match")
    plain, durational = _split_targets(targets)
    target_names = [n for t in [*plain, *durational] for n in t.names(start)]
    if target_stats is None:
        if durational:
            raise ValueError(f"give target_stats=: the network has no observed values of "
                             f"{[n for t in durational for n in t.names(start)]}, its ties' ages")
        target_stats = np.asarray(bind(start, as_formula(targets)).observed(), dtype=float)
    target_stats = np.asarray(target_stats, dtype=float)
    if target_stats.shape != (len(target_names),):
        raise ValueError(f"target_stats must have {len(target_names)} values: {target_names}")
    free = model.free
    p = int(free.sum())
    if len(target_names) < p:
        raise ValueError(f"the EGMME needs at least as many targets ({len(target_names)}) as "
                         f"parameters ({p})")
    names = [n.replace("(1)~", "~", 1) for n in model.names]
    c = {**_EGMME_DEFAULTS, **(control or {})}
    rng = np.random.default_rng(seed)
    draw = lambda: int(rng.integers(2**63))  # noqa: E731
    process = _Process(model, start, plain, durational, c)
    theta = np.asarray(init, dtype=float) if init is not None else \
        _initial(model, start, plain, durational, target_stats, model.names)
    theta = np.where(model.fixed, model.fixed_values, theta)

    def gradient(theta):
        """d E[targets] / d theta, by central differences with common random numbers."""
        state = (process.edges, process.ages)
        g = np.zeros((len(target_names), p))
        for k, j in enumerate(np.flatnonzero(free)):
            s = draw()
            means = []
            for h in (c["step"], -c["step"]):
                shifted = theta.copy()
                shifted[j] += h
                values, _ = process.run(shifted, c["burn"] + c["gradient_steps"], s, keep=False, state=state)
                means.append(values[c["burn"]:].mean(axis=0))
            g[:, k] = (means[0] - means[1]) / (2 * c["step"])
        return g

    def weights(values):
        cov = np.atleast_2d(np.cov(values, rowvar=False))
        return np.linalg.pinv(cov + 1e-9 * np.eye(len(cov)))

    # Phase 1: burn in, the gradient and the weights.
    values, _ = process.run(theta, c["burnin"], draw())
    W = weights(values[len(values) // 2:])
    D = gradient(theta)
    history = [theta.copy()]
    # Phase 2: stochastic approximation, with Polyak averaging in each subphase.
    gain = c["gain"]
    for subphase in range(c["subphases"]):
        visited = []
        for _ in range(int(c["iterations"] * 1.5 ** subphase)):
            values, _ = process.run(theta, c["run_length"], draw())
            gap = values.mean(axis=0) - target_stats
            step = np.linalg.lstsq(D.T @ W @ D, D.T @ W @ gap, rcond=None)[0]
            theta[free] -= gain * step
            visited.append(theta.copy())
            history.append(theta.copy())
        theta = np.mean(visited, axis=0)
        gain /= 2
        if subphase == c["subphases"] // 2:
            D = gradient(theta)
    # Phase 3: the targets at the estimate, and the standard errors.
    values, _ = process.run(theta, c["final_steps"], draw())
    simulated = values.mean(axis=0)
    D = gradient(theta)
    sigma = np.atleast_2d(np.cov(values, rowvar=False))
    W = np.linalg.pinv(sigma)
    bread = np.linalg.pinv(D.T @ W @ D)
    cov = np.zeros((model.n_params, model.n_params))
    cov[np.ix_(free, free)] = bread
    # The coefficients' shift that would move the simulated targets to theirs,
    # in standard errors: converged if under 0.5 (the final run's own noise,
    # a few thousand time steps of an autocorrelated process, shifts them by
    # up to a few tenths).
    shift = bread @ D.T @ W @ (simulated - target_stats) / np.sqrt(np.maximum(np.diag(bread), 1e-300))
    converged = bool(np.all(np.abs(shift) < 0.5))
    fit = EgmmeFit(network, formula, model, names, theta, cov, target_names, target_stats, simulated,
                   np.array(history), converged, c)
    fit._shift = shift
    return fit


#: Settings of the EGMME's stochastic approximation (``control=`` of tergm()).
_EGMME_DEFAULTS = {
    "burnin": 200, "burn": 30, "gradient_steps": 1500, "step": 0.3, "gain": 0.3, "subphases": 4,
    "iterations": 30, "run_length": 10, "final_steps": 2500, "min_steps": 1000, "max_steps": 100_000,
    "pval": 0.5, "add": 1.0, "triadic_weight": None,
}


def _varying(model) -> bool:
    """Whether some operator's coefficients vary between transitions (lm=,
    subset=, offset=)."""
    return any(isinstance(t.term if t.is_offset else t, BlockOperator) and
               (t.term if t.is_offset else t).varies(model.network) for t in model.formula)


def simulate_varying(model, params, time_slices: int, *, nsim: int = 1, seed=None, monitor=None,
                     triadic_weight=None, min_steps: int = 1000, max_steps: int = 100_000,
                     pval: float = 0.5, add: float = 1.0):
    """Continue a fitted series whose coefficients vary over time (lm=) from
    its last network: transition k has the time .Time + k .TimeDelta and
    .TimeID + k of the last one, and its coefficients are the linear models'
    predictions for it. Returns DynamicSimulation objects."""
    from ._simulate import summary_stats
    from .terms import Formula, Offset

    network = model.network
    last = network.blocks[-1]
    start = last.network
    attributes = [b.attributes for b in network.blocks]
    step_attributes = []
    for k in range(1, int(time_slices) + 1):
        a = dict(last.attributes)
        a[".Time"] = last.attributes[".Time"] + k * last.attributes[".TimeDelta"]
        a[".TimeID"] = last.attributes[".TimeID"] + k
        step_attributes.append(a)
    # The model of one transition with each operator's intercept only, and
    # each term's coefficients at every future time.
    terms, coefficients = [], []
    for term, cols in model.term_columns():
        inner = term.term if term.is_offset else term
        beta = np.asarray(params, dtype=float)[cols]
        if not isinstance(inner, BlockOperator):
            terms.append(term)
            coefficients.append(np.tile(beta, (int(time_slices), 1)))
            continue
        _, columns = inner._design(network)
        rows = []
        for a in step_attributes:
            try:
                kept, x, future, offsets = inner._frame([*attributes, a])
            except ValueError as e:
                raise ValueError(f"{inner!r}: can't predict the coefficients of a future transition: {e}") from None
            if future != columns:
                raise ValueError(f"{inner!r}: a future transition has new levels in its linear model "
                                 f"({future} instead of {columns}); its coefficients can't be predicted")
            shift = 0.0 if offsets is None else offsets[-1]
            rows.append(beta.reshape(-1, len(columns)) @ x[-1] + shift * kept[-1])
        plain = BlockOperator(inner.op, inner.formula)
        terms.append(Offset(plain) if term.is_offset else plain)
        coefficients.append(np.array(rows))
    step_model = bind(NetSeries(start, start), Formula(terms), model.constraints,
                      offset_coef=None)
    theta = np.hstack(coefficients)  # time steps x parameters of step_model
    rng = np.random.default_rng(seed)
    states = [start.edges] * int(nsim)
    collected = [([], [], []) for _ in range(int(nsim))]
    for k in range(int(time_slices)):
        eta = step_model.eta(theta[k])
        runs = step_model.core.simulate_series(
            states, [float(v) for v in eta], 1, int(rng.integers(2**63)), min_steps=int(min_steps),
            max_steps=int(max_steps), pval=float(pval), add=float(add),
            triadic_weight=step_model.triadic_weight(triadic_weight), space=step_model.space)
        states = []
        for (edges, stats, steps, _), (all_edges, all_stats, all_steps) in zip(runs, collected):
            all_edges.append(edges[0])
            all_stats.append(np.asarray(stats)[0])
            all_steps.append(steps[0])
            states.append(edges[0])
    results = []
    for all_edges, all_stats, all_steps in collected:
        tracked = _monitor(start, all_edges, monitor, summary_stats) if monitor is not None else None
        results.append(DynamicSimulation(start, all_edges, np.array(all_stats), list(step_model.stat_names),
                                         all_steps, tracked))
    return results[0] if nsim == 1 else results
