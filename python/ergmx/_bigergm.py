"""ERGMs of large networks with local dependence, as R's bigergm (Schweinberger
and Handcock 2015; Babkin et al. 2020): the
vertices fall in blocks, the dyads within a block follow an ERGM with
dependence (triangles, shared partners...), and those between blocks are
independent. bigergm() finds the blocks with the MM algorithm of a
stochastic block model's variational approximation (optionally with the
covariates of the model's nodematch terms), then fits the dyad-independent
terms to the dyads between blocks and the whole model to those within."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from . import _core
from ._fit import ErgmFit
from ._network import Network, as_network, to_graph
from .terms import Formula, NodeMatch, as_formula

_MIN_TAU = 1e-4
_MIN_PI = 1e-4


# -- The MM algorithm ---------------------------------------------------------------------------


def _clip(p: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(p) & (p >= _MIN_PI), p, _MIN_PI)


def _pairs_weight(tau: np.ndarray) -> np.ndarray:
    """sum over ordered pairs i != j of tau_ik tau_jl (k x l)."""
    return tau.T @ (tau.sum(axis=0)[None, :] - tau)


def _quadratic(tau, g, directed: bool):
    """The surrogate's quadratic coefficients and the lower bound's part of
    the dyads, without covariates (bigergm's compute_quadratic_term)."""
    pi1 = _clip((tau.T @ (g @ tau)) / _pairs_weight(tau))
    pi0 = _clip(1 - pi1)
    log0, log1 = np.log(pi0), np.log(pi1 / pi0)
    rest = tau.sum(axis=0)[None, :] - tau
    gt = g @ tau
    if directed:
        gtt = g.T @ tau
        a = rest @ (log0 + log0.T).T + gtt @ log1 + gt @ log1.T
        lb = np.sum((tau.T @ rest) * log0) + np.sum((tau.T @ gt) * log1)
    else:
        a = rest @ log0.T + gt @ log1.T
        lb = (np.sum((tau.T @ rest) * log0) + np.sum((tau.T @ gt) * log1)) / 2
    return a, lb


def _features_matrices(g, features: list) -> list:
    """bigergm's elementwise products of the adjacency matrix and the
    covariates' match matrices: first the pairs that match on no covariate
    (less 1, by inclusion and exclusion), then, for each combination of the
    tie (bit 0) and the covariates' matches (bits 1...), the pairs with
    exactly that combination."""
    n = g.shape[0]
    denominator = sparse.csr_matrix((n, n))
    m = len(features)
    for s in range(1, 2**m):
        bits = [(s >> t) & 1 for t in range(m)]
        x = None
        for t, bit in enumerate(bits):
            if bit:
                x = features[t] if x is None else x.multiply(features[t])
        denominator = denominator + (-1) ** sum(bits) * x
    out = [sparse.csr_matrix(denominator)]
    mats = [g, *features]
    for s in range(1, 2 ** (m + 1)):
        bits = [(s >> t) & 1 for t in range(m + 1)]
        x = None
        for t, bit in enumerate(bits):
            if bit:
                x = mats[t] if x is None else x.multiply(mats[t])
        for t, bit in enumerate(bits):
            if not bit:
                x = x - x.multiply(mats[t])
        x = sparse.csr_matrix(x)
        x.eliminate_zeros()
        out.append(x)
    return out


def _quadratic_features(tau, mats: list, directed: bool):
    """The surrogate's quadratic coefficients with covariates (bigergm's
    compute_quadratic_term_with_features)."""
    weight = _pairs_weight(tau) + tau.T @ (mats[0] @ tau)
    weight = np.where(weight < 1e-10, 1.0, weight)
    s0 = mats[1]
    pi_d1x0 = _clip((tau.T @ (s0 @ tau)) / weight)
    pi_d0x0 = _clip(1 - pi_d1x0)
    log00 = np.log(pi_d0x0)
    rest = tau.sum(axis=0)[None, :] - tau
    log10 = np.log(pi_d1x0 / pi_d0x0)
    half = 1.0 if directed else 0.5

    def spread(x, log):
        """x tau log' (and x' tau log, if directed): the coefficients of x's pairs."""
        out = x @ tau @ log.T
        return out + x.T @ tau @ log if directed else out

    a = rest @ ((log00 + log00.T) if directed else log00).T + spread(s0, log10)
    lb = half * (np.sum((tau.T @ rest) * log00) + np.sum((tau.T @ (s0 @ tau)) * log10))
    for s in range(1, len(mats) // 2):
        d0x, d1x = mats[2 * s], mats[2 * s + 1]
        denominator = tau.T @ ((d0x + d1x) @ tau)
        numerator = tau.T @ (d1x @ tau)
        pi_d1 = _clip(numerator / denominator)
        pi_d0 = _clip(1 - pi_d1)
        log_d0, log_d1 = np.log(pi_d0 / pi_d0x0), np.log(pi_d1 / pi_d0x0)
        a = a + spread(d0x, log_d0) + spread(d1x, log_d1)
        lb += half * (np.sum((denominator - numerator) * log_d0) + np.sum(numerator * log_d1))
    return a, lb


def _normalize(tau: np.ndarray) -> np.ndarray:
    """bigergm's normalizeTau: rows summing to 1, with no value below the minimum."""
    tau = tau / tau.sum(axis=1, keepdims=True)
    low = tau < _MIN_TAU
    if low.any():
        tau = np.where(low, _MIN_TAU, tau)
        rows = low.any(axis=1)
        tau[rows] /= tau[rows].sum(axis=1, keepdims=True)
    return tau


def mm(g, tau: np.ndarray, directed: bool, mats: list | None = None, max_steps: int = 100, tol: float = 1e-4):
    """bigergm's MM iterations from the posterior membership probabilities
    `tau`: the final ones and the lower bounds of the likelihood."""
    tau = np.array(tau, dtype=float)
    gamma = tau.mean(axis=0)
    lower = []
    for step in range(max_steps):
        a, lb = _quadratic(tau, g, directed) if mats is None else _quadratic_features(tau, mats, directed)
        a = (1 - np.minimum(a, 0) / 2) / tau
        lb += np.sum(tau * (np.log(gamma)[None, :] - np.log(tau)))
        s = 1 + np.log(gamma)[None, :] - np.log(tau)
        tau = _normalize(_core.mm_solve_qp(np.ascontiguousarray(a), np.ascontiguousarray(s),
                                           np.ascontiguousarray(tau), _MIN_TAU))
        gamma = tau.mean(axis=0)
        lower.append(lb)
        if step > 0 and abs(lower[-1] - lower[-2]) / abs(lower[-2]) < tol:
            break
    return tau, lower


def _first_appearance(labels) -> np.ndarray:
    """Labels 1, 2... in order of first appearance, as bigergm's relabel()."""
    order = {}
    return np.array([order.setdefault(v, len(order) + 1) for v in labels])


def _fix_small_blocks(blocks: np.ndarray, g, k: int, rng, min_size: int = 2) -> np.ndarray:
    """bigergm's check of the clusters: a block with fewer than `min_size`
    vertices takes, with the largest block, two k-means clusters of the
    vertices' leading eigenvectors."""
    from scipy.cluster.vq import kmeans2
    from scipy.sparse.linalg import eigsh

    blocks = blocks.copy()
    sizes = np.bincount(blocks, minlength=k + 1)[1:]
    if np.all(sizes >= min_size):
        return blocks
    n = g.shape[0]
    sym = (g + g.T).astype(bool).astype(float)
    vectors = eigsh(sym, k=min(int(np.ceil(np.sqrt(n))), n - 1), which="LA")[1]
    for _ in range(100):
        sizes = np.bincount(blocks, minlength=k + 1)[1:]
        bad = np.flatnonzero(sizes < min_size) + 1
        if not len(bad):
            return blocks
        donor = int(np.argmax(sizes)) + 1
        chosen = np.isin(blocks, [bad[0], donor])
        best = None
        for _ in range(20):  # k-means from several starts, as bigergm's (100, Hartigan-Wong)
            centers, labels = kmeans2(vectors[chosen], 2, minit="++", seed=rng)
            spread = np.sum((vectors[chosen] - centers[labels]) ** 2)
            if best is None or spread < best[0]:
                best = (spread, labels)
        blocks[np.flatnonzero(chosen)] = np.where(best[1] == 0, bad[0], donor)
    raise RuntimeError("bigergm(): removing the blocks with too few vertices failed; the number of blocks may be "
                       "too high")


def _initial(network, g, k: int, initialization, seed) -> np.ndarray:
    """The starting blocks (1 to k; 0 for none): given, or infomap's,
    walktrap's or random."""
    import random

    if not isinstance(initialization, str):
        start = np.asarray(initialization)
        if len(start) != network.n:
            raise ValueError("bigergm(initialization=): one block per vertex")
        return _first_appearance(start.tolist())
    if initialization == "random":
        return np.random.default_rng(seed).integers(1, k + 1, network.n)
    try:
        import igraph as ig
    except ImportError:  # pragma: no cover
        raise ImportError(f"bigergm(initialization={initialization!r}) needs igraph") from None
    graph = ig.Graph(n=network.n, edges=network.edges.tolist(), directed=network.directed)
    state = random.getstate()
    random.seed(seed)
    try:
        if initialization == "infomap":
            membership = graph.community_infomap().membership
        elif initialization == "walktrap":
            membership = graph.as_undirected().community_walktrap().as_clustering().membership
        else:
            raise ValueError("bigergm(initialization=): 'infomap', 'walktrap', 'random' or one block per vertex")
    finally:
        random.setstate(state)
    membership = np.asarray(membership) + 1
    return np.where(membership <= k, membership, 0)  # as bigergm, the communities beyond k start undecided


# -- bigergm() ----------------------------------------------------------------------------------


def _adjacency(network: Network):
    n, e = network.n, network.edges.astype(np.int64)
    g = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return g if network.directed else (g + g.T).tocsr()


def _match_matrix(values) -> sparse.csr_matrix:
    """The pairs i != j with the same (known) value."""
    n = len(values)
    rows, cols = [], []
    groups: dict = {}
    for i, v in enumerate(values):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            groups.setdefault(v, []).append(i)
    for members in groups.values():
        m = np.array(members)
        r, c = np.meshgrid(m, m)
        keep = r != c
        rows.append(r[keep])
        cols.append(c[keep])
    if not rows:
        return sparse.csr_matrix((n, n))
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))


def _with_blocks(network: Network, blocks: np.ndarray, edges=None) -> Network:
    attributes = {**network.attributes, "block": [int(b) for b in blocks]}
    return Network(network.n, network.directed, network.edges if edges is None else edges, attributes, network.source,
                   network.graph_attributes, network.missing)


def bigergm(network, formula, n_blocks: int | None = None, *, blocks=None, initialization="infomap",
            clustering_with_features: bool = True, add_intercepts: bool = False, method_within: str = "MPLE",
            n_mm_steps: int = 100, tol: float = 1e-4, weight_for_initialization: float = 1000.0, seed=None,
            estimate_parameters: bool = True, **options) -> BigErgmFit:
    """Fit an ERGM with local dependence to a large network, as R's
    ``bigergm()``.

    Parameters
    ----------
    network, formula
        The network and the model: its dyad-independent terms are fitted to
        the dyads between blocks, all of them to the dyads within blocks.
    n_blocks : int
        The number of blocks to find (with ``blocks``, theirs).
    blocks : array, optional
        The blocks, if known: no clustering.
    initialization : {"infomap", "walktrap", "random"} or array
        The starting blocks of the MM algorithm: igraph's infomap or walktrap
        communities (those beyond ``n_blocks`` start undecided), random ones,
        or given.
    clustering_with_features : bool
        Whether the clustering uses the covariates of the model's
        ``nodematch`` terms, as bigergm.
    add_intercepts : bool
        Whether each pair of blocks (between) and each block (within) gets its
        own intercept (and, with ``clustering_with_features``, its own
        covariate effects). ergmx leaves out the edges terms they make
        redundant (bigergm's are NA).
    method_within : {"MPLE", "MLE"}
        The within-block model's estimate: the pseudo-likelihood's, as
        bigergm's default, or the Monte Carlo MLE.
    n_mm_steps, tol
        The MM algorithm's iterations, at most, and its convergence: the
        relative change of the likelihood's lower bound.
    weight_for_initialization : float
        How much the starting blocks weigh in the starting membership
        probabilities.
    seed
        For the communities, the k-means of blocks too small and the MCMC.
    options
        Passed to :func:`ergmx.ergm` for the within-block model.
    """
    from . import ergm

    net = as_network(network, options.pop("bipartite", None))
    if net.combined or net.bipartite:
        raise ValueError("bigergm() takes a single one-mode network")
    terms = list(as_formula(formula))
    if any(t.is_offset for t in terms):
        raise ValueError("bigergm(): offset() terms are not supported")
    method_within = {"ML": "MLE"}.get(method_within.upper(), method_within.upper())
    if method_within not in ("MPLE", "MLE"):
        raise ValueError("bigergm(method_within=): 'MPLE' or 'MLE'")
    nodematch = [t for t in terms if isinstance(t, NodeMatch)]
    if clustering_with_features and not nodematch:
        clustering_with_features = False
    g = _adjacency(net)
    rng = np.random.default_rng(seed)
    mm_info = None
    if blocks is not None:
        found = _first_appearance(list(blocks))
        n_blocks = int(found.max())
    else:
        if n_blocks is None:
            raise ValueError("bigergm(): give n_blocks, or the blocks")
        start = _initial(net, g, int(n_blocks), initialization, None if seed is None else int(rng.integers(2**31)))
        tau = np.ones((net.n, n_blocks))
        known = start > 0
        tau[np.flatnonzero(known), start[known] - 1] = weight_for_initialization
        tau /= tau.sum(axis=1, keepdims=True)
        mats = None
        if clustering_with_features:
            names = list(dict.fromkeys(t.attr for t in nodematch))
            features = [_match_matrix(net.attribute(a)) for a in names]
            mats = _features_matrices(g, features)
        tau, lower = mm(g, tau, net.directed, mats, n_mm_steps, tol)
        found = np.argmax(tau, axis=1) + 1
        found = _fix_small_blocks(found, g, n_blocks, rng)
        found = _first_appearance(found.tolist())
        tau = np.where(tau <= 1e-6, 0.0, tau)
        mm_info = {"lower_bound": lower, "alpha": tau, "initial": start}
    between_fit = within_fit = None
    found = np.asarray(found)
    if estimate_parameters:
        between_fit, within_fit = _estimate(net, terms, found, add_intercepts, clustering_with_features,
                                            method_within, seed, options, ergm)
    return BigErgmFit(net, Formula(terms), found, between_fit, within_fit, mm_info, add_intercepts,
                      clustering_with_features)


def _between_mix(k: int, directed: bool, inner: str | None = None) -> str:
    """ergm's nodemix('block') cells of the pairs of different blocks (with
    a dyad-independent term, their interactions), named as bigergm's:
    mix.block.1_2..."""
    cells = [(a, b) for b in range(1, k + 1) for a in range(1, (k if directed else b) + 1)]  # ergm's order
    diagonal = [c + 1 for c, (a, b) in enumerate(cells) if a == b]
    names = [f"mix.block.{a}_{b}" for a, b in cells if a != b]
    mix = f"nodemix('block', levels2 = -c({', '.join(map(str, diagonal))}))"
    if inner is not None:
        mix, names = f"{inner[0]}:{mix}", [f"{inner[1]}:{n}" for n in names]
    return f"Label(~{mix}, c({', '.join(repr(n) for n in names)}), pos = 'replace')"


def _estimate(net, terms, blocks, add_intercepts, features, method_within, seed, options, ergm):
    k = int(blocks.max())
    independent = [t for t in terms if t.dyad_independent]
    kept = [t for t in independent if t.r_call() != "edges"]  # the intercepts make edges redundant
    if add_intercepts:
        between = [_between_mix(k, net.directed)]
        if features:
            between += [_between_mix(k, net.directed, (t.r_call(), t.names(net)[0])) for t in kept]
        else:
            between += [t.r_call() for t in kept]
    else:
        between = [t.r_call() for t in independent]
    if not between:
        raise ValueError("bigergm(): the model has no dyad-independent terms for the dyads between blocks")
    between_fit = ergm(_with_blocks(net, blocks), " + ".join(between),
                       constraints="Dyads(fix = ~nodematch('block'))")
    # The ties within blocks, on which the model's terms count.
    edges = net.edges
    inside = edges[blocks[edges[:, 0]] == blocks[edges[:, 1]]] if len(edges) else edges
    if add_intercepts:
        match = "nodematch('block', diff = TRUE)"
        within = [match] + ([f"{t.r_call()}:{match}" for t in kept] if features else [t.r_call() for t in kept]) \
            + [t.r_call() for t in terms if not t.dyad_independent]
    else:
        within = [t.r_call() for t in terms]
    within_fit = ergm(_with_blocks(net, blocks, inside), " + ".join(within), constraints="blockdiag('block')",
                      estimate=method_within, seed=seed, **options)
    return between_fit, within_fit


class BigErgmFit:
    """A fit of :func:`ergmx.bigergm`: the blocks, the MM algorithm's lower
    bounds and membership probabilities, and the fits of the dyads between
    blocks (``between``, the exact MLE of a logistic regression) and within
    them (``within``)."""

    def __init__(self, network, formula, blocks, between, within, mm_info, add_intercepts, features):
        self.network, self.formula = network, formula
        self.blocks = np.asarray(blocks)
        self.between: ErgmFit | None = between
        self.within: ErgmFit | None = within
        self._mm = mm_info
        self.add_intercepts, self.clustering_with_features = add_intercepts, features

    @property
    def n_blocks(self) -> int:
        return int(self.blocks.max())

    @property
    def lower_bound(self) -> list[float] | None:
        """The MM algorithm's lower bound of the log-likelihood at each iteration."""
        return None if self._mm is None else list(self._mm["lower_bound"])

    @property
    def membership(self) -> np.ndarray | None:
        """The MM algorithm's posterior membership probabilities (vertices x blocks, in its numbering)."""
        return None if self._mm is None else self._mm["alpha"]

    def summary(self) -> str:
        sizes = np.bincount(self.blocks, minlength=self.n_blocks + 1)[1:]
        lines = [f"Blocks: {self.n_blocks}, of sizes {', '.join(map(str, sizes))}."]
        if self._mm is not None:
            lines.append(f"MM algorithm: {len(self._mm['lower_bound'])} iterations, lower bound "
                         f"{self._mm['lower_bound'][-1]:.4f}.")
        if self.within is not None:
            lines += ["", "Within blocks:", str(self.within.summary()), "", "Between blocks:",
                      str(self.between.summary())]
        return "\n".join(lines)

    __str__ = summary

    def simulate(self, nsim: int = 1, *, seed=None, output: str = "network", burnin: int | None = None,
                 interval: int | None = None):
        """Networks from the fitted model: the dyads within blocks from the
        within-block model (by MCMC, from the network), those between from
        the between-block model (independently). ``output="edges"`` gives
        their edge lists."""
        from ._estimation import Control

        if self.within is None:
            raise ValueError("the blocks' parameters were not estimated (estimate_parameters=False)")
        rng = np.random.default_rng(seed)
        model, defaults = self.within._model, Control()
        _, _, inside = model.simulate(
            [model.network.edges], self.within.params, defaults.burnin if burnin is None else burnin,
            defaults.interval if interval is None else interval, nsim, int(rng.integers(2**63)),
            keep_networks=True, triadic_weight=model.triadic_weight(None))
        pairs, p = self._between_probabilities()
        network = _with_blocks(self.network, self.blocks)
        out = []
        for k in range(nsim):
            drawn = pairs[rng.random(len(p)) < p]
            edges = np.vstack([np.asarray(inside[0][k], dtype=np.int64).reshape(-1, 2), drawn]).astype(np.uint32)
            out.append(edges if output == "edges" else to_graph(network, edges))
        return out

    def _between_probabilities(self):
        """The dyads between blocks, and the between-block model's probability of a tie in each."""
        model = self.between._model
        x, _, pairs = model.core.mple_data(model.network.edges, model.space)
        eta = x @ model.eta(self.between.params)
        return np.asarray(pairs, dtype=np.int64), 1 / (1 + np.exp(-eta))

    def gof(self, nsim: int = 100, *, seed=None, stats=None, **options):
        """Goodness of fit, as bigergm's gof(): the degree, edgewise shared
        partner and geodesic distance distributions of networks simulated
        from the fit against the network's."""
        from ._gof import GofResult, GofTable, _labels, _many, default_stats

        stats = stats or [s for s in default_stats(self.network) if s != "model"]
        sims = self.simulate(nsim, seed=seed, output="edges", **options)
        n, directed = self.network.n, self.network.directed
        tables = {s: GofTable(s, _labels(n, s), _many(n, directed, [self.network.edges], s)[0],
                              _many(n, directed, sims, s)) for s in stats}
        return GofResult(tables, nsim)

