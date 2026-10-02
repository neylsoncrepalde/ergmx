"""The constrained samplers against exact enumeration of the networks each
constraint allows, on networks small enough to list them all."""

import itertools

import igraph as ig
import numpy as np
import pytest
from conftest import load

import ergmx
from ergmx._model import bind
from ergmx.constraints import parse_constraints

UNDIRECTED = list(itertools.combinations(range(6), 2))
DIRECTED = [(i, j) for i in range(4) for j in range(4) if i != j]


def every_network(pairs):
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        yield np.array([p for p, b in zip(pairs, bits) if b], dtype=np.uint32).reshape(-1, 2)


def degrees(edges, n, directed):
    out = np.bincount(edges[:, 0], minlength=n) if len(edges) else np.zeros(n, int)
    into = np.bincount(edges[:, 1], minlength=n) if len(edges) else np.zeros(n, int)
    return (out, into) if directed else (out + into, out + into)


def check(model, theta, allowed, *, conditional=False, weights=(0.0, 0.5), seeds=12, steps=8000):
    """Mean statistics of the chains against their exact expectation over the
    networks for which `allowed(edges)` is true."""
    pairs = DIRECTED if model.network.directed else UNDIRECTED
    networks = [e for e in every_network(pairs) if allowed(e)]
    assert len(networks) > 1
    stats = np.array([model.core.summary(e) for e in networks])
    logp = stats @ theta
    p = np.exp(logp - logp.max())
    expected = (p / p.sum()) @ stats
    for weight in weights:
        runs = np.array([
            model.simulate([model.network.edges] * 4, theta, 1000, 10, steps // 10, seed,
                           conditional=conditional, triadic_weight=weight)[0]
            .reshape(-1, len(theta)).mean(axis=0)
            for seed in range(seeds)
        ])
        se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
        assert np.all(np.abs(runs.mean(axis=0) - expected) < 5 * se + 1e-9), (weight, runs.mean(0), expected)
    return len(networks)


def graph(n, edges, directed=False, **attributes):
    g = ig.Graph(n=n, edges=edges, directed=directed)
    for name, values in attributes.items():
        g.vs[name] = values
    return g


def test_bounded_degrees_undirected():
    g = graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    model = bind(g, "edges + triangle + gwesp(0.5, fixed=TRUE)", "bd(maxout=2, minout=1)")
    theta = np.array([0.2, 0.5, 0.3])
    allowed = lambda e: all(1 <= d <= 2 for d in degrees(e, 6, False)[0])  # noqa: E731
    check(model, theta, allowed)


def test_bounded_degrees_directed():
    g = graph(4, [(0, 1), (1, 2), (2, 0)], directed=True)
    model = bind(g, "edges + mutual + ttriple", "bd(maxout=2, maxin=c(1, 2, 2, 3))")
    theta = np.array([0.3, 0.6, 0.2])
    maxin = np.array([1, 2, 2, 3])

    def allowed(e):
        out, into = degrees(e, 4, True)
        return (out <= 2).all() and (into <= maxin).all()

    check(model, theta, allowed)


@pytest.mark.parametrize("start", [
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],          # all degrees 2
    [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4), (4, 5), (2, 5)],  # degrees 3, 2, 3, 2, 2, 2
])
def test_degrees_undirected(start):
    g = graph(6, start)
    model = bind(g, "triangle + gwesp(0.5, fixed=TRUE)", "degrees")
    target = degrees(model.network.edges, 6, False)[0]
    allowed = lambda e: np.array_equal(degrees(e, 6, False)[0], target)  # noqa: E731
    check(model, np.array([0.4, 0.3]), allowed, weights=(0.0,))


def _same_degrees(model):
    out, into = degrees(model.network.edges, 4, True)

    def allowed(e):
        o, i = degrees(e, 4, True)
        return np.array_equal(o, out) and np.array_equal(i, into)

    return allowed


def test_degrees_directed():
    g = graph(4, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 1)], directed=True)
    model = bind(g, "mutual + ttriple + ctriple", "degrees")
    assert check(model, np.array([0.5, 0.4, -0.3]), _same_degrees(model), weights=(0.0,)) > 2


def test_degrees_directed_reverses_cyclic_triples():
    """0 -> 1 -> 2 -> 0 and 0 -> 2 -> 1 -> 0 have the same in- and out-degrees,
    but no swap of tie endpoints turns one into the other: only reversing the
    cycle does. An asymmetric covariate tells them apart."""
    g = graph(4, [(0, 1), (1, 2), (2, 0)], directed=True)
    x = np.zeros((4, 4))
    x[0, 1] = 1.0
    g["x"] = x
    model = bind(g, "edgecov('x')", "degrees")
    assert check(model, np.array([0.8]), _same_degrees(model), weights=(0.0,)) == 2


@pytest.mark.parametrize("kind", ["odegrees", "idegrees"])
def test_out_and_in_degrees(kind):
    g = graph(4, [(0, 1), (0, 2), (1, 2), (2, 3), (3, 0)], directed=True)
    model = bind(g, "edges + mutual + ttriple + istar(2)", kind)
    out, into = degrees(model.network.edges, 4, True)
    kept = 0 if kind == "odegrees" else 1

    def allowed(e):
        return np.array_equal(degrees(e, 4, True)[kept], (out, into)[kept])

    check(model, np.array([0.0, 0.7, 0.3, -0.2]), allowed, weights=(0.0,))


def test_edges_constraint_keeps_the_number_of_edges():
    g = graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5)])
    model = bind(g, "triangle + gwesp(0.5, fixed=TRUE) + kstar(2)", "edges")
    allowed = lambda e: len(e) == 5  # noqa: E731
    assert check(model, np.array([0.6, 0.3, -0.2]), allowed, weights=(0.0,)) == 3003


@pytest.mark.parametrize("kind", ["b1degrees", "b2degrees"])
def test_mode_degrees(kind):
    """Vertices 0, 2 and 4 are the first mode; the modes interleave on purpose."""
    g = graph(6, [(0, 1), (0, 3), (2, 3), (4, 5), (2, 5)], type=[False, True] * 3)
    model = bind(g, "b1star(2) + b2star(2) + cycle(4)", kind, bipartite="type")
    first = np.array([True, False] * 3)
    kept = first if kind == "b1degrees" else ~first
    target = degrees(model.network.edges, 6, False)[0][kept]

    def allowed(e):
        across = all(first[a] != first[b] for a, b in e)
        return across and np.array_equal(degrees(e, 6, False)[0][kept], target)

    check(model, np.array([0.4, -0.3, 0.5]), allowed, weights=(0.0,))


def test_bounds_by_alter_class_undirected():
    sex = np.array([0, 0, 0, 1, 1, 1])
    g = graph(6, [(0, 3), (1, 4), (2, 5), (0, 1)], sex=sex.tolist())
    g["classes"] = np.column_stack([sex == 0, sex == 1])
    g["most"] = np.array([[1, 1], [1, 2], [2, 1], [1, 1], [2, 1], [1, 2]])
    model = bind(g, "edges + triangle + nodematch('sex')", "bd(attribs='classes', maxout='most')")

    def allowed(e):
        adj = np.zeros((6, 6), dtype=int)
        adj[e[:, 0], e[:, 1]] = adj[e[:, 1], e[:, 0]] = 1
        return (adj @ g["classes"] <= g["most"]).all()

    check(model, np.array([0.3, 0.5, 0.4]), allowed)


def test_bounds_by_alter_class_directed():
    g = graph(4, [(0, 2), (1, 3), (2, 1)], directed=True)
    g["classes"] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    g["most_in"] = np.array([[1, 1], [2, 1], [1, 2], [1, 1]])
    model = bind(g, "edges + mutual + ttriple", "bd(attribs='classes', maxin='most_in', minout=matrix(0, 4, 2))")

    def allowed(e):
        adj = np.zeros((4, 4), dtype=int)
        adj[e[:, 0], e[:, 1]] = 1
        return (adj.T @ g["classes"] <= g["most_in"]).all()

    check(model, np.array([0.2, 0.6, 0.3]), allowed)


def test_blocks_fix_dyads():
    g = graph(6, [(0, 1), (2, 3), (1, 4), (3, 5)], level=["a", "a", "a", "b", "b", "b"])
    # Fix the a-b dyads: only ties within levels change.
    model = bind(g, "edges + triangle + gwesp(0.5, fixed=TRUE)", "blocks('level', levels2=2)")
    level = np.array(["a", "a", "a", "b", "b", "b"])
    between = {(i, j) for i, j in UNDIRECTED if level[i] != level[j]}
    observed_between = {tuple(p) for p in model.network.edges.tolist()} & between
    allowed = lambda e: {tuple(p) for p in e.tolist()} & between == observed_between  # noqa: E731
    check(model, np.array([0.1, 0.5, 0.2]), allowed)


def test_missing_dyads_are_sampled_conditional_on_the_observed_ones():
    g = graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (1, 5)])
    g.es["na"] = [False, False, False, False, True, True]  # (0, 5) and (1, 5) are unknown
    model = bind(g, "edges + triangle + gwesp(0.5, fixed=TRUE)")
    assert model.network.missing.tolist() == [[0, 5], [1, 5]]
    assert len(model.network.edges) == 4
    observed = {tuple(p) for p in model.network.edges.tolist()}
    unknown = {(0, 5), (1, 5)}
    allowed = lambda e: {tuple(p) for p in e.tolist()} - unknown == observed  # noqa: E731
    check(model, np.array([0.2, 0.6, 0.1]), allowed, conditional=True)


def test_constraints_parse_like_r():
    c = parse_constraints("~bd(maxout=4) + blocks('Grade', levels2=-1)")
    assert repr(c) == "bd(maxout=4) + blocks('Grade', levels2=-1)"
    assert c.dyad_dependent
    assert not parse_constraints("blocks('Grade')").dyad_dependent
    assert parse_constraints("degrees").preserves == frozenset({"in", "out", "b1", "b2"})
    assert parse_constraints("b1degrees").preserves == frozenset({"b1"})
    assert parse_constraints("edges").preserves == frozenset({"edges"})
    with pytest.raises(ValueError, match="one degree-preserving"):
        parse_constraints("degrees + odegrees")
    with pytest.raises(ergmx.FormulaError, match="unknown constraint"):
        parse_constraints("degreedist")


def test_observed_network_must_satisfy_the_constraints():
    with pytest.raises(ValueError, match="violates bd"):
        ergmx.ergm(load("samplk3"), "edges + mutual", constraints="bd(maxout=3)")
    with pytest.raises(ValueError, match="directed network"):
        ergmx.ergm(load("flomarriage"), "edges + triangle", constraints="odegrees")


def test_statistics_constant_under_the_constraints_are_fixed():
    g = load("flomarriage")
    with pytest.warns(UserWarning, match=r"\['edges'\] are constant"):
        fit = ergmx.ergm(g, "edges + triangle", constraints="degrees", seed=1)
    assert fit.coef["edges"] == 0 and np.isnan(fit.stderr["edges"])
    assert fit.df == 1
    assert "constant under the constraints" in str(fit.summary())


def test_simulations_keep_the_constraints():
    g = load("samplk3")
    out = np.bincount(np.array(g.get_edgelist())[:, 0], minlength=18)
    for h in ergmx.simulate(g, "edges + mutual", [-1.0, 1.0], 5, constraints="odegrees", seed=1):
        assert np.array_equal(np.bincount(np.array(h.get_edgelist())[:, 0], minlength=18), out)
    for h in ergmx.simulate(g, "edges + mutual", [0.0, 1.0], 5, constraints="bd(maxout=4)", seed=1):
        assert np.bincount(np.array(h.get_edgelist())[:, 0], minlength=18).max() <= 4


def test_compare_refuses_different_constraints():
    g = load("samplk3")
    free = ergmx.ergm(g, "edges + mutual", seed=1)
    bounded = ergmx.ergm(g, "edges + mutual", constraints="bd(maxout=4)", seed=1)
    assert bounded.loglik_relative and not free.loglik_relative
    with pytest.raises(ValueError, match="different constraints"):
        ergmx.compare(free, bounded)


def test_bipartite_networks_only_have_ties_between_modes():
    """3 x 3 bipartite networks: every one of the 512 is enumerated."""
    g = graph(6, [(0, 3), (1, 4), (2, 5)], type=[False, False, False, True, True, True])
    model = bind(g, "edges + b1star(2) + gwb2degree(0.5, fixed=TRUE) + cycle(4)", bipartite="type")
    cross = {(i, j) for i in range(3) for j in range(3, 6)}
    allowed = lambda e: {tuple(sorted(p)) for p in e.tolist()} <= cross  # noqa: E731
    assert check(model, np.array([-0.3, 0.4, -0.2, 0.3]), allowed, weights=(0.0,)) == 512


def test_transitive_triads_sampler():
    """The MCMC with transitive triads against every directed network on 4 vertices."""
    import warnings

    g = graph(4, [(0, 1), (1, 2), (0, 2)], directed=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ergmx.ErgmDifferenceWarning)
        model = bind(g, "edges + mutual + transitive")
    check(model, np.array([-0.4, 0.5, 0.6]), lambda e: True)
