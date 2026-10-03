"""Samples of networks (Networks() and N(), as ergm.multi): their statistics,
linear models, sampler and fits, beyond the comparisons with R in
test_statistics.py and test_estimation.py."""

import itertools

import igraph as ig
import numpy as np
import pytest
from conftest import load

import ergmx
from ergmx._lm import LmError, design
from ergmx._model import bind


def monks():
    return [load(f"samplk{k}") for k in (1, 2, 3)]


def test_terms_inside_n_are_summed_over_the_networks():
    nets = monks()
    formula = "edges + mutual + gwesp(0.5, fixed=TRUE) + dsp(0:2) + nodematch('group')"
    each = [ergmx.summary_stats(g, formula) for g in nets]
    total = ergmx.summary_stats(ergmx.Networks(nets), f"N(~{formula})")
    assert list(total) == [f"N(1)~{name}" for name in each[0]]
    np.testing.assert_allclose(list(total.values()), np.sum([list(s.values()) for s in each], axis=0))


def test_linear_models_weight_each_network():
    nets = monks()
    for k, g in enumerate(nets):
        g["x"], g["kind"] = [1.5, 2.0, -1.0][k], ["b", "a", "c"][k]
    edges = np.array([g.ecount() for g in nets])
    stats = ergmx.summary_stats(ergmx.Networks(nets), "N(~edges, lm=~x + kind) + N(~edges, lm=~0 + kind)")
    assert stats == {
        "N(1)~edges": edges.sum(), "N(x)~edges": edges @ [1.5, 2.0, -1.0],
        "N(kindb)~edges": edges[0], "N(kindc)~edges": edges[2],
        "N(kinda)~edges": edges[1], "N(kindb)~edges.1": edges[0], "N(kindc)~edges.1": edges[2],
    }


@pytest.mark.parametrize("lm, columns", [
    ("~1", ["1"]),
    ("~log(n) - 1", ["log(n)"]),
    ("~I(n<=3) + I(n >= 5)", ["1", "I(n <= 3)TRUE", "I(n >= 5)TRUE"]),
    ("~I(x^2) + I(x/2) + I(x > 1 & x < 2)", ["1", "I(x^2)", "I(x/2)", "I(x > 1 & x < 2)TRUE"]),
    ("~0 + factor(.NetworkID)", ["factor(.NetworkID)1", "factor(.NetworkID)2", "factor(.NetworkID)3"]),
    ("~weekday + kind", ["1", "weekdayTRUE", "kindb"]),
    ("~!weekday", ["1", "!weekdayTRUE"]),
])
def test_linear_model_columns_are_named_as_in_r(lm, columns):
    attributes = [{"n": 3, "x": 1.5, "weekday": True, "kind": "a", ".NetworkID": 1},
                  {"n": 5, "x": 2.0, "weekday": False, "kind": "b", ".NetworkID": 2},
                  {"n": 6, "x": -1.0, "weekday": True, "kind": "a", ".NetworkID": 3}]
    x, names = design(lm, attributes)
    assert names == columns and x.shape == (3, len(columns))


def test_bad_linear_models():
    attributes = [{"n": 3, "g": "a"}, {"n": 4, "g": None}]
    with pytest.raises(LmError, match="no network attribute"):
        design("~size", attributes)
    with pytest.raises(LmError, match="missing for some"):
        design("~g", attributes)
    with pytest.raises(LmError, match="interactions"):
        design("~n:g", attributes)
    with pytest.raises(LmError, match="unsupported function"):
        design("~poly(n, 2)", attributes)


def test_formulas_take_r_syntax():
    f = ergmx.parse_formula("N(~edges, ~I(n<=3) + I(.NetworkID >= 2)) + N(~kstar(2), lm = ~log(n))")
    assert [t.lm for t in f] == ["~I(n<=3) + I(.NetworkID >= 2)", "~log(n)"]
    with pytest.raises(ValueError, match="can't be inside"):
        ergmx.parse_formula("N(~N(~edges))")
    t, = ergmx.parse_formula("N(~edges, ~log(n), subset=~n>=4 & .NetworkID != 2, offset=~log(n), label='hh')")
    assert (t.lm, t.subset, t.offset, t.name_label) == ("~log(n)", "~n>=4 & .NetworkID != 2", "~log(n)", "hh")
    t, = ergmx.parse_formula("N(~edges, subset=c(TRUE, FALSE), offset=2)")
    assert (t.subset, t.offset) == ("c(TRUE, FALSE)", "2")
    with pytest.raises(NotImplementedError, match="weights"):
        ergmx.summary_stats(ergmx.Networks(monks()), "N(~edges, weights=~n)")
    assert len(ergmx.summary_stats(ergmx.Networks(monks()), "N(~edges, weights=~1)")) == 1
    with pytest.raises(ergmx.FormulaError, match="contrasts must be a dict"):
        ergmx.parse_formula("N(~edges, contrasts=1)")
    t, = ergmx.parse_formula("N(~edges, ~factor(n), contrasts=list(`factor(n)`=contr.sum, x='contr.poly'))")
    assert t.contrasts == {"factor(n)": "contr.sum", "x": "contr.poly"}


def _sized():
    nets = monks()
    for k, g in enumerate(nets):
        g["x"], g["big"] = [1.5, 2.0, -1.0][k], k > 0
    return nets, ergmx.Networks(nets)


def test_subset_keeps_networks():
    nets, combined = _sized()
    edges = np.array([g.ecount() for g in nets])
    for subset, kept in [("~big", [1, 2]), ("~.NetworkID != 2", [0, 2]), ([True, False], [0, 2]),
                         ([3], [2]), ([-1], [1, 2]), ("c(1, 3)", [0, 2]), ("TRUE", [0, 1, 2])]:
        stats = ergmx.summary_stats(combined, ergmx.N("~edges", lm="~x", subset=subset))
        assert stats == {"N(1)~edges": edges[kept].sum(), "N(x)~edges": edges[kept] @ np.array([1.5, 2.0, -1.0])[kept]}
    # The linear model's levels are those of the kept networks.
    assert list(ergmx.summary_stats(combined, "N(~edges, ~factor(x), subset=~big)")) == \
        ["N(1)~edges", "N(factor(x)2)~edges"]
    with pytest.raises(ValueError, match="keeps no network"):
        ergmx.summary_stats(combined, "N(~edges, subset=~x > 5)")
    with pytest.raises(ValueError, match="beyond"):
        ergmx.summary_stats(combined, "N(~edges, subset=4)")


def test_offsets_add_statistics_with_coefficient_one():
    nets, combined = _sized()
    edges = np.array([g.ecount() for g in nets])
    mutual = np.array([ergmx.summary_stats(g, "mutual")["mutual"] for g in nets])
    model = bind(combined, "N(~edges + mutual, lm=~big, offset=~x / 2)")
    assert model.names == ["N(1)~edges", "N(bigTRUE)~edges", "N(1)~mutual", "N(bigTRUE)~mutual"]
    assert model.stat_names == ["N(1)~edges", "N(bigTRUE)~edges", "offset1", "N(1)~mutual",
                                "N(bigTRUE)~mutual", "offset2"]
    x = np.array([1.5, 2.0, -1.0]) / 2
    np.testing.assert_allclose(model.observed(), [edges.sum(), edges[1:].sum(), edges @ x, mutual.sum(),
                                                  mutual[1:].sum(), mutual @ x])
    theta = np.array([-2.0, 0.5, 1.0, -0.3])
    np.testing.assert_allclose(model.eta(theta), [-2.0, 0.5, 1.0, 1.0, -0.3, 1.0])
    jac, h = model.jacobian(theta), 1e-6
    numeric = np.column_stack([(model.eta(theta + h * e) - model.eta(theta - h * e)) / (2 * h) for e in np.eye(4)])
    np.testing.assert_allclose(jac, numeric, atol=1e-8)
    # The same as offset() in the linear model, or numbers.
    same = ergmx.summary_stats(combined, "N(~edges + mutual, lm=~big + offset(x / 2))")
    assert list(same.values()) == pytest.approx(model.observed())
    numbers = ergmx.summary_stats(combined, ergmx.N("~edges", offset=list(x)))
    assert numbers["offset1"] == pytest.approx(edges @ x)
    # Each network's coefficient is its prediction plus its offset: the MPLE
    # of edges alone is the logit of the density shifted by the offsets.
    fit = ergmx.ergm(combined, "N(~edges, offset=~x / 2)", estimate="MPLE")
    p = 1 / (1 + np.exp(-(fit.params[0] + x)))
    assert p @ np.full(3, 18 * 17) == pytest.approx(edges.sum(), rel=1e-6)


def test_labels_name_the_statistics():
    _, combined = _sized()
    assert list(ergmx.summary_stats(combined, "N(~edges, lm=~big, label='hh')")) == \
        ["N(hh,1)~edges", "N(hh,bigTRUE)~edges"]
    assert list(ergmx.summary_stats(combined, ergmx.N("~edges + mutual", lm="~x",
                                                      label=lambda name, column: f"{name}[{column}]"))) == \
        ["edges[1]", "edges[x]", "mutual[1]", "mutual[x]"]


def test_subsets_and_offsets_of_curved_terms():
    """Curved terms keep the statistics of the kept networks only (N#1, N#2
    for the networks kept), and each network's decay and coefficient are the
    linear model's predictions plus the offset."""
    nets, combined = _sized()
    model = bind(combined, "N(~edges + gwesp(0.5), subset=~big, offset=~x)")
    assert model.stat_names[0].startswith("N#1~") and not any(n.startswith("N#3") for n in model.stat_names)
    theta = np.array([-2.0, 0.4, 0.7])
    total = model.eta(theta) @ model.observed()
    expected = 0.0
    for g, x in zip(nets[1:], [2.0, -1.0]):
        edges, coef, decay = theta + x
        fixed = ergmx.summary_stats(g, f"edges + gwesp({decay}, fixed=TRUE)")
        expected += edges * fixed["edges"] + coef * fixed[f"gwesp.OTP.fixed.{decay:g}"]
    assert total == pytest.approx(expected, rel=1e-10)
    jac, h = model.jacobian(theta), 1e-6
    numeric = np.column_stack([(model.eta(theta + h * e) - model.eta(theta - h * e)) / (2 * h) for e in np.eye(3)])
    np.testing.assert_allclose(jac, numeric, atol=1e-6)


def test_sampler_with_subsets_and_offsets_is_exact():
    a = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
    b = ig.Graph(n=3, edges=[(0, 2)])
    c = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
    for g, size in zip((a, b, c), (0.5, -0.5, 1.0)):
        g["size"] = size
    model = bind(ergmx.Networks(a, b, c), "N(~edges + triangle, subset=~size < 1, offset=~size) + N(~kstar(2))")
    theta = np.array([-0.4, 0.3, 0.2])
    eta = model.eta(theta)
    expected = _enumerate(model, eta)
    runs = np.array([model.simulate([model.network.edges] * 4, theta, 1000, 10, 800, seed)[0]
                     .reshape(-1, len(eta)).mean(axis=0) for seed in range(12)])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(runs.mean(axis=0) - expected) < 5 * se + 1e-9)


def test_n_needs_combined_networks_and_equal_parameters():
    with pytest.raises(ValueError, match="Networks"):
        ergmx.summary_stats(load("samplk1"), "N(~edges)")
    a, b = ig.Graph(n=3, edges=[(0, 1)]), ig.Graph(n=3, edges=[(1, 2)])
    a.vs["g"], b.vs["g"] = ["x", "y", "y"], ["x", "x", "x"]
    with pytest.raises(ValueError, match="different numbers of parameters"):
        ergmx.summary_stats(ergmx.Networks(a, b), "N(~nodefactor('g'))")
    with pytest.raises(ValueError, match="all directed or all undirected"):
        ergmx.Networks(a, load("samplk1"))


def test_curved_terms_inside_n_get_each_networks_statistics():
    """With curved terms, N() keeps each network's histogram counts, whose
    coefficients follow the linear model: eta . counts is the sum over the
    networks of the fixed-decay terms at each network's coefficients."""
    nets = monks()
    for k, g in enumerate(nets):
        g["x"] = [0.5, 1.0, -0.5][k]
    combined = ergmx.Networks(nets)
    model = bind(combined, "N(~edges + gwesp(0.5), lm=~x)")
    assert model.names == ["N(1)~edges", "N(x)~edges", "N(1)~gwesp.OTP", "N(x)~gwesp.OTP",
                           "N(1)~gwesp.OTP.decay", "N(x)~gwesp.OTP.decay"]
    theta = np.array([-2.0, 0.3, 0.4, -0.2, 0.7, 0.1])
    total = model.eta(theta) @ model.observed()
    expected = 0.0
    for g, x in zip(nets, [0.5, 1.0, -0.5]):
        edges, coef, decay = theta[[0, 2, 4]] + x * theta[[1, 3, 5]]
        fixed = ergmx.summary_stats(g, f"edges + gwesp({decay}, fixed=TRUE)")
        expected += edges * fixed["edges"] + coef * fixed[f"gwesp.OTP.fixed.{decay:g}"]
    assert total == pytest.approx(expected, rel=1e-10)
    # The Jacobian, against finite differences.
    jac, h = model.jacobian(theta), 1e-6
    numeric = np.column_stack([(model.eta(theta + h * e) - model.eta(theta - h * e)) / (2 * h)
                               for e in np.eye(len(theta))])
    np.testing.assert_allclose(jac, numeric, atol=1e-6)
    # The decays start at the decay argument in every network.
    start = model.initial()
    assert start[4] == pytest.approx(0.5) and start[5] == pytest.approx(0.0, abs=1e-12)


def _enumerate(model, theta):
    """Exact expected statistics over every network with ties only within blocks."""
    blocks = model.network.blocks
    directed = model.network.directed
    pairs = [(b.start + i, b.start + j) for b in blocks for i, j in
             (itertools.permutations(range(b.network.n), 2) if directed
              else itertools.combinations(range(b.network.n), 2))]
    stats = np.array([model.core.summary(np.array([p for p, x in zip(pairs, bits) if x],
                                                  dtype=np.uint32).reshape(-1, 2))
                      for bits in itertools.product((0, 1), repeat=len(pairs))])
    logp = stats @ theta
    p = np.exp(logp - logp.max())
    return (p / p.sum()) @ stats


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_sampler_of_several_networks_is_exact(weight):
    a = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
    b = ig.Graph(n=3, edges=[(0, 2)])
    a.vs["g"], b.vs["g"] = [1, 1, 2, 2], [1, 2, 2]
    a["big"], b["big"] = True, False
    model = bind(ergmx.Networks(a, b), "N(~edges + triangle, lm=~big) + N(~nodematch('g')) + kstar(2)")
    theta = np.array([-0.4, 0.3, 0.5, -0.6, 0.4, 0.2])
    expected = _enumerate(model, theta)
    runs = np.array([model.simulate([model.network.edges] * 4, theta, 1000, 10, 800, seed,
                                    triadic_weight=weight)[0].reshape(-1, len(theta)).mean(axis=0)
                     for seed in range(12)])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(runs.mean(axis=0) - expected) < 5 * se + 1e-9)
    # No tie ever crosses between the networks.
    _, last, _ = model.simulate([model.network.edges], theta, 5000, 1, 1, 1)
    assert np.all((last[0] < 4).all(axis=1) | (last[0] >= 4).all(axis=1))


def test_simulated_networks_come_back_split():
    nets = monks()
    sims = ergmx.simulate(ergmx.Networks(nets), "N(~edges + mutual)", [-2.0, 2.0], nsim=3, seed=1)
    assert len(sims) == 3 and all(len(s) == 3 for s in sims)
    assert all(isinstance(g, ig.Graph) and g.vcount() == 18 for s in sims for g in s)
    assert sims[0][0].vs["group"] == nets[0].vs["group"]


def test_gof_pools_the_networks():
    nets = monks()
    combined = ergmx.Networks(nets)
    result = ergmx.gof(combined, "N(~edges + mutual)", [-2.1, 2.2], nsim=20, seed=1)
    pooled = sum(np.bincount(g.degree(mode="in"), minlength=18) for g in nets)
    np.testing.assert_array_equal(result["idegree"].observed, pooled)
    # Distances only between vertices of the same network.
    unreachable = sum((np.isinf(np.array(g.distances(mode="out"))) & ~np.eye(18, dtype=bool)).sum()
                      for g in nets)
    assert result["distance"].observed[-1] == unreachable


def test_fit_of_several_networks():
    fit = ergmx.ergm(ergmx.Networks(monks()), "N(~edges + mutual)", seed=1)
    assert fit.method == "MCMLE" and fit.converged
    assert "Fitted to 3 networks." in str(fit.summary())
    assert fit._model.n_observations == 3 * 18 * 17


def _grades():
    mesa = load("faux.mesa.high")
    return [mesa.induced_subgraph([v.index for v in mesa.vs if v["Grade"] == g]) for g in range(7, 13)]


def _simulated_networks():
    """20 networks of 40 vertices simulated from known coefficients (decay
    0.7), where the decay is well identified: the six grades of
    faux.mesa.high identify it so poorly that whether the Monte Carlo MLE
    converges depends on the random path (the number of chains, the
    platform)."""
    nets = []
    for seed in range(100, 120):
        (g,) = ergmx.simulate(ig.Graph(n=40), "edges + gwesp(0.7)", [-3.5, 0.8, 0.7], seed=seed,
                              burnin=100_000)
        nets.append(g)
    return ergmx.Networks(nets)


def _simulated_series():
    """A series simulated from known coefficients (decay 0.7)."""
    sim = ergmx.simulate_dynamic(_grades()[0], "Form(~edges + gwesp(0.7)) + Persist(~edges)",
                                 [-4.0, 0.8, 0.7, 1.0], time_slices=4, seed=3)
    return ergmx.NetSeries(sim.start, *sim.networks)


@pytest.mark.parametrize("combined, op, rest", [
    (_simulated_networks, "N", ""),
    (_simulated_series, "Form", " + Persist(~edges)"),
])
def test_curved_fits_inside_operators_are_consistent(combined, op, rest):
    """R can't fit curved terms inside N() or Form() (its MPLE fails). At the
    curved MPLE's decay, the fixed-decay MPLE has the same coefficients; the
    decay is near the truth (0.7) of the simulated networks and series, and
    the Monte Carlo MLE converges near the MPLE."""
    nets = combined()
    curved = ergmx.ergm(nets, f"{op}(~edges + gwesp(0.5)){rest}", estimate="MPLE")
    decay = curved.coef[f"{op}(1)~gwesp.decay"]
    assert 0.5 < decay < 0.9
    fixed = ergmx.ergm(nets, f"{op}(~edges + gwesp({decay!r}, fixed=TRUE)){rest}", estimate="MPLE")
    others = [v for k, v in curved.coef.items() if not k.endswith(".decay")]
    np.testing.assert_allclose(others, list(fixed.coef.values()), rtol=1e-8)
    mle = ergmx.ergm(nets, f"{op}(~edges + gwesp(0.5)){rest}", seed=1, eval_loglik=False)
    assert mle.converged
    edges = f"{op}(1)~edges"
    assert abs(mle.coef[edges] - curved.coef[edges]) < 3 * mle.stderr[edges]


def test_bipartite_networks_combine():
    davis = ergmx.datasets.load("davis")
    other = davis.copy()
    other.delete_edges(list(range(0, other.ecount(), 3)))
    combined = ergmx.Networks(davis, other, bipartite=True)
    stats = ergmx.summary_stats(combined, "N(~edges + b1star(2) + gwb1dsp(0.5, fixed=TRUE))")
    each = [ergmx.summary_stats(g, "edges + b1star(2) + gwb1dsp(0.5, fixed=TRUE)", bipartite=True)
            for g in (davis, other)]
    np.testing.assert_allclose(list(stats.values()), np.sum([list(e.values()) for e in each], axis=0))
    model = bind(combined, "N(~edges)")
    assert model.n_observations == 2 * 18 * 14  # only ties between modes, within networks
    with pytest.raises(ValueError, match="give bipartite= to Networks"):
        ergmx.summary_stats(ergmx.Networks(davis, other), "N(~edges)", bipartite=True)
