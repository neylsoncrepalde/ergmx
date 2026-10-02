"""Series of networks (NetSeries(), tergm's operators and CMLE) and dynamic
simulation, beyond the comparisons with R in test_statistics.py and
test_estimation.py."""

import itertools
import json

import igraph as ig
import numpy as np
import pytest
from conftest import DATA, load
from scipy.special import expit

import ergmx
from ergmx._model import bind


def monks():
    return [load(f"samplk{k}") for k in (1, 2, 3)]


def _matrix(g):
    return np.array(g.get_adjacency().data)


def test_operators_see_the_union_intersection_and_changes():
    nets = monks()
    stats = ergmx.summary_stats(ergmx.NetSeries(nets), "Form(~edges + mutual) + Persist(~edges + mutual) + "
                                "Diss(~edges) + Cross(~edges) + Change(~edges)")
    form = persist = change = cross = mutual_form = mutual_persist = 0
    for before, after in zip(nets, nets[1:]):
        p, c = _matrix(before), _matrix(after)
        union, both = np.maximum(p, c), p * c
        form, persist, cross = form + union.sum(), persist + both.sum(), cross + c.sum()
        change += np.abs(c - p).sum()
        mutual_form += (union * union.T).sum() / 2
        mutual_persist += (both * both.T).sum() / 2
    assert stats == {"Form(1)~edges": form, "Form(1)~mutual": mutual_form,
                     "Persist(1)~edges": persist, "Persist(1)~mutual": mutual_persist,
                     "Diss(1)~edges": -persist, "Cross(1)~edges": cross, "Change(1)~edges": change}


def test_series_attributes():
    series = ergmx.NetSeries(monks(), times=[0, 2, 5])
    assert [b.attributes[".Time"] for b in series.blocks] == [2, 5]
    assert [b.attributes[".TimeDelta"] for b in series.blocks] == [2, 3]
    assert [b.attributes[".TimeID"] for b in series.blocks] == [1, 2]
    assert repr(series) == "<NetSeries: 2 transitions of 18 vertices, directed>"
    with pytest.raises(ValueError, match="increasing"):
        ergmx.NetSeries(monks(), times=[0, 1, 1])
    with pytest.raises(ValueError, match="same vertices"):
        ergmx.NetSeries(load("samplk1"), load("faux.dixon.high"))
    with pytest.raises(ValueError, match="series of networks"):
        ergmx.summary_stats(ergmx.Networks(monks()), "Form(~edges)")


def test_dyad_independent_cmle_is_exact():
    """Form(~edges) + Persist(~edges): the logits of the share of non-ties that
    became ties, and of ties that persisted."""
    nets = monks()
    fit = ergmx.tergm(nets, "Form(~edges) + Persist(~edges)")
    formed = possible = kept = had = 0
    for before, after in zip(nets, nets[1:]):
        p, c = _matrix(before), _matrix(after)
        off = ~np.eye(18, dtype=bool)
        formed += ((p == 0) & (c == 1) & off).sum()
        possible += ((p == 0) & off).sum()
        kept, had = kept + ((p == 1) & (c == 1)).sum(), had + p.sum()
    assert fit.method == "MLE"
    assert fit.coef["Form(1)~edges"] == pytest.approx(np.log(formed / (possible - formed)), rel=1e-8)
    assert fit.coef["Persist(1)~edges"] == pytest.approx(np.log(kept / (had - kept)), rel=1e-8)
    assert str(fit.summary()).startswith("Conditional Maximum Likelihood Results:")


def _enumerate(model, theta, prev_blocks=None):
    blocks = model.network.blocks
    pairs = [(b.start + i, b.start + j) for b in blocks for i, j in itertools.permutations(range(b.network.n), 2)]
    stats = np.array([model.core.summary(np.array([p for p, x in zip(pairs, bits) if x],
                                                  dtype=np.uint32).reshape(-1, 2))
                      for bits in itertools.product((0, 1), repeat=len(pairs))])
    logp = stats @ theta
    p = np.exp(logp - logp.max())
    return (p / p.sum()) @ stats


FORMULA = ("Form(~edges + mutual + gwesp(0.5, fixed=TRUE)) + Persist(~edges + mutual) + "
           "Change(~edges) + Cross(~ctriple) + ttriple")
THETA = np.array([-0.8, 0.6, 0.3, 0.7, -0.4, -0.2, 0.3, 0.2])


def _small_series():
    a = ig.Graph(n=3, edges=[(0, 1), (1, 2), (2, 0)], directed=True)
    b = ig.Graph(n=3, edges=[(0, 1), (1, 0)], directed=True)
    c = ig.Graph(n=3, edges=[(1, 2)], directed=True)
    return ergmx.NetSeries(a, b, c)


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_sampler_of_transitions_is_exact(weight):
    """The conditional model of two transitions of 3-vertex directed networks,
    with discordant-dyad proposals, against every network."""
    model = bind(_small_series(), FORMULA)
    expected = _enumerate(model, THETA)
    runs = np.array([model.simulate([model.network.edges] * 4, THETA, 1000, 10, 800, seed,
                                    triadic_weight=weight)[0].reshape(-1, len(THETA)).mean(axis=0)
                     for seed in range(12)])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(runs.mean(axis=0) - expected) < 5 * se + 1e-9)


def test_dynamic_steps_draw_from_the_conditional_model():
    """One time step from a network is a draw from the model of the
    transition from it."""
    start = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 0), (3, 2)], directed=True)
    formula = "Form(~edges + mutual + gwesp(0.5, fixed=TRUE)) + Persist(~edges + mutual) + Cross(~ctriple)"
    theta = np.array([-0.8, 0.6, 0.3, 0.7, -0.4, 0.2])
    model = bind(ergmx.NetSeries(start, start), formula)
    expected = _enumerate(model, theta)
    sims = ergmx.simulate_dynamic(start, formula, theta, nsim=4000, seed=1, min_steps=500, max_steps=500)
    stats = np.array([s.stats[0] for s in sims])
    se = stats.std(axis=0, ddof=1) / np.sqrt(len(stats))
    assert np.all(np.abs(stats.mean(axis=0) - expected) < 5 * se + 1e-9)


def test_dynamic_rates_match_a_dyad_independent_model():
    """With tergm's stopping rule, ties form and dissolve at the model's
    rates: the probabilities expit(theta) per dyad."""
    g = load("faux.mesa.high")
    dyads = g.vcount() * (g.vcount() - 1) / 2
    form, persist = -6.0, 1.5
    sims = ergmx.simulate_dynamic(g, "Form(~edges) + Persist(~edges)", [form, persist], time_slices=30,
                                  nsim=4, seed=1)
    edges = np.array([s.edges for s in sims])
    formed, dissolved = np.array([s.formed for s in sims]), np.array([s.dissolved for s in sims])
    assert formed.sum() == pytest.approx((expit(form) * (dyads - edges[:, :-1])).sum(), rel=0.05)
    assert dissolved.sum() == pytest.approx(((1 - expit(persist)) * edges[:, :-1]).sum(), rel=0.08)
    assert np.all(edges[:, 1:] - edges[:, :-1] == formed - dissolved)


def test_durations_and_monitor():
    g = load("samplk1")
    sim = ergmx.simulate_dynamic(g, "Form(~edges) + Persist(~edges)", {"Form(1)~edges": -3.0,
                                 "Persist(1)~edges": 1.0}, time_slices=20, seed=2, monitor="edges + mutual")
    assert len(sim) == 20 and len(sim.networks) == 20 and sim.start.ecount() == g.ecount()
    np.testing.assert_array_equal(sim.monitor["edges"], sim.edges[1:])
    finished, ongoing = sim.durations()
    assert len(ongoing) == sim.edges[-1]
    assert finished.min() >= 1 and ongoing.max() <= 20
    assert "time   edges  formed" in repr(sim)


def test_dynamic_simulation_needs_constant_linear_models():
    with pytest.raises(NotImplementedError, match="time-varying"):
        ergmx.simulate_dynamic(load("samplk1"), "Form(~edges, lm=~.Time)", [-3.0, 0.1])
    with pytest.raises(ValueError, match="single network"):
        ergmx.simulate_dynamic(ergmx.Networks(monks()), "Form(~edges)", [-3.0])


def test_fits_continue_the_series():
    nets = monks()
    fit = ergmx.tergm(nets, "Form(~edges + mutual) + Persist(~edges)", seed=1)
    assert fit.converged and "2 transitions" in str(fit.summary())
    last = fit.simulate(time_slices=3, seed=1)
    assert last.start.get_edgelist() == nets[-1].get_edgelist()
    first = fit.simulate(time_slices=2, nsim=2, nw_start="first", seed=1)
    assert len(first) == 2 and first[0].start.get_edgelist() == nets[0].get_edgelist()
    assert fit.simulate(time_slices=1, nw_start=2, seed=1).start.get_edgelist() == nets[1].get_edgelist()
    with pytest.raises(ValueError, match="nw_start"):
        fit.simulate(time_slices=1, nw_start=4)
    with pytest.raises(ValueError, match="series of networks"):
        ergmx.ergm(load("samplk1"), "edges + mutual", seed=1).simulate(time_slices=2)


def test_fits_with_time_trends_continue_them():
    """The coefficients of transition k after the last are the linear models'
    predictions at .Time + k .TimeDelta: here formation's log-odds."""
    fit = ergmx.tergm(monks(), "Form(~edges, lm=~.Time) + Persist(~edges)", times=[0, 1, 2])
    b0, b1, persist = fit.params
    sims = fit.simulate(time_slices=4, nsim=200, seed=1, monitor="edges")
    dyads = 18 * 17
    for k in range(4):
        before, after = zip(*((s._sets()[k], s._sets()[k + 1]) for s in sims))
        formed = sum(len(b - a) for a, b in zip(before, after))
        free = sum(dyads - len(a) for a in before)
        p = expit(b0 + b1 * (3 + k))
        assert abs(formed / free - p) < 4 * np.sqrt(p * (1 - p) / free)
        kept = sum(len(a & b) for a, b in zip(before, after))
        q = expit(persist)
        assert abs(kept / sum(map(len, before)) - q) < 4 * np.sqrt(q * (1 - q) / sum(map(len, before)))
    np.testing.assert_array_equal(sims[0].monitor["edges"], sims[0].edges[1:])
    with pytest.raises(ValueError, match="last network"):
        fit.simulate(time_slices=2, nw_start="first")
    levels = ergmx.tergm(monks(), "Form(~edges, lm=~factor(.TimeID)) + Persist(~edges)")
    with pytest.raises(ValueError, match="new levels"):
        levels.simulate(time_slices=1)


def test_tergm_arguments():
    nets = monks()
    pseudo = ergmx.tergm(nets, "Form(~edges + mutual) + Persist(~edges)", estimate="CMPLE")
    assert pseudo.method == "MPLE"
    assert str(pseudo.summary()).startswith("Conditional Maximum Pseudolikelihood Results:")
    with pytest.raises(ValueError, match="EGMME"):
        ergmx.tergm(nets, "Form(~edges)", estimate="EGMME")
    with pytest.raises(TypeError, match="list of networks"):
        ergmx.tergm(nets[0], "Form(~edges)")
    g = nets[0].copy()
    g.es["na"] = False
    g.add_edge(0, 5, na=True)
    with pytest.raises(ValueError, match="time.s. 0 have missing dyads"):
        ergmx.NetSeries(g, nets[1])
    with pytest.raises(ValueError, match="unknown NA imputation"):
        ergmx.NetSeries(g, nets[1], na_impute="last")
    with pytest.raises(ValueError, match="NetSeries"):
        ergmx.tergm(ergmx.NetSeries(nets), "Form(~edges)", na_impute="next")


def test_na_impute_fills_the_networks_transitioned_from():
    """Each imputer fills the missing dyads it can, in turn; the networks
    transitioned to keep theirs (the fits are compared with R's in
    test_estimation.py: series_na_*)."""
    first, second, last = load("samplk1.na"), load("samplk2.na"), load("samplk3.nonresponse")

    def state(g, a, b):
        e = g.get_eid(a, b, error=False)
        return None if e < 0 else ("NA" if g.es[e]["na"] else 1)

    series = ergmx.NetSeries(first, second, last, na_impute="next")
    before = [b.prev for b in series.blocks]
    ties = [set(map(tuple, p.edges.tolist())) for p in before]
    # Vertex 1's nominations of 2-5 come from the second wave, 3's of 6-7
    # from the third (they're missing in the second as well).
    for t, a, b, source in [(0, 0, 1, second), (0, 0, 4, second), (0, 2, 5, last), (0, 2, 6, last),
                            (1, 2, 8, last)]:
        assert ((a, b) in ties[t]) == (state(source, a, b) == 1)
    assert not any(len(p.missing) for p in before)
    assert len(series.missing) == len(load("samplk2.na").es.select(na=True)) + \
        len(last.es.select(na=True))
    # The first wave has no previous one; 3's nominations of 6-7 are missing
    # in both.
    with pytest.warns(UserWarning, match="first network"), \
            pytest.raises(ValueError, match="time.s. 0, 1 have missing dyads"):
        ergmx.NetSeries(first, second, last, na_impute="previous")
    zero = ergmx.NetSeries(first, second, last, na_impute=["prev", "1"])
    assert {(0, 1), (0, 4), (2, 5), (2, 6)} <= set(map(tuple, zero.blocks[0].prev.edges.tolist()))
    tied = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=False)
    tied.es["na"] = [False, True]
    with pytest.warns(UserWarning, match="majority"), pytest.raises(ValueError, match="missing dyads"):
        ergmx.NetSeries(tied, tied, na_impute="majority")


# -- Tie ages and the EGMME ---------------------------------------------------------------------


def test_durational_monitors_follow_the_ties():
    """mean.age, edge.ages and edges.ageinterval of a simulation, against the
    ages counted from its ties' formations."""
    rng = np.random.default_rng(1)
    pairs = [(i, j) for i in range(25) for j in range(i + 1, 25)]
    g = ig.Graph(n=25, edges=[pairs[k] for k in rng.choice(len(pairs), 20, replace=False)])
    sim = ergmx.simulate_dynamic(g, "Form(~edges) + Persist(~edges)", [-4.0, 2.0], 30, seed=2,
                                 monitor="edges + mean.age + edge.ages + edges.ageinterval(c(1, 3), c(3, Inf))")
    since = {tuple(sorted(e)): 0 for e in g.get_edgelist()}
    for t, h in enumerate(sim.networks, start=1):
        current = {tuple(sorted(e)) for e in h.get_edgelist()}
        since = {e: since.get(e, t) for e in current}
        ages = np.array([t - s + 1 for s in since.values()])
        assert sim.monitor["edges"][t - 1] == len(current)
        assert sim.monitor["mean.age"][t - 1] == pytest.approx(ages.mean() if len(ages) else 0)
        assert sim.monitor["edge.ages"][t - 1] == ages.sum()
        assert sim.monitor["edges.age1to3"][t - 1] == np.sum(ages < 3)
        assert sim.monitor["edges.age3toInf"][t - 1] == np.sum(ages >= 3)
    with pytest.raises(ValueError, match="tie ages"):
        ergmx.summary_stats(g, "edges + mean.age")


def test_egmme_matches_the_exact_edges_model():
    """With Form(~edges) + Persist(~edges), each dyad is a two-state Markov
    chain: the edges target fixes the stationary density and the mean.age
    target (ties aged 1 when they form, as tergm) the persistence."""
    g = ig.Graph(n=30)
    fit = ergmx.tergm(g, "Form(~edges) + Persist(~edges)", estimate="EGMME", targets="edges + mean.age",
                      target_stats=[15, 10], seed=1)
    assert fit.converged and fit.names == ["Form~edges", "Persist~edges"]
    density, persist = 15 / 435, 1 - 1 / 10  # 435 dyads
    form = density * (1 - persist) / (1 - density)
    exact = {"Form~edges": np.log(form / (1 - form)), "Persist~edges": np.log(persist / (1 - persist))}
    # The delta method's standard errors, from the stationary (co)variances: 0.36 and 0.27.
    se = {"Form~edges": 0.36, "Persist~edges": 0.27}
    for name, value in exact.items():
        assert abs(fit.coef[name] - value) < 0.5 * se[name], name
        assert fit.stderr[name] == pytest.approx(se[name], rel=0.35), name


def test_egmme_matches_r():
    """tergm's EGMME of a dyad-dependent model, with three seeds."""
    ref = json.loads((DATA / "r_egmme_reference.json").read_text())["models"]["degree_duration"]
    g = ig.Graph(n=ref["n"])
    fit = ergmx.tergm(g, ref["formula"], estimate="EGMME", targets=ref["targets"],
                      target_stats=ref["target_stats"], seed=1)
    assert fit.converged
    for name in fit.names:
        r = np.mean([f["coef"][name] for f in ref["fits"]])
        r_se = [f["se"][name] for f in ref["fits"]]
        assert abs(fit.coef[name] - r) < 0.4 * fit.stderr[name], name
        assert min(r_se) / 1.5 < fit.stderr[name] < max(r_se) * 1.5, name
    assert "Equilibrium Generalized Method of Moments" in fit.summary()
    with pytest.raises(ValueError, match="target_stats"):
        ergmx.tergm(g, ref["formula"], estimate="EGMME", targets=ref["targets"])
