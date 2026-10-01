"""Series of networks (NetSeries(), tergm's operators and CMLE) and dynamic
simulation, beyond the comparisons with R in test_statistics.py and
test_estimation.py."""

import itertools

import igraph as ig
import numpy as np
import pytest
from conftest import load
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


def test_tergm_arguments():
    nets = monks()
    pseudo = ergmx.tergm(nets, "Form(~edges + mutual) + Persist(~edges)", estimate="CMPLE")
    assert pseudo.method == "MPLE"
    assert str(pseudo.summary()).startswith("Conditional Maximum Pseudolikelihood Results:")
    with pytest.raises(ValueError, match="EGMME"):
        ergmx.tergm(nets, "Form(~edges)", estimate="EGMME")
    with pytest.raises(TypeError, match="list of networks"):
        ergmx.tergm(nets[0], "Form(~edges)")
    with pytest.raises(NotImplementedError, match="missing dyads"):
        g = nets[0].copy()
        g.es["na"] = False
        g.add_edge(0, 5, na=True)
        ergmx.NetSeries(g, nets[1])
