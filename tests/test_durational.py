"""tergm's durational terms: statistics of tie ages against tergm's (scripts/
r_durational_reference.R), as terms of dynamic models (tracked statistics,
hazards of dissolution by age, simulations against tergm's), and where they
can't be."""

import json

import igraph as ig
import numpy as np
import pytest
from conftest import DATA, load
from scipy.special import expit

import ergmx
from ergmx._durational import Ages
from ergmx._network import as_network
from ergmx.formula import parse_formula

REFERENCE = DATA / "r_durational_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


def _aged(s):
    g = ig.Graph(n=s["n"], edges=[tuple(e) for e in s["edges"]], directed=s["directed"])
    g.vs["sex"], g.vs["grade"] = s["sex"], s["grade"]
    g["w"] = np.array(s["w"])
    return g


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("kind", ["undirected", "directed"])
def test_durational_statistics_match_tergm(kind):
    """Each durational term's statistics of a network whose ties have known
    ages, as tergm's (with ergmx's names of repeated statistics made unique)."""
    s = R["statistics"][kind]
    net = as_network(_aged(s))
    edges, ages = np.array(s["edges"], dtype=np.uint32), np.array(s["ages"])
    names, values = [], []
    for term in parse_formula(s["formula"]):
        names += term.names(net)
        values += list(term.value(net, edges, ages))
    assert names == s["names"]
    np.testing.assert_allclose(values, s["values"], rtol=1e-12, atol=1e-12)


FORMULA = ("Form(~edges + mean.age) + Persist(~edges + edge.ages + mean.age(log=TRUE) + "
           "edges.ageinterval(c(1, 3), c(3, Inf)) + nodefactor.mean.age('sex') + degree.mean.age(1:3)) + "
           "edgecov.mean.age('w', emptyval=2) + nodemix.mean.age('grade') + degrange.mean.age(c(1, 3), c(3, Inf), "
           "byarg='sex') + EdgeAges(~edges + nodematch('sex')) + edgecov.ages('w')")


def _expected(g, networks, start_ages):
    """The model's statistics of each step, computed in Python from the
    networks and their ties' ages: Form() on the union of the previous and
    current networks, Persist() on their intersection, the other terms on the
    current network; a tie's age is its previous age plus one if it was a tie
    then, else 1."""
    net = as_network(g)
    terms = list(parse_formula(FORMULA))
    form, persist, plain = terms[0].formula, terms[1].formula, terms[2:]
    ages, prev = start_ages.copy(), net.edges
    rows = []
    for edges in networks:
        before = ages.copy()
        now = ages.step(edges)

        def age(e, before=before):
            key = tuple(map(int, e)) if g.is_directed() else tuple(sorted(map(int, e)))
            return before.age.get(key, 0) + 1

        prev_set = {tuple(sorted(map(int, e))) for e in prev}
        cur_set = {tuple(sorted(map(int, e))) for e in edges}
        union = np.array(sorted(prev_set | cur_set), dtype=np.uint32).reshape(-1, 2)
        inter = np.array(sorted(prev_set & cur_set), dtype=np.uint32).reshape(-1, 2)
        row = []
        for formula, ties in ((form, union), (persist, inter)):
            for term in formula:
                ties_ages = np.array([age(e) for e in ties])
                row += list(term.value(net, ties, ties_ages)) if getattr(term, "durational", False) else \
                    list(ergmx.summary_stats(ergmx._network.to_graph(net, ties), term).values())
        for term in plain:
            row += list(term.value(net, edges, now))
        rows.append(row)
        prev = edges
    return np.array(rows)


def test_tracked_durational_terms_are_exact():
    """The statistics the sampler tracks, change by change, of a model with
    every durational term (in Form(), in Persist() and on the current
    network), against the statistics computed from the simulated networks."""
    rng = np.random.default_rng(3)
    pairs = [(i, j) for i in range(18) for j in range(i + 1, 18)]
    g = ig.Graph(n=18, edges=[pairs[k] for k in rng.choice(len(pairs), 25, replace=False)])
    g.vs["sex"] = rng.choice(["F", "M"], 18).tolist()
    g.vs["grade"] = rng.choice([7, 8, 9], 18).tolist()
    g["w"] = np.round(rng.uniform(0, 2, (18, 18)), 2)
    g.es["age"] = rng.integers(1, 6, g.ecount()).tolist()
    names = list(ergmx._model.bind(ergmx.NetSeries(g, g), FORMULA, dynamic=True).names)
    coef = np.zeros(len(names))
    coef[names.index("Form(1)~edges")], coef[names.index("Persist(1)~edges")] = -3.0, 1.5
    sim = ergmx.simulate_dynamic(g, FORMULA, coef, 25, seed=4, ages="age", min_steps=200, max_steps=200)
    start = as_network(g)
    expected = _expected(g, [h for h in sim._edges], Ages(False, start.edges, np.array(g.es["age"])))
    np.testing.assert_allclose(sim.stats, expected, rtol=1e-9, atol=1e-9)
    assert len({len(e) for e in sim._edges}) > 1


def test_duration_dependent_dissolution_has_its_hazards():
    """With Persist(~edges + edges.ageinterval(3, 7)), a tie of age a at the
    start of the step persists with probability expit(2 - 1) if a is in
    [3, 7), else expit(2): as tergm's, the step's model sees the ages at its
    start (its change statistics come before the tick)."""
    g = ig.Graph(n=40)
    sim = ergmx.simulate_dynamic(g, "Form(~edges) + Persist(~edges + edges.ageinterval(3, 7))", [-4.0, 2.0, -1.0],
                                 1500, seed=1)
    ages = Ages(False, np.zeros((0, 2), dtype=np.uint32))
    kept = {True: [0, 0], False: [0, 0]}
    for edges in sim._edges:
        before = dict(ages.age)
        ages.step(edges)
        current = {tuple(sorted(map(int, e))) for e in edges}
        for tie, age in before.items():
            inside = 3 <= age < 7
            kept[inside][0] += tie in current
            kept[inside][1] += 1
    for inside, p in ((True, expit(1.0)), (False, expit(2.0))):
        persisted, total = kept[inside]
        se = np.sqrt(p * (1 - p) / total)
        assert abs(persisted / total - p) < 4 * se, (inside, persisted / total, p)


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["simulations"]) if R else [])
def test_dynamic_simulations_match_tergm(name):
    """The time averages of the monitored statistics after a burn-in, against
    tergm's simulations of the same models, by batch means."""
    r = R["simulations"][name]
    sim = ergmx.simulate_dynamic(ig.Graph(n=r["n"]), r["formula"], r["coef"], r["slices"], seed=1,
                                 monitor="edges + edges.ageinterval(c(1, 3, 7), c(3, 7, Inf)) + mean.age + "
                                         "degree.mean.age(1:2)")
    values = np.column_stack([sim.monitor[n] for n in r["names"]])[r["burnin"]:]
    batches = values[: len(values) // 100 * 100].reshape(-1, 100, values.shape[1]).mean(axis=1)
    se = batches.std(axis=0, ddof=1) / np.sqrt(len(batches))
    z = (values.mean(axis=0) - np.array(r["mean"])) / np.sqrt(se**2 + np.array(r["se"]) ** 2)
    assert np.all(np.abs(z) < 4), dict(zip(r["names"], np.round(z, 2)))


def test_durational_terms_need_a_dynamic_model():
    g = load("flomarriage")
    for call in (lambda: ergmx.ergm(g, "edges + mean.age"),
                 lambda: ergmx.tergm([g, g], "Form(~edges) + Persist(~edges + edges.ageinterval(2))"),
                 lambda: ergmx.simulate(g, "edges + edge.ages", [-1.0, 0.1]),
                 lambda: ergmx.summary_stats(g, "edges + mean.age")):
        with pytest.raises(ValueError, match="statistic of tie ages, which needs the ties' history"):
            call()
    with pytest.raises(ValueError, match="dyad-independent"):
        ergmx.simulate_dynamic(g, "Form(~edges) + Persist(~edges + EdgeAges(~triangle))", [-3.0, 1.0, 0.0], 2)
    with pytest.raises(ValueError, match="undirected"):
        ergmx.simulate_dynamic(load("samplk1"), "Form(~edges) + Persist(~edges + degree.mean.age(1))",
                               [-3.0, 1.0, 0.0], 2)


def test_starting_ages():
    """With every tie persisting and none forming, a step adds one to the
    ages the starting network's ties are given (or to 1)."""
    g = load("flomarriage")
    g.es["age"] = list(range(1, g.ecount() + 1))
    model = "Form(~edges) + Persist(~edges)"
    sim = ergmx.simulate_dynamic(g, model, [-30.0, 30.0], 2, seed=1, ages="age", monitor="mean.age + edge.ages")
    assert sim.monitor["edge.ages"].tolist() == [sum(g.es["age"]) + g.ecount() * k for k in (1, 2)]
    sim = ergmx.simulate_dynamic(g, model, [-30.0, 30.0], 1, seed=1, monitor="mean.age")
    assert sim.monitor["mean.age"].tolist() == [2.0]
    g.es["age"] = [0] * g.ecount()
    with pytest.raises(ValueError, match="1 or more"):
        ergmx.simulate_dynamic(g, model, [-3.0, 3.0], 1, ages="age")
