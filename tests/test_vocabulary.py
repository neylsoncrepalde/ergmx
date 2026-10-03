"""More of ergm's vocabulary: terms, interactions and constraints.

Their statistics, MPLEs and MLEs are compared with R in test_statistics.py and
test_estimation.py (the reference models *_vocab*, *_dyadind and those of the
constraints). Here: the statistics the sampler tracks for every new term,
what each constraint keeps, ergm's documented definitions where ergm computes
something else, and errors."""

import json
import warnings

import igraph as ig
import numpy as np
import pytest
from conftest import DATA, REFERENCE, load

import ergmx
from ergmx.terms import ErgmDifferenceWarning

DIFFERENCES = json.loads((DATA / "r_reference.json").read_text())["ergm_differences"]

TRACKED = [
    ("faux.mesa.high", None,
     "edges + degrange(1:2, 4) + degree(1:2, by='Race') + degree(0:1, by='Sex', homophily=TRUE) + "
     "concurrent(by='Sex') + concurrentties(by='Sex') + degree1.5 + isolatededges + altkstar(2, fixed=TRUE) + "
     "triadcensus(0:3) + threetrail + opentriad + transitiveties(attr='Race') + cyclicalties + "
     "kstar(2:3, attr='Sex') + triangle(attr='Grade', diff=TRUE) + nodecovrange('Grade') + "
     "nodefactordistinct('Race') + gwdegree(0.5, fixed=TRUE, attr='Sex') + localtriangle('nbhd_mesa') + "
     "degrange(1, 3, by='Race', homophily=TRUE) + balance"),
    ("faux.dixon.high", None,
     "edges + triadcensus + simmelianties + transitiveties(attr='sex') + cyclicalties(attr='race') + threetrail + "
     "idegrange(1, 3, by='race', homophily=TRUE) + odegree1.5 + nodeicovrange('grade') + nodecovrange('grade') + "
     "mutual(by='race') + mutual(same='race', diff=TRUE) + asymmetric(attr='sex', diff=TRUE) + "
     "ostar(2:3, attr='race') + nodefactordistinct('race') + nodeifactordistinct('race') + "
     "ttriple(attr='sex', diff=TRUE) + ctriple(attr='race') + gwodegree(0.5, fixed=TRUE, attr='race') + "
     "triangle(attr='race') + balance"),
    ("samplk3", None, "edges + threetrail + hamming('wave2') + localtriangle('nbhd_samplk') + simmelianties"),
    ("faux.mesa.high", None, "edges + tripercent + tripercent('Grade') + tripercent('Sex', diff=TRUE)"),
    ("davis", "type", "edges + coincidence(levels=c(1, 2, 5, 40))"),
    ("bipartite_sim", "type",
     "edges + b1twostar('g') + b2starmix(2, 'g') + b1nodematch('g', beta=0.5) + b2nodematch('g', alpha=0.25) + "
     "b1nodematch('g', diff=TRUE, byb2attr='g') + b2degrange(0, 2, by='g', homophily=TRUE) + b1covrange('x') + "
     "b2factordistinct('g') + isolatededges + b2star(2, attr='g') + b1starmix(2, 'g', diff=FALSE) + "
     "gwb1degree(0.5, fixed=TRUE, attr='g')"),
]


@pytest.mark.parametrize(("network", "bipartite", "formula"), TRACKED)
def test_new_terms_tracked_by_the_sampler_are_exact(network, bipartite, formula):
    """Recomputing the statistics of sampled networks from scratch gives those
    the sampler updated, toggle by toggle, with the change statistics."""
    g = load(network)
    bip = {"bipartite": bipartite} if bipartite else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        names = list(ergmx.summary_stats(g, formula, **bip))
        if bipartite:
            n1 = sum(not t for t in g.vs[bipartite])
            dyads = n1 * (g.vcount() - n1)
        else:
            dyads = g.vcount() * (g.vcount() - 1) / (1 if g.is_directed() else 2)
        p = g.ecount() / dyads
        coef = [np.log(p / (1 - p)) if name == "edges" else 0.0 for name in names]
        tracked = ergmx.simulate(g, formula, coef, 15, seed=4, interval=4096, output="stats", **bip)
        networks = ergmx.simulate(g, formula, coef, 15, seed=4, interval=4096, **bip)
        recomputed = [list(ergmx.summary_stats(h, formula, **bip).values()) for h in networks]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9, atol=1e-9)
    assert len({h.ecount() for h in networks}) > 1


def test_interactions_multiply_change_statistics():
    g = load("faux.mesa.high")
    stats = ergmx.summary_stats(g, "nodecov('Grade')*nodematch('Sex')")
    assert list(stats) == ["nodecov.Grade", "nodematch.Sex", "nodecov.Grade:nodematch.Sex"]
    grade = np.array(g.vs["Grade"])
    same = [(grade[a] + grade[b]) for a, b in g.get_edgelist() if g.vs[a]["Sex"] == g.vs[b]["Sex"]]
    assert stats["nodecov.Grade:nodematch.Sex"] == pytest.approx(sum(same))
    # Several statistics on each side: the first term's vary fastest, as in ergm.
    names = list(ergmx.summary_stats(g, "(nodefactor('Sex') + edges):nodefactor('Race', levels=c('Hisp','White'))"))
    assert names == ["nodefactor.Sex.M:nodefactor.Race.Hisp", "edges:nodefactor.Race.Hisp",
                     "nodefactor.Sex.M:nodefactor.Race.White", "edges:nodefactor.Race.White"]
    with pytest.raises(ValueError, match="dyad-independent"):
        ergmx.summary_stats(g, "edges:triangle")


def test_r_syntax_in_formulas():
    g = load("faux.mesa.high")
    stats = ergmx.summary_stats(g, "degree1.5 + diff('Grade', pow=2, sign.action='abs') + altkstar(lambda=2, fixed=TRUE)")
    assert list(stats) == ["degree1.5", "diff2.abs.Grade", "altkstar.2"]
    assert ergmx.summary_stats(g, "triangles + nodemain('Grade') + threepath") == \
        ergmx.summary_stats(g, "triangle + nodecov('Grade') + threetrail")
    assert ergmx.summary_stats(g, ergmx.degree1_5()) == {"degree1.5": stats["degree1.5"]}


def test_ergm_documents_intransitive_as_triads():
    """intransitive counts the triads ergm documents (111D, 201, 111U, 021C,
    030C); ergm computes twopath - ttriple."""
    for name, record in DIFFERENCES["intransitive"].items():
        with pytest.warns(ErgmDifferenceWarning, match="intransitive triples"):
            value = ergmx.summary_stats(load(name), "intransitive")["intransitive"]
        assert value == record["triads"]
        stats = ergmx.summary_stats(load(name), "twopath + ttriple")
        assert stats["twopath"] - stats["ttriple"] == record["ergm_intransitive"] == record["twopath_minus_ttriple"]


def test_dyadcov_states_as_ergm_documents():
    """utri is the dyads whose only tie is in the adjacency matrix's upper
    triangle (lower- to higher-numbered vertex); ergm counts them as ltri."""
    g = ig.Graph(n=3, edges=[(0, 1)], directed=True)
    g["ones"] = np.ones((3, 3))
    with pytest.warns(ErgmDifferenceWarning, match="other way round"):
        ours = ergmx.summary_stats(g, "dyadcov('ones')")
    r = DIFFERENCES["dyadcov_lower_to_higher_tie"]
    assert ours == {"dyadcov.ones.mutual": 0, "dyadcov.ones.utri": 1, "dyadcov.ones.ltri": 0}
    assert (r["dyadcov.ones.utri"], r["dyadcov.ones.ltri"]) == (0, 1)


def test_missing_attribute_values_are_refused():
    g = load("faux.mesa.high")
    race = g.vs["Race"]
    race[3] = None
    g.vs["Race"] = race
    for formula in ("nodefactor('Race')", "nodematch('Race')", "nodemix('Race')", "degree(1, by='Race')"):
        with pytest.raises(ValueError, match="missing values"):
            ergmx.summary_stats(g, formula)


def test_unsupported_options_say_so():
    g = load("faux.mesa.high")
    with pytest.raises(NotImplementedError, match="gwdegree"):
        ergmx.summary_stats(g, "altkstar(2)")
    with pytest.raises(NotImplementedError, match="fixed=TRUE"):
        ergmx.summary_stats(g, "gwdegree(0.5, attr='Sex')")
    with pytest.raises(ValueError, match="triad type"):
        ergmx.summary_stats(g, "triadcensus(levels=7)")
    with pytest.raises(ValueError, match="bipartite"):
        ergmx.summary_stats(g, "b1degrange(1)")


# -- Constraints --------------------------------------------------------------------------------


def _simulate(g, formula, coef, constraints, **options):
    return ergmx.simulate(g, formula, coef, nsim=10, constraints=constraints, seed=2, interval=2048, **options)


def test_edges_constraint_keeps_the_number_of_edges():
    g = load("faux.mesa.high")
    stats = _simulate(g, "edges + triangle + nodematch('Grade')", [0.0, 0.2, 0.5], "edges", output="stats")
    assert set(stats[:, 0]) == {g.ecount()}
    assert len(set(stats[:, 2])) > 1


def test_mode_degrees_are_kept():
    g = load("bipartite_sim")
    for constraint, kept, free in (("b1degrees", False, True), ("b2degrees", True, False)):
        sims = _simulate(g, "edges + b1star(2) + b2star(2)", [-3.0, 0.1, 0.1], constraint, bipartite="type")
        for h in sims:
            assert h.degree(g.vs.select(type=kept)) == g.degree(g.vs.select(type=kept))
        assert any(h.degree(g.vs.select(type=free)) != g.degree(g.vs.select(type=free)) for h in sims)
    with pytest.raises(ValueError, match="bipartite"):
        ergmx.simulate(load("flomarriage"), "edges", [-1.0], constraints="b1degrees")


def test_bounds_by_alter_attribute_hold():
    g = load("faux.dixon.high")
    classes = np.asarray(g["sexmat"], dtype=bool)
    sims = _simulate(g, "edges + mutual", [-3.5, 2.0], "bd(attribs='sexmat', maxout='maxsex')")
    for h in sims:
        counts = np.array(h.get_adjacency().data) @ classes
        assert (counts <= g["maxsex"]).all()
    tight = np.array(g.get_adjacency().data) @ classes - 1
    g["tight"] = tight
    with pytest.raises(ValueError, match="violates"):
        ergmx.simulate(g, "edges", [-3.0], constraints="bd(attribs='sexmat', maxout='tight')")


def test_fixed_dyads_keep_their_values():
    g = load("flomarriage")
    ties = {tuple(sorted(e)) for e in g.get_edgelist()}

    def dyads(h):
        return {tuple(sorted(e)) for e in h.get_edgelist()}

    present, absent = [(1, 9), (2, 6)], [(1, 2), (3, 4)]  # R's vertex numbers
    assert {(a - 1, b - 1) for a, b in present} <= ties
    c = f"fixedas(present=matrix(c(1,9,2,6), ncol=2, byrow=TRUE), absent={[[1, 2], [3, 4]]})"
    for h in _simulate(g, "edges", [-1.0], c):
        assert {(a - 1, b - 1) for a, b in present} <= dyads(h)
        assert not {(a - 1, b - 1) for a, b in absent} & dyads(h)
    with pytest.raises(ValueError, match="not ties"):
        ergmx.simulate(g, "edges", [-1.0], constraints="fixedas(present=matrix(c(1, 2), ncol=2))")
    free = [(0, 1), (0, 2), (3, 4), (5, 6)]
    sims = _simulate(g, "edges", [0.0], f"fixallbut({[[a + 1, b + 1] for a, b in free]})")
    for h in sims:
        assert dyads(h) - set(free) == ties - set(free)
    assert any(dyads(h) != ties for h in sims)


def test_dyads_and_blockdiag_constraints():
    g = load("faux.mesa.high")
    sex = np.array(g.vs["Sex"])
    for h in _simulate(g, "edges", [-3.0], "Dyads(fix=~nodematch('Sex'))"):
        same = {tuple(sorted(e)) for e in h.get_edgelist() if sex[e[0]] == sex[e[1]]}
        assert same == {tuple(sorted(e)) for e in g.get_edgelist() if sex[e[0]] == sex[e[1]]}
    within = load("faux.mesa.within")
    grade = np.array(within.vs["Grade"])
    for h in _simulate(within, "edges", [-3.0], "blockdiag('Grade')"):
        assert all(grade[a] == grade[b] for a, b in h.get_edgelist())
    with pytest.raises(ValueError, match="between blocks"):
        ergmx.simulate(g, "edges", [-3.0], constraints="blockdiag('Grade')")


def test_observed_constraint_only_varies_missing_dyads():
    g = load("faux.mesa.high.missing")
    observed = {tuple(sorted(e.tuple)) for e in g.es if not e["na"]}
    for h in _simulate(g, "edges", [-3.0], "observed"):
        # Ties of observed dyads are kept; other observed dyads stay empty.
        assert observed <= {tuple(sorted(e)) for e in h.get_edgelist()}


def test_constrained_reference_models_are_in_the_reference():
    for name in ("mesa_dyads_fix", "mesa_dyads_vary", "flo_fixedas", "flo_fixallbut", "mesa_blockdiag",
                 "mesa_edges_constraint", "bip_b1degrees", "dixon_bd_attribs"):
        assert name in REFERENCE


def test_degcor_and_degcrossprod_are_linearized_at_the_network():
    """Their values are the network's correlation and mean, but their change
    statistics, as ergm's, are those of the sum over ties of the product of
    the degrees, scaled at the network."""
    g = load("flomarriage")
    degree = np.array(g.degree(), dtype=float)
    a, b = np.array(g.get_edgelist()).T
    ends = np.concatenate([degree[a], degree[b]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        stats = ergmx.summary_stats(g, "degcor + degcrossprod")
        assert stats["degcor"] == pytest.approx(np.corrcoef(ends, np.concatenate([degree[b], degree[a]]))[0, 1])
        assert stats["degcrossprod"] == pytest.approx(np.mean(degree[a] * degree[b]))
        sims = ergmx.simulate(g, "edges + degcor + degcrossprod", [-1.5, 0, 0], 5, seed=1)
        tracked = ergmx.simulate(g, "edges + degcor + degcrossprod", [-1.5, 0, 0], 5, seed=1, output="stats")
    squares = np.sum((ends - ends.mean()) ** 2)
    for h, row in zip(sims, tracked):
        d = np.array(h.degree(), dtype=float)
        x, y = np.array(h.get_edgelist()).T
        cross = np.sum(d[x] * d[y])
        assert row[1] == pytest.approx(2 / squares * cross - ends.mean() ** 2 / (squares / len(ends)))
        assert row[2] == pytest.approx(cross / g.ecount())


def test_degcrossprod_and_coincidence_follow_ergms_documentation():
    r = DIFFERENCES["degcrossprod"]
    with pytest.warns(ErgmDifferenceWarning, match="half"):
        ours = ergmx.summary_stats(load("flomarriage"), "degcrossprod")["degcrossprod"]
    assert ours == pytest.approx(r["mean"]) == pytest.approx(2 * r["ergm_degcrossprod"])
    r = DIFFERENCES["coincidence_active"]
    with pytest.warns(ErgmDifferenceWarning, match="at least"):
        names = list(ergmx.summary_stats(load("davis"), f"coincidence(active={r['active']})", bipartite="type"))
    assert names == r["at_least"] and set(r["ergm"]) < set(names)


def test_tripercent_counts_triangles_among_two_paths():
    g = load("faux.mesa.high")
    stats = ergmx.summary_stats(g, "tripercent + triangle + kstar(2)")
    t, s = stats["triangle"], stats["kstar2"]
    assert stats["tripercent"] == pytest.approx(100 * t / (s - 2 * t))
    assert ergmx.summary_stats(ig.Graph.Ring(5), "tripercent")["tripercent"] == 0
    with pytest.raises(ValueError, match="undirected"):
        ergmx.summary_stats(load("samplk3"), "tripercent")
    with pytest.raises(ValueError, match="bipartite"):
        ergmx.summary_stats(g, "coincidence")


@pytest.mark.parametrize(("network", "constraint", "kept", "free"), [
    ("faux.mesa.high", "degreedist", ("all",), ("all",)),
    ("faux.dixon.high", "degreedist", ("out", "in"), ("out", "in")),
    ("faux.dixon.high", "odegreedist", ("out",), ("out", "in")),
    ("faux.dixon.high", "idegreedist", ("in",), ("out", "in")),
    ("bipartite_sim", "degreedist", ("all",), ("all",)),
])
def test_degree_distributions_are_kept_but_not_the_degrees(network, constraint, kept, free):
    """ergm documents degreedist as keeping the degree distribution: the
    vertices' degrees change, and, in directed networks, both distributions
    are kept; odegreedist and idegreedist keep only theirs."""
    from collections import Counter

    g = load(network)
    bip = {"bipartite": "type"} if network == "bipartite_sim" else {}
    sims = ergmx.simulate(g, "edges + triangle" if not bip else "edges", [0.0, 0.1] if not bip else [0.0],
                          5, seed=3, constraints=constraint, interval=10000, **bip)
    for h in sims:
        assert h.ecount() == g.ecount()
        for mode in kept:
            assert Counter(h.degree(mode=mode)) == Counter(g.degree(mode=mode)), mode
        for mode in free:
            assert h.degree(mode=mode) != g.degree(mode=mode), mode
        if bip:
            t = g.vs["type"]
            assert all(t[a] != t[b] for a, b in h.get_edgelist())
            for side in (False, True):
                assert Counter(d for d, x in zip(h.degree(), t) if x == side) == \
                    Counter(d for d, x in zip(g.degree(), t) if x == side)


def test_degree_distribution_statistics_are_constant_under_degreedist():
    g = load("faux.mesa.high")
    with pytest.warns(UserWarning, match="constant under the constraints"):
        fit = ergmx.ergm(g, "edges + kstar(2) + degree(1:2) + gwdegree(0.5, fixed=TRUE) + nodematch('Grade') "
                            "+ sociality(nodes=c(2, 5))", constraints="degreedist", seed=1, eval_loglik=False)
    assert all(fit.coef[k] == 0 for k in ("edges", "kstar2", "degree1", "degree2", "gwdeg.fixed.0.5"))
    assert fit.coef["nodematch.Grade"] != 0 and fit.coef["sociality2"] != 0


def test_egocentric_fixes_the_egos_dyads():
    g = load("faux.dixon.high")
    ego = np.array(g.vs["ego"], dtype=bool)
    ties = set(g.get_edgelist())
    for direction, fixed in (("both", lambda a, b: ego[a] or ego[b]), ("out", lambda a, b: ego[a]),
                             ("in", lambda a, b: ego[b])):
        for h in _simulate(g, "edges", [-3.0], f"egocentric('ego', direction='{direction}')"):
            mine = set(h.get_edgelist())
            assert {e for e in mine if fixed(*e)} == {e for e in ties if fixed(*e)}
            assert mine != ties
    with pytest.raises(ValueError, match="direction"):
        ergmx.simulate(load("faux.mesa.high"), "edges", [-3.0], constraints="egocentric('ego', direction='out')")
    with pytest.raises(ValueError, match="logical"):
        ergmx.simulate(g, "edges", [-3.0], constraints="egocentric('grade')")
