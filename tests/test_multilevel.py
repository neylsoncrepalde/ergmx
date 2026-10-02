"""MPNet's multilevel configurations (Wang, Robins, Pattison and Lazega 2013),
which no R package has: against independent matrix formulas, and the sampler
against exact enumeration. S(), which ergm has, is compared with R in
test_statistics.py and test_estimation.py."""

import itertools

import igraph as ig
import numpy as np
import pytest

import ergmx
from ergmx._model import bind


def _g(d, decay):
    return np.exp(decay) * (1 - (1 - np.exp(-decay)) ** d)


def expected(g: ig.Graph, attr: str, a, b, decay: float) -> dict:
    """Every configuration, from the adjacency matrices of the two levels."""
    m = np.array(g.get_adjacency().data)
    level = np.array(g.vs[attr], dtype=object)
    ia, ib = np.flatnonzero(level == a), np.flatnonzero(level == b)
    A, B, X = m[np.ix_(ia, ia)], m[np.ix_(ib, ib)], m[np.ix_(ia, ib)]
    dA, dB, dXa, dXb = A.sum(1), B.sum(1), X.sum(1), X.sum(0)
    sa, sb = X @ X.T, X.T @ X  # shared partners at the other level
    up_a, up_b = np.triu(A, 1), np.triu(B, 1)
    g_ = lambda d: _g(d, decay)  # noqa: E731
    suffix = f".{attr}"
    alt = f"{suffix}.{decay:g}"
    return {
        "Star2AX" + suffix: (dA * dXa).sum(), "Star2BX" + suffix: (dB * dXb).sum(),
        "AXS1A" + alt: (dA * g_(dXa)).sum(), "AXS1B" + alt: (dB * g_(dXb)).sum(),
        "AAS1X" + alt: (g_(dA) * dXa).sum(), "ABS1X" + alt: (g_(dB) * dXb).sum(),
        "AAAXS" + alt: (g_(dA) * g_(dXa)).sum(), "ABAXS" + alt: (g_(dB) * g_(dXb)).sum(),
        "TXAX" + suffix: (up_a * sa).sum(), "TXBX" + suffix: (up_b * sb).sum(),
        "ATXAX" + alt: (up_a * g_(sa)).sum(), "ATXBX" + alt: (up_b * g_(sb)).sum(),
        "L3XAX" + suffix: (up_a * np.outer(dXa, dXa)).sum(), "L3XBX" + suffix: (up_b * np.outer(dXb, dXb)).sum(),
        "L3AXB" + suffix: (X * np.outer(dA, dB)).sum(),
        "C4AXB" + suffix: np.trace(A @ X @ B @ X.T) / 2,
        "EXTA" + suffix: (np.diag(A @ A @ A) / 2 * dXa).sum(), "EXTB" + suffix: (np.diag(B @ B @ B) / 2 * dXb).sum(),
        "ASAXASB" + alt: (X * np.outer(g_(dA), g_(dB))).sum(),
    }


def formula(attr, decay, levels=None):
    lv = "" if levels is None else f", levels={list(levels)!r}"
    plain = ["star2ax", "star2bx", "txax", "txbx", "l3xax", "l3xbx", "l3axb", "c4axb", "exta", "extb"]
    alternating = ["axs1a", "axs1b", "aas1x", "abs1x", "aaaxs", "abaxs", "atxax", "atxbx", "asaxasb"]
    return " + ".join([f"{t}('{attr}'{lv})" for t in plain] +
                      [f"{t}('{attr}', decay={float(decay)!r}{lv})" for t in alternating])


def random_two_level(seed, n=24, p=(0.25, 0.3, 0.15), others=3):
    rng = np.random.default_rng(seed)
    levels = ["lab"] * (n // 2) + ["researcher"] * (n - n // 2) + ["other"] * others
    g = ig.Graph(n=len(levels))
    g.vs["level"] = levels
    edges = []
    for i, j in itertools.combinations(range(len(levels)), 2):
        kind = 0 if levels[i] == levels[j] == "lab" else 1 if levels[i] == levels[j] == "researcher" else 2
        if rng.random() < p[kind]:
            edges.append((i, j))
    g.add_edges(edges)
    return g


@pytest.mark.parametrize("decay", [0.5, np.log(2), 1.3])
def test_configurations_match_their_definitions(decay):
    for seed in range(3):
        g = random_two_level(seed)
        stats = ergmx.summary_stats(g, formula("level", decay, ("lab", "researcher")))
        truth = expected(g, "level", "lab", "researcher", decay)
        assert set(stats) == set(truth)
        for name, value in truth.items():
            assert stats[name] == pytest.approx(value, rel=1e-12, abs=1e-9), name


def test_bundled_multilevel_network():
    g = ergmx.datasets.load("linked_sim")
    stats = ergmx.summary_stats(g, formula("level", 0.7))
    truth = expected(g, "level", "individual", "organization", 0.7)
    for name, value in truth.items():
        assert stats[name] == pytest.approx(value, rel=1e-12), name


def test_statistics_tracked_by_the_sampler_are_exact():
    g = random_two_level(5)
    f = "edges + " + formula("level", 0.6, ("lab", "researcher"))
    model = bind(g, f)
    theta = np.zeros(model.n_stats)
    theta[0] = -1.0
    sample, _, networks = model.simulate([model.network.edges], theta, 2000, 500, 20, 1, keep_networks=True,
                                         triadic_weight=0.5)
    for stats, edges in zip(sample[0], networks[0]):
        np.testing.assert_allclose(stats, model.core.summary(edges), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_sampler_matches_exact_enumeration(weight):
    """Every network on 3 A and 2 B vertices (and one of neither level)."""
    g = ig.Graph(n=6, edges=[(0, 1), (0, 3), (1, 4), (3, 4), (2, 5)])
    g.vs["level"] = ["A", "A", "A", "B", "B", "C"]
    f = ("edges + star2ax('level', levels=['A', 'B']) + atxax('level', decay=0.8, levels=['A', 'B']) + "
         "l3axb('level', levels=['A', 'B']) + c4axb('level', levels=['A', 'B']) + "
         "abs1x('level', decay=0.5, levels=['A', 'B']) + S(~triangle, ~level != 'C')")
    model = bind(g, f)
    theta = np.array([-0.8, 0.3, 0.4, -0.2, 0.5, 0.3, 0.2])
    pairs = list(itertools.combinations(range(6), 2))
    networks = [np.array([p for p, x in zip(pairs, bits) if x], dtype=np.uint32).reshape(-1, 2)
                for bits in itertools.product((0, 1), repeat=len(pairs))]
    stats = np.array([model.core.summary(e) for e in networks])
    logp = stats @ theta
    p = np.exp(logp - logp.max())
    truth = (p / p.sum()) @ stats
    runs = np.array([model.simulate([model.network.edges] * 4, theta, 1000, 10, 800, seed,
                                    triadic_weight=weight)[0].reshape(-1, len(theta)).mean(axis=0)
                     for seed in range(12)])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(runs.mean(axis=0) - truth) < 5 * se + 1e-9)


def test_levels_and_errors():
    g = ergmx.datasets.load("linked_sim")
    assert list(ergmx.summary_stats(g, "txax('level')")) == ["TXAX.level"]
    swapped = ergmx.summary_stats(g, "txax('level', levels=['organization', 'individual'])")
    assert swapped["TXAX.level"] == ergmx.summary_stats(g, "txbx('level')")["TXBX.level"]
    with pytest.raises(ValueError, match="3 levels"):
        ergmx.summary_stats(random_two_level(1), "txax('level')")
    with pytest.raises(ValueError, match="undirected"):
        ergmx.summary_stats(ergmx.datasets.load("samplk3"), "txax('group', levels=['Turks', 'Loyal'])")
    with pytest.raises(ValueError, match="no vertices"):
        ergmx.summary_stats(g, "txax('level', levels=['individual', 'lab'])")


def test_subgraph_operator():
    g = ergmx.datasets.load("linked_sim")
    individuals = [v.index for v in g.vs if v["level"] == "individual"]
    inside = g.induced_subgraph(individuals)
    ours = ergmx.summary_stats(g, "S(~edges + triangle + gwesp(0.5, fixed=TRUE), ~level == 'individual')")
    alone = ergmx.summary_stats(inside, "edges + triangle + gwesp(0.5, fixed=TRUE)")
    assert list(ours.values()) == pytest.approx(list(alone.values()))
    # By vertex indices (1-based, as in R), and negative ones to leave out.
    by_index = ergmx.summary_stats(g, ergmx.S("edges", [i + 1 for i in individuals]))
    others = [-(v.index + 1) for v in g.vs if v["level"] != "individual"]
    assert list(by_index.values()) == list(ergmx.summary_stats(g, ergmx.S("edges", others)).values()) \
        == [inside.ecount()]
    with pytest.raises(ValueError, match="disjoint"):
        ergmx.summary_stats(g, "S(~edges, (level == 'individual') ~ (type | !type))")
    # In a directed network, the arcs from the first set to the second, as ergm (R gives 0, 7 and 2).
    stats = ergmx.summary_stats(ergmx.datasets.load("samplk3"),
                                "S(~edges + b1degree(0), (group == 'Turks') ~ (group == 'Loyal')) + "
                                "S(~edges, (group == 'Loyal') ~ (group == 'Turks'))")
    assert list(stats.values()) == [0, 7, 2]


def test_multilevel_fit():
    g = ergmx.datasets.load("linked_sim")
    f = ("S(~edges + gwesp(0.5, fixed=TRUE), ~level == 'individual') + S(~edges, ~level == 'organization') + "
         "S(~edges, (level == 'individual') ~ (level == 'organization')) + star2ax('level') + txax('level') + "
         "c4axb('level')")
    fit = ergmx.ergm(g, f, seed=1, eval_loglik=False)
    assert fit.converged
    assert "C4AXB.level" in fit.coef and np.isfinite(fit.stderr["C4AXB.level"])


def test_labs_sim_recovers_its_known_model():
    """labs_sim is simulated from a known model (scripts/simulate_labs.py):
    fitted given its affiliations, every estimate is near its true value."""
    g = ergmx.datasets.load("labs_sim")
    level = np.array(g.vs["level"])
    researchers, labs = np.flatnonzero(level == "researcher"), np.flatnonzero(level == "laboratory")
    affiliations = [len([u for u in g.neighbors(r) if u in set(labs)]) for r in researchers]
    assert (len(researchers), len(labs), sum(affiliations)) == (120, 30, 150)
    assert set(affiliations) == {1, 2}
    truth = {
        'S(level=="researcher")~edges': -3.6,
        'S(level=="researcher")~gwesp.fixed.0.693147': 0.3,
        'S(level=="laboratory")~edges': -2.8,
        "TXBX.level": 1.5,
        "TXAX.level": 1.5,
        "C4AXB.level": 0.4,
    }
    formula = ("S(~edges + gwesp(0.693147, fixed=TRUE), ~level == 'researcher') + S(~edges, ~level == 'laboratory')"
               " + txbx('level') + txax('level') + c4axb('level')")
    fit = ergmx.ergm(g, formula, constraints="blocks('level', levels2=2)", seed=1, eval_loglik=False)
    assert fit.converged and list(fit.coef) == list(truth)
    for name, value in truth.items():
        assert abs(fit.coef[name] - value) < 2.5 * fit.stderr[name], name


# -- Directed two-level networks ------------------------------------------------------------------


def expected_directed(g: ig.Graph, attr: str, a, b, decay: float) -> dict:
    """MPNet's directed configurations, from the adjacency matrices: A and B
    the arcs within each level, X the arcs from A to B."""
    m = np.array(g.get_adjacency().data)
    level = np.array(g.vs[attr], dtype=object)
    ia, ib = np.flatnonzero(level == a), np.flatnonzero(level == b)
    A, B, X = m[np.ix_(ia, ia)], m[np.ix_(ib, ib)], m[np.ix_(ia, ib)]
    inA, outA, inB, outB = A.sum(0), A.sum(1), B.sum(0), B.sum(1)
    xA, xB = X.sum(1), X.sum(0)
    SA, SB = X @ X.T, X.T @ X
    MA, MB = A * A.T, B * B.T
    C = X @ B @ X.T

    def g_(d):
        return _g(d, decay)

    s, alt = f".{attr}", f".{attr}.{decay:g}"
    return {
        "In2StarAX" + s: (inA * xA).sum(), "In2StarBX" + s: (inB * xB).sum(),
        "Out2StarAX" + s: (outA * xA).sum(), "Out2StarBX" + s: (outB * xB).sum(),
        "AXS1Ain" + alt: (inA * g_(xA)).sum(), "AXS1Bin" + alt: (inB * g_(xB)).sum(),
        "AXS1Aout" + alt: (outA * g_(xA)).sum(), "AXS1Bout" + alt: (outB * g_(xB)).sum(),
        "AAinS1X" + alt: (g_(inA) * xA).sum(), "ABinS1X" + alt: (g_(inB) * xB).sum(),
        "AAoutS1X" + alt: (g_(outA) * xA).sum(), "ABoutS1X" + alt: (g_(outB) * xB).sum(),
        "TXAXarc" + s: (A * SA).sum(), "TXBXarc" + s: (B * SB).sum(),
        "TXAXreciprocity" + s: (np.triu(MA, 1) * SA).sum(), "TXBXreciprocity" + s: (np.triu(MB, 1) * SB).sum(),
        "ATXAXarc" + alt: (A * g_(SA)).sum(), "ATXBXarc" + alt: (B * g_(SB)).sum(),
        "ATXAXreciprocity" + alt: (np.triu(MA, 1) * g_(SA)).sum(),
        "ATXBXreciprocity" + alt: (np.triu(MB, 1) * g_(SB)).sum(),
        "L3XAX" + s: (A * np.outer(xA, xA)).sum(), "L3XBX" + s: (B * np.outer(xB, xB)).sum(),
        "L3XAXreciprocity" + s: (np.triu(MA, 1) * np.outer(xA, xA)).sum(),
        "L3XBXreciprocity" + s: (np.triu(MB, 1) * np.outer(xB, xB)).sum(),
        "L3AXBin" + s: (X * np.outer(inA, inB)).sum(), "L3AXBout" + s: (X * np.outer(outA, outB)).sum(),
        "L3AXBpath" + s: (X * np.outer(inA, outB)).sum(), "L3BXApath" + s: (X * np.outer(outA, inB)).sum(),
        "C4AXBentrainment" + s: (A * C).sum(), "C4AXBexchange" + s: (A * C.T).sum(),
        "C4AXBexchangeAreciprocity" + s: (MA * C).sum(),
        "C4AXBexchangeBreciprocity" + s: (A * (X @ MB @ X.T)).sum(),
        "C4AXBreciprocity" + s: (MA * (X @ MB @ X.T)).sum() / 2,
        "AinASXAinBS" + alt: (X * np.outer(g_(inA), g_(inB))).sum(),
        "AoutASXAoutBS" + alt: (X * np.outer(g_(outA), g_(outB))).sum(),
        "AinASXAoutBS" + alt: (X * np.outer(g_(inA), g_(outB))).sum(),
        "AoutASXAinBS" + alt: (X * np.outer(g_(outA), g_(inB))).sum(),
    }


DIRECTED_PLAIN = ["in2starax", "in2starbx", "out2starax", "out2starbx", "txaxarc", "txbxarc", "txaxreciprocity",
                  "txbxreciprocity", "l3xax", "l3xbx", "l3xaxreciprocity", "l3xbxreciprocity", "l3axbin", "l3axbout",
                  "l3axbpath", "l3bxapath", "c4axbentrainment", "c4axbexchange", "c4axbexchangeareciprocity",
                  "c4axbexchangebreciprocity", "c4axbreciprocity"]
DIRECTED_ALTERNATING = ["axs1ain", "axs1bin", "axs1aout", "axs1bout", "aains1x", "abins1x", "aaouts1x", "abouts1x",
                        "atxaxarc", "atxbxarc", "atxaxreciprocity", "atxbxreciprocity", "ainasxainbs", "aoutasxaoutbs",
                        "ainasxaoutbs", "aoutasxainbs"]


def directed_formula(attr, decay, levels=None):
    lv = "" if levels is None else f", levels={list(levels)!r}"
    return " + ".join([f"{t}('{attr}'{lv})" for t in DIRECTED_PLAIN] +
                      [f"{t}('{attr}', decay={float(decay)!r}{lv})" for t in DIRECTED_ALTERNATING])


def random_directed_two_level(seed, n=(14, 12), others=3, p=(0.2, 0.25, 0.25)):
    """Arcs within each level, affiliations from A to B, and vertices of
    neither level with arcs to anyone."""
    rng = np.random.default_rng(seed)
    levels = ["lab"] * n[0] + ["researcher"] * n[1] + ["other"] * others
    g = ig.Graph(n=len(levels), directed=True)
    g.vs["level"] = levels
    edges = []
    for i, j in itertools.permutations(range(len(levels)), 2):
        li, lj = levels[i], levels[j]
        if li == "researcher" and lj == "lab":
            continue
        kind = 0 if li == lj == "lab" else 1 if li == lj == "researcher" else 2
        if rng.random() < p[kind]:
            edges.append((i, j))
    g.add_edges(edges)
    return g


@pytest.mark.parametrize("decay", [0.5, np.log(2)])
def test_directed_configurations_match_their_definitions(decay):
    for seed in range(3):
        g = random_directed_two_level(seed)
        stats = ergmx.summary_stats(g, directed_formula("level", decay, ("lab", "researcher")))
        truth = expected_directed(g, "level", "lab", "researcher", decay)
        assert set(stats) == set(truth)
        for name, value in truth.items():
            assert stats[name] == pytest.approx(value, rel=1e-12, abs=1e-9), name


def test_directed_statistics_tracked_by_the_sampler_are_exact():
    g = random_directed_two_level(7)
    f = "edges + mutual + " + directed_formula("level", 0.6, ("lab", "researcher"))
    model = bind(g, f, "blocks('level', levels=c('lab', 'researcher'), levels2=2)")
    theta = np.zeros(model.n_stats)
    theta[0] = -1.5
    sample, _, networks = model.simulate([model.network.edges], theta, 2000, 500, 20, 1, keep_networks=True,
                                         triadic_weight=0.5)
    for stats, edges in zip(sample[0], networks[0]):
        np.testing.assert_allclose(stats, model.core.summary(edges), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_directed_sampler_matches_exact_enumeration(weight):
    """Every network on 2 A and 2 B vertices with no arcs from B to A."""
    g = ig.Graph(n=4, edges=[(0, 1), (0, 2), (1, 3), (2, 3)], directed=True)
    g.vs["level"] = ["A", "A", "B", "B"]
    f = ("edges + c4axbentrainment('level') + c4axbexchange('level') + c4axbreciprocity('level') + "
         "txaxarc('level') + atxbxreciprocity('level', decay=0.7) + l3axbpath('level') + aains1x('level', decay=0.5)")
    model = bind(g, f, "blocks('level', levels2=2)")
    theta = np.array([-0.6, 0.5, -0.4, 0.6, 0.3, 0.4, -0.2, 0.3])
    pairs = [(i, j) for i, j in itertools.permutations(range(4), 2) if not (i >= 2 and j < 2)]
    networks = [np.array([p for p, x in zip(pairs, bits) if x], dtype=np.uint32).reshape(-1, 2)
                for bits in itertools.product((0, 1), repeat=len(pairs))]
    stats = np.array([model.core.summary(e) for e in networks])
    logp = stats @ theta
    prob = np.exp(logp - logp.max())
    truth = (prob / prob.sum()) @ stats
    runs = np.array([model.simulate([model.network.edges] * 4, theta, 1000, 10, 800, seed,
                                    triadic_weight=weight)[0].reshape(-1, len(theta)).mean(axis=0)
                     for seed in range(12)])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(runs.mean(axis=0) - truth) < 5 * se + 1e-9)


def test_directed_affiliations_go_from_a_to_b():
    g = random_directed_two_level(2)
    g.add_edge(g.vs.find(level="researcher").index, g.vs.find(level="lab").index)
    with pytest.raises(ValueError, match="arcs from level A"):
        ergmx.summary_stats(g, "in2starax('level', levels=['lab', 'researcher'])")
    with pytest.raises(ValueError, match="directed"):
        ergmx.summary_stats(random_two_level(0), "in2starax('level', levels=['lab', 'researcher'])")
    with pytest.raises(ValueError, match="undirected"):
        ergmx.summary_stats(random_directed_two_level(0), "exta('level', levels=['lab', 'researcher'])")


# -- Estimated decays ------------------------------------------------------------------------------

CURVED_UNDIRECTED = ["axs1a", "axs1b", "aas1x", "abs1x", "atxax", "atxbx"]
CURVED_DIRECTED = ["axs1ain", "axs1bin", "axs1aout", "axs1bout", "aains1x", "abins1x", "aaouts1x", "abouts1x",
                   "atxaxarc", "atxbxarc", "atxaxreciprocity", "atxbxreciprocity"]


@pytest.mark.parametrize(("make", "terms"), [(random_two_level, CURVED_UNDIRECTED),
                                             (random_directed_two_level, CURVED_DIRECTED)])
def test_curved_histograms_weigh_to_the_fixed_statistic(make, terms):
    """With decay alpha, eta . counts is theta times the fixed-decay statistic;
    with a smaller cutoff, the last statistic counts everything above it."""
    from ergmx._model import bind

    g = make(4)
    for term in terms:
        curved = f"{term}('level', decay=0.8, levels=['lab', 'researcher'], fixed=FALSE)"
        fixed = f"{term}('level', decay=0.8, levels=['lab', 'researcher'])"
        model = bind(g, curved)
        counts = model.observed()
        eta = model.eta(np.array([1.0, 0.8]))
        assert eta @ counts == pytest.approx(next(iter(ergmx.summary_stats(g, fixed).values())), rel=1e-12), term
        short = bind(g, curved.replace("fixed=FALSE", "fixed=FALSE, cutoff=2")).observed()
        np.testing.assert_allclose(short, [counts[0], counts[1], counts[2:].sum()], err_msg=term)


def test_curved_multilevel_statistics_tracked_by_the_sampler_are_exact():
    g = random_two_level(5)
    f = "edges + " + " + ".join(f"{t}('level', decay=0.6, levels=['lab', 'researcher'], fixed=FALSE, cutoff=3)"
                                for t in CURVED_UNDIRECTED)
    model = bind(g, f)
    theta = np.zeros(model.n_stats)
    theta[0] = -1.0
    sample, _, networks = model.simulate([model.network.edges], theta, 2000, 500, 20, 1, keep_networks=True,
                                         triadic_weight=0.5, canonical=True)
    for stats, edges in zip(sample[0], networks[0]):
        np.testing.assert_allclose(stats, model.core.summary(edges), rtol=1e-10, atol=1e-9)
    d = random_directed_two_level(5)
    f = "edges + " + " + ".join(f"{t}('level', decay=0.6, levels=['lab', 'researcher'], fixed=FALSE, cutoff=3)"
                                for t in CURVED_DIRECTED)
    model = bind(d, f, "blocks('level', levels=c('lab', 'researcher'), levels2=2)")
    theta = np.zeros(model.n_stats)
    theta[0] = -1.5
    sample, _, networks = model.simulate([model.network.edges], theta, 2000, 500, 20, 1, keep_networks=True,
                                         canonical=True)
    for stats, edges in zip(sample[0], networks[0]):
        np.testing.assert_allclose(stats, model.core.summary(edges), rtol=1e-10, atol=1e-9)


def test_curved_multilevel_mple_is_consistent_with_its_fixed_decay():
    """At the curved MPLE's decay, the fixed-decay MPLE has the same coefficients."""
    g = ergmx.datasets.load("labs_sim")
    given = "blocks('level', levels2=2)"
    base = "S(~edges, ~level == 'researcher') + S(~edges, ~level == 'laboratory') + "
    curved = ergmx.ergm(g, base + "atxbx('level', decay=0.5, fixed=FALSE)", constraints=given, estimate="MPLE")
    decay = curved.mple["ATXBX.level.decay"]
    assert 0.05 < decay < 10
    fixed = ergmx.ergm(g, base + f"atxbx('level', decay={decay!r})", constraints=given, estimate="MPLE")
    for name in ("S(level==\"researcher\")~edges", "S(level==\"laboratory\")~edges"):
        assert fixed.mple[name] == pytest.approx(curved.mple[name], abs=1e-6)
    assert fixed.mple[f"ATXBX.level.{decay:g}"] == pytest.approx(curved.mple["ATXBX.level"], abs=1e-6)
    with pytest.raises(NotImplementedError, match="two alternating parts"):
        ergmx.summary_stats(g, "aaaxs('level', fixed=FALSE)")


# -- Goodness of fit by level ------------------------------------------------------------------------


def test_gof_by_level_observed_distributions():
    g = ergmx.datasets.load("labs_sim")
    result = ergmx.gof(g, "edges + nodemix('level', levels2=TRUE)", [0.0, -3.0, -3.0, -2.0], nsim=5, seed=1,
                       by="level")
    for level in ("researcher", "laboratory"):
        inside = g.induced_subgraph(g.vs.select(level=level))
        assert list(result[f"degree.{level}"].observed[:len(inside.degree())]) == \
            list(np.bincount(inside.degree(), minlength=inside.vcount()))
        shared = [len(set(inside.neighbors(a)) & set(inside.neighbors(b))) for a, b in inside.get_edgelist()]
        np.testing.assert_array_equal(result[f"espartners.{level}"].observed[:max(shared) + 1], np.bincount(shared))
        others = [v.index for v in g.vs if v["level"] != level]
        counts = [len(set(g.neighbors(v)) & set(others)) for v in g.vs.select(level=level).indices]
        np.testing.assert_array_equal(result[f"affiliations.{level}"].observed[:max(counts) + 1], np.bincount(counts))
        assert result[f"degree.{level}"].simulated.shape[0] == 5
    assert "affiliations of researcher" in str(result)


def test_gof_by_level_in_directed_networks():
    g = random_directed_two_level(1, others=0)
    result = ergmx.gof(g, "edges + mutual", [-2.0, 1.0], nsim=4, seed=2, by="level")
    assert "idegree.lab" in result.tables and "odegree.researcher" in result.tables
    level = np.array(g.vs["level"])
    adj = np.array(g.get_adjacency().data)
    labs, researchers = np.flatnonzero(level == "lab"), np.flatnonzero(level == "researcher")
    np.testing.assert_array_equal(result["affiliations.lab"].observed[:adj[np.ix_(labs, researchers)].sum(1).max() + 1],
                                  np.bincount(adj[np.ix_(labs, researchers)].sum(1)))
    with pytest.raises(ValueError, match="unknown statistics"):
        ergmx.gof(g, "edges", [-2.0], nsim=2, by="level", stats=["degree"])


def test_estimated_multilevel_decay_is_recovered():
    """Networks simulated with ATXAX's decay 0.7 (and its coefficient 0.9)
    give them back, within their standard errors."""
    rng = np.random.default_rng(3)
    n_a, n_b = 150, 20
    g = ig.Graph(n=n_a + n_b)
    g.vs["level"] = ["A"] * n_a + ["B"] * n_b
    # Each A vertex in four B vertices, so that A-ties have 0 to 4 shared partners.
    g.add_edges([(a, n_a + b) for a in range(n_a) for b in rng.choice(n_b, 4, replace=False)])
    f = "S(~edges, ~level == 'A') + S(~edges, ~level == 'B') + atxax('level', decay=0.7)"
    given = "blocks('level', levels2=2)"
    for seed in (1, 2):
        sim = ergmx.simulate(g, f, [-5.0, -1.0, 0.9], constraints=given, seed=seed, burnin=500_000)[0]
        fit = ergmx.ergm(sim, f.replace("decay=0.7", "decay=0.3, fixed=FALSE"), constraints=given, seed=2,
                         eval_loglik=False)
        assert fit.converged
        assert abs(fit.coef["ATXAX.level.decay"] - 0.7) < 2.5 * fit.stderr["ATXAX.level.decay"]
        assert abs(fit.coef["ATXAX.level"] - 0.9) < 2.5 * fit.stderr["ATXAX.level"]
