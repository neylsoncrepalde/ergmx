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
    }


def formula(attr, decay, levels=None):
    lv = "" if levels is None else f", levels={list(levels)!r}"
    plain = ["star2ax", "star2bx", "txax", "txbx", "l3xax", "l3xbx", "l3axb", "c4axb"]
    alternating = ["axs1a", "axs1b", "aas1x", "abs1x", "aaaxs", "abaxs", "atxax", "atxbx"]
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
    with pytest.raises(ValueError, match="undirected"):
        ergmx.summary_stats(ergmx.datasets.load("samplk3"), "S(~edges, cloisterville ~ !cloisterville)")


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
