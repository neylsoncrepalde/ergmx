import json

import igraph as ig
import networkx as nx
import numpy as np
import pytest
from conftest import DATA, REFERENCE, load, options, without_overflow

import ergmx


def test_statistics_match_r(reference):
    _, model, g = reference
    stats = without_overflow(ergmx.summary_stats(g, model["formula"], **_bipartite(model)))
    assert list(stats) == list(model["stats"])  # same names, same order
    np.testing.assert_allclose(list(stats.values()), list(model["stats"].values()), rtol=1e-12)


def _bipartite(model):
    return {k: v for k, v in options(model).items() if k == "bipartite"}


@pytest.mark.parametrize("name", ["samplk_mutual", "mesa_gwesp", "flo_dyadind", "mesa_terms",
                                  "dixon_terms", "mesa_terms2", "dixon_terms2", "mesa_terms3",
                                  "dixon_terms3", "davis_terms", "bipartite_terms", "mesa_curved"])
def test_statistics_tracked_by_the_sampler_are_exact(name):
    """The sampler updates statistics with change statistics; recomputing them
    from scratch on the sampled networks must give the same values."""
    model = REFERENCE[name]
    g = load(model["network"])
    # Coefficients that keep the density near the observed one.
    bip = _bipartite(model)
    dyads = g.vcount() * (g.vcount() - 1) / (1 if g.is_directed() else 2)
    if bip:
        n1 = sum(not t for t in g.vs["type"])
        dyads = n1 * (g.vcount() - n1)
    density = g.ecount() / dyads
    formula = model["formula"] if "edges" in model["stats"] else "edges + " + model["formula"]
    params = ergmx.ergm(g, formula, estimate="MPLE", **bip).names if "gwesp(0.5)" in formula else \
        list(ergmx.summary_stats(g, formula, **bip))
    coef = [np.log(density / (1 - density)) if term == "edges" else
            (0.5 if term.endswith(".decay") else 0.0) for term in params]
    tracked = ergmx.simulate(g, formula, coef, 20, seed=3, interval=4096, output="stats", **bip)
    networks = ergmx.simulate(g, formula, coef, 20, seed=3, interval=4096, **bip)
    recomputed = [list(ergmx.summary_stats(h, formula, **bip).values()) for h in networks]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9, atol=1e-9)
    assert len({h.ecount() for h in networks}) > 1  # the chain moved


def test_networkx_gives_the_same_statistics():
    g = load("faux.mesa.high")
    G = g.to_networkx()
    formula = REFERENCE["mesa_gwesp"]["formula"]
    assert ergmx.summary_stats(G, formula) == pytest.approx(ergmx.summary_stats(g, formula))


def test_directed_networkx():
    g = load("samplk3")
    assert ergmx.summary_stats(g.to_networkx(), "edges + mutual") == {"edges": 56, "mutual": 15}
    assert isinstance(g.to_networkx(), nx.DiGraph)


@pytest.mark.parametrize(
    ("network", "formula", "error", "message"),
    [
        ("samplk3", "edges + degree(1)", ValueError, "undirected networks"),
        ("samplk3", "kstar(2)", ValueError, "undirected networks"),
        ("flomarriage", "edges + mutual", ValueError, "directed networks"),
        ("flomarriage", "ttriple + nodeicov('wealth')", ValueError, "directed networks"),
        ("flomarriage", "edgecov('trade')", KeyError, "no graph attribute 'trade'"),
        ("flomarriage", "nodematch('Grade')", KeyError, "no vertex attribute 'Grade'"),
    ],
)
def test_term_errors(network, formula, error, message):
    with pytest.raises(error, match=message):
        ergmx.summary_stats(load(network), formula)


def test_networks_must_be_binary_without_loops():
    with pytest.raises(ValueError, match="self-loops"):
        ergmx.summary_stats(ig.Graph([(0, 1), (1, 1)]), "edges")
    with pytest.raises(ValueError, match="multiple edges"):
        ergmx.summary_stats(ig.Graph([(0, 1), (0, 1)]), "edges")
    with pytest.raises(TypeError, match="igraph or networkx"):
        ergmx.summary_stats(np.eye(3), "edges")


def test_reciprocated_two_paths_follow_the_definition():
    """ergm's edgewise RTP statistics depend on the order of the vertices (they
    change when the same network is relabeled) and don't match the definition;
    ergmx's match the definition, computed in R."""
    rtp = json.loads((DATA / "r_reference.json").read_text())["rtp_direct"]
    assert rtp["summary_original"] != rtp["summary_relabeled"]  # ergm's bug, as recorded
    stats = ergmx.summary_stats(load("faux.dixon.high"),
                                "esp(0:3, type='RTP') + gwesp(0.5, fixed=TRUE, type='RTP')")
    assert list(stats.values())[:4] == rtp["esp"]
    assert stats["gwesp.RTP.fixed.0.5"] == pytest.approx(rtp["gwesp"], rel=1e-12)


def test_curved_terms_weight_their_histograms_like_the_fixed_terms():
    """eta . counts of a curved term is theta times the fixed-decay term."""
    from ergmx._model import bind

    g = load("faux.mesa.high")
    for curved, fixed in [("gwesp(0.5, cutoff=300)", "gwesp(0.7, fixed=TRUE)"),
                          ("gwdegree(0.5, cutoff=300)", "gwdegree(0.7, fixed=TRUE)"),
                          ("gwdsp(0.5, cutoff=300)", "gwdsp(0.7, fixed=TRUE)")]:
        model = bind(g, curved)
        counts = model.observed()
        assert not any("#>" in n for n in model.stat_names)  # no overflow with this cutoff
        theta = np.array([1.3, 0.7])
        expected = 1.3 * ergmx.summary_stats(g, fixed)[list(ergmx.summary_stats(g, fixed))[0]]
        assert model.eta(theta) @ counts == pytest.approx(expected, rel=1e-10)
        # The Jacobian matches finite differences.
        h = 1e-6
        numeric = np.column_stack([(model.eta(theta + h * e) - model.eta(theta - h * e)) / (2 * h)
                                   for e in np.eye(2)])
        np.testing.assert_allclose(model.jacobian(theta), numeric, rtol=1e-5, atol=1e-8)
