import igraph as ig
import networkx as nx
import numpy as np
import pytest
from conftest import REFERENCE, load

import ergmx


def test_statistics_match_r(reference):
    _, model, g = reference
    stats = ergmx.summary_stats(g, model["formula"])
    assert list(stats) == list(model["stats"])  # same names, same order
    np.testing.assert_allclose(list(stats.values()), list(model["stats"].values()), rtol=1e-12)


@pytest.mark.parametrize("name", ["samplk_mutual", "mesa_gwesp", "flo_dyadind"])
def test_statistics_tracked_by_the_sampler_are_exact(name):
    """The sampler updates statistics with change statistics; recomputing them
    from scratch on the sampled networks must give the same values."""
    model = REFERENCE[name]
    g = load(model["network"])
    coef = list(model["mle"].values())
    tracked = ergmx.simulate(g, model["formula"], coef, 20, seed=3, interval=4096, output="stats")
    networks = ergmx.simulate(g, model["formula"], coef, 20, seed=3, interval=4096)
    recomputed = [list(ergmx.summary_stats(h, model["formula"]).values()) for h in networks]
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
        ("samplk3", "edges + triangle", ValueError, "undirected networks"),
        ("flomarriage", "edges + mutual", ValueError, "directed networks"),
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
