import igraph as ig
import networkx as nx
import numpy as np
from conftest import REFERENCE, load

import ergmx


def test_simulated_graphs_keep_the_vertex_attributes():
    g = load("faux.mesa.high")
    model = REFERENCE["mesa_gwesp"]
    networks = ergmx.simulate(g, model["formula"], model["mle"], nsim=3, seed=1)
    assert len(networks) == 3
    for h in networks:
        assert isinstance(h, ig.Graph) and h.vcount() == g.vcount()
        assert h.vs["Grade"] == g.vs["Grade"]


def test_networkx_in_networkx_out():
    G = load("samplk3").to_networkx()
    networks = ergmx.simulate(G, "edges + mutual", [-2.15, 2.3], nsim=2, seed=1)
    assert all(isinstance(h, nx.DiGraph) and set(h) == set(G) for h in networks)


def test_simulating_at_the_mle_reproduces_the_observed_statistics():
    g = load("samplk3")
    model = REFERENCE["samplk_mutual"]
    stats = ergmx.simulate(g, "edges + mutual", model["mle"], nsim=2000, seed=1, output="stats")
    observed = np.array(list(model["stats"].values()))
    se = stats.std(axis=0) / np.sqrt(len(stats)) * 3  # generous for autocorrelation
    assert np.all(np.abs(stats.mean(axis=0) - observed) < 3 * se)


def test_simulate_from_a_fit():
    fit = ergmx.ergm(load("samplk3"), "edges + mutual", seed=1)
    stats = fit.simulate(5, seed=2, output="stats")
    assert stats.shape == (5, 2)
