"""The MCMC samples the right distribution: checks against exact enumeration of
every network on a few vertices, for each proposal mixture."""

import itertools
from collections import Counter

import igraph as ig
import numpy as np
import pytest
from conftest import load

from ergmx._model import bind

WEIGHTS = [0.0, 0.5, 0.9]  # plain TNT, and two shares of triadic proposals
EMPTY = np.zeros((0, 2), dtype=np.uint32)


def all_networks(n):
    pairs = list(itertools.combinations(range(n), 2))
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        yield np.array([p for p, b in zip(pairs, bits) if b], dtype=np.uint32).reshape(-1, 2)


def exact_distribution(model, n, theta):
    networks = list(all_networks(n))
    stats = np.array([model.core.summary(e) for e in networks])
    logp = stats @ theta
    p = np.exp(logp - logp.max())
    return networks, stats, p / p.sum()


@pytest.mark.parametrize("weight", WEIGHTS)
def test_each_network_is_visited_with_its_probability(weight):
    model = bind(ig.Graph(n=3), "edges + triangle")
    theta = [-0.8, 1.5]
    networks, _, p = exact_distribution(model, 3, theta)
    samples = 200_000
    _, _, visited = model.core.simulate([EMPTY], theta, 100, 10, samples, 5,
                                        keep_networks=True, triadic_weight=weight)
    counts = Counter(tuple(sorted(map(tuple, e.tolist()))) for e in visited[0])
    freq = np.array([counts[tuple(sorted(map(tuple, e.tolist())))] for e in networks]) / samples
    assert np.all(np.abs(freq - p) < 5 * np.sqrt(p * (1 - p) / samples))


@pytest.mark.parametrize("weight", WEIGHTS)
def test_mean_statistics_match_exact_expectations(weight):
    g = ig.Graph(n=6)
    g.vs["grp"] = [0, 0, 0, 1, 1, 1]
    model = bind(g, "edges + triangle + gwesp(0.7, fixed=TRUE) + nodematch('grp')")
    theta = np.array([-0.8, 0.4, 0.3, 0.5])
    _, stats, p = exact_distribution(model, 6, theta)
    expected = p @ stats
    runs = np.array([
        model.core.simulate([EMPTY] * 4, theta.tolist(), 1000, 20, 10_000, seed,
                            triadic_weight=weight)[0].reshape(-1, 4).mean(axis=0)
        for seed in range(12)
    ])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(runs.mean(axis=0) - expected) < 5 * se)


def test_triadic_proposals_are_the_default_for_triangle_models():
    mesa, samplk = load("faux.mesa.high"), load("samplk3")
    assert bind(mesa, "edges + gwesp(0.5, fixed=TRUE)").triadic_weight(None) == 0.5
    assert bind(mesa, "edges + nodematch('Grade')").triadic_weight(None) == 0.0
    assert bind(samplk, "edges + mutual").triadic_weight(None) == 0.0
    assert bind(mesa, "edges + triangle").triadic_weight(0.2) == 0.2


def test_triadic_weight_is_validated():
    directed = bind(load("samplk3"), "edges + mutual")
    with pytest.raises(ValueError, match="triadic_weight"):
        directed.core.simulate([directed.network.edges], [0.0, 0.0], 10, 1, 1, 1, triadic_weight=0.5)
    undirected = bind(load("flomarriage"), "edges")
    with pytest.raises(ValueError, match="triadic_weight"):
        undirected.core.simulate([undirected.network.edges], [0.0], 10, 1, 1, 1, triadic_weight=1.0)
