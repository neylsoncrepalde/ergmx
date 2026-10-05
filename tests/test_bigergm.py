"""bigergm(): the MM algorithm's blocks, the between- and within-block fits
and the simulations, against R's bigergm and ergm."""

import json

import igraph as ig
import numpy as np
import pytest
from conftest import DATA

import ergmx
from ergmx._bigergm import _fix_small_blocks, _adjacency
from ergmx.datasets import load

REFERENCE = DATA / "r_bigergm_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None
pytestmark = pytest.mark.skipif(R is None, reason="no R reference")


def toy():
    """bigergm's toyNet: 200 vertices in 4 blocks, with covariates x and y."""
    t = R["toyNet"]
    g = ig.Graph(n=t["n"], edges=(np.array(t["edges"]) - 1).tolist())
    g.vs["block"], g.vs["x"], g.vs["y"] = t["block"], t["x"], t["y"]
    return g


TOY = "edges + nodematch('x') + nodematch('y') + triangle"


@pytest.mark.parametrize("name", list(R["mm"]) if R else [])
def test_mm_algorithm_matches_bigergm(name):
    """From the same starting blocks, the same iterations, lower bounds,
    membership probabilities and blocks."""
    r = R["mm"][name]
    network, start, k = (toy(), R["toy_start"], 4) if r["network"] == "toyNet" else \
        (load("faux.dixon.high"), R["dixon_start"], 6)
    fit = ergmx.bigergm(network, r["formula"], k, initialization=start, clustering_with_features=r["features"],
                        estimate_parameters=False)
    assert len(fit.lower_bound) == r["iterations"]
    np.testing.assert_allclose(fit.lower_bound, r["lower_bound"], rtol=1e-10)
    np.testing.assert_allclose(fit.membership, np.array(r["alpha"]), atol=1e-9)
    assert fit.blocks.tolist() == r["block"]


@pytest.mark.parametrize("name", list(R["fits"]) if R else [])
def test_fits_from_the_blocks_match_bigergm(name):
    """The between-block logistic regression and the within-block MPLE
    exactly, the within-block Monte Carlo MLE within 0.3 standard errors.
    ergmx leaves out the edges terms the blocks' intercepts make redundant
    (bigergm's are NA), and doesn't quote the names of interactions."""
    r = R["fits"][name]
    formula = TOY if name != "mle" else "edges + nodematch('x') + nodematch('y') + gwesp(0.5, fixed = TRUE)"
    options = {"mple": {}, "intercepts": {"add_intercepts": True},
               "intercepts_plain": {"add_intercepts": True, "clustering_with_features": False},
               "mle": {"method_within": "MLE", "seed": 1}}[name]
    fit = ergmx.bigergm(toy(), formula, blocks=R["toyNet"]["block"], **options)
    for part, ours in (("between", fit.between), ("within", fit.within)):
        names = [n.strip("`") for n in r[f"{part}_names"]]
        estimated = [n for n, v in zip(names, r[part]) if v != "NA"]
        assert sorted(ours.names) == sorted(estimated), part
        expected = dict(zip(names, r[part]))
        se = dict(zip(names if len(r[f"{part}_se"]) == len(names) else estimated, r[f"{part}_se"]))
        got = dict(zip(ours.names, ours.params))
        mine_se = dict(zip(ours.names, np.sqrt(np.diag(ours.cov))))
        for n in estimated:
            if name == "mle" and part == "within":
                assert abs(got[n] - expected[n]) < 0.3 * se[n], (n, got[n], expected[n])
            else:
                assert got[n] == pytest.approx(expected[n], rel=1e-7, abs=1e-9), n
                assert mine_se[n] == pytest.approx(se[n], rel=1e-5), n


def test_simulated_networks_follow_the_fitted_model():
    """The dyads within blocks as ergm's simulation of the within-block model,
    those between with the between-block model's expected ties. (bigergm's
    simulate() reproduces neither.)"""
    fit = ergmx.bigergm(toy(), TOY, blocks=R["toyNet"]["block"])
    sims = fit.simulate(150, seed=1, burnin=100000, interval=10000, output="edges")
    blocks = np.array(R["toyNet"]["block"])
    within = [s[blocks[s[:, 0].astype(int)] == blocks[s[:, 1].astype(int)]] for s in sims]
    between = [len(s) - len(w) for s, w in zip(sims, within)]
    model = fit.within._model
    stats = np.array([model.core.summary(np.ascontiguousarray(w, dtype=np.uint32)) for w in within])
    r = R["within_simulated"]
    tolerance = 4 * np.array(r["sd"]) * np.sqrt(1 / 150 + 1 / 1000)
    assert np.all(np.abs(stats.mean(axis=0) - r["mean"]) < tolerance), (stats.mean(axis=0), r["mean"])
    pairs, p = fit._between_probabilities()
    assert p.sum() == pytest.approx(R["expected_between"], rel=1e-9)
    assert abs(np.mean(between) - p.sum()) < 4 * np.sqrt(np.sum(p * (1 - p)) / 150)
    graphs = fit.simulate(2, seed=1)
    assert isinstance(graphs[0], ig.Graph) and graphs[0].vs["block"] == fit.blocks.tolist()


def test_blocks_found_from_infomap():
    fit = ergmx.bigergm(toy(), TOY, 4, seed=1)
    assert fit.n_blocks == 4
    pairs = set(zip(fit.blocks.tolist(), R["toyNet"]["block"]))
    assert len(pairs) == 4  # toyNet's blocks, relabelled
    text = fit.summary()
    assert "Blocks: 4, of sizes 50, 50, 50, 50." in text and "Within blocks:" in text
    gof = fit.gof(10, seed=1)
    assert [t.name for t in gof] == ["degree", "espartners", "distance"]
    assert gof["degree"].simulated.shape[0] == 10


def test_blocks_too_small_take_vertices_of_the_largest():
    g = _adjacency(ergmx._network.as_network(toy()))
    blocks = np.array(R["toyNet"]["block"])
    blocks[blocks == 4] = 3
    blocks[0] = 4  # a block of one vertex
    fixed = _fix_small_blocks(blocks, g, 4, np.random.default_rng(1))
    assert np.all(np.bincount(fixed, minlength=5)[1:] >= 2)
    assert np.array_equal(fixed[np.isin(blocks, [1, 2]) & (np.arange(200) != 0)],
                          blocks[np.isin(blocks, [1, 2]) & (np.arange(200) != 0)])


def test_errors():
    with pytest.raises(ValueError, match="n_blocks"):
        ergmx.bigergm(toy(), TOY)
    with pytest.raises(ValueError, match="one-mode"):
        ergmx.bigergm(load("davis"), "edges", 2, bipartite=True)
    with pytest.raises(ValueError, match="'MPLE' or 'MLE'"):
        ergmx.bigergm(toy(), TOY, blocks=R["toyNet"]["block"], method_within="CD")
    with pytest.raises(ValueError, match="no dyad-independent terms"):
        ergmx.bigergm(toy(), "triangle", blocks=R["toyNet"]["block"])
