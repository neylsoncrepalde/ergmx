"""btergm: pseudo-likelihood estimates against R's btergm on Knecht's pupils
(scripts/r_btergm_reference.R), the bootstrap against its exact
distribution (all resamples of the few time steps), and the percentile
intervals against boot's."""

import itertools
import json
import warnings

import igraph as ig
import networkx as nx
import numpy as np
import pytest
from conftest import DATA, load

import ergmx
from ergmx._btergm import _norm_inter
from ergmx._estimation import logistic_regression

REFERENCE = DATA / "r_btergm_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


def knecht():
    """The pupils' friendship networks at four times (pupils who left are not
    in them), with their sex, degrees, and the primary school covariate."""
    primary, labels = np.array(R["primary"]), R["primary_names"]
    graphs = []
    for s in R["networks"].values():
        g = ig.Graph(n=len(s["names"]), edges=[tuple(e) for e in s["edges"]], directed=True)
        g.vs["name"] = [str(v) for v in s["names"]]
        for a in ("sex", "idegsqrt", "odegsqrt"):
            g.vs[a] = s[a]
        rows = [labels.index(str(v)) for v in s["names"]]
        g["primary"] = g["primaries"] = primary[np.ix_(rows, rows)]
        graphs.append(g)
    return graphs


def _formula(r) -> str:
    """R's formula, with the covariates' names quoted and the transform as an expression."""
    return (r["formula"].replace("edgecov(primary)", "edgecov('primary')")
            .replace("timecov(primaries", "timecov('primaries'").replace("function(t) t^2", "'t^2'"))


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["models"]) if R else [])
def test_btergm_matches_r(name):
    r = R["models"][name]
    fit = ergmx.btergm(knecht(), _formula(r), R=200, offset=r["offset"], seed=1)
    np.testing.assert_allclose(fit.params, r["coef"], rtol=1e-6, atol=1e-6)
    assert (fit.time_steps, fit.nobs) == (r["time_steps"], r["nobs"])
    assert len(fit.names) == len(r["names"])


def _exact_bootstrap(fit):
    """The bootstrap's exact distribution: the estimates of every resample of
    the time steps (equally likely), without those that can't estimate every
    coefficient."""
    x, y, w, time = fit._data
    steps = fit.time_steps
    estimates = []
    for resample in itertools.product(range(steps), repeat=steps):
        weights = w * np.bincount(resample, minlength=steps)[time]
        used = weights > 0
        if np.linalg.matrix_rank(x[used] * np.sqrt(weights[used])[:, None]) < x.shape[1]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimates.append(logistic_regression(x[used], y[used], weights[used])[0])
    return np.array(estimates)


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["models"]) if R else [])
def test_bootstrap_matches_its_exact_distribution(name):
    """btergm's bootstrap means and ergmx's, against the exact bootstrap
    distribution's mean, within their Monte Carlo errors."""
    r = R["models"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ergmx.btergm(knecht(), _formula(r), R=2000, offset=r["offset"], seed=1)
        exact = _exact_bootstrap(fit)
        ours = fit.boot[np.all(np.isfinite(fit.boot), axis=1)]
    mean, sd = exact.mean(axis=0), exact.std(axis=0)
    assert np.all(np.abs(np.array(r["boot_mean"]) - mean) < 4 * sd / np.sqrt(r["complete"]) + 1e-9)
    assert np.all(np.abs(ours.mean(axis=0) - mean) < 4 * sd / np.sqrt(len(ours)) + 1e-9)
    np.testing.assert_allclose(r["boot_sd"], sd, rtol=0.15)
    assert len(ours) / len(fit.boot) == pytest.approx(len(exact) / fit.time_steps**fit.time_steps, abs=0.02)


@pytest.mark.skipif(R is None, reason="no R reference")
def test_percentile_intervals_interpolate_as_boot():
    p = R["percentile"]
    t = np.array(p["t"])
    for alpha, value in zip(p["alpha"], p["value"]):
        assert _norm_inter(t, alpha) == pytest.approx(value, rel=1e-12)


@pytest.mark.skipif(R is None, reason="no R reference")
def test_btergm_results():
    graphs = knecht()
    fit = ergmx.btergm(graphs, "edges + mutual + nodematch('sex') + memory + delrecip", R=300, seed=2)
    table = fit.confint()
    assert list(table.columns) == ["Estimate", "Boot mean", "2.5%", "97.5%"]
    assert np.all(table["2.5%"] < table["97.5%"])
    assert "Time steps: 3" in fit.summary() and "bootstrap replications: 300" in fit.summary()
    assert list(fit.coef) == ["edges", "mutual", "nodematch.sex", "memory.stability", "delrecip"]
    assert ergmx.btergm(graphs, "edges + mutual + memory", R=50, seed=2).boot.shape == (50, 3)
    # The same seed gives the same bootstrap; networkx graphs, matched by their nodes, the same fit.
    np.testing.assert_array_equal(fit.boot, ergmx.btergm(graphs, "edges + mutual + nodematch('sex') + memory + "
                                                                 "delrecip", R=300, seed=2).boot)
    converted = []
    for g in graphs:
        h = nx.DiGraph()
        h.add_nodes_from((v["name"], {"sex": v["sex"]}) for v in g.vs)
        h.add_edges_from((g.vs[a]["name"], g.vs[b]["name"]) for a, b in g.get_edgelist())
        converted.append(h)
    other = ergmx.btergm(converted, "edges + mutual + nodematch('sex') + memory + delrecip", R=10, seed=2)
    np.testing.assert_allclose(other.params, fit.params, rtol=1e-10)
    # Two memory terms (btergm keeps one).
    two = ergmx.btergm(graphs, "edges + mutual + memory(type='autoregression') + memory(type='stability', lag=2)",
                       R=10, seed=1)
    assert np.all(np.isfinite(two.params)) and list(two.coef)[2:] == ["memory.autoregression",
                                                                    "memory.stability.lag2"]
    # Resamples of a single time step repeated can't estimate a time trend.
    with pytest.warns(UserWarning, match="left out, as btergm does"):
        ergmx.btergm(graphs, "edges + mutual + timecov()", R=200, seed=1).confint()


def test_btergm_errors():
    g = load("samplk1")
    with pytest.raises(ValueError, match="at least 2"):
        ergmx.btergm([g], "edges")
    with pytest.raises(ValueError, match="leaves no network"):
        ergmx.btergm([g, g], "edges + memory(lag=2)")
    with pytest.raises(ValueError, match="memory\\(type=\\)"):
        ergmx.btergm([g, g], "edges + memory(type='forever')")
    with pytest.raises(ValueError, match="no offset"):
        ergmx.btergm([g, g], "edges + offset(mutual)")
    with pytest.raises(ValueError, match="vertex names"):
        ergmx.btergm([g, ig.Graph(n=5, directed=True)], "edges + mutual")
    with pytest.raises(ValueError, match="is a term of btergm"):
        ergmx.ergm(g, "edges + memory")
    with pytest.raises(ValueError, match="needs the networks before"):
        ergmx.ergm(g, ergmx.edges() + ergmx.memory())
