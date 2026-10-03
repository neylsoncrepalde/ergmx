"""Bayesian ERGMs (bergm(), as R's Bergm): the exchange algorithm against the
exact posterior of dyad-independent models (also with missing dyads), and
against R's Bergm (scripts/r_bergm_reference.R)."""

import json

import numpy as np
import pytest
from conftest import DATA, load

import ergmx

REFERENCE = DATA / "r_bergm_reference.json"
R = json.loads(REFERENCE.read_text())["models"] if REFERENCE.exists() else {}


def _exact_posterior(model, prior_sd=10.0, half_width=7.0, points=241):
    """Posterior means and standard deviations of a two-parameter
    dyad-independent model, by integrating its exact likelihood (a logistic
    regression on the free observed dyads) over a grid."""
    from ergmx._estimation import mple

    x, y, w = model.mple_table()
    est = mple(model)
    centre, se = est.theta, np.sqrt(np.diag(est.cov))
    axes = [np.linspace(c - half_width * s, c + half_width * s, points) for c, s in zip(centre, se)]
    a, b = np.meshgrid(*axes, indexing="ij")
    thetas = np.stack([a.ravel(), b.ravel()], axis=1)
    eta = thetas @ x.T
    loglik = (w * (y * eta - np.logaddexp(0, eta))).sum(axis=1)
    logpost = loglik - 0.5 * (thetas**2).sum(axis=1) / prior_sd**2
    p = np.exp(logpost - logpost.max())
    p /= p.sum()
    mean = p @ thetas
    sd = np.sqrt(p @ (thetas - mean) ** 2)
    return mean, sd


@pytest.mark.parametrize(("network", "formula", "aux_iters"), [
    ("samplk3", "edges + nodematch('group')", None),
    ("faux.mesa.high.missing", "edges + nodematch('Grade')", 5000),
])
def test_exchange_algorithm_gives_the_exact_posterior(network, formula, aux_iters):
    """For a dyad-independent model the likelihood, and so the posterior, is
    exact: the exchange algorithm's draws must have its moments. With missing
    dyads, the chains impute them, and the posterior is that of the observed
    dyads."""
    from ergmx._model import bind

    g = load(network)
    model = bind(g, formula, fitting=True)
    mean, sd = _exact_posterior(model)
    fit = ergmx.bergm(g, formula, main_iters=1500, aux_iters=aux_iters, seed=2)
    se = np.array(list(fit.sd.values())) * np.sqrt(fit.tau / fit.draws.shape[0])
    assert np.all(np.abs(fit.mean - mean) < 4 * se), (fit.mean, mean, se)
    np.testing.assert_allclose(list(fit.sd.values()), sd, rtol=0.15)
    assert 0.01 < fit.acceptance_rate < 0.9
    assert all(r < 1.1 for r in fit.rhat.values())


@pytest.mark.skipif(not R, reason="no R reference")
@pytest.mark.parametrize("name", list(R))
def test_bergm_matches_r(name):
    r = R[name]
    fit = ergmx.bergm(load(r["network"]), r["formula"], main_iters=r["main_iters"], aux_iters=r["aux_iters"], seed=1)
    assert fit.names == r["names"]
    sd = np.array(list(fit.sd.values()))
    se = sd * np.sqrt(fit.tau / fit.draws.shape[0])
    both = np.sqrt(se**2 + np.array(r["ts_se"]) ** 2)
    assert np.all(np.abs(fit.mean - r["mean"]) < 4 * both), (fit.mean, r["mean"], both)
    np.testing.assert_allclose(sd, r["sd"], rtol=0.2)


def test_bergm_results():
    g = load("samplk3")
    fit = ergmx.bergm(g, "edges + mutual", main_iters=300, burn_in=50, seed=1)
    assert fit.chains.shape == (4, 300, 2) and fit.draws.shape == (1200, 2)
    text = str(fit.summary())
    assert "Time-series SE" in text and "Acceptance rate" in text and "mutual" in text
    assert list(fit.summary().to_frame().columns)[:4] == ["Mean", "SD", "Naive SE", "Time-series SE"]
    assert fit.to_frame().shape == (1200, 3)
    assert fit.quantiles().shape == (2, 5)
    gof = fit.gof(20, aux_iters=2000, seed=1)
    assert {"idegree", "odegree", "espartners", "distance", "model"} <= {t.name for t in gof}
    sims = fit.simulate(3, seed=1, aux_iters=2000)
    assert len(sims) == 3 and all(h.is_directed() for h in sims)
    assert fit.simulate(5, seed=1, output="stats", aux_iters=500).shape == (5, 2)
    assert len(fit.plot().axes) == 6
    # Reproducible with a seed.
    again = ergmx.bergm(g, "edges + mutual", main_iters=300, burn_in=50, seed=1)
    np.testing.assert_array_equal(again.draws, fit.draws)


def test_offsets_priors_and_decays():
    g = load("samplk3")
    fit = ergmx.bergm(g, "edges + offset(mutual)", offset_coef=[2.0], main_iters=200, burn_in=20, seed=1,
                      nchains=4)
    assert fit.names == ["edges"] and fit.params[1] == 2.0
    # A tight prior pulls the posterior to its mean.
    tight = ergmx.bergm(g, "edges + mutual", prior_mean=[-3, 2], prior_sigma=0.01, main_iters=200, burn_in=50,
                        seed=1)
    np.testing.assert_allclose(tight.mean, [-3, 2], atol=0.3)
    curved = ergmx.bergm(g, "edges + mutual + gwesp(0.5)", main_iters=100, burn_in=20, seed=1, aux_iters=500)
    assert curved.names[-1].endswith(".decay") and curved.draws[:, -1].min() >= 0
    with pytest.raises(ValueError, match="at least 4 chains"):
        ergmx.bergm(g, "edges + mutual", nchains=3)
    with pytest.raises(ValueError, match="positive definite"):
        ergmx.bergm(g, "edges + mutual", prior_sigma=[[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="prior_mean"):
        ergmx.bergm(g, "edges + mutual", prior_mean=[0])
