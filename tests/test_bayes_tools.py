"""Bergm's model-choice tools: the adjusted pseudo-likelihood, the model
evidence and the calibrated pseudo-posterior, against the exact answers of
a dyad-independent model (whose pseudo-likelihood is its likelihood) and
R's Bergm (scripts/r_bergm_tools_reference.R)."""

import json
import warnings

import numpy as np
import pytest
from conftest import DATA, load
from scipy.stats import multivariate_normal

import ergmx
from ergmx import ErgmDifferenceWarning
from ergmx._model import bind

REFERENCE = DATA / "r_bergm_tools_reference.json"
R = json.loads(REFERENCE.read_text())["models"] if REFERENCE.exists() else None
DYADIND = "edges + nodecov('wealth')"
MONKS = "edges + mutual + nodematch('group')"


@pytest.fixture(scope="module")
def exact():
    """The exact posterior of the dyad-independent model with Bergm's prior
    N(0, 100 I), by importance sampling: its log evidence, mean and sd."""
    model = bind(load("flomarriage"), DYADIND)
    x, y, w = model.mple_table()
    fit = ergmx.ergm(load("flomarriage"), DYADIND)
    rng = np.random.default_rng(1)
    cov = 1.5 * np.atleast_2d(np.asarray(fit._estimate.cov))
    draws = rng.multivariate_normal(fit.params, cov, 200_000)
    lin = draws @ x.T
    loglik = (w * (y * lin - np.logaddexp(0, lin))).sum(axis=1)
    logw = loglik + multivariate_normal(np.zeros(2), 100 * np.eye(2)).logpdf(draws) \
        - multivariate_normal(fit.params, cov).logpdf(draws)
    weights = np.exp(logw - logw.max())
    mean = weights @ draws / weights.sum()
    sd = np.sqrt(weights @ (draws - mean) ** 2 / weights.sum())
    return {"log_evidence": float(np.log(weights.mean()) + logw.max()), "mean": mean, "sd": sd}


def test_adjusted_pseudo_likelihood():
    """Of a dyad-independent model, the pseudo-likelihood itself; of others,
    one with the likelihood's maximum, curvature and value there."""
    apl = ergmx.ergm_apl(load("flomarriage"), DYADIND, seed=1)
    np.testing.assert_allclose(apl.W, np.eye(2), atol=1e-8)
    np.testing.assert_allclose(apl.theta_mle, apl.theta_pl, atol=1e-6)
    assert abs(apl.log_c) < 1e-6
    monks = load("samplk3")
    fit = ergmx.ergm(monks, MONKS, seed=1)
    apl = ergmx.ergm_apl(monks, MONKS, fit=fit)
    information = np.cov(fit._estimate.sample.reshape(-1, 3), rowvar=False)
    np.testing.assert_allclose(-apl.hessian(apl.theta_mle), information, rtol=1e-8, atol=1e-8)
    assert apl.loglik(apl.theta_mle) == pytest.approx(fit.loglik, abs=1e-8)
    assert np.all(apl.score(apl.theta_mle[None, :]) < 1e-6)
    if R is not None:  # R's log C is noisy: within its spread
        r = np.array(R["monks"]["logC"])
        assert abs(apl.log_c - r.mean()) < 3 * r.std(ddof=1)


def test_evidence_matches_the_exact_marginal_likelihood(exact):
    g = load("flomarriage")
    apl = ergmx.ergm_apl(g, DYADIND, seed=1)
    with pytest.warns(ErgmDifferenceWarning, match="leaves the prior out"):
        cj = ergmx.evidence(g, DYADIND, method="CJ", seed=1, apl=apl)
    assert cj.log_evidence == pytest.approx(exact["log_evidence"], abs=0.03)
    np.testing.assert_allclose(cj.draws.mean(axis=0), exact["mean"], atol=0.1 * exact["sd"].max())
    # Power posteriors, with fewer draws: within their discretization error (0.2 with 50 temperatures).
    pp = ergmx.evidence(g, DYADIND, method="PP", seed=1, apl=apl, main_iters=3000, burn_in=500)
    assert pp.log_evidence == pytest.approx(exact["log_evidence"], abs=0.5)
    assert "log evidence" in repr(pp)


@pytest.mark.skipif(R is None, reason="no R reference")
def test_evidence_of_a_dyad_dependent_model():
    """The monks' evidence by Chib and Jeliazkov's method, against R's power
    posteriors (Bergm's Chib and Jeliazkov estimates, from 5 networks per
    curvature, are noisier and biased up)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        values = [ergmx.evidence(load("samplk3"), MONKS, method="CJ", seed=s).log_evidence for s in (1, 2)]
    assert abs(values[0] - values[1]) < 0.3
    pp = np.array(R["monks"]["evidence_pp"])
    assert abs(np.mean(values) - pp.mean()) < 3 * max(pp.std(ddof=1), 0.3)


def test_bergmC_calibrates_the_pseudo_posterior(exact):  # noqa: N802
    g = load("flomarriage")
    with pytest.warns(ErgmDifferenceWarning, match="wrong sign"):
        fit = ergmx.bergmC(g, DYADIND, main_iters=20000, burn_in=2000, seed=1)
    assert isinstance(fit, ergmx.BergmFit)
    np.testing.assert_allclose(fit.mean, exact["mean"], atol=0.1 * exact["sd"].max())
    np.testing.assert_allclose(list(fit.sd.values()), exact["sd"], rtol=0.1)
    if R is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErgmDifferenceWarning)
            monks = ergmx.bergmC(load("samplk3"), MONKS, seed=1)
        r = R["monks"]
        assert np.all(np.abs(monks.mean - r["bergmC_mean"]) < 0.4 * np.array(r["bergmC_sd"]))
        np.testing.assert_allclose(list(monks.sd.values()), r["bergmC_sd"], rtol=0.3)
        assert "calibrated pseudo-posterior" in str(monks.summary())


def test_tools_errors():
    g = load("flomarriage")
    with pytest.raises(ValueError, match="curved"):
        ergmx.ergm_apl(g, "edges + gwesp(0.5)")
    with pytest.raises(ValueError, match="at least 2"):
        ergmx.evidence(g, "edges")
    with pytest.raises(ValueError, match="'CJ' or 'PP'"):
        ergmx.evidence(g, DYADIND, method="AIC")
