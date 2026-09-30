import numpy as np
import pytest
from conftest import REFERENCE, load

import ergmx

DEPENDENT = [name for name, m in REFERENCE.items() if not m["dyad_independent"]]


def test_mple_matches_r(reference):
    _, model, g = reference
    fit = ergmx.ergm(g, model["formula"], estimate="MPLE")
    assert fit.names == list(model["mple"])
    np.testing.assert_allclose(list(fit.mple.values()), list(model["mple"].values()),
                               rtol=1e-6, atol=1e-8)


def test_dyad_independent_models_get_the_exact_mle():
    model = REFERENCE["flo_dyadind"]
    fit = ergmx.ergm(load(model["network"]), model["formula"])
    assert fit.method == "MLE"
    np.testing.assert_allclose(list(fit.coef.values()), list(model["mle"].values()), rtol=1e-6)
    # R's glm stops at a deviance tolerance of 1e-8, so its SEs are accurate to ~1e-6.
    np.testing.assert_allclose(list(fit.stderr.values()), list(model["se"].values()), rtol=1e-5)
    assert fit.loglik == pytest.approx(model["loglik"], rel=1e-9)
    assert fit.aic == pytest.approx(-2 * model["loglik"] + 2 * 3)


@pytest.mark.parametrize("name", DEPENDENT)
def test_monte_carlo_mle_matches_r(name):
    model = REFERENCE[name]
    fit = ergmx.ergm(load(model["network"]), model["formula"], seed=2026)
    assert fit.method == "MCMLE" and fit.converged
    for term, r_estimate in model["mle"].items():
        r_se = model["se"][term]
        assert abs(fit.coef[term] - r_estimate) < 0.25 * r_se, term
        assert fit.stderr[term] == pytest.approx(r_se, rel=0.2), term


def test_fits_are_reproducible():
    g = load("samplk3")
    first = ergmx.ergm(g, "edges + mutual", seed=7, n_chains=2)
    second = ergmx.ergm(g, "edges + mutual", seed=7, n_chains=2)
    assert first.coef == second.coef


def test_summary_table():
    fit = ergmx.ergm(load("samplk3"), "edges + mutual", seed=1)
    text = str(fit.summary())
    assert text.startswith("Monte Carlo Maximum Likelihood Results:")
    assert "edges" in text and "mutual" in text and "Converged" in text
    assert "mutual" in repr(fit)


def test_mple_only():
    fit = ergmx.ergm(load("samplk3"), "edges + mutual", estimate="MPLE")
    assert fit.method == "MPLE"
    assert "unreliable" in str(fit.summary())


def test_bad_arguments():
    g = load("samplk3")
    with pytest.raises(ValueError, match="estimate must be"):
        ergmx.ergm(g, "edges", estimate="Bayes")
    with pytest.raises(ValueError, match="init must have 2 values"):
        ergmx.ergm(g, "edges + mutual", init=[0.0])
    with pytest.raises(TypeError):
        ergmx.ergm(g, "edges + mutual", not_a_setting=1)
