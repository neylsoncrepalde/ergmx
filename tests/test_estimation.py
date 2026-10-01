import numpy as np
import pytest
from conftest import REFERENCE, estimated, load, models_with, options

import ergmx

FITTED = models_with("mle")
#: Curved models whose likelihood is nearly flat in the decay (R's standard
#: error of the decay is twice its estimate): where in that flat region an
#: estimate lands changes the standard errors of the correlated parameters, so
#: only the estimates are compared.
WEAKLY_IDENTIFIED = {"bipartite_curved"}
DEPENDENT = [name for name in FITTED if not REFERENCE[name]["dyad_independent"]]
INDEPENDENT = [name for name in FITTED if REFERENCE[name]["dyad_independent"]]


@pytest.mark.parametrize("name", models_with("mple"))
def test_mple_matches_r(name):
    model = REFERENCE[name]
    g = load(model["network"])
    fit = ergmx.ergm(g, model["formula"], estimate="MPLE", **options(model))
    assert fit.names == list(model["mple"])
    r_mple = np.array(list(model["mple"].values()))
    if not fit._model.curved:
        np.testing.assert_allclose(list(fit.mple.values()), r_mple, rtol=1e-6, atol=1e-8)
        return
    # Curved: R's optimizer stops near a flat optimum; ergmx's is at least as high.
    from ergmx._estimation import _pseudo_loglik, fixed_part

    np.testing.assert_allclose(list(fit.mple.values()), r_mple, rtol=1e-3, atol=1e-6)
    model_ = fit._model
    x, y = model_.mple_data()

    def pseudo(theta):
        lin = fixed_part(x, np.arange(model_.n_stats), model_.eta(theta))
        return _pseudo_loglik(lin, y, np.ones(len(y)))

    assert pseudo(fit._mple.theta) >= pseudo(r_mple) - 1e-8


@pytest.mark.parametrize("name", INDEPENDENT)
def test_dyad_independent_models_get_the_exact_mle(name):
    model = REFERENCE[name]
    fit = ergmx.ergm(load(model["network"]), model["formula"], **options(model))
    assert fit.method == "MLE"
    np.testing.assert_allclose(list(fit.coef.values()), list(model["mle"].values()), rtol=1e-6)
    # R's glm stops at a relative deviance change of 1e-8 and computes its
    # covariance from the weights of its previous iteration: its SEs are
    # accurate to ~1e-4, less for rare categories.
    terms = estimated(model)
    np.testing.assert_allclose([fit.stderr[t] for t in terms], [model["se"][t] for t in terms],
                               rtol=1e-3)
    assert fit.loglik == pytest.approx(model["loglik"], rel=1e-9)
    assert fit.aic == pytest.approx(-2 * model["loglik"] + 2 * len(terms))
    assert fit.bic == pytest.approx(-2 * model["loglik"] + np.log(model["nobs"]) * len(terms))


@pytest.mark.parametrize("name", DEPENDENT)
def test_monte_carlo_mle_matches_r(name):
    model = REFERENCE[name]
    fit = ergmx.ergm(load(model["network"]), model["formula"], seed=2026, eval_loglik=False,
                     **options(model))
    assert fit.method == "MCMLE" and fit.converged
    for term in estimated(model):
        r_estimate, r_se = model["mle"][term], model["se"][term]
        assert abs(fit.coef[term] - r_estimate) < 0.25 * r_se, term
        if name not in WEAKLY_IDENTIFIED:
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


def test_contrastive_divergence_estimate_is_its_fixed_point():
    """Samples of 8 MCMC proposals from the observed network, at the CD
    estimate, average to the observed statistics: the definition of CD."""
    from ergmx._estimation import hotelling_pvalue
    from ergmx._model import bind

    g = load("samplk3")
    fit = ergmx.ergm(g, "edges + mutual", estimate="CD", seed=1)
    assert fit.method == "CD" and fit.converged
    assert all(np.isnan(v) for v in fit.stderr.values())
    model = bind(g, "edges + mutual")
    sample, _, _ = model.core.simulate([model.network.edges] * 20_000, list(fit.params), 7, 1, 1, 5)
    sample = sample.reshape(1, -1, 2)
    assert hotelling_pvalue(sample, model.observed(), np.ones(2)) > 0.001
    assert "no standard errors" in str(fit.summary())


@pytest.mark.parametrize("name", ["samplk_mutual", "mesa_gwesp"])
def test_mle_from_contrastive_divergence_start_matches_r(name):
    model = REFERENCE[name]
    fit = ergmx.ergm(load(model["network"]), model["formula"], init="CD", seed=3)
    assert fit.method == "MCMLE" and fit.converged
    for term, r_estimate in model["mle"].items():
        assert abs(fit.coef[term] - r_estimate) < 0.25 * model["se"][term], term


def test_density_guard_stops_degenerate_simulations():
    g = load("samplk3")
    with pytest.raises(ergmx.DegeneracyError, match="more than 60 edges"):
        ergmx.ergm(g, "edges + mutual", init=[3.0, 0.0], seed=1, density_guard=1.0,
                   density_guard_min=60)


def test_stalled_estimation_explains_why():
    g = load("samplk3")
    with pytest.raises(ergmx.DegeneracyError, match="not making progress") as error:
        ergmx.ergm(g, "edges + mutual", init=[4.0, 0.0], seed=1, stall_iterations=2)
    assert "simulated" in str(error.value) and "observed" in str(error.value)


def test_bad_init():
    with pytest.raises(ValueError, match="init must be"):
        ergmx.ergm(load("samplk3"), "edges + mutual", init="SA")
