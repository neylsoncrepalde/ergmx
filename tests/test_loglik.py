import itertools

import igraph as ig
import numpy as np
import pytest
from conftest import REFERENCE, load, models_with

import ergmx
from ergmx._estimation import Control
from ergmx._loglik import bridge_loglik
from ergmx._model import bind

DEPENDENT = [n for n in models_with("mle") if not REFERENCE[n]["dyad_independent"]]


def exact_loglik(model, n, theta, directed=False):
    pairs = ([(i, j) for i in range(n) for j in range(n) if i != j] if directed
             else list(itertools.combinations(range(n), 2)))
    stats = np.array([
        model.core.summary(np.array([p for p, b in zip(pairs, bits) if b], dtype=np.uint32).reshape(-1, 2))
        for bits in itertools.product((0, 1), repeat=len(pairs))
    ])
    return float(model.observed() @ theta - np.logaddexp.reduce(stats @ theta))


@pytest.mark.parametrize("directed", [False, True])
def test_path_sampling_matches_the_exact_loglikelihood(directed):
    """On networks small enough to enumerate, the estimate is unbiased and its
    standard error is right."""
    if directed:
        g = ig.Graph(n=4, edges=[(0, 1), (1, 0), (1, 2), (2, 3), (0, 2)], directed=True)
        g.vs["sex"] = [1, 1, 2, 2]
        formula = "edges + mutual + ttriple + gwesp(0.5, fixed=TRUE) + nodeofactor('sex')"
        theta = np.array([-0.6, 0.8, 0.3, 0.2, 0.3])
    else:
        g = ig.Graph(n=6, edges=[(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)])
        g.vs["grp"] = [0, 0, 0, 1, 1, 1]
        formula = "edges + triangle + gwesp(0.7, fixed=TRUE) + nodematch('grp')"
        theta = np.array([-0.8, 0.4, 0.3, 0.5])
    model = bind(g, formula)
    exact = exact_loglik(model, g.vcount(), theta, directed)
    estimates = np.array([bridge_loglik(model, theta, Control(), 64, np.random.default_rng(s))
                          for s in range(12)])
    values, ses = estimates[:, 0], estimates[:, 1]
    assert abs(values.mean() - exact) < 4 * values.std(ddof=1) / np.sqrt(len(values))
    assert ses.mean() == pytest.approx(values.std(ddof=1), rel=0.5)


@pytest.mark.parametrize("name", DEPENDENT)
def test_loglikelihood_is_close_to_r(name):
    """R's ergm integrates with a 16-point midpoint rule, which is off by up to
    a few units on these models, so the comparison is loose; the exact tests
    above check the estimator."""
    model = REFERENCE[name]
    fit = ergmx.ergm(load(model["network"]), model["formula"], seed=1)
    assert fit.loglik_se > 0
    assert abs(fit.loglik - model["loglik"]) < 2.0 + 3 * fit.loglik_se


def test_compare_models():
    g = load("faux.mesa.high")
    small = ergmx.ergm(g, "edges + nodematch('Grade')")
    large = ergmx.ergm(g, "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)", seed=1)
    other = ergmx.ergm(g, "edges + nodematch('Race')")
    comparison = ergmx.compare(small, large, other)
    tests = comparison.lr_tests()
    assert tests[0] is None and tests[2] is None  # "other" is not nested in "large"
    chi2, df, p = tests[1]
    assert df == 1 and chi2 > 100 and p < 1e-10
    assert np.argmin(comparison.aic) == 1
    text = str(comparison)
    assert "Model comparison" in text and "Monte Carlo estimates" in text


def test_compare_needs_loglikelihoods():
    g = load("samplk3")
    fit = ergmx.ergm(g, "edges + mutual", seed=1, eval_loglik=False)
    assert fit.loglik is None and fit.aic is None
    assert "not computed" in str(fit.summary())
    with pytest.raises(ValueError, match="eval_loglik=True"):
        ergmx.compare(ergmx.ergm(g, "edges"), fit)
    with pytest.raises(ValueError, match="at least two"):
        ergmx.compare(fit)
