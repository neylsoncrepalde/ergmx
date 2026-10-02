import numpy as np
import pytest
from conftest import load

import ergmx
from ergmx import F, edges, gwesp, nodematch, offset, triangle
from ergmx._estimation import logistic_regression
from ergmx._model import bind


def test_python_and_string_operators_agree():
    g = load("faux.mesa.high")
    as_string = ergmx.summary_stats(
        g, "F(~edges + triangle, ~!nodematch('Race')) + offset(gwesp(0.5, fixed=TRUE))")
    as_terms = ergmx.summary_stats(
        g, F(edges() + triangle(), nodematch("Race"), negate=True) + offset(gwesp(0.5, fixed=True)))
    assert as_string == as_terms
    assert list(as_string) == ['F(!nodematch("Race"))~edges', 'F(!nodematch("Race"))~triangle',
                               "offset(gwesp.fixed.0.5)"]


def test_operator_errors():
    g = load("faux.mesa.high")
    with pytest.raises(ValueError, match="dyad-independent"):
        F(edges(), triangle())
    with pytest.raises(ValueError, match="can't be filtered"):
        F(offset(edges()), nodematch("Race"))
    with pytest.raises(ValueError, match="offset_coef"):
        ergmx.ergm(g, "offset(edges) + nodematch('Race')")
    with pytest.raises(ValueError, match="1 values"):
        ergmx.ergm(g, "offset(edges) + nodematch('Race')", offset_coef=[1.0, 2.0])
    with pytest.raises(ValueError, match="no offset"):
        ergmx.ergm(g, "edges", offset_coef=[1.0])
    with pytest.raises(ValueError, match="forbids"):
        ergmx.ergm(g, "edges + offset(nodematch('Race'))", offset_coef=[float("-inf")])


def test_minus_infinity_offsets_forbid_ties():
    """A -inf offset on a dyad-independent term fixes the dyads it counts at 0:
    the MLE is the logistic regression on the other dyads."""
    g = load("flomarriage")
    n = g.vcount()
    ties = set(map(frozenset, g.get_edgelist()))
    forbidden = [(i, j) for i in range(n) for j in range(i + 1, n) if frozenset((i, j)) not in ties][:25]
    matrix = np.zeros((n, n))
    for i, j in forbidden:
        matrix[i, j] = matrix[j, i] = 1.0
    g["forbidden"] = matrix
    fit = ergmx.ergm(g, "edges + nodecov('wealth') + offset(edgecov('forbidden'))",
                     offset_coef=[float("-inf")])
    assert fit.method == "MLE" and fit.df == 2
    model = bind(g, "edges + nodecov('wealth')")
    x, y, pairs = model.core.mple_data(model.network.edges)
    keep = np.array([tuple(pair) not in set(forbidden) for pair in pairs.tolist()])
    beta, _, loglik = logistic_regression(x[keep], y[keep], np.ones(keep.sum()))
    np.testing.assert_allclose([fit.coef["edges"], fit.coef["nodecov.wealth"]], beta, rtol=1e-8)
    assert fit.loglik == pytest.approx(loglik)
    for h in fit.simulate(10, seed=1):
        assert not set(map(frozenset, h.get_edgelist())) & set(map(frozenset, forbidden))


def test_gof_with_missing_dyads_compares_with_imputed_networks():
    g = load("samplk3.nonresponse")
    fit = ergmx.ergm(g, "edges + mutual", seed=1)
    result = fit.gof(nsim=50, seed=1)
    observed = result["odegree"].observed
    assert observed.sum() == pytest.approx(18)  # every vertex has some out-degree
    assert np.any(observed != np.round(observed))  # averages over imputations
    edges = result["model"].observed[0]
    assert edges > ergmx.summary_stats(g, "edges")["edges"]  # imputed ties added
    assert fit.mcmc_diagnostics().n_chains == fit.control.n_chains
