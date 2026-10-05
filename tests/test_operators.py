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


# -- ergm's operators on formulas (Sum, Prod, Log, Exp, Symmetrize, Label...) -------------------

import json  # noqa: E402

from conftest import DATA  # noqa: E402

OPS_REFERENCE = DATA / "r_operators_reference.json"
OPS = json.loads(OPS_REFERENCE.read_text()) if OPS_REFERENCE.exists() else None


@pytest.mark.skipif(OPS is None, reason="no R reference")
def test_formula_operators_match_r():
    """Each operator's statistics and names, as ergm's, on the Florentine
    marriages (undirected) and Sampson's monks (directed)."""
    for key, r in OPS["stats"].items():
        g = load("flomarriage" if key.startswith("u") else "samplk3")
        stats = ergmx.summary_stats(g, r["formula"])
        names, values = np.atleast_1d(r["names"]).tolist(), np.atleast_1d(r["values"])  # R's JSON unboxes 1
        assert list(stats) == names, r["formula"]
        # Prod() and Exp() add up changes of exponentials, in R and ergmx: 1e-8 apart.
        np.testing.assert_allclose(list(stats.values()), values, rtol=1e-6, err_msg=r["formula"])


@pytest.mark.skipif(OPS is None, reason="no R reference")
@pytest.mark.parametrize("name", list(OPS["fits"]) if OPS else [])
def test_formula_operator_fits_match_r(name):
    from ergmx._operators import Curve

    r = OPS["fits"][name]
    g = load(r["network"])
    formula = r["formula"]
    if name == "curve_mle":  # R's map and gradient functions, as Python's
        formula = ergmx.Formula([Curve("edges + nodematch('Grade') + nodematch('Race')", {"a": 0, "b": 0},
                                       map=lambda x, n: np.array([x[0], x[1], x[1]]),
                                       gradient=lambda x, n: np.array([[1, 0, 0], [0, 1, 1]]))])
    fit = ergmx.ergm(g, formula, estimate=r["estimate"], seed=1, eval_loglik=False)
    assert fit.names == r["names"]
    if name == "symmetrize_mle":
        # The weakly symmetrized ties are the ties less the mutual pairs: the
        # reciprocity model, reparametrized, whose MLE is exact. (ergm 4.12's
        # Monte Carlo MLE ends far from it, at 1.56 and -6.26.)
        exact = ergmx.ergm(g, "edges + mutual").params
        expected = np.array([exact[0] + exact[1], -exact[1]])
        assert np.all(np.abs(fit.params - expected) < 0.2 * np.array(list(fit.stderr.values()))), (fit.params, expected)
        return
    if r["estimate"] == "MPLE":
        np.testing.assert_allclose(fit.params, r["coef"], rtol=1e-6)
    else:
        se = np.array(r["se"])
        assert np.all(np.abs(fit.params - r["coef"]) < 0.3 * se), (fit.params, r["coef"], se)


@pytest.mark.parametrize("directed", [False, True])
def test_formula_operators_track_their_statistics(directed):
    """The statistics the sampler tracks, change by change, against those of
    the simulated networks."""
    if directed:
        g = load("samplk3")
        formula = ("edges + Log(~mutual + istar(2)) + Symmetrize(~triangle + kstar(2)) + "
                   "Symmetrize(~edges + nodematch('group'), 'upper') + Symmetrize(~edges, 'strong') + "
                   "Prod(list(~mutual, ~edges), 'p') + Sum(list(c(1, -1) ~ istar(2:3)), 's') + Exp(~isolates)")
    else:
        g = load("flomarriage")
        formula = ("edges + Log(~triangle + kstar(2), log0 = -3) + Sum(list(2 ~ kstar(2:3), ~degree(1:2)), 'w') + "
                   "Label(~gwesp(0.5, fixed = TRUE), 'L') + Passthrough(~degree(0))")
    names = list(ergmx.summary_stats(g, formula))
    coef = np.zeros(len(names))
    coef[0] = -1.5
    tracked = ergmx.simulate(g, formula, coef, 15, seed=2, interval=300, output="stats")
    networks = ergmx.simulate(g, formula, coef, 15, seed=2, interval=300)
    recomputed = [list(ergmx.summary_stats(h, formula).values()) for h in networks]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-7, atol=1e-7)


def test_formula_operator_errors():
    g = load("flomarriage")
    with pytest.raises(ValueError, match="directed networks"):
        ergmx.summary_stats(g, "Symmetrize(~edges)")
    with pytest.raises(ValueError, match="differ in number"):
        ergmx.summary_stats(g, "Sum(list(~edges, ~kstar(1:2)), 'x')")
    with pytest.raises(NotImplementedError, match="curved"):
        ergmx.summary_stats(g, "Sum(~gwesp(), 'x')")
    with pytest.raises(ValueError, match="2 labels for 1"):
        ergmx.summary_stats(g, "Label(~edges, c('a', 'b'), pos='replace')")
    with pytest.raises(ValueError, match="needs its gradient"):
        from ergmx._operators import Curve

        Curve("edges", ["a"], map=lambda x, n: x)
    with pytest.raises(ValueError, match="rule="):
        ergmx.summary_stats(load("samplk3"), "Symmetrize(~edges, 'both')")
    # Offset() fixes some parameters; For() and I() splice terms in.
    fit = ergmx.ergm(g, "edges + Offset(~kstar(2) + triangle, c(0.1), 'triangle')", seed=1, estimate="MPLE")
    assert fit.names == ["edges", "kstar2"]
    assert ergmx.summary_stats(g, "For(~degree(d) + nodecov(a), d = 1:2, a = c('wealth'))") == \
        ergmx.summary_stats(g, "degree(1) + nodecov('wealth') + degree(2) + nodecov('wealth')")
