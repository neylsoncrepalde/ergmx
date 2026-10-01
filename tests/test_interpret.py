"""Interpreting fitted models: tie probabilities against R's predict(),
marginal effects against ergMargins, confidence intervals against R's
confint(), and tables against texreg's (scripts/r_interpret_reference.R)."""

import json

import numpy as np
import pytest
from conftest import DATA, load
from scipy.special import expit

import ergmx
from ergmx._interpret import _dyads

R = json.loads((DATA / "r_interpret_reference.json").read_text())
#: ergmx names edgecov by its graph attribute; R by the covariate's name.
RENAME = {"edgecov.business": "edgecov.flobusiness"}


def _network(name):
    if name == "Goeyvaerts.weekday20":
        households = ergmx.datasets.load("Goeyvaerts")
        return ergmx.Networks([g for g in households if g["included"] and g["weekday"]][:20])
    return load(name)


@pytest.mark.parametrize("name", sorted(R["predict"]))
def test_conditional_probabilities_match_r(name):
    case = R["predict"][name]
    pred = ergmx.predict(_network(case["network"]), case["formula"], case["coef"])
    ours = {(int(t), int(h)): p for t, h, p in zip(pred.tail, pred.head, pred.p)}
    theirs = {(t, h): p for t, h, p in zip(case["tail"], case["head"], case["p"])}
    # R's formula method leaves out missing dyads; ergmx predicts them too.
    assert set(theirs) <= set(ours)
    if name != "missing":
        assert set(ours) == set(theirs)
    np.testing.assert_allclose([ours[k] for k in theirs], list(theirs.values()), rtol=1e-12)


def test_predictions_of_missing_dyads_and_links():
    g = load("samplk3.nonresponse")
    pred = ergmx.predict(g, "edges + mutual", [-2.0, 1.8])
    assert len(pred) == 18 * 17
    link = ergmx.predict(g, "edges + mutual", [-2.0, 1.8], type="link")
    np.testing.assert_allclose(expit(link.p), pred.p)
    assert set(np.round(link.p, 12)) == {-2.0, -0.2}


def test_prediction_matrix_and_means_by_attribute():
    g = load("faux.mesa.high")
    fit = ergmx.ergm(g, "edges + nodematch('Grade')")
    pred = fit.predict()
    m = pred.matrix()
    assert np.allclose(m, m.T) and np.all(np.diag(m) == 0) and len(pred) == 205 * 204 // 2
    by = pred.mean_by("Grade")
    same, other = expit(sum(fit.params)), expit(fit.params[0])
    assert by["7"][0] == pytest.approx(same) and by["8"][0] == pytest.approx(other)
    frame = pred.to_frame()
    assert list(frame.columns) == ["tail", "head", "p"] and len(frame) == len(pred)


def test_unconditional_probabilities():
    g = load("samplk3")
    fit = ergmx.ergm(g, "edges + mutual", seed=1)
    pred = fit.predict(conditional=False, nsim=400, seed=1)
    # Every dyad has the same probability: the expected density.
    density = fit.simulate(400, seed=2, output="stats")[:, 0].mean() / (18 * 17)
    assert pred.p.mean() == pytest.approx(density, rel=0.03)
    assert not pred.conditional and np.allclose(pred.p * 400, np.round(pred.p * 400))
    with pytest.raises(ValueError, match="link"):
        fit.predict(conditional=False, type="link")


def test_predictions_of_bipartite_and_combined_networks():
    davis = ergmx.datasets.load("davis")
    pred = ergmx.predict(davis, "edges + gwb1dsp(0.5, fixed=TRUE)", [-1.2, 0.3], bipartite=True)
    assert len(pred) == 18 * 14
    assert np.isnan(pred.matrix()[0, 1])  # two women: not a dyad of the network
    nets = ergmx.Networks(load("samplk1"), load("samplk2"))
    combined = ergmx.predict(nets, "N(~edges + mutual)", [-2.0, 2.0])
    assert len(combined) == 2 * 18 * 17 and set(combined.network) == {0, 1}
    assert [m.shape for m in combined.matrix()] == [(18, 18), (18, 18)]
    assert "network" in combined.to_frame().columns


@pytest.mark.parametrize("name", sorted(R["exact"]))
def test_marginal_effects_match_ergmargins(name):
    """Average marginal effects equal ergMargins' (which rounds to 5
    significant digits). Its delta-method standard errors hold the tie
    probabilities fixed: with them fixed, ergmx's are the same."""
    case = R["exact"][name]
    fit = ergmx.ergm(load(case["network"]), case["formula"])
    effects = {RENAME.get(k, k): v for k, v in fit.marginal_effects().to_dict().items()}
    assert set(effects) == set(case["ame"]) | {"edges"}
    x, *_ = _dyads(fit._model)
    p = expit(x @ fit.params)
    fixed = (p * (1 - p)).mean()
    for term, r in case["ame"].items():
        assert effects[term]["AME"] == pytest.approx(r["ame"], rel=1e-4, abs=1e-9)
        k = [RENAME.get(n, n) for n in fit.names].index(term)
        assert fixed * np.sqrt(fit.cov[k, k]) == pytest.approx(r["se"], rel=1e-3)


def test_marginal_effect_standard_errors_are_the_delta_method():
    """Against a numerical delta method, with the probabilities recomputed."""
    fit = ergmx.ergm(load("faux.mesa.high"), "edges + nodematch('Grade') + absdiff('Grade')")
    x, *_ = _dyads(fit._model)

    def ame(theta, k):
        p = expit(x @ theta)
        return theta[k] * (p * (1 - p)).mean()

    effects = fit.marginal_effects()
    for k, name in enumerate(fit.names):
        h = 1e-6
        grad = np.array([(ame(fit.params + h * e, k) - ame(fit.params - h * e, k)) / (2 * h)
                         for e in np.eye(len(fit.params))])
        r = effects.index.index(name)
        assert effects["AME"][r] == pytest.approx(ame(fit.params, k), rel=1e-12)
        assert effects["Delta SE"][r] == pytest.approx(np.sqrt(grad @ fit.cov @ grad), rel=1e-5)


def test_marginal_effects_of_curved_models_skip_the_decay():
    fit = ergmx.ergm(load("flomarriage"), "edges + gwdegree(0.5)", estimate="MPLE")
    effects = fit.marginal_effects()
    assert effects.index == ["edges", "gwdegree"]
    assert np.all(np.isfinite(effects["Delta SE"]))
    with pytest.raises(ValueError, match="covariance"):
        ergmx.ergm(load("samplk3"), "edges + mutual", estimate="CD", seed=1).marginal_effects()


@pytest.mark.parametrize("name", sorted(R["exact"]))
def test_confidence_intervals_match_r(name):
    case = R["exact"][name]
    fit = ergmx.ergm(load(case["network"]), case["formula"])
    ci = fit.confint()
    index = [RENAME.get(n, n) for n in ci.index]
    # R's glm standard errors are accurate to about 1e-4 (test_estimation.py).
    np.testing.assert_allclose(ci["2.5 %"], [case["confint"]["low"][n] for n in index], rtol=1e-3)
    np.testing.assert_allclose(ci["97.5 %"], [case["confint"]["high"][n] for n in index], rtol=1e-3)
    odds = fit.odds_ratios(0.9)
    np.testing.assert_allclose(odds["Odds ratio"], np.exp([fit.coef[n] for n in odds.index]))
    assert list(odds.columns) == ["Odds ratio", "5 %", "95 %"]


def _flo_fits():
    g = load("flomarriage")
    return [ergmx.ergm(g, f) for f in ("edges", "edges + nodecov('wealth')",
                                        "edges + nodecov('wealth') + absdiff('wealth') + edgecov('business')")]


def test_tables_are_texregs():
    f1, f2, f3 = _flo_fits()
    f4 = ergmx.ergm(load("faux.mesa.high"), "edges + nodematch('Grade') + nodefactor('Sex') + absdiff('Grade')")
    tables = R["tables"]
    assert ergmx.table(f1, f2, f3, rename=RENAME).to_text() == tables["three"].lstrip("\n")
    assert ergmx.table(f2, digits=3).to_text() == tables["digits3"].lstrip("\n")
    assert ergmx.table([f1, f2], names=["Null", "Wealth"]).to_text() == tables["named"].lstrip("\n")
    assert ergmx.table(f4, f2).to_text() == tables["mesa"].lstrip("\n")
    assert ergmx.table(f1, f2).to_latex() == tables["latex"].lstrip("\n")
    assert ergmx.table(f1, f2).to_html() == tables["html"].lstrip("\n")


def test_table_options():
    f1, f2, f3 = _flo_fits()
    t = ergmx.table(f1, f2, f3, omit="wealth", include_bic=False, stars=(0.01, 0.05, 0.1))
    text = str(t)
    assert "wealth" not in text and "BIC" not in text and "AIC" in text
    assert text.rstrip().endswith("*** p < 0.01; ** p < 0.05; * p < 0.1")
    md = t.to_markdown()
    assert md.startswith("| | Model 1 | Model 2 | Model 3 |") and r"\*\*\*" in md
    assert t._repr_html_() == t.to_html()
    with pytest.raises(TypeError, match="fitted models"):
        ergmx.table(load("flomarriage"))
    with pytest.raises(ValueError, match="names"):
        ergmx.table(f1, f2, names=["one"])


def test_tables_of_monte_carlo_fits_and_offsets():
    g = load("samplk3")
    fit = ergmx.ergm(g, "edges + mutual", seed=1)
    offset = ergmx.ergm(g, "edges + offset(mutual)", offset_coef=[2.0], seed=1)
    text = str(ergmx.table(fit, offset))
    assert "offset(mutual)" in text and "2.00" in text
    row = next(line for line in text.splitlines() if line.startswith("offset(mutual)"))
    assert "*" not in row  # fixed, not tested
    frame = fit.to_frame()
    assert list(frame.columns) == ["Estimate", "Std. Error", "MCMC %", "z value", "Pr(>|z|)"]
    assert frame.loc["mutual", "Estimate"] == fit.coef["mutual"]
