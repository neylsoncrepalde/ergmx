"""Valued models of a series of networks (NetSeries() with response=),
against R's tergm and ergm.count: Sampson's monks' liking at three times,
the values the ranks of each monk's choices; and edgecov(".PrevNet"), the
network before each transition, in binary series too."""

import json
import warnings

import numpy as np
import pytest
from conftest import DATA

import ergmx
from ergmx.datasets import load

REFERENCE = DATA / "r_valued_series_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None
pytestmark = pytest.mark.skipif(R is None, reason="no R reference")


def _monks():
    return ergmx.NetSeries(load("samplk1"), load("samplk2"), load("samplk3"))


@pytest.mark.parametrize("name", ["valued", "binary"])
def test_series_statistics_match_r(name):
    r = R["stats"][name]
    response = {"response": "score"} if name == "valued" else {}
    stats = ergmx.summary_stats(_monks(), r["formula"], **response)
    assert list(stats) == r["names"]
    np.testing.assert_allclose(list(stats.values()), r["values"], rtol=1e-12)


def test_valued_n_of_a_series():
    """N() of valued terms on a series (which R's valued N() refuses): the
    statistics of R's N() on Networks() of the transitions, with .TimeID for
    .NetworkID."""
    r = R["stats"]["valued_N"]
    stats = ergmx.summary_stats(_monks(), "N(~sum + nonzero + mutual(form = 'min'), ~.TimeID)", response="score")
    assert list(stats) == [n.replace(".NetworkID", ".TimeID") for n in r["names"]]
    np.testing.assert_allclose(list(stats.values()), r["values"], rtol=1e-12)


@pytest.mark.parametrize("name", list(R["fits"]) if R else [])
def test_series_fits_match_r(name):
    """The estimates within 0.3 standard errors of the mean of R's over three
    seeds, the standard errors within 15%."""
    r = R["fits"][name]
    options = {"response": r["response"], "reference": r["reference"]} if "response" in r else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ergmx.ergm(_monks(), r["formula"], seed=1, **options)
    assert fit.converged and fit.names == r["names"]
    se = np.array(r["se"])
    assert np.all(np.abs(fit.params - r["coef"]) < 0.3 * se), (fit.params, r["coef"])
    np.testing.assert_allclose(list(fit.stderr.values()), se, rtol=0.15)


def test_valued_series_tools():
    """tergm() takes valued series, as R's; simulations are each
    transition's network, given the one before, with the values in the
    response's attribute; gof() and the summary are a series'."""
    nets = [load("samplk1"), load("samplk2"), load("samplk3")]
    formula = "sum + nonzero + mutual(form = 'min') + edgecov('.PrevNet', 'score')"
    fit = ergmx.tergm(nets, formula, response="score", reference="Binomial(3)", seed=1)
    same = ergmx.ergm(_monks(), formula, response="score", reference="Binomial(3)", seed=1)
    np.testing.assert_allclose(fit.params, same.params)
    text = str(fit.summary())
    assert "Conditional" in text and "2 transitions, each conditional on the network before it" in text
    sims = fit.simulate(2, seed=1)
    assert len(sims) == 2 and len(sims[0]) == 2
    assert sims[0][0].vcount() == 18 and set(sims[0][0].es["score"]) <= {1.0, 2.0, 3.0}
    stats = np.asarray(fit.simulate(200, seed=2, output="stats"))
    observed = np.array(list(ergmx.summary_stats(_monks(), formula, response="score").values()))
    assert np.all(np.abs(stats.mean(axis=0) - observed) < 4 * stats.std(axis=0) / np.sqrt(200) + 0.05 * observed)
    gof = fit.gof(nsim=20, seed=1)
    assert [t.name for t in gof][0] == "model"
    with pytest.raises(ValueError, match="time_slices= is for binary series"):
        fit.simulate(2, time_slices=3)
    with pytest.raises(ValueError, match="estimate='CMLE'"):
        ergmx.tergm(nets, "sum", response="score", reference="Poisson", estimate="CMPLE")


def test_tergm_operators_are_binary():
    with pytest.raises(ValueError, match="tergm's operators are for binary series"):
        ergmx.ergm(_monks(), "Form(~sum)", response="score", reference="Poisson")
    with pytest.raises(ValueError, match="tergm's operators are for binary series"):
        ergmx.summary_stats(_monks(), "sum + Cross(~nonzero)", response="score")


def test_previous_networks():
    """edgecov(".PrevNet") is the network before each transition: with
    imputed ties, their values are unknown; elsewhere than in a series, it's
    an ordinary graph attribute."""
    nets = [load("samplk1"), load("samplk2"), load("samplk3")]
    first = nets[0].copy()
    # A tie at the first time, unknown, that the next network's tie imputes.
    kept = next(e.index for e in first.es if nets[1].get_eid(*e.tuple, error=False) >= 0)
    first.es["na"] = [e.index == kept for e in first.es]
    series = ergmx.NetSeries(first, *nets[1:], na_impute="next")
    assert ergmx.summary_stats(series, "edgecov('.PrevNet')")["edgecov..PrevNet..PrevNet"] == \
        ergmx.summary_stats(_monks(), "edgecov('.PrevNet')")["edgecov..PrevNet..PrevNet"]
    with pytest.raises(ValueError, match="imputed .* have no values"):
        ergmx.summary_stats(series, "edgecov('.PrevNet', 'score')")
    with pytest.raises(KeyError, match="no graph attribute '.PrevNet'"):
        ergmx.summary_stats(nets[0], "edgecov('.PrevNet')")
    # A graph attribute holding a network, with an edge attribute's values, named as ergm's.
    g = nets[1].copy()
    g["before"] = nets[0]
    stats = ergmx.summary_stats(g, "edgecov('before') + edgecov('before', 'score')")
    assert list(stats) == ["edgecov.before.before", "edgecov.before.score"]
    prev = {(e.source, e.target): e["score"] for e in nets[0].es}
    assert stats["edgecov.before.score"] == sum(prev.get((e.source, e.target), 0) for e in g.es)
    with pytest.raises(ValueError, match="x must be a network"):
        ergmx.summary_stats(g, ergmx.terms.edgecov(np.eye(18), "score"))
