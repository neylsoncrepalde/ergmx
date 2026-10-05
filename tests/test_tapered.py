"""Tapered ERGMs (ergm_tapered), against R's ergm.tapered, and the Taper()
operator's statistics."""

import json
import pickle

import numpy as np
import pytest
from conftest import DATA

import ergmx
from ergmx.datasets import load

REFERENCE = DATA / "r_tapered_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["fits"]) if R else [])
def test_tapered_fits_match_r(name):
    """The estimates, within 0.3 standard errors of the mean of R's over 3
    seeds, the tapering coefficients exactly, and the standard errors within
    25%, but where ergm.tapered's own vary by a factor of 2 between seeds
    (its sample's means are far from the network's)."""
    r = R["fits"][name]
    options = {k: r[k] for k in ("r", "tau", "taper_terms") if k in r}
    fit = ergmx.ergm_tapered(load(r["network"]), r["formula"], seed=1, eval_loglik=False, **options)
    assert fit.names == r["names"]
    se = np.array(r["se"])
    assert np.all(np.abs(fit.params - r["coef"]) < 0.3 * se), (fit.params, r["coef"], se)
    np.testing.assert_allclose([fit.tapering_coef[n] for n in r["names"]],
                               [r["tapering_coef"][n] for n in r["names"]], rtol=1e-8)
    stable = np.ptp(np.array(r["se_seeds"]), axis=0) < 0.3 * se
    np.testing.assert_allclose(np.array(list(fit.stderr.values()))[stable], se[stable], rtol=0.25)


def test_taper_statistic_and_its_changes():
    """Taper()'s penalty sum_k tau_k (g_k - m_k)^2, and the sampler's tracked
    statistics against those of the simulated networks."""
    g = load("flomarriage")
    stats = ergmx.summary_stats(g, "Taper(~edges + triangle, coef = c(0.1, 0.5), m = c(18, 5))")
    assert stats == {"edges": 20.0, "triangle": 3.0, "Taper_Penalty": pytest.approx(0.1 * 4 + 0.5 * 4)}
    # A single coefficient is a multiplier, as ergm.tapered's: coef / (4 m).
    default = ergmx.summary_stats(g, "Taper(~edges + kstar(2), coef = 2, m = c(10, 40))")
    assert default["Taper_Penalty"] == pytest.approx(2 / 40 * 100 + 2 / 160 * 49)
    formula = "edges + Taper(~kstar(2) + triangle, coef = c(0.01, 0.2), m = c(30, 2))"
    coef = [-1.0, 0.05, 0.1, -1.0]
    tracked = ergmx.simulate(g, formula, coef, 10, seed=1, interval=100, output="stats")
    recomputed = [list(ergmx.summary_stats(h, formula).values())
                  for h in ergmx.simulate(g, formula, coef, 10, seed=1, interval=100)]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9)


def test_tapering_stops_degeneracy():
    """edges + kstar(2) + triangle on the Florentine marriages: degenerate as
    an ERGM, fitted when tapered, with simulated statistics around the
    network's."""
    g = load("flomarriage")
    fit = ergmx.ergm_tapered(g, "edges + kstar(2) + triangle", seed=1)
    assert fit.converged and fit.loglik is not None
    sims = np.asarray(fit.simulate(200, seed=2, output="stats"))[:, :3]
    observed = np.array(list(ergmx.summary_stats(g, "edges + kstar(2) + triangle").values()))
    assert np.all(np.abs(sims.mean(axis=0) - observed) < 3 * sims.std(axis=0) / np.sqrt(20))
    text = str(fit.summary())
    assert "Tapered (r = 2)" in text and "Taper_Penalty" not in text.split("Tapered")[0]
    assert pickle.loads(pickle.dumps(fit)).tapering_coef == fit.tapering_coef


def test_tapered_options():
    g = load("faux.mesa.high")
    formula = "edges + nodematch('Grade') + gwesp(0.5, fixed = TRUE)"
    # Only the gwesp term, and its center elsewhere: the Monte Carlo MLE's standard errors.
    fit = ergmx.ergm_tapered(g, formula, taper_terms="gwesp(0.5, fixed = TRUE)", tapering_centers={"gwesp.fixed.0.5": 180},
                             seed=1, eval_loglik=False)
    assert fit.tapering_coef == {"edges": 0.0, "nodematch.Grade": 0.0, "gwesp.fixed.0.5": 1 / (4 * 180)}
    assert fit.tapering_centers == {"gwesp.fixed.0.5": 180.0}
    mple = ergmx.ergm_tapered(g, formula, estimate="MPLE", beta=10)
    assert mple.method == "MPLE" and mple.tapering_coef["edges"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="not terms of the formula"):
        ergmx.ergm_tapered(g, formula, taper_terms="triangle")
    with pytest.raises(ValueError, match="'MLE' or 'MPLE'"):
        ergmx.ergm_tapered(g, formula, estimate="CD")
