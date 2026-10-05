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


# -- Estimated tapering (fixed=False) -----------------------------------------------------------


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["objective_checks"]) if R else [])
def test_the_strengths_objective_is_ergm_tapereds(name):
    """The kurtosis-penalized objective, on the sample of ergm.tapered's
    first fit (at strength 1), at several strengths, and the strength it
    proposes, with its importance weights' effective size: R's exactly."""
    from ergmx._tapered import TaperingControl, propose_strength, strength_objective

    check = R["objective_checks"][name]
    x, s0 = np.array(check["sample"], dtype=float), check["strength0"]
    control = TaperingControl()
    np.testing.assert_allclose([strength_objective(x, s, s0, control) for s in check["strengths"]],
                               check["objective"], rtol=0, atol=1e-10)
    proposed, objective, size = propose_strength(x, s0, control)
    assert proposed == pytest.approx(check["proposed"], abs=1e-12)
    assert objective == pytest.approx(np.ravel(check["proposed_objective"])[0], abs=1e-10)
    assert size == pytest.approx(check["effective_size"], rel=1e-10)


def test_brents_method_is_rs_optimize():
    """The minima R's optimize() finds (R 4.5), to the last digit (but the
    power function's)."""
    from ergmx._tapered import _brent_min

    cases = [(lambda x: np.sin(3 * x) + 0.1 * x * x, (-2, 2), -0.51220133121067679),
             (lambda x: (x - 1 / 3) ** 2, (0, 1), 0.33333333333333331),
             (lambda x: abs(x - 2.5) + 0.01 * x, (0, 3), 2.500007253369108),
             (lambda x: -x, (1 / 3 + 1e-4, 3 - 1e-4), 2.9998326700838729),
             (lambda x: x**4 - 3 * x, (-1, 2), 0.90857123365776771)]
    for f, interval, minimum in cases:
        assert _brent_min(f, *interval) == pytest.approx(minimum, rel=0, abs=4e-16)


ESTIMATED = list(R["estimated"]) if R and "estimated" in R else []


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", ESTIMATED)
def test_estimated_tapering_matches_r(name):
    """The estimated strength within the range of R's three seeds' (mostly
    the interval's top, 3), the coefficients within 0.3 of R's standard
    errors (and twice the spread of R's seeds' estimates) of their mean, and
    the tapering coefficients those given times the strength, as R's.
    faux.mesa.high's edges + triangle, degenerate at the first strength, 1,
    starts at 2, where R's iterations pass; it is near-degenerate even
    tapered (R's standard errors vary by a factor of 3 between seeds)."""
    from ergmx._tapered import TaperingControl

    r = R["estimated"][name]
    control = TaperingControl(init=2.0) if name == "mesa_triangle" else None
    fit = ergmx.ergm_tapered(load(r["network"]), r["formula"], fixed=False, tapering_control=control, seed=1,
                             eval_loglik=False)
    strengths = [s["strength"] for s in r["seeds"]]
    assert min(strengths) - 0.01 <= fit.tapering_strength <= max(strengths) + 0.01
    assert fit.r == pytest.approx(2 / np.sqrt(fit.tapering_strength))
    assert fit.names == r["names"]
    coefs = np.array([s["coef"] for s in r["seeds"]])
    coef, spread = coefs.mean(axis=0), coefs.std(axis=0, ddof=1)
    se = np.mean([s["se"] for s in r["seeds"]], axis=0)
    assert np.all(np.abs(fit.params - coef) < 0.3 * se + 2 * spread), (fit.params, coef, se)
    base = {n: t / r["seeds"][-1]["strength"] for n, t in r["tapering_coef"].items()}
    np.testing.assert_allclose([fit.tapering_coef[n] / fit.tapering_strength for n in r["names"]],
                               [base[n] for n in r["names"]], rtol=1e-8)
    assert fit.tapering_converged and fit.tapering_history[-1]["strength"] == fit.tapering_strength


def test_estimated_tapering_after_a_degenerate_fit(monkeypatch):
    """A fit that stops for degeneracy moves the strength halfway to the
    interval's top; the strength counts as a parameter (AIC, BIC), and the
    summary says it was estimated."""
    import ergmx._simulate

    real, calls = ergmx._simulate.ergm, []

    def first_degenerate(*args, **kwargs):
        calls.append(kwargs.get("init"))
        if len(calls) == 1:
            raise ergmx.DegeneracyError("a simulated network had more than 10000 edges")
        return real(*args, **kwargs)

    monkeypatch.setattr(ergmx, "ergm", first_degenerate)
    g = load("flomarriage")
    fit = ergmx.ergm_tapered(g, "edges + kstar(2) + triangle", fixed=False, seed=1)
    first, second = fit.tapering_history[:2]
    assert np.isnan(first["proposed"]) and first["strength"] == 1.0
    assert second["strength"] == pytest.approx((1 + 3 - 1e-4) / 2)
    assert fit.tapering_converged and fit.loglik is not None
    fixed = ergmx.ergm_tapered(g, "edges + kstar(2) + triangle", seed=1, eval_loglik=False)
    assert fit.df == fixed.df + 1 == 4
    assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * 4)
    text = str(fit.summary())
    assert "the strength estimated (2.9998; r = 1.155)" in text and "Taper_Penalty" not in text
    assert pickle.loads(pickle.dumps(fit)).tapering_strength == fit.tapering_strength


def test_estimated_tapering_options():
    from ergmx._tapered import TaperingControl

    g = load("flomarriage")
    with pytest.raises(ValueError, match="estimate='MLE'"):
        ergmx.ergm_tapered(g, "edges + triangle", fixed=False, estimate="MPLE")
    with pytest.raises(ValueError, match="fixed=False"):
        ergmx.ergm_tapered(g, "edges + triangle", tapering_control=TaperingControl())
    with pytest.raises(ValueError, match="in the interval"):
        TaperingControl(init=5)
    # One iteration: not converged, with a warning; the strength is the fit's.
    with pytest.warns(UserWarning, match="did not converge after 1 iterations"):
        fit = ergmx.ergm_tapered(g, "edges + kstar(2) + triangle", fixed=False, seed=1, eval_loglik=False,
                                 tapering_control=TaperingControl(maxit=1))
    assert fit.tapering_strength == 1.0 and not fit.tapering_converged
    assert "not converged" in str(fit.summary())
