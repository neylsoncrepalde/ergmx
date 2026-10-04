"""Egocentric ERGMs, against R's ergm.ego (scripts/r_ego_reference.R): the
population statistics egocentric data estimate, their covariances, and the
fits."""

import json

import numpy as np
import pytest
from conftest import DATA, load

import ergmx
from ergmx import EgoData

R = json.loads((DATA / "r_ego_reference.json").read_text())


def _data(which: str) -> EgoData:
    census = EgoData.from_network(load("faux.mesa.high"))
    return census if which == "census" else census.sample(np.array(R["sampled_egos"]) - 1)


@pytest.mark.parametrize("which", ["census", "sample"])
def test_ego_statistics_match_r(which):
    data, r = _data(which), R[which]
    stats, cov = ergmx.ego_stats(R["stat_formula"], data, scaleto=205)
    assert list(stats) == r["names"]
    np.testing.assert_allclose(list(stats.values()), r["mean"], rtol=1e-12)
    # R leaves out the covariances of meandeg, which doesn't scale.
    r_cov = np.array([[np.nan if v == "NA" else v for v in row] for row in r["survey"]], dtype=float)
    known = np.isfinite(r_cov)
    np.testing.assert_allclose(cov[known], r_cov[known], rtol=1e-9, atol=1e-9)
    scaling = R["stat_formula"].replace(" + meandeg", "")
    # R's other estimators scale to its pseudo-population: 2 replicates of each of 100 egos.
    size = 205 if which == "census" else 200
    for est in ("asymptotic", "naive", "jackknife"):
        m, v = ergmx.ego_stats(scaling, data, scaleto=size, stats_est=est)
        np.testing.assert_allclose(list(m.values()), r[est]["mean"], rtol=1e-9, err_msg=est)
        np.testing.assert_allclose(v, np.array(r[est]["cov"]), rtol=1e-7, atol=1e-9, err_msg=est)


def test_census_statistics_are_the_networks():
    """Every vertex an ego: the estimates are the network's statistics."""
    g = load("faux.mesa.high")
    formula = R["stat_formula"].replace(" + meandeg", "")
    stats, _ = ergmx.ego_stats(formula, EgoData.from_network(g), scaleto=g.vcount())
    np.testing.assert_allclose(list(stats.values()), list(ergmx.summary_stats(g, formula).values()), rtol=1e-12)


@pytest.mark.parametrize("name", list(R["fits"]))
def test_ergm_ego_matches_r(name):
    r = R["fits"][name]
    fit = ergmx.ergm_ego(r["formula"], _data(r["data"]), popsize=r["popsize"], seed=1)
    assert fit.ppopsize == r["ppopsize"]
    np.testing.assert_allclose(list(fit.stats.values()), list(r["stats"].values()), rtol=1e-9)
    # R's fits are Monte Carlo MLEs, even of dyad-independent models.
    tolerance = 0.25 if fit.method == "MCMLE" else 0.1
    for term, value in r["coef"].items():
        name_ = "offset(edges)" if term == "offset(netsize.adj)" else term
        if term.startswith("offset("):
            assert fit.coef[name_] == pytest.approx(value)
            continue
        assert abs(fit.coef[name_] - value) < tolerance * r["se"][term], term
        # R's standard errors use its Monte Carlo estimate of the information, which
        # the sandwich amplifies: up to 15% off the exact one.
        assert fit.stderr[name_] == pytest.approx(r["se"][term], rel=0.2), term
    assert "egos" in str(fit.summary())
    text = str(ergmx.table(fit, omit="offset"))
    assert all(term in text for term in r["coef"] if not term.startswith("offset(")) and "AIC" not in text


def test_ego_gof_observes_the_egos():
    """gof()'s observed statistics are those the egos estimate, per capita, as
    ergm.ego's; of a census, the network's own."""
    r = next(f for f in R["fits"].values() if "gof" in f)
    fit = ergmx.ergm_ego(r["formula"], _data(r["data"]), popsize=r["popsize"], seed=1)
    gof = fit.gof(nsim=10, seed=1)
    assert [t.name for t in gof] == ["degree", "espartners", "model"]
    assert gof["degree"].labels[-1] == r["gof"]["degree"]["names"][-1].replace("deg", "")
    for stat in ("model", "degree", "espartners"):
        np.testing.assert_allclose(gof[stat].observed, r["gof"][stat]["obs"], rtol=1e-12, atol=1e-12, err_msg=stat)
    assert gof["degree"].simulated.shape == (10, len(r["gof"]["degree"]["obs"]))
    np.testing.assert_allclose(gof["degree"].simulated.sum(axis=1), 1)
    g = load("faux.mesa.high")
    census = ergmx.ergm_ego("edges + nodematch('Grade')", EgoData.from_network(g), seed=1).gof(nsim=5, seed=1)
    top = len(census["degree"].observed) - 1
    degrees = np.bincount(g.degree(), minlength=top + 1) / g.vcount()
    np.testing.assert_allclose(census["degree"].observed, [*degrees[:top], degrees[top:].sum()], rtol=1e-12)
    with pytest.raises(ValueError, match="can't be estimated from egocentric data"):
        fit.gof(stats=["distance"])
    no_ties = EgoData(_data("sample").egos, {".egoID": [1], "Grade": [7]})
    with pytest.raises(ValueError, match="ties among the alters"):
        ergmx.ergm_ego("edges", no_ties, seed=1).gof(stats=["espartners"])


def test_ego_data_from_tables():
    egos = {"id": [1, 2, 3], "sex": ["F", "M", "F"]}
    alters = {"id": [1, 1, 2, 3], "alter": [10, 11, 12, 13], "sex": ["M", "F", "F", "F"]}
    aaties = {"id": [1], ".srcID": [10], ".tgtID": [11]}
    data = EgoData(egos, alters, aaties, ego_id="id", alter_id="alter")
    assert data.n == 3 and data.degree.tolist() == [2, 1, 1]
    stats, _ = ergmx.ego_stats("edges + nodematch('sex') + transitiveties + triangle", data)
    # Half of each ego's ties; ties with a shared partner; triangles seen by three egos.
    assert stats == pytest.approx({"edges": 2.0, "nodematch.sex": 1.0, "transitiveties": 1.0, "triangle": 1 / 3})
    with pytest.raises(ValueError, match="not supported"):
        ergmx.ego_stats("edges + kstar(2)", data)
    with pytest.raises(ValueError, match="no ego has"):
        ergmx.ego_stats("nodefactor('sex')", EgoData(egos, {**alters, "sex": ["M", "F", "X", "F"]}, ego_id="id"))
