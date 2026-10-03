"""Goodness of fit by network (gofN(), as ergm.multi's): against exact
expectations of dyad-independent models, and against R's."""

import json
import warnings

import numpy as np
import pytest
from conftest import DATA, load
from scipy.special import expit

import ergmx
from ergmx import ErgmDifferenceWarning

R = json.loads((DATA / "r_reference.json").read_text())["gofn"]


def monks():
    return [load(f"samplk{k}") for k in (1, 2, 3)]


def _exact_edges(nets, theta):
    """Each network's expected number of ties and its variance, under
    N(~edges + nodematch('group')): independent dyads."""
    means, variances = [], []
    for g in nets:
        group = np.array(g.vs["group"])
        same = (group[:, None] == group[None, :])[~np.eye(g.vcount(), dtype=bool)]
        p = expit(theta[0] + theta[1] * same)
        means.append(p.sum())
        variances.append((p * (1 - p)).sum())
    return np.array(means), np.array(variances)


def test_fitted_values_and_variances_are_the_models():
    nets = monks()
    fit = ergmx.ergm(ergmx.Networks(nets), "N(~edges + nodematch('group'))")
    nsim = 2000
    result = ergmx.gofN(fit, "edges", nsim=nsim, seed=1)
    mean, var = _exact_edges(nets, fit.params)
    t = result["edges"]
    np.testing.assert_array_equal(t.observed, [g.ecount() for g in nets])
    assert np.all(np.abs(t.fitted - mean) < 4 * np.sqrt(var / nsim))
    np.testing.assert_allclose(t.var, var, rtol=4 * np.sqrt(2 / nsim))
    np.testing.assert_allclose(t.pearson, (t.observed - t.fitted) / np.sqrt(t.var))
    assert np.all(t.var_obs == 0)
    # By default, the model's statistics, each network's share.
    default = ergmx.gofN(fit, nsim=200, seed=1)
    assert default.names == ["N(1)~edges", "N(1)~nodematch.group"]
    np.testing.assert_array_equal(default["N(1)~edges"].observed, t.observed)


def test_missing_dyads_are_imputed():
    """The observed statistics are the mean over networks drawn conditional
    on the observed dyads, and var_obs their variance."""
    nets = [load("samplk1"), load("samplk3.nonresponse")]
    fit = ergmx.ergm(ergmx.Networks(nets), "N(~edges + nodematch('group'))")
    nsim = 2000
    t = ergmx.gofN(fit, "edges", nsim=nsim, seed=2)["edges"]
    g = nets[1]
    group = np.array(g.vs["group"])
    missing = [e.tuple for e in g.es if e["na"]]
    p = np.array([expit(fit.params[0] + fit.params[1] * (group[a] == group[b])) for a, b in missing])
    ties = sum(not e["na"] for e in g.es)
    assert t.var_obs[0] == 0 and t.observed[0] == nets[0].ecount()
    assert abs(t.observed[1] - (ties + p.sum())) < 4 * np.sqrt((p * (1 - p)).sum() / nsim)
    np.testing.assert_allclose(t.var_obs[1], (p * (1 - p)).sum(), rtol=4 * np.sqrt(2 / nsim))
    np.testing.assert_allclose(t.pearson, (t.observed - t.fitted) / np.sqrt(t.var - t.var_obs))


def test_subsets_summaries_and_series():
    nets = monks()
    for k, g in enumerate(nets):
        g["wave"] = k + 1
    fit = ergmx.ergm(ergmx.Networks(nets), "N(~edges + nodematch('group'))")
    some = ergmx.gofN(fit, "edges + mutual", subset="~wave >= 2", nsim=50, seed=1)
    np.testing.assert_array_equal(some.networks, [2, 3])
    assert some["mutual"].labels == ["2", "3"] and len(some["mutual"].pearson) == 2
    assert np.array_equal(ergmx.gofN(fit, "edges", subset=[1, 3], nsim=20, seed=1).networks, [1, 3])
    s = some.summary()
    assert "Pearson residuals" in str(s) and "Variance and std. dev." in str(s)
    assert list(some.summary(by="~wave == 3").groups) == ["wave == 3 = FALSE", "wave == 3 = TRUE"]
    # A series: one row per transition.
    series = ergmx.tergm(nets, "Form(~edges) + Persist(~edges)", estimate="CMPLE")
    by_transition = ergmx.gofN(series, nsim=50, seed=1)
    assert by_transition.names == ["Form(1)~edges", "Persist(1)~edges"]
    assert len(by_transition["Form(1)~edges"].observed) == 2
    with pytest.raises(ValueError, match="several networks"):
        ergmx.gofN(ergmx.ergm(nets[0], "edges"))


def test_frames_and_plots():
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fit = ergmx.ergm(ergmx.Networks(monks()), "N(~edges + nodematch('group'))")
    result = ergmx.gofN(fit, "edges + mutual", nsim=50, seed=1)
    frame = result.to_frame()
    assert list(frame.columns) == ["network", "statistic", "observed", "fitted", "var", "var_obs", "pearson"]
    assert len(frame) == 6
    fig = result.plot(which=(1, 2, 3))
    assert len(fig.axes) == 6
    assert len(result.plot("edges", against="~.NetworkID").axes) == 2


def test_empty_network_statistics_differ_from_r():
    nets = monks()
    fit = ergmx.ergm(ergmx.Networks(nets), "N(~edges)")
    with pytest.warns(ErgmDifferenceWarning, match="isolates"):
        ergmx.gofN(fit, "edges + isolates", nsim=10, seed=1)


# -- Against R ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def households():
    fit = ergmx.ergm(load(R["network"]), R["formula"])
    for name, value in R["coef"].items():
        assert fit.coef[name] == pytest.approx(value, abs=1e-6)  # the exact MLE, as R's
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        return (ergmx.gofN(fit, nsim=R["nsim"], seed=1),
                ergmx.gofN(fit, R["gof_formula"], nsim=R["nsim"], seed=2))


def _r(values) -> np.ndarray:
    return np.array([np.nan if v is None or v == "NA" else v for v in values], dtype=float)


@pytest.mark.parametrize("which", ["default", "gof"])
def test_gofn_matches_r(households, which):
    ours = households[0 if which == "default" else 1]
    assert ours.names == list(R[which])
    size = np.array(R["network_size"], dtype=float)
    for name, r in R[which].items():
        t = ours[name]
        r_var, r_fitted, r_observed = _r(r["var"]), _r(r["fitted"]), _r(r["observed"])
        # Statistics that never vary in a network have no residual; rare ones
        # may not vary in one of the samples.
        assert np.all(np.nan_to_num(t.var[np.isnan(r_var)]) < 0.05), name
        assert np.all(np.nan_to_num(r_var[np.isnan(t.var)]) < 0.05), name
        both = np.isfinite(t.var) & np.isfinite(r_var) & (r_var > 0.05)
        if name in ("degree0", "isolates"):  # ergm.multi leaves out the empty network's value
            r_observed, r_fitted = r_observed + size, r_fitted + size
        np.testing.assert_allclose(t.observed[both], r_observed[both], err_msg=name)
        both &= r_var > 0.25  # rarer counts' variances are too noisy to compare
        if both.sum() < 20:
            continue
        # R's simulated statistics are autocorrelated (see gofN's interval):
        # its fitted values are about twice as far from the exact ones as
        # independent draws would be.
        z = (t.fitted[both] - r_fitted[both]) / np.sqrt(r_var[both] / R["nsim"])
        assert np.std(z) < 3.5 and np.max(np.abs(z)) < 11, name
        ratio = np.log(t.var[both] / r_var[both])
        assert abs(np.median(ratio)) < 0.05 and np.quantile(np.abs(ratio), 0.9) < 0.3, name
        r_pearson = _r(r["pearson"])
        assert np.nanstd(t.pearson[both] - r_pearson[both]) < 0.2, name


def test_lm_gofn_matches_r():
    """ergmx's linear models of R's gofN() table are R's lm.gofN()'s."""
    from ergmx._gofn import GofNResult
    from ergmx._network import as_network

    names = list(R["gof"])
    table = {ours: np.column_stack([_r(R["gof"][n][theirs]) for n in names])
             for ours, theirs in (("observed", "observed"), ("fitted", "fitted"), ("var", "var"),
                                  ("var_obs", "var.obs"), ("pearson", "pearson"))}
    blocks = as_network(load(R["network"])).blocks
    gof = GofNResult(names, table, np.arange(1, len(blocks) + 1), [b.attributes for b in blocks], R["nsim"])
    fits = ergmx.lm_gofN(R["lm_formula"], gof)
    assert list(fits) == list(R["lm"])
    for name, r in R["lm"].items():
        fit = fits[name]
        np.testing.assert_allclose(list(fit.coef.values()), list(r["coef"].values()), rtol=1e-9)
        np.testing.assert_allclose(list(fit.stderr.values()), list(r["se"].values()), rtol=1e-9)
        assert fit.sigma == pytest.approx(r["sigma"]) and fit.df == r["df"]
        assert fit.r_squared == pytest.approx(r["r_squared"]) and fit.adj_r_squared == pytest.approx(r["adj_r_squared"])
        np.testing.assert_allclose(fit.fstatistic, r["fstatistic"])
        assert fit.dropped == r["dropped"]
    assert "Residual standard error" in str(fits["edges"])
    # Backquoted names and indices pick the statistics too.
    assert list(gof.lm("`edges` ~ n")) == ["edges"] and list(ergmx.lm_gofN("1:2 ~ n", gof)) == names[:2]
    with pytest.raises(ValueError, match="no statistics"):
        ergmx.lm_gofN("esp1 ~ n", gof)
