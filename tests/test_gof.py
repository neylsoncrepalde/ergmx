import json

import numpy as np
import pytest
from conftest import DATA, load

import ergmx
from ergmx import GofTable

#: R's gof() at R's MLE (scripts/r_gof_reference.R).
GOF_REFERENCE = json.loads((DATA / "r_gof_reference.json").read_text())
CASES = [(name, stat) for name, m in sorted(GOF_REFERENCE.items()) for stat in m["tables"]]


@pytest.fixture(scope="module")
def ergmx_gof():
    """ergmx's gof at R's MLE, once per model, with R's nearly independent samples."""
    results = {}
    for name, m in GOF_REFERENCE.items():
        results[name] = ergmx.gof(load(m["network"]), m["formula"], m["coef"], nsim=100, seed=1,
                                  burnin=1_000_000, interval=100_000)
    return results


@pytest.mark.parametrize(("name", "stat"), CASES)
def test_observed_distributions_match_r(ergmx_gof, name, stat):
    r = GOF_REFERENCE[name]["tables"][stat]
    np.testing.assert_allclose(ergmx_gof[name][stat].observed, r["observed"], rtol=1e-12)


@pytest.mark.parametrize(("name", "stat"), CASES)
def test_monte_carlo_pvalues_are_computed_as_in_r(name, stat):
    r = GOF_REFERENCE[name]["tables"][stat]
    table = GofTable(stat, [], np.array(r["observed"]), np.array(r["simulated"]))
    np.testing.assert_allclose(table.pvalue, r["pvalue"], atol=1e-12)


@pytest.mark.parametrize(("name", "stat"), CASES)
def test_simulated_distributions_match_r(ergmx_gof, name, stat):
    """Both simulate 100 networks from the same model: the means must agree
    within Monte Carlo error."""
    ours = ergmx_gof[name][stat].simulated
    theirs = np.array(GOF_REFERENCE[name]["tables"][stat]["simulated"])
    se = np.sqrt(ours.var(axis=0) / len(ours) + theirs.var(axis=0) / len(theirs))
    assert np.all(np.abs(ours.mean(axis=0) - theirs.mean(axis=0)) <= 5 * se + 0.5)


def test_default_statistics_follow_the_network_direction():
    assert [t.name for t in ergmx.gof(load("samplk3"), "edges", [-2.0], nsim=5, seed=1)] == [
        "idegree", "odegree", "espartners", "distance", "model"]
    fit = ergmx.ergm(load("flomarriage"), "edges")
    result = fit.gof(nsim=5, seed=1)
    assert [t.name for t in result] == ["degree", "espartners", "distance", "model"]
    assert "Goodness-of-fit for degree" in str(result)


def test_gof_errors():
    g = load("samplk3")
    with pytest.raises(TypeError, match="formula and the coefficients"):
        ergmx.gof(g)
    with pytest.raises(ValueError, match="does not apply to directed"):
        ergmx.gof(g, "edges", [-2.0], stats=["degree"])
    with pytest.raises(ValueError, match="unknown goodness-of-fit statistic"):
        ergmx.gof(g, "edges", [-2.0], stats=["triadcensus"])


def test_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fit = ergmx.ergm(load("samplk3"), "edges + mutual", seed=1)
    figure = fit.gof(nsim=20, seed=1).plot()
    assert len(figure.axes) == 5


def test_distributions_of_empty_networks():
    from ergmx._gof import _distribution

    empty = np.zeros((0, 2), dtype=np.uint32)
    for directed in (False, True):
        assert _distribution(4, directed, empty, "espartners").tolist() == [0, 0, 0]
        assert _distribution(4, directed, empty, "distance").tolist() == [0, 0, 0, 12 if directed else 6]
