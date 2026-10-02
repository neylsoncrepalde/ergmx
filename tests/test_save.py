"""Fits saved and reloaded (fit.save(), ergmx.load_fit()) work as before."""

import pickle

import numpy as np
import pytest
from conftest import load

import ergmx

FITS = {
    "curved": lambda: ergmx.ergm(load("faux.mesa.high"), "edges + nodematch('Grade') + gwesp(0.5)", seed=1),
    "constrained, missing dyads": lambda: ergmx.ergm(load("samplk3.nonresponse"), "edges + mutual",
                                                     constraints="bd(maxout=6)", seed=1),
    "networks": lambda: ergmx.ergm(ergmx.Networks([load(f"samplk{k}") for k in (1, 2, 3)]),
                                   "N(~edges + mutual, offset=~.NetworkID / 10)", seed=1),
    "series": lambda: ergmx.tergm([load(f"samplk{k}") for k in (1, 2, 3)],
                                  "Form(~edges + mutual) + Persist(~edges)", seed=1),
    "exact": lambda: ergmx.ergm(load("flomarriage"), "edges + nodecov('wealth')"),
}


@pytest.mark.parametrize("kind", FITS)
def test_reloaded_fits_work_as_before(kind, tmp_path):
    fit = FITS[kind]()
    path = tmp_path / "fit.pkl"
    fit.save(path)
    again = ergmx.load_fit(path)
    assert again.coef == fit.coef and str(again.summary()) == str(fit.summary())
    np.testing.assert_array_equal(again.simulate(3, seed=2, output="stats"), fit.simulate(3, seed=2, output="stats"))
    np.testing.assert_allclose(again.predict().p, fit.predict().p)
    assert again.gof(nsim=5, seed=1, stats=["model"])["model"].observed.tolist() == \
        fit.gof(nsim=5, seed=1, stats=["model"])["model"].observed.tolist()
    if kind == "series":
        assert len(again.simulate(time_slices=2, seed=1)) == 2


def test_egmme_fits_and_pickle(tmp_path):
    import igraph as ig

    fit = ergmx.tergm(ig.Graph(n=20), "Form(~edges) + Persist(~edges)", estimate="EGMME",
                      targets="edges + mean.age", target_stats=[10, 5], seed=1)
    fit.save(tmp_path / "egmme.pkl")
    again = ergmx.load_fit(tmp_path / "egmme.pkl")
    assert again.coef == fit.coef and again.summary() == fit.summary()
    assert len(again.simulate(3, seed=1)) == 3
    # Plain pickle works too.
    assert pickle.loads(pickle.dumps(fit)).coef == fit.coef


def test_loading_checks_the_file(tmp_path):
    fit = FITS["exact"]()
    with open(tmp_path / "old.pkl", "wb") as f:
        pickle.dump({"ergmx": "0.0.1", "fit": fit}, f)
    with pytest.warns(UserWarning, match="saved with ergmx 0.0.1"):
        assert ergmx.load_fit(tmp_path / "old.pkl").coef == fit.coef
    with open(tmp_path / "other.pkl", "wb") as f:
        pickle.dump([1, 2], f)
    with pytest.raises(ValueError, match="not a fit"):
        ergmx.load_fit(tmp_path / "other.pkl")
