import numpy as np
import pytest
from conftest import load

import ergmx
from ergmx import McmcDiagnostics


def ar1(chains, n, rho, seed=0, shift=None):
    """AR(1) chains with unit innovations: autocorrelation time (1 + rho) / (1 - rho)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((chains, n, 1))
    for t in range(1, n):
        x[:, t, 0] = rho * x[:, t - 1, 0] + rng.standard_normal(chains)
    if shift is not None:
        x[:, :, 0] += np.asarray(shift)[:, None]
    return x


def test_effective_size_of_an_ar1_process():
    rho = 0.8
    d = McmcDiagnostics(["x"], ar1(4, 20_000, rho), [0.0])
    tau = (1 + rho) / (1 - rho)
    assert d.autocorrelation_time[0] == pytest.approx(tau, rel=0.1)
    assert d.effective_size[0] == pytest.approx(80_000 / tau, rel=0.1)
    assert d.timeseries_se[0] == pytest.approx(d.naive_se[0] * np.sqrt(tau), rel=0.1)
    assert d.rhat[0] == pytest.approx(1.0, abs=0.01)
    assert np.all(np.abs(d.geweke) < 4)


def test_rhat_flags_chains_that_disagree():
    d = McmcDiagnostics(["x"], ar1(4, 2000, 0.5, shift=[0, 0, 3, 3]), [0.0])
    assert d.rhat[0] > 1.1
    assert "chains disagree" in str(d)


def test_diagnostics_of_a_fit():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fit = ergmx.ergm(load("samplk3"), "edges + mutual", seed=1)
    d = fit.mcmc_diagnostics()
    text = str(d)
    for section in ("Time-series SE", "R-hat", "Hotelling", "Geweke"):
        assert section in text
    assert d.n_chains == fit.control.n_chains
    assert 0 <= d.pvalue <= 1
    # The fit reproduces the observed statistics: mean deviations are small.
    assert np.all(np.abs(d.mean / d.sd) < 0.25)
    assert len(d.plot().axes) == 2 * 2


def test_no_diagnostics_without_mcmc():
    fit = ergmx.ergm(load("flomarriage"), "edges + nodecov('wealth')")
    with pytest.raises(ValueError, match="without MCMC"):
        fit.mcmc_diagnostics()
