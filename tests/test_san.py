"""Simulated annealing (san(), as ergm's) and models fitted to target
statistics (ergm(target_stats=)). The fits are compared with R's in
test_estimation.py (the reference models *_target)."""

import igraph as ig
import numpy as np
import pytest
from conftest import load

import ergmx


def test_san_reaches_reachable_targets():
    g = ig.Graph(n=40)
    g.vs["group"] = [k % 3 for k in range(40)]
    formula = "edges + nodematch('group') + kstar(2)"
    target = [80, 40, 340]
    h = ergmx.san(g, formula, target, seed=1)
    assert list(ergmx.summary_stats(h, formula).values()) == target
    # By name, and every run's network.
    runs = ergmx.san(g, formula, dict(zip(["edges", "nodematch.group", "kstar2"], target)), seed=2,
                     only_last=False, maxit=3)
    assert 1 <= len(runs) <= 3 and list(ergmx.summary_stats(runs[-1], formula).values()) == target


def test_san_keeps_the_constraints_and_infinite_offsets():
    g = load("faux.mesa.high")
    empty = g.copy()
    empty.delete_edges(empty.es)
    h = ergmx.san(empty, "edges + nodematch('Grade')", [300, 150], constraints="bd(maxout=6)", seed=1)
    assert max(h.degree()) <= 6 and h.ecount() > 250
    h = ergmx.san(empty, "edges + offset(nodematch('Sex'))", [150], offset_coef=[-np.inf], seed=1)
    assert h.ecount() == 150 and ergmx.summary_stats(h, "nodematch('Sex')")["nodematch.Sex"] == 0
    with pytest.raises(ValueError, match="offset_coef"):
        ergmx.san(g, "edges + offset(nodematch('Sex'))", [150])
    with pytest.raises(ValueError, match="2 values"):
        ergmx.san(g, "edges + nodematch('Grade')", [150])


def test_dyad_independent_targets_get_the_exact_mle():
    """The MLE for target statistics solves: expected statistics = targets,
    whether or not the annealed network reaches them."""
    g = load("flomarriage")
    fit = ergmx.ergm(g, "edges + nodecov('wealth')", target_stats=[25, 3000], seed=1)
    assert fit.method == "MLE"
    from scipy.special import expit

    x, _, w = fit._model.mple_table()
    np.testing.assert_allclose((w * expit(x @ fit.params)) @ x, [25, 3000], rtol=1e-8)
    assert "target statistics" in str(fit.summary())


def test_dependent_targets_are_matched_by_the_fit():
    g = load("faux.mesa.high")
    formula = "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)"
    target = [250, 190, 200]
    fit = ergmx.ergm(g, formula, target_stats=target, seed=1, eval_loglik=False)
    assert fit.converged
    sims = fit.simulate(200, seed=2, output="stats")
    se = sims.std(axis=0) / np.sqrt(len(sims)) * 3
    assert np.all(np.abs(sims.mean(axis=0) - target) < 5 * se + 1)
