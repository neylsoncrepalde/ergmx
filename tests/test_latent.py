"""Latent space models (ergmm), against R's latentnet: the likelihood and
priors exactly, the starting values, and the posteriors of 13 models within
the Monte Carlo error of R's (three seeds)."""

import json
import warnings

import numpy as np
import pytest
from conftest import DATA

import ergmx
from ergmx._latent import bind_latent, find_mpe, initial_values, label_switch, procrustes_rotation
from ergmx.datasets import load

REFERENCE = DATA / "r_latentnet_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None
pytestmark = pytest.mark.skipif(R is None, reason="no R reference")
NAMES = list(R["fits"]) if R else []


def _options(r):
    out = {k: r[k] for k in ("response", "family") if k in r}
    if r.get("family") == "binomial":
        out["fam_par"] = {"trials": r["trials"]}
    if r.get("family") == "normal":
        out["fam_par"] = {"prior_var": 1, "prior_var_df": 2}
    if r["network"] == "davis":
        out["bipartite"] = True
    return out


def _config(values: dict) -> dict:
    out = {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in values.items()}
    if "beta" in out:
        out["beta"] = np.atleast_1d(out["beta"])
    return out


def _seeds(r, key):
    """R's three seeds' values: their mean and spread."""
    values = np.array([np.atleast_1d(s[key]) for s in r["seeds"]], dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


@pytest.mark.parametrize("name", NAMES)
def test_likelihood_and_priors_match_latentnet(name):
    """A posterior draw's log-likelihood and log-priors, and the model's
    coefficients and default priors, as latentnet's."""
    r = R["fits"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bind_latent(load(r["network"]), r["formula"], **_options(r))
    assert model.coef_names == list(np.atleast_1d(r["coef_names"]))
    for key, value in r["prior"].items():
        if key in model.prior:
            np.testing.assert_allclose(np.atleast_1d(model.prior[key]), np.atleast_1d(value), rtol=1e-12, err_msg=key)
    lp = model.lp(_config(r["draw"]))
    for key, value in r["draw_lp"].items():
        assert lp[key] == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("name", ["sampson_e2", "sampson_rreceiver", "sampson_nodematch", "tribes_normal",
                                  "sampson_poisson", "tribes_sociality"])
def test_starting_values_match_latentnet(name):
    """The MCMC starts, as R's, at a mode of the conditional posterior, with a
    log posterior close to that of R's start. The conditional posterior has
    several nearby modes, and which one the optimizer reaches depends on
    floating-point rounding (the platform, SciPy's version), in R too: on
    macOS they are R's to 0.006, elsewhere up to 0.64 apart."""
    r = R["fits"][name]
    model = bind_latent(load(r["network"]), r["formula"], **_options(r))
    start = initial_values(model, np.random.default_rng(1))
    ours, theirs = sum(model.lp(start).values()), sum(model.lp(_config(r["start"])).values())
    again = find_mpe(model, start, maxit=2000)
    assert sum(model.lp(again).values()) - ours < 1e-3  # a mode
    assert ours == pytest.approx(theirs, abs=1)


@pytest.mark.parametrize("name", NAMES)
def test_posteriors_match_latentnet(name):
    """The coefficients' posterior means within 0.25 posterior standard
    deviations of R's, their standard deviations within 15%, the variances
    within 10%, the mean distances and tie probabilities and the clusters'
    co-membership within a few times R's own spread between seeds."""
    r = R["fits"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ergmx.ergmm(load(r["network"]), r["formula"], seed=1, **_options(r))
    beta, sd = fit.sample["beta"], _seeds(r, "beta_sd")[0]
    mean, spread = _seeds(r, "beta_mean")
    assert np.all(np.abs(beta.mean(axis=0) - mean) < 0.25 * sd + 3 * spread), (beta.mean(axis=0), mean)
    np.testing.assert_allclose(beta.std(axis=0), sd, rtol=0.15)
    if fit.model.G <= 1 and "Z_var_mean" in r["seeds"][0]:
        np.testing.assert_allclose(fit.sample["Z.var"].mean(axis=0), _seeds(r, "Z_var_mean")[0], rtol=0.1)
    for key in ("sender_var", "receiver_var", "sociality_var", "dispersion"):
        if key in r["seeds"][0]:
            assert np.mean(fit.sample[key.replace("_", ".")]) == pytest.approx(_seeds(r, key)[0][0], rel=0.1), key
    observed = fit.model.observed | fit.model.observed.T
    p_mean, p_spread = (x.reshape(fit.model.n, fit.model.n) for x in _seeds(r, "tie_probability"))
    assert np.max(np.abs(fit.predict("post") - p_mean)[observed]) < 0.02 + 3 * np.max(p_spread[observed])
    if fit.model.d:
        z = fit.sample["Z"]
        distance = np.mean([np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)) for x in z], axis=0)
        d_mean, d_spread = (x.reshape(distance.shape) for x in _seeds(r, "distance_mean"))
        assert np.max(np.abs(distance - d_mean)) < 0.03 * d_mean.max() + 3 * np.max(d_spread)
    if "co_cluster" in r["seeds"][0]:
        same = np.mean([np.equal.outer(k, k) for k in fit.sample["Z.K"]], axis=0)
        c_mean, c_spread = (x.reshape(same.shape) for x in _seeds(r, "co_cluster"))
        assert np.mean(np.abs(same - c_mean)) < 0.02 + 3 * np.mean(c_spread)
    # The MKL coefficients; the BIC's likelihood part, which R's MKL makes noisy too.
    mkl, mkl_spread = _seeds(r, "mkl_beta")
    shift = 0.35 if name == "tribes_sociality" else 0.0  # ergmx moves twice the sociality's mean into the intercept
    assert np.all(np.abs(np.atleast_1d(fit.mkl["beta"]) - mkl) < (0.15 + shift) * sd + 3 * mkl_spread)
    bic_y = np.array([s["bic"]["Y"] for s in r["seeds"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # with 16 vertices in 3 clusters, the EM may leave clusters empty, as R's
        assert abs(fit.bic()["Y"] - bic_y.mean()) < 3 + 3 * bic_y.std(ddof=1)


def test_fit_tools():
    g = load("samplike")
    fit = ergmx.ergmm(g, "euclidean(d=2, G=3) + rreceiver", seed=2, sample_size=800, burnin=4000,
                      tofit=("mcmc", "mkl", "mkl.mbc", "procrustes", "klswitch", "pmode", "mle"))
    assert fit.pmode is not None and fit.mle is not None and fit.mkl["mbc"]["Z.K"].shape == (18,)
    text = str(fit.summary())
    assert "Covariate coefficients posterior means:" in text and "Overall BIC:" in text and "MLE:" in text
    pm = fit.pmean
    np.testing.assert_allclose(pm["Z.pZK"].sum(axis=1), 1)
    for kind in ("post", "mkl", "pmean", "mle", "pmode", "start", 0):
        p = fit.predict(kind)
        assert p.shape == (18, 18) and np.all((p >= 0) & (p <= 1))
    sims = fit.simulate(3, seed=1)
    assert len(sims) == 3 and all(s.is_directed() and s.vcount() == 18 for s in sims)
    gof = fit.gof(20, seed=1)
    assert [t.name for t in gof] == ["idegree", "odegree", "espartners", "distance"]
    diag = fit.mcmc_diagnostics()
    assert diag.n_chains == 4 and "R-hat" in str(diag)


def test_undirected_and_valued_simulations_follow_the_model():
    """An undirected network's simulated ties are symmetric, with the
    predicted probabilities (latentnet's draw two per pair); valued ones,
    with the family's means."""
    tribes = load("tribes")
    fit = ergmx.ergmm(tribes, "euclidean(d=2)", response="pos", seed=1, sample_size=400, burnin=2000)
    y = np.array(fit.simulate(400, seed=2, output="adjacency"))
    assert np.allclose(y, np.transpose(y, (0, 2, 1)))
    observed = fit.model.observed
    assert abs(y.mean(axis=0)[observed].sum() - fit.predict("post")[observed].sum()) < 4 * np.sqrt(observed.sum())
    poisson = ergmx.ergmm(load("samplike"), "euclidean(d=2)", response="nominations", family="Poisson", seed=1,
                          sample_size=400, burnin=2000)
    graphs = poisson.simulate(2, seed=1)
    assert "nominations" in graphs[0].es.attributes()


def test_label_switching_and_rotations():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(30, 2)) + np.repeat([[0, 0], [6, 0], [0, 6]], 10, axis=0)
    truth = np.repeat([1, 2, 3], 10)
    sample = {"Z": np.broadcast_to(z, (40, 30, 2)), "Z.mean": [], "Z.var": [], "Z.pK": [], "Z.K": []}
    means = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
    for s in range(40):
        perm = rng.permutation(3)  # the draws' labels permuted at random
        sample["Z.mean"].append(means[perm])
        sample["Z.var"].append(np.ones(3))
        sample["Z.pK"].append(np.full(3, 1 / 3))
        sample["Z.K"].append(np.argsort(perm)[truth - 1] + 1)
    sample = {k: np.array(v) if k != "Z" else v for k, v in sample.items()}
    switched, q = label_switch(sample, truth)
    assert np.all(switched["Z.K"] == truth)
    np.testing.assert_allclose(switched["Z.mean"], np.broadcast_to(means, (40, 3, 2)))
    assert np.all(np.argmax(q, axis=1) + 1 == truth)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(z @ rotation @ procrustes_rotation(z @ rotation, z), z, atol=1e-10)


def test_errors():
    g = load("samplike")
    with pytest.raises(ValueError, match="one latent space term"):
        ergmx.ergmm(g, "euclidean(d=2) + bilinear(d=2)")
    with pytest.raises(ValueError, match="rsociality"):
        ergmx.ergmm(load("tribes"), "euclidean(d=2) + rsender", response="pos")
    with pytest.raises(ValueError, match="family"):
        ergmx.ergmm(g, "euclidean(d=2)", family="gamma")
    with pytest.raises(ValueError, match="prior_var"):
        ergmx.ergmm(load("tribes"), "euclidean(d=2)", response="sign", family="normal")
    with pytest.raises(ValueError, match="trials"):
        ergmx.ergmm(load("tribes"), "euclidean(d=2)", response="sign.012", family="binomial")
    with pytest.raises(ValueError, match="multiple of n_chains"):
        ergmx.ergmm(g, "euclidean(d=2)", sample_size=1001)
    with pytest.raises(ValueError, match="tofit"):
        ergmx.ergmm(g, "euclidean(d=2)", tofit=("mcmc", "bic"))
