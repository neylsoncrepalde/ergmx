"""Valued networks (ergm.count): statistics against R's and the definitions,
the sampler against exact enumeration and the statistics it tracks, and
fits against exact MLEs and R's (scripts/r_valued_reference.R)."""

import json
import math
import warnings

import igraph as ig
import numpy as np
import pytest
from conftest import DATA, load

import ergmx
from ergmx import ErgmDifferenceWarning

REFERENCE = DATA / "r_valued_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


def zach():
    return ergmx.datasets.load("zach")


def nominations():
    """Sampson's monks' liking, summed over the three times: counts of 0 to 3."""
    gs = [load(f"samplk{k}") for k in (1, 2, 3)]
    y = sum(np.array(g.get_adjacency().data) for g in gs)
    i, j = np.nonzero(y)
    h = ig.Graph(n=gs[0].vcount(), edges=list(zip(i.tolist(), j.tolist())), directed=True)
    h.es["nominations"] = y[i, j].astype(float).tolist()
    for a in gs[0].vs.attributes():
        h.vs[a] = gs[0].vs[a]
    return h


def continuous(name):
    """Continuous values of R's reference script (normal, reciprocal or proportions)."""
    y = np.array(R["continuous"][name])
    directed = name == "reciprocal"
    n = len(y)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and (directed or i < j) and y[i, j] != 0]
    g = ig.Graph(n=n, edges=pairs, directed=directed)
    g.es["value"] = [float(y[i, j]) for i, j in pairs]
    if n == len(R["continuous"]["group"]):
        g.vs["group"] = R["continuous"]["group"]
    return g


def empty():
    """The karate club without its interactions (to fit to target statistics)."""
    g = zach()
    g.delete_edges(g.es)
    return g


NETWORKS = {"zach": (zach, "contexts", "Poisson"), "nominations": (nominations, "nominations", "Binomial(3)"),
            "empty": (empty, "contexts", "Poisson"),
            **{name: (lambda name=name: continuous(name), "value", None)
               for name in ("normal", "reciprocal", "proportions")}}

UNDIRECTED = ("sum + nonzero + nodematch('role') + nodematch('role', form='nonzero') + nodefactor('role') + "
              "nodecov('faction.id') + absdiff('faction.id') + atleast(2) + atmost(c(1,3)) + greaterthan(4) + "
              "smallerthan(2) + equalto(3) + ininterval(1, 3) + ininterval(1,3, open=c(FALSE, TRUE)) + "
              "sum(pow=0.5) + CMP + transitiveweights('min','max','min') + "
              "transitiveweights('geomean','sum','geomean') + cyclicalweights + nodecovar + "
              "nodecovar(center=TRUE, transform='sqrt') + mm('role')")
DIRECTED = ("sum + nonzero + mutual + mutual(form='nabsdiff') + mutual(form='product') + mutual(form='geometric') + "
            "transitiveweights + cyclicalweights + transitiveweights('geomean','sum','geomean') + "
            "cyclicalweights('geomean','sum','min') + nodeocovar + nodeicovar + nodeocovar(center=TRUE) + "
            "nodeicovar(transform='sqrt') + transitiveties + transitiveties(threshold=1) + "
            "nodematch('group', form='sum') + sender(form='nonzero')")


def test_valued_statistics_match_the_definitions():
    g = zach()
    y = np.zeros((34, 34))
    for e in g.es:
        y[e.source, e.target] = y[e.target, e.source] = e["contexts"]
    stats = ergmx.summary_stats(g, "sum + nonzero + CMP + atleast(3) + sum(pow=2)", response="contexts")
    upper = y[np.triu_indices(34, 1)]
    assert stats["sum"] == upper.sum() and stats["nonzero"] == np.count_nonzero(upper)
    assert stats["CMP"] == pytest.approx(sum(math.lgamma(v + 1) for v in upper))
    assert stats["atleast.3"] == np.sum(upper >= 3) and stats["sum2"] == pytest.approx(np.sum(upper**2))
    # transitiveweights(min, max, min): each tie's value, capped by its strongest two-path.
    expected = 0.0
    for t, h in zip(*np.triu_indices(34, 1)):
        if y[t, h]:
            paths = [min(y[t, k], y[k, h]) for k in range(34) if k not in (t, h)]
            expected += min(max(paths), y[t, h])
    assert ergmx.summary_stats(g, "transitiveweights", response="contexts")["transitiveweights.min.max.min"] == expected


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("network", ["zach", "nominations"])
def test_valued_statistics_match_r(network):
    make, response, _ = NETWORKS[network]
    formula = UNDIRECTED if network == "zach" else DIRECTED
    stats = ergmx.summary_stats(make(), formula, response=response)
    r = R["stats"][network]
    # R repeats names that ergmx makes unique (nodecovar, nodecovar.1).
    from ergmx._model import _make_unique

    assert list(stats) == _make_unique(r["names"])
    np.testing.assert_allclose(list(stats.values()), r["values"], rtol=1e-10)


@pytest.mark.parametrize("network", ["zach", "nominations"])
def test_tracked_valued_statistics_are_exact(network):
    """Recomputing the statistics of sampled valued networks gives those the
    sampler tracked, change by change."""
    make, response, reference = NETWORKS[network]
    formula = (UNDIRECTED if network == "zach" else DIRECTED)
    g = make()
    names = list(ergmx.summary_stats(g, formula, response=response))
    coef = [0.0] * len(names)
    tracked = ergmx.simulate(g, formula, coef, 10, seed=3, response=response, reference=reference,
                             interval=500, output="stats")
    sims = ergmx.simulate(g, formula, coef, 10, seed=3, response=response, reference=reference, interval=500)
    recomputed = [list(ergmx.summary_stats(h, formula, response=response).values()) for h in sims]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9, atol=1e-9)
    assert len({tuple(h.es[response]) for h in sims}) > 1


@pytest.mark.parametrize(("reference", "support"), [("Poisson", range(0, 25)), ("Binomial(2)", range(3)),
                                                    ("Geometric", range(0, 60)), ("DiscUnif(0, 3)", range(4))])
def test_valued_mcmc_matches_exact_enumeration(reference, support):
    """On 3 vertices (3 dyads), the expected statistics of a dependent model,
    by enumerating the values (truncated where the reference's tail is
    negligible), against the sampler's."""
    g = ig.Graph(n=3)
    g.add_edges([(0, 1)])
    g.es["w"] = [1.0]
    formula = "sum + nonzero + transitiveweights + CMP"
    theta = np.array([-0.4, 0.3, 0.2, 0.1]) if reference != "Geometric" else np.array([-0.7, 0.3, 0.2, 0.0])
    name = reference.split("(")[0]

    def log_h(y):
        if name == "Poisson":
            return -math.lgamma(y + 1)
        if name == "Binomial":
            return math.lgamma(3) - math.lgamma(y + 1) - math.lgamma(3 - y)
        return 0.0

    grid = np.array(np.meshgrid(*[np.array(list(support), dtype=float)] * 3, indexing="ij")).reshape(3, -1)
    a, b, c = grid  # the dyads 0-1, 0-2 and 1-2

    def capped(y, p, q):  # a tie's value, capped by its two-path's (the min of its links)
        return np.where(y > 0, np.minimum(np.minimum(p, q), y), 0.0)

    lgamma = np.vectorize(math.lgamma)
    s = np.stack([a + b + c, (a > 0) * 1.0 + (b > 0) + (c > 0),
                  capped(a, b, c) + capped(b, a, c) + capped(c, a, b),
                  lgamma(a + 1) + lgamma(b + 1) + lgamma(c + 1)])
    log_h = np.vectorize(log_h)
    logw = theta @ s + log_h(a) + log_h(b) + log_h(c)
    w = np.exp(logw - logw.max())
    exact = (s * w).sum(axis=1) / w.sum()
    sample = ergmx.simulate(g, formula, theta, 20000, seed=1, response="w", reference=reference, interval=20,
                            output="stats")
    se = sample.std(axis=0) / np.sqrt(len(sample)) * 3  # autocorrelation allowance
    assert np.all(np.abs(sample.mean(axis=0) - exact) < 4 * se + 1e-9), (sample.mean(axis=0), exact)


def test_poisson_and_binomial_sums_get_their_exact_mles():
    g = zach()
    fit = ergmx.ergm(g, "sum", response="contexts", reference="Poisson", seed=1)
    d = 34 * 33 / 2
    total = sum(g.es["contexts"])
    assert fit.coef["sum"] == pytest.approx(math.log(total / d), abs=0.03)
    assert fit.stderr["sum"] == pytest.approx(1 / math.sqrt(total), rel=0.1)
    exact = math.log(total / d) * total - total - sum(math.lgamma(v + 1) for v in g.es["contexts"])
    assert abs(fit.loglik - exact) < 4 * fit.loglik_se + 0.5
    # exp(coef) of a valued model is no odds ratio of a tie.
    ratios = fit.odds_ratios()
    assert "exp(coef)" in ratios.columns and "a dyad's value relative to the value one lower" in str(ratios)
    assert ratios.columns["exp(coef)"][0] == pytest.approx(total / d, rel=0.03)
    h = nominations()
    fit = ergmx.ergm(h, "sum", response="nominations", reference="Binomial(3)", seed=1)
    p = sum(h.es["nominations"]) / (3 * 18 * 17)
    assert fit.coef["sum"] == pytest.approx(math.log(p / (1 - p)), abs=0.03)


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["fits"]) if R else [])
def test_valued_fits_match_r(name):
    r = R["fits"][name]
    make, response, _ = NETWORKS[r["network"]]
    target = None
    if r.get("target"):  # fitted to another network's statistics
        other, other_response, _ = NETWORKS[r["target"]]
        target = ergmx.summary_stats(other(), r["formula"], response=other_response)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        fit = ergmx.ergm(make(), r["formula"], response=response, reference=r["reference"], seed=1,
                         eval_loglik=False, target_stats=target)
    assert fit.converged
    # R's random-walk proposals of continuous values leave more Monte Carlo
    # error (0.2 standard errors from the exact MLE, for reciprocal_product);
    # test_continuous_fits_get_their_exact_mles checks those against the exact MLEs.
    tolerance = 0.5 if r["reference"].split("(")[0] in ("StdNormal", "Unif") else 0.3
    for term, value in r["coef"].items():
        assert abs(fit.coef[term] - value) < tolerance * r["se"][term], term
        assert fit.stderr[term] == pytest.approx(r["se"][term], rel=0.25), term


def test_valued_errors():
    g = zach()
    with pytest.raises(ValueError, match="reference measure"):
        ergmx.ergm(g, "sum", response="contexts")
    with pytest.raises(ValueError, match="no edge attribute"):
        ergmx.summary_stats(g, "sum", response="weight")
    with pytest.raises(ergmx.FormulaError, match="unknown valued term"):
        ergmx.summary_stats(g, "sum + triangle", response="contexts")
    with pytest.warns(ErgmDifferenceWarning, match="transitiveties"):
        ergmx.summary_stats(g, "transitiveties", response="contexts")
    with pytest.raises(ValueError, match="no MPLE"):
        ergmx.ergm(g, "sum", response="contexts", reference="Poisson", estimate="MPLE")
    with pytest.raises(ValueError, match="unknown reference"):
        ergmx.ergm(g, "sum", response="contexts", reference="Negbin")


@pytest.mark.skipif(R is None or "gof" not in R, reason="no R reference")
def test_valued_gof_matches_r():
    """gof() of a valued model: ergm's cdf points, the observed counts and,
    from nearly independent networks, the simulated means."""
    from ergmx._gof import cdf_points

    r = R["gof"]
    fractional = zach()
    fractional.es["scaled"] = [0.35 * v for v in fractional.es["contexts"]]
    for name, (g, response, nmax) in {"zach": (zach(), "contexts", 100),
                                      "nominations": (nominations(), "nominations", 100),
                                      "fractional": (fractional, "scaled", 100),
                                      "fractional_nmax": (fractional, "scaled", 5)}.items():
        points = cdf_points(g.es[response], nmax=nmax)
        assert [f".gof.cdf#{x:.15g}" for x in points] == r["cdf_points"][name], name
    nsim = 200
    result = ergmx.gof(zach(), r["formula"], r["coef"], response="contexts", reference="Poisson", nsim=nsim,
                       interval=20000, seed=1)
    assert list(result.tables) == ["model", "cdf"]
    for name in ("cdf", "model"):
        table, ref = result[name], r[name]
        assert table.labels == ref["names"]
        np.testing.assert_array_equal(table.observed, ref["obs"])
        se = np.sqrt(np.square(ref["sd"]) / r["nsim"] + table.simulated.var(axis=0) / nsim)
        assert np.all(np.abs(table.mean - ref["mean"]) < 4 * np.maximum(se, 1e-9)), name


def test_valued_gof_of_a_fit():
    fit = ergmx.ergm(zach(), "sum + nonzero + nodematch('faction.id')", response="contexts",
                     reference="Poisson", seed=1, eval_loglik=False)
    result = fit.gof(nsim=50, seed=1, stats=["cdf", "degree", "espartners", "distance", "model"])
    assert list(result.tables) == ["cdf", "degree", "espartners", "distance", "model"]
    degrees = np.bincount(zach().degree(), minlength=34)
    np.testing.assert_array_equal(result["degree"].observed, degrees)  # of the nonzero dyads as ties
    assert result["cdf"].observed[-1] == 34 * 33 / 2
    assert np.all(np.diff(result["cdf"].simulated, axis=1) >= 0)
    assert "dyads with value at most x" in str(result)
    with pytest.raises(ValueError, match="valued network"):
        ergmx.ergm(load("flomarriage"), "edges").gof(nsim=5, stats=["cdf"])
    with pytest.raises(ValueError, match="by="):
        fit.gof(nsim=5, by="faction.id")
    with pytest.raises(ValueError, match="reference measure"):
        ergmx.gof(zach(), "sum", [0.5], response="contexts")


@pytest.mark.skipif(R is None or "continuous" not in R, reason="no R reference")
def test_continuous_fits_get_their_exact_mles():
    """Models of continuous values whose MLEs are known: normal values (a
    regression on the groups; a log-likelihood too), reciprocal normal
    values (bivariate normal pairs) and proportions (an exponential tilt of
    the uniform, by group)."""
    from scipy.optimize import brentq

    g = continuous("normal")
    fit = ergmx.ergm(g, "sum + sum(pow=2) + nodematch('group', form='sum')", response="value",
                     reference="StdNormal", seed=1)
    y = np.array(R["continuous"]["normal"])[np.triu_indices(20, 1)]
    group = np.array(R["continuous"]["group"])
    same = (group[:, None] == group[None, :])[np.triu_indices(20, 1)]
    mean0, mean1 = y[~same].mean(), y[same].mean()
    var = np.mean(np.where(same, y - mean1, y - mean0) ** 2)
    exact = {"sum": mean0 / var, "sum2": (1 - 1 / var) / 2, "nodematch.sum.group": (mean1 - mean0) / var}
    for name, value in exact.items():
        assert abs(fit.coef[name] - value) < 0.25 * fit.stderr[name], name
    loglik = -len(y) / 2 * (np.log(2 * np.pi * var) + 1)
    assert abs(fit.loglik - loglik) < 4 * fit.loglik_se + 0.5

    # Reciprocal normal values: each pair is bivariate normal, with precision
    # [[1 - 2 sum2, -mutual.product], [-mutual.product, 1 - 2 sum2]] and both
    # means sum / (1 - 2 sum2 - mutual.product).
    from scipy.optimize import minimize

    g = continuous("reciprocal")
    fit = ergmx.ergm(g, "sum + sum(pow=2) + mutual(form='product')", response="value", reference="StdNormal",
                     seed=1, eval_loglik=False)
    y = np.array(R["continuous"]["reciprocal"])
    pairs = np.column_stack([y[np.triu_indices(len(y), 1)], y.T[np.triu_indices(len(y), 1)]])

    def minus_loglik(t):
        a, c = 1 - 2 * t[1], t[2]
        if a <= abs(c):
            return np.inf
        precision = np.array([[a, -c], [-c, a]])
        d = pairs - t[0] / (a - c)
        return -(len(d) * np.log(np.linalg.det(precision)) / 2
                 - np.einsum("ni,ij,nj->", d, precision, d) / 2)

    exact = minimize(minus_loglik, [0.0, -0.1, 0.4], method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000}).x
    for name, value in zip(["sum", "sum2", "mutual.product"], exact):
        assert abs(fit.coef[name] - value) < 0.25 * fit.stderr[name], name

    g = continuous("proportions")
    fit = ergmx.ergm(g, "sum + nodematch('group', form='sum')", response="value", reference="Unif(0, 1)",
                     seed=1)
    u = np.array(R["continuous"]["proportions"])[np.triu_indices(20, 1)]

    def tilt(mean):  # the coefficient whose tilted uniform has this mean
        return brentq(lambda e: np.exp(e) / np.expm1(e) - 1 / e - mean, -50, 50)

    eta0, eta1 = tilt(u[~same].mean()), tilt(u[same].mean())
    assert abs(fit.coef["sum"] - eta0) < 0.25 * fit.stderr["sum"]
    assert abs(fit.coef["nodematch.sum.group"] - (eta1 - eta0)) < 0.25 * fit.stderr["nodematch.sum.group"]


def test_continuous_mcmc_matches_the_exact_moments():
    """On 3 vertices, directed (3 pairs): normal values with reciprocity are
    bivariate normal within each pair; uniform ones, tilted by their
    minimum, by numerical integration."""
    g = ig.Graph(n=3, directed=True)
    theta = np.array([0.3, -0.4, 0.5])  # sum, sum2, mutual.product
    sample = ergmx.simulate(g, "sum + sum(pow=2) + mutual(form='product')", theta, 20000, seed=1, response="w",
                            reference="StdNormal", interval=50, output="stats")
    a, c = 1 - 2 * theta[1], theta[2]  # the pair's precision matrix is [[a, -c], [-c, a]]
    mean = theta[0] / (a - c)
    exact = 3 * np.array([2 * mean, 2 * (a / (a * a - c * c) + mean**2), c / (a * a - c * c) + mean**2])
    se = sample.std(axis=0) / np.sqrt(len(sample)) * 3  # autocorrelation allowance
    assert np.all(np.abs(sample.mean(axis=0) - exact) < 4 * se), (sample.mean(axis=0), exact)

    theta = np.array([-1.0, 2.0])  # sum, mutual.min
    sample = ergmx.simulate(g, "sum + mutual", theta, 20000, seed=1, response="w", reference="Unif(0, 1)",
                            interval=20, output="stats")
    x = (np.arange(1000) + 0.5) / 1000
    y1, y2 = np.meshgrid(x, x, indexing="ij")
    stats = np.stack([y1 + y2, np.minimum(y1, y2)])
    w = np.exp(np.tensordot(theta, stats, axes=1))
    exact = 3 * (stats * w).sum(axis=(1, 2)) / w.sum()
    se = sample.std(axis=0) / np.sqrt(len(sample)) * 3
    assert np.all(np.abs(sample.mean(axis=0) - exact) < 4 * se), (sample.mean(axis=0), exact)
    assert sample.min() >= 0


def test_reference_supports():
    g = zach()
    g.es["half"] = [v / 2 for v in g.es["contexts"]]
    g.es["signed"] = [v - 3 for v in g.es["contexts"]]
    with pytest.raises(ValueError, match="Poisson reference takes whole numbers"):
        ergmx.ergm(g, "sum", response="half", reference="Poisson")
    with pytest.raises(ValueError, match="Binomial reference takes whole numbers from 0 to 3"):
        ergmx.ergm(g, "sum", response="contexts", reference="Binomial(3)")
    with pytest.raises(ValueError, match=r"Unif reference takes values from 1 to 9 \(and the dyads"):
        ergmx.ergm(g, "sum", response="contexts", reference="Unif(1, 9)")
    with pytest.raises(ValueError, match="a < b"):
        ergmx.ergm(g, "sum", response="contexts", reference="Unif(2, 1)")
    with pytest.raises(ValueError, match="CMP needs nonnegative values"):
        ergmx.ergm(g, "sum + CMP", response="signed", reference="StdNormal")
    with pytest.raises(ValueError, match="needs nonnegative values"):
        ergmx.ergm(g, "sum + nodesqrtcovar", response="signed", reference="StdNormal")
    # Statistics need no reference, and negative values are fine.
    assert ergmx.summary_stats(g, "sum", response="signed")["sum"] == sum(g.es["signed"])


def test_tracked_continuous_statistics_are_exact():
    """As for counts, with normal values: negative, and with every dyad nonzero."""
    rng = np.random.default_rng(1)
    g = ig.Graph.Full(12, directed=True)
    g.es["value"] = rng.normal(size=g.ecount()).tolist()
    formula = ("sum + sum(pow=2) + nonzero + mutual + mutual(form='nabsdiff') + mutual(form='product') + "
               "transitiveweights + cyclicalweights('min', 'sum', 'min') + nodeocovar + nodeicovar(center=TRUE) + "
               "transitiveties + atleast(0.5) + ininterval(-1, 1)")
    coef = [0.0, -0.3] + [0.0] * (len(ergmx.summary_stats(g, formula, response="value")) - 2)
    options = dict(seed=3, response="value", reference="StdNormal", interval=500)
    tracked = ergmx.simulate(g, formula, coef, 10, output="stats", **options)
    sims = ergmx.simulate(g, formula, coef, 10, **options)
    recomputed = [list(ergmx.summary_stats(h, formula, response="value").values()) for h in sims]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9, atol=1e-9)
    assert min(min(h.es["value"]) for h in sims) < 0


def _with_missing(g, pairs, response="contexts", how="na"):
    """The network with these dyads missing: na=True edges, or no value."""
    g = g.copy()
    for i, j in pairs:
        e = g.get_eid(i, j, error=False)
        if e < 0:
            g.add_edge(i, j, **({"na": True} if how == "na" else {response: float("nan")}))
        elif how == "na":
            g.es[e]["na"] = True
        else:
            g.es[e][response] = None
    return g


def test_valued_missing_dyads_get_the_exact_mle():
    """With dyads missing at random, a dyad-independent model's MLE is that
    of the observed dyads; missing dyads are na=True edges, or edges without
    a value."""
    rng = np.random.default_rng(7)
    upper = [(i, j) for i in range(34) for j in range(i + 1, 34)]
    missing = [upper[k] for k in rng.choice(len(upper), 60, replace=False)]
    values = {tuple(sorted(e.tuple)): e["contexts"] for e in zach().es}
    observed = [values.get(p, 0.0) for p in upper if p not in set(missing)]
    mean = np.mean(observed)
    exact = sum(y * math.log(mean) - mean - math.lgamma(y + 1) for y in observed)
    for how in ("na", "none"):
        fit = ergmx.ergm(_with_missing(zach(), missing, how=how), "sum", response="contexts", reference="Poisson",
                         seed=1)
        assert fit.coef["sum"] == pytest.approx(math.log(mean), abs=0.02)
        assert fit.stderr["sum"] == pytest.approx(1 / math.sqrt(sum(observed)), rel=0.1)
        assert abs(fit.loglik - exact) < 4 * fit.loglik_se + 0.5
        assert "Missing dyads: 60" in str(fit.summary())
        assert fit._model.n_observations == len(upper) - 60
    # The goodness of fit's observed values average the missing dyads' imputations:
    # some of the 60 are imputed nonzero.
    result = fit.gof(nsim=20, seed=1)
    assert result["cdf"].labels[:3] == ["0", "1", "2"]
    zeros = sum(1 for y in observed if y == 0)
    assert zeros < result["cdf"].observed[0] < zeros + 60


def test_valued_conditional_mcmc_matches_exact_enumeration():
    """On 3 vertices with two dyads missing, the conditional sampler (DiscTNT
    on the free dyads only) against the exact conditional distribution."""
    from ergmx._valued import bind_valued

    g = ig.Graph(n=3, edges=[(0, 1), (0, 2), (1, 2)])
    g.es["w"] = [2.0, None, None]  # 0-2 and 1-2 missing
    formula = "sum + nonzero + transitiveweights + CMP"
    theta = np.array([-0.4, 0.3, 0.2, 0.1])
    model = bind_valued(g, formula, "w", "Poisson")
    sample, _, _ = model.simulate([model.triples], theta, 1000, 20, 20000, 1, conditional=True)
    support = np.arange(25.0)
    b, c = (x.ravel() for x in np.meshgrid(support, support, indexing="ij"))
    a = np.full_like(b, 2.0)

    def capped(y, p, q):
        return np.where(y > 0, np.minimum(np.minimum(p, q), y), 0.0)

    lgamma = np.vectorize(math.lgamma)
    s = np.stack([a + b + c, 1.0 + (b > 0) + (c > 0), capped(a, b, c) + capped(b, a, c) + capped(c, a, b),
                  lgamma(a + 1) + lgamma(b + 1) + lgamma(c + 1)])
    logw = theta @ s - lgamma(b + 1) - lgamma(c + 1)
    w = np.exp(logw - logw.max())
    exact = (s * w).sum(axis=1) / w.sum()
    sample = sample[0]
    se = sample.std(axis=0) / np.sqrt(len(sample)) * 3
    assert np.all(np.abs(sample.mean(axis=0) - exact) < 4 * se + 1e-9), (sample.mean(axis=0), exact)


def test_valued_san_and_target_stats():
    """san() reaches a valued network's statistics, counts or continuous, and
    a fit to target statistics has no log-likelihood where it would need
    the values (it does for the uniform references)."""
    formula = "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')"
    target = ergmx.summary_stats(zach(), formula, response="contexts")
    g = ergmx.san(empty(), formula, target, response="contexts", reference="Poisson", seed=1)
    assert ergmx.summary_stats(g, formula, response="contexts") == target
    flo = load("flomarriage")
    flo.delete_edges(flo.es)
    g = ergmx.san(flo, "sum + sum(pow=2)", [10.0, 60.0], response="w", reference="StdNormal", seed=1)
    reached = ergmx.summary_stats(g, "sum + sum(pow=2)", response="w")
    assert reached["sum"] == pytest.approx(10, abs=0.01) and reached["sum2"] == pytest.approx(60, abs=0.01)
    fit = ergmx.ergm(empty(), "sum", response="contexts", reference="Poisson", target_stats=[231], seed=1)
    assert fit.coef["sum"] == pytest.approx(math.log(231 / 561), abs=0.03)
    assert fit.loglik is None and "log-density of the values is unknown" in str(fit.summary())
    assert "Fitted to target statistics" in str(fit.summary())
    fit = ergmx.ergm(empty(), "sum", response="contexts", reference="DiscUnif(0, 5)", target_stats=[231], seed=1)
    assert fit.loglik is not None
