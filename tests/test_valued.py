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


NETWORKS = {"zach": (zach, "contexts", "Poisson"), "nominations": (nominations, "nominations", "Binomial(3)")}

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        fit = ergmx.ergm(make(), r["formula"], response=response, reference=r["reference"], seed=1,
                         eval_loglik=False)
    assert fit.converged
    for term, value in r["coef"].items():
        assert abs(fit.coef[term] - value) < 0.3 * r["se"][term], term
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
