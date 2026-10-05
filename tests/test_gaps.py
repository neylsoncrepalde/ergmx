"""Semicycles, interactions in N()'s linear models, L() of curved terms and
the gw layer terms with an estimated decay, bipartite layers, ergm's
projection operators and valued models of several networks, against R."""

import json
import warnings

import igraph as ig
import numpy as np
import pytest
from conftest import DATA

import ergmx
from ergmx._lm import model_frame
from ergmx._model import _make_unique, bind
from ergmx.datasets import load

REFERENCE = DATA / "r_gaps_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


def flo():
    return ergmx.Layer(m=load("flomarriage"), b=load("flobusiness"))


def davis_pair():
    """Davis's Southern Women, and the attendances whose woman's and event's
    numbers (from 1, in each mode) don't add up to a multiple of 3."""
    davis = load("davis")
    women = [v.index for v in davis.vs if not v["type"]]
    events = [v.index for v in davis.vs if v["type"]]
    second = davis.copy()
    second.delete_edges([e.index for e in davis.es
                         if (women.index(min(e.tuple)) + events.index(max(e.tuple)) + 2) % 3 == 0])
    return davis, second


def zachs():
    """The karate club's counts of contexts, and the counts less one (at least 1)."""
    first = load("zach")
    second = first.copy()
    second.es["contexts"] = [max(1, v - 1) for v in first.es["contexts"]]
    return ergmx.Networks(first, second)


def households():
    return ergmx.Networks([g for g in load("Goeyvaerts") if g["included"]][:80])


def mesa_layers():
    mesa = load("faux.mesa.high")
    random = ig.Graph(n=205, edges=[(a - 1, b - 1) for a, b in R["mesa_random"]])
    for a in mesa.vs.attributes():
        random.vs[a] = mesa.vs[a]
    return ergmx.Layer(friends=mesa, random=random)


def _network(key):
    if key in ("curved_L", "curved_gwespL"):
        return flo()
    if key == "bipartite_layers":
        davis, second = davis_pair()
        return ergmx.Layer(a=davis, b=second, bipartite=True)
    return {"semicycles": lambda: load("samplk3"), "lm_interactions": households, "projections": lambda: load("davis"),
            "valued_N": zachs, "households": households, "mesa_layers": mesa_layers, "davis": lambda: load("davis"),
            "zachs": zachs}[key]()


def _options(key):
    return {"valued_N": {"response": "contexts"}, "projections": {"bipartite": True}}.get(key, {})


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("key", list(R["stats"]) if R else [])
def test_statistics_match_r(key):
    r = R["stats"][key]
    stats = ergmx.summary_stats(_network(key), r["formula"], **_options(key))
    # ergmx's names are unique, as R's make.unique() makes them for fits.
    assert list(stats) == _make_unique(np.atleast_1d(r["names"]).tolist()), r["formula"]
    if key != "bipartite_layers":  # ergm.multi 0.3.0 misplaces bipartite layers' ties: below
        np.testing.assert_allclose(list(stats.values()), np.atleast_1d(r["values"]), rtol=1e-8, atol=1e-8)


def _shared_partners(first, second, mode, k_mode):
    """For each pair of vertices of `mode`, the vertices of the other mode tied
    to one of them in `first` and to the other in `second`, by brute force."""
    a = np.array(first.get_adjacency().data, dtype=bool)
    b = np.array(second.get_adjacency().data, dtype=bool)
    mine = [v.index for v in first.vs if v["type"] == (mode == 2)]
    others = [v.index for v in first.vs if v["type"] == (k_mode == 2)]
    counts = []
    for x, i in enumerate(mine):
        for j in mine[x + 1:]:
            counts.append(sum((a[i, k] and b[j, k]) or (b[i, k] and a[j, k]) for k in others))
    return np.array(counts)


def test_bipartite_layer_terms_count_their_definitions():
    davis, second = davis_pair()
    layers = ergmx.Layer(a=davis, b=second, bipartite=True)
    formula = ("L(~edges, ~a & b) + b1dspL(0:2, Ls.path = c(~a, ~b)) + b2dspL(1:3, Ls.path = c(~a, ~b)) + "
               "gwb1dspL(0.5, fixed = TRUE, Ls.path = c(~a, ~b)) + gwb2dspL(0.25, fixed = TRUE, Ls.path = ~b) + "
               "L(~b1degree(1:2) + gwb2degree(0.5, fixed = TRUE), ~a | b)")
    stats = list(ergmx.summary_stats(layers, formula).values())
    b1 = _shared_partners(davis, second, 1, 2)
    b2 = _shared_partners(davis, second, 2, 1)
    gw = lambda sp, a: np.sum(np.exp(a) * (1 - (1 - np.exp(-a)) ** sp))  # noqa: E731
    # The second layer's ties are the first's too: a & b is b, and a | b is a.
    plain_b = ergmx.summary_stats(second, "gwb2dsp(0.25, fixed=TRUE)", bipartite=True)
    plain_a = ergmx.summary_stats(davis, "b1degree(1:2) + gwb2degree(0.5, fixed=TRUE)", bipartite=True)
    expected = [second.ecount(), *[np.sum(b1 == d) for d in (0, 1, 2)], *[np.sum(b2 == d) for d in (1, 2, 3)],
                gw(b1, 0.5), *plain_b.values(), *plain_a.values()]
    np.testing.assert_allclose(stats, expected, rtol=1e-10)


@pytest.mark.skipif(R is None, reason="no R reference")
def test_design_matrices_match_r_model_matrix():
    for case in R["designs"]:
        x, names, _ = model_frame(case["formula"], R["frame"])
        assert ["(Intercept)" if n == "1" else n for n in names] == case["names"], case["formula"]
        np.testing.assert_allclose(x, np.array(case["matrix"]), err_msg=case["formula"])


@pytest.mark.skipif(R is None, reason="no R reference")
def test_curved_parameter_names_match_r():
    fit = ergmx.ergm(flo(), "L(~edges, ~m) + L(~gwesp, ~m | b)", estimate="MPLE")
    assert fit.names == R["param_names"]["curved_L"]
    # ergm.multi leaves gwespL's parameters' names unwrapped (gwesp,
    # gwesp.decay), which collide for two such terms: ergmx wraps them.
    fit = ergmx.ergm(flo(), "L(~edges, ~m) + gwespL(L.base = ~m, Ls.path = ~b)", estimate="MPLE")
    assert fit.names == ["L(m)~edges", "L(pth=(b),bse=m,inord=FALSE)~gwesp",
                         "L(pth=(b),bse=m,inord=FALSE)~gwesp.decay"]


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["fits"]) if R else [])
def test_fits_match_r(name):
    r = R["fits"][name]
    options = {"response": "contexts", "reference": "Poisson"} if name == "valued_N" else \
        {"bipartite": True} if name == "projection" else {}
    fit = ergmx.ergm(_network(r["network"]), r["formula"], seed=1, eval_loglik=False, **options)
    assert fit.names == np.atleast_1d(r["names"]).tolist()
    se = np.array(r["se"])
    if name == "lm_interactions":  # dyad-independent: the exact MLE
        np.testing.assert_allclose(fit.params, r["coef"], rtol=1e-6)
        return
    assert np.all(np.abs(fit.params - r["coef"]) < 0.3 * se), (fit.params, r["coef"], se)
    np.testing.assert_allclose(list(fit.stderr.values()), se, rtol=0.2)


def test_curved_terms_in_layers_and_projections_track_their_statistics():
    davis, second = davis_pair()
    cases = [
        (flo(), "L(~edges + gwesp + gwdegree(0.25), ~m | b) + gwespL(L.base = ~m, Ls.path = c(~m, ~b), cutoff = 2) + "
                "gwdspL(Ls.path = ~b)", {}),
        (ergmx.Layer(a=davis, b=second, bipartite=True),
         "L(~edges, ~a) + L(~edges, ~b) + b1dspL(0:2, Ls.path = c(~a, ~b)) + gwb2dspL(Ls.path = c(~a, ~b)) + "
         "L(~b1degree(1:2), ~a & !b)", {}),
        (davis, "edges + Proj1(~sum + nonzero + transitiveweights + nodecovar(center = TRUE)) + "
                "Proj2(~sum(pow = 2) + atleast(2) + CMP)", {"bipartite": True}),
    ]
    for network, formula, options in cases:
        coef = np.zeros(bind(network, formula, **options).n_params)
        coef[0] = -1.0
        tracked = ergmx.simulate(network, formula, coef, 10, seed=2, interval=200, output="stats", **options)
        draws = ergmx.simulate(network, formula, coef, 10, seed=2, interval=200, **options)
        if isinstance(draws[0], dict):
            draws = [ergmx.Layer(d, bipartite=network.bipartite or None) for d in draws]
        recomputed = [list(ergmx.summary_stats(d, formula, **options).values()) for d in draws]
        np.testing.assert_allclose(tracked, recomputed, rtol=1e-9, atol=1e-9, err_msg=formula)


def test_valued_models_of_several_networks():
    networks = zachs()
    formula = "sum + N(~sum + nonzero + nodecovar(center = TRUE) + transitiveweights, ~.NetworkID)"
    stats = ergmx.summary_stats(networks, formula, response="contexts")
    each = [ergmx.summary_stats(b.network.source, "sum + nonzero + nodecovar(center = TRUE) + transitiveweights",
                                response="contexts") for b in networks.blocks]
    sums = np.sum([list(s.values()) for s in each], axis=0)
    weighted = np.sum([(k + 1) * np.array(list(s.values())) for k, s in enumerate(each)], axis=0)
    np.testing.assert_allclose(list(stats.values()), [sums[0], *np.ravel(np.column_stack([sums, weighted]))])
    # Simulated: a graph per network, their statistics the tracked ones.
    model_formula = "N(~sum + nonzero + nodecovar(center = TRUE))"
    coef = [0.5, -1.5, 0.1]
    tracked = ergmx.simulate(networks, model_formula, coef, 5, seed=1, response="contexts", reference="Poisson",
                             output="stats")
    draws = ergmx.simulate(networks, model_formula, coef, 5, seed=1, response="contexts", reference="Poisson")
    assert len(draws[0]) == 2
    recomputed = [list(ergmx.summary_stats(ergmx.Networks(*d), model_formula, response="contexts").values())
                  for d in draws]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9)


def test_semicycles_are_the_cycles_of_the_symmetrized_network():
    g = load("samplk3")
    semi = ergmx.summary_stats(g, "cycle(3:5, semi = TRUE)")
    symmetrized = ergmx.summary_stats(g, "Symmetrize(~cycle(3:5))")
    assert list(semi.values()) == list(symmetrized.values())
    # Undirected networks ignore semi, as ergm.
    assert ergmx.summary_stats(load("flomarriage"), "cycle(3, semi = TRUE)") == {"cycle3": 3.0}


def test_errors():
    with pytest.raises(ValueError, match="length 3 or more"):
        ergmx.summary_stats(load("samplk3"), "cycle(2, semi = TRUE)")
    with pytest.raises(ValueError, match="bipartite"):
        ergmx.summary_stats(load("flomarriage"), "Proj1(~sum)")
    with pytest.raises(ValueError, match="same modes"):
        davis, second = davis_pair()
        flipped = davis.copy()
        flipped.vs["type"] = [not t for t in davis.vs["type"]]
        ergmx.Layer(davis, flipped, bipartite=True)
    with pytest.raises(ValueError, match="bipartite"):
        ergmx.summary_stats(flo(), "b1dspL(1, Ls.path = c(~m, ~b))")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="linear terms"):
            ergmx.ergm(zachs(), "N(~sum, offset = 1)", response="contexts", reference="Poisson")
