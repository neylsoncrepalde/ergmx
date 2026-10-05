import json
import warnings

import numpy as np
import pytest
from conftest import DATA

import ergmx
from ergmx import ErgmDifferenceWarning, Layer
from ergmx.datasets import load
from ergmx._layers import L, LayerLogic

REFERENCE = DATA / "r_multilayer_reference.json"
R = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


def flo():
    return Layer(m=load("flomarriage"), b=load("flobusiness"))


def monks():
    return Layer(load("samplk1"), load("samplk2"), load("samplk3"))


def _adjacency(name):
    return np.array(load(name).get_adjacency().data, dtype=bool)


def _shared_partners(kind, sp_type, path, base, any_order, ds_or_decay):
    """ergm.multi's documented OSP and ISP shared partners on the monks'
    layers, by brute force: of ordered pairs (i, j), the first tie (the one
    at i) in the first path layer, unless in any order."""
    layers = [_adjacency(f"samplk{t}") for t in (1, 2, 3)]
    p1, p2 = layers[path[0] - 1], layers[path[1] - 1]
    n = len(p1)

    def two(a1, b1, a2, b2):
        return (p1[a1, b1] and p2[a2, b2]) or (any_order and p2[a1, b1] and p1[a2, b2])

    counts = []
    for i in range(n):
        for j in range(n):
            if i == j or (kind == "esp" and not layers[base - 1][i, j]) or \
                    (kind == "nsp" and layers[base - 1][i, j]):
                continue
            ks = [k for k in range(n) if k not in (i, j)]
            counts.append(sum(two(i, k, j, k) if sp_type == "OSP" else two(k, i, k, j) for k in ks))
    counts = np.array(counts)
    if isinstance(ds_or_decay, float):
        a = ds_or_decay
        return [np.sum(np.exp(a) * (1 - (1 - np.exp(-a)) ** counts))]
    return [np.sum(counts == d) for d in ds_or_decay]


# ergm.multi 0.3.0's OSP and ISP shared partners in order depend on the order
# of the ties (its cache keys them by unordered pairs): these sets' terms, by
# their documented definition instead (kind, type, path layers, base layer,
# any order, degrees or decay).
IN_ORDER = {
    "d7": [("esp", "OSP", (1, 3), 2, True, [0, 1]), ("esp", "ISP", (1, 3), 2, False, [0, 1])],
    "d9": [("dsp", "OSP", (1, 3), None, True, [0, 1, 2]), ("dsp", "ISP", (1, 3), None, False, [0, 1, 2]),
           ("nsp", "OSP", (1, 3), 2, False, [0, 1, 2]), ("nsp", "ISP", (1, 3), 2, True, [0, 1, 2])],
    "d10": [("nsp", "ISP", (1, 3), 2, False, 0.5), ("esp", "OSP", (2, 3), 1, False, 0.5),
            ("dsp", "ISP", (1, 2), None, True, 0.5)],
}


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("key", list(R["stats"]) if R else [])
def test_layer_statistics_match_r(key):
    """L() with Layer Logic and the layer-aware terms, as ergm.multi's, on
    the Florentine marriages and business ties (undirected) and the monks'
    liking at three times (directed)."""
    r = R["stats"][key]
    g = flo() if key.startswith("u") else monks()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = ergmx.summary_stats(g, r["formula"])
    assert list(stats) == np.atleast_1d(r["names"]).tolist(), r["formula"]
    expected = np.atleast_1d(r["values"])
    if key in IN_ORDER:
        assert any(issubclass(w.category, ErgmDifferenceWarning) for w in caught)
        expected = np.concatenate([_shared_partners(*spec) for spec in IN_ORDER[key]])
    np.testing.assert_allclose(list(stats.values()), expected, rtol=1e-8, atol=1e-8, err_msg=r["formula"])


@pytest.mark.skipif(R is None, reason="no R reference")
@pytest.mark.parametrize("name", list(R["fits"]) if R else [])
def test_layer_fits_match_r(name):
    r = R["fits"][name]
    fit = ergmx.ergm(flo() if r["network"] == "flo" else monks(), r["formula"], seed=1, eval_loglik=False)
    assert fit.names == r["names"]
    se = np.array(r["se"])
    assert np.all(np.abs(fit.params - r["coef"]) < 0.3 * se), (fit.params, r["coef"], se)
    np.testing.assert_allclose(list(fit.stderr.values()), se, rtol=0.25)


@pytest.mark.parametrize("directed", [False, True])
def test_layer_terms_track_their_statistics(directed):
    """The statistics the sampler tracks, change by change, against those of
    the simulated networks (dicts of layers, which Layer() takes back)."""
    if directed:
        g = monks()
        formula = ("edges + L(~mutual + edges, ~t(`1`) & `2`) + L(~edges, ~(`1` + `2` + `3`) >= 2) + "
                   "CMBL(c(~`1`, ~`3`)) + mutualL(Ls = c(~`1`, ~`2`)) + "
                   "mutualL(same = 'group', diff = TRUE, Ls = c(~`3`, ~`3`)) + twostarL(c(~`1`, ~`2`), 'path') + "
                   "twostarL(c(~`1`, ~`3`), 'in', distinct = FALSE) + "
                   "despL(0:1, 'OSP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`), L.in_order = TRUE) + "
                   "dnspL(0:1, 'ISP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`)) + "
                   "ddspL(0:1, 'ITP', Ls.path = c(~`1`, ~`3`), L.in_order = TRUE) + "
                   "dgwespL(0.5, fixed = TRUE, type = 'OTP', L.base = ~`3`, Ls.path = c(~`1`, ~`2`))")
    else:
        g = flo()
        formula = ("L(~edges + triangle, ~m | b) + L(~kstar(2), c(2 ~ m, -1 ~ (b & !m))) + CMBL + "
                   "twostarL(c(~m, ~b)) + twostarL(c(~m, ~b), distinct = FALSE) + "
                   "despL(0:2, L.base = ~m, Ls.path = c(~b, ~m), L.in_order = TRUE) + "
                   "dgwnspL(0.5, fixed = TRUE, L.base = ~b, Ls.path = c(~m, ~b)) + ddspL(0:1, Ls.path = c(~m, ~b))")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErgmDifferenceWarning)
        names = list(ergmx.summary_stats(g, formula))
        coef = np.zeros(len(names))
        coef[0] = -1.5
        tracked = ergmx.simulate(g, formula, coef, 10, seed=2, interval=300, output="stats")
        networks = ergmx.simulate(g, formula, coef, 10, seed=2, interval=300)
        assert list(networks[0]) == g.graph_attributes["_layers"]
        recomputed = [list(ergmx.summary_stats(Layer(h), formula).values()) for h in networks]
    np.testing.assert_allclose(tracked, recomputed, rtol=1e-9, atol=1e-9)


def test_layer_logic_labels_and_dyad_independence():
    assert LayerLogic.parse("(`1` + `2`) >= 2").label() == "(`1`+`2`)>=2"
    assert LayerLogic.parse("m & !b").label() == "m&!b"
    assert LayerLogic.parse("`1`").label() == "1"
    # A logical layer of a lone layer leaves the dyads independent; of two, it ties them.
    assert L("edges", "~m").dyad_independent and L("edges", "c(~m, ~t(b))").dyad_independent
    assert not L("edges", "~m & b").dyad_independent
    assert not L("edges", "~m & t(m)").dyad_independent
    assert not L("triangle", "~m").dyad_independent


def test_layer_errors():
    with pytest.raises(ValueError, match="at least 2"):
        Layer(load("flomarriage"))
    with pytest.raises(ValueError, match="same vertices"):
        Layer(load("flomarriage"), load("samplk1"))
    with pytest.raises(ValueError, match="all directed or all undirected"):
        Layer(load("samplk1"), load("samplk2").as_undirected())
    with pytest.raises(ValueError, match="Layer"):
        ergmx.summary_stats(load("flomarriage"), "L(~edges, ~m)")
    with pytest.raises(ValueError, match="no layer"):
        ergmx.summary_stats(flo(), "L(~edges, ~x)")
    with pytest.raises(ValueError, match="no layer has one"):
        ergmx.summary_stats(flo(), "L(~edges, ~!m)")
    with pytest.raises((ValueError, NotImplementedError, ergmx.FormulaError), match="Ls"):
        ergmx.summary_stats(monks(), "mutualL")
    with pytest.raises((ValueError, NotImplementedError, ergmx.FormulaError), match="reciprocated two-paths"):
        ergmx.summary_stats(monks(), "despL(1, 'RTP', L.base = ~`1`)")
    with pytest.raises((ValueError, NotImplementedError, ergmx.FormulaError), match="fixed=TRUE"):
        ergmx.summary_stats(flo(), "gwespL(0.5, L.base = ~m)")
