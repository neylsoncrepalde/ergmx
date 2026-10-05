"""Terms written in Python: dyad-independent ones (tabulated) against
ergmx's own terms' exact fits, dyad-dependent ones (called back) against
their statistics and the sampler's tracked changes."""

import numpy as np
import pytest
from conftest import load

import ergmx
from ergmx import UserTerm, register_term


class PyTriangles(UserTerm):
    """Triangles of an undirected network: the shared partners of (i, j)."""

    name = "pytriangle"
    directed = False
    triadic = True

    def change(self, net, i, j, adding):
        common = len(set(net.neighbours(i)) & set(net.neighbours(j)))
        return [common if adding else -common]


class SameAttribute(UserTerm):
    """Ties between vertices with the same value of an attribute (dyad-independent)."""

    dyad_independent = True

    def __init__(self, attr):
        self.attr = attr

    def names(self, network):
        return [f"same.{self.attr}"]

    def change(self, net, i, j, adding):
        values = self.network.attributes[self.attr]
        return [1.0 if values[i] == values[j] else 0.0]


class Stars(UserTerm):
    """Two-stars (sum over vertices of C(degree, 2)) and isolates, with the
    empty network's isolates."""

    def names(self, network):
        return ["py2star", "pyisolates"]

    def empty(self, network):
        return [0.0, network.n]

    def change(self, net, i, j, adding):
        di, dj = net.degree(i), net.degree(j)
        if adding:
            return [di + dj, -(di == 0) - (dj == 0)]
        return [-(di - 1) - (dj - 1), (di == 1) + (dj == 1)]


register_term("pytriangle", PyTriangles)
register_term("same", SameAttribute)
register_term("pystars", Stars)


def test_python_terms_statistics():
    g = load("flomarriage")
    stats = ergmx.summary_stats(g, "pytriangle + triangle + pystars + kstar(2) + isolates + same('priorates')")
    assert stats["pytriangle"] == stats["triangle"]
    assert (stats["py2star"], stats["pyisolates"]) == (stats["kstar2"], stats["isolates"])


def test_dyad_independent_python_terms_get_the_exact_mle():
    g = load("faux.mesa.high")
    python = ergmx.ergm(g, "edges + same('Grade')", seed=1)
    builtin = ergmx.ergm(g, "edges + nodematch('Grade')", seed=1)
    np.testing.assert_allclose(python.params, builtin.params, rtol=1e-10)
    assert python.loglik == pytest.approx(builtin.loglik, rel=1e-10)


def test_python_terms_track_their_statistics():
    g = load("flomarriage")
    formula = "edges + pytriangle + pystars"
    coef = [-1.5, 0.2, -0.05, 0.1]
    tracked = ergmx.simulate(g, formula, coef, 8, seed=1, interval=200, output="stats")
    networks = ergmx.simulate(g, formula, coef, 8, seed=1, interval=200)
    recomputed = [list(ergmx.summary_stats(h, formula).values()) for h in networks]
    np.testing.assert_allclose(tracked, recomputed)
    fit = ergmx.ergm(g, "edges + pytriangle", seed=1, eval_loglik=False, samplesize=256)
    assert fit.converged


def test_python_term_errors():
    with pytest.raises(ValueError, match="one of ergmx's terms"):
        register_term("edges", PyTriangles)
    with pytest.raises(ValueError, match="only implemented for undirected"):
        ergmx.summary_stats(load("samplk3"), "pytriangle")

    class Wrong(UserTerm):
        def change(self, net, i, j, adding):
            return [1.0, 2.0]

    with pytest.raises(BaseException, match="2 values for 1 statistics"):
        ergmx.summary_stats(load("flomarriage"), ergmx.Formula([Wrong()]))
