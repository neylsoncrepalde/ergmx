"""Statistics of tie ages, as tergm's durational terms: targets of the EGMME
(:func:`ergmx.tergm` with ``estimate="EGMME"``) and monitors of dynamic
simulations. A tie's age is the number of time steps it has existed, 1 in
the step it formed, as in tergm. They describe a network with the ages of its
ties, so they can't be terms of a model to fit."""

from __future__ import annotations

import numpy as np

from ._network import Network
from .terms import Term, _level_codes, _level_name, _select_levels, _sorted_levels, _values


class Durational(Term):
    """A statistic of the ages of a network's ties."""

    durational = True
    dyad_independent = False

    def value(self, network: Network, edges: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """The statistics of the network with these edges, whose ages are `ages`."""
        raise NotImplementedError

    def spec(self, network):
        raise ValueError(f"{self!r} is a statistic of tie ages: use it as a target of "
                         "tergm(estimate='EGMME') or in simulate_dynamic(monitor=...), not in a model")

    def __repr__(self) -> str:
        return self.names(None)[0] if self._fixed_names else super().__repr__()

    _fixed_names = False


def _mean(values: np.ndarray, emptyval: float, log: bool) -> float:
    if not len(values):
        return float(emptyval)
    return float(np.mean(np.log(values) if log else values))


class EdgeAges(Durational):
    """Sum over ties of their ages."""

    _fixed_names = True

    def names(self, network):
        return ["edge.ages"]

    def value(self, network, edges, ages):
        return np.array([float(np.sum(ages))])


class MeanAge(Durational):
    """Mean age of the ties (of their logarithms, with ``log``), ``emptyval`` without ties."""

    _fixed_names = True

    def __init__(self, emptyval: float = 0, log: bool = False):
        self.emptyval, self.log = float(emptyval), bool(log)

    def names(self, network):
        return ["mean.log.age" if self.log else "mean.age"]

    def value(self, network, edges, ages):
        return np.array([_mean(np.asarray(ages, dtype=float), self.emptyval, self.log)])


class EdgesAgeInterval(Durational):
    """Number of ties with age in [from, to)."""

    def __init__(self, frm, to=float("inf")):
        frm = [frm] if np.isscalar(frm) else list(frm)
        to = [to] if np.isscalar(to) else list(to)
        if len(frm) == 1 and len(to) > 1:
            frm = frm * len(to)
        if len(to) == 1 and len(frm) > 1:
            to = to * len(frm)
        if len(frm) != len(to) or any(f < 1 for f in frm):
            raise ValueError("edges.ageinterval: from must be 1 or more (a tie's age is at least 1), "
                             "with as many values as to")
        self.ranges = list(zip(frm, to))

    def names(self, network):
        return [f"edges.age{f:g}to{t:g}".replace("toinf", "toInf") for f, t in self.ranges]

    def value(self, network, edges, ages):
        ages = np.asarray(ages)
        return np.array([float(np.sum((ages >= f) & (ages < t))) for f, t in self.ranges])

    def __repr__(self) -> str:
        return f"edges.ageinterval({self.ranges!r})"


class EdgecovAges(Durational):
    """Sum over ties of a dyadic covariate times their age."""

    def __init__(self, x):
        self.x = x

    def names(self, network):
        return [f"edgecov.ages.{self.x}" if isinstance(self.x, str) else "edgecov.ages"]

    def value(self, network, edges, ages):
        from .terms import _matrix_argument

        m, _ = _matrix_argument(network, self.x, "edgecov.ages")
        if not network.directed:
            m = np.triu(m) + np.triu(m, 1).T  # the upper triangle, as edgecov
        if not len(edges):
            return np.zeros(1)
        i, j = edges[:, 0].astype(int), edges[:, 1].astype(int)
        lo, hi = (i, j) if network.directed else (np.minimum(i, j), np.maximum(i, j))
        return np.array([float(np.sum(m[lo, hi] * ages))])


class NodefactorMeanAge(Durational):
    """For each level of ``attr`` (all by default), the mean age of the ties'
    ends at vertices of that level (a tie between two of them counts twice)."""

    def __init__(self, attr: str, levels=None, emptyval: float = 0, log: bool = False):
        self.attr, self.levels, self.emptyval, self.log = attr, levels, float(emptyval), bool(log)

    def _chosen(self, network):
        values = _values(network, self.attr)
        return values, _select_levels(_sorted_levels(values), self.levels)

    def names(self, network):
        base = "nodefactor.mean.log.age" if self.log else "nodefactor.mean.age"
        return [f"{base}.{self.attr}.{_level_name(v)}" for v in self._chosen(network)[1]]

    def value(self, network, edges, ages):
        values, chosen = self._chosen(network)
        codes = np.array(_level_codes(values, chosen))
        ages = np.asarray(ages, dtype=float)
        out = []
        for k in range(len(chosen)):
            ends = np.concatenate([ages[codes[edges[:, 0].astype(int)] == k],
                                   ages[codes[edges[:, 1].astype(int)] == k]]) if len(edges) else np.zeros(0)
            out.append(_mean(ends, self.emptyval, self.log))
        return np.array(out)

    def __repr__(self) -> str:
        return f"nodefactor.mean.age({self.attr!r})"


def edge_ages() -> Term:
    """Sum over ties of their ages (tergm's ``edge.ages``)."""
    return EdgeAges()


def mean_age(emptyval: float = 0, log: bool = False) -> Term:
    """Mean age of the ties, or of their logarithms with ``log``; ``emptyval``
    if there is none (tergm's ``mean.age``)."""
    return MeanAge(emptyval, log)


def edges_ageinterval(frm, to=float("inf")) -> Term:
    """Number of ties with age in [from, to) (tergm's ``edges.ageinterval``)."""
    return EdgesAgeInterval(frm, to)


def edgecov_ages(x) -> Term:
    """Sum over ties of a dyadic covariate times their age (tergm's ``edgecov.ages``)."""
    return EdgecovAges(x)


def nodefactor_mean_age(attr: str, levels=None, emptyval: float = 0, log: bool = False) -> Term:
    """For each level of ``attr`` (all by default), the mean age of the ties'
    ends at its vertices (tergm's ``nodefactor.mean.age``)."""
    return NodefactorMeanAge(attr, levels, emptyval, log)


#: The durational statistics by their R names.
DURATIONAL = {"edge.ages": edge_ages, "mean.age": mean_age, "edges.ageinterval": edges_ageinterval,
              "edgecov.ages": edgecov_ages, "nodefactor.mean.age": nodefactor_mean_age}


class Ages:
    """The ages of a network's ties over a dynamic simulation."""

    def __init__(self, directed: bool, edges: np.ndarray, ages=None):
        self.directed = directed
        keys = self._keys(edges)
        self.age = dict(zip(keys, (np.ones(len(keys), dtype=int) if ages is None else ages).tolist()))

    def _keys(self, edges: np.ndarray) -> list[tuple[int, int]]:
        if self.directed:
            return [(int(a), int(b)) for a, b in edges]
        return [(int(min(a, b)), int(max(a, b))) for a, b in edges]

    def step(self, edges: np.ndarray) -> np.ndarray:
        """Advance one time step to the network with these edges: ties that
        persist age by one, new ones are 1. Returns the ages, as the edges."""
        keys = self._keys(edges)
        self.age = {k: self.age.get(k, 0) + 1 for k in keys}
        return np.array([self.age[k] for k in keys], dtype=int)

    def of(self, edges: np.ndarray) -> np.ndarray:
        return np.array([self.age[k] for k in self._keys(edges)], dtype=int)

    def copy(self) -> Ages:
        other = Ages.__new__(Ages)
        other.directed, other.age = self.directed, dict(self.age)
        return other
