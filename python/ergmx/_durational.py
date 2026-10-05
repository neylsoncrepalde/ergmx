"""Statistics of tie ages, as tergm's durational terms: targets of the EGMME
(:func:`ergmx.tergm` with ``estimate="EGMME"``), monitors of dynamic
simulations, and terms of the models they simulate. A tie's age is the
number of time steps it has existed, 1 in the step it formed, as in tergm.
As in tergm, they need the ties' history, so they can't be terms of a model
fitted to networks by CMLE or ergm()."""

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

    def aged_spec(self, network) -> tuple:
        """The term as the Rust core's durational terms expect it."""
        raise NotImplementedError

    def full_spec(self, network):
        spec = self.aged_spec(network)
        return spec if len(spec) == 4 else (*spec, [])

    def spec(self, network):
        return self.full_spec(network)[:3]

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

    def aged_spec(self, network):
        return ("aged.edge_ages", [], [])


class MeanAge(Durational):
    """Mean age of the ties (of their logarithms, with ``log``), ``emptyval`` without ties."""

    _fixed_names = True

    def __init__(self, emptyval: float = 0, log: bool = False):
        self.emptyval, self.log = float(emptyval), bool(log)

    def names(self, network):
        return ["mean.log.age" if self.log else "mean.age"]

    def value(self, network, edges, ages):
        return np.array([_mean(np.asarray(ages, dtype=float), self.emptyval, self.log)])

    def aged_spec(self, network):
        return ("aged.mean_age", [self.emptyval, float(self.log)], [])


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

    def aged_spec(self, network):
        return ("aged.edges_ageinterval", [float(v) for r in self.ranges for v in r], [])

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

    def aged_spec(self, network):
        return ("aged.edgecov_ages", _covariate(network, self.x, "edgecov.ages"), [])


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

    def aged_spec(self, network):
        values, chosen = self._chosen(network)
        return ("aged.nodefactor_mean_age", [self.emptyval, float(self.log)], list(_level_codes(values, chosen)))

    def __repr__(self) -> str:
        return f"nodefactor.mean.age({self.attr!r})"


class EdgecovMeanAge(Durational):
    """The mean age of the ties weighted by a dyadic covariate: the sum over
    ties of the covariate times their age (or its log), over the sum of the
    covariate; ``emptyval`` if that is 0."""

    def __init__(self, x, emptyval: float = 0, log: bool = False):
        self.x, self.emptyval, self.log = x, float(emptyval), bool(log)

    def names(self, network):
        base = "mean.log.age" if self.log else "mean.age"
        return [f"{base}.{self.x}" if isinstance(self.x, str) else base]

    def value(self, network, edges, ages):
        weights = _edge_weights(network, self.x, edges, "edgecov.mean.age")
        total = float(np.sum(weights))
        if not len(edges) or total == 0:
            return np.array([self.emptyval])
        ages = np.asarray(ages, dtype=float)
        return np.array([float(np.sum(weights * (np.log(ages) if self.log else ages)) / total)])

    def aged_spec(self, network):
        return ("aged.edgecov_mean_age", [self.emptyval, float(self.log), *_covariate(network, self.x,
                                                                                      "edgecov.mean.age")], [])


def _covariate(network, x, term: str) -> list[float]:
    """A dyadic covariate as the Rust terms read it: row by row, its upper
    triangle if undirected."""
    from .terms import _matrix_argument

    m, _ = _matrix_argument(network, x, term)
    if not network.directed:
        m = np.triu(m) + np.triu(m, 1).T
    return [float(v) for v in np.asarray(m, dtype=float).ravel()]


def _edge_weights(network, x, edges, term: str) -> np.ndarray:
    """A dyadic covariate's value at each tie (its upper triangle, if undirected)."""
    from .terms import _matrix_argument

    m, _ = _matrix_argument(network, x, term)
    if not network.directed:
        m = np.triu(m) + np.triu(m, 1).T
    if not len(edges):
        return np.zeros(0)
    i, j = edges[:, 0].astype(int), edges[:, 1].astype(int)
    lo, hi = (i, j) if network.directed else (np.minimum(i, j), np.maximum(i, j))
    return np.asarray(m[lo, hi], dtype=float)


def _emptyvals(emptyval, k: int, term: str) -> np.ndarray:
    values = np.atleast_1d(np.asarray(emptyval, dtype=float))
    if len(values) == 1:
        return np.repeat(values, k)
    if len(values) != k:
        raise ValueError(f"{term}: emptyval must have 1 value or one per statistic ({k})")
    return values


class NodemixMeanAge(Durational):
    """For each mixing type of ``attr`` (as nodemix's, but all by default), the
    mean age of its ties."""

    def __init__(self, attr: str, levels=None, levels2=None, emptyval=0, log: bool = False):
        self.attr, self.levels, self.levels2, self.emptyval, self.log = attr, levels, levels2, emptyval, bool(log)

    def _types(self, network):
        from .terms import mixing_types

        return mixing_types(network, self.attr, self.levels, True if self.levels2 is None else self.levels2)

    def names(self, network):
        chosen, _, types = self._types(network)
        base = "nodemix.mean.log.age" if self.log else "nodemix.mean.age"
        return [f"{base}.{self.attr}.{_level_name(chosen[r])}.{_level_name(chosen[c])}" for r, c in types]

    def value(self, network, edges, ages):
        chosen, codes, types = self._types(network)
        emptyval = _emptyvals(self.emptyval, len(types), "nodemix.mean.age")
        codes = np.asarray(codes)
        ages = np.asarray(ages, dtype=float)
        a = codes[edges[:, 0].astype(int)] if len(edges) else np.zeros(0, dtype=int)
        b = codes[edges[:, 1].astype(int)] if len(edges) else np.zeros(0, dtype=int)
        out = []
        for k, (r, c) in enumerate(types):
            hit = (a == r) & (b == c)
            if not network.directed:
                hit |= (a == c) & (b == r)
            out.append(_mean(ages[hit], emptyval[k], self.log))
        return np.array(out)

    def aged_spec(self, network):
        chosen, codes, types = self._types(network)
        size = len(chosen)
        mapping = np.full((size, size), -1, dtype=np.int64)
        for stat, (row, col) in enumerate(types):
            mapping[row, col] = stat
            if not network.directed:
                mapping[col, row] = stat
        emptyval = _emptyvals(self.emptyval, len(types), "nodemix.mean.age")
        return ("aged.nodemix_mean_age", [*emptyval.tolist(), float(self.log)],
                [*codes, size, *mapping.ravel().tolist()])

    def __repr__(self) -> str:
        return f"nodemix.mean.age({self.attr!r})"


class DegreeMeanAge(Durational):
    """For each degree in ``d`` (or degree range, as degrange), the mean age
    of the ties at vertices of that degree, a tie counted once per such end
    (by level of ``byarg`` too); undirected networks."""

    def __init__(self, ranges, byarg=None, emptyval=0, log: bool = False, single: bool = True):
        self.ranges, self.byarg, self.emptyval, self.log, self.single = ranges, byarg, emptyval, bool(log), single

    def check(self, network):
        if network.directed:
            raise ValueError(f"{self!r} is for undirected networks")
        if any(f == 0 for f, _ in self.ranges):
            raise ValueError("the age of ties at vertices of degree 0 is meaningless")

    def _levels(self, network):
        if self.byarg is None:
            return None, None
        values = _values(network, self.byarg)
        levels = _sorted_levels(values)
        if len(levels) < 2:
            raise ValueError(f"the attribute {self.byarg!r} has only one value")
        return values, levels

    def _cells(self, network):
        """(from, to, level) of each statistic, levels outermost."""
        _, levels = self._levels(network)
        if levels is None:
            return [(f, t, None) for f, t in self.ranges]
        return [(f, t, v) for v in levels for f, t in self.ranges]

    def names(self, network):
        log = ".log" if self.log else ""
        n = network.n if network is not None else None
        out = []
        for f, t, v in self._cells(network):
            by = "" if v is None else f".{self.byarg}{_level_name(v)}"
            if self.single:
                out.append(f"degree{f:g}.mean{log}.age" if v is None
                           else f"deg{f:g}{by}.mean{log}.age")
            else:
                span = f"{f:g}+" if t == np.inf or (n is not None and t >= n + 1) else f"{f:g}to{t:g}"
                out.append(f"deg{span}{by}.mean{log}.age")
        return out

    def value(self, network, edges, ages):
        values, _ = self._levels(network)
        cells = self._cells(network)
        emptyval = _emptyvals(self.emptyval, len(cells), "degree.mean.age")
        degree = np.bincount(edges.ravel().astype(int), minlength=network.n) if len(edges) else np.zeros(network.n)
        ages = np.asarray(ages, dtype=float)
        out = []
        for k, (f, t, v) in enumerate(cells):
            def counts(end):
                ok = (degree[end] >= f) & (degree[end] < t)
                if v is not None:
                    ok &= np.array([values[e] == v for e in end], dtype=bool)
                return ok

            if not len(edges):
                out.append(emptyval[k])
                continue
            ends = [ages[counts(edges[:, side].astype(int))] for side in (0, 1)]
            out.append(_mean(np.concatenate(ends), emptyval[k], self.log))
        return np.array(out)

    def aged_spec(self, network):
        values, levels = self._levels(network)
        cells = self._cells(network)
        codes = [-1] * network.n if levels is None else list(_level_codes(values, levels))
        triples = [x for f, t, v in cells
                   for x in (int(f), -1 if t == np.inf else int(t), -1 if v is None else levels.index(v))]
        emptyval = _emptyvals(self.emptyval, len(cells), "degree.mean.age")
        return ("aged.degree_mean_age", [*emptyval.tolist(), float(self.log)], [*codes, *triples])

    def __repr__(self) -> str:
        return f"{'degree' if self.single else 'degrange'}.mean.age({self.ranges!r})"


class EdgeAgesOperator(Durational):
    """tergm's EdgeAges(formula): for each statistic of a dyad-independent
    formula, the sum over ties of their age times the statistic's change on
    adding the tie."""

    def __init__(self, formula):
        self.formula = formula

    def _model(self, network):
        from ._model import bind

        model = bind(network, self.formula)
        if not model.dyad_independent:
            raise ValueError("EdgeAges() takes a dyad-independent formula")
        return model

    def names(self, network):
        return [f"EdgeAges~{name}" for name in self._model(network).stat_names]

    def aged_spec(self, network):
        model = self._model(network)
        return ("aged.EdgeAges", [], [], [t.full_spec(network) for t in model.formula])

    def value(self, network, edges, ages):
        model = self._model(network)
        empty = np.asarray(model.core.summary(np.zeros((0, 2), dtype=np.uint32)))
        out = np.zeros(model.n_stats)
        for (i, j), age in zip(np.asarray(edges, dtype=np.uint32).reshape(-1, 2), np.asarray(ages, dtype=float)):
            single = np.asarray(model.core.summary(np.array([[i, j]], dtype=np.uint32)))
            out += age * (single - empty)
        return out

    def __repr__(self) -> str:
        return f"EdgeAges({self.formula!r})"


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


def edgecov_mean_age(x, emptyval: float = 0, log: bool = False) -> Term:
    """Mean age of the ties weighted by a dyadic covariate (tergm's ``edgecov.mean.age``)."""
    return EdgecovMeanAge(x, emptyval, log)


def nodemix_mean_age(attr: str, levels=None, levels2=None, emptyval=0, log: bool = False) -> Term:
    """For each mixing type of ``attr`` (all by default), the mean age of its
    ties (tergm's ``nodemix.mean.age``)."""
    return NodemixMeanAge(attr, levels, levels2, emptyval, log)


def _ranges(frm, to):
    frm = [frm] if np.isscalar(frm) else list(frm)
    to = [to] if np.isscalar(to) else list(to)
    if len(to) == 1 and len(frm) > 1:
        to = to * len(frm)
    elif len(frm) == 1 and len(to) > 1:
        frm = frm * len(to)
    if len(frm) != len(to) or any(f >= t for f, t in zip(frm, to)):
        raise ValueError("degrange.mean.age: from and to of the same length (or one of them a single value), from < to")
    return [(float(f), float(t)) for f, t in zip(frm, to)]


def degree_mean_age(d, byarg=None, emptyval=0, log: bool = False) -> Term:
    """For each degree in ``d``, the mean age of the ties at vertices of that
    degree, by level of ``byarg`` (tergm's ``degree.mean.age``; undirected)."""
    d = [d] if np.isscalar(d) else list(d)
    return DegreeMeanAge([(float(k), float(k) + 1) for k in d], byarg, emptyval, log, single=True)


def degrange_mean_age(frm, to=float("inf"), byarg=None, emptyval=0, log: bool = False) -> Term:
    """For each degree range [from, to), the mean age of the ties at vertices
    with degrees in it, by level of ``byarg`` (tergm's ``degrange.mean.age``; undirected)."""
    return DegreeMeanAge(_ranges(frm, to), byarg, emptyval, log, single=False)


def edge_ages_operator(formula) -> Term:
    """For each statistic of a dyad-independent formula, the sum over ties of
    their age times the statistic's change on adding the tie (tergm's ``EdgeAges``)."""
    return EdgeAgesOperator(formula)


#: The durational statistics by their R names.
DURATIONAL = {"edge.ages": edge_ages, "mean.age": mean_age, "edges.ageinterval": edges_ageinterval,
              "edgecov.ages": edgecov_ages, "nodefactor.mean.age": nodefactor_mean_age,
              "edgecov.mean.age": edgecov_mean_age, "nodemix.mean.age": nodemix_mean_age,
              "degree.mean.age": degree_mean_age, "degrange.mean.age": degrange_mean_age,
              "EdgeAges": edge_ages_operator}


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

    def rows(self) -> list[tuple[int, int, int]]:
        """The ties with their ages, as (i, j, age), for the Rust core."""
        return [(i, j, int(a)) for (i, j), a in self.age.items()]

    def copy(self) -> Ages:
        other = Ages.__new__(Ages)
        other.directed, other.age = self.directed, dict(self.age)
        return other
