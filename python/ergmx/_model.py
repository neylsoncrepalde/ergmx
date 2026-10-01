"""A formula bound to a network: the bridge between the terms and the Rust core."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import _core
from ._network import Network, as_network
from .constraints import Constraints, parse_constraints
from .terms import Formula, as_formula


@dataclass(frozen=True)
class BoundModel:
    network: Network
    formula: Formula
    names: list[str]
    core: _core.Model
    constraints: Constraints = field(default_factory=Constraints)
    #: Coefficients fixed rather than estimated: offsets, and statistics that the
    #: constraints keep constant (fixed at 0), with their values.
    fixed: np.ndarray = None
    fixed_values: np.ndarray = None
    #: Statistics that the constraints keep constant.
    constant: np.ndarray = None

    @property
    def dyad_independent(self) -> bool:
        """Whether every term is dyad-independent."""
        return all(t.dyad_independent for t in self.formula)

    @property
    def exact(self) -> bool:
        """Whether the MLE is a logistic regression: dyad-independent terms and
        constraints."""
        return self.dyad_independent and not self.constraints.dyad_dependent

    @property
    def n_stats(self) -> int:
        return len(self.names)

    @property
    def free(self) -> np.ndarray:
        """Statistics whose coefficients are estimated."""
        return ~self.fixed

    @property
    def has_missing(self) -> bool:
        return len(self.network.missing) > 0

    def triadic_weight(self, requested: float | None) -> float:
        """Share of triadic MCMC proposals: 0.5 by default for models with
        triangle or shared partner terms, as ergm does, and 0 otherwise."""
        if requested is not None:
            return float(requested)
        return 0.5 if any(t.triadic for t in self.formula) else 0.0

    def term_columns(self) -> list:
        """Each term with the positions of its statistics."""
        columns, start = [], 0
        for term in self.formula:
            count = len(term.names(self.network))
            columns.append((term, list(range(start, start + count))))
            start += count
        return columns

    def observed(self) -> np.ndarray:
        """Statistics of the observed network, with missing dyads as non-ties."""
        return np.array(self.core.summary(self.network.edges))

    def theta(self, free_values) -> np.ndarray:
        """Full coefficients from the estimated ones."""
        theta = self.fixed_values.copy()
        theta[self.free] = free_values
        return theta

    # -- Sample spaces ----------------------------------------------------------------------

    @cached_property
    def fixed_dyads(self) -> np.ndarray:
        """n x n mask of the dyads fixed by the constraints or by -inf offsets."""
        mask = self.constraints.fixed(self.network)
        for k in np.flatnonzero(self.fixed & np.isneginf(self.fixed_values)):
            mask |= self._dyads_counted_by(k)
        return mask

    def _dyads_counted_by(self, stat: int) -> np.ndarray:
        """Dyads where adding a tie changes a dyad-independent statistic."""
        x, _ = self.core.mple_data(self.network.edges)
        n = self.network.n
        if self.network.directed:
            rows, cols = np.nonzero(~np.eye(n, dtype=bool))
        else:
            rows, cols = np.triu_indices(n, 1)
        mask = np.zeros((n, n), dtype=bool)
        hit = x[:, stat] != 0
        mask[rows[hit], cols[hit]] = True
        return mask | mask.T if not self.network.directed else mask

    def _space(self, free: np.ndarray | None, bounds=True, preserve=True):
        n = self.network.n
        b = self.constraints.bounds(self.network) if bounds else None
        kind = self.constraints.preserve_kind if preserve else ""
        if free is None and b is None and not kind:
            return None
        flat = None if free is None else np.ascontiguousarray(free, dtype=bool).ravel()
        return _core.Space(n, self.network.directed, flat, b, kind)

    @cached_property
    def space(self):
        """Where the networks of the model live (None: every network)."""
        fixed = self.fixed_dyads
        return self._space(~fixed if fixed.any() else None)

    @cached_property
    def space_obs(self):
        """The networks that agree with the observed dyads, for conditional samples."""
        if not self.has_missing:
            return None
        free = self.network.dyad_mask(self.network.missing) & ~self.fixed_dyads
        return self._space(free)

    @cached_property
    def space_mple(self):
        """The dyads of the logistic regressions: free and observed."""
        free = ~self.fixed_dyads & ~self.network.dyad_mask(self.network.missing)
        np.fill_diagonal(free, False)
        if free.all(where=~np.eye(self.network.n, dtype=bool)):
            return None
        return self._space(free, bounds=False, preserve=False)

    @property
    def n_observations(self) -> int:
        """Free observed dyads: the sample size of BIC, as in ergm."""
        if self.space_mple is not None:
            return int(self.space_mple.n_free)
        n = self.network.n
        return n * (n - 1) // (1 if self.network.directed else 2)

    def mple_data(self):
        return self.core.mple_data(self.network.edges, self.space_mple)

    def simulate(self, starts, theta, burnin, interval, samples, seed, *, conditional=False, **options):
        """Runs chains in the model's sample space (conditional on the observed
        dyads with `conditional`)."""
        space = self.space_obs if conditional else self.space
        return self.core.simulate(starts, [float(t) for t in theta], burnin, interval, samples, seed,
                                  space=space, **options)


def _make_unique(names: list[str]) -> list[str]:
    """Names made unique as R's make.unique does: repeats get .1, .2..."""
    seen, unique = set(names), []
    counts: dict[str, int] = {}
    for name in names:
        if name in counts:
            k = counts[name]
            while f"{name}.{k}" in seen:
                k += 1
            counts[name] = k + 1
            seen.add(f"{name}.{k}")
            unique.append(f"{name}.{k}")
        else:
            counts[name] = 1
            unique.append(name)
    return unique


def bind(network, formula, constraints=None, offset_coef=None, fitting=False) -> BoundModel:
    """Bind a formula to a network, with constraints and offset coefficients.

    With `fitting`, offset terms need their coefficients, and statistics that
    the constraints keep constant are reported."""
    network = as_network(network)
    formula = as_formula(formula)
    if not len(formula):
        raise ValueError("the formula has no terms")
    for term in formula:
        term.check(network)
    constraints = parse_constraints(constraints)
    constraints.check(network)
    names = _make_unique([name for term in formula for name in term.names(network)])
    core = _core.Model(network.n, network.directed, [t.full_spec(network) for t in formula])

    offsets = np.zeros(len(names), dtype=bool)
    constant = np.zeros(len(names), dtype=bool)
    kept = constraints.preserves
    for term, start in zip(formula, np.cumsum([0] + [len(t.names(network)) for t in formula])):
        count = len(term.names(network))
        if term.is_offset:
            offsets[start:start + count] = True
        elif kept is not None and term.degree_dependence is not None and term.degree_dependence <= kept:
            constant[start:start + count] = True
    values = np.zeros(len(names))
    if offsets.any() and (fitting or offset_coef is not None):
        if offset_coef is None:
            raise ValueError(f"give the values of the offset coefficients with offset_coef= "
                             f"({int(offsets.sum())} values)")
        offset_coef = np.atleast_1d(np.asarray(offset_coef, dtype=float))
        if offset_coef.shape != (offsets.sum(),):
            raise ValueError(f"offset_coef must have {int(offsets.sum())} values, one per offset "
                             f"statistic, not {offset_coef.size}")
        if np.any(np.isposinf(offset_coef)) or np.any(np.isnan(offset_coef)):
            raise ValueError("offset coefficients must be finite or -inf")
        values[offsets] = offset_coef
    elif offset_coef is not None and len(np.atleast_1d(offset_coef)):
        raise ValueError("offset_coef given, but the formula has no offset() terms")
    if constant.any() and fitting:
        warnings.warn(
            f"{[n for n, c in zip(names, constant) if c]} are constant under the constraints "
            f"({constraints!r}): their coefficients can't be estimated and are fixed at 0",
            stacklevel=3,
        )
    model = BoundModel(network, formula, names, core, constraints, offsets | constant, values, constant)
    for k in np.flatnonzero(model.fixed & np.isneginf(values)):
        term = next(t for t, cols in model.term_columns() if k in cols)
        if not term.dyad_independent:
            raise NotImplementedError("-inf offsets are only supported for dyad-independent terms")
    if np.isneginf(values).any():
        forbidden = model.fixed_dyads & network.dyad_mask(network.edges)
        if forbidden.any():
            raise ValueError("the observed network has ties that a -inf offset forbids")
    return model
