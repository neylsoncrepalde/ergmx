"""A formula bound to a network: the bridge between the terms and the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from ._network import Network, as_network
from .terms import Formula, as_formula


@dataclass(frozen=True)
class BoundModel:
    network: Network
    formula: Formula
    names: list[str]
    core: _core.Model

    @property
    def dyad_independent(self) -> bool:
        return all(t.dyad_independent for t in self.formula)

    @property
    def n_stats(self) -> int:
        return len(self.names)

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
        return np.array(self.core.summary(self.network.edges))


def bind(network, formula) -> BoundModel:
    network = as_network(network)
    formula = as_formula(formula)
    if not len(formula):
        raise ValueError("the formula has no terms")
    for term in formula:
        term.check(network)
    names = [name for term in formula for name in term.names(network)]
    core = _core.Model(network.n, network.directed, [t.spec(network) for t in formula])
    return BoundModel(network, formula, names, core)
