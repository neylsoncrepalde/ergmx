"""Log-likelihood of dyad-dependent ERGMs, by path sampling from the
dyad-independent submodel, as ergm's "bridge sampling" does.

Along the straight path theta(u) = (1 - u) theta_from + u theta_to,

    loglik(theta_to) = loglik(theta_from) + integral over u of f(u),
    f(u) = Delta . (g_obs - E_u[g]),    f'(u) = -Var_u(Delta . g),

with Delta = theta_to - theta_from. The path starts at the MLE of the model's
dyad-independent terms plus an edges term, with the dependent coefficients at
0, whose log-likelihood is exact.

ergm integrates with the midpoint rule, whose error falls as 1/bridges^2. As
the derivative is a variance, known from the same samples, ergmx uses the
Euler-Maclaurin corrected trapezoidal rule instead:

    integral = h (f_0 / 2 + f_1 + ... + f_{K-1} + f_K / 2) + h^2 / 12 (f'(0) - f'(1)),

whose error falls as 1/bridges^4 (on a network of 6 vertices, 0.0015 against
0.06 for the midpoint rule with 32 bridges).
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._estimation import (
    Control,
    _density_guard_error,
    _max_edges,
    autocorrelation_time,
    logistic_regression,
)
from ._model import BoundModel, bind
from .terms import edges


def _dyad_independent_start(model: BoundModel) -> tuple[np.ndarray, float]:
    """Coefficients of the dyad-independent submodel, 0 for the other terms,
    and its exact log-likelihood."""
    x, y = model.core.mple_data(model.network.edges)
    columns = [c for term, cols in model.term_columns() if term.dyad_independent for c in cols]
    rows, counts = np.unique(np.column_stack([x[:, columns], y]), axis=0, return_counts=True)
    beta, _, loglik = logistic_regression(rows[:, :-1], rows[:, -1], counts.astype(float))
    theta = np.zeros(model.n_stats)
    theta[columns] = beta
    return theta, loglik


def bridge_loglik(model: BoundModel, theta: np.ndarray, control: Control, interval: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    """The log-likelihood at `theta` and its Monte Carlo standard error."""
    augmented = model
    if "edges" not in model.names:
        augmented = bind(model.network, model.formula + edges())
    theta_to = np.append(theta, np.zeros(augmented.n_stats - model.n_stats))
    theta_from, loglik_from = _dyad_independent_start(augmented)
    delta = theta_to - theta_from

    k, chains, samples = control.bridges, control.bridge_chains, control.bridge_samplesize
    u = np.arange(k + 1) / k  # both ends included
    points = [(1 - v) * theta_from + v * theta_to for v in u for _ in range(chains)]
    max_edges = _max_edges(augmented, control)
    try:
        sample, _, _ = augmented.core.simulate(
            [augmented.network.edges] * len(points), [], 16 * interval, interval, samples,
            int(rng.integers(2**63)), triadic_weight=augmented.triadic_weight(control.triadic_weight),
            max_edges=max_edges, chain_thetas=[p.tolist() for p in points],
        )
    except _core.DensityGuardError:
        raise _density_guard_error(augmented, max_edges) from None
    # f = Delta . (g_obs - g), by point: points x chains x samples
    f = ((augmented.observed() - sample) @ delta).reshape(k + 1, chains, samples)
    means = f.mean(axis=(1, 2))
    variances = np.array([
        f[b].var(ddof=1) * autocorrelation_time(f[b][:, :, None])[0] / (chains * samples)
        for b in range(k + 1)
    ])
    h = 1 / k
    weights = np.full(k + 1, h)
    weights[[0, -1]] = h / 2
    slope_from, slope_to = -f[0].var(), -f[-1].var()
    llr = weights @ means + h * h / 12 * (slope_from - slope_to)
    return float(loglik_from + llr), float(np.sqrt(weights**2 @ variances))
