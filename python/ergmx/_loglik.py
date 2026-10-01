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

With missing dyads (`Handcock and Gile 2010 <https://doi.org/10.1214/08-AOAS221>`__), the log-likelihood is that of
the observed dyads, and the integrand becomes

    f(u) = Delta . (E_u[g | observed dyads] - E_u[g]),
    f'(u) = Var_u(Delta . g | observed dyads) - Var_u(Delta . g),

with conditional samples at every point of the path. Under dyad-dependent
constraints (bd, degrees...) there is no dyad-independent submodel to start
from: the path starts at 0, and the log-likelihood is relative to that null
model, the uniform distribution on the constrained networks, as in ergm.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._estimation import (
    Control,
    _density_guard_error,
    _max_edges,
    autocorrelation_time,
    regression,
)
from ._model import BoundModel, bind
from .terms import edges


def _dyad_independent_start(model: BoundModel) -> tuple[np.ndarray, float]:
    """Parameters of the dyad-independent submodel (with the fixed values of
    its offsets), 0 for the other terms, and its exact log-likelihood on the
    free observed dyads."""
    x, y = model.mple_data()
    independent = np.zeros(model.n_params, dtype=bool)
    for term, cols in model.term_columns():
        if term.dyad_independent:
            independent[cols] = True
    theta, _, loglik = regression(x, y, model, params=independent, zero=~independent)
    return theta, loglik


def _sample(model, points, interval, samples, rng, control, conditional):
    max_edges = _max_edges(model, control)
    try:
        sample, _, _ = model.simulate(
            [model.network.edges] * len(points), [], 16 * interval, interval, samples,
            int(rng.integers(2**63)), conditional=conditional, max_edges=max_edges,
            triadic_weight=model.triadic_weight(control.triadic_weight),
            chain_thetas=[p.tolist() for p in points],
        )
    except _core.DensityGuardError:
        raise _density_guard_error(model, max_edges) from None
    return sample


def bridge_loglik(model: BoundModel, theta: np.ndarray, control: Control, interval: int,
                  rng: np.random.Generator) -> tuple[float, float, bool]:
    """The log-likelihood at `theta`, its Monte Carlo standard error, and
    whether it is relative to the null model."""
    relative = model.constraints.dyad_dependent
    augmented = model
    if not relative and "edges" not in model.names and "offset(edges)" not in model.names:
        augmented = bind(model.network, model.formula + edges(), model.constraints,
                         model.fixed_values[model.fixed & ~model.constant] if model.fixed.any() else None)
    # The path runs between the statistics' coefficients (eta), in which the
    # likelihood is an exponential family whatever the parameters.
    theta_to = augmented.eta(np.append(theta, np.zeros(augmented.n_params - model.n_params)))
    if relative:
        theta_from, loglik_from = np.zeros(augmented.n_stats), 0.0
    else:
        start, loglik_from = _dyad_independent_start(augmented)
        theta_from = augmented.eta(start)
    # Coefficients equal at both ends (including -inf offsets) don't move.
    same = theta_from == theta_to
    theta_from[same] = theta_to[same]
    delta = np.where(same, 0.0, theta_to - theta_from)

    k, chains, samples = control.bridges, control.bridge_chains, control.bridge_samplesize
    u = np.arange(k + 1) / k  # both ends included
    points = [theta_from + v * delta for v in u for _ in range(chains)]

    def projected(sample):
        """Delta . g, by point: points x chains x samples."""
        changing = delta != 0
        return (sample[..., changing] @ delta[changing]).reshape(k + 1, chains, samples)

    def moments(values):
        means = values.mean(axis=(1, 2))
        variances = np.array([
            values[b].var(ddof=1) * autocorrelation_time(values[b][:, :, None])[0] / (chains * samples)
            for b in range(k + 1)
        ])
        return means, variances, np.array([values[b].var() for b in range(k + 1)])

    means, errors, spread = moments(projected(_sample(augmented, points, interval, samples, rng,
                                                     control, conditional=False)))
    if augmented.has_missing:
        means_obs, errors_obs, spread_obs = moments(projected(_sample(
            augmented, points, interval, samples, rng, control, conditional=True)))
    else:
        observed = augmented.observed()
        changing = delta != 0
        means_obs = np.full(k + 1, observed[changing] @ delta[changing])
        errors_obs, spread_obs = np.zeros(k + 1), np.zeros(k + 1)
    f = means_obs - means
    h = 1 / k
    weights = np.full(k + 1, h)
    weights[[0, -1]] = h / 2
    slope = spread_obs - spread  # f'(u)
    llr = weights @ f + h * h / 12 * (slope[0] - slope[-1])
    se = float(np.sqrt(weights**2 @ (errors + errors_obs)))
    return float(loglik_from + llr), se, relative
