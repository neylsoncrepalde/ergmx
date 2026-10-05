---
file_format: mystnb
kernelspec:
  name: python3
---

# Tapered ERGMs

Many ERGMs with triangle or star terms are *degenerate*. Most of their
probability lies on networks far from the observed one (nearly empty or
nearly complete), so their MCMC can't fit them. Monks' liking with transitive
and cyclic triples is one:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

monks = datasets.load("samplk3")
try:
    ergmx.ergm(monks, "edges + ttriple + ctriple", seed=1)
except ergmx.DegeneracyError as e:
    print(e)
```

A *tapered* ERGM ([Fellows and Handcock 2017](https://proceedings.mlr.press/v54/fellows17a.html);
[Blackburn and Handcock 2023](https://doi.org/10.1080/10618600.2022.2116444))
multiplies the model's probabilities by a penalty that shrinks as the
statistics move away from their centers $m$, the observed statistics:

$$
P(Y = y) \propto \exp\Big(\theta \cdot g(y) - \sum_k \tau_k \big(g_k(y) - m_k\big)^2\Big).
$$

The penalty keeps the networks near the observed one, which removes the
degeneracy. The model's statistics still have the network's values as their
expectations at the estimate. {func}`ergmx.ergm_tapered` fits it, as R's
ergm.tapered:

```{code-cell} ipython3
fit = ergmx.ergm_tapered(monks, "edges + ttriple + ctriple", seed=1)
print(fit.summary())
```

The coefficients $\tau_k = 1 / (r^2 \max(1, |m_k|))$ make the penalty 1 when a
statistic is $r$ Poisson standard deviations from its center. The default is
$r = 2$, and a smaller $r$ tapers more. Networks simulated from the fit stay
around the observed one:

```{code-cell} ipython3
import numpy as np

stats = np.asarray(fit.simulate(200, seed=2, output="stats"))[:, :3]
stats.mean(axis=0).round(1), list(ergmx.summary_stats(monks, "edges + ttriple + ctriple").values())
```

The standard errors are those of the tapered likelihood, from
ergm.tapered's formula for the derivative of the mean statistics. That
formula accounts for centers that are themselves the observed statistics.

## Options

- `taper_terms="dependent"` tapers only the dyad-dependent terms;
  `taper_terms="ttriple"` tapers only the terms of that formula.
- `r=`, `tau=` (one per tapered statistic, or one for all) and `beta=`
  (`tau = 1 / beta**2`) set the tapering.
- `tapering_centers={"edges": 50, ...}` centers the penalty elsewhere. With
  such centers, the standard errors are the Monte Carlo MLE's, as in
  ergm.tapered.
- `target_stats=` fits the model to target statistics, which are also the
  centers.
- `estimate="MPLE"` fits the tapered model by pseudo-likelihood.

The `Taper()` operator itself, `Taper(~edges + triangle, coef = c(0.01, 0.1))`,
adds the penalty as a statistic, `Taper_Penalty`, for models written by hand.

## Estimating the tapering

With `fixed=False`, the tapering's strength $s$, a multiplier of the
coefficients $\tau_k$, is estimated as ergm.tapered's (Blackburn and Handcock
2023). Each fit's sample proposes the strength that maximizes the
log-likelihood ratio, plus a penalty of the statistics' kurtosis away from a
normal distribution's (3), for the next fit; the iterations stop once a
fit's sample proposes its own strength. The strength is estimated between
1/3 and 3, and the likelihood favors the strongest tapering, so the estimate
is often 3, where $r$ becomes $2/\sqrt{3} = 1.15$:

```{code-cell} ipython3
fit = ergmx.ergm_tapered(monks, "edges + ttriple + ctriple", fixed=False, seed=1)
print(fit.summary())
```

`fit.tapering_strength` is the strength, and `fit.tapering_history` each
iteration's strength and the one its sample proposed. The strength counts as
a parameter in AIC and BIC. {class}`ergmx.TaperingControl` sets the
iterations and the objective's penalty, as ergm.tapered's
`control.ergm.tapered()`.

## When tapering is not enough

A weak tapering can leave a model near-degenerate: with $r = 2$,
faux.mesa.high's `edges + triangle` still moves between regimes. Its fit
stops with a {class}`~ergmx.DegeneracyError` once each effective draw of the
MCMC costs hundreds of sweeps of the network, after about a minute; with
`r=1`, or the estimated tapering, it fits in seconds. (ergm.tapered's ends
after 60 iterations without converging, at coefficients whose networks have
nearly five times the observed triangles.)

## Differences from ergm.tapered

- With `taper.terms = "dependent"`, ergm.tapered takes its standard errors
  from its last Monte Carlo sample. That sample's mean statistics can be far
  from the network's (35 edges below them on faux.mesa.high), so its standard
  errors vary by a factor of 2 between seeds. ergmx samples at the estimate.
- For curved terms, ergm.tapered corrects the covariance of the linear
  parameters only and leaves the decays' as the Monte Carlo MLE's. ergmx maps
  the whole tapered covariance to the parameters.
- ergm.tapered's `Taper()` reads any single coefficient as a multiplier, so
  that tapering a single statistic divides its documented $\tau$ by $4m$
  once more. ergmx's `ergm_tapered()` uses the documented $\tau$.
- With the estimated tapering, a fit that stops for degeneracy (at the first
  strength, 1, say) moves the strength halfway to the interval's top, the
  strongest tapering. ergm.tapered's fits don't stop: they return samples
  from chains that hadn't mixed, which then propose the next strength.
