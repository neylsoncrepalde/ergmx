---
file_format: mystnb
kernelspec:
  name: python3
---

# Fitting models

{func}`ergmx.ergm` fits a model to a network:

```python
fit = ergmx.ergm(network, formula, seed=1)
```

How it fits depends on whether the model's terms are *dyad-independent*:
whether the statistics' change from adding a tie depends only on that tie
(`edges`, `nodematch`, `nodecov`...) or also on the rest of the network
(`mutual`, `triangle`, `gwesp`...).

## Dyad-independent models: the exact MLE

Without dependence, an ERGM is a logistic regression on the dyads, and
`ergmx` computes its maximum likelihood estimate exactly, with the
log-likelihood, AIC and BIC. Do wealthier Florentine families marry more?

```{code-cell} ipython3
import ergmx
from ergmx import datasets

flomarriage = datasets.load("flomarriage")
fit = ergmx.ergm(flomarriage, "edges + nodecov('wealth')")
fit.summary()
```

## Dyad-dependent models: the Monte Carlo MLE

With dependence, the likelihood has a normalizing constant that sums over
every possible network, so `ergmx` estimates it by MCMC, as ergm does:

1. It starts from the maximum pseudo-likelihood estimate (MPLE).
2. It simulates networks at the current coefficients, in parallel chains,
   and moves the coefficients towards values whose simulated networks have,
   on average, the observed statistics ([Hummel et al. 2012](https://doi.org/10.1080/10618600.2012.679224) step lengths with
   a log-normal approximation).
3. It stops when two consecutive steps are full steps, then draws a larger
   final sample for the standard errors.

```{code-cell} ipython3
mesa = datasets.load("faux.mesa.high")
fit = ergmx.ergm(
    mesa, "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)", seed=1
)
fit.summary()
```

The standard errors include the Monte Carlo error of the estimate; the
`MCMC %` column is its share of the variance, as in ergm. A last line says
whether the estimation converged. The log-likelihood is estimated too: see
[](model-comparison.md).

The estimates are also available directly:

```{code-cell} ipython3
fit.coef
```

```{code-cell} ipython3
fit.stderr
```

`fit.params` and `fit.cov` give the same as an array and a matrix, in the
order of `fit.names`.

## Seeds and threads

The MCMC is random: pass `seed=` for reproducible results. Chains run in
parallel threads, four by default (or one per CPU, if fewer), and a fit is
reproducible for a given seed and number of chains. `n_chains=1` runs on a
single thread.

To watch the estimation, turn on logging:

```{code-cell} ipython3
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
fit = ergmx.ergm(mesa, "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)", seed=2)
logging.getLogger().setLevel(logging.WARNING)
```

Each iteration reports the MCMC interval (proposals between sampled
networks), the effective sample size and the step length. When the samples
are too autocorrelated, the interval grows.

## Settings

{class}`ergmx.Control` holds the settings of the MCMC and the estimation, with
ergm's defaults. Pass a `Control`, or override single settings as keyword
arguments:

```python
ergmx.ergm(mesa, formula, samplesize=2048, n_chains=8)
ergmx.ergm(mesa, formula, control=ergmx.Control(interval=4096))
```

The most useful are `samplesize` (networks per iteration), `interval`,
`burnin`, `n_chains` and `effective_size` (the effective sample size the
interval adapts to).

## Other estimates and starting values

`estimate="MPLE"` stops at the MPLE, which is quick but has unreliable
standard errors for dyad-dependent models. `estimate="CD"` stops at the
*contrastive divergence* estimate: the coefficients at which networks a few
MCMC steps away from the observed one have, on average, its statistics. It has
no standard errors.

`init` sets where the Monte Carlo MLE starts: `"MPLE"` (the default), `"CD"`,
or your own coefficients, such as those of a previous fit:

```python
ergmx.ergm(mesa, formula, init="CD")
ergmx.ergm(mesa, formula, init=previous_fit.params)
```

When a fit doesn't converge, or stops with a `DegeneracyError`, see
[](diagnostics.md).

## Estimating the decay: curved models

`gwesp(0.5, fixed=TRUE)` fixes the decay at 0.5. Without `fixed=TRUE`, as is
ergm's default, the decay is estimated with the other coefficients, starting
from 0.5, and the model is a *curved* exponential family:

```{code-cell} ipython3
curved = ergmx.ergm(
    mesa, "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5)", seed=1
)
curved.summary()
```

`gwesp.decay` is the estimated decay, with its standard error. The model's
statistics are then the counts of ties with each number of shared partners
(`curved.observed`), which the two parameters weight: see the
[term reference](../terms.md#curved-terms). R's ergm fits this model only
from a contrastive divergence start (`init.method = "CD"`): from its default
start, a simulated network exceeds its cutoff of 30 shared partners and the
fit stops with an error. ergmx counts those ties in an overflow statistic, and
its estimates match R's.

## Fixed coefficients

`offset(term)` fixes a term's coefficients instead of estimating them, at the
values given as `offset_coef`, in formula order:

```{code-cell} ipython3
fixed = ergmx.ergm(flomarriage, "offset(edges) + nodecov('wealth')", offset_coef=[-2.6])
fixed.summary()
```

Offsets don't count as parameters in AIC and BIC. A coefficient of `-inf`
forbids the ties the term counts: see [](multilevel.md#fixed-coefficients).

## Constraints and missing ties

`constraints=` restricts the networks the model puts probability on, for
example to bounded degrees in fixed-choice designs: see [](constraints.md).
Dyads whose value is unknown are marked as edges with `na=True`; the fit is
then conditional on the observed ones: see [](missing-data.md).

## Saving fits

A fit takes seconds to minutes; {meth}`fit.save(path) <ergmx.ErgmFit.save>`
keeps it, with its network, estimates, MCMC sample and settings, and
{func}`ergmx.load_fit` brings it back, to summarize, simulate, check or
compare as before (R's `saveRDS()` and `readRDS()`):

```python
fit.save("mesa_fit.pkl")
fit = ergmx.load_fit("mesa_fit.pkl")
```

The file is a Python pickle (the fits pickle, too, with `pickle.dump()`),
so load only files you trust; loading a file saved by another version of
ergmx warns.

