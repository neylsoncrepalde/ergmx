---
file_format: mystnb
kernelspec:
  name: python3
---

# Simulation

{func}`ergmx.simulate` draws networks from a model with given coefficients,
like R's `simulate(net ~ formula, coef = ...)`. The MCMC starts from the
network passed, which also provides the vertex attributes:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

flomarriage = datasets.load("flomarriage")
networks = ergmx.simulate(flomarriage, "edges + triangle", [-1.7, 0.2], nsim=3, seed=1)
[g.ecount() for g in networks]
```

The networks are of the same kind as the one passed, here igraph graphs, with
its vertex attributes. `output="stats"` returns only their statistics, one
row per network, which is much faster for many networks:

```{code-cell} ipython3
stats = ergmx.simulate(flomarriage, "edges + triangle", [-1.7, 0.2], nsim=1000, seed=1,
                       output="stats")
stats.mean(axis=0)
```

Coefficients can be given by name:

```python
ergmx.simulate(flomarriage, "edges + triangle", {"edges": -1.7, "triangle": 0.2})
```

## From a fit

{meth}`ErgmFit.simulate() <ergmx.ErgmFit.simulate>` uses the estimates. At
the MLE, the simulated networks have, on average, the observed statistics:

```{code-cell} ipython3
fit = ergmx.ergm(flomarriage, "edges + triangle", seed=1)
stats = fit.simulate(1000, seed=1, output="stats")
dict(zip(fit.names, stats.mean(axis=0).round(2))), fit.observed
```

## MCMC settings

`burnin` (16,384 proposals by default) is the number of proposals before the
first network and `interval` (1,024) the number between networks. Networks
closer together are more alike; raise `interval` when consecutive networks
must be nearly independent.

## Over time

A temporal model simulates a process, each network drawn given the one
before: {meth}`fit.simulate(time_slices=...) <ergmx.ErgmFit.simulate>` for a
fit of {func}`ergmx.tergm`, and {func}`ergmx.simulate_dynamic` for any
coefficients and starting network, with the ties that form and dissolve,
their durations, and `monitor=` statistics of each network, tie ages
included: see [](temporal.md#simulating-the-process).


## Networks with given statistics

{func}`ergmx.san`, as ergm's `san()`, searches for a network whose statistics
are given targets, by simulated annealing: proposals (the MCMC's, within the
constraints) that bring the statistics nearer the targets are accepted, and,
while the temperature is high, some that take them farther. It is a search,
not a draw from a distribution. Starting from an empty network of 50
vertices:

```{code-cell} ipython3
import igraph as ig

empty = ig.Graph(n=50)
target = ergmx.san(empty, "edges + triangle", [100, 20], seed=1)
ergmx.summary_stats(target, "edges + triangle")
```

Terms in `offset()` are not targeted: their coefficients (`offset_coef`)
bias the search, and a coefficient of `-inf` forbids the ties they count.
{class}`ergmx.SanControl` holds ergm's settings (`nsteps`, `maxit`, `tau`...).
With `response=` and `reference=`, it searches for a valued network, with
the valued model's proposals ([](valued.md#fitting)).

### Fitting to target statistics

`ergmx.ergm(..., target_stats=...)`, as ergm's `target.stats`, fits a model
to statistics rather than to an observed network: the estimate whose
expected statistics are the targets, for instance a population's known
density and clustering. It anneals a network towards the targets, then fits
the model with the targets as the observed statistics; dyad-independent
models get the exact MLE.

```{code-cell} ipython3
mesa = datasets.load("faux.mesa.high")
fit = ergmx.ergm(mesa, "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)",
                 target_stats=[250, 190, 200], seed=1, eval_loglik=False)
fit.summary()
```

This is the cross-sectional half of how EpiModel parameterizes network
models; the dynamic half is the EGMME of [](temporal.md#a-process-from-one-network-the-egmme).
