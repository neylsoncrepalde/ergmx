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

