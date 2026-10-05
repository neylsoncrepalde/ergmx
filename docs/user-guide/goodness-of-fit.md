---
file_format: mystnb
kernelspec:
  name: python3
---

# Goodness of fit

A model converges to the coefficients that reproduce the statistics in its
formula, but does it reproduce the rest of the network's structure?
{func}`ergmx.gof` simulates networks from the model and compares their
distributions with the observed network's, like R's `gof()`:

- the **degree** distribution (in- and out-degrees if directed),
- the **edgewise shared partners**: for each tie, the number of partners its
  two vertices share (outgoing two-paths if directed),
- the **minimum geodesic distances** between pairs of vertices,
- and the **model statistics** themselves.

```{code-cell} ipython3
import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")
fit = ergmx.ergm(
    mesa, "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)", seed=1
)
result = fit.gof(seed=1)
result["espartners"]
```

For each value, `obs` is the observed count, `min`, `mean` and `max` those of
the simulated networks, and the Monte Carlo p-value is ergm's: twice the
share of simulated networks at least as far out as the observed one, on the
smaller side. Small p-values point at what the model misses.

```{code-cell} ipython3
result.plot();
```

The model reproduces the shared partners well, but not everything: the
observed network has more pairs of students 8 or more steps apart than its
simulations, so it is more elongated, made of longer chains of friendships,
than the model's networks. Print `result["distance"]` for the numbers.

## Options

`nsim` sets the number of simulated networks (100 by default) and `stats`
which distributions to compute, among `"degree"`, `"idegree"`, `"odegree"`,
`"b1degree"`, `"b2degree"` (bipartite), `"espartners"`, `"dspartners"`,
`"distance"` and `"model"`. The simulated networks are spaced
by the MCMC interval the fit ended with; `interval=` and `burnin=` change it.
For valued models, the default, as ergm's, is the model statistics and
`"cdf"`, the distribution of the values ([](valued.md#goodness-of-fit)).

`gof` also works without a fit, from a formula and coefficients:

```python
ergmx.gof(mesa, "edges + nodematch('Grade')", [-6.0, 2.0])
```

## By level, and network by network

`by="level"` computes the distributions within each value of a vertex
attribute, and the ties between two values, as for the levels of a
[multilevel network](multilevel.md). For models of several networks,
{func}`ergmx.gofN` checks each network's statistics against its simulations,
with Pearson residuals, as ergm.multi's `gofN()`: see
[](multiple-networks.md#goodness-of-fit-network-by-network).

