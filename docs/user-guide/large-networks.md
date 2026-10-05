---
file_format: mystnb
kernelspec:
  name: python3
---

# Large networks with local dependence

In a large network, a tie rarely depends on ties far away: friendships form
within classrooms, collaborations within departments. An ERGM whose
triangles or shared partners range over the whole network ignores that, and
is often degenerate. In an ERGM with *local dependence*
([Schweinberger and Handcock 2015](https://doi.org/10.1111/rssb.12081)), the
vertices fall in blocks. The ties within a block depend on each other, and
those between blocks are independent. Such models scale to networks of tens
of thousands of vertices.

{func}`ergmx.bigergm`, as R's bigergm
([Babkin et al. 2020](https://scholar.google.com/scholar?q=%22Large-scale+estimation+of+random+graph+models+with+local+dependence%22)),
fits them in three steps:

1. it finds the blocks, by the MM algorithm of a stochastic block model's
   variational approximation, started from infomap's communities;
2. it fits the model's dyad-independent terms to the dyads between blocks,
   a logistic regression;
3. it fits the whole model to the dyads within blocks, by MPLE (as bigergm,
   by default) or Monte Carlo MLE.

bigergm's `toyNet` has 200 vertices in 4 blocks, and the covariates `x` and
`y`:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

toy = datasets.load("toyNet")
fit = ergmx.bigergm(toy, "edges + nodematch('x') + nodematch('y') + triangle", n_blocks=4, seed=1)
print(fit.summary())
```

The blocks found are toyNet's:

```{code-cell} ipython3
import numpy as np

np.unique(np.column_stack([fit.blocks, np.array(toy.vs["block"], dtype=int)]), axis=0)
```

Within blocks, ties close triangles (a positive `triangle` coefficient) and
join vertices with the same covariates. Between blocks, ties are rarer, and
the covariates matter less.

## Options

- `blocks=` gives known blocks, and skips the clustering.
- `initialization=` starts the MM algorithm from `"walktrap"`'s communities,
  `"random"` blocks, or given ones.
- `clustering_with_features=True` (the default) makes the clustering use the
  covariates of the model's `nodematch` terms.
- `add_intercepts=True` gives each pair of blocks, and each block, its own
  intercept. With `clustering_with_features`, it also gives them their own
  covariate effects.
- `method_within="MLE"` fits the within-block model by Monte Carlo MLE.

`fit.lower_bound` and `fit.membership` are the MM algorithm's lower bounds of
the log-likelihood, and its posterior probabilities of each vertex's block.

## Simulation and goodness of fit

`fit.simulate()` draws the ties within blocks from the within-block model
(by MCMC) and those between from the between-block model (independently).
`fit.gof()` compares their degrees, shared partners and distances with the
network's:

```{code-cell} ipython3
fit.gof(50, seed=1).plot();
```

## Differences from bigergm

- bigergm 1.2.6's `simulate()` follows neither model. On toyNet, its
  networks have 262 ties between blocks on average, where the between-block
  model expects 254. Its ties within blocks average 1149, where ergm's
  simulation of the within-block model gives 1158. ergmx's simulations agree
  with both.
- With `add_intercepts`, bigergm keeps the `edges` terms, whose estimates are
  NA beside the intercepts. ergmx leaves them out.
- Infomap's communities and the k-means that splits blocks of one vertex
  are random, so they differ from bigergm's even with the same seed. From
  the same starting blocks, ergmx's MM algorithm gives bigergm's iterations,
  lower bounds and blocks.
