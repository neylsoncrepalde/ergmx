---
file_format: mystnb
kernelspec:
  name: python3
---

# Latent space models

A latent space model ([Hoff, Raftery and Handcock 2002](https://doi.org/10.1198/016214502388618906))
places each vertex at an unobserved position in a "social space". The closer
two vertices are, the likelier their tie, and the dyads are independent given
the positions. Distances are symmetric and satisfy the triangle inequality:
if i is close to j and j to k, then i is close to k. So the model captures
reciprocity and transitivity without the dependence terms that can make an
ERGM degenerate. The positions can fall in clusters
([Handcock, Raftery and Tantrum 2007](https://doi.org/10.1111/j.1467-985X.2007.00471.x)),
and vertices can have random sender, receiver or sociality effects
([Krivitsky et al. 2009](https://doi.org/10.1016/j.socnet.2009.04.001)).

{func}`ergmx.ergmm` fits these models, as R's latentnet, with the log-odds of
a tie from i to j

$$
\eta_{ij} = \beta \cdot x_{ij} - \lVert Z_i - Z_j \rVert + \delta_i + \gamma_j ,
$$

by Bayesian MCMC.

## Clusters of positions

Sampson's monks named whom they liked:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

monks = datasets.load("samplike")
fit = ergmx.ergmm(monks, "euclidean(d=2, G=3)", seed=1)
print(fit.summary())
```

The intercept is the log-odds of a tie between two monks at the same
position: ties are likely between close monks. The three clusters are
Sampson's own groups:

```{code-cell} ipython3
import numpy as np

np.unique(np.column_stack([fit.pmean["Z.K"], monks.vs["group"]]), axis=0)
```

The positions are those of the minimum Kullback-Leibler (MKL) estimate
([Shortreed, Handcock and Hoff 2006](https://doi.org/10.1027/1614-2241.2.1.24)).
That is the configuration whose tie probabilities are closest to the
posterior's. The draws are rotated onto it, so that their positions are
comparable, and their clusters are relabelled ([Stephens 2000](https://doi.org/10.1111/1467-9868.00265)):

```{code-cell} ipython3
fit.plot(labels=True);
```

## The terms

- **Latent space.** `euclidean(d, G=0)` (the negative distance),
  `euclidean2(d, G)` (its square) or `bilinear(d, G)` (the inner product).
  `d` is the dimension and `G` the number of clusters. Their priors are
  latentnet's (`var.mul`, `mean.var.mul`... as keyword arguments).
- **Random effects.** `rsender`, `rreceiver` (directed) and `rsociality`,
  normal around 0, with `var=1` and `var_df=3` for their variance's prior.
- **Covariates.**
  - `latentcov(x)` takes a matrix, a graph attribute or a network.
  - `sendercov(attr)`, `receivercov(attr)` and `socialitycov(attr)` use an
    attribute of the vertex.
  - Any ergm term is also a covariate, through its change statistics (as
    `nodematch('group')`).
  - The intercept is included, unless the formula has `- 1`.

Random receiver effects let some monks be more popular than their position
explains:

```{code-cell} ipython3
fit2 = ergmx.ergmm(monks, "euclidean(d=2, G=3) + rreceiver", seed=1)
print(fit2.summary())
```

latentnet's BIC (lower is better) prefers this model. Valued networks take
`response=` and a family: `"binomial"` (with `fam_par={"trials": m}`),
`"Poisson"` or `"normal"` (with the prior of its variance).

## Using the fit

- `fit.sample`: the draws (`beta`, `Z`, `Z.K`, `Z.mean`, `Z.var`,
  `receiver`...).
- `fit.mkl`: the MKL estimate, with `fit.mkl["mbc"]`, the Bayesian clusters
  of its positions.
- `fit.pmean`: the posterior means, with each vertex's cluster probabilities
  `Z.pZK`.
- `tofit=(... "pmode", "mle")` adds the conditional posterior mode and the
  maximum likelihood estimate.
- `fit.predict()`: each dyad's posterior tie probability (`type="mkl"`, at
  the MKL).
- `fit.simulate(n)`: networks from random posterior draws.
- `fit.gof()`: the degree, shared partner and distance distributions of
  those networks against the observed ones.
- `fit.mcmc_diagnostics()`: effective sizes and R-hat across the chains
  (four by default, in parallel).

```{code-cell} ipython3
fit2.gof(seed=1).plot();
```

## Differences from latentnet

Fitted with ergmx's random numbers, the posteriors are latentnet's within
their Monte Carlo error (see [](../validation.md)). Where latentnet's code
differs from its models, ergmx follows the models:

- `predict()`'s posterior probabilities use the covariates the right way
  round for directed networks (latentnet's transpose them). For undirected
  networks they are given in both triangles.
- `simulate()` draws each pair of an undirected network once (latentnet's
  draw twice, so that a tie's probability is $1 - (1-p)^2$). It keeps
  bipartite networks bipartite and supports the binomial family.
- `gof()`'s observed statistics are those of the response, the dyads with
  values (latentnet's are those of the network object).
- To centre the random effects, the intercept takes twice the mean of
  undirected sociality effects, as $\eta$ has $s_i + s_j$ (latentnet's takes
  it once). The MKL intercept of such models differs accordingly.
- The chains' tuning is latentnet's (pilot runs aiming at an acceptance rate
  of 0.234), chain by chain. With no covariates, the joint proposal of the
  positions' scale and the random effects' shifts stays tuned (latentnet's
  stops moving).
