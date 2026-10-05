---
file_format: mystnb
kernelspec:
  name: python3
---

# ergmx

**Exponential-family random graph models (ERGMs) in Python, with a Rust core.**

`ergmx` fits, simulates and checks ERGMs with a high-level API in the spirit
of R's [ergm](https://github.com/statnet/ergm) and statnet: R-style formulas,
the same terms and statistics, and `summary()`, `gof()` and
`mcmc_diagnostics()` that read like R's. Its MCMC sampler is written in Rust
and runs chains in parallel threads, so fits take seconds.

:::{note}
`ergmx` has 189 terms and 25 operators for directed, undirected and bipartite
networks, and interactions, and terms written in Python; curved ERGMs, sample
space constraints, missing ties, multilevel networks (as MPNet), samples of
networks and multilayer networks (as ergm.multi), temporal ERGMs, EGMME and
dynamic simulation with tie ages (as tergm), and btergm; valued ERGMs (of
single networks, several networks and series of them), egocentric and
Bayesian ERGMs, with model evidence; tapered ERGMs, with an estimated
tapering (as ergm.tapered), ERGMs with local dependence for large networks
(as bigergm) and latent space models (as latentnet); MPLE, contrastive
divergence and Monte Carlo MLE, MCMC diagnostics, log-likelihoods, model
comparison, goodness of fit, tie probabilities, marginal effects and tables
of results, for networks of up to tens of thousands of vertices, all
[validated against R](validation.md).
:::

## A first look

`faux.mesa.high` is a friendship network of 205 high school students. Do
students befriend others of the same grade and race, and friends of their
friends?

```{code-cell} ipython3
import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")   # an igraph.Graph

fit = ergmx.ergm(
    mesa,
    "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)",
    seed=1,
)
fit.summary()
```

Same-grade friendships are much more likely than others (the
`nodematch.Grade` coefficient), and so are ties that close triangles
(`gwesp`). R's ergm gives the same estimates. Then check that the model
reproduces the network's structure:

```{code-cell} ipython3
fit.gof(seed=1).plot();
```

The black lines are the observed network's distributions and the boxplots
those of 100 networks simulated from the model.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Quick start
:link: user-guide/quickstart/index
:link-type: doc

Seven complete analyses, from the data to a table of results: an ERGM, and
multilevel, valued, egocentric, Bayesian, temporal and bipartite ERGMs.
:::

:::{grid-item-card} {fas}`book` User guide
:link: user-guide/index
:link-type: doc

Networks, formulas, fitting, curved models, constraints, missing ties,
multilevel and bipartite networks, samples of networks, networks over time,
diagnostics, goodness of fit, model comparison, interpreting and reporting
results, simulation, and saving fits.
:::

:::{grid-item-card} {fas}`list` Term reference
:link: terms
:link-type: doc

The 189 terms and 25 operators, their statistics and their names.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api
:link-type: doc

Every function and class, its parameters and what it returns.
:::

:::{grid-item-card} {fab}`r-project` Coming from R
:link: coming-from-r
:link-type: doc

ergm and statnet functions and their ergmx equivalents.
:::

:::{grid-item-card} {fas}`check` Validation
:link: validation
:link-type: doc

How ergmx's results compare with R's ergm, and how fast it is.
:::

:::{grid-item-card} {fas}`clock-rotate-left` Changelog
:link: changelog
:link-type: doc

What changed in each version.
:::
::::

```{toctree}
:hidden:

user-guide/index
terms
api
coming-from-r
validation
changelog
references
```
