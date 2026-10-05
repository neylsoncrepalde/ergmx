---
file_format: mystnb
kernelspec:
  name: python3
---

# Bipartite networks

A *bipartite* (two-mode) network has two kinds of vertices, with ties only
between kinds: people and the groups they belong to, firms and the boards
their directors sit on, authors and papers. [Davis, Gardner and Gardner's
(1941)](https://doi.org/10.7208/chicago/9780226817996.001.0001) Southern Women is the classic one: 18 women and the 14 social events each
attended.

```{code-cell} ipython3
import ergmx
from ergmx import datasets

davis = datasets.load("davis")
print(datasets.describe("davis"))
```

[A complete bipartite ERGM analysis](quickstart/bipartite.md) goes from the
attendance matrix to a table of results, with the networks the two modes
induce on each other.

## Declaring the modes

`ergmx` reads each vertex's mode from a vertex attribute: false (or 0) for the
first mode, true (or 1) for the second. Pass its name as `bipartite=`, or
`bipartite=True` for igraph's `type` or networkx's `bipartite` attribute,
which their bipartite functions create:

```{code-cell} ipython3
ergmx.summary_stats(davis, "edges + b1star(2) + b2star(2) + cycle(4)", bipartite=True)
```

`b1star(2)` counts the pairs of events each woman attended, `b2star(2)` the
pairs of women at each event, and `cycle(4)` the pairs of women who attended
the same two events. In R's ergm, the first mode is the first `bipartite`
vertices of the network; in `ergmx` the vertices can be in any order.

## Fitting

Only ties between the modes are modeled: the dyads within a mode are fixed,
and the sample size of BIC is the 18 x 14 = 252 pairs of a woman and an event.
Do women who attended many of the same events attend more together, beyond
what their activity explains?

```{code-cell} ipython3
fit = ergmx.ergm(davis, "edges + gwb1dsp(0.5, fixed=TRUE)", bipartite=True, seed=1)
fit.summary()
```

The positive `gwb1dsp` coefficient says yes: pairs of women tend to share
several events. R's ergm gives −1.25 and 0.33 for this model.

The terms of bipartite networks are in the
[term reference](../terms.md#bipartite-terms): stars, degrees, factors,
covariates and homophily (`b1nodematch`) for each mode, and their shared
partners. `edgecov` takes a first-mode by second-mode matrix, as in ergm.

## Projections

ergm's `Proj1()` and `Proj2()` (or `Project(formula, mode)`) evaluate
[valued terms](../terms.md#valued-terms) on the projection of the network onto
a mode: the network of the women whose value for a pair is the number of
events they both attended. `Proj1(~sum + nonzero)` is the women's
co-attendances, and the pairs that attended any event together:

```{code-cell} ipython3
ergmx.summary_stats(davis, "Proj1(~sum + nonzero) + Proj2(~sum + nonzero + atleast(2))", bipartite=True)
```

They are dyad-dependent terms of the bipartite model: with them,
`edges + Proj1(~nonzero)` asks whether attendances spread over more pairs of
women than chance would.

## Goodness of fit

For bipartite networks, `gof()` compares, as ergm does, the degrees of each
mode, the dyadwise shared partners and the geodesic distances:

```{code-cell} ipython3
fit.gof(seed=1).plot();
```

Simulation, constraints, missing ties and the other features work as for
other networks; pass `bipartite=` to {func}`ergmx.simulate` and
{func}`ergmx.gof` when you give them a network rather than a fit.
