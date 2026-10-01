---
file_format: mystnb
kernelspec:
  name: python3
---

# Multilevel networks

A multilevel network has vertices at several levels, such as individuals and
the organizations they belong to, with ties within each level and ties across
levels (Lazega and Snijders 2016). In `ergmx`, it is one network with a vertex
attribute for the level, and the model separates what happens within and
across levels with three tools:

- [`nodemix`](../terms.md#attribute-terms) counts ties by mixing type: one
  density per kind of tie;
- the [`F()` operator](../terms.md#operators) evaluates terms on the ties that
  pass a filter, such as the ties within levels;
- the [`blocks` constraint](constraints.md) fixes the dyads of some mixing
  types, to model some kinds of ties given the others.

`linked_sim`, from the [multinets](https://github.com/neylsoncrepalde/multinets)
package, has 100 individuals and 50 organizations:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

linked = datasets.load("linked_sim")
ergmx.summary_stats(linked, "nodemix('level', levels2=TRUE)")
```

## Densities by kind of tie

`nodemix('level', levels2=TRUE)` has one statistic per mixing type, so
without an `edges` term each coefficient is the log-odds of a tie of that
kind:

```{code-cell} ipython3
densities = ergmx.ergm(linked, "nodemix('level', levels2=TRUE)")
densities.summary()
```

Affiliation ties are rarer, relative to the number of pairs, than ties within
either level: about 4% of individual–organization pairs are tied, against 6%
of pairs of individuals and 7% of pairs of organizations.

## Closure within levels

Do ties within a level close triangles? `F(~gwesp(0.5, fixed=TRUE),
~nodematch('level'))` computes gwesp on the network of the ties within
levels only:

```{code-cell} ipython3
closure = ergmx.ergm(
    linked,
    "nodemix('level', levels2=TRUE) + F(~gwesp(0.5, fixed=TRUE), ~nodematch('level'))",
    seed=1,
)
closure.summary()
```

```{code-cell} ipython3
ergmx.compare(densities, closure)
```

Not in this simulated network: the coefficient is close to 0, and the
comparison prefers the model without it.

The filter is any dyad-independent term with one statistic; a tie passes if
adding it would change that statistic. `~!nodematch('level')` keeps the
other ties, here the affiliations.

## Given the affiliations

To model the ties within levels taking the affiliations as given, fix the
individual–organization dyads with `blocks`. Mixing types are numbered as
`nodemix` orders them: (individual, individual), (individual, organization),
(organization, organization), so the affiliations are type 2:

```{code-cell} ipython3
within = ergmx.ergm(
    linked,
    "nodemix('level', levels2=c(1, 3)) + F(~gwesp(0.5, fixed=TRUE), ~nodematch('level'))",
    constraints="blocks('level', levels2=2)",
    seed=1,
)
within.summary()
```

`blocks` is dyad-independent, so the log-likelihood is still absolute, over
the free dyads, and BIC counts only those.

## Fixed coefficients

`offset()` fixes a coefficient rather than estimating it, for example to
compare networks of different sizes with a density adjusted for the
number of vertices (Krivitsky, Handcock and Morris 2011), or to forbid a
kind of tie with a coefficient of `-inf`:

```{code-cell} ipython3
# The same network without ties between organizations.
organizations = set(linked.vs.select(level="organization").indices)
without = linked.copy()
without.delete_edges([e for e in without.es if {e.source, e.target} <= organizations])

forbidden = ergmx.ergm(
    without,
    "nodemix('level', levels2=c(1, 2)) + offset(nodemix('level', levels2=3))",
    offset_coef=[float("-inf")],
)
forbidden.summary()
```

With the `-inf` offset, ties between organizations are impossible: the
model is fitted on the other dyads, and its simulated networks never have
them.
