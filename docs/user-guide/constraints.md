---
file_format: mystnb
kernelspec:
  name: python3
---

# Constraints

A model's *sample space* is the set of networks it puts probability on: by
default every network on the vertices. Constraints restrict it, for when some
networks could not have been observed, or to condition on some features of
the network. They are given as in R's ergm:

```python
ergmx.ergm(network, formula, constraints="bd(maxout=4)")
ergmx.ergm(network, formula, constraints="~degrees")              # a leading ~ is fine
ergmx.ergm(network, formula, constraints="bd(maxout=4) + blocks('level', levels2=2)")
```

| Constraint | Networks allowed |
|---|---|
| `bd(maxout=, maxin=, minout=, minin=)` | Degrees within bounds (one value, or one per vertex). Undirected networks use `minout` and `maxout` for degrees. |
| `blocks(attr, levels=, levels2=)` | The dyads of some mixing types of `attr` are fixed at their observed values: those `nodemix(attr, levels, levels2)` would count. `levels2` selects them as for [nodemix](../terms.md#attribute-terms); by default none. |
| `degrees` | Every vertex keeps its degree (its in- and out-degrees, if directed). |
| `odegrees`, `idegrees` | Every vertex keeps its out-degree, or its in-degree (directed networks). |

## Bounded degrees: fixed-choice designs

In Sampson's monastery, each monk named the three brothers he liked most (four
when he couldn't choose). No network with more than four nominations per
monk could have been observed, so the model shouldn't put probability on any:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

samplk3 = datasets.load("samplk3")
bounded = ergmx.ergm(samplk3, "edges + mutual", constraints="bd(maxout=4)", seed=1)
bounded.summary()
```

Without the bound, the model explains the low density with a large negative
`edges` coefficient; with it, much of the sparsity comes from the design,
and `edges` is closer to zero (compare R's ergm: −1.64 and 2.32).

## Fixing blocks of dyads

`blocks` fixes the dyads of chosen mixing types. With `levels2=-1`, every
mixing type but the first is fixed, so only the ties among 7th graders are
modeled; the dyads between them and everyone else keep their observed
(absent) values:

```{code-cell} ipython3
mesa = datasets.load("faux.mesa.high")
fit = ergmx.ergm(mesa, "edges + nodematch('Race')", constraints="blocks('Grade', levels2=-1)")
fit.summary()
```

This model is dyad-independent and `blocks` fixes dyads independently of
each other, so the fit is still an exact logistic regression, over the 1,891
pairs of 7th graders.

## Preserving degrees

Conditioning on the degrees asks whether a network has more triangles, or
more ties within groups, than networks with the same degrees: degree
heterogeneity alone can produce clustering.

```{code-cell} ipython3
fit = ergmx.ergm(mesa, "nodematch('Grade') + gwesp(0.5, fixed=TRUE)", constraints="degrees",
                 seed=1)
fit.summary()
```

Same-grade ties and triadic closure remain strong given the degrees. Terms
whose statistics the constraint keeps constant, such as `edges`, `kstar`,
`gwdegree` or `nodefactor` under `degrees`, can't be estimated: `ergmx`
warns, fixes their coefficients at 0 and reports them as constant.

The MCMC uses moves that keep the degrees: it swaps the endpoints of two
ties and, in directed networks, also reverses cyclic triples ($i \to j \to k
\to i$), without which some networks with the same degrees can't reach each
other.

## What changes with dyad-dependent constraints

`bd` and the degree constraints are *dyad-dependent*: whether a dyad may
change depends on the others. Then:

- the MPLE ignores the constraint, so `estimate="MPLE"` is not available, and
  the Monte Carlo MLE starts from the contrastive divergence estimate, which
  respects it (`init="CD"`), as in ergm;
- dyad-independent models still need the Monte Carlo MLE;
- the log-likelihood is relative to the null model, the uniform distribution
  over the allowed networks, as in ergm. The summary says so. Relative
  log-likelihoods, AICs and BICs compare models with the same constraints,
  and {func}`ergmx.compare` refuses models with different ones.

{func}`ergmx.simulate` and {func}`ergmx.gof` take the same `constraints`, and
a fit's simulations and goodness of fit use its own.
