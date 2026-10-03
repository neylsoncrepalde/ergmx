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
| `bd(attribs=, maxout=, ...)` | Bounds by the alters' classes: `attribs` (a graph attribute, or a vertices x classes logical matrix) marks each vertex's classes, and the bounds (same shape) limit each vertex's ties to alters of each class. |
| `blockdiag(attr)` | Ties only between vertices with the same value of `attr`: the dyads between blocks are fixed at no tie. |
| `Dyads(fix=~terms, vary=~terms)` | With `fix`, the dyads that the dyad-independent terms count are fixed; with `vary`, only those may vary; with both, the dyads either lets vary. |
| `fixedas(fixed.dyads=, present=, absent=)` | These dyads are fixed (`present` checked to be ties, `absent` non-ties). |
| `fixallbut(free.dyads)` | Every dyad but these is fixed. |
| `observed` | The observed dyads are fixed: only the missing ones vary, to simulate them. |
| `degrees` | Every vertex keeps its degree (its in- and out-degrees, if directed). |
| `odegrees`, `idegrees` | Every vertex keeps its out-degree, or its in-degree (directed networks). |
| `b1degrees`, `b2degrees` | The vertices of the first (second) mode of a bipartite network keep their degrees. |
| `degreedist` | The degree distribution is kept (the in- and out-degree distributions, if directed; each mode's, if bipartite): how many vertices have each degree, not which. |
| `odegreedist`, `idegreedist` | The out-degree, or in-degree, distribution is kept (directed networks). |
| `edges` | The number of edges is kept. |
| `egocentric(attr=, direction='both')` | The dyads of the vertices whose `attr` (a logical vertex attribute) is true are fixed: those with such a vertex at either end, or, in directed networks, those they send (`'out'`) or receive (`'in'`). Without `attr`, of the vertices whose `na` attribute is false. |

Dyads are given as an edge list of R's vertex numbers, from 1
(`matrix(c(1, 9, 2, 6), ncol=2, byrow=TRUE)`, or a list of pairs), a
logical $n \times n$ matrix, a graph, or the name of a graph attribute
holding one. R's argument names with dots (`fixed.dyads`) work in strings;
in Python they have underscores.

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
other. Under `b1degrees` (`b2degrees`) it moves the other end of a tie to
another vertex of that mode, and under `edges` it swaps a tie for a non-tie.

`degreedist` conditions on less: networks whose vertices have the same
degrees between them, in any arrangement. Besides the moves of `degrees`,
the MCMC moves an end of a tie from a vertex to one with one tie fewer, so
that the two swap degrees (in directed networks, the head of a tie to swap
in-degrees, or its tail to swap out-degrees). Statistics of the degree
distribution alone (`edges`, `kstar`, `degree`, `gwdegree`, `isolates`...)
are constant, but not those of particular vertices' degrees (`sociality`,
`nodefactor`). That is how ergm documents `degreedist`, `idegreedist` and
`odegreedist`; ergm 4.12's MCMC keeps every vertex's out-degree when it swaps
in-degrees and the reverse, so its `odegreedist` and `idegreedist` keep the
out- or in-degrees themselves, as `odegrees` and `idegrees`.

## What changes with dyad-dependent constraints

`bd`, `edges` and the degree and degree distribution constraints are *dyad-dependent*: whether a
dyad may change depends on the others. Then:

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
