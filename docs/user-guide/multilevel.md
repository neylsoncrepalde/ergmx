---
file_format: mystnb
kernelspec:
  name: python3
---

# Multilevel networks

A multilevel network has vertices at several levels, such as researchers and
the laboratories they belong to, with ties within each level (advice among
researchers, collaboration among laboratories) and ties across levels (the
affiliations) ([Lazega and Snijders 2016](https://doi.org/10.1007/978-3-319-24520-1)). In `ergmx` it is one network with a
vertex attribute for the level. [Wang, Robins, Pattison and Lazega (2013)](https://doi.org/10.1016/j.socnet.2013.01.004)
proposed ERGMs for such two-level networks, as their program MPNet fits them:
a level A, a level B, the A-ties within A, the B-ties within B, and the X-ties
between them, with effects for each network and for how they depend on each
other.

`linked_sim`, from the [multinets](https://github.com/neylsoncrepalde/multinets)
package, has 100 individuals and 50 organizations:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

linked = datasets.load("linked_sim")
ergmx.summary_stats(linked, "nodemix('level', levels2=TRUE)")
```

## Each level, and the ties between them

The operator {func}`~ergmx.S`, as ergm's, evaluates terms on a subgraph. With
one set of vertices, `~level == 'individual'`, it is the network within that
level, so any ergm term describes it; with two, `(level == 'individual') ~
(level == 'organization')`, it is the bipartite network of the ties between
them, whose `b1` terms are about the first set:

```{code-cell} ipython3
ergmx.summary_stats(
    linked,
    "S(~edges + gwesp(0.693147, fixed=TRUE), ~level == 'individual') "
    "+ S(~edges + b1star(2) + b2star(2), (level == 'individual') ~ (level == 'organization'))",
)
```

So MPNet's effects within each level and within the affiliation network are
ergm terms inside `S()`, with MPNet's lambda as `exp(decay)` (MPNet's default
lambda = 2 is a decay of log 2 = 0.693147):

| MPNet | `ergmx`, inside `S(~..., ~level == A)` (or B) |
|---|---|
| EdgeA, Star2A, TriangleA | `edges`, `kstar(2)`, `triangle` |
| ATA (alternating triangles) | `gwesp(log(lambda), fixed=TRUE)`, the same statistic |
| A2PA (alternating two-paths) | `gwdsp(log(lambda), fixed=TRUE)`, the same statistic |
| ASA (alternating stars) | `gwdegree(log(lambda), fixed=TRUE)`: with `edges`, the same model |

| MPNet | `ergmx`, inside `S(~..., (level == A) ~ (level == B))` |
|---|---|
| XEdge, XStar2A, XStar2B | `edges`, `b1star(2)`, `b2star(2)` |
| XASA, XASB (alternating stars) | `gwb1degree`, `gwb2degree`: with `edges`, the same model |
| XC4 (four-cycles) | `cycle(4)` |
| XACA, XACB (alternating two-paths) | `gwb1dsp`, `gwb2dsp` |

## How the levels depend on each other

The configurations that join ties of different kinds are MPNet's, written as
`ergmx` terms whose first argument is the level attribute, A and B being its
two values in sorted order (or `levels=(A, B)`). In [Wang et al.'s](https://doi.org/10.1016/j.socnet.2013.01.004)
interpretation, with laboratories as A and researchers as B:

| Term | Configuration | Interpretation |
|---|---|---|
| `star2ax`, `star2bx` | a vertex with a within-level tie and an affiliation | affiliation-based popularity: those active within their level have more affiliations |
| `axs1a`, `aas1x`, `aaaxs` (and `b`) | the same, alternating in one or both kinds of tie | the same, attenuated |
| `txax`, `txbx` | a within-level tie whose ends share an affiliation | affiliation-based closure: researchers of the same laboratory seek each other's advice |
| `atxax`, `atxbx` | the same, alternating in the shared affiliations | the same, attenuated |
| `l3xax`, `l3xbx` | an affiliation, a within-level tie, an affiliation | meso-level popularity and within-level activity |
| `l3axb` | an A-tie, an affiliation, a B-tie | assortativity of activity across levels: active researchers belong to active laboratories |
| `c4axb` | an A-tie, a B-tie and the two affiliations that join their ends | cross-level alignment: members of collaborating laboratories seek each other's advice |

The term reference gives each one's statistic. They're for undirected
networks; MPNet's directed variants are not available yet. A model of the
three networks and their interdependence:

```{code-cell} ipython3
mpnet = ergmx.ergm(
    linked,
    "S(~edges + gwesp(0.693147, fixed=TRUE), ~level == 'individual') "
    "+ S(~edges + gwesp(0.693147, fixed=TRUE), ~level == 'organization') "
    "+ S(~edges + gwb1degree(0.693147, fixed=TRUE), (level == 'individual') ~ (level == 'organization')) "
    "+ star2ax('level') + star2bx('level') + txax('level') + txbx('level') + c4axb('level')",
    seed=1,
)
mpnet.summary()
```

In this simulated network, the cross-level effects are small: only the
individuals' within-level activity is slightly negatively associated with
their affiliations (`Star2AX`), and there is no cross-level alignment
(`C4AXB`). The tools below model the same network by kinds of tie.

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
number of vertices ([Krivitsky, Handcock and Morris 2011](https://doi.org/10.1016/j.stamet.2011.01.005)), or to forbid a
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
