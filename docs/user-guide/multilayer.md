---
file_format: mystnb
kernelspec:
  name: python3
---

# Multilayer networks

Actors are often tied by several relations at once: families by marriage and
by business, colleagues by advice and by friendship, countries by trade and
by alliances. Fitting one ERGM per relation leaves out how the relations
depend on each other, such as whether families tied by marriage also tend to
do business together. A multilayer ERGM, as R's
[ergm.multi](https://github.com/statnet/ergm.multi) fits ([Krivitsky, Koehly
and Marcum 2020](https://doi.org/10.1007/s11336-020-09720-7)), models the
relations jointly. Each relation is a layer of one network, and its terms can
involve several layers.

## Layer()

{func}`ergmx.Layer` combines networks on the same vertices into the layers
of one network. They can be passed as arguments (layers `1`, `2`...) or by
name. The Florentine families' marriages and business ties
([Padgett and Ansell 1993](https://doi.org/10.1086/230190)) are:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

flo = ergmx.Layer(marriage=datasets.load("flomarriage"),
                  business=datasets.load("flobusiness"))
flo
```

The layers are the blocks of one larger network, made of a copy of the
families per layer. Ties form only within a layer, and every layer has the
first one's vertex attributes. Terms outside `L()` see that combined network,
so `edges` counts the ties of every layer and `triangle` the triangles within
any one layer.

## L() and Layer Logic

`L(formula, Ls)` evaluates a formula's terms on a *logical layer*: one layer,
or an expression of several, in R's syntax:

- the logical operators `&`, `|`, `!` and `xor()`;
- comparisons, and the arithmetic `+ - * / %% ^`;
- `abs()`, `sign()` and `round()`.

A dyad of a logical layer has a tie where its expression is nonzero (TRUE),
evaluated on the dyad's ties in each layer (1 or 0):

```{code-cell} ipython3
ergmx.summary_stats(flo, "edges + L(~edges, ~marriage) + L(~edges, ~business) + "
                         "L(~edges, ~marriage & business) + L(~edges, ~marriage | business) + "
                         "L(~triangle, ~(marriage + business) >= 1)")
```

`Ls` can also be a list of logical layers, whose statistics are summed, each
weighted by the left side of its formula: `L(~edges, c(2 ~ marriage, -1 ~
business))`. Without `Ls` (or with `~.`), the terms are summed over the
layers. Unnamed layers are numbers in backticks: `` ~`1` & `2` ``. In
directed networks, `t()` transposes a layer: `` ~`1` & t(`2`) `` has a tie
i → j where layer 1 has i → j and layer 2 has j → i. As in ergm.multi, a
logical layer can't have ties where no layer has one (as `~!marriage` would).

## A model of multiplexity

The density of each layer, and how much likelier a pair is to have both
ties:

```{code-cell} ipython3
fit = ergmx.ergm(flo, "L(~edges, ~marriage) + L(~edges, ~business) + L(~edges, ~marriage & business)",
                 seed=1)
print(fit.summary())
```

The two ties of a pair depend on each other, so the model is fitted by MCMC,
as ergm.multi fits it. Each pair's two ties are four possible states, and the
third coefficient is the log of their odds ratio: a marriage multiplies the
odds of a business tie by $e^{2.19} \approx 9$, and vice versa.

ergm.multi's `CMBL` term models the same dependence for any number of layers:
its statistic is the sum, over the dyads, of $\log\{E!(R-E)!/R!\}$, where $E$
is the number of the $R$ layers with a tie. A positive coefficient makes the
layers agree. With two layers, it is the previous model reparametrized, with
the same log-likelihood:

```{code-cell} ipython3
cmb = ergmx.ergm(flo, "L(~edges, ~marriage) + L(~edges, ~business) + CMBL", seed=1)
print(cmb.summary())
```

## Layer-aware terms

ergm.multi's terms of several layers:

- `twostarL(Ls, type)`: two-stars whose two ties are in two logical layers;
- `mutualL(Ls=)`: reciprocity between two logical layers, in directed
  networks;
- the shared partner terms `espL`, `dspL`, `nspL` and their geometrically
  weighted versions `gwespL`, `gwdspL` and `gwnspL`. A partner's two ties
  are in the layers `Ls.path` (in that order with `L.in_order=TRUE`). The
  edges are those of the layer `L.base`.

```{code-cell} ipython3
ergmx.summary_stats(flo, "twostarL(c(~marriage, ~business)) + "
                         "espL(1:2, L.base = ~marriage, Ls.path = ~business) + "
                         "gwespL(0.5, fixed = TRUE, L.base = ~marriage, Ls.path = c(~marriage, ~business))")
```

`espL(1, L.base = ~marriage, Ls.path = ~business)` counts the marriages whose
two families do business with exactly one common partner. See
[](../terms.md) for all of them.

## Simulating

Networks simulated from a Layer() network are dicts of their layers, by
name, which `Layer()` takes back:

```{code-cell} ipython3
draws = fit.simulate(3, seed=2)
[ergmx.summary_stats(ergmx.Layer(d), "L(~edges, ~marriage & business)") for d in draws]
```

## Differences from ergm.multi

The statistics and names are ergm.multi's, and fits agree with its fits
within their Monte Carlo error (see [](../validation.md)). There are three
differences:

- ergm.multi 0.3.0's OSP and ISP shared partners with `L.in_order=TRUE`
  depend on the order its ties were added in. Its cache of shared partners
  forgets which tie is at which end of the pair. ergmx follows ergm.multi's
  documentation, where the first tie is the one at the pair's first vertex
  (the base tie's tail), and warns with `ErgmDifferenceWarning`.
- `L()` of curved terms, and the gw layer terms with an estimated decay, are
  not supported yet: fix the decay.
- As in ergm.multi, `mutualL()` needs `Ls`; without it, use `mutual`.
