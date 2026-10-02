---
file_format: mystnb
kernelspec:
  name: python3
---

# Missing ties

Network data often have dyads whose value is unknown: survey respondents who
didn't answer, people who left before the second wave, ties that couldn't be
coded. Treating them as absent biases the density, and the rest of the
model, downwards. `ergmx` treats them as R's ergm does.

## Marking missing dyads

A missing dyad is an edge with a true `na` attribute, as in R's network
package (`net[i, j] <- NA`). In igraph:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

samplk3 = datasets.load("samplk3")
# Two monks didn't answer: their nominations are unknown.
nonrespondents = [0, 6]
g = samplk3.copy()
g.delete_edges(g.es.select(_source_in=nonrespondents))
g.es["na"] = False
g.add_edges([(i, j) for i in nonrespondents for j in range(g.vcount()) if j != i],
            attributes={"na": True})
print(g.ecount() - sum(g.es["na"]), "ties,", sum(g.es["na"]), "missing dyads")
```

In networkx: `G.add_edge(u, v, na=True)`. Statistics (`summary_stats`) count
missing dyads as non-ties, as R's `summary()` does.

## Fitting

```{code-cell} ipython3
fit = ergmx.ergm(g, "edges + mutual", seed=1)
fit.summary()
```

R's ergm gives −1.98 and 1.82 for this model, within Monte Carlo error. With
the complete network, the estimates are −2.15 and 2.30, with smaller standard
errors (0.22 and 0.48): the 34 unknown dyads carry no information, so the
two monks' nominations, and how many of them were returned, are inferred from
the model.

## The method

The likelihood is that of the observed dyads, summing over every possible
value of the missing ones ([Handcock and Gile 2010](https://doi.org/10.1214/08-AOAS221)):

$$
L(\theta) = \sum_{y_{mis}} P_\theta(y_{obs}, y_{mis}).
$$

It assumes the dyads are **missing at random**: whether a dyad is missing
doesn't depend on its unobserved value. A student who skipped the survey
because they had few friends breaks the assumption, and no method can tell
from the data alone.

- **Estimating equation.** The MLE solves
  $E_\theta[g(Y) \mid y_{obs}] = E_\theta[g(Y)]$: the expected statistics
  given the observed dyads equal the expected statistics overall. Each Monte
  Carlo iteration draws two samples, one unconditional and one in which only
  the missing dyads change, and takes the Newton step
  $(\Sigma - \Sigma_{obs})\,\delta = \mu_{obs} - \mu$.
- **Standard errors** come from the information $\Sigma - \Sigma_{obs}$ (the
  missing information principle, [Louis 1982](https://doi.org/10.1111/j.2517-6161.1982.tb01203.x)), plus the MCMC error of both
  samples.
- **Dyad-independent models** are still fitted exactly: by logistic
  regression on the observed dyads, as the missing ones factor out.
- **Starting values** are the MPLE on the observed dyads, with missing dyads as
  non-ties in the change statistics. ergm imputes them at random first, so its
  MPLE differs a little between runs; the MLE is the same.
- **The log-likelihood** is that of the observed dyads, by path sampling with
  conditional samples along the path, so AIC, BIC and {func}`ergmx.compare`
  work as usual.

## Goodness of fit

With missing dyads, {func}`ergmx.gof` compares the simulated networks with
networks *imputed* from the model given the observed dyads, averaged, rather
than with the network with missing dyads as non-ties, which would make every
model look like it predicts too many ties:

```{code-cell} ipython3
fit.gof(seed=1)["odegree"]
```

The observed values are averages over the imputations, so they need not be
whole numbers. They show what `edges + mutual` misses: nearly every monk named
three brothers, the fixed-choice design that the constraint `bd(maxout=4)`,
or `odegrees`, represents (see [](constraints.md)).

## Several networks, and waves

In networks combined with {func}`ergmx.Networks`, each network's missing
dyads are missing in the joint model, and {func}`ergmx.gofN` imputes them
too. In a series of networks ({func}`ergmx.tergm`), the missing dyads of the
networks transitioned *to* are missing as here; those of the networks
transitioned *from*, which the next transition is conditional on, are
imputed first with `na_impute=`, as tergm's `NA.impute`: see
[](temporal.md#missing-dyads).

