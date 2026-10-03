---
file_format: mystnb
kernelspec:
  name: python3
---

# Valued networks

Some ties come with counts: the number of contexts in which two people meet,
of messages, of times one monk named another. R's ergm, with ergm.count
([Krivitsky 2012](https://doi.org/10.1214/12-EJS696)), models them by giving
each dyad a value drawn from a *reference measure*, tilted by the model's
statistics. `ergmx` does the same: give the edge attribute with the counts as
`response`, the reference as `reference`, and ergm's valued terms.

```{code-cell} ipython3
import ergmx
from ergmx import datasets

zach = datasets.load("zach")  # Zachary's karate club, with counts of contexts
ergmx.summary_stats(zach, "sum + nonzero + nodematch('faction.id')", response="contexts")
```

## Reference measures

| `reference` | Values | The model with no other terms |
|---|---|---|
| `"Poisson"` | 0, 1, 2... | each dyad's count is Poisson; `sum`'s coefficient is the log of its mean |
| `"Binomial(trials)"` | 0 to `trials` | each count is binomial; `sum`'s coefficient is the log-odds of each trial |
| `"Geometric"` | 0, 1, 2... | each count is geometric (the reference can't be normalized, so there is no log-likelihood) |
| `"DiscUnif(a, b)"` | `a` to `b` | uniform; with terms, any distribution on those values |

Counts of zero are dyads without an edge; the network's edges carry the
nonzero counts.

## Terms

The valued terms are ergm's:

- `sum` (and `sum(pow=)`), the sum of the values, and `nonzero`, the number of
  nonzero dyads, which together model zero inflation;
- dyad-independent binary terms with `form="sum"` (the default: each dyad's
  statistic weighted by its value) or `form="nonzero"`: `nodematch`,
  `nodemix`, `nodefactor`, `nodecov`, `absdiff`, `absdiffcat`, `edgecov`,
  `attrcov`, `diff`, `sociality`, `sender`, `receiver`, `mm` and the
  bipartite `b1cov`, `b1factor`...; their names say the form,
  `nodematch.sum.faction.id`;
- thresholds: `atleast`, `atmost`, `greaterthan`, `smallerthan`, `equalto`,
  `ininterval`;
- `mutual(form=)` (directed): reciprocity of values, by their minimum,
  absolute difference, product or geometric mean;
- `transitiveweights` and `cyclicalweights` (Krivitsky 2012, eq. 13), each
  value capped by its strongest two-path, and `transitiveties(threshold=)`;
- `nodecovar` (`nodeocovar`, `nodeicovar`): the covariance of the values of
  each vertex's dyads, a measure of heterogeneity in sociality;
- `CMP`, the sum of log(y!), which turns the Poisson reference into the
  Conway-Maxwell-Poisson, for over- or underdispersion.

## Fitting

Valued models have no pseudo-likelihood: as in ergm, they start from
contrastive divergence (or `init="zeros"`), then run the Monte Carlo MLE.
The MCMC proposes, for a random dyad, a new count near its current one, and,
a fifth of the time, a nonzero dyad's jump to 0 (ergm.count's `DiscTNT`).

```{code-cell} ipython3
fit = ergmx.ergm(zach, "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')",
                 response="contexts", reference="Poisson", seed=1)
fit.summary()
```

The negative `nonzero` with a positive `sum` is zero inflation: more pairs
never meet than a Poisson with this mean would give, and those who meet do
so in several contexts. Members of the same faction meet in more contexts.
The log-likelihood is estimated by path sampling from the reference measure,
as for binary models.

{meth}`fit.simulate() <ergmx.ErgmFit.simulate>` and {func}`ergmx.simulate`
(with `response` and `reference`) return networks with the simulated counts
in the edge attribute `response`.

Not available for valued networks: missing dyads, dyad-dependent constraints,
several networks, goodness of fit and `target_stats`.
