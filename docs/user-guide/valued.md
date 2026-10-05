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
`response`, the reference as `reference`, and ergm's valued terms. Values
that aren't counts take ergm's continuous references
([](#continuous-values)).

```{code-cell} ipython3
import ergmx
from ergmx import datasets

zach = datasets.load("zach")  # Zachary's karate club, with counts of contexts
ergmx.summary_stats(zach, "sum + nonzero + nodematch('faction.id')", response="contexts")
```

[A complete valued ERGM analysis](quickstart/valued.md) goes from the data
to a table of results.

## Reference measures

| `reference` | Values | The model with no other terms |
|---|---|---|
| `"Poisson"` | 0, 1, 2... | each dyad's count is Poisson; `sum`'s coefficient is the log of its mean |
| `"Binomial(trials)"` | 0 to `trials` | each count is binomial; `sum`'s coefficient is the log-odds of each trial |
| `"Geometric"` | 0, 1, 2... | each count is geometric (the reference can't be normalized, so there is no log-likelihood) |
| `"DiscUnif(a, b)"` | `a` to `b` | uniform; with terms, any distribution on those values |
| `"StdNormal"` | any real number | each value is standard normal; with `sum` and `sum(pow=2)`, normal with any mean and variance |
| `"Unif(a, b)"` | any number from `a` to `b` | uniform; with `sum`, an exponential truncated to [a, b] |

Values of zero are dyads without an edge; the network's edges carry the
nonzero values. The values must be in the reference's support: whole
numbers for the first four.

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
  each vertex's dyads, a measure of heterogeneity in sociality; like
  `kstar`, it can make a model degenerate (with the karate club's other
  terms, its networks drift to ones where the leaders interact with nearly
  everyone; R's ergm reports a fit in that case, from chains that hadn't
  left the observed network yet, and ergmx a {class}`~ergmx.DegeneracyError`);
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
in the edge attribute `response`. {func}`ergmx.san` searches for a valued
network with given statistics, and `ergm(..., target_stats=)` fits a valued
model to statistics rather than to a network
([](simulation.md#networks-with-given-statistics)). Without the values, the
log-likelihood can't be computed for the Poisson, binomial and normal
references, whose densities depend on them.

## Goodness of fit

{meth}`fit.gof() <ergmx.ErgmFit.gof>`, as ergm's `gof()` of a valued
model, compares the model statistics and the distribution of the values
with those of simulated networks. The distribution (`"cdf"`) is cumulative:
the number of dyads whose value is at most each of a range of values, from
the smallest nonzero value to the largest, widened by a tenth of their
range, plotted as proportions of the dyads:

```{code-cell} ipython3
fit.gof(seed=1).plot();
```

The model reproduces its statistics and the number of pairs who never
meet, but its networks have about three times as many pairs who meet in a
single context as the club (18 against 6). The other
statistics of {func}`ergmx.gof` (`stats=["degree", "espartners",
"distance"]`) are those of the nonzero dyads as ties, which ergm's gof
doesn't compute for valued networks.

## Continuous values

Values that aren't counts, such as standardized trade volumes or
correlations, take a continuous reference (ergm's): `"StdNormal"` or
`"Unif(a, b)"`. With `"StdNormal"`, `sum` and `sum(pow=2)`, with
coefficients θ₁ and θ₂ < 1/2, make each dyad's value normal with variance
1/(1 - 2θ₂) and mean θ₁ times that variance. Other terms shift the mean
(`nodematch(..., form="sum")`, `absdiff`), or relate the values, as
`mutual(form="product")`, which correlates the two values of each pair of
a directed network.

For example, values simulated for the 16 Florentine families from a model
where families of similar wealth have higher values, and the model fitted
to them:

```{code-cell} ipython3
flo = datasets.load("flomarriage")
flo.delete_edges(flo.es)  # start from values of 0
formula = "sum + sum(pow=2) + absdiff('wealth')"
families = ergmx.simulate(flo, formula, [0.6, -1.0, -0.01], seed=1, response="trade",
                          reference="StdNormal", burnin=200000)[0]
continuous = ergmx.ergm(families, formula, response="trade", reference="StdNormal", seed=1)
continuous.summary()
```

The estimates are within a standard error of the coefficients that
simulated the values. The MCMC proposes, for a random dyad, its value plus
a normal step, with a standard deviation of 0.2 (ergm's default;
`Control(normal_sd=)` changes it), so values on a much larger scale than
the standard normal's mix slowly: standardize them, or take larger steps.
With `"Unif(a, b)"`, the MCMC proposes a value uniformly in [a, b]. Terms
with square roots or log-factorials (`sum(pow=0.5)`, `nodesqrtcovar`,
`CMP`, geometric means) need nonnegative values, so not `"StdNormal"`.

## Missing dyads

A dyad whose value is unknown is an edge with a true `na` attribute, as in
binary networks ([Missing ties](missing-data.md)), or an edge without a
value (`None` or NaN). The MLE is that of the observed dyads, with the
missing ones' values imputed from the model at each iteration (they start
at 0, or at the reference's value nearest 0), as ergm's:

```{code-cell} ipython3
import numpy as np

unknown = zach.copy()
rng = np.random.default_rng(1)
for e in rng.choice(unknown.ecount(), 10, replace=False):  # 10 of the pairs who interact
    unknown.es[int(e)]["contexts"] = None
for i, j in [(0, 9), (2, 30), (5, 20)]:  # and three who don't
    unknown.add_edge(i, j, na=True)
partial = ergmx.ergm(unknown, "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')",
                     response="contexts", reference="Poisson", seed=1)
partial.summary()
```

The estimates are close to those of the complete network, with somewhat
larger standard errors. The log-likelihood is that of the observed dyads, as
is the BIC's sample size; the goodness of fit's observed values average
networks with the missing values imputed.

## Several valued networks

Valued networks combined with {func}`ergmx.Networks` take `N()` of valued
terms, as in ergm.multi: each network's statistics on its own values,
combined through `N()`'s linear model. Here, the karate club's counts and
the same counts less one (at least 1):

```{code-cell} ipython3
fewer = zach.copy()
fewer.es["contexts"] = [max(1, v - 1) for v in zach.es["contexts"]]
clubs = ergmx.Networks(zach, fewer)
fit = ergmx.ergm(clubs, "N(~sum + nonzero, ~.NetworkID)", response="contexts", reference="Poisson", seed=1)
print(fit.summary())
```

Networks simulated from such a model are lists of graphs, one per network.

## What is not supported

Not available for valued networks: dyad-dependent constraints, series of
networks (`NetSeries()`), and `N()` with offsets.
