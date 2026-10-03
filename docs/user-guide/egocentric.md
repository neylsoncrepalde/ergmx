---
file_format: mystnb
kernelspec:
  name: python3
---

# Egocentric data

Surveys often observe a network through a sample of *egos*: each respondent
names their contacts (*alters*), reports their attributes, and perhaps which
of them know each other. {func}`ergmx.ergm_ego`, as R's ergm.ego ([Krivitsky
and Morris 2017](https://doi.org/10.1214/16-AOAS1010)), fits an ERGM of the
population network from such data:

1. each ego's contribution to each statistic (half its ties, half its ties
   to alters of its own group, its degree...) averaged over the egos,
   weighted by their sampling weights, estimates the statistics of the
   population, per person;
2. the model is fitted to those statistics, scaled to a *pseudo-population*
   network of replicates of the egos, with {func}`ergmx.ergm`'s
   `target_stats`;
3. an offset on `edges` (and on `transitiveties`, with triadic terms) of
   -log(pseudo-population / population size) gives the coefficients of a
   network of the population's size: with `popsize=1`, the default, *per
   capita* coefficients, which don't depend on the population size;
4. the standard errors come from the sampling variance of the egos'
   contributions, through the model.

## The data

{class}`ergmx.EgoData` holds the egos (one row each), their alters (one row
each, with the ego who named them) and, optionally, the ties among each
ego's alters, as pandas DataFrames, dicts of columns or lists of records,
with egor's column names by default (`.egoID`, `.altID`, `.srcID`,
`.tgtID`). `EgoData.from_network()` takes a whole network's egocentric
census, as egor's `as.egor()`, and `sample()` some of its egos:

```{code-cell} ipython3
import ergmx
from ergmx import datasets
import numpy as np

mesa = datasets.load("faux.mesa.high")
census = ergmx.EgoData.from_network(mesa)
rng = np.random.default_rng(1)
sample = census.sample(rng.choice(census.n, 100, replace=False))
sample
```

## The statistics

{func}`ergmx.ego_stats`, as R's `summary(egor ~ ...)`, estimates the
population's statistics, scaled to `scaleto` people (the number of egos by
default), with their covariance. On the census, they are the network's
statistics; on the sample, estimates:

```{code-cell} ipython3
formula = "edges + nodematch('Grade') + degree(0) + gwesp(0.5, fixed=TRUE)"
stats, cov = ergmx.ego_stats(formula, sample, scaleto=mesa.vcount())
{name: (round(value, 1), round(float(se), 1)) for (name, value), se in
 zip(stats.items(), np.sqrt(np.diag(cov)))}, ergmx.summary_stats(mesa, formula)
```

The terms ergmx estimates from egocentric data are ergm.ego's: `edges`,
`nodecov`, `nodefactor`, `nodematch`, `nodemix`, `absdiff`, `absdiffcat`,
`degree` (also by an attribute, and with `homophily`), `degrange`,
`concurrent`, `concurrentties`, `degree1.5`, `gwdegree`, `meandeg`, and, with
the ties among alters, `esp`, `gwesp`, `transitiveties` and `triangle`.
Attributes of alters must have the levels the egos have. The covariance is
the survey linearization's by default; `stats_est` also takes ergm.ego's
`"asymptotic"`, `"naive"`, `"bootstrap"` and `"jackknife"`.

## The model

```{code-cell} ipython3
fit = ergmx.ergm_ego("edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)", sample,
                     popsize=mesa.vcount(), seed=1)
fit.summary()
```

With `popsize`, the coefficients are those of a network of that many
people, comparable with a fit of the whole network (`edges` near -6.4 for
faux.mesa.high's 205 students). The pseudo-population has as many people
as the population (or the egos, without `popsize`), times `ppopsize_mul`,
each ego replicated in proportion to its weight. {meth}`EgoFit.simulate()
<ergmx.EgoFit.simulate>` and {meth}`EgoFit.gof() <ergmx.EgoFit.gof>` work on
the pseudo-population.

ergm.ego fits even dyad-independent models by Monte Carlo MLE; ergmx fits
them exactly, so their estimates differ from R's by R's Monte Carlo error.
The standard errors use the model's exact information (ergm.ego's, its
Monte Carlo estimate, which the sandwich amplifies: up to 15% apart on a
sample of 100 egos).
