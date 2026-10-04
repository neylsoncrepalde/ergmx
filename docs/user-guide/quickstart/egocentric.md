---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete egocentric analysis

Often the whole network can't be observed, but a sample of its members can
be surveyed: each respondent, an *ego*, names their contacts, the *alters*,
and describes them. This page fits an ERGM of a whole network from such
data, as R's ergm.ego does ([Krivitsky and Morris
2017](https://doi.org/10.1214/16-AOAS1010)): put the survey's tables
together, estimate the network's statistics, fit models, check them, and
compare the estimates with those of the whole network, which, here, we
have. The [Egocentric data](../egocentric.md) page of the user guide has the
details.

## The survey

`faux.mesa.high` is a friendship network of the 205 students of a high
school ([A complete ERGM analysis](ergm.md) fits it whole). Suppose we had
surveyed 100 of them, at random: each named their friends in the school,
with each friend's grade, race and sex, and said which of their friends are
friends with each other. A survey gives three tables: the egos, with their
attributes; the alters, each with the ego who named them; and the ties
among each ego's alters. Here we make them from the network; with real
data, they would be read from the survey's files. The columns that link
them have egor's names (R's package for egocentric data): `.egoID` for the
ego, `.altID` for the alter (unique within each ego's alters), and `.srcID`
and `.tgtID` for the two alters of a tie.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")  # the whole network, to make the survey and check it
rng = np.random.default_rng(1)
surveyed = np.sort(rng.choice(mesa.vcount(), 100, replace=False))


def attributes(student):
    return {"Grade": int(student["Grade"]), "Race": student["Race"], "Sex": student["Sex"]}


egos = pd.DataFrame([{".egoID": i + 1, **attributes(mesa.vs[i])} for i in surveyed])
alters = pd.DataFrame([
    {".egoID": i + 1, ".altID": j + 1, **attributes(mesa.vs[j])}
    for i in surveyed for j in mesa.neighbors(i)
])
alter_ties = pd.DataFrame([
    {".egoID": i + 1, ".srcID": j + 1, ".tgtID": k + 1}
    for i in surveyed for j in mesa.neighbors(i) for k in mesa.neighbors(i) if j < k and mesa.are_adjacent(j, k)
])
egos.head()
```

```{code-cell} ipython3
alters.head()
```

```{code-cell} ipython3
alter_ties.head()
```

Ego 8 named three friends, 30, 104 and 160, who are all friends with each
other. {class}`ergmx.EgoData` puts the tables together (as egor's `egor()`);
they may also be dicts of columns, or lists of records:

```{code-cell} ipython3
survey = ergmx.EgoData(egos, alters, alter_ties)
survey
```

## Describing the survey

The egos' numbers of friends are their degrees, and the grades of egos and
their friends, the mixing by grade:

```{code-cell} ipython3
degrees = alters.groupby(".egoID").size().reindex(egos[".egoID"], fill_value=0)
print(f"mean number of friends {degrees.mean():.2f}; {np.mean(degrees == 0):.0%} name none")
pd.crosstab(alters[".egoID"].map(egos.set_index(".egoID")["Grade"]), alters["Grade"],
            rownames=["ego's grade"], colnames=["friend's grade"])
```

Most friends are in the same grade, as in the school.

## Estimating the network's statistics

Each ego contributes to the network's statistics: half of each of their
friendships to `edges` (the other half is the friend's), half of each with a
friend of the same grade to `nodematch('Grade')`, themselves to `degree(0)`
if they named no one, and half of each of their friendships' contributions
to `gwesp`, which depend on how many friends the two share (as the ties
among the alters tell). Their averages, times the school's
size, estimate the school's statistics; {func}`ergmx.ego_stats`, as R's
`summary(egor ~ formula, scaleto=)`, computes them, with their covariance
from the sampling of the egos. Next to them, the whole school's, which a
survey wouldn't have:

```{code-cell} ipython3
formula = "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + degree(0) + gwesp(0.5, fixed=TRUE)"
estimates, cov = ergmx.ego_stats(formula, survey, scaleto=mesa.vcount())
pd.DataFrame({
    "Estimate": estimates.values(), "Std. Error": np.sqrt(np.diag(cov)),
    "School": ergmx.summary_stats(mesa, formula).values(),
}, index=list(estimates)).round(1)
```

Every estimate is within one standard error of the school's value. The
sample has somewhat fewer friendships, and more triadic closure, than the
school.

## A first model: homophily

{func}`ergmx.ergm_ego` fits the model to these estimates on a
*pseudo-population*, a network of copies of the egos with their
attributes, as many as `popsize`, the population's size (here, rounded to
two copies of each of the 100 egos); an offset on `edges` makes the
coefficients those of a network of `popsize` people. The standard errors
come from the egos' sampling variance:

```{code-cell} ipython3
homophily = ergmx.ergm_ego("edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race')", survey,
                           popsize=mesa.vcount())
homophily.summary()
```

Friendships within a grade are much more likely, and within a race, more
likely. The model is dyad-independent, so ergmx fits it exactly (ergm.ego
fits it by Monte Carlo MLE).

## Adding triadic closure

With the ties among the alters, the survey also tells how many friends
each friendship's two students share, and `gwesp` can model triadic
closure:

```{code-cell} ipython3
closure = ergmx.ergm_ego(
    "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)", survey,
    popsize=mesa.vcount(), seed=1,
)
closure.summary()
```

Closure is strong, and part of the grade homophily was friends of friends,
often in the same grade. With triadic terms, ergm.ego's adjustment for the
network's size also puts an offset on `transitiveties`. The model is
dyad-dependent, fitted by Monte Carlo MLE on the pseudo-population, from
networks simulated by MCMC; its diagnostics are those of a fit of a whole
network ([Convergence and MCMC diagnostics](../diagnostics.md)):

```{code-cell} ipython3
closure.mcmc_diagnostics().plot();
```

The four chains mix and agree, around the observed statistics.

## Does the model fit?

{meth}`EgoFit.gof() <ergmx.EgoFit.gof>`, as ergm.ego's `gof()`, compares
the distributions that the egos estimate, per person, with those of
networks of the pseudo-population simulated from the model: the degrees,
the edgewise shared partners (from the ties among the alters), and the model
statistics. Other distributions, such as distances, can't be estimated
from egocentric data. The black lines are the survey's, and the boxplots
the simulated networks':

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
homophily.gof(seed=1).plot(axes[0])
closure.gof(seed=1).plot(axes[1])
for model, row in zip(["Homophily", "Closure"], axes):
    for ax in row:
        ax.set_title(f"{model}: {ax.get_title()}")
fig.tight_layout()
```

Without closure, almost no friends share a friend, and too few students
have no friends; the closure model reproduces both distributions.

## How good are the estimates?

The point of an egocentric model is to estimate the whole network's model
without observing it. Here, we can check: {func}`ergmx.table` puts the
estimates from the 100 egos next to those of the whole network's ERGM
(`omit=` leaves out the offsets; egocentric fits have no log-likelihood):

```{code-cell} ipython3
whole = ergmx.ergm(
    mesa, "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)", seed=1,
)
ergmx.table(closure, whole, names=["100 egos", "Whole network"], omit="offset")
```

The survey of half the school recovers its model: the same significant
effects, with the same signs (sex has none in either), each estimate
within about two of its standard errors of the whole network's. The differences are those of the
sample's statistics, with fewer friendships and more closure than the
school's; the standard errors, larger, account for them.

## Coefficients per capita

Without `popsize`, the coefficients are *per capita*, as ergm.ego's
default: they apply to a network of any size N with the same mean degree,
after subtracting log(N) from `edges`'s. Fitted on the same
pseudo-population of 200 (`ppopsize=200`; without it, the pseudo-population
has as many people as the egos):

```{code-cell} ipython3
per_capita = ergmx.ergm_ego("edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race')", survey,
                            ppopsize=200)
pd.DataFrame({"Per capita": per_capita.coef, "School of 205": homophily.coef}).round(3)
```

`edges`'s coefficient for the school is the per capita one less
log(205) = 5.32, and the others are the same. Per capita coefficients
compare populations of different sizes, such as surveys of several schools.

Next, [A complete Bayesian ERGM analysis](bayesian.md) samples the
posterior distribution of a model's coefficients.
