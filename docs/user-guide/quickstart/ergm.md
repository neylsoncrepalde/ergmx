---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete ERGM analysis

This page takes a friendship network from the data to a table of results:
describe it, fit a model, check that the estimation converged and that the
model fits, compare it with a simpler model, interpret it and report it.

## The data

`faux.mesa.high` is a friendship network of 205 students of a high school,
simulated from the Add Health study design, with each student's grade, race
and sex:

```{code-cell} ipython3
import random

import igraph as ig
import matplotlib.pyplot as plt

import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")  # an igraph.Graph
print(f"{mesa.vcount()} students, {mesa.ecount()} friendships, density {mesa.density():.4f}")
print(f"{len(mesa.vs.select(_degree=0))} students have no friends in the school")
```

Drawn with igraph, each student coloured by grade:

```{code-cell} ipython3
random.seed(1)  # igraph's layouts use Python's random numbers
grades = sorted(set(mesa.vs["Grade"]))
colors = plt.get_cmap("viridis", len(grades))

fig, ax = plt.subplots(figsize=(8, 7))
ig.plot(
    mesa, target=ax, layout=mesa.layout("fr"), vertex_size=6, edge_width=0.4, edge_color="grey",
    vertex_color=[colors(grades.index(g)) for g in mesa.vs["Grade"]],
)
ax.legend(
    handles=[plt.Line2D([], [], marker="o", linestyle="", color=colors(k), label=f"Grade {g:g}")
             for k, g in enumerate(grades)],
    loc="upper left", bbox_to_anchor=(1, 1), frameon=False,
);
```

Friends tend to be in the same grade, and they form clusters. Before
modelling, {func}`ergmx.summary_stats` counts the statistics a model could
use, as R's `summary(formula)`
([Formulas](../formulas.md), [Term reference](../../terms.md)):

```{code-cell} ipython3
ergmx.summary_stats(mesa, "edges + nodematch('Grade') + nodematch('Race') + nodematch('Sex') + triangle")
```

Of the 203 friendships, 163 are between students of the same grade, and
there are 62 triangles of friends.

## A first model: homophily

{func}`ergmx.ergm` fits a model given as an R-style formula. The first one
has the baseline log-odds of a friendship (`edges`), a difference between
boys and girls in their number of friendships (`nodefactor('Sex')`), and
homophily on grade and race (`nodematch`):

```{code-cell} ipython3
homophily = ergmx.ergm(mesa, "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race')")
homophily.summary()
```

Students of the same grade are much more likely to be friends, and so,
less, are students of the same race; boys have fewer friendships than
girls. These terms are dyad-independent: each friendship is independent of
the others, so the estimates are a logistic regression's, exact
([Fitting models](../fitting.md)).

## Adding triadic closure

Friendships are not independent, though: friends of friends become friends.
`gwesp`, the geometrically weighted edgewise shared partners, counts the
friendships whose two students share friends, with less weight on each
further shared friend:

```{code-cell} ipython3
closure = ergmx.ergm(
    mesa,
    "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)",
    seed=1,
)
closure.summary()
```

Triadic closure is strong (`gwesp`), and with it in the model the effect of
grade is smaller: part of what looked like homophily was friends of friends,
who are often in the same grade. The difference between boys and girls is
no longer significant. The model is dyad-dependent, so it is fitted by Monte
Carlo maximum likelihood, from networks simulated by MCMC; `seed=` makes
the result reproducible.

## Did the estimation converge?

At the estimate, the networks simulated from the model should have the
observed statistics on average, and the MCMC chains should agree
([Convergence and MCMC diagnostics](../diagnostics.md)):

```{code-cell} ipython3
diagnostics = closure.mcmc_diagnostics()
diagnostics
```

The mean deviations are a small fraction of the SDs, the four chains agree
(R-hat close to 1), and the samples are worth hundreds of independent ones
(the effective sizes). The traces should look like noise around zero, and
the chains' densities overlap:

```{code-cell} ipython3
diagnostics.plot();
```

## Does the model fit?

Each model reproduces the statistics in its formula, but does it reproduce
the rest of the network's structure? {meth}`~ergmx.ErgmFit.gof` simulates
100 networks from the model and compares their degree, edgewise shared
partner and geodesic distance distributions with the observed network's
([Goodness of fit](../goodness-of-fit.md)). The black lines are the observed
distributions and the boxplots the simulated ones, here for each model:

```{code-cell} ipython3
stats = ["degree", "espartners", "distance"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
homophily.gof(stats=stats, seed=1).plot(axes[0])
closure.gof(stats=stats, seed=1).plot(axes[1])
for model, row in zip(["Homophily", "Closure"], axes):
    for ax in row:
        ax.set_title(f"{model}: {ax.get_title()}")
fig.tight_layout()
```

The homophily model misses the shared partners: in its networks, almost no
two friends share a friend. Its networks are also too connected, with fewer
students without friends, and fewer pairs of students who can't reach each
other, than the school. The closure model reproduces the shared partners
and, roughly, the degrees. The school still has more pairs of students 8 to
12 steps apart than its networks: it is made of longer chains of
friendships.

## Which model is better?

{func}`ergmx.compare` puts the models side by side, with their AIC and BIC
and, as they are nested, a likelihood-ratio test
([Model comparison](../model-comparison.md)):

```{code-cell} ipython3
ergmx.compare(homophily, closure)
```

Adding triadic closure lowers the AIC by 183, far more than the Monte Carlo
error of the log-likelihood.

## Interpreting the coefficients

A coefficient is the change in the log-odds of a friendship, given the rest
of the network, for each unit of its statistic. Its exponential is an odds
ratio ([Interpreting and reporting results](../interpretation.md)):

```{code-cell} ipython3
closure.odds_ratios()
```

Students of the same grade have about 7 times the odds of a friendship of
others with the same friends, race and sex, and those of the same race 1.3
times. A first shared friend multiplies the odds by about 3.4. The same page
has tie probabilities and marginal effects.

## Reporting the results

{func}`ergmx.table` puts the models side by side for a paper, as R's texreg,
with `rename=` for readable labels:

```{code-cell} ipython3
results = ergmx.table(
    homophily,
    closure,
    names=["Homophily", "Closure"],
    rename={
        "edges": "Edges",
        "nodefactor.Sex.M": "Male",
        "nodematch.Grade": "Same grade",
        "nodematch.Race": "Same race",
        "gwesp.fixed.0.5": "GWESP (decay 0.5)",
    },
)
results
```

`print(results)` gives the text table, and `.to_latex()`, `.to_html()` and
`.to_markdown()` the others:

```python
with open("models.tex", "w") as f:
    f.write(results.to_latex())
```

Next, [A complete multilevel ERGM analysis](multilevel.md) does the same for
a network with two levels.
