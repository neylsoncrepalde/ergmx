---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete temporal ERGM analysis

A panel study observes the same people's network several times. A temporal
ERGM, as R's tergm fits ([Krivitsky and Handcock
2014](https://doi.org/10.1111/rssb.12014)), models each network given the one
before it: which ties form, and which persist. This page fits such models to
a three-wave panel: describe the changes, fit models of formation and
persistence, check them, test for a trend, respect the survey's design,
simulate the process forward, and report the results. The
[Networks over time](../temporal.md) page of the user guide has the details,
and the EGMME, for a process known from a single network.

## The data

Sampson (1968) asked the 18 novices of a monastery, at three times during
a crisis that ended with several of them expelled, which three of the
others they liked most ([A complete Bayesian ERGM analysis](bayesian.md)
models the last time alone). Between the times, some nominations were
new, some were dropped, and the rest persisted:

```{code-cell} ipython3
import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

import ergmx
from ergmx import datasets

waves = [datasets.load(f"samplk{t}") for t in (1, 2, 3)]  # igraph.Graphs, the same 18 monks
changes = []
for t, (before, after) in enumerate(zip(waves, waves[1:]), start=1):
    old, new = set(before.get_edgelist()), set(after.get_edgelist())
    changes.append({"transition": f"{t} to {t + 1}", "formed": len(new - old), "dissolved": len(old - new),
                    "persisted": len(old & new)})
pd.DataFrame(changes).set_index("transition")
```

About a third of the nominations are new at each transition, fewer at
the second. Drawn with the same layout at each time, the monks coloured by
Sampson's groups and the new nominations in red:

```{code-cell} ipython3
random.seed(1)  # igraph's layouts use Python's random numbers
layout = ig.union(waves, byname=False).layout("fr")
groups = sorted(set(waves[0].vs["group"]))
colors = plt.get_cmap("Set2")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for t, (ax, g) in enumerate(zip(axes, waves)):
    before = set(waves[t - 1].get_edgelist()) if t else set(g.get_edgelist())
    ig.plot(
        g, target=ax, layout=layout, vertex_size=14, edge_arrow_size=7,
        vertex_color=[colors(groups.index(x)) for x in g.vs["group"]],
        edge_color=["grey" if e in before else "crimson" for e in g.get_edgelist()],
        edge_width=[0.6 if e in before else 1.4 for e in g.get_edgelist()],
    )
    ax.set_title(f"Time {t + 1}" + (": new nominations in red" if t else ""))
axes[-1].legend(
    handles=[plt.Line2D([], [], marker="o", linestyle="", color=colors(k), label=x) for k, x in enumerate(groups)],
    loc="upper left", bbox_to_anchor=(1, 1), frameon=False,
);
```

Most new nominations are within a group (22 of 38), and many return one
(16).

## Formation and persistence

tergm's operators evaluate a formula on views of each transition:
`Form(~...)` on the union of the previous and the current network, so its
statistics change only when a tie forms, and `Persist(~...)` on their
intersection, so its statistics change only when a tie persists or
dissolves. A model of `Form()` and `Persist()` terms is *separable*:
formation and persistence are each an ERGM of its own, given the previous
network. {func}`ergmx.tergm` fits it to the transitions of the series by
conditional maximum likelihood, as `tergm(..., estimate="CMLE")`. With
`edges` alone, the model is two rates, and its MLE is exact:

```{code-cell} ipython3
rates = ergmx.tergm(waves, "Form(~edges) + Persist(~edges)")
rates.summary()
```

```{code-cell} ipython3
print(f"a nomination forms with probability {expit(rates.coef['Form(1)~edges']):.3f}; "
      f"one persists with probability {expit(rates.coef['Persist(1)~edges']):.3f}")
```

Of the 500 nominations that could have formed (the pairs not nominated at
times 1 and 2), 38 did, 7.6%; of the 112 that could have persisted, 75 did,
67%. The names follow tergm, with each coefficient's column of the
operator's linear model, `(1)` for its intercept.

## Reciprocity and groups

Does a monk start liking those who like him, and those of his group? And
do such nominations last longer? The same terms in both operators:

```{code-cell} ipython3
terms = "edges + mutual + nodematch('group')"
groups_model = ergmx.tergm(waves, f"Form(~{terms}) + Persist(~{terms})", seed=1)
groups_model.summary()
```

A nomination is far more likely to form if it returns one (`Form(1)~mutual`)
and within a group (`Form(1)~nodematch.group`). Once there, nominations
within a group persist more; returned ones, no more than others. The model
is dyad-dependent, fitted by Monte Carlo MLE, from networks simulated by
MCMC given the previous ones; `seed=` makes it reproducible. Triadic
closure adds nothing (`gwesp` in both operators), as
{func}`ergmx.compare` shows ([Model comparison](../model-comparison.md)):

```{code-cell} ipython3
closure_terms = terms + " + gwesp(0.5, fixed=TRUE)"
closure = ergmx.tergm(waves, f"Form(~{closure_terms}) + Persist(~{closure_terms})", seed=1)
ergmx.compare(rates, groups_model, closure)
```

## Did the estimation converge, and does the model fit?

The diagnostics and the goodness of fit are those of cross-sectional fits
([Convergence and MCMC diagnostics](../diagnostics.md),
[Goodness of fit](../goodness-of-fit.md)), with the networks simulated
from the model given the previous ones, summed over the transitions:

```{code-cell} ipython3
groups_model.mcmc_diagnostics().plot();
```

```{code-cell} ipython3
groups_model.gof(seed=1).plot();
```

The chains mix and agree. The model reproduces the in-degrees, shared
partners and distances, but not the out-degrees: almost every monk names
three, by the survey's design, while the model's networks spread the
nominations more widely.

## Did the process change?

Every transition has the same coefficients unless the operators' `lm=`
says otherwise: `lm=~.Time` adds a linear trend in the time of the
transition's later network (`Form(.Time)~edges`), here for the baseline
rates of formation and persistence:

```{code-cell} ipython3
trend = ergmx.tergm(
    waves,
    "Form(~edges, lm=~.Time) + Form(~mutual + nodematch('group'))"
    " + Persist(~edges, lm=~.Time) + Persist(~mutual + nodematch('group'))",
    seed=1,
)
trend.summary()
```

Neither trend is significant: the fewer changes of the second transition
are within chance.

## The survey's design

The constraint `"odegrees"` keeps each monk's number of nominations at the
observed one, as the survey's design did, in every simulated network
([Constraints](../constraints.md)). The number of nominations is then fixed,
so each one formed means one fewer kept: `Form(~edges)` and
`Persist(~edges)` are the same statistic, up to sign and a constant, and
only `Persist`'s stays:

```{code-cell} ipython3
design = ergmx.tergm(waves, f"Form(~mutual + nodematch('group')) + Persist(~{terms})", constraints="odegrees",
                     seed=1)
design.summary()
```

The conclusions hold: new nominations return old ones and stay within the
groups, and nominations within groups last longer. `Persist(1)~edges` is
now the tendency of a nomination to be kept rather than replaced by
another, given the monk names the same number: strong.

## Simulating the process

A fitted temporal model is a process: from a network, draw the next one,
and so on. {meth}`fit.simulate(time_slices=...) <ergmx.ErgmFit.simulate>`
runs it forward from the last network of the series, as tergm's
`simulate(fit, time.slices=)`, here for 50 steps, with the statistics of
`monitor=` at each one (the dots are the observed three times):

```{code-cell} ipython3
future = groups_model.simulate(time_slices=50, seed=1, monitor="edges + mutual + nodematch('group')")
observed = pd.DataFrame([ergmx.summary_stats(g, "edges + mutual + nodematch('group')") for g in waves])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, name in zip(axes, ["edges", "mutual", "nodematch.group"]):
    ax.plot(np.arange(1, len(future.monitor[name]) + 1), future.monitor[name], color="grey")
    ax.scatter([-2, -1, 0], observed[name], color="black", zorder=3)
    ax.set_title(name)
    ax.set_xlabel("time steps after time 3")
fig.tight_layout()
print({name: round(float(np.mean(future.monitor[name][10:])), 1) for name in observed})
```

Run forward, the process keeps about as many nominations and mutual pairs
as the monastery had, but more of them within the groups (43, against 30
to 38 observed): had it gone on, the groups would have closed in on
themselves further. The result, a
{class}`ergmx.DynamicSimulation`, also has the ties that formed and
dissolved at each step, and their durations:

```{code-cell} ipython3
finished, ongoing = future.durations()
print(f"{len(finished)} nominations ended, after {finished.mean():.1f} steps on average; "
      f"{len(ongoing)} still present")
```

## Reporting the results

{func}`ergmx.table` puts the models side by side, as R's texreg; `rename=`
gives readable labels, and `.to_latex()`, `.to_html()` and `.to_markdown()`
its other formats:

```{code-cell} ipython3
ergmx.table(rates, groups_model, closure, names=["Rates", "Groups", "Closure"])
```

Next, [A complete bipartite ERGM analysis](bipartite.md) models a network
of women and the events they attended.
