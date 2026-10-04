---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete bipartite ERGM analysis

A bipartite (two-mode) network has two kinds of vertices, with ties only
between kinds: people and the events they attend, directors and the boards
they sit on. This page models the classic one, Davis, Gardner and Gardner's
(1941) Southern Women: describe both modes and the networks they induce on
each other, fit models of who attends what, check them, compare them,
interpret them and report them. The [Bipartite networks](../bipartite.md)
page of the user guide has the details.

## The data

In the 1930s, Davis, Gardner and Gardner recorded which of 14 social events
in Natchez, Mississippi, each of 18 women attended, in their study of the
town's social classes, *Deep South*
([1941](https://doi.org/10.7208/chicago/9780226817996.001.0001)). In
`davis`, the vertex attribute `type` is false for the women (the first
mode) and true for the events (the second), as igraph's bipartite
functions have it:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ergmx
from ergmx import datasets

davis = datasets.load("davis")  # an igraph.Graph
women = [v.index for v in davis.vs if not v["type"]]
events = [v.index for v in davis.vs if v["type"]]
attended = pd.DataFrame(np.array(davis.get_adjacency().data)[np.ix_(women, events)],
                        index=davis.vs[women]["name"], columns=davis.vs[events]["name"])
print(f"{len(women)} women, {len(events)} events, {davis.ecount()} attendances")
```

The attendance matrix, in Davis, Gardner and Gardner's order:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(attended.values, cmap="Greys", aspect="auto")
ax.set_xticks(range(len(events)), attended.columns)
ax.set_yticks(range(len(women)), attended.index)
ax.set_title("Attendance (black: attended)");
```

Two blocks stand out: the first women attended the first events, the last
women the last ones, and both the events in the middle. Davis, Gardner and
Gardner read them as two groups of women.

## Describing both modes

Each mode has its degrees: the events each woman attended, and the women at
each event. The *dyadwise shared partners* link the two modes: two women
share the events both attended, and two events the women who attended
both. With `bipartite=True`, {func}`ergmx.summary_stats` takes the mode
from `type`; `b1` terms are about the first mode, `b2` terms the second:

```{code-cell} ipython3
print("events attended by each woman:", sorted(attended.sum(axis=1).tolist()))
print("women at each event:", sorted(attended.sum(axis=0).tolist()))
ergmx.summary_stats(davis, "edges + b1star(2) + b2star(2) + b1dsp(0) + b2dsp(0)", bipartite=True)
```

The women attended 2 to 8 events, and the events had 3 to 14 women. There
are 322 meetings of two women at an event (`b2star2`, the two-stars
centered on events), 14 pairs of women who never met at one (`b1dsp0`),
and 25 pairs of events with no woman in common (`b2dsp0`). The full
distributions of these shared partners, which the models below should
reproduce:

```{code-cell} ipython3
def shared(graph):
    """The pairs of women by events in common (0 to 7+), and the pairs of
    events by women in common (0 to 9+)."""
    a = np.array(graph.get_adjacency().data)[np.ix_(women, events)]
    pairs_of_women = (a @ a.T)[np.triu_indices(len(women), 1)]
    pairs_of_events = (a.T @ a)[np.triu_indices(len(events), 1)]
    return np.bincount(np.minimum(pairs_of_women, 7), minlength=8), \
        np.bincount(np.minimum(pairs_of_events, 9), minlength=10)


observed = shared(davis)
print("pairs of women by events in common:", observed[0].tolist())
print("pairs of events by women in common:", observed[1].tolist())
```

## Activity and popularity

{func}`ergmx.ergm` fits bipartite models with `bipartite=True`; only the
dyads between the modes, 18 x 14 = 252, are modelled. The Bernoulli model
(`edges`) gives every woman the same chance to attend every event. But
women differ in how much they go out, and events in how many they draw:
`b1sociality` gives each woman her own effect (all but the first), and
`b2sociality` each event, a Rasch model of attendance. Both models are
dyad-independent, so their estimates are a logistic regression's, exact:

```{code-cell} ipython3
bernoulli = ergmx.ergm(davis, "edges", bipartite=True)
rasch = ergmx.ergm(davis, "edges + b1sociality + b2sociality", bipartite=True)
ergmx.compare(bernoulli, rasch)
```

Activity and popularity explain much of the attendance (the
likelihood-ratio test), though BIC, which charges more for the 30
coefficients, prefers the Bernoulli model.

## Do the women split into groups?

If women and events were matched only by activity and popularity, how
many pairs of events would have no woman in common? Simulating 100
networks from the Rasch model, as {meth}`fit.simulate()
<ergmx.ErgmFit.simulate>` does, and comparing their shared partners with
the observed ones (the black lines):

```{code-cell} ipython3
def check(**fits):
    """The observed shared partners against those of 100 networks simulated from each fit."""
    titles = ["pairs of women by events in common", "pairs of events by women in common"]
    labels = [[*map(str, range(7)), "7+"], [*map(str, range(9)), "9+"]]
    fig, axes = plt.subplots(len(fits), 2, figsize=(12, 3.6 * len(fits)), squeeze=False)
    for row, (name, fit) in zip(axes, fits.items()):
        simulated = [shared(g) for g in fit.simulate(100, seed=1)]
        for k, ax in enumerate(row):
            ax.boxplot(np.array([s[k] for s in simulated]), tick_labels=labels[k], showfliers=False,
                       medianprops={"color": "grey"})
            ax.plot(np.arange(1, len(labels[k]) + 1), observed[k], color="black", marker="o", markersize=3)
            ax.set_title(f"{name}: {titles[k]}")
    fig.tight_layout()


check(Rasch=rasch)
```

The Rasch model gives about 14 pairs of events with no woman in common, and
the data have 25: the events split into sets that different women
attended, more than activity and popularity explain. `b2dsp(0)`, the
number of pairs of events without a common attendee, measures this
directly. With it, the model is dyad-dependent: whether a woman attends an
event depends on which other events she, and the others, attended. It is
fitted by Monte Carlo MLE, from networks simulated by MCMC; `seed=` makes
it reproducible:

```{code-cell} ipython3
groups = ergmx.ergm(davis, "edges + b1sociality + b2sociality + b2dsp(0)", bipartite=True, seed=1)
pd.DataFrame({"Estimate": groups.coef, "Std. Error": groups.stderr}).loc[["edges", "b2dsp0"]].round(3)
```

(`groups.summary()` has all 32 coefficients.) `b2dsp0`'s is positive, 2.6
standard errors from 0: pairs of events tend to have no attendee in
common.

## Did the estimation converge, and does the model fit?

The diagnostics are those of any fit ([Convergence and MCMC
diagnostics](../diagnostics.md)); printed, they have a row for each of the
32 statistics, summarised here:

```{code-cell} ipython3
diagnostics = groups.mcmc_diagnostics()
print(f"largest mean deviation from the observed: {np.max(np.abs(diagnostics.mean / diagnostics.sd)):.2f} SD; "
      f"R-hat up to {diagnostics.rhat.max():.3f}; effective sizes from {diagnostics.effective_size.min():.0f}")
```

The simulated statistics center on the observed ones, and the chains agree
and mix (`diagnostics.plot()` draws their traces).

For bipartite networks, {meth}`~ergmx.ErgmFit.gof` compares, as ergm's
does, the degrees of each mode, the dyadwise shared partners (of both
modes together) and the geodesic distances, and the model statistics
(left out here: they are 32) ([Goodness of fit](../goodness-of-fit.md)):

```{code-cell} ipython3
groups.gof(stats=["b1degree", "b2degree", "dspartners", "distance"], seed=1).plot();
```

The sociality terms reproduce the degrees, and the model the shared
partners and distances. The shared partners of each mode separately:

```{code-cell} ipython3
check(Groups=groups)
```

The model now gives as many pairs of events without a common attendee as
the data. The pairs of women are reproduced less closely: more pairs share
exactly two events, and fewer five, than the model's networks have.

## Which model is better?

```{code-cell} ipython3
ergmx.compare(bernoulli, rasch, groups)
```

AIC prefers the model with the groups.

## Interpreting the coefficients

The exponential of a coefficient multiplies the conditional odds that a
woman attends an event, for each unit its statistic gains when she does
([Interpreting and reporting results](../interpretation.md)):

```{code-cell} ipython3
groups.odds_ratios().to_frame().loc[["edges", "b2dsp0"]]
```

A woman's attendance changes `b2dsp0` by minus the number of her other
events that had no attendee in common with this one: each such pair of
events her attendance would join for the first time divides the odds that
she attends by about 1.6. The women keep to their set of events.

## Reporting the results

{func}`ergmx.table` puts the models side by side, as R's texreg, here
without the 30 sociality coefficients (`omit=`, a regular expression);
`.to_latex()`, `.to_html()` and `.to_markdown()` give its other formats:

```{code-cell} ipython3
ergmx.table(bernoulli, rasch, groups, names=["Bernoulli", "Rasch", "Groups"], omit="sociality",
            rename={"edges": "Edges", "b2dsp0": "Pairs of events without a common attendee"})
```

The [user guide](../index.md) covers each step of these analyses in depth.
