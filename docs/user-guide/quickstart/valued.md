---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete valued ERGM analysis

Some ties are counts: of meetings, messages, shared events. This page models
the counts of a small social network with a valued ERGM, as R's ergm.count
does ([Krivitsky 2012](https://doi.org/10.1214/12-EJS696)): describe the
counts, choose a reference distribution, fit models of who interacts more,
check that the estimation converged and that the models reproduce the
counts, compare them, interpret them and report them. The
[Valued networks](../valued.md) page of the user guide has the details.

## The data

Zachary's karate club is the 34 members of a university karate club, which
split in two after a dispute between the instructor and the club's
president ([Zachary 1977](https://doi.org/10.1086/jar.33.4.3629752)). For
each pair of members, Zachary counted the contexts, of 8 (classes, the
instructor's private studio, the same bar...), in which they interacted. The
edges of `zach` are the pairs that interacted, with the counts in the edge
attribute `contexts`; `faction.id` places each member from -2 (strongly the
instructor's faction) to 2 (strongly the president's), and `role` is
`Instructor`, `President` or `Member`.

```{code-cell} ipython3
import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

import ergmx
from ergmx import datasets

zach = datasets.load("zach")  # an igraph.Graph
contexts = np.array(zach.es["contexts"], dtype=int)
pairs = zach.vcount() * (zach.vcount() - 1) // 2
print(f"{zach.vcount()} members, {pairs} pairs, {zach.ecount()} of them interact")
print("pairs by number of contexts:", dict(enumerate(np.bincount(contexts).tolist()[1:], start=1)))
```

Drawn with igraph, with each tie as wide as its count, and each member
coloured by faction (the instructor is the `I`, the president the `P`):

```{code-cell} ipython3
random.seed(1)  # igraph's layouts use Python's random numbers
colors = plt.get_cmap("coolwarm", 5)

fig, ax = plt.subplots(figsize=(8, 7))
ig.plot(
    zach, target=ax, layout=zach.layout("fr", weights="contexts"), vertex_size=14,
    vertex_color=[colors(int(f) + 2) for f in zach.vs["faction.id"]],
    vertex_label=[role[0] if role != "Member" else "" for role in zach.vs["role"]], vertex_label_size=8,
    edge_width=[0.6 * c for c in contexts], edge_color="grey",
)
ax.legend(
    handles=[plt.Line2D([], [], marker="o", linestyle="", color=colors(k + 2), label=f"{k:+d}")
             for k in range(-2, 3)],
    title="faction.id", loc="upper left", bbox_to_anchor=(1, 1), frameon=False,
);
```

The two factions gather around the instructor and the president, who
interact with many members and in many contexts. Few ties cross the
factions, and they are weak.

## Choosing a reference distribution

A valued ERGM gives each pair's count a *reference* distribution, which the
model's statistics then tilt. The Poisson is the usual one for counts (these
are of 8 contexts, so `"Binomial(8)"` would also do, but a Poisson with
these means gives counts above 8 almost no probability). With no other
terms, every pair's count would be Poisson with the same mean; but
the club's counts are far more variable than that, and far more pairs never
interact:

```{code-cell} ipython3
values = np.concatenate([contexts, np.zeros(pairs - len(contexts), dtype=int)])
mean = values.mean()
print(f"mean {mean:.2f}, variance {values.var():.2f}")
print(f"pairs that never interact: {np.mean(values == 0):.0%}; a Poisson with that mean: {np.exp(-mean):.0%}")
```

This is *zero inflation*: whether two members interact at all is one
question, and in how many contexts, another. The `nonzero` term, the number
of pairs who interact, models the first, and `sum`, the sum of the counts,
the second. With the counts in `response=`, {func}`ergmx.summary_stats`
gives valued statistics, as R's `summary(formula, response=)`; the
dyad-independent terms of binary ERGMs then add up each pair's counts
(their `form="sum"`, the default, as in their names):

```{code-cell} ipython3
ergmx.summary_stats(zach, "sum + nonzero + nodefactor('role', levels=-2) + absdiff('faction.id')",
                    response="contexts")
```

The 78 pairs who interact do so in 231 contexts; the instructor's pairs in
42 and the president's in 48. `absdiff('faction.id')` adds up each pair's
difference in faction, times its count.

## A Poisson model, with zero inflation

{func}`ergmx.ergm` fits a valued model given the counts as `response` and
the reference as `reference` (`"Poisson"`, `"Geometric"`,
`"Binomial(trials)"` or `"DiscUnif(a, b)"`). Valued models are fitted by
Monte Carlo MLE, from networks of counts simulated by MCMC; `seed=` makes
the results reproducible. First, `sum` alone, the Poisson model, and then
with `nonzero`:

```{code-cell} ipython3
valued = dict(response="contexts", reference="Poisson", seed=1)
poisson = ergmx.ergm(zach, "sum", **valued)
inflated = ergmx.ergm(zach, "sum + nonzero", **valued)
inflated.summary()
```

`sum`'s coefficient alone (-0.89, in `poisson`) is the log of the mean
count, 0.41. With `nonzero`, few pairs interact (its large negative
coefficient), and those who do, in several contexts: their counts are as a
Poisson's with mean e{sup}`1.02` = 2.8, without the zeros.

## Who interacts more?

The instructor and the president are the club's leaders, and the factions
divide it. `nodefactor('role', levels=-2)` lets the pairs of the instructor
and of the president have more, or fewer, contexts than pairs of two
members (`levels=-2` leaves out the second level, `Member`, as the
baseline), and `absdiff('faction.id')` lets the count fall, or rise, with
the members' difference in faction:

```{code-cell} ipython3
roles = ergmx.ergm(zach, "sum + nonzero + nodefactor('role', levels=-2) + absdiff('faction.id')", **valued)
roles.summary()
```

The leaders' pairs interact in more contexts, and members farther apart in
the dispute in fewer. The model is dyad-independent: given the
coefficients, each pair's count is independent of the others.

## Adding closure

Members who share strong ties to a third tend to have a strong tie
themselves. `transitiveweights('min', 'max', 'min')` (Krivitsky 2012) gives
each pair its strongest two-path, the larger, over third members, of the
smaller of the two counts that link them through that member, and adds up,
over the pairs, the smaller of that and the pair's own count:

```{code-cell} ipython3
closure = ergmx.ergm(
    zach,
    "sum + nonzero + nodefactor('role', levels=-2) + absdiff('faction.id')"
    " + transitiveweights('min', 'max', 'min')",
    **valued,
)
closure.summary()
```

Closure is strong, and with it, the leaders' advantage is smaller: some of
their contexts were with members who shared their other ties.

## Did the estimation converge?

At the estimate, the networks simulated from the model should have the
observed statistics on average, and the MCMC chains should agree and mix
([Convergence and MCMC diagnostics](../diagnostics.md)):

```{code-cell} ipython3
diagnostics = closure.mcmc_diagnostics()
diagnostics
```

The mean deviations are small next to the SDs (at most a quarter of an SD),
the four chains agree (R-hats close to 1), and the samples are worth
hundreds of independent ones. The traces look like noise around zero:

```{code-cell} ipython3
diagnostics.plot();
```

## Does the model reproduce the counts?

{meth}`fit.gof() <ergmx.ErgmFit.gof>`, as ergm's for valued models,
compares the model statistics and the distribution of the counts with those
of simulated networks ([Goodness of fit](../goodness-of-fit.md)). The
distribution is cumulative: the share of pairs with at most each number of
contexts.

```{code-cell} ipython3
closure.gof(seed=1).plot();
```

The model reproduces its statistics, as a converged fit should, and the
number of pairs who never interact. But more of its pairs have at most one
context than the club's (the boxplot at 1 is above the line): too many
pairs interact in a single context.

Other distributions take a few lines of code.
{meth}`fit.simulate() <ergmx.ErgmFit.simulate>` returns networks simulated
from a valued model, with their counts in the edge attribute `weight`.
Compare three distributions of 100 such networks with the club's: the pairs
who interact by their number of contexts, the members by their number of
partners, and the pairs who interact by their strongest two-path (the black
lines are the club's, the boxplots the simulated networks'):

```{code-cell} ipython3
def distributions(graph, weight):
    """The pairs who interact by number of contexts (1 to 7+), the members by
    number of partners (0 to 10+), and the pairs who interact by their
    strongest two-path (0 to 6+)."""
    counts = np.asarray(graph.es[weight], dtype=int)
    a = np.zeros((graph.vcount(), graph.vcount()), dtype=int)
    for (i, j), count in zip(graph.get_edgelist(), counts):
        a[i, j] = a[j, i] = count
    strongest = np.minimum(a[:, :, None], a[None, :, :]).max(axis=1)  # max over k of min(a_ik, a_kj)
    paths = [strongest[i, j] for i, j in graph.get_edgelist()]
    return (np.bincount(np.minimum(counts, 7), minlength=8)[1:],
            np.bincount(np.minimum(graph.degree(), 10), minlength=11),
            np.bincount(np.minimum(paths, 6), minlength=7))


def check(**fits):
    """The club's distributions against those of 100 networks simulated from each fit."""
    observed = distributions(zach, "contexts")
    titles = ["contexts of a pair", "partners of a member", "strongest two-path of a pair"]
    labels = [[*map(str, range(1, 7)), "7+"], [*map(str, range(10)), "10+"], [*map(str, range(6)), "6+"]]
    fig, axes = plt.subplots(len(fits), 3, figsize=(13, 3.5 * len(fits)), squeeze=False)
    for row, (name, fit) in zip(axes, fits.items()):
        simulated = [distributions(g, "weight") for g in fit.simulate(100, seed=1)]
        for k, ax in enumerate(row):
            ax.boxplot(np.array([s[k] for s in simulated]), tick_labels=labels[k], showfliers=False,
                       medianprops={"color": "grey"})
            ax.plot(np.arange(1, len(labels[k]) + 1), observed[k], color="black", marker="o", markersize=3)
            ax.set_title(f"{name}: {titles[k]}")
    fig.tight_layout()


check(Roles=roles, Closure=closure)
```

Without closure, the model's networks have nearly three times as many pairs
who interact with no two-path between them as the club, which the closure
model reproduces. Both models give too many pairs who interact in a single
context, though, and too few in three: among the pairs who interact, the
counts vary less than a Poisson's. Neither reproduces the many members with
two partners, or the four with ten or more: members differ in sociability
more than the leaders' terms allow. (ergm.count's `nodecovar` term models
that, but on these data it makes the model degenerate: its networks drift
to ones where the leaders interact with nearly everyone, in three times the
club's contexts, and the fit stops with a {class}`~ergmx.DegeneracyError`
that says so.)

## Dispersion

`CMP` adds the sum of the counts' log-factorials, which turns the Poisson
reference into the Conway-Maxwell-Poisson distribution: a negative
coefficient makes the counts less dispersed than a Poisson's, a positive one
more:

```{code-cell} ipython3
final = ergmx.ergm(
    zach,
    "sum + nonzero + CMP + nodefactor('role', levels=-2) + absdiff('faction.id')"
    " + transitiveweights('min', 'max', 'min')",
    **valued,
)
final.summary()
```

```{code-cell} ipython3
check(Final=final)
```

The counts are underdispersed (`CMP`'s negative coefficient), and the model
now reproduces the numbers of pairs with each count. The other effects keep
their signs and significance; the leaders' grow a little, and closure's
shrinks.

## Which model is better?

{func}`ergmx.compare` puts the models side by side, with their AIC and BIC
and, as they are nested, likelihood-ratio tests
([Model comparison](../model-comparison.md)). The log-likelihoods of valued
models are estimated by path sampling from the reference distribution, as
those of binary models from the empty network:

```{code-cell} ipython3
ergmx.compare(poisson, inflated, roles, closure, final)
```

Every step improves the model, beyond the Monte Carlo error of the
log-likelihoods.

## Interpreting the coefficients

In a valued model, the exponential of a coefficient is the factor by which
a unit increase in its statistic multiplies the probability of a pair's
count, relative to the count one lower, given the rest of the network
([Interpreting and reporting results](../interpretation.md)). For a binary
network, these are the odds of a tie, and {meth}`~ergmx.ErgmFit.odds_ratios`
gives them for valued models too:

```{code-cell} ipython3
final.odds_ratios()
```

For a pair of the instructor's or the president's, each further context is
about 1.6 times as likely, relative to one fewer, as for two members with
the same other ties. Each unit of difference in faction multiplies it by
0.81: for members at opposite ends of the dispute (a difference of 4), by
0.81{sup}`4` = 0.42.

## Reporting the results

{func}`ergmx.table` puts the models side by side for a paper, as R's
texreg, with `rename=` for readable labels; `print(results)`,
`.to_latex()`, `.to_html()` and `.to_markdown()` give its formats:

```{code-cell} ipython3
results = ergmx.table(
    inflated,
    roles,
    closure,
    final,
    names=["Zero-inflated", "Roles", "Closure", "Final"],
    rename={
        "sum": "Sum of contexts",
        "nonzero": "Pairs who interact",
        "CMP": "Dispersion (CMP)",
        "nodefactor.sum.role.Instructor": "Instructor",
        "nodefactor.sum.role.President": "President",
        "absdiff.sum.faction.id": "Faction difference",
        "transitiveweights.min.max.min": "Transitive weights",
    },
)
results
```

Next, [A complete egocentric analysis](egocentric.md) estimates a network's
model from a survey of some of its members.
