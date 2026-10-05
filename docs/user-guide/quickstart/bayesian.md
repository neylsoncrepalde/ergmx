---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete Bayesian ERGM analysis

This page fits a Bayesian ERGM to a small network with {func}`ergmx.bergm`,
as R's Bergm does ([Caimo and Friel
2011](https://doi.org/10.1016/j.socnet.2010.09.004)): check that the chains
converged, compare the posterior with the MLE, read probabilities off it,
check the model with its posterior predictive distribution, improve it,
bring in a prior from earlier data, and report the results. The
[Bayesian ERGMs](../bayesian.md) page of the user guide has the details.

## What is a Bayesian ERGM?

The model is the same as {func}`ergmx.ergm`'s: a network's probability
depends on its statistics, weighted by the coefficients. What differs is
how the coefficients are estimated.

{func}`ergmx.ergm` finds the *maximum likelihood estimate* (MLE): the
coefficients under which the observed network is most probable. Its
uncertainty is a standard error, from the curvature of the likelihood at
the maximum. Confidence intervals and p-values then take the estimate to be
normally distributed around the true coefficients. With enough data, it
nearly is.

A Bayesian ERGM treats the coefficients as unknown quantities with a
probability distribution. It starts from a *prior*, what is known or
assumed about the coefficients before seeing the network. The likelihood of
the network updates the prior into the *posterior*: the coefficients'
probable values, given the network and the prior. {func}`ergmx.bergm`
returns thousands of draws from the posterior, and every summary comes from
them. These include means and standard deviations, and *credible
intervals*: a 95% credible interval holds the coefficient with probability
0.95. The draws also give the probability that a coefficient is positive,
or that one effect is stronger than another.

With a nearly flat prior and enough data, the two approaches agree: the
posterior mean is close to the MLE, and the posterior standard deviation to
the standard error. A Bayesian ERGM is most useful in these cases:

- **Small networks.** With a few dozen vertices, the uncertainty about the
  coefficients may be far from normal: skewed, or strongly correlated. The
  posterior describes it as it is, without the normal approximation.
- **Prior knowledge.** An earlier wave of the same network, a similar
  network, or published estimates can become the prior, and the posterior
  combines them with the new data.
- **Questions about probabilities.** "How probable is it that closure is
  positive?" has a direct answer in the posterior. A p-value answers a
  different question: how surprising the data would be if the coefficient
  were 0.
- **Checks that include the uncertainty.** The posterior predictive
  distribution simulates networks at many draws of the posterior, rather
  than at one estimate, so the simulated range includes the coefficients'
  uncertainty.

The price is computing time. Each step of each chain simulates a network,
so a posterior takes much longer than an MLE, and the cost grows with the
network's size: for large networks, the MLE is often the practical choice.

## The data

Sampson (1968) observed 18 novices of a New England monastery during a
crisis that ended with several of them expelled or leaving. At three times,
he asked each novice which three of the others he liked most. `samplk3` is
the last time, just before the expulsions, with Sampson's division of the
novices into groups: the Loyal opposition, the Turks, the Outcasts, and
the Waverers between them.

```{code-cell} ipython3
import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ergmx
from ergmx import datasets

monks = datasets.load("samplk3")  # an igraph.Graph
print(f"{monks.vcount()} monks, {monks.ecount()} nominations")
pd.Series(monks.vs["group"]).value_counts()
```

Drawn with igraph, each monk coloured by group:

```{code-cell} ipython3
random.seed(1)  # igraph's layouts use Python's random numbers
groups = sorted(set(monks.vs["group"]))
colors = plt.get_cmap("Set2")

fig, ax = plt.subplots(figsize=(7, 6))
ig.plot(
    monks, target=ax, layout=monks.layout("fr"), vertex_size=16, edge_width=0.8, edge_color="grey",
    edge_arrow_size=8, vertex_color=[colors(groups.index(g)) for g in monks.vs["group"]],
)
ax.legend(
    handles=[plt.Line2D([], [], marker="o", linestyle="", color=colors(k), label=g) for k, g in enumerate(groups)],
    loc="upper left", bbox_to_anchor=(1, 1), frameon=False,
);
```

The groups are clusters of mutual liking, with few nominations between
them. The model has the baseline log-odds of a nomination (`edges`),
reciprocity (`mutual`), liking within the groups (`nodematch('group')`), and
transitive closure, the liking of the liked ones' liked ones (`gwesp`; in a
directed network, it counts by default the two-paths i → k → j of each
nomination i → j):

```{code-cell} ipython3
formula = "edges + mutual + nodematch('group') + gwesp(0.5, fixed=TRUE)"
ergmx.summary_stats(monks, formula)
```

Of the 56 nominations, 15 pairs are mutual and 38 nominations are within a
group.

## The posterior

{func}`ergmx.bergm` samples the posterior with the exchange algorithm: at
each step, each chain proposes new coefficients, simulates an auxiliary
network from the model at them, and accepts or rejects the proposal by
comparing the observed network's statistics with the auxiliary network's.
The prior is normal, with mean 0 and variance 100, nearly flat on the scale
of most coefficients:

```{code-cell} ipython3
posterior = ergmx.bergm(monks, formula, seed=1)
posterior.summary()
```

The table gives each coefficient's posterior mean and standard deviation,
the Monte Carlo error of the mean (naive, and accounting for the
autocorrelation of the chains), the effective sample size, R-hat, and the
posterior quantiles.

## Did the chains converge?

The 8 chains, of 1000 draws each, should agree: R-hat compares the variance
between them with the variance within them, and should be close to 1. Here
it is up to 1.1 (`nodematch.group`), and the draws, autocorrelated, are
worth only about 200 independent ones. Running the chains longer,
`main_iters=3000`:

```{code-cell} ipython3
posterior = ergmx.bergm(monks, formula, main_iters=3000, seed=1)
posterior.summary()
```

R-hats are now below 1.02, and the effective sizes about 500 or more. The plot has
each coefficient's posterior density, the chains' traces, which should look
like noise around a stable level, and the autocorrelations of the draws:

```{code-cell} ipython3
posterior.plot();
```

## The posterior and the MLE

With a nearly flat prior, the posterior mean and standard deviation should
be close to the maximum likelihood estimate and its standard error:

```{code-cell} ipython3
mle = ergmx.ergm(monks, formula, seed=1)
pd.DataFrame({
    "MLE": mle.coef, "Std. Error": mle.stderr, "Posterior mean": posterior.coef, "Posterior SD": posterior.sd,
}).round(2)
```

They are, within a third of a standard error. What the posterior adds is
in its draws, which answer questions directly.

## What the posterior says

`posterior.draws` holds every draw, one row each. The posterior probability
that each coefficient is positive is the share of its draws that are:

```{code-cell} ipython3
draws = pd.DataFrame(posterior.draws, columns=posterior.names)
(draws > 0).mean().round(3)
```

Reciprocity and liking within groups are almost certainly positive. Closure
is positive with probability 0.75: three to one, weak evidence. The odds
ratios' posterior medians and 95% credible intervals are the exponentials
of the coefficients' quantiles:

```{code-cell} ipython3
np.exp(draws).quantile([0.025, 0.5, 0.975]).T.round(2)
```

A nomination within a group has about 7 times the odds of one between
groups, with a 95% posterior probability of 3.5 to 16 times; a nomination
returning another, about 4 times.

## Does the model fit?

{meth}`posterior.gof() <ergmx.BergmFit.gof>`, as Bergm's `bgof()`, compares
the network's degree, shared partner and distance distributions with those
of networks from the *posterior predictive* distribution, each simulated at
a draw of the posterior, so that they include the coefficients'
uncertainty:

```{code-cell} ipython3
posterior.gof(seed=1).plot();
```

The model misses the out-degrees: Sampson asked each monk for the three he
liked most, and almost all named three (a few named four), while the
model's networks spread the nominations from 0 to 7. The in-degrees,
shared partners and distances are reproduced.

## Fixing the out-degrees

The out-degrees are the survey's design, not something to model: the
constraint `"odegrees"` restricts the networks to those with the observed
out-degrees, as in {func}`ergmx.ergm` ([Constraints](../constraints.md)).
They fix the number of nominations, so `edges` leaves the formula:

```{code-cell} ipython3
constrained = "mutual + nodematch('group') + gwesp(0.5, fixed=TRUE)"
final = ergmx.bergm(monks, constrained, constraints="odegrees", main_iters=3000, seed=1)
final.summary()
```

```{code-cell} ipython3
final.gof(seed=1).plot();
```

The model now reproduces the out-degrees, by construction, and the rest as
well. And the conclusion about closure changes: given that each monk names
three, the liked ones' liked ones are liked, with a coefficient around 0.55
and a 95% credible interval above 0. Getting the out-degrees wrong had
hidden it.

## Bringing in a prior

The prior can bring in what is known before the data. Sampson's second
time, `samplk2`, is a natural source: its posterior, with the same
model, summarises what the monks' liking showed earlier, and as the
prior of the third time gives the posterior of both. (The two times are
not independent evidence, being the same monks a few months apart, so
this overstates the information somewhat; it shows how a prior combines
with the data.) `prior_mean` and `prior_sigma` give the normal prior's
mean and covariance:

```{code-cell} ipython3
earlier = ergmx.bergm(datasets.load("samplk2"), constrained, constraints="odegrees", main_iters=3000, seed=1)
informed = ergmx.bergm(monks, constrained, constraints="odegrees", main_iters=3000,
                       prior_mean=earlier.mean, prior_sigma=earlier.cov, seed=1)


def mean_sd(fit):
    return [f"{fit.coef[name]:.2f} ({fit.sd[name]:.2f})" for name in fit.names]


pd.DataFrame({"Time 3": mean_sd(final), "Time 2": mean_sd(earlier), "Time 3, time 2 as prior": mean_sd(informed)},
             index=final.names)
```

With the prior, the posterior standard deviations shrink by about a third.
Closure's estimate falls between the two times' (it was weaker at the
second), and the others barely move.

## Reporting the results

A table of a Bayesian ERGM has the posterior means, standard deviations
and credible intervals; {meth}`posterior.summary().to_frame()
<ergmx.BergmSummary.to_frame>` has them as a pandas DataFrame, to which the
probabilities of a positive coefficient add a column:

```{code-cell} ipython3
report = final.summary().to_frame()[["Mean", "SD", "2.5%", "97.5%"]]
report["P(> 0)"] = (final.draws > 0).mean(axis=0)
report.round(2)
```

pandas' `report.round(2).to_latex()`, `.to_html()` and `.to_markdown()`
give it for a paper.

Next, [A complete temporal ERGM analysis](temporal.md) models the same
monks' liking over the three times.
