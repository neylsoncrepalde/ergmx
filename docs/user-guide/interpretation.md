---
file_format: mystnb
kernelspec:
  name: python3
---

# Interpreting and reporting results

An ERGM's coefficients are changes in the log-odds of a tie, given the rest
of the network, per unit of each term's statistic. This page turns them into
quantities that are easier to read and to report: odds ratios, tie
probabilities and marginal effects, and tables for papers.

```{code-cell} ipython3
import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")
homophily = ergmx.ergm(mesa, "edges + nodematch('Grade') + nodematch('Race') + nodefactor('Sex')")
closure = ergmx.ergm(
    mesa, "edges + nodematch('Grade') + nodematch('Race') + nodefactor('Sex') + gwesp(0.5, fixed=TRUE)",
    seed=1,
)
```

## Odds ratios and confidence intervals

{meth}`~ergmx.ErgmFit.odds_ratios` exponentiates the coefficients: the factor
by which one more unit of a statistic multiplies the conditional odds of a
tie. {meth}`~ergmx.ErgmFit.confint` gives Wald intervals of the coefficients,
as R's `confint()`:

```{code-cell} ipython3
closure.odds_ratios()
```

Students of the same grade have about 7 times the odds of a friendship of
other students with the same friends, race and sex. A first shared friend
adds one unit to gwesp (later ones add less, with this decay), which
multiplies the odds by about 3.4.

## Tie probabilities

{meth}`~ergmx.ErgmFit.predict` gives every dyad's probability of a tie, as
R's `predict(fit)`. By default it is *conditional*: given the rest of the
observed network, computed exactly from the dyad's change statistics, so it
accounts for the friends two students already share. With
`conditional=False` it is the share of networks simulated from the model in
which the dyad is a tie:

```{code-cell} ipython3
probabilities = closure.predict()
probabilities
```

`.matrix()` arranges them in an n x n array, `.to_frame()` in a pandas
DataFrame, and `.mean_by(attribute)` averages them by pairs of levels:

```{code-cell} ipython3
probabilities.mean_by("Grade")
```

As in ergm, conditional probabilities ignore the sample space constraints;
unlike ergm's formula method, they include dyads whose value is missing,
which is how to predict them.

## Average marginal effects

Odds ratios are multiplicative, and depend on the baseline odds. The
*average marginal effect* of a term ([Duxbury 2023](https://doi.org/10.1177/0049124120986178), R's [ergMargins](https://CRAN.R-project.org/package=ergMargins)) is the
change in a dyad's conditional tie probability per unit of the term's
statistic, theta * p (1 - p), averaged over the dyads:

```{code-cell} ipython3
closure.marginal_effects()
```

A shared grade adds about 1.4 percentage points to the probability of a
friendship, against an average probability of 0.9%. The standard errors
use the delta method; ergMargins holds the tie probabilities fixed in it,
which ergmx doesn't, so ergmx's standard errors differ somewhat from
ergMargins' (the effects are the same).

## Tables for papers

{func}`ergmx.table` puts models side by side, as R's texreg: printed, it is
`screenreg()`'s table, character for character.

```{code-cell} ipython3
results = ergmx.table(homophily, closure, names=["Homophily", "Closure"])
results
```

In a notebook it shows as HTML. `.to_latex()`, `.to_html()` and
`.to_markdown()` give the table in each format (the first two as texreg's
`texreg()` and `htmlreg()` write them):

```{code-cell} ipython3
print(results.to_markdown())
```

`rename=` relabels coefficients (`{"nodematch.Grade": "Same grade"}`),
`omit=` leaves some out, `digits=` and `stars=` change the numbers and the
significance thresholds. {meth}`ErgmFit.to_frame() <ergmx.ErgmFit.to_frame>`
returns the coefficient table as a pandas DataFrame, with R's columns.
