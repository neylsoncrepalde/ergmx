---
file_format: mystnb
kernelspec:
  name: python3
---

# Samples of networks

Many studies observe not one network but many: the friendships in each
classroom of a school, the contacts within each household of a survey, the
advice networks of several firms. Fitting one ERGM per network gives as many
estimates as networks, each noisy if the networks are small. Fitting a single
model to all of them, as R's [ergm.multi](https://github.com/statnet/ergm.multi)
does ([Krivitsky, Coletti and Hens 2023](https://doi.org/10.1080/01621459.2023.2242627)), pools their information, and can let
the effects depend on each network's characteristics.

The Goeyvaerts data ([Goeyvaerts et al. 2018](https://doi.org/10.1098/rspb.2018.2201)), from ergm.multi, are the contacts within 318 households
of Flanders and Brussels, each from a one-day contact diary:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

households = datasets.load("Goeyvaerts")   # a list of igraph.Graph
print(datasets.describe("Goeyvaerts"))
households[0].vs["role"], households[0]["weekday"]
```

Each network has vertex attributes (`age`, `gender`, `role`) and graph
attributes (`weekday`, whether the diary was kept on a weekday; `included`,
whether the original study analysed it). Following the study, we keep the
included households observed on a weekday:

```{code-cell} ipython3
weekday = [g for g in households if g["included"] and g["weekday"]]
len(weekday)
```

## Networks() and N()

{func}`ergmx.Networks` combines networks for a joint model, and the operator
{func}`~ergmx.N` evaluates terms on each network, as in ergm.multi. With
`N(~edges + triangle)`, the statistics are the sums over the households of
their edges and triangles, so each coefficient is the same in every
household:

```{code-cell} ipython3
networks = ergmx.Networks(weekday)
ergmx.summary_stats(networks, "N(~edges + triangle)")
```

The networks are the blocks of one larger network, with no ties possible
between them: each household's model is the same ERGM, and the households are
independent. Terms outside `N()` are evaluated on that combined network, as
in ergm.multi; for most terms (`edges`, `triangle`, `nodematch`...) that is
the same as summing them, but not for terms that count pairs of vertices,
such as `dsp`, which would count the pairs in different households too.

## Effects that depend on the network

`N()`'s second argument, `lm`, is a linear model for the coefficients, in R's
`lm()` syntax, over network-level attributes: each network's graph
attributes, `n` (its number of vertices), `.NetworkID` (1, 2...) and
`.NetworkName`. `N(~edges, ~I(n <= 3) + I(n >= 5))` has three parameters:
the edges coefficient of medium households (3 to 4 members, the intercept),
and how much higher it is in small and in large ones. This is the model of
ergm.multi's vignette, with the terms ergmx has:

```{code-cell} ipython3
formula = (
    "N(~edges, ~I(n <= 3) + I(n >= 5)) "
    "+ N(~kstar(2) + nodematch('gender') + absdiff('age')) "
    "+ N(~triangle, ~I(n >= 6))"
)
fit = ergmx.ergm(networks, formula, seed=1)
fit.summary()
```

Contacts are denser in small households and sparser in large ones. They
close triangles strongly: two members in contact with a third tend to be in
contact with each other, less so in the largest households. And they are
more likely between members of different ages (parents and children) than
the rest of the model predicts. R's ergm.multi gives the same estimates, in about
eight times as long (see [](../validation.md)).

The names follow ergm.multi: `N(1)~edges` for the intercept, then each
column of the linear model, named as R's `model.matrix()` names it, such as
`N(I(n <= 3)TRUE)~edges` for a logical term or `N(log(n))~edges` for a
numeric one. Factors and character attributes get one column per level but
the first; `~0 + factor(.NetworkID)` gives each network its own coefficient.

The `lm` formula accepts attributes, arithmetic, comparisons, `&`, `|`, `!`,
`I()`, `log()`, `exp()`, `sqrt()`, `abs()`, `factor()` and `offset()`;
interactions (`a:b`) are not supported yet.

`N()`'s other arguments are ergm.multi's. `subset` restricts the terms to
some networks, an expression of their attributes such as `~n >= 4` (or
logical values, or indices): the others contribute nothing, and the linear
model's levels are those of the networks kept. `offset` adds a known amount
to every coefficient of the formula in each network, such as `~log(n)`: the
statistics get companions `offset1`, `offset2`..., whose coefficients are
fixed at 1. `label` names the operator in the statistics' names,
`N(<label>,1)~edges`, which tells apart the parameters of several `N()`
terms:

```{code-cell} ipython3
ergmx.summary_stats(networks, "N(~edges, subset=~n >= 4, label='big') + N(~edges, offset=~log(n))")
```

## Curved terms

A curved term inside `N()`, such as `gwesp(0.5)`, has parameters that the
linear model predicts for each network, like any term's:
`N(~edges + gwesp(0.5), ~log(n))` estimates how the gwesp coefficient and its
decay change with the network size. Because a curved term's statistics enter
the likelihood nonlinearly in its parameters, its statistics can't be summed
over the networks: the model keeps each network's histogram counts instead,
named `N#1~esp#1`, `N#1~esp#2`... as in ergm.multi.

## Checking and using the fit

Everything else works as for one network: {meth}`~ergmx.ErgmFit.mcmc_diagnostics`,
{func}`ergmx.compare`, the log-likelihood (summed over the networks) and BIC
(with the number of dyads within networks). {meth}`~ergmx.ErgmFit.gof`
compares the distributions summed over the networks:

```{code-cell} ipython3
fit.gof(seed=1, stats=["degree", "espartners", "model"]).plot();
```

Distances and shared partners only count pairs of vertices in the same
network; ergm's `gof()` also counts the pairs in different networks, as
unreachable. {meth}`~ergmx.ErgmFit.simulate` returns, for each simulation,
a list with one network per household:

```{code-cell} ipython3
first = fit.simulate(1, seed=1)[0]
len(first), [g.ecount() for g in first[:5]], [g.ecount() for g in weekday[:5]]
```

## Goodness of fit network by network

Summed distributions can hide a model that fits some networks and not
others. {func}`ergmx.gofN`, as ergm.multi's `gofN()`, compares each network's
statistics with those simulated from the fit: their mean (the fitted value),
variance, and the Pearson residual, (observed - fitted) / sd. By default the
statistics are the model's, each network's share; `GOF=` checks others. Its
summary describes the residuals over the networks, whose variance is near 1
for a model that fits:

```{code-cell} ipython3
by_household = ergmx.gofN(fit, "edges + triangle + kstar(2)", seed=1)
by_household.summary()
```

Indexing by statistic gives the table of the networks (`.to_frame()` as a
pandas DataFrame), and `plot()` the residuals against the fitted values (or
`against=`, an expression of the networks' attributes) and their
scale-location plot, with a weighted local regression to show a trend and
the most extreme networks labelled:

```{code-cell} ipython3
by_household.plot(["edges", "triangle"]);
```

`summary(by=)` splits the summary by an attribute of the networks, such as
`by="~n"`, and `subset=` keeps some networks. With missing dyads, the
observed statistics are averaged over networks imputed from the model, as
in ergm.multi.

ergm.multi's `gofN()` leaves out the statistics of the empty network, so it
reports `degree0` and `isolates` minus the network size; ergmx reports the
statistics themselves, and warns (`ErgmDifferenceWarning`) when the
statistics checked have this difference. Its default interval between
simulated networks is three times the number of dyads, where ergm.multi uses
the fit's (1024 by default): with many small networks, each network's
simulated statistics are then autocorrelated, and its fitted values and
residuals about twice as noisy as `nsim` independent draws.
