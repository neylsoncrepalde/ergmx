---
file_format: mystnb
kernelspec:
  name: python3
---

# Networks over time

A panel study observes the same people's network several times: the
friendships in a school each year, the advice ties in a firm before and after
a merger. A *temporal* ERGM, as R's [tergm](https://github.com/statnet/tergm)
fits, models each network given the one before it: which ties form, which
persist, and which dissolve.

Sampson's monks named the brothers they liked at three times, before a crisis
split the monastery:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

waves = [datasets.load(f"samplk{t}") for t in (1, 2, 3)]
[g.ecount() for g in waves]
```

[A complete temporal ERGM analysis](quickstart/temporal.md) goes from the
three waves to a table of results, with the survey's fixed out-degrees and
a simulation of the process.

## Formation and persistence

tergm's operators evaluate terms on views of each transition, from a
previous network to the current one:

| Operator | Evaluates its terms on | A positive coefficient means |
|---|---|---|
| {func}`~ergmx.Form` | the union of the previous and the current network | more ties form |
| {func}`~ergmx.Persist` | their intersection: the ties that persisted | more ties persist |
| {func}`~ergmx.Diss` | the same, with the statistics negated | more ties dissolve |
| {func}`~ergmx.Cross` | the current network | (a cross-sectional effect) |
| {func}`~ergmx.Change` | the dyads that changed | more change |

`Form(~edges)` only changes when a tie forms, and `Persist(~edges)` when one
persists or dissolves, so a model of only `Form()` and `Persist()` (or
`Diss()`) is *separable* ([Krivitsky and Handcock 2014](https://doi.org/10.1111/rssb.12014)): formation and
dissolution are independent given the previous network, each an ERGM of its
own. Terms outside the operators describe the current network, as `Cross()`.

{func}`ergmx.tergm` fits the model to the transitions of a series by
conditional maximum likelihood (CMLE), as `tergm(..., estimate="CMLE")`:

```{code-cell} ipython3
fit = ergmx.tergm(
    waves,
    "Form(~edges + mutual + gwesp(0.5, fixed=TRUE)) + Persist(~edges + mutual)",
    seed=1,
)
fit.summary()
```

Formation is rare (the `Form(1)~edges` coefficient), but much more likely
for a tie that reciprocates one (`mutual`) or closes two-paths (`gwesp`).
Liking, once there, tends to persist, a little more so when reciprocated.
R's tergm gives the same estimates. The names follow tergm, with the linear
model's column as in [`N()`](multiple-networks.md): `Form(1)~edges`.

Every transition has the same coefficients by default. The operators' `lm`
argument lets them change over time, with the attributes `.Time` (the time of
the current network: 1 and 2 here), `.TimeID` (the transition's position) and
`.TimeDelta` (the time since the previous network): `Form(~edges, lm=~.Time)`
estimates a trend in formation. {func}`ergmx.NetSeries` builds the series
with its times, when they are not 0, 1, 2...; `tergm()` builds it from a
list. The operators also take [`N()`](multiple-networks.md)'s `subset`,
`offset` and `label`.

With a dyad-independent model, the CMLE is a logistic regression and exact:
`Form(~edges) + Persist(~edges)` gives the log-odds that a non-tie became a
tie, and that a tie persisted. Dyad-dependent models are fitted by Monte
Carlo MLE; their samples mix, as in tergm, proposals that toggle a dyad that
differs from the previous network, which undo formations and dissolutions
efficiently. Diagnostics, goodness of fit (also by transition, with
{func}`ergmx.gofN`) and model comparison work as for other fits.

## Missing dyads

Missing dyads (`na` edges) of the networks transitioned *to* are treated as
missing, as in a cross-sectional fit. Those of the networks transitioned
*from* (all but the last) are what the next transition is conditioned on,
and must be imputed first, as tergm's `NA.impute`: `na_impute="next"` (each
missing dyad takes its value in the next wave), `"previous"`, `"majority"`
(the more common value of the network's observed dyads), `"0"` or `"1"`;
several apply in turn, so `["previous", "0"]` fills what the previous wave
can't with non-ties. `NetSeries()` and `tergm()` take it:

```python
fit = ergmx.tergm(waves_with_nonresponse, "Form(~edges + mutual) + Persist(~edges)",
                  na_impute="next")
```

## Simulating the process

A fitted temporal model describes a process: start from a network, and draw
each next one from the model given the current one.
{meth}`fit.simulate(time_slices=...) <ergmx.ErgmFit.simulate>` runs it forward
from the last network of the series (or `nw_start="first"`, a position, or a
network), as tergm's `simulate(fit, nw.start=, time.slices=)`. Coefficients
that change over time (`lm=~.Time`) continue their trend: the k-th step
after the last network has the time `.Time + k .TimeDelta` and the position
`.TimeID + k`, and the linear models' predictions for them.

```{code-cell} ipython3
future = fit.simulate(time_slices=20, seed=1, monitor="edges + mutual")
future
```

The result, a {class}`ergmx.DynamicSimulation`, has the networks
(`future.networks`), the model's statistics of each transition
(`future.stats`), the `monitor` formula's statistics of each network, the
ties that formed and dissolved at each step, and their durations:

```{code-cell} ipython3
finished, ongoing = future.durations()
print(f"{len(finished)} ties dissolved after {finished.mean():.1f} steps on average; "
      f"{len(ongoing)} still present")
future.monitor["mutual"]
```

{func}`ergmx.simulate_dynamic` simulates from any coefficients and starting
network. A time step's Markov chain runs as in tergm: from the current
network, until the number of dyads that differ from it stops growing (a test
on its exponentially weighted increments), then as long again. Its length
adapts to the model and the network size; `min_steps` and `max_steps` bound
it, and equal values fix it.

## A process from one network: the EGMME

Often there is a single network, a cross-section, and some knowledge of how
long ties last: a survey of current partnerships, with their mean duration.
tergm's equilibrium generalized method of moments (EGMME) finds a process
whose equilibrium matches both: coefficients of formation and persistence
such that, simulated for a long time, the network has the observed
statistics and its ties the observed ages. `tergm(estimate="EGMME")` takes
the network, the model, and `targets`, a formula of the statistics to match,
which can include statistics of tie ages: `mean.age`, `edge.ages` (their
sum), `edges.ageinterval(from, to)`, `edgecov.ages(x)` and
`nodefactor.mean.age(attr)`, with a tie's age 1 in the step it formed, as in
tergm.

The Florentine marriages, as a process where marriages last ten time steps
on average and families with a common partner marry more readily:

```{code-cell} ipython3
flo = datasets.load("flomarriage")
observed = ergmx.summary_stats(flo, "edges + gwesp(0, fixed=TRUE)")
egmme = ergmx.tergm(
    flo,
    "Form(~edges + gwesp(0, fixed=TRUE)) + Persist(~edges)",
    estimate="EGMME",
    targets="edges + gwesp(0, fixed=TRUE) + mean.age",
    target_stats=[*observed.values(), 10],
    seed=1,
)
egmme
```

The estimate is found as tergm does: from EpiModel's approximation of the
edges coefficients (formation, the log-odds of the density minus the log of
the mean duration; persistence, the log of the duration minus one),
stochastic approximation with Polyak averaging, and the delta method's
standard errors, with the gradient of the targets estimated by central
differences under common random numbers. It needs at least as many targets
as coefficients. The fit's {meth}`~ergmx.EgmmeFit.simulate` runs the
process, and its tie ages are monitors there too:

```{code-cell} ipython3
run = egmme.simulate(time_slices=500, seed=2, monitor="edges + mean.age")
run.monitor["edges"][100:].mean(), run.monitor["mean.age"][100:].mean()
```
