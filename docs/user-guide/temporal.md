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
`Diss()`) is *separable* (Krivitsky and Handcock 2014): formation and
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
list.

With a dyad-independent model, the CMLE is a logistic regression and exact:
`Form(~edges) + Persist(~edges)` gives the log-odds that a non-tie became a
tie, and that a tie persisted. Dyad-dependent models are fitted by Monte
Carlo MLE; their samples mix, as in tergm, proposals that toggle a dyad that
differs from the previous network, which undo formations and dissolutions
efficiently. Diagnostics, goodness of fit and model comparison work as for
other fits.

## Simulating the process

A fitted temporal model describes a process: start from a network, and draw
each next one from the model given the current one.
{meth}`fit.simulate(time_slices=...) <ergmx.ErgmFit.simulate>` runs it forward
from the last network of the series (or `nw_start="first"`, a position, or a
network), as tergm's `simulate(fit, nw.start=, time.slices=)`:

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

tergm's other estimator, EGMME, which fits a process to a single network and
the durations of its ties, is not supported.
