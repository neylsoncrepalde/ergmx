---
file_format: mystnb
kernelspec:
  name: python3
---

# Convergence and MCMC diagnostics

A Monte Carlo MLE is only as good as its MCMC sample. At the estimate,
networks simulated from the model should have, on average, the observed
statistics, and the chains should have explored the same distribution.
{meth}`ErgmFit.mcmc_diagnostics() <ergmx.ErgmFit.mcmc_diagnostics>` checks
both on the sample of the last iteration, like R's `mcmc.diagnostics()`:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")
fit = ergmx.ergm(
    mesa, "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)", seed=1
)
diagnostics = fit.mcmc_diagnostics()
diagnostics
```

What to look for:

Mean deviations
: Near zero, compared with the SD: the simulated networks reproduce the
  observed statistics. With thousands of effective samples, Hotelling's test
  flags even negligible deviations, so look at their size too.

Effective size
: How many independent samples the autocorrelated sample is worth. Small
  values (below a hundred or so) make the estimate and its standard errors
  noisy; raise `interval` or `samplesize`.

R-hat
: Compares the parallel chains. Close to 1 means they agree; above 1.1, they
  are exploring different parts of the distribution, a sign of poor mixing or
  of a degenerate model.

Geweke z-scores
: Compare the start and the end of each chain. About one in twenty beyond ±2
  is expected by chance; many more suggest the chains had not reached their
  stationary distribution.

The plots show the same: traces should look like noise around zero, and the
chains' densities should overlap.

```{code-cell} ipython3
diagnostics.plot();
```

## Degenerate models

Some models put almost all their probability on nearly empty or nearly
complete networks: they are *degenerate*. No coefficients make their
simulated networks look like the observed one, and the estimation can't
converge. `ergmx` detects this and stops with a {class}`ergmx.DegeneracyError`
that says why:

```{code-cell} ipython3
samplk3 = datasets.load("samplk3")

try:
    ergmx.ergm(samplk3, "edges + mutual", init=[4.0, 0.0], seed=1, stall_iterations=2)
except ergmx.DegeneracyError as error:
    print(error)
```

Here the starting coefficients are just poor, which `init="MPLE"` (the
default) avoids. Three checks raise the error:

- **The density guard**, as in ergm: a simulated network with many more edges
  than the observed one (`Control.density_guard`, by default about 20 times,
  and at least `density_guard_min` edges). The chains stop as soon as it
  trips, so it costs little time.
- **A stalled estimation**: `stall_iterations` consecutive iterations (10 by
  default) in which the observed statistics are far outside the range of the
  simulated ones. The time this takes depends on the model, as later
  iterations use longer chains. `stall_iterations=None` keeps iterating.
- **Chains that barely mix**: the chains move between very different
  regimes, so that longer intervals between samples hardly help. The
  estimation stops rather than lengthen its chains past 16 times their
  starting interval, to more than 2^26 (67 million) proposals per chain and
  iteration, when each effective draw already costs more than `max_sweeps`
  sweeps of the network (proposals per dyad; 300 by default, where
  well-specified models of large networks need at most tens; a small
  network's long chains are cheap, and go on), or when three samples at the
  same coefficients gained little from longer intervals.
  `max_sweeps=None` and `stall_iterations=None` keep sampling.

Degeneracy is usually fixed by changing the model rather than the settings:
`gwesp` with a smaller decay instead of `triangle`, adding `gwdegree` or
attribute terms, or a starting point closer to the MLE (`init="CD"`).
