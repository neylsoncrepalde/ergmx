---
file_format: mystnb
kernelspec:
  name: python3
---

# Bayesian ERGMs

{func}`ergmx.bergm`, as R's Bergm ([Caimo and Friel
2011](https://doi.org/10.1016/j.socnet.2010.09.004)), samples the posterior
distribution of a model's coefficients, given a normal prior, rather than
maximizing the likelihood. The posterior gives every coefficient's
uncertainty directly, with its correlations and skewness, and a prior can
bring in what is known before the data.

The likelihood of an ERGM can't be evaluated (its normalizing constant sums
over every network), so neither can the usual Metropolis-Hastings ratio.
The *exchange algorithm* draws, for each proposed coefficient vector, an
auxiliary network from the model at those coefficients, by MCMC, and the
normalizing constants cancel in the acceptance ratio. A population of chains
proposes moves along the differences between chains (adaptive direction
sampling), so they learn the posterior's scale and correlations together.

```{code-cell} ipython3
import ergmx
from ergmx import datasets

flomarriage = datasets.load("flomarriage")
posterior = ergmx.bergm(flomarriage, "edges + kstar(2)", seed=1)
posterior.summary()
```

The table gives the posterior means and standard deviations, the Monte Carlo
standard errors of the means (naive, and accounting for the chains'
autocorrelation), the effective sample sizes, R-hat (near 1 when the chains
agree), and the posterior quantiles. `posterior.coef`, `posterior.sd`,
`posterior.cov` and `posterior.quantiles()` return them; `posterior.draws`
holds every draw, and `posterior.chains` each chain's.

```{code-cell} ipython3
posterior.plot();
```

## Settings

`prior_mean` and `prior_sigma` set the normal prior (mean 0 and covariance
100 I by default, as Bergm's: nearly flat on the scale of most
coefficients). `burn_in` and `main_iters` are each chain's discarded and kept
iterations, `nchains` the number of chains (twice the number of
coefficients by default), and `gamma` and `v_proposal` scale the proposals.

`aux_iters` is the number of MCMC proposals that draw each auxiliary
network, from the observed one. Too few leave the auxiliary networks too like
the observed network, which widens the posterior and shifts it. ergmx's
default is at least one proposal per dyad (and at least Bergm's 1000):
Bergm's default of 1000 is too few for networks beyond about 50 vertices.
On faux.mesa.high's 20,910 dyads, with `edges + nodematch('Grade') +
gwesp(0.5, fixed=TRUE)`, 1000 proposals put the gwesp coefficient's
posterior mean at 1.40 with a standard deviation of 0.11; 30,000 give 1.25
and 0.10, and 100,000, 1.24 and 0.08: the MLE and its standard error (1.25
and 0.08), as they should be with a nearly flat prior.

Missing dyads are imputed by every chain, at each iteration, from the model
at its current coefficients, as Bergm's `bergmM()` does; the posterior is
then that of the observed dyads. Constraints and offsets work as in
{func}`ergmx.ergm`.

## Checking the model

{meth}`posterior.gof() <ergmx.BergmFit.gof>`, as Bergm's `bgof()`, compares
the observed network's degree, shared partner and distance distributions
with those of networks from the *posterior predictive* distribution, each
simulated at a posterior draw, so they include the coefficients'
uncertainty:

```{code-cell} ipython3
posterior.gof(50, seed=1).plot();
```

{meth}`posterior.simulate() <ergmx.BergmFit.simulate>` returns such networks.

Bergm updates the chains one after the other; ergmx updates half of them
at a time, with proposals from the other half, and draws their auxiliary
networks in parallel. Both sample the same posterior. Bergm's model
evidence (`evidence()`) and calibrated pseudo-likelihood (`bergmC()`) are
not available.
