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

[A complete Bayesian ERGM analysis](quickstart/bayesian.md) goes from the
data to a table of results, with a constrained model and an informative
prior.

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
networks in parallel. Both sample the same posterior.

## Model choice: the evidence

A model's evidence, its marginal likelihood p(y), is the probability of the
network averaged over the prior; the ratio of two models' evidence, the
Bayes factor, says how much more the network supports one than the other.
{func}`ergmx.evidence`, as Bergm's `evidence()` ([Bouranis, Friel and Maire
2018](https://doi.org/10.1080/10618600.2018.1448832)), estimates it from
the *adjusted pseudo-likelihood* ({func}`ergmx.ergm_apl`, Bergm's
`ergmAPL()`): the pseudo-likelihood moved to the MLE, stretched to the
likelihood's curvature there and scaled to its value, a cheap stand-in for
the likelihood that can be evaluated anywhere. Chib and Jeliazkov's method
(`method="CJ"`, the default) estimates the posterior's density at its mean;
power posteriors (`"PP"`) integrate over a ladder of temperatures, and are
slower.

```{code-cell} ipython3
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ergmx.ErgmDifferenceWarning)
    two_stars = ergmx.evidence(flomarriage, "edges + kstar(2)", seed=1)
    wealth = ergmx.evidence(flomarriage, "edges + nodecov('wealth')", seed=1)
two_stars.log_evidence, wealth.log_evidence
```

The two-star model has the larger evidence, by a Bayes factor of
exp(1.1), about 3: weak evidence. The adjustment comes from the Monte Carlo
MLE: its final sample of networks gives the likelihood's curvature, and its
bridge-sampled log-likelihood the scale. Bergm's `evidence()` takes both
from 5 simulated networks, which leaves its estimates noisy and biased: on
the wealth model, whose exact log evidence, -62.92, can be computed (its
pseudo-likelihood is its likelihood), five runs of Bergm's Chib and
Jeliazkov estimate gave -61.0 on average, from -59.2 to -63.8, and ergmx's
gives -62.93 for any seed. With 50 temperatures, power posteriors are off
by about 0.2 (-63.1 here), in ergmx and Bergm: more temperatures (`temps=`)
reduce that.

## Fast posteriors: bergmC

{func}`ergmx.bergmC`, as Bergm's `bergmC()` ([Bouranis, Friel and Maire
2017](https://doi.org/10.1016/j.socnet.2017.03.013)), samples the
pseudo-posterior, which needs no auxiliary networks, and calibrates the
sample to the true posterior: from its mode to the posterior's mode, and
from its curvature to the posterior's. For large networks it is much faster
than {func}`ergmx.bergm`, and the result is a {class}`~ergmx.BergmFit`
with the same summaries and checks:

```{code-cell} ipython3
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ergmx.ErgmDifferenceWarning)
    calibrated = ergmx.bergmC(datasets.load("samplk3"), "edges + mutual + nodematch('group')", seed=1)
calibrated.summary()
```

The calibration is exact when the posterior is normal, and good when it is
nearly so, as here (the exact posterior means, from `bergm()`, are -2.71,
1.39 and 2.07). It is poor for skewed posteriors: for the two-star model of
the Florentine marriages, its mean for `edges` is -1.7, the exact one -1.1.
