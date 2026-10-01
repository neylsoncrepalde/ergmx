---
file_format: mystnb
kernelspec:
  name: python3
---

# Model comparison

Every fit has a log-likelihood, and so an AIC and a BIC:

- for dyad-independent models, the exact one;
- for the others, an estimate with its Monte Carlo standard error, computed
  after the fit (`eval_loglik=False` skips it).

{func}`ergmx.compare` puts fits of the same network side by side, like R's
`AIC()` and `anova()`. Consecutive nested models, each with all the terms of
the previous one, also get a likelihood-ratio test:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")
homophily = ergmx.ergm(mesa, "edges + nodematch('Grade') + nodematch('Race')")
triads = ergmx.ergm(
    mesa, "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)", seed=1
)
degrees = ergmx.ergm(
    mesa,
    "edges + nodematch('Grade') + nodematch('Race') + gwdegree(0.5, fixed=TRUE) + gwesp(0.5, fixed=TRUE)",
    seed=1,
)
ergmx.compare(homophily, triads, degrees)
```

`dAIC` is each model's AIC minus the smallest. Triadic closure (`gwesp`)
improves the fit enormously. Adding `gwdegree` lowers the AIC a little more and
is significant in the likelihood-ratio test, but BIC, which penalizes
parameters more, prefers the second model.

## How the log-likelihood is estimated

The log-likelihood of a dyad-dependent model has a normalizing constant
that sums over every possible network. `ergmx` estimates it by *path
sampling*, as ergm does: it starts from the model's dyad-independent terms,
whose log-likelihood is exact, and integrates the change in log-likelihood
along a straight path to the estimate, with MCMC samples at points along the
path.

ergm integrates with a 16-point midpoint rule, which is biased: by 0.1 to 2
log-likelihood units on the models of the [validation](../validation.md).
`ergmx` uses 33 points and the Euler–Maclaurin corrected trapezoidal rule,
whose error falls as the inverse fourth power of the number of points rather
than the square. Its estimates are unbiased against exact enumeration on
small networks.

The standard error is printed with the log-likelihood. Differences between
models much larger than it are reliable; differences of the same size are
noise. To make it smaller, sample more:

```python
ergmx.ergm(mesa, formula, bridge_samplesize=1024, bridges=64)
```
