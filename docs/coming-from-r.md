# Coming from R

`ergmx` follows R's ergm closely: the same formulas, term names, statistics,
defaults and output, so a model fitted in R can be refitted in Python and the
tables compared line by line.

## Functions

| R (ergm, statnet) | ergmx |
|---|---|
| `ergm(net ~ edges + triangle)` | {func}`ergmx.ergm(g, "edges + triangle") <ergmx.ergm>` |
| `summary(fit)` | {meth}`fit.summary() <ergmx.ErgmFit.summary>` |
| `coef(fit)`, `vcov(fit)` | `fit.coef`, `fit.cov` |
| `logLik(fit)`, `AIC(fit)`, `BIC(fit)` | `fit.loglik`, `fit.aic`, `fit.bic` |
| `AIC(fit1, fit2)`, `anova(fit1, fit2)` | {func}`ergmx.compare(fit1, fit2) <ergmx.compare>` |
| `summary(net ~ edges + triangle)` | {func}`ergmx.summary_stats(g, "edges + triangle") <ergmx.summary_stats>` |
| `simulate(fit, nsim = 10)` | {meth}`fit.simulate(10) <ergmx.ErgmFit.simulate>` |
| `simulate(net ~ edges, coef = -2)` | {func}`ergmx.simulate(g, "edges", [-2]) <ergmx.simulate>` |
| `gof(fit)`, `plot(gof(fit))` | {meth}`fit.gof() <ergmx.ErgmFit.gof>`, `fit.gof().plot()` |
| `mcmc.diagnostics(fit)` | {meth}`fit.mcmc_diagnostics() <ergmx.ErgmFit.mcmc_diagnostics>`, `.plot()` |
| `ergm(..., estimate = "MPLE")` | `ergmx.ergm(..., estimate="MPLE")` |
| `ergm(..., estimate = "CD")` | `ergmx.ergm(..., estimate="CD")` |
| `control.ergm(init.method = "CD")` | `ergmx.ergm(..., init="CD")` |
| `control.ergm(init = c(-2, 0.1))` | `ergmx.ergm(..., init=[-2, 0.1])` |
| `control.ergm(seed = 1)` | `ergmx.ergm(..., seed=1)` |
| `control.ergm(MCMC.samplesize = 2048)` | `ergmx.ergm(..., samplesize=2048)` |
| `control.ergm(MCMC.interval = 4096)` | `ergmx.ergm(..., interval=4096)` |
| `control.ergm(MCMC.burnin = 65536)` | `ergmx.ergm(..., burnin=65536)` |
| `control.ergm(parallel = 8)` | `ergmx.ergm(..., n_chains=8)` |
| `ergm(..., eval.loglik = FALSE)` | `ergmx.ergm(..., eval_loglik=False)` |
| `ergm(net ~ ..., constraints = ~bd(maxout = 5))` | `ergmx.ergm(g, ..., constraints="bd(maxout=5)")` |
| `ergm(net ~ offset(edges) + ..., offset.coef = -3)` | `ergmx.ergm(g, "offset(edges) + ...", offset_coef=[-3])` |
| `net[i, j] <- NA` | `g.add_edge(i, j, na=True)` (an edge marked missing) |
| `simulate(net ~ ..., constraints = ~degrees)` | `ergmx.simulate(g, ..., constraints="degrees")` |
| `data(faux.mesa.high)` | {func}`ergmx.datasets.load("faux.mesa.high") <ergmx.datasets.load>` |

## Formulas

Formula strings are R's, so `"edges + gwesp(0.5, fixed=TRUE)"`,
`"kstar(2:3)"`, `"nodematch('Grade', diff=TRUE)"`,
`"nodemix('Race', levels2=-c(1, 3))"` and
`"F(~gwesp(0.5, fixed=TRUE), ~!nodematch('Grade'))"` work as written, and so
do constraint strings such as `"~bd(maxout=4) + blocks('level', levels2=2)"`. The
left-hand side of `net ~ ...` is ignored, since the network is the first
argument. Strings can use single or double quotes. Only literal arguments are
accepted: R expressions such as `log(n)` must be computed first.

## What is different

**Networks are igraph or networkx graphs.**
: There is no `network` class. Vertex attributes play the role of
  `net %v% "attr"`, and graph attributes hold the matrices for `edgecov`.

**Parallel chains by default.**
: ergmx runs four MCMC chains in threads (ergm runs one, unless `parallel`
  is set), which also gives the R-hat diagnostic.

**A more accurate log-likelihood.**
: Both estimate it by path sampling, but ergmx integrates with a higher-order
  rule and more samples. Its log-likelihoods, and so AICs and BICs, can differ
  from ergm's by up to about 2 units on dyad-dependent models, ergm's being
  the biased ones (see [](validation.md)).

**Degeneracy stops the fit with an explanation.**
: A {class}`ergmx.DegeneracyError` shows the simulated and observed statistics
  and suggests what to try, after a density guard like ergm's or when the
  estimate stops moving.

**Missing dyads are not imputed for the MPLE.**
: ergm imputes them at random before its MPLE, so its MPLE varies between
  runs; ergmx treats them as non-ties in the change statistics. Both are
  only starting values: the MLEs agree.

**The goodness of fit of networks with missing dyads uses imputations.**
: ergmx compares the simulated networks with networks imputed from the model
  given the observed dyads. ergm 4.12 computes such imputations but compares
  with the network with missing dyads as non-ties.

**Not yet available.**
: Curved ERGMs (geometrically weighted terms with an estimated decay),
  `bd()` bounds by attribute, valued and bipartite networks, and many of
  ergm's terms and constraints.
