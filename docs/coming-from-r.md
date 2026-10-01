# Coming from R

:::{important}
**Last updated on 1 October 2026**, against ergm 4.12.0, the version on CRAN
then, and its development version on GitHub (4.13.0-8214, commit
[a85e6a9](https://github.com/statnet/ergm/tree/a85e6a9839e711286f59d95b16c14733929ba0ec)),
ergm.multi 0.3.0 and tergm 4.2.2. The differences below describe these
packages at that point; later versions may have changed them.
:::

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
| `ergm(Networks(g1, g2) ~ N(~edges, lm = ~log(n)))` | `ergmx.ergm(ergmx.Networks(g1, g2), "N(~edges, lm=~log(n))")` |
| `tergm(list(g1, g2, g3) ~ Form(~edges) + Persist(~edges), estimate = "CMLE")` | {func}`ergmx.tergm([g1, g2, g3], "Form(~edges) + Persist(~edges)") <ergmx.tergm>` |
| `ergm(NetSeries(g1, g2) ~ Form(~edges) + Diss(~edges))` | `ergmx.ergm(ergmx.NetSeries(g1, g2), "Form(~edges) + Diss(~edges)")` |
| `simulate(fit, nw.start = "last", time.slices = 10)` | {meth}`fit.simulate(time_slices=10) <ergmx.ErgmFit.simulate>` |
| `simulate(g ~ Form(...) + Persist(...), coef = c(...), time.slices = 10, dynamic = TRUE)` | {func}`ergmx.simulate_dynamic(g, "Form(...) + Persist(...)", [...], 10) <ergmx.simulate_dynamic>` |
| `data(Goeyvaerts)` (ergm.multi) | `ergmx.datasets.load("Goeyvaerts")`, a list of graphs |
| `predict(fit)`, `predict(fit, conditional = FALSE)` | {meth}`fit.predict() <ergmx.ErgmFit.predict>`, `fit.predict(conditional=False)` |
| `predict(net ~ edges + mutual, eta = c(-2, 1.8))` | {func}`ergmx.predict(g, "edges + mutual", [-2, 1.8]) <ergmx.predict>` |
| `confint(fit)`, `exp(coef(fit))` | {meth}`fit.confint() <ergmx.ErgmFit.confint>`, {meth}`fit.odds_ratios() <ergmx.ErgmFit.odds_ratios>` |
| `ergMargins::ergm.AME(fit, "nodematch.Grade")` | {meth}`fit.marginal_effects() <ergmx.ErgmFit.marginal_effects>` (every term) |
| `ergm(net ~ S(~edges + gwesp(0.5, fixed = TRUE), ~level == "A"))` | `ergmx.ergm(g, "S(~edges + gwesp(0.5, fixed=TRUE), ~level == 'A')")` |
| MPNet's multilevel effects (TXAX, C4AXB...) | {func}`~ergmx.txax`, {func}`~ergmx.c4axb`... (see [](user-guide/multilevel.md)) |
| `screenreg(list(fit1, fit2))` | {func}`print(ergmx.table(fit1, fit2)) <ergmx.table>` |
| `texreg(...)`, `htmlreg(...)` | `ergmx.table(...).to_latex()`, `.to_html()` |

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
  given the observed dyads. ergm computes such imputations but compares with
  the network with missing dyads as non-ties (in 4.12.0 and the development
  version).

**Bipartite networks are declared, not ordered.**
: ergm's bipartite networks have the first mode first (`bipartite = n1`);
  ergmx reads each vertex's mode from an attribute (`bipartite="type"`), in
  any order.

**Curved terms start from their decay argument.**
: In ergm, `gwesp(0.5)` without `fixed=TRUE` ignores the 0.5 (it warns, and
  the start comes from `init`); ergmx starts the decay there. Both estimate
  it. When a network has more shared partners (or degree) than the cutoff,
  ergm stops with an error; ergmx counts them in an overflow statistic.

**`transitive` counts transitive triads, as ergm documents.**
: ergm documents `transitive` as a count of transitive triads but computes
  transitive triples, the same as `ttriple`, in 4.12.0 and the development
  version. ergmx counts the triads and warns; to reproduce a model fitted in
  R with `transitive`, use `ttriple`, which gives ergm's statistic and
  estimates exactly.

**Edgewise RTP statistics are right, as in ergm without its cache.**
: ergm 4.12.0's shared-partner cache, used by default, reads the edgewise
  RTP statistics (`esp`, `gwesp` and `nsp` with `type = "RTP"`) with the
  wrong key, so their values depend on the order of the vertices. The
  development version fixes it
  ([statnet/ergm#656](https://github.com/statnet/ergm/pull/656)). Until the
  fix reaches CRAN, `term.options = list(cache.sp = FALSE)` in `summary()` or
  `ergm()` gives the right values in ergm 4.12.0, which match ergmx's.

**Several networks are pooled in goodness of fit.**
: For models of several networks (`Networks()`, `NetSeries()`), ergmx's
  `gof()` sums the distributions over the networks and only counts pairs of
  vertices in the same network; ergm's counts the pairs in different
  networks too, as unreachable (with distance `Inf`) and without shared
  partners.

**N()'s linear models cover the common cases.**
: `lm` formulas take attributes, arithmetic, comparisons, `&`, `|`, `!`,
  `I()`, `log()`, `exp()`, `sqrt()`, `abs()` and `factor()`, with R's column
  names; not interactions (`a:b`), nor `N()`'s `subset`, `offset` and `label`
  arguments.

**Dynamic simulation uses tergm's stopping rule and discordant proposals.**
: Each time step's chain runs until the number of changed dyads stops
  growing, as tergm's `MCMC.burnin.min`, `.max`, `.pval` and `.add`
  (`min_steps`, `max_steps`, `pval`, `add`), with tergm's discordTNT proposal
  mixed into the TNT and triadic proposals. Linear models that vary over
  time (`lm=~.Time`) can be fitted but not simulated forward.

**Predictions include missing dyads; unconditional ones respect the constraints.**
: ergm's `predict()` for a formula leaves out the dyads whose value is
  missing, and simulates unconditional probabilities without the fit's
  constraints. ergmx predicts missing dyads (conditional on the observed
  network, with missing dyads as non-ties) and simulates with the
  constraints. ergm's unconditional predictions also leave out the dyads
  never tied in the simulations; ergmx lists every dyad.

**Marginal effects' standard errors use the full delta method.**
: ergMargins' `ergm.AME()` holds the tie probabilities fixed when it
  differentiates the average marginal effect; ergmx also counts how they
  change with the coefficients. The effects are the same; the standard
  errors differ, by up to a quarter on the models tested.

**Not yet available.**
: `bd()` bounds by attribute, valued networks, tergm's EGMME estimator and
  duration terms (`edgeages`, `mean.age`), ergm.multi's `gofN()`, and many of
  ergm's terms and constraints.
