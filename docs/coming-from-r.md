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
| `tergm(g ~ Form(~edges) + Persist(~edges), targets = ~edges + mean.age, target.stats = c(20, 10), estimate = "EGMME")` | `ergmx.tergm(g, "Form(~edges) + Persist(~edges)", estimate="EGMME", targets="edges + mean.age", target_stats=[20, 10])` |
| `tergm(..., control = control.tergm(CMLE.NA.impute = "next"))`, `NetSeries(..., NA.impute = "next")` | `ergmx.tergm(..., na_impute="next")`, `ergmx.NetSeries(..., na_impute="next")` |
| `gofN(fit, GOF = ~edges + triangle)` (ergm.multi) | {func}`ergmx.gofN(fit, "edges + triangle") <ergmx.gofN>` |
| `lm.gofN(edges ~ n, data = g)` (ergm.multi) | {func}`ergmx.lm_gofN("edges ~ n", g) <ergmx.lm_gofN>` |
| `ergm(net ~ sum + nonzero, response = "contexts", reference = ~Poisson)` (ergm.count) | `ergmx.ergm(g, "sum + nonzero", response="contexts", reference="Poisson")` |
| `bergm(net ~ edges + kstar(2))`, `summary(b)`, `plot(b)`, `bgof(b)` (Bergm) | {func}`ergmx.bergm(g, "edges + kstar(2)") <ergmx.bergm>`, `.summary()`, `.plot()`, `.gof()` |
| `san(net ~ edges + triangle, target.stats = c(100, 20))` | {func}`ergmx.san(g, "edges + triangle", [100, 20]) <ergmx.san>` |
| `ergm(net ~ edges + triangle, target.stats = c(100, 20))` | `ergmx.ergm(g, "edges + triangle", target_stats=[100, 20])` |
| `as.egor(net)` (egor) | {meth}`ergmx.EgoData.from_network(g) <ergmx.EgoData.from_network>`; `egor(egos, alters, aaties)`: `ergmx.EgoData(egos, alters, aaties)` |
| `summary(egor ~ edges + degree(0), scaleto = 1000)` (ergm.ego) | {func}`ergmx.ego_stats("edges + degree(0)", data, scaleto=1000) <ergmx.ego_stats>` |
| `ergm.ego(egor ~ edges + nodematch("sex"), popsize = 1000)` | {func}`ergmx.ergm_ego("edges + nodematch('sex')", data, popsize=1000) <ergmx.ergm_ego>` |
| `saveRDS(fit, "fit.rds")`, `readRDS("fit.rds")` | {meth}`fit.save("fit.pkl") <ergmx.ErgmFit.save>`, {func}`ergmx.load_fit("fit.pkl") <ergmx.load_fit>` |
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

**`intransitive` counts intransitive triads, as ergm documents.**
: The same for `intransitive`: documented as intransitive triads (types
  111D, 201, 111U, 021C and 030C), computed by ergm 4.12.0 as intransitive
  triples, two-paths without a shortcut. ergmx counts the triads and warns;
  ergm's statistic is `twopath` minus `ttriple`.

**`dyadcov`'s `utri` and `ltri` are as ergm documents.**
: ergm documents `dyadcov`'s second and third statistics, in directed
  networks, as the dyads in the upper- and lower-triangular asymmetric
  states, but counts a lone tie from the lower- to the higher-numbered
  vertex, in the upper triangle of the adjacency matrix, as `ltri`. ergmx
  follows the documentation and warns: its `utri` is ergm's `ltri`.

**`degcrossprod` is the mean cross-product of degrees, as ergm documents.**
: ergm documents `degcrossprod` as the mean over ties of the product of
  their vertices' degrees, but computes half of it. ergmx follows the
  documentation and warns: its coefficients are half of ergm's.

**`coincidence(active=)` keeps the pairs with at least `active` partners, as ergm documents.**
: ergm's code keeps those with more than `active`; ergmx warns when the two
  differ.

**`degreedist`, `odegreedist` and `idegreedist` keep the distributions, as ergm documents.**
: ergm's MCMC for these constraints, in directed networks, moves the head of
  a tie keeping every out-degree, or its tail keeping every in-degree, so its
  `odegreedist` and `idegreedist` keep the out- or in-degrees themselves
  (as `odegrees`, `idegrees`). ergmx's keep only the distributions, as
  documented; undirected `degreedist` samples the same networks in both.

**The Monte Carlo MLE has a trust region.**
: Both stop by ergm's confidence test (`MCMLE.termination = "confidence"`,
  ergmx's `termination="confidence"`; `"Hummel"` is there too). ergmx also
  checks each step with the next sample: a step that lost log-likelihood is
  replaced by a shorter one, and the next steps are bounded, the bound
  growing back as steps succeed. ergm takes every step, so curved fits that
  ergm leaves stalled in a degenerate region, from a step that overshot,
  converge in ergmx. Decays are bounded below by 0, as ergm's.

**`smalldiff` counts differences up to the cutoff, as ergm's code.**
: ergm's documentation says less than the cutoff, but its code, and so
  ergmx, counts the ties whose difference is at most the cutoff.

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
  `I()`, `log()`, `exp()`, `sqrt()`, `abs()`, `factor()` and `offset()`, with
  R's column names; not interactions (`a:b`). `subset`, `offset`, `label`
  and `contrasts` are ergm.multi's, but `label` as a function is a Python
  function of one statistic's name and one column (R's is vectorized), and a
  numeric `offset` may be a single number (R requires one per network).
  `weights` must be 1, as in ergm.multi.

**gofN() reports degree0 and isolates themselves, from nearly independent draws.**
: ergm.multi's `gofN()` leaves out the empty network's statistics, so its
  observed and fitted `degree0` and `isolates` are the counts minus the
  network size (the residuals are unaffected); ergmx reports the counts and
  warns. ergmx's default interval between simulated networks is three times
  the number of dyads, so that each network's draws are nearly independent;
  ergm.multi's (the fit's, 1024 by default) leaves those of many small
  networks autocorrelated. `gofN()`'s `subset` selects the networks reported.

**Attribute arguments are names.**
: Terms take vertex attributes by name, and `levels=` the specifications of
  ergm's `?nodal_attributes` (values, 1-based indices, negative indices, `TRUE`,
  `NULL`); not functions of attributes (`~Grade > 9`) or `I()`. Covariate
  matrices and reference networks (`edgecov`, `dyadcov`, `hamming`,
  `localtriangle`, `bd(attribs=)`) are graph attributes, arrays or graphs.
  `altkstar` needs `fixed=TRUE`, and `gwdegree(attr=)` a fixed decay, as in
  ergm.

**Dynamic simulation uses tergm's stopping rule and discordant proposals.**
: Each time step's chain runs until the number of changed dyads stops
  growing, as tergm's `MCMC.burnin.min`, `.max`, `.pval` and `.add`
  (`min_steps`, `max_steps`, `pval`, `add`), with tergm's discordTNT proposal
  mixed into the TNT and triadic proposals. A fit whose coefficients vary
  over time (`lm=~.Time`) continues its trend when simulated forward, at
  `.Time + k .TimeDelta`.

**The EGMME follows tergm's algorithm, with its own convergence check.**
: Starting values from EpiModel's approximation, a gradient by central
  differences under common random numbers, stochastic approximation with
  Polyak averaging, and the delta method's standard errors, as tergm. The
  fit is reported as converged when the shift of the coefficients that would
  bring the final run's targets to theirs is under half a standard error.
  Tie ages start at 1, as in tergm.

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

**ergm.ego's dyad-independent fits are exact.**
: ergm.ego fits every model by Monte Carlo MLE; ergmx fits dyad-independent
  ones exactly (from the egos' estimated statistics), so their estimates
  differ from R's by R's Monte Carlo error, and their standard errors use
  the exact information. Its size adjustment is `offset(edges)` (and
  `offset(transitiveties)`, with triadic terms) where ergm.ego has one
  `offset(netsize.adj)`; `adjust_size=False` is R's `popsize = I(...)`. R
  reports no covariance of non-scaling statistics (`meandeg`) with the others;
  ergmx does.

**Valued `transitiveties` follows the binary term in undirected networks.**
: ergm 4.12's valued `transitiveties(threshold=)` counts every tie above the
  threshold in an undirected network (78 of zach's 78, where 67 have a
  shared partner); ergmx counts those with a two-path above it, as the
  binary term and directed networks do, and warns. ergm's valued
  `cyclicalties` has no change statistic (it stops with an error); ergmx's
  counts the ties with a two-path back. Valued models have no MPLE in either.

**Bergm's auxiliary networks are longer by default.**
: Bergm draws each auxiliary network with 1000 MCMC proposals, which leaves
  those of networks beyond about 50 vertices close to the observed one, and
  the posterior wider and shifted. ergmx's `aux_iters` default is at least
  one proposal per dyad; `aux_iters=1000` reproduces Bergm's. ergmx updates
  half the chains at a time (Bergm, one at a time), with the auxiliary
  networks of each half drawn in parallel; the samplers differ, the
  posterior is the same. `bergm()` takes missing dyads directly (Bergm's
  `bergmM()`). Bergm's `evidence()`, `bergmC()` and `ergmAPL()` are not
  available.

**Not yet available.**
: Valued networks with continuous values (ergm's `StdNormal` reference),
  missing dyads or several networks; tergm's durational *model* terms (as
  opposed to EGMME targets and monitors); and among ergm's binary terms,
  the projection and `Sum`/`Prod`/`Exp`-style operators.
