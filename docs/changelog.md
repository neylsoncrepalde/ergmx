# Changelog

## 0.3.1 (2026-10-04)

- **Quick start**: complete analyses of a valued network (the karate club's
  counts of contexts), of egocentric data (a survey of half a high school,
  against the whole network's model), of a Bayesian ERGM (Sampson's monks,
  with fixed out-degrees and a prior from an earlier time), of a temporal
  ERGM (the monks' three times: formation, persistence, a trend, the
  survey's design and the process run forward) and of a bipartite ERGM
  (the Southern Women: activity, popularity and the events' groups).
- {meth}`EgoFit.gof() <ergmx.EgoFit.gof>` now follows ergm.ego's `gof()`:
  its observed degree and edgewise shared partner distributions and model
  statistics are those the egos estimate, per person, rather than those of
  the pseudo-population network made to match the targets. It has
  `"degree"`, `"espartners"` (with the ties among alters) and `"model"`;
  checked against ergm.ego.
- {func}`ergmx.table` takes egocentric fits.
- Fits with many coefficients no longer print numpy's overflow warnings
  from the convergence test.
- The Monte Carlo MLE stops with a {class}`~ergmx.DegeneracyError` when
  chains barely mix and longer intervals don't help (three samples at the
  same coefficients without the effective size growing, the interval at
  least 16 times the starting one): a near-degenerate model whose
  simulated networks move between regimes. It used to keep growing the
  interval, up to 2^20 proposals between samples, for hours. A valued model
  of the karate club with `nodecovar(center=TRUE, transform='sqrt')` is one:
  R's ergm reports a fit, but simulating from it gives networks with three
  times the observed counts. `Control(stall_iterations=None)` keeps
  iterating.
- {meth}`~ergmx.ErgmFit.odds_ratios` of a valued model are labelled as what
  they are: the factor of the probability of a dyad's value relative to the
  value one lower, not the odds of a tie.

## 0.3.0 (2026-10-03)

- **Robust Monte Carlo MLE.** It now stops by ergm's default convergence
  test (`termination="confidence"`): an equivalence test that the estimate
  is within the tolerance region, with the sample growing until the test can
  tell; `termination="Hummel"` keeps the old rule. A trust region checks each
  step with the next sample, and replaces a step that lost log-likelihood by
  one half as long (Fisher scoring's, in curved models), bounding the next
  ones; steps wait for samples of at least `effective_size` effective draws;
  curved models are fitted with their decays held first, then freed; decays
  stay at or above 0, as ergm's. Curved fits inside `N()` that stalled
  on some random paths (some numbers of chains, some platforms) now converge
  on all of them.
- **Terms**: `degcor`, `degcrossprod`, `tripercent` and `coincidence`,
  checked against R. ergm computes half of `degcrossprod`'s documented mean,
  and `coincidence(active=)` keeps the pairs with more than `active`
  partners, not at least: ergmx follows the documentation and warns.
- **Constraints**: `degreedist`, `odegreedist`, `idegreedist` (the degree
  distributions kept, as ergm documents them) and `egocentric`.
- **Several networks**: N()'s `contrasts`, as R's `contrasts.arg`
  (`contr.sum`, `contr.helmert`, `contr.poly`, `contr.SAS`, matrices), and
  {func}`~ergmx.lm_gofN`, ergm.multi's `lm.gofN()`: linear models of
  `gofN()`'s residuals on the networks' attributes. N() accepts `weights` of
  1, as ergm.multi. Backquoted names (``list(`factor(n)` = ...)``) work in
  formula strings.
- The tests run in the continuous integration with 2 and 3 Monte Carlo
  chains too (`ERGMX_TEST_CHAINS`), the random paths of other machines.
- **Networks with given statistics**: {func}`~ergmx.san`, ergm's simulated
  annealing, and `ergm(target_stats=)`, models fitted to target statistics
  (exactly, for dyad-independent ones), as ergm's `target.stats`.
- **Egocentric ERGMs**, as R's ergm.ego, checked against it:
  {class}`~ergmx.EgoData` (egos, alters and the ties among them, or a
  network's census), {func}`~ergmx.ego_stats` (the population statistics
  egocentric data estimate, with the survey, asymptotic, naive, bootstrap
  and jackknife variances) and {func}`~ergmx.ergm_ego` (the fit, on a
  pseudo-population, with the network size adjustment and the egos'
  sampling variance in the standard errors).
- **Valued networks**, as R's ergm with ergm.count, checked against it:
  `ergmx.ergm(g, formula, response="contexts", reference="Poisson")`, with
  the Poisson, binomial, geometric and discrete uniform reference measures,
  ergm's valued terms (`sum`, `nonzero`, the dyad-independent terms with
  `form=`, thresholds, `mutual`, `transitiveweights`, `cyclicalweights`,
  `nodecovar`, `CMP`...), contrastive divergence, the Monte Carlo MLE, the
  log-likelihood and simulation (ergm.count's `DiscTNT` proposals). The
  dataset `zach`, ergm.count's karate club with counts of contexts.
- **Bayesian ERGMs**, as R's Bergm, checked against it and against exact
  posteriors: {func}`~ergmx.bergm`, the approximate exchange algorithm with
  adaptive direction sampling over a population of chains (half of them
  updated in parallel), normal priors, offsets, constraints and missing
  dyads (imputed by every chain, as `bergmM()`); posterior summaries,
  diagnostics (effective sample sizes, R-hat), plots, and the posterior
  predictive goodness of fit (`bgof()`) and simulation. Its auxiliary
  networks get at least one MCMC proposal per dyad by default, where
  Bergm's 1000 bias the posterior of larger networks.

## 0.2.0 (2026-10-02)

- **More of ergm's vocabulary**, checked against R: `degrange`,
  `idegrange`, `odegrange`, `degree1.5` and its in- and out- versions,
  `concurrentties`, `isolatededges`, `density`, `meandeg`, `dyadcov`,
  `hamming`, `attrcov`, `mm`, `diff`, `smalldiff`, the covariate ranges
  (`nodecovrange`...) and distinct neighbour types (`nodefactordistinct`...),
  `altkstar`, the triad census and `balance`, `intransitive`, `simmelian`,
  `nearsimmelian`, `simmelianties`, `transitiveties`, `cyclicalties`,
  `threetrail`, `opentriad`, `localtriangle`, `m2star`, the directed `d*sp`
  aliases, and for bipartite networks `b1degrange`, `b1mindegree`,
  `b1sociality`, `b1starmix`, `b1twostar`, `b1covrange`, `b1factordistinct`
  and their `b2` twins.
- **Term options**: `levels=` (and the older `keep=`, `base=`) for
  `nodematch`, `nodefactor` and the bipartite factors; `by=` and
  `homophily=` for the degree terms and `concurrent`; `attr=` for the star
  and triangle terms, `mutual(same=, by=)`, `asymmetric`, `sociality` and the
  geometrically weighted degrees; `nodes=` for `sender`, `receiver` and
  `sociality`; `b1nodematch(diff=, alpha=, beta=, byb2attr=)`.
- **Interactions** of dyad-independent terms, `a:b` and `a*b`.
- **Constraints**: `edges`, `b1degrees`, `b2degrees`, `Dyads(fix=, vary=)`,
  `fixedas`, `fixallbut`, `observed`, `blockdiag` and `bd(attribs=)`.
- Attributes with missing values are refused with a clear error, as in ergm.
- ergm documents `intransitive` as intransitive triads and `dyadcov`'s
  `utri` as the upper triangle's asymmetric dyads, but computes
  intransitive triples and swaps `utri` and `ltri`: ergmx follows the
  documentation and warns (`ErgmDifferenceWarning`).
- **Multilevel networks**: MPNet's configurations of directed two-level
  networks (in- and out-stars with affiliations, triangles, alternating
  triangles and three-paths of arcs and reciprocated pairs, cross-level
  three-paths, entrainment and exchange four-cycles, alternating stars at
  both ends), and the undirected EXTA, EXTB and ASAXASB; estimated decays
  (`fixed=FALSE`) for the configurations with one alternating part;
  goodness of fit by level, `gof(by="level")`; and `S()` between two sets of
  a directed network, the arcs from the first to the second, as in ergm.
- **Datasets**: `labs_sim`, a multilevel network of 120 researchers and 30
  laboratories simulated from a known model, with effects within each level
  and across levels.
- **tergm's EGMME**: `tergm(network, ..., estimate="EGMME", targets=,
  target_stats=)` fits a process to a single network and the ages of its
  ties, with tergm's algorithm (`EgmmeFit`); tergm's statistics of tie ages,
  `edge.ages`, `mean.age`, `edges.ageinterval`, `edgecov.ages` and
  `nodefactor.mean.age`, as targets and as monitors of dynamic simulations.
- **Series of networks**: forward simulation of fits whose coefficients vary
  over time (`lm=~.Time`); missing dyads in the networks transitioned from,
  imputed as tergm's `NA.impute` (`na_impute=` of `NetSeries()` and
  `tergm()`), and in the networks transitioned to, missing.
- **Samples of networks**: `gofN()`, goodness of fit network by network as
  ergm.multi's, with its summary and residual plots; `N()`'s `subset`,
  `offset` (and `offset()` in `lm`) and `label`, also for tergm's operators.
- ergm.multi's `gofN()` reports `degree0` and `isolates` minus the network
  size; ergmx reports them, and warns.
- **Scale**: no array has a row or a cell per dyad any more. The MPLE builds
  the distinct rows of change statistics with their counts, in parallel
  (70 s and 6.8 GB on 10,000 vertices before, 0.5 s and 0.45 GB now); the
  sample spaces of combined and bipartite networks, missing dyads and the
  constraints `fixedas`, `fixallbut`, `observed` and `blockdiag` are
  described by groups of vertices and lists of dyads (500 classrooms of 20
  in 0.27 GB rather than 0.75, and 1,500 in 0.57 GB rather than about 7);
  goodness of fit's distances and shared partners come from the Rust core
  (0.1 s and 0.17 GB rather than 5.6 s and 2.3 GB on 10,000 vertices).
- **Speed**: a shared partner cache, as ergm's, for the shared partner terms
  on networks that aren't sparse (10% faster on faux.mesa.high and
  faux.dixon.high), and tabulated geometric weights.
- **Saving fits**: `fit.save(path)` and `ergmx.load_fit(path)`; fits also
  pickle.
- `benchmarks/scale.R` and `scale.py` time ergm and ergmx on networks of
  1,461 to 10,000 vertices and on 500 classrooms.
- **Documentation**: a Quick start, with two complete analyses: an ERGM and
  a multilevel ERGM.
- **Fixes**: `datasets.load()` reads each bundled network once and returns
  copies (python-igraph leaves a C file stream open at each read, and
  Windows allows 512); the Monte Carlo MLE of curved models no longer fails
  in the linear algebra when the decay runs off along a flat direction of
  the approximation, and moves each decay by at most 1 per iteration.

## 0.1.0 (2026-10-01)

The first version.

- **Terms**: 58 of ergm's terms for directed, undirected and bipartite networks, with ergm's
  definitions and names, among them `edges`, `mutual`, `edgecov`, `kstar`, `istar`,
  `ostar`, `degree`, `idegree`, `odegree`, `isolates`, `gwdegree`,
  `gwidegree`, `gwodegree`, `triangle`, `ttriple`, `ctriple`, `gwesp` and
  `gwdsp` (OTP if directed), `esp`, `dsp`, `nodematch` (with `diff=TRUE`),
  `nodemix`, `nodefactor`, `nodeifactor`, `nodeofactor`, `nodecov`,
  `nodeicov`, `nodeocov` and `absdiff`.
- **More terms**: `asymmetric`, `sender`, `receiver`, `sociality`,
  `concurrent`, `twopath`, `transitive` (transitive triads, as ergm documents
  it; ergm computes `ttriple`), `cycle`, `gwnsp`, `nsp`,
  `absdiffcat`, and the shared partner types of directed networks (OTP,
  ITP, RTP, OSP, ISP) for every shared partner term.
- **Bipartite networks**, with 18 terms (`b1star`, `b1degree`,
  `gwb1degree`, `b1concurrent`, `b1factor`, `b1cov`, `b1nodematch`, `b1dsp`,
  `gwb1dsp` and their `b2` twins) and ergm's goodness of fit statistics.
- **Curved ERGMs**: the decay of the geometrically weighted terms
  estimated (`fixed=FALSE`), in the MPLE, contrastive divergence, the Monte
  Carlo MLE and the log-likelihood.
- **Operators**: `offset()` (with `-inf` to forbid ties) and `F()`.
- **Samples of networks**, as R's ergm.multi: `Networks()` and the `N()`
  operator, with linear models of network-level attributes (`lm=`), curved
  terms, and pooled goodness of fit.
- **Temporal ERGMs**, as R's tergm: `NetSeries()`, the operators `Form()`,
  `Persist()`, `Diss()`, `Cross()` and `Change()`, `tergm()` for the
  conditional MLE (and MPLE), and dynamic simulation (`simulate_dynamic()`,
  `fit.simulate(time_slices=)`) with tergm's per-step stopping rule and
  discordant-dyad proposals.
- **Constraints**: `bd`, `blocks`, `degrees`, `odegrees` and `idegrees`, with
  degree-preserving MCMC moves.
- **Missing ties**: likelihood inference conditional on the observed dyads
  ([Handcock and Gile 2010](https://doi.org/10.1214/08-AOAS221)), for estimates, standard errors,
  log-likelihoods and goodness of fit.
- **Estimation**: the exact MLE of dyad-independent models; the Monte Carlo
  MLE of the others, with [Hummel et al. (2012)](https://doi.org/10.1080/10618600.2012.679224) step lengths, an adaptive MCMC
  interval and standard errors that include the MCMC error; MPLE and
  contrastive divergence estimates and starting values.
- **MCMC** in Rust: tie/no-tie and triadic proposals, parallel chains, a
  density guard.
- **Checking models**: MCMC diagnostics, goodness of fit, log-likelihoods by
  path sampling with an Euler–Maclaurin corrected rule, and model comparison.
- **Multilevel networks**: ergm's `S()` operator (terms on the network
  within a level, or on the bipartite network between two), and MPNet's
  configurations of two-level networks ([Wang et al. 2013](https://doi.org/10.1016/j.socnet.2013.01.004)): `star2ax`,
  `axs1a`, `aas1x`, `aaaxs`, `txax`, `atxax`, `l3xax` and their B twins,
  `l3axb` and `c4axb`.
- **Interpreting and reporting**: tie probabilities (`predict()`,
  conditional and unconditional, as ergm's), average marginal effects (as
  ergMargins, with full delta-method standard errors), odds ratios and
  confidence intervals, tables of models identical to texreg's
  (`table()`: text, LaTeX, HTML and Markdown), and `to_frame()` for pandas.
- **Datasets**: the networks of R's ergm documentation, ergm.multi's
  household networks `Goeyvaerts`, and multinets' multilevel network
  `linked_sim`.
- **Packaging**: Python 3.11 or newer; wheels with the compiled Rust core for
  Linux, macOS and Windows, one per platform for every Python version;
  development with uv, and the Rust version pinned in `rust-toolchain.toml`.
