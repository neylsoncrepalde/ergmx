# Validation

`ergmx` is tested against R's ergm 4.12, ergm.multi 0.3.0, tergm 4.2.2, ergm.count
4.1.3, ergm.ego 1.1.4 and Bergm 5.0.7 on
ergm's own networks (flomarriage, samplk1 to samplk3, faux.mesa.high and
faux.dixon.high, the latter two also with missing dyads), multinets'
`linked_sim`, Davis's Southern Women, a simulated bipartite network,
Sampson's monks as a sample and as a series of networks, and ergm.multi's
225 weekday household networks (Goeyvaerts), and against exact results on
networks small enough to enumerate every possible network. The R scripts in
`scripts/` store R's results in `tests/data/`, and the test suite compares.

## Against R's ergm

| Check | Result |
|---|---|
| Statistics of ergm's terms and operators, with their options (`levels=`, `by=`, `homophily=`, `attr=`, `nodes=`...), and interactions, 109 models | identical to R's `summary()` (to 1e-12; `tripercent` to 1e-7, as ergm computes its ratios in single precision), names included, except `transitive`, `intransitive`, `dyadcov`'s `utri` and `ltri`, and the edgewise RTP statistics (below) |
| MPLE, 68 models, with offsets, interactions, `F()`, `S()`, `N()` (with `subset`, `offset`, `label` and `contrasts`), tergm's operators (also with missing dyads imputed by `NA.impute`), `blocks`, `Dyads`, `fixedas`, `fixallbut`, `blockdiag`, bipartite networks and curved terms | identical to R (to 1e-6; curved models to 1e-3, where R's optimizer stops on a flat optimum: ergmx's pseudo-likelihood is at least R's) |
| Dyad-independent MLE, standard errors, log-likelihood and BIC, 31 models, with offsets, interactions, `N()`'s `subset`, `offset` and `contrasts`, target statistics, `egocentric`, `NA.impute`, `blocks`, `Dyads`, `fixedas`, `fixallbut`, `blockdiag`, `S()`, missing dyads, bipartite networks, samples and series of networks | identical to R (to 1e-6; standard errors to 1e-3, the tolerance of R's `glm`) |
| Monte Carlo MLE, 42 dyad-dependent models x 5 seeds (`scripts/mcmle_seeds.py`): directed and undirected, with constraints (`bd`, also by alter class, `blocks`, `degrees`, `odegrees`, `degreedist`, `edges`, `b1degrees`), missing dyads, offsets, `F()`, `S()` and multilevel models, `esp`, OSP shared partners, `concurrent`, `twopath`, `degree(by=)`, curved models (gwesp, directed and undirected, gwb1degree), bipartite models, samples of networks (Sampson's monks; 225 households with `N()` linear models, `subset` and `label`), series (tergm's CMLE, also with missing dyads imputed) and a fit to target statistics | all 210 fits converged, within 0.23 standard errors of R's estimates; standard errors 0.87 to 1.16 times R's |
| Valued networks (ergm.count): 28 statistics on the karate club's counts and 18 on the monks' summed liking (directed) | identical to R's (to 1e-10), but undirected `transitiveties` (below) |
| Valued Monte Carlo MLE, 4 models (Poisson and binomial references, `nonzero`, `nodefactor`, `nodematch`, `mutual`, `transitiveweights`) | within 0.3 standard errors of R's estimates; standard errors within 25% of R's |
| Egocentric data (ergm.ego): 37 estimated statistics on faux.mesa.high's census and a sample of 100 egos, with the survey, asymptotic, naive and jackknife variances | identical to R's (to 1e-9) |
| `ergm.ego()`, 5 models, census and sample, per capita and for a population of 205 | within 0.1 standard errors of R's estimates (0.25 for the Monte Carlo ones); standard errors within 20% of R's (R estimates the information by MCMC, ergmx exactly where it can) |
| Bayesian ERGMs (Bergm), 3 models (undirected, directed, gwesp on 205 vertices), 4 to 6 chains of 2000 draws | posterior means within 4 Monte Carlo standard errors of R's, standard deviations within 20% |
| `san()` and `ergm(target_stats=)`, dyad-independent and with gwesp | the fit to target statistics is R's exactly (dyad-independent) or within its Monte Carlo error |
| Goodness of fit, directed and undirected | observed distributions and p-values identical to R's; simulated distributions agree within Monte Carlo error |
| ergm.multi's multilayer networks: `L()` with Layer Logic, `CMBL`, `twostarL`, `mutualL` and the layer-aware shared partner terms, 20 sets of statistics on the Florentine marriages and business ties and the monks' three waves as layers; 4 fits | statistics and names identical to R's (but the OSP and ISP shared partners in order, which match their documented definition, counted by brute force: see [](coming-from-r.md)); estimates within 0.1 standard errors of R's, standard errors within 10% |
| ergm.multi's `gofN()`, 225 households, the model's statistics and five others, 2,000 simulations | observed statistics identical to R's (but `degree0` and `isolates`, below); fitted values, variances and Pearson residuals agree within R's Monte Carlo error |
| tergm's EGMME, formation and persistence of edges with a mean duration, and with `degree(1)` too | within 0.4 standard errors of the mean of R's estimates over 3 seeds; standard errors within R's range, which spans a factor of 1.7 between its seeds; the edges model's estimate is also the exact one (below) |
| Conditional tie probabilities (`predict()`), 5 models: undirected, directed, missing dyads, curved, 20 household networks | identical to R's `predict()` (to 1e-12), on the same dyads (R's formula method leaves out missing dyads) |
| Average marginal effects, 4 models | identical to ergMargins' (to its 5 significant digits); the standard errors match a numerical delta method (to 1e-5), and ergMargins' when the probabilities are held fixed, as it holds them |
| Confidence intervals, 4 models | identical to R's `confint()` (to 1e-3, the accuracy of R's `glm` standard errors) |
| Tables of results | identical to texreg's `screenreg()`, `texreg()` and `htmlreg()`, character for character, on exactly fitted models |
| Contrastive divergence | a fixed point of its defining equation; the Monte Carlo MLE from a CD start matches R's |

R's own standard errors vary by about 15% between seeds on the small
networks, so the comparison of standard errors is only as tight as R allows.
ergm imputes missing dyads at random before its MPLE, so MPLEs of
dyad-dependent models with missing dyads are not compared. R fits two of the
curved models only from a contrastive divergence start; ergmx fits them from
its default start. ergm.multi and tergm can't fit curved terms inside `N()`
or `Form()` (their MPLE's gradient is not finite, and contrastive divergence
stops with an error), so only their statistics are compared; ergmx's fits of
them are checked below.

### Discrepancies in ergm 4.12.0

- **Edgewise RTP statistics.** ergm 4.12.0's `esp`, `gwesp` and `nsp` with
  `type = "RTP"` depend on the order of the vertices: relabeling
  faux.dixon.high changes `summary(net ~ esp(0:3, type = "RTP"))` from 889,
  269, 37, 2 to 886, 273, 36, 2. Its shared-partner cache, on by default,
  reads them with the wrong key; ergm's development version fixes it
  ([statnet/ergm#656](https://github.com/statnet/ergm/pull/656)). With the
  cache off (`term.options = list(cache.sp = FALSE)`), ergm 4.12.0 gives 847,
  309, 39, 2, as the definition computed in R does, and ergmx's `esp`,
  `gwesp`, `nsp` and `gwnsp` of type RTP are identical to ergm's.
- **`transitive`.** ergm documents it as the number of transitive triads
  (types 030T, 120D, 120U and 300: 371 on faux.dixon.high, 13 on samplk3,
  counted by R's igraph) but computes the number of transitive triples, the
  same as `ttriple` (1,254 and 49). ergmx counts the triads, which match the
  triad census on every directed network of 4 vertices, and warns that ergm
  differs; `ttriple` reproduces ergm's statistic.
- **`intransitive`.** The same: documented as the number of intransitive
  triads (types 111D, 201, 111U, 021C and 030C: 5,991 on faux.dixon.high,
  70 on samplk3, by igraph's triad census) but computed as the number of
  intransitive triples, twopath minus ttriple (6,752 and 92). ergmx counts
  the triads, which match R's `triadcensus` and igraph's, and warns.
- **`dyadcov`.** Documented as the covariate summed over mutual,
  upper-triangular asymmetric and lower-triangular asymmetric dyads; on a
  network whose only tie goes from vertex 1 to vertex 2, in the upper
  triangle, ergm's `utri` is 0 and its `ltri` 1. ergmx follows the
  documentation, so its `utri` and `ltri` are ergm's `ltri` and `utri`, and
  warns.
- **`gofN()`'s `degree0` and `isolates`.** ergm.multi's `gofN()` leaves out
  the statistics of the empty network, which for `degree0` and `isolates` are
  the network size: in the households, it reports the observed and fitted
  isolates as their counts minus the size (-4 for a household of four
  without isolates). The Pearson residuals are unaffected. ergmx reports the
  counts, and warns.
- **`smalldiff`** is documented as counting differences less than the
  cutoff, but its code counts those at most the cutoff (193 ties of
  faux.mesa.high with grades at most 2 apart, against 178 strictly less);
  ergmx follows the code, as the argument's description, "maximum", does.

- **`degcrossprod`** is documented as the mean over ties of the product of
  their degrees, but computed as half of it (5.25 on flomarriage, whose mean
  is 10.5). ergmx follows the documentation and warns.
- **`coincidence(active=)`** is documented as keeping the pairs with at least
  `active` partners in common, but keeps those with more (21 pairs of
  Davis's events with `active=3`, where 42 have at least 3). ergmx follows
  the documentation and warns.
- **`degreedist`, `odegreedist`, `idegreedist`** are documented as keeping
  the degree distributions, but ergm's proposals for directed networks keep
  every vertex's out-degree (moving heads) or in-degree (moving tails).
  ergmx keeps only the distributions.
- **Valued `transitiveties`** of an undirected network counts every tie above
  the threshold (all 78 of zach's) rather than those with a two-path above
  it (67); ergmx counts the latter, as the binary term and the directed
  valued term do, and warns.

## Against exact results

| Check | Result |
|---|---|
| MCMC stationary distribution, with 3 mixtures of TNT and triadic proposals | expected statistics match exact enumeration of every network on 3 and 6 vertices (undirected) and 4 vertices (directed) |
| Constrained MCMC: `bd` (directed and undirected, also by alter class), `degrees` (both), `odegrees`, `idegrees`, `b1degrees`, `b2degrees`, `edges`, `blocks`, and sampling conditional on observed dyads | expected statistics match exact enumeration of every network each constraint allows; a test that only reversing cyclic triples can pass checks that directed `degrees` reaches every network |
| `-inf` offsets | the MLE equals the logistic regression without the forbidden dyads |
| Bipartite MCMC | expected statistics match exact enumeration of the 512 networks of a 3 x 3 bipartite network |
| Curved terms | eta . counts equals theta times the fixed-decay statistic exactly, and the Jacobian matches finite differences; inside `N()`, with a linear model, too |
| Several networks: `Networks()` and `NetSeries()` | the MCMC matches exact enumeration of every network with ties within the networks (two undirected networks; two transitions of directed networks, with discordant-dyad and triadic proposals), and never ties networks together |
| Curved terms inside `N()` and `Form()` | at the curved MPLE's decay, the fixed-decay MPLE has the same coefficients (to 1e-8); a series simulated with decay 0.7 gives back 0.64 |
| Dynamic simulation | one time step is a draw from the transition's model (exact enumeration); with tergm's stopping rule, ties form and dissolve at the dyad-independent model's exact rates (within 5% and 8%) |
| Dyad-independent CMLE | the log-odds that non-ties became ties and that ties persisted, exactly |
| Forward simulation with time trends (`lm=~.Time`) | each step's formation and persistence rates are those predicted for its time, within Monte Carlo error |
| `NA.impute` | each option fills the dyads it can, in turn, as tergm's (`next` from the next wave, also through a wave missing them too) |
| `N()`'s `offset` | the offset statistics' coefficients are 1 and the Jacobian matches finite differences, for curved terms too; the MCMC with `subset` and `offset` matches exact enumeration |
| `gofN()` | each network's fitted value and variance of its edges are the dyad-independent model's exact ones (within Monte Carlo error); with missing dyads, the observed value and its variance are the exact conditional ones |
| EGMME | the edges model, whose equilibrium is known (formation and persistence probabilities that give the density and mean duration), within 0.5 standard errors; its standard errors are the delta method's from the exact stationary covariances; tie ages follow every tie of a simulation |
| The sampler's tracked statistics for every new term (degree ranges, by attribute and with homophily, triad census, trails, Simmelian and transitive ties, covariate ranges, distinct neighbour types, the bipartite terms...) | equal the statistics recomputed from scratch on the sampled networks (to 1e-9), so the change statistics of removing ties are right too |
| Constraints `edges`, `b1degrees`, `b2degrees`, `bd(attribs=)`, `fixedas`, `fixallbut`, `Dyads`, `blockdiag`, `observed` | every simulated network keeps what the constraint fixes, and moves elsewhere |
| MPNet's directed configurations (35, from its manual) and EXTA, EXTB and ASAXASB | equal to their definitions computed from adjacency matrices (to 1e-12) on random directed two-level networks; the MCMC matches exact enumeration of every directed network on 2 + 2 vertices with affiliations from A to B; `S()` between two sets of a directed network matches R's |
| Estimated decays of the multilevel configurations (18 terms) | eta . counts equals theta times the fixed-decay statistic exactly; at the curved MPLE's decay, the fixed-decay MPLE has the same coefficients; networks simulated with a decay of 0.7 give it back within 1.1 standard errors |
| Goodness of fit by level | the observed distributions within each level and of the affiliations equal igraph's |
| MPNet's 16 multilevel configurations, which no R package has | equal to their definitions computed from the levels' adjacency matrices (to 1e-12), on random two-level networks and `linked_sim`, for three decays; the MCMC matches exact enumeration of every network on 6 vertices of two levels (and one of neither), with S() too |
| Log-likelihood, directed and undirected | unbiased against exact enumeration (6 and 4 vertices), with standard errors that match the spread across seeds |
| Valued MCMC, Poisson, binomial, geometric and discrete uniform references, with `transitiveweights` and `CMP` | expected statistics match exact enumeration of every valued network on 3 vertices (truncated where the reference's tail is negligible); the statistics the sampler tracks equal those recomputed (to 1e-9); the Poisson and binomial `sum` models give their exact MLEs and log-likelihood |
| Bayesian ERGMs: the exchange algorithm on dyad-independent models, also with missing dyads imputed by the chains | the posterior means and standard deviations of the exact posterior (the likelihood times the prior on a grid), within Monte Carlo error |
| Simulated annealing (`san()`) | reaches reachable targets exactly, within the constraints and infinite offsets |
| Constraints `degreedist`, `odegreedist`, `idegreedist` and `egocentric` | every simulated network keeps the distributions (each mode's, if bipartite) or the egos' dyads, and the degrees themselves move |
| The Monte Carlo MLE's trust region | curved fits inside `N()` that stalled on some numbers of chains converge on all of them (2, 3 and 4 chains x 5 seeds, the decay within 0.01 of each other) |

## The log-likelihood

For 5 dyad-dependent models, `ergmx`'s log-likelihood is within one standard
error of a high-precision estimate (128 points along the path, 4 times the
samples). R's ergm, which integrates with a 16-point midpoint rule, is off by
0.1 to 2.1 units:

| Model | R ergm | High-precision | ergmx (3 seeds) |
|---|---|---|---|
| samplk3, edges + mutual | -133.99 | -133.88 | -133.86, -133.90, -133.87 |
| samplk3, + gwesp(0.5) | -131.36 | -131.04 | -130.99, -131.04, -131.15 |
| faux.mesa.high, gwesp(0.5) | -867.56 | -867.09 | -867.13, -867.15, -867.09 |
| faux.mesa.high, gwdegree + gwesp | -866.01 | -865.27 | -865.36, -865.20, -865.74 |
| faux.dixon.high (directed), gwesp(0.1) | -4206.10 | -4208.20 | -4208.56, -4208.57, -4207.76 |

## Performance

`benchmarks/benchmark.R` and `benchmarks/benchmark.py` fit three models with
both packages' defaults, which include the log-likelihood; median of 3 seeds
on an Apple M4 Pro. R's ergm runs on one thread, and the single-threaded
ergmx run is limited to one thread too.

| Model | Vertices | R ergm 4.12 | ergmx, 1 thread | ergmx, all threads | Largest difference |
|---|---|---|---|---|---|
| faux.mesa.high, gwesp(0.5) | 205 | 16.8 s | 10.1 s (1.7x) | 2.4 s (7.1x) | 0.04 SE |
| faux.magnolia.high, gwesp(0.25) | 1,461 | 17.3 s | 14.0 s (1.2x) | 3.3 s (5.2x) | 0.07 SE |
| faux.dixon.high (directed), gwesp(0.1) | 248 | 142 s | 80 s (1.8x) | 17 s (8.6x) | 0.07 SE |
| 225 household networks (ergm.multi), with `N()` linear models | 2 to 7 each | 23.3 s | 13.2 s (1.8x) | 2.9 s (8.0x) | 0.06 SE |

The last column is the largest difference between the two packages'
estimates, in R's standard errors (the households: one R run, from
`scripts/r_reference.R`). On small models, such as the series of Sampson's
monks, both take about a second. `ergmx`'s log-likelihood samples about 17
times more than ergm's, which is what makes it accurate; without the
log-likelihood on either side, a single thread was 1.9 to 3.6 times faster
than R on these models.

### Larger networks

`benchmarks/make_scale.py`, `scale.R` and `scale.py` fit the magnolia model
to faux.magnolia.high and to networks like it of 2,500 to 10,000 vertices
(simulated from R's fit, with the same mean degree), and a model with
homophily and gwesp to 500 classrooms of 20 students, combined with
`Networks()` (`N()` in ergm.multi); one seed each, the same machine:

| Network | Vertices | MPLE: R | ergmx | MLE: R | ergmx, 1 thread | ergmx, all threads | Largest difference |
|---|---|---|---|---|---|---|---|
| faux.magnolia.high | 1,461 | 0.2 s | 0.01 s | 17.2 s | 13.9 s (1.2x) | 3.2 s (5.5x) | 0.10 SE |
| magnolia-like | 2,500 | 0.4 s | 0.02 s | 11.4 s | 6.6 s (1.7x) | 1.4 s (8.4x) | 0.06 SE |
| magnolia-like | 5,000 | 1.4 s | 0.08 s | 12.0 s | 26.8 s (0.4x) | 5.8 s (2.1x) | 0.11 SE |
| magnolia-like | 10,000 | 5.6 s | 0.3 s | 36.9 s | 27.5 s (1.3x) | 5.6 s (6.6x) | 0.07 SE |
| 500 classrooms | 10,000 | 2.6 s | 0.02 s | 91.3 s | 59.7 s (1.5x) | 15.3 s (6.0x) | 0.02 SE |

The MPLE is 17 to 110 times faster: its data are the distinct rows of change
statistics, built in parallel threads, and the logistic regression runs on
those. The Monte Carlo MLEs take a varying number of iterations: on 5,000
vertices, ergmx's took 8 where the others took 4 or 5, and half of each
fit's time is the log-likelihood. Memory stays below 0.6 GB for all of them.
