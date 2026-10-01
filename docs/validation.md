# Validation

`ergmx` is tested against R's ergm 4.12 on ergm's own networks (flomarriage,
samplk3, faux.mesa.high and faux.dixon.high, the second and third also with
missing dyads), multinets' `linked_sim`, Davis's Southern Women and a
simulated bipartite network, and against exact results on networks small
enough to enumerate every possible network. The R scripts in
`scripts/` store R's results in `tests/data/`, and the test suite compares.

## Against R's ergm

| Check | Result |
|---|---|
| Statistics of all 58 terms and both operators, 56 models | identical to R's `summary()` (to 1e-12), names included, except `transitive` and the edgewise RTP statistics (below) |
| MPLE, 38 models, with offsets, `F()`, `blocks`, bipartite networks and curved terms | identical to R (to 1e-6; curved models to 1e-3, where R's optimizer stops on a flat optimum: ergmx's pseudo-likelihood is at least R's) |
| Dyad-independent MLE, standard errors, log-likelihood and BIC, 10 models, with offsets, `blocks`, missing dyads and bipartite networks | identical to R (to 1e-6; standard errors to 1e-3, the tolerance of R's `glm`) |
| Monte Carlo MLE, 7 models (4 undirected, 3 directed), 3 to 10 seeds each | within 0.12 standard errors of R's estimates; standard errors 0.89 to 1.12 times R's |
| Monte Carlo MLE with constraints (`bd`, `blocks`, `degrees`, `odegrees`), missing dyads, offsets, `F()`, `esp` and multilevel models, 12 models x 5 seeds | all converged, within 0.17 standard errors of R's estimates; standard errors 0.91 to 1.09 times R's |
| Monte Carlo MLE of `concurrent`, `twopath`, OSP shared partners, bipartite models and curved models (gwesp, directed and undirected, and gwb1degree), 10 models x 3 to 5 seeds | all converged, within 0.2 standard errors of R's estimates; standard errors 0.90 to 1.12 times R's, except the curved bipartite model, whose likelihood is nearly flat in the decay (R's standard error of the decay is twice its estimate) |
| Goodness of fit, directed and undirected | observed distributions and p-values identical to R's; simulated distributions agree within Monte Carlo error |
| Contrastive divergence | a fixed point of its defining equation; the Monte Carlo MLE from a CD start matches R's |

R's own standard errors vary by about 15% between seeds on the small
networks, so the comparison of standard errors is only as tight as R allows.
ergm imputes missing dyads at random before its MPLE, so MPLEs of
dyad-dependent models with missing dyads are not compared. R fits two of the
curved models only from a contrastive divergence start; ergmx fits them from
its default start.

### Two discrepancies in ergm 4.12.0

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

## Against exact results

| Check | Result |
|---|---|
| MCMC stationary distribution, with 3 mixtures of TNT and triadic proposals | expected statistics match exact enumeration of every network on 3 and 6 vertices (undirected) and 4 vertices (directed) |
| Constrained MCMC: `bd` (directed and undirected), `degrees` (both), `odegrees`, `idegrees`, `blocks`, and sampling conditional on observed dyads | expected statistics match exact enumeration of every network each constraint allows; a test that only reversing cyclic triples can pass checks that directed `degrees` reaches every network |
| `-inf` offsets | the MLE equals the logistic regression without the forbidden dyads |
| Bipartite MCMC | expected statistics match exact enumeration of the 512 networks of a 3 x 3 bipartite network |
| Curved terms | eta . counts equals theta times the fixed-decay statistic exactly, and the Jacobian matches finite differences |
| Log-likelihood, directed and undirected | unbiased against exact enumeration (6 and 4 vertices), with standard errors that match the spread across seeds |

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
| faux.mesa.high, gwesp(0.5) | 205 | 16.8 s | 10.6 s (1.6x) | 2.5 s (6.6x) | 0.04 SE |
| faux.magnolia.high, gwesp(0.25) | 1,461 | 17.3 s | 15.3 s (1.1x) | 5.5 s (3.1x) | 0.07 SE |
| faux.dixon.high (directed), gwesp(0.1) | 248 | 142 s | 83 s (1.7x) | 17 s (8.1x) | 0.07 SE |

The last column is the largest difference between the two packages'
estimates, in R's standard errors. `ergmx`'s log-likelihood samples about 17
times more than ergm's, which is what makes it accurate; without the
log-likelihood on either side, a single thread was 1.9 to 3.6 times faster
than R on these models.
