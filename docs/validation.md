# Validation

`ergmx` is tested against R's ergm 4.12 on ergm's own networks (flomarriage,
samplk3, faux.mesa.high and faux.dixon.high, the last two also with missing
dyads) and multinets' `linked_sim`, and against exact results on networks
small enough to enumerate every possible network. The R scripts in
`scripts/` store R's results in `tests/data/`, and the test suite compares.

## Against R's ergm

| Check | Result |
|---|---|
| Statistics of all 29 terms and both operators, 35 models | identical to R's `summary()` (to 1e-12), names included |
| MPLE, 25 models, with offsets, `F()` and `blocks` | identical to R (to 1e-6) |
| Dyad-independent MLE, standard errors, log-likelihood and BIC, 8 models, with offsets, `blocks` and missing dyads | identical to R (to 1e-6; standard errors to 1e-4, the tolerance of R's `glm`) |
| Monte Carlo MLE, 7 models (4 undirected, 3 directed), 3 to 10 seeds each | within 0.12 standard errors of R's estimates; standard errors 0.89 to 1.12 times R's |
| Monte Carlo MLE with constraints (`bd`, `blocks`, `degrees`, `odegrees`), missing dyads, offsets, `F()`, `esp` and multilevel models, 12 models x 5 seeds | all converged, within 0.17 standard errors of R's estimates; standard errors 0.91 to 1.09 times R's |
| Goodness of fit, directed and undirected | observed distributions and p-values identical to R's; simulated distributions agree within Monte Carlo error |
| Contrastive divergence | a fixed point of its defining equation; the Monte Carlo MLE from a CD start matches R's |

R's own standard errors vary by about 15% between seeds on the small
networks, so the comparison of standard errors is only as tight as R allows.
ergm imputes missing dyads at random before its MPLE, so MPLEs of
dyad-dependent models with missing dyads are not compared.

## Against exact results

| Check | Result |
|---|---|
| MCMC stationary distribution, with 3 mixtures of TNT and triadic proposals | expected statistics match exact enumeration of every network on 3 and 6 vertices (undirected) and 4 vertices (directed) |
| Constrained MCMC: `bd` (directed and undirected), `degrees` (both), `odegrees`, `idegrees`, `blocks`, and sampling conditional on observed dyads | expected statistics match exact enumeration of every network each constraint allows; a test that only reversing cyclic triples can pass checks that directed `degrees` reaches every network |
| `-inf` offsets | the MLE equals the logistic regression without the forbidden dyads |
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
| faux.mesa.high, gwesp(0.5) | 205 | 16.8 s | 10.0 s (1.7x) | 2.7 s (6.2x) | 0.04 SE |
| faux.magnolia.high, gwesp(0.25) | 1,461 | 17.3 s | 14.9 s (1.2x) | 5.8 s (3.0x) | 0.07 SE |
| faux.dixon.high (directed), gwesp(0.1) | 248 | 142 s | 80 s (1.8x) | 18 s (8.0x) | 0.07 SE |

The last column is the largest difference between the two packages'
estimates, in R's standard errors. `ergmx`'s log-likelihood samples about 17
times more than ergm's, which is what makes it accurate; without the
log-likelihood on either side, a single thread was 1.9 to 3.6 times faster
than R on these models.
