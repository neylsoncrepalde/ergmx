# ergmx

**Exponential-family random graph models (ERGMs) in Python, with a Rust core.**

`ergmx` fits, simulates, summarizes and checks ERGMs with a high-level API in
the spirit of R's [ergm](https://github.com/statnet/ergm) and statnet: R-style
formulas, the same term names and statistics, and `summary()` and `gof()`
that read like R's.

> **Status: proof of concept.** 22 terms for directed and undirected
> networks, MPLE and Monte Carlo MLE, goodness of fit, validated against R's
> ergm. See [what's missing](#not-yet).

```python
import igraph as ig
import ergmx

g = ig.Graph.Read_GraphML("tests/data/faux.mesa.high.graphml")
fit = ergmx.ergm(
    g, "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)",
    seed=1,
)
fit.summary()
```

```
Monte Carlo Maximum Likelihood Results:

                   Estimate  Std. Error  MCMC %  z value  Pr(>|z|)
edges               -6.1870      0.1719       0  -36.000    <1e-04 ***
nodefactor.Sex.M    -0.1261      0.0743       0   -1.696   0.08989 .
nodematch.Grade      1.9762      0.1747       0   11.314    <1e-04 ***
nodematch.Race       0.2650      0.1149       0    2.307   0.02105 *
gwesp.fixed.0.5      1.2146      0.0841       0   14.447    <1e-04 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-likelihood: not computed for dyad-dependent models yet.
Converged after 6 iterations (4 chains, 1024 samples).
```

R's ergm gives `-6.1884, -0.1293, 1.9761, 0.2699, 1.2178` on the same model.
It took 15.8 s; `ergmx` took 1 s.

Then check the fit, as with R's `gof()`:

```python
result = fit.gof()   # degree, edgewise shared partners, geodesic distances, model statistics
print(result["degree"])
result.plot()
```

![](docs/figures/gof-mesa.png)

## Features

- **Networks**: `igraph.Graph` or `networkx.Graph`/`DiGraph`, directed or
  undirected. Vertex attributes are available to the terms.
- **Formulas** in R syntax (`"edges + gwesp(0.5, fixed=TRUE)"`, even with
  `net ~` in front), or terms combined with `+`: `edges() + gwesp(0.5, fixed=True)`.
  Nothing is evaluated: arguments must be literals.
- **Terms**, with ergm's definitions and names:

  | | Undirected | Directed |
  |---|---|---|
  | Dyadic | `edges`, `edgecov` | `edges`, `edgecov`, `mutual` |
  | Degree | `kstar(k)`, `gwdegree` | `istar(k)`, `ostar(k)`, `gwidegree`, `gwodegree` |
  | Triads | `triangle`, `gwesp`, `gwdsp` | `triangle`, `ttriple`, `ctriple`, `gwesp` (OTP) |
  | Attributes | `nodematch` (`diff=TRUE` too), `nodefactor`, `nodecov`, `absdiff` | the same, plus `nodeifactor`, `nodeofactor`, `nodeicov`, `nodeocov` |

  The geometrically weighted terms take a fixed decay (`fixed=TRUE`).
  `edgecov('name')` reads an n x n matrix from a graph attribute.
- **Estimation**:
  - dyad-independent models: the exact MLE (logistic regression), with
    log-likelihood, AIC and BIC;
  - other models: Monte Carlo MLE starting from the MPLE, with Hummel et al.
    (2012) step lengths, the log-normal approximation, an adaptive MCMC interval
    that targets an effective sample size, and standard errors that include the
    MCMC error;
  - `estimate="MPLE"` for the pseudo-likelihood estimate only.
- **MCMC** in Rust: tie/no-tie (TNT) proposals mixed with triadic proposals,
  which close or open triangles, for models with triangle or shared partner
  terms, directed or not (like ergm's `MH_SPDyad` default). Chains run in
  parallel threads.
- **Goodness of fit**: `fit.gof()` or `ergmx.gof(network, formula, coef)`
  compares the degree (in- and out-degree if directed), edgewise shared
  partner and geodesic distance distributions, and the model statistics,
  with simulated networks: tables with Monte Carlo p-values like R's, and
  `plot()`.
- `ergmx.simulate(network, formula, coef, nsim)` returns graphs of the same
  kind as the input, or their statistics; `fit.simulate()` uses the estimates.
- `ergmx.summary_stats(network, formula)`: R's `summary(net ~ formula)`.

## Validation against R

`scripts/r_reference.R` fits the models below with ergm 4.12 and stores the
results in `tests/data/r_reference.json`; the test suite compares.

| Check | Result |
|---|---|
| Statistics of all 22 terms, 13 models | identical to R's `summary()` (1e-12) |
| MPLE, 13 models | identical to R (1e-6) |
| Dyad-independent MLE, standard errors, log-likelihood (3 models, incl. `edgecov` and the directed attribute terms) | identical to R (1e-6; SEs 1e-4, R's `glm` tolerance) |
| Monte Carlo MLE, 7 models (4 undirected, 3 directed) x 3–10 seeds | within 0.12 standard errors of R; SEs within 0.89–1.12 of R's |
| Goodness of fit, directed and undirected | observed distributions and p-values identical to R's; simulated distributions agree within Monte Carlo error |
| MCMC stationary distribution, 3 proposal mixtures | matches exact enumeration of all networks on 3 and 6 vertices (undirected) and 4 vertices (directed) |

The networks are ergm's flomarriage, samplk3, faux.mesa.high and
faux.dixon.high (248 students, directed friendship nominations). R's own
standard errors vary by about 15% between seeds on the small networks, so the
SE comparison is only as tight as R allows.

## Performance

`benchmarks/benchmark.R` and `benchmarks/benchmark.py`: median of 3 seeds on an
Apple M4 Pro, R's ergm with its default settings and without the log-likelihood
(which ergmx doesn't compute yet).

| Model | Vertices | R ergm 4.12 | ergmx, 1 chain | ergmx, 4 chains | Max \|difference\| |
|---|---|---|---|---|---|
| faux.mesa.high, gwesp(0.5) | 205 | 15.8 s | 5.8 s (2.7x) | 1.7 s (9.5x) | 0.04 SE |
| faux.magnolia.high, gwesp(0.25) | 1,461 | 13.7 s | 7.2 s (1.9x) | 2.9 s (4.7x) | 0.07 SE |
| faux.dixon.high (directed), gwesp(0.1) | 248 | 132 s | 37 s (3.6x) | 10 s (13x) | 0.07 SE |

The proposal matters as much as the language: with plain TNT proposals, R
takes 128 s on faux.magnolia.high.

## Architecture

```
python/ergmx/         Python API
  formula.py            R-style formula parsing (Python's ast, no eval)
  terms.py              term definitions: names, parameters, directedness
  _estimation.py        MPLE, Monte Carlo MLE, MCMC diagnostics
  _fit.py               ErgmFit and its summary table
  _simulate.py          ergm(), simulate(), summary_stats()
src/                  Rust core (PyO3), exposed as ergmx._core.Model
  network.rs            sorted neighbour lists + edge list for O(1) random ties
  terms.rs              the Term trait and the change statistics
  sampler.rs            Metropolis-Hastings with TNT + triadic proposals
  lib.rs                bindings; chains run in parallel with rayon
```

Adding a term means implementing its change statistic in `src/terms.rs` and
describing it in `python/ergmx/terms.py`.

## Not yet

- Curved ERGMs (the geometrically weighted terms with an estimated decay).
- More terms: `degree(k)`, `isolates`, `nodemix`, `concurrent`, `gwnsp`,
  bipartite terms, `edgecov` of a network attribute given as a network, and
  directed shared partner types other than OTP.
- Log-likelihood of dyad-dependent models (bridge sampling), for AIC/BIC.
- Better starting values than the MPLE for strongly dependent models
  (contrastive divergence), sample space constraints (`bd`, `blocks`,
  `degrees`) and missing ties.
- MCMC diagnostic plots.
- Documentation, and wheels built for every platform in CI.

## Development

Requires Rust (<https://rustup.rs>) and Python 3.9 or newer.

```bash
python -m venv .venv
.venv/bin/pip install maturin numpy scipy igraph networkx pytest
.venv/bin/maturin develop --release     # build the Rust core and install ergmx
.venv/bin/python -m pytest              # Python tests
cargo test --release                    # Rust tests
```

To regenerate the R reference results or rerun the benchmark (needs R with
ergm, igraph and jsonlite):

```bash
Rscript scripts/r_reference.R && Rscript scripts/r_gof_reference.R
Rscript benchmarks/benchmark.R && .venv/bin/python benchmarks/benchmark.py
```

## License

GPL-3.0, like R's ergm.
