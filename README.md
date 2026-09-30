# ergmx

**Exponential-family random graph models (ERGMs) in Python, with a Rust core.**

`ergmx` fits, simulates and summarizes ERGMs with a high-level API in the spirit
of R's [ergm](https://github.com/statnet/ergm) and statnet: R-style formulas,
the same term names and statistics, and a `summary()` that reads like R's.

> **Status: proof of concept.** Eight terms, MPLE and Monte Carlo MLE,
> validated against R's ergm. See [what's missing](#not-yet).

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

## Features

- **Networks**: `igraph.Graph` or `networkx.Graph`/`DiGraph`, directed or
  undirected. Vertex attributes are available to the terms.
- **Formulas** in R syntax (`"edges + gwesp(0.5, fixed=TRUE)"`, even with
  `net ~` in front), or terms combined with `+`: `edges() + gwesp(0.5, fixed=True)`.
  Nothing is evaluated: arguments must be literals.
- **Terms**: `edges`, `mutual`, `triangle`, `gwesp` (fixed decay), `nodematch`,
  `nodefactor`, `nodecov`, `absdiff`, with ergm's definitions and names.
- **Estimation**:
  - dyad-independent models: the exact MLE (logistic regression), with
    log-likelihood, AIC and BIC;
  - other models: Monte Carlo MLE starting from the MPLE, with Hummel et al.
    (2012) step lengths, the log-normal approximation, an adaptive MCMC interval
    that targets an effective sample size, and standard errors that include the
    MCMC error;
  - `estimate="MPLE"` for the pseudo-likelihood estimate only.
- **MCMC** in Rust: tie/no-tie (TNT) proposals mixed with triadic proposals,
  which close or open triangles, for models with triangle terms (like ergm's
  `MH_SPDyad` default). Chains run in parallel threads.
- `ergmx.simulate(network, formula, coef, nsim)` returns graphs of the same
  kind as the input, or their statistics; `fit.simulate()` uses the estimates.
- `ergmx.summary_stats(network, formula)`: R's `summary(net ~ formula)`.

## Validation against R

`scripts/r_reference.R` fits the models below with ergm 4.12 and stores the
results in `tests/data/r_reference.json`; the test suite compares.

| Check | Result |
|---|---|
| Statistics of all terms (incl. gwesp) | identical to R's `summary()` (1e-13) |
| MPLE | identical to R (1e-8) |
| Dyad-independent MLE, standard errors, log-likelihood | identical to R (1e-6) |
| Monte Carlo MLE, 3 models x 10 seeds | within 0.08 standard errors of R; SEs within 0.89–1.12 of R's |
| MCMC stationary distribution, 3 proposal mixtures | matches exact enumeration of all networks on 3 and 6 vertices |

Models: `flomarriage ~ edges + nodecov('wealth') + absdiff('wealth')`,
`flomarriage ~ edges + triangle`, `samplk3 ~ edges + mutual`, and the
faux.mesa.high model above. R's own standard errors vary by about 15% between
seeds on the small networks, so the SE comparison is only as tight as R allows.

## Performance

`benchmarks/benchmark.R` and `benchmarks/benchmark.py`: median of 3 seeds on an
Apple M4 Pro, R's ergm with its default settings and without the log-likelihood
(which ergmx doesn't compute yet).

| Model | Vertices | R ergm 4.12 | ergmx, 1 chain | ergmx, 4 chains | Max \|difference\| |
|---|---|---|---|---|---|
| faux.mesa.high, gwesp(0.5) | 205 | 15.8 s | 5.4 s (2.9x) | 1.0 s (16x) | 0.05 SE |
| faux.magnolia.high, gwesp(0.25) | 1,461 | 13.6 s | 7.0 s (2.0x) | 2.8 s (4.9x) | 0.07 SE |

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

- Curved ERGMs (`gwesp`, `gwdsp`, `gwdegree` with an estimated decay).
- More terms: `kstar`, `gwdegree`, `gwdsp`, `istar`/`ostar`, `nodematch(diff=TRUE)`,
  `edgecov`, bipartite terms, directed triangles (`ttriple`, `ctriple`, ...).
- Log-likelihood of dyad-dependent models (bridge sampling), for AIC/BIC.
- Sample space constraints (`bd`, `blocks`, `degrees`), missing ties.
- Goodness of fit (`gof`) and MCMC diagnostic plots.
- Triadic proposals for directed networks.
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
Rscript scripts/r_reference.R
Rscript benchmarks/benchmark.R && .venv/bin/python benchmarks/benchmark.py
```

## License

GPL-3.0, like R's ergm.
