# ergmx

**Exponential-family random graph models (ERGMs) in Python, with a Rust core.**

`ergmx` fits, simulates, summarizes and checks ERGMs with a high-level API in
the spirit of R's [ergm](https://github.com/statnet/ergm) and statnet: R-style
formulas, the same term names and statistics, and `summary()` and `gof()`
that read like R's.

> 167 terms and 9 operators for directed, undirected and bipartite
> networks, and interactions; curved ERGMs; sample space constraints; missing
> ties; multilevel networks (as MPNet); samples of networks (as ergm.multi);
> temporal ERGMs, EGMME and dynamic simulation (as tergm); MPLE, contrastive
> divergence and Monte Carlo MLE; MCMC diagnostics, log-likelihoods, model
> comparison and goodness of fit; tie probabilities, marginal effects and
> tables of results; networks of tens of thousands of vertices; all
> validated against R. See [what's missing](#not-yet).

```python
import ergmx
from ergmx import datasets

g = datasets.load("faux.mesa.high")   # an igraph.Graph; ergm's networks are bundled
fit = ergmx.ergm(
    g, "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)",
    seed=1,
)
fit.summary()
```

```
Monte Carlo Maximum Likelihood Results:

                   Estimate  Std. Error  MCMC %  z value  Pr(>|z|)
edges               -6.1846      0.1734       0  -35.669    <1e-04 ***
nodefactor.Sex.M    -0.1256      0.0747       0   -1.681   0.09274 .
nodematch.Grade      1.9710      0.1758       0   11.210    <1e-04 ***
nodematch.Race       0.2657      0.1188       0    2.235   0.02539 *
gwesp.fixed.0.5      1.2166      0.0853       0   14.271    <1e-04 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-likelihood: -867.1531 (MC SE 0.194)   AIC: 1744.3062   BIC: 1784.0461
Converged after 6 iterations (4 chains, 1024 samples).
```

R's ergm gives `-6.1884, -0.1293, 1.9761, 0.2699, 1.2178` and a log-likelihood
of -867.56 on the same model, in 16.8 s; `ergmx` took 2.4 s. (A high-precision
estimate of the log-likelihood is -867.09: R's is off by 0.47, see
[validation](#validation-against-r).)

Then check the fit, as with R's `gof()`:

```python
result = fit.gof()   # degree, edgewise shared partners, geodesic distances, model statistics
print(result["degree"])
result.plot()
```

![](docs/figures/gof-mesa.png)

Check the MCMC and compare models, as with R's `mcmc.diagnostics()` and
`anova()`:

```python
fit.mcmc_diagnostics()        # effective sizes, R-hat across chains, Geweke; .plot() for traces
simpler = ergmx.ergm(g, "edges + nodefactor('Sex') + nodematch('Grade') + nodematch('Race')")
ergmx.compare(simpler, fit)   # log-likelihoods, AIC, BIC, likelihood-ratio test
```

## Features

- **Networks**: `igraph.Graph` or `networkx.Graph`/`DiGraph`, directed or
  undirected. Vertex attributes are available to the terms; edges with
  `na=True` mark missing dyads.
- **Formulas** in R syntax (`"edges + gwesp(0.5, fixed=TRUE)"`, even with
  `net ~` in front), or terms combined with `+`: `edges() + gwesp(0.5, fixed=True)`.
  Nothing is evaluated: arguments must be literals.
- **Terms**, with ergm's definitions and names:

  | | Undirected | Directed |
  |---|---|---|
  | Dyadic | `edges`, `edgecov`, `sociality` | `edges`, `edgecov`, `mutual`, `asymmetric`, `sender`, `receiver` |
  | Degree | `kstar(k)`, `degree(d)`, `isolates`, `concurrent`, `twopath`, `gwdegree` | `istar(k)`, `ostar(k)`, `idegree(d)`, `odegree(d)`, `isolates`, `twopath`, `gwidegree`, `gwodegree` |
  | Triads and cycles | `triangle`, `cycle(k)`, `gwesp`, `gwdsp`, `gwnsp`, `esp(d)`, `dsp(d)`, `nsp(d)` | `triangle`, `ttriple`, `ctriple`, `transitive`, `cycle(k)`, and the shared partner terms with any `type` (OTP, ITP, RTP, OSP, ISP) |
  | Attributes | `nodematch` (`diff=TRUE` too), `nodemix`, `nodefactor`, `nodecov`, `absdiff`, `absdiffcat` | the same, plus `nodeifactor`, `nodeofactor`, `nodeicov`, `nodeocov` |
  | Operators | `offset(term)` (with `-inf` to forbid ties), `F(~terms, ~filter)`, `S(~terms, ~attrs)` | the same |

  And the rest of ergm's common vocabulary: degree ranges (`degrange`,
  `idegrange`, `odegrange`), `degree1.5`, `concurrentties`, `isolatededges`,
  `density`, `meandeg`, the triad census, `balance`, `intransitive`, the
  Simmelian terms, `transitiveties`, `cyclicalties`, `threetrail`,
  `localtriangle`, covariates (`dyadcov`, `hamming`, `attrcov`, `mm`, `diff`,
  `smalldiff`, the covariate ranges and distinct neighbour types), `altkstar`
  and more, with ergm's options (`levels=`, `by=`, `homophily=`, `attr=`,
  `nodes=`...), and interactions of dyad-independent terms (`a:b`, `a*b`).
  Bipartite networks have `b1star`, `b1degree`, `b1degrange`,
  `b1mindegree`, `gwb1degree`, `b1concurrent`, `b1factor`, `b1cov`,
  `b1covrange`, `b1nodematch`, `b1starmix`, `b1twostar`, `b1sociality`,
  `b1dsp`, `gwb1dsp` and the rest, and their `b2` twins. The geometrically
  weighted terms take a fixed decay (`fixed=TRUE`) or, as in ergm by default,
  estimate it: curved ERGMs. `edgecov('name')` reads an n x n matrix from a
  graph attribute. See the [term reference](docs/terms.md).
- **Constraints**, as in ergm: `bd` (bounded degrees, for fixed-choice
  designs, also by the alters' attributes), `blocks` (fix the dyads of some
  mixing types), `blockdiag`, `Dyads(fix=, vary=)`, `fixedas`, `fixallbut`,
  `observed`, and `degrees`, `odegrees`, `idegrees`, `b1degrees`,
  `b2degrees` and `edges`, with moves that preserve them.
- **Missing ties**: likelihood inference conditional on the observed dyads
  ([Handcock and Gile 2010](https://doi.org/10.1214/08-AOAS221)), assuming they are missing at random, for the
  estimates, standard errors, log-likelihood and goodness of fit.
- **Bipartite networks** (`bipartite=True`): only the ties between modes are
  modeled, with their own terms and ergm's goodness of fit statistics.
- **Curved ERGMs**: the decays of gwesp, gwdegree and the other
  geometrically weighted terms estimated in the MPLE, contrastive divergence,
  the Monte Carlo MLE and the log-likelihood, with an overflow statistic
  where ergm's cutoff stops with an error.
- **Multilevel networks** as one network with a level attribute, as MPNet
  models them ([Wang et al. 2013](https://doi.org/10.1016/j.socnet.2013.01.004)): ergm's `S()` for terms within a level or
  between two (`S(~edges + gwesp(0.5, fixed=TRUE), ~level == 'A')`), MPNet's
  cross-level configurations (`star2ax`, `txax`, `atxax`, `l3axb`,
  `c4axb`...) and those of directed networks from MPNet's manual, with
  estimated decays (`fixed=FALSE`), `nodemix`, `F()` and `blocks` to model
  kinds of ties, and goodness of fit by level, `gof(by="level")`.
- **Samples of networks**, as R's ergm.multi: `ergmx.Networks(g1, g2, ...)`
  models many networks (classrooms, households) jointly, and
  `N(~terms, lm=~log(n) + weekday)` lets the coefficients depend on
  network-level attributes, with R's syntax and names (and N()'s `subset`,
  `offset` and `label`); `ergmx.gofN(fit)` checks the fit network by network.
- **Temporal ERGMs**, as R's tergm: `ergmx.tergm([wave1, wave2, wave3],
  "Form(~edges + mutual) + Persist(~edges)")` fits the conditional MLE of a
  series of networks, with `Form()`, `Persist()`, `Diss()`, `Cross()` and
  `Change()`, missing dyads (tergm's `NA.impute`) and trends over time;
  `fit.simulate(time_slices=20)` and `ergmx.simulate_dynamic()` run the
  process forward, with the ties that form and dissolve and their durations;
  and `estimate="EGMME"` fits a process to a single network and the ages of
  its ties.
- **Estimation**:
  - dyad-independent models: the exact MLE (logistic regression), with
    log-likelihood, AIC and BIC;
  - other models: Monte Carlo MLE starting from the MPLE (or from the
    contrastive divergence estimate, `init="CD"`), with [Hummel et al. (2012)](https://doi.org/10.1080/10618600.2012.679224)
    step lengths, the log-normal approximation, an adaptive MCMC interval that
    targets an effective sample size, and standard errors that include the
    MCMC error;
  - `estimate="MPLE"` or `estimate="CD"` for those estimates only;
  - degenerate models stop with a `DegeneracyError` that says why: a density
    guard (as in ergm) and a check that the estimate is still moving, with the
    simulated and observed statistics side by side.
- **Log-likelihood** of dyad-dependent models, with its Monte Carlo standard
  error, by path sampling from the dyad-independent submodel as ergm does,
  but integrated with the Euler-Maclaurin corrected trapezoidal rule: its
  error falls as 1/bridges^4 instead of 1/bridges^2 for ergm's midpoint rule.
  `ergmx.compare(fit1, fit2, ...)` tabulates log-likelihoods, AIC, BIC and
  likelihood-ratio tests of nested models.
- **MCMC diagnostics**: `fit.mcmc_diagnostics()` reports mean deviations from
  the observed statistics, naive and time-series standard errors, effective
  sizes, split R-hat across the parallel chains and Geweke z-scores, with
  trace and density plots.
- **MCMC** in Rust: tie/no-tie (TNT) proposals mixed with triadic proposals,
  which close or open triangles, for models with triangle or shared partner
  terms, directed or not (like ergm's `MH_SPDyad` default). Chains run in
  parallel threads.
- **Goodness of fit**: `fit.gof()` or `ergmx.gof(network, formula, coef)`
  compares the degree (in- and out-degree if directed), edgewise shared
  partner and geodesic distance distributions, and the model statistics,
  with simulated networks: tables with Monte Carlo p-values like R's, and
  `plot()`; by level of a vertex attribute with `by=`, and network by network
  for samples of networks with `ergmx.gofN(fit)`.
- `ergmx.simulate(network, formula, coef, nsim)` returns graphs of the same
  kind as the input, or their statistics; `fit.simulate()` uses the estimates.
- **Interpreting and reporting**: `fit.predict()` (tie probabilities, as
  ergm's `predict()`), `fit.marginal_effects()` (as ergMargins),
  `fit.odds_ratios()`, `fit.confint()`, and `ergmx.table(fit1, fit2)`, a
  table of models identical to texreg's `screenreg()`, also as LaTeX, HTML
  and Markdown; `to_frame()` for pandas.
- `ergmx.summary_stats(network, formula)`: R's `summary(net ~ formula)`.
- **Scale**: no array has a cell per dyad, so a network of 10,000 vertices,
  or 1,500 classrooms combined, fits in less than 0.6 GB; the MPLE is built
  from the distinct rows of change statistics, in parallel; shared partner
  counts are cached, as in ergm, where that helps.
- **Saving fits**: `fit.save(path)` and `ergmx.load_fit(path)` (R's
  `saveRDS()` and `readRDS()`); fits also pickle.
- `ergmx.datasets`: the networks of R's ergm documentation (flomarriage,
  flobusiness, samplk1-3, faux.mesa.high, faux.dixon.high,
  faux.magnolia.high), ergm.multi's 318 household networks `Goeyvaerts`,
  multinets' multilevel `linked_sim`, and `labs_sim`, a multilevel network
  of researchers and laboratories simulated from a known model.

## Documentation

A user guide, a term reference, the API reference and a guide for R users,
built with Sphinx in `docs/`; every example runs when it is built:

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -d docs/_build/doctrees docs docs/_build/html
```

`.github/workflows/docs.yml` publishes it to GitHub Pages.

## Validation against R

`scripts/r_reference.R` fits the models below with ergm 4.12 (ergm.multi 0.3.0
and tergm 4.2.2 for samples and series of networks) and stores the
results in `tests/data/r_reference.json`; the test suite compares.

| Check | Result |
|---|---|
| Statistics of ergm's terms and operators, 97 models | identical to R's `summary()` (1e-12), names included, except where ergm 4.12.0 computes something other than its documentation and ergmx follows the documentation: `transitive`, `intransitive`, `dyadcov`'s `utri` and `ltri`, and the edgewise RTP statistics, which ergmx gives as ergm does with its shared-partner cache off (see [validation](docs/validation.md)) |
| MPLE, 60 models | identical to R (1e-6; curved models 1e-3, with a pseudo-likelihood at least R's) |
| Dyad-independent MLE, standard errors, log-likelihood and BIC (25 models, with offsets, `blocks`, `S()`, missing dyads, bipartite networks, samples and series of networks, `N()`'s `subset` and `offset`, `NA.impute`) | identical to R (1e-6; SEs 1e-3, R's `glm` tolerance) |
| Monte Carlo MLE, 7 models (4 undirected, 3 directed) x 3–10 seeds | within 0.12 standard errors of R; SEs within 0.89–1.12 of R's |
| Monte Carlo MLE with constraints, missing dyads, offsets, `F()` and multilevel models, 12 models x 5 seeds | within 0.17 standard errors of R; SEs within 0.91–1.09 of R's |
| Monte Carlo MLE of the new terms, bipartite and curved models (directed and undirected), 10 models x 3–5 seeds | within 0.2 standard errors of R; SEs within 0.90–1.12 of R's, except one curved model with a nearly flat decay |
| Monte Carlo MLE of samples of networks (ergm.multi, also with N()'s `subset`) and of series (tergm's CMLE, also with missing dyads), 6 models x 5 seeds | within 0.10 standard errors of R; SEs within 0.93–1.05 of R's |
| Monte Carlo MLE of a multilevel model with `S()`, 5 seeds | within 0.08 standard errors of R; SEs within 0.96–1.04 of R's |
| MPNet's 16 multilevel configurations (no R implementation exists) | equal to their matrix definitions (1e-12); the MCMC matches exact enumeration |
| Goodness of fit, directed and undirected | observed distributions and p-values identical to R's; simulated distributions agree within Monte Carlo error |
| ergm.multi's `gofN()` and tergm's EGMME | observed statistics identical to R's, fitted values and residuals within R's Monte Carlo error; EGMME estimates within 0.4 standard errors of R's |
| Tie probabilities, marginal effects, confidence intervals and tables | identical to R's `predict()` (1e-12), ergMargins' effects, `confint()` and texreg's tables (character for character) |
| MCMC stationary distribution, unconstrained and under every constraint | matches exact enumeration of every network each constraint allows on 3 to 6 vertices, directed and undirected |
| Log-likelihood, directed and undirected | unbiased against exact enumeration (6 and 4 vertices), with calibrated standard errors |
| Log-likelihood, 5 dyad-dependent models | within one standard error of high-precision estimates (128 bridges); R's 16-point midpoint rule is off by 0.1 to 2.1 |
| Contrastive divergence | a fixed point of its defining equation; the MLE from a CD start matches R's |

The networks are ergm's flomarriage, samplk3, faux.mesa.high and
faux.dixon.high (248 students, directed friendship nominations), the second
and third also with missing dyads, and multinets' linked_sim. R's own
standard errors vary by about 15% between seeds on the small networks, so the
SE comparison is only as tight as R allows. Full details:
[docs/validation.md](docs/validation.md).

## Performance

`benchmarks/benchmark.R` and `benchmarks/benchmark.py`: median of 3 seeds on an
Apple M4 Pro, both with their defaults, which include the log-likelihood. R's
ergm runs on one thread; the single-threaded ergmx run is limited to one thread
too.

| Model | Vertices | R ergm 4.12 | ergmx, 1 thread | ergmx, all threads | Max \|difference\| |
|---|---|---|---|---|---|
| faux.mesa.high, gwesp(0.5) | 205 | 16.8 s | 10.1 s (1.7x) | 2.4 s (7.1x) | 0.04 SE |
| faux.magnolia.high, gwesp(0.25) | 1,461 | 17.3 s | 14.0 s (1.2x) | 3.3 s (5.2x) | 0.07 SE |
| faux.dixon.high (directed), gwesp(0.1) | 248 | 142 s | 80 s (1.8x) | 17 s (8.6x) | 0.07 SE |

ergmx's log-likelihood samples about 17 times more than ergm's, which is what
makes it accurate. Without the log-likelihood on either side (`eval_loglik=False`
and `eval.loglik = FALSE`), a single thread was 1.9 to 3.6 times faster than R
on these models.

The proposal matters as much as the language: with plain TNT proposals, R
takes 128 s on faux.magnolia.high.

On networks of 2,500 to 10,000 vertices and on 500 classrooms combined
(`benchmarks/scale.py`), ergmx's MPLE is 17 to 110 times faster than R's, and
its Monte Carlo MLE 2 to 8 times faster on all threads, in less than 0.6 GB;
see [validation](docs/validation.md#larger-networks).

## Architecture

```
python/ergmx/         Python API
  formula.py            R-style formula parsing (Python's ast, no eval)
  terms.py              term definitions: names, parameters, directedness
  _estimation.py        MPLE, contrastive divergence, Monte Carlo MLE, degeneracy checks
  _loglik.py            log-likelihood by path sampling
  _diagnostics.py       MCMC diagnostics
  _gof.py               goodness of fit
  _compare.py           model comparison
  constraints.py        sample space constraints
  datasets.py           ergm's and multinets' networks, bundled in data/
  _fit.py               ErgmFit and its summary table
  _simulate.py          ergm(), simulate(), summary_stats()
src/                  Rust core (PyO3), exposed as ergmx._core.Model
  network.rs            sorted neighbour lists + edge list for O(1) random ties
  terms.rs              the Term trait and the change statistics
  sampler.rs            Metropolis-Hastings: TNT, triadic and degree-preserving moves
  space.rs              the sample space: free dyads, degree bounds
  lib.rs                bindings; chains run in parallel with rayon
```

Adding a term means implementing its change statistic in `src/terms.rs` and
describing it in `python/ergmx/terms.py`.

## Not yet

- A few of ergm's terms (`degcor`, `tripercent`, the projection operators)
  and constraints (`degreedist`, `egocentric`).
- Valued networks (ergm.count) and egocentric data (ergm.ego).
- tergm's durational model terms; ergm.multi's `lm.gofN()`.

## Installation

ergmx needs Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv add "ergmx[igraph,plot]"        # in a uv project; or: uv pip install / pip install
```

Releases come as wheels with the Rust core already compiled, for Linux,
macOS and Windows, so no Rust compiler is needed. To install the development
version from GitHub, which builds it from source:

```bash
uv add "ergmx[igraph,plot] @ git+https://github.com/neylsoncrepalde/ergmx"
```

Building from source needs a C linker (Xcode Command Line Tools,
`build-essential` or the Visual Studio C++ Build Tools) and Rust 1.88 or
newer. If Rust isn't installed, maturin downloads a private copy (about 500
MB, cached). The [installation guide](docs/user-guide/installation.md) has
the details.

## Development

With [uv](https://docs.astral.sh/uv/) and [rustup](https://rustup.rs):

```bash
git clone https://github.com/neylsoncrepalde/ergmx.git && cd ergmx
rustup toolchain install    # the Rust pinned in rust-toolchain.toml
uv sync                     # Python 3.14, locked dependencies, ergmx built in release mode
uv run pytest               # Python tests
cargo clippy --release --all-targets -- -D warnings && cargo test --release
```

`uv run` rebuilds the Rust core after changes to it. `uv build` makes a wheel
and a source distribution in `dist/`; `.github/workflows/release.yml` builds
the wheels of every platform and publishes them to PyPI when a GitHub release
is published.

To regenerate the R reference results or rerun the benchmark (needs R with
ergm, igraph and jsonlite):

```bash
Rscript scripts/r_reference.R && Rscript scripts/r_gof_reference.R
Rscript benchmarks/benchmark.R && uv run python benchmarks/benchmark.py
```

## License

GPL-3.0, like R's ergm.
