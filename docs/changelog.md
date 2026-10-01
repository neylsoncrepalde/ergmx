# Changelog

## Unreleased

- **Datasets**: `labs_sim`, a multilevel network of 120 researchers and 30
  laboratories simulated from a known model, with effects within each level
  and across levels.
- **Documentation**: a Quick start, with two complete analyses: an ERGM and
  a multilevel ERGM.

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
