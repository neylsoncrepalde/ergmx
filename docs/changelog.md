# Changelog

## 0.1.0 (unreleased)

The first version, a proof of concept.

- **Terms**: 58 terms for directed, undirected and bipartite networks, with ergm's
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
  (Handcock and Gile 2010), for estimates, standard errors,
  log-likelihoods and goodness of fit.
- **Estimation**: the exact MLE of dyad-independent models; the Monte Carlo
  MLE of the others, with Hummel et al. (2012) step lengths, an adaptive MCMC
  interval and standard errors that include the MCMC error; MPLE and
  contrastive divergence estimates and starting values.
- **MCMC** in Rust: tie/no-tie and triadic proposals, parallel chains, a
  density guard.
- **Checking models**: MCMC diagnostics, goodness of fit, log-likelihoods by
  path sampling with an Euler–Maclaurin corrected rule, and model comparison.
- **Datasets**: the networks of R's ergm documentation, ergm.multi's
  household networks `Goeyvaerts`, and multinets' multilevel network
  `linked_sim`.
- **Packaging**: Python 3.11 or newer; wheels with the compiled Rust core for
  Linux, macOS and Windows, one per platform for every Python version;
  development with uv, and the Rust version pinned in `rust-toolchain.toml`.
