# Changelog

## 0.1.0 (unreleased)

The first version, a proof of concept.

- **Terms**: 29 terms for directed and undirected networks, with ergm's
  definitions and names: `edges`, `mutual`, `edgecov`, `kstar`, `istar`,
  `ostar`, `degree`, `idegree`, `odegree`, `isolates`, `gwdegree`,
  `gwidegree`, `gwodegree`, `triangle`, `ttriple`, `ctriple`, `gwesp` and
  `gwdsp` (OTP if directed), `esp`, `dsp`, `nodematch` (with `diff=TRUE`),
  `nodemix`, `nodefactor`, `nodeifactor`, `nodeofactor`, `nodecov`,
  `nodeicov`, `nodeocov` and `absdiff`.
- **Operators**: `offset()` (with `-inf` to forbid ties) and `F()`.
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
- **Datasets**: the networks of R's ergm documentation, and multinets'
  multilevel network `linked_sim`.
- **Packaging**: Python 3.11 or newer; wheels with the compiled Rust core for
  Linux, macOS and Windows, one per platform for every Python version;
  development with uv, and the Rust version pinned in `rust-toolchain.toml`.
