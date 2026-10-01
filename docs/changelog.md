# Changelog

## 0.1.0 (unreleased)

The first version, a proof of concept.

- **Terms**: 22 terms for directed and undirected networks, with ergm's
  definitions and names: `edges`, `mutual`, `edgecov`, `kstar`, `istar`,
  `ostar`, `gwdegree`, `gwidegree`, `gwodegree`, `triangle`, `ttriple`,
  `ctriple`, `gwesp` (OTP if directed), `gwdsp`, `nodematch` (with
  `diff=TRUE`), `nodefactor`, `nodeifactor`, `nodeofactor`, `nodecov`,
  `nodeicov`, `nodeocov` and `absdiff`.
- **Estimation**: the exact MLE of dyad-independent models; the Monte Carlo
  MLE of the others, with Hummel et al. (2012) step lengths, an adaptive MCMC
  interval and standard errors that include the MCMC error; MPLE and
  contrastive divergence estimates and starting values.
- **MCMC** in Rust: tie/no-tie and triadic proposals, parallel chains, a
  density guard.
- **Checking models**: MCMC diagnostics, goodness of fit, log-likelihoods by
  path sampling with an Euler–Maclaurin corrected rule, and model comparison.
- **Datasets**: the networks of R's ergm documentation.
