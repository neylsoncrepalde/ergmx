# Reference fits from R's ergm.tapered (1.2-0, from github.com/statnet/
# ergm.tapered; no longer on CRAN for R 4.5) for tests/test_tapered.py:
# estimates, standard errors and tapering coefficients of tapered ERGMs,
# including models whose ordinary MLE is degenerate. Each model is fitted
# with seeds 1 to 3, whose estimates and standard errors vary: with
# taper.terms = "dependent", ergm.tapered's standard errors come from the
# last Monte Carlo sample, whose means can be far from the network's (35
# edges below it on faux.mesa.high), and vary by a factor of 2 between seeds.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_tapered_reference.R

suppressMessages({
  library(ergm.tapered)
})
data(florentine)
data(samplk)
data(faux.mesa.high)

fits <- list(
  flo_triangle = list(network = "flomarriage", formula = "edges + kstar(2) + triangle", r = 2),
  flo_r1 = list(network = "flomarriage", formula = "edges + kstar(2) + triangle", r = 1),
  mesa_gwesp = list(network = "faux.mesa.high", formula = "edges + nodematch('Grade') + gwesp(0.5, fixed = TRUE)", r = 2),
  mesa_dependent = list(network = "faux.mesa.high", formula = "edges + nodematch('Grade') + gwesp(0.5, fixed = TRUE)",
                        r = 2, taper_terms = "dependent"),
  samplk_ttriple = list(network = "samplk3", formula = "edges + mutual + ttriple", r = 1.5),
  flo_tau = list(network = "flomarriage", formula = "edges + kstar(2) + triangle", tau = c(0.05, 0.01, 0.2))
)
results <- list()
for (name in names(fits)) {
  m <- fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  coefs <- list(); ses <- list()
  for (seed in 1:3) {
    args <- list(f, control = control.ergm.tapered(seed = seed))
    if (!is.null(m$r)) args$r <- m$r
    if (!is.null(m$tau)) args$tau <- m$tau
    if (!is.null(m$taper_terms)) args$taper.terms <- m$taper_terms
    seconds <- system.time(fit <- do.call(ergm.tapered, args))[["elapsed"]]
    keep <- !grepl("Taper_Penalty", names(coef(fit)), fixed = TRUE)
    coefs[[seed]] <- unname(coef(fit)[keep])
    ses[[seed]] <- unname(sqrt(diag(vcov(fit)))[keep])
  }
  coefs <- do.call(rbind, coefs); ses <- do.call(rbind, ses)
  results[[name]] <- c(m, list(names = names(coef(fit))[keep], coef = colMeans(coefs), se = colMeans(ses),
                              coef_seeds = coefs, se_seeds = ses,
                              tapering_coef = as.list(fit$tapering.coefficients),
                              centers = as.list(fit$tapering.centers), seconds = seconds))
  message(sprintf("%-16s %6.1f s  %s", name, seconds,
                  paste(sprintf("%s=%.4f (%.4f)", names(coef(fit))[keep], colMeans(coefs), colMeans(ses)),
                        collapse = " ")))
}
jsonlite::write_json(list(ergm_tapered_version = as.character(packageVersion("ergm.tapered")), fits = results),
                     file.path("tests", "data", "r_tapered_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
