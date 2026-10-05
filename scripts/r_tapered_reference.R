# Reference fits from R's ergm.tapered (1.2-0, from github.com/statnet/
# ergm.tapered; no longer on CRAN for R 4.5) for tests/test_tapered.py:
# estimates, standard errors and tapering coefficients of tapered ERGMs,
# including models whose ordinary MLE is degenerate. Each model is fitted
# with seeds 1 to 3, whose estimates and standard errors vary: with
# taper.terms = "dependent", ergm.tapered's standard errors come from the
# last Monte Carlo sample, whose means can be far from the network's (35
# edges below it on faux.mesa.high), and vary by a factor of 2 between seeds.
#
# Estimated tapering (fixed = FALSE): the strength multiplying the tapering
# coefficients, from ergm.tapered's kurtosis-penalized profile iterations,
# with each model's estimates; and, to check the profile objective and its
# maximization exactly, an inner fit's sample (its statistics relative to
# the network's) with the objective at several strengths and the strength
# ergm.tapered proposes from it.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_tapered_reference.R
# or, for the estimated tapering only (the other fits kept from the JSON):
#   Rscript scripts/r_tapered_reference.R estimated

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
only_estimated <- identical(commandArgs(trailingOnly = TRUE), "estimated")
out_path <- file.path("tests", "data", "r_tapered_reference.json")
results <- if (only_estimated) jsonlite::read_json(out_path, simplifyVector = TRUE)$fits else list()
for (name in if (only_estimated) character(0) else names(fits)) {
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
estimated_fits <- list(
  flo_triangle = list(network = "flomarriage", formula = "edges + kstar(2) + triangle"),
  samplk_ttriple = list(network = "samplk3", formula = "edges + mutual + ttriple"),
  mesa_gwesp = list(network = "faux.mesa.high", formula = "edges + nodematch('Grade') + gwesp(0.5, fixed = TRUE)"),
  mesa_triangle = list(network = "faux.mesa.high", formula = "edges + triangle")
)
estimated <- list()
for (name in names(estimated_fits)) {
  m <- estimated_fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  by_seed <- list()
  for (seed in 1:3) {
    seconds <- system.time(fit <- suppressWarnings(ergm.tapered(f, fixed = FALSE, eval.loglik = FALSE,
                                                                 control = control.ergm.tapered(seed = seed))))[["elapsed"]]
    keep <- !grepl("Taper_Penalty", names(coef(fit)), fixed = TRUE)
    by_seed[[seed]] <- list(coef = unname(coef(fit)[keep]), se = unname(sqrt(diag(vcov(fit)))[keep]),
                            strength = fit$tapering.strength, r = fit$r, converged = fit$tapering.converged,
                            history = fit$tapering.history, seconds = seconds)
  }
  estimated[[name]] <- c(m, list(names = names(coef(fit))[keep], tapering_coef = as.list(fit$tapering.coefficients),
                                 seeds = by_seed))
  message(sprintf("%-16s strengths %s  %s", name, paste(sapply(by_seed, function(b) round(b$strength, 4)), collapse = " "),
                  paste(round(colMeans(do.call(rbind, lapply(by_seed, `[[`, "coef"))), 4), collapse = " ")))
}

# The profile objective on an inner fit's sample: the first outer iteration
# (strength 1) of two models.
objective_checks <- list()
for (name in c("flo_triangle", "mesa_gwesp")) {
  m <- estimated_fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  control <- control.ergm.tapered(seed = 1, MCMLE.tapering.maxit = 1)
  fit <- suppressWarnings(ergm.tapered(f, fixed = FALSE, eval.loglik = FALSE, control = control))
  for (arg in c("MCMLE.metric", "MCMLE.termination", ergm.tapered:::STATIC_TAPERING_CONTROLS))
    control$loglik[arg] <- list(control[[arg]])
  xsim <- statnet.common::as.logwmatrix(as.matrix(fit$sample))
  index <- grep("Taper_Penalty", param_names(fit, canonical = FALSE), fixed = TRUE)
  theta0 <- coef(fit)
  eta0 <- ergm.eta(theta0, fit$etamap)
  strengths <- c(0.34, 0.5, 0.8, 1, 1.2, 1.7, 2.3, 2.9985, 2.9995)
  values <- sapply(strengths, function(s) {
    theta <- theta0
    theta[index] <- s
    ergm.tapered:::llik.fun.Kpenalty(theta, xsim = xsim, eta0 = eta0, etamap = fit$etamap, control.llik = control$loglik)
  })
  update <- ergm.tapered:::.estimate.tapering.strength(fit, control)
  objective_checks[[name]] <- list(strength0 = unname(theta0[index]), sample = unname(as.matrix(fit$sample)),
                                   strengths = strengths, objective = values, proposed = update$strength,
                                   proposed_objective = update$objective, effective_size = update$effectiveSize)
}

jsonlite::write_json(list(ergm_tapered_version = as.character(packageVersion("ergm.tapered")), fits = results,
                          estimated = estimated, objective_checks = objective_checks),
                     out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
