# Reference results from R's Bergm, used by tests/test_bayes.py: posterior
# summaries of three models (means, standard deviations, time-series
# standard errors of the means, quantiles, acceptance rates), with longer
# chains than Bergm's default for tighter comparisons, and auxiliary chains
# of at least one sweep of the dyads (ergmx's default; Bergm's 1000 steps
# leave the auxiliary networks of larger networks too close to the observed
# one, which biases the posterior). Writes tests/data/r_bergm_reference.json.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_bergm_reference.R

suppressMessages({
  library(Bergm)
})
data(florentine)
data(samplk)
data(faux.mesa.high)

models <- list(
  flo_kstar = list(network = "flomarriage", formula = "edges + kstar(2)"),
  samplk_mutual = list(network = "samplk3", formula = "edges + mutual"),
  mesa_gwesp = list(network = "faux.mesa.high", formula = "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)")
)
main_iters <- 2000
results <- lapply(models, function(m) {
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  aux_iters <- max(1000, network.dyadcount(net))
  set.seed(1)
  seconds <- system.time(fit <- bergm(f, main.iters = main_iters, aux.iters = aux_iters))[["elapsed"]]
  theta <- as.matrix(fit$Theta)
  tsvar <- apply(theta, 2, function(x) coda::spectrum0.ar(x)$spec)
  message(sprintf("%-14s %6.1f s  means %s", m$formula, seconds, paste(round(colMeans(theta), 4), collapse = " ")))
  list(network = m$network, formula = m$formula, names = fit$specs, mean = unname(colMeans(theta)),
       sd = unname(apply(theta, 2, sd)), ts_se = unname(sqrt(tsvar / nrow(theta))),
       quantiles = unname(t(apply(theta, 2, quantile, c(0.025, 0.5, 0.975)))),
       acceptance = fit$AR, seconds = seconds, main_iters = main_iters, aux_iters = aux_iters)
})
jsonlite::write_json(list(bergm_version = as.character(packageVersion("Bergm")), models = results),
                     file.path("tests", "data", "r_bergm_reference.json"), auto_unbox = TRUE, digits = NA,
                     pretty = TRUE)
