# Reference results from R's Bergm for its model-choice tools, used by
# tests/test_bayes_tools.py: the adjusted pseudo-likelihood (ergmAPL), the
# model evidence by Chib and Jeliazkov's method and by power posteriors
# (evidence), and the calibrated pseudo-posterior (bergmC). They are
# Monte Carlo estimates: each is run with several seeds, to give their
# spread.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_bergm_tools_reference.R

suppressMessages({
  library(Bergm)
})
data(florentine)
data(samplk)

models <- list(
  flo_kstar = list(network = "flomarriage", formula = "edges + kstar(2)"),
  flo_dyadind = list(network = "flomarriage", formula = "edges + nodecov('wealth')"),
  monks = list(network = "samplk3", formula = "edges + mutual + nodematch('group')")
)
seeds <- 1:5
results <- list()
for (name in names(models)) {
  m <- models[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  apl <- lapply(seeds, function(s) ergmAPL(f, seed = s))
  # evidence() passes its arguments on unevaluated: give it their values.
  cj <- sapply(seeds, function(s) suppressMessages(do.call(evidence, list(evidence.method = "CJ", formula = f,
                                                                           seed = s)))$log.evidence)
  pp <- sapply(seeds[1:3], function(s) suppressMessages(do.call(evidence, list(evidence.method = "PP", formula = f,
                                                                                seed = s)))$log.evidence)
  set.seed(1)
  bc <- suppressMessages(bergmC(f, seed = 1))
  draws <- as.matrix(bc$Theta)
  results[[name]] <- list(
    network = m$network, formula = m$formula,
    theta_mle = unname(apl[[1]]$Theta_MLE), theta_pl = unname(apl[[1]]$Theta_PL),
    W = lapply(apl, function(a) unname(a$W)), logC = sapply(apl, `[[`, "logC"),
    ll_true = sapply(apl, `[[`, "ll_true"), ll_adjpseudo = sapply(apl, `[[`, "ll_adjpseudo"),
    evidence_cj = cj, evidence_pp = pp,
    bergmC_mean = unname(colMeans(draws)), bergmC_sd = unname(apply(draws, 2, sd)))
  message(sprintf("%-12s logC %s  CJ %s  PP %s", name, paste(round(results[[name]]$logC, 3), collapse = " "),
                  paste(round(cj, 3), collapse = " "), paste(round(pp, 3), collapse = " ")))
}
jsonlite::write_json(list(bergm_version = as.character(packageVersion("Bergm")), models = results),
                     file.path("tests", "data", "r_bergm_tools_reference.json"), auto_unbox = TRUE, digits = NA,
                     pretty = TRUE)
