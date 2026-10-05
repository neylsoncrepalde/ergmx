# Reference results from R's tergm 4.2 and ergm.count 4.1 for
# tests/test_valued_series.py: valued models of a series of networks
# (NetSeries() with response=), Sampson's monks' liking at three times, whose
# values are the ranks of the monks' choices (score: 3 for the first, 1 for
# the third). The transitions are modelled conditionally on the networks
# before them; edgecov(".PrevNet", "score") is the previous network's values,
# a lagged effect. Statistics, and fits with seeds 1 to 3.
#
# R's valued N() doesn't accept a NetSeries() (its LHS must be Networks()):
# the statistics of N() are those of Networks() of the transitions.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_valued_series_reference.R

suppressMessages({
  library(tergm)
  library(ergm.count)
  library(ergm.multi)
})
data(samplk)
monks <- NetSeries(samplk1, samplk2, samplk3)
transitions <- Networks(samplk2, samplk3)

stats_of <- function(net, formula, ...) {
  s <- summary(as.formula(paste("net ~", formula)), ...)
  list(formula = formula, names = names(s), values = unname(as.numeric(s)))
}
stats <- list(
  valued = stats_of(monks, paste("sum + nonzero + mutual(form = 'min') + edgecov('.PrevNet', 'score') +",
                                 "edgecov('.PrevNet') + nodematch('group', form = 'sum') +",
                                 "transitiveweights('min', 'max', 'min') + nodeocovar + atleast(2) + sum(pow = 0.5) + CMP"),
                    response = "score"),
  binary = stats_of(monks, "edges + mutual + edgecov('.PrevNet') + edgecov('.PrevNet', 'score')"),
  valued_N = stats_of(transitions, "N(~sum + nonzero + mutual(form = 'min'), ~.NetworkID)", response = "score")
)

fits <- list(
  valued_lag = list(formula = "sum + nonzero + mutual(form = 'min') + edgecov('.PrevNet', 'score')",
                    response = "score", reference = "Binomial(3)"),
  valued_group = list(formula = "sum + nonzero + nodematch('group', form = 'sum') + edgecov('.PrevNet', 'score')",
                      response = "score", reference = "Poisson"),
  binary_lag = list(formula = "edges + mutual + edgecov('.PrevNet')")
)
results <- list()
for (name in names(fits)) {
  m <- fits[[name]]
  f <- as.formula(paste("monks ~", m$formula))
  environment(f) <- environment()
  coefs <- list(); ses <- list()
  for (seed in 1:3) {
    args <- list(f, control = control.ergm(seed = seed))
    if (!is.null(m$response)) {
      args$response <- m$response
      args$reference <- as.formula(paste("~", m$reference))
    }
    seconds <- system.time(fit <- suppressWarnings(suppressMessages(do.call(ergm, args))))[["elapsed"]]
    coefs[[seed]] <- unname(coef(fit))
    ses[[seed]] <- unname(sqrt(diag(vcov(fit))))
  }
  coefs <- do.call(rbind, coefs); ses <- do.call(rbind, ses)
  results[[name]] <- c(m, list(names = names(coef(fit)), coef = colMeans(coefs), se = colMeans(ses),
                              coef_seeds = coefs, se_seeds = ses, seconds = seconds))
  message(sprintf("%-14s %6.1f s  %s", name, seconds,
                  paste(sprintf("%s=%.4f (%.4f)", names(coef(fit)), colMeans(coefs), colMeans(ses)), collapse = " ")))
}

jsonlite::write_json(list(versions = list(tergm = as.character(packageVersion("tergm")),
                                          ergm.count = as.character(packageVersion("ergm.count"))),
                          stats = stats, fits = results),
                     file.path("tests", "data", "r_valued_series_reference.json"), auto_unbox = TRUE, digits = NA,
                     pretty = TRUE)
