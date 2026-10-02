# Reference results of tergm's EGMME (Krivitsky 2009), for the ergmx test
# suite: tests/data/r_egmme_reference.json has, for each model, its estimates
# and standard errors with three seeds (the EGMME is stochastic).
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_egmme_reference.R

suppressMessages(library(tergm))

models <- list(
  edges_duration = list(n = 30, formula = "Form(~edges) + Persist(~edges)", targets = "edges + mean.age",
                        target_stats = c(15, 10)),
  degree_duration = list(n = 30, formula = "Form(~edges + degree(1)) + Diss(~edges)",
                         targets = "edges + degree(1) + mean.age", target_stats = c(15, 18, 20))
)
results <- list()
for (name in names(models)) {
  m <- models[[name]]
  g0 <- network.initialize(m$n, directed = FALSE)
  fits <- lapply(1:3, function(seed) {
    set.seed(seed)
    seconds <- system.time(fit <- tergm(as.formula(paste("g0 ~", m$formula)), targets = as.formula(paste("~", m$targets)),
                                        target.stats = m$target_stats, estimate = "EGMME",
                                        control = control.tergm(seed = seed)))[["elapsed"]]
    s <- summary(fit)$coefficients
    cat(name, seed, sprintf("%.1f s", seconds), sprintf("%s=%.3f (%.3f)", rownames(s), s[, 1], s[, 2]), "\n")
    list(coef = as.list(s[, 1]), se = as.list(s[, 2]), seconds = seconds)
  })
  results[[name]] <- c(m, list(fits = fits))
}
jsonlite::write_json(list(tergm_version = as.character(packageVersion("tergm")), models = results),
                     file.path("tests", "data", "r_egmme_reference.json"), auto_unbox = TRUE, digits = NA,
                     pretty = TRUE)
