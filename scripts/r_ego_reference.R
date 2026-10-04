# Reference results from R's ergm.ego, used by tests/test_ego.py.
#
# The egocentric census of faux.mesa.high (every student an ego, as egor's
# as.egor()) and a sample of 100 of its egos: their estimated population
# statistics and covariances (summary(egor ~ ..., scaleto=)), with each of
# ergm.ego's variance estimators, and ergm.ego() fits, with the observed
# statistics of one's gof(). Writes
# tests/data/r_ego_reference.json.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_ego_reference.R

suppressMessages({
  library(ergm.ego)
})

data(faux.mesa.high)
census <- as.egor(faux.mesa.high)
set.seed(2026)
sampled_egos <- sort(sample(network.size(faux.mesa.high), 100))
sampled <- census[sampled_egos, ]

named <- function(x) as.list(x)

stat_formula <- paste(
  "edges + nodefactor('Sex') + nodefactor('Race', levels=c('Black','White')) + nodematch('Grade') +",
  "nodematch('Sex', diff=TRUE) + nodemix('Sex') + nodecov('Grade') + absdiff('Grade') + absdiffcat('Grade') +",
  "degree(0:3) + degree(1:2, by='Sex') + degree(1, by='Grade', homophily=TRUE) + degrange(2, 4) +",
  "concurrent + concurrentties + degree1.5 + gwdegree(0.5, fixed=TRUE) + esp(0:2) +",
  "gwesp(0.5, fixed=TRUE) + transitiveties + triangle + meandeg")
stats_for <- function(egor, scaleto) {
  out <- list()
  for (est in c("survey", "asymptotic", "naive", "jackknife")) {
    s <- if (est == "survey") summary(as.formula(paste("egor ~", stat_formula)), scaleto = scaleto)
         else NULL
    if (est == "survey") {
      out$names <- names(s)
      out$mean <- as.numeric(s)
      out$survey <- unname(vcov(s))
    }
  }
  # The other estimators, through ergm.ego without fitting.
  for (est in c("asymptotic", "naive", "jackknife")) {
    # meandeg doesn't scale: these estimators need scaling statistics only.
    f <- as.formula(paste("egor ~", sub(" \\+ meandeg$", "", stat_formula)))
    e <- ergm.ego(f, popsize = scaleto, control = control.ergm.ego(stats.est = est, ppopsize = scaleto),
                  do.fit = FALSE)
    out[[est]] <- list(mean = unname(e$m), cov = unname(e$v))
  }
  out
}

fits <- list(
  census_dyadind = list(data = "census", popsize = 205,
                        formula = "edges + nodefactor('Sex') + nodematch('Grade') + nodecov('Grade')"),
  census_percapita = list(data = "census", popsize = 1,
                          formula = "edges + nodefactor('Sex') + nodematch('Grade') + nodecov('Grade')"),
  sample_dyadind = list(data = "sample", popsize = 205,
                        formula = "edges + nodefactor('Sex') + nodematch('Grade') + nodecov('Grade')"),
  sample_degree = list(data = "sample", popsize = 205, gof = TRUE,
                       formula = "edges + nodematch('Grade') + degree(1) + gwesp(0.5, fixed=TRUE)"),
  census_gwesp = list(data = "census", popsize = 1,
                      formula = "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)")
)
fit_results <- lapply(fits, function(m) {
  egor <- if (m$data == "census") census else sampled
  f <- as.formula(paste("egor ~", m$formula))
  environment(f) <- environment()
  seconds <- system.time(fit <- ergm.ego(f, popsize = m$popsize,
                                         control = control.ergm.ego(ergm = control.ergm(seed = 1))))[["elapsed"]]
  cat(sprintf("%-18s %6.1f s\n", m$formula, seconds))
  out <- list(data = m$data, popsize = m$popsize, formula = m$formula, coef = named(coef(fit)),
              se = named(sqrt(diag(vcov(fit)))), ppopsize = fit$ppopsize, stats = named(fit$m),
              seconds = seconds)
  if (isTRUE(m$gof)) {
    # The observed (per capita) statistics of gof(), which the egos estimate.
    out$gof <- lapply(c(model = "model", degree = "degree", espartners = "espartners"), function(G) {
      g <- gof(fit, GOF = G, control = control.gof.ergm(seed = 1, nsim = 10))
      obs <- g[[grep("^obs", names(g))[1]]]
      list(names = names(obs), obs = unname(as.numeric(obs)))
    })
  }
  out
})

jsonlite::write_json(
  list(ergm_ego_version = as.character(packageVersion("ergm.ego")),
       sampled_egos = sampled_egos, stat_formula = stat_formula,
       census = stats_for(census, 205), sample = stats_for(sampled, 205), fits = fit_results),
  file.path("tests", "data", "r_ego_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE
)
