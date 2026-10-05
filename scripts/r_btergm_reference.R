# Reference results from R's btergm, used by tests/test_btergm.py.
#
# The friendship networks of Knecht's pupils at four times, prepared as in
# btergm's documentation (Leifeld, Cranmer and Desmarais 2018): pupils who
# left are removed at those times (so the networks' vertices differ), other
# missing nominations are filled with the modal value, and each network has
# the pupils' sex and the square roots of their in- and out-degrees. The
# networks (with their vertex names) and covariates are written to the JSON,
# with the pseudo-likelihood estimates of several models (exact) and their
# bootstrap results (random). (btergm keeps a single memory() term: a second
# one replaces the first's covariate.)
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_btergm_reference.R

suppressMessages({
  library(btergm)
  library(sna)
})

data("knecht")
for (i in seq_along(friendship)) {
  rownames(friendship[[i]]) <- colnames(friendship[[i]]) <- seq_len(nrow(friendship[[i]]))
}
rownames(primary) <- colnames(primary) <- rownames(friendship[[1]])
sex <- demographics$sex
names(sex) <- seq_along(sex)
friendship <- handleMissings(friendship, na = 10, method = "remove")
friendship <- handleMissings(friendship, na = NA, method = "fillmode")
for (i in seq_along(friendship)) {
  s <- adjust(sex, friendship[[i]])
  friendship[[i]] <- network(friendship[[i]])
  friendship[[i]] <- set.vertex.attribute(friendship[[i]], "sex", s)
  friendship[[i]] <- set.vertex.attribute(friendship[[i]], "idegsqrt",
                                          sqrt(degree(friendship[[i]], cmode = "indegree")))
  friendship[[i]] <- set.vertex.attribute(friendship[[i]], "odegsqrt",
                                          sqrt(degree(friendship[[i]], cmode = "outdegree")))
}

primaries <- rep(list(primary), length(friendship))  # timecov() takes one matrix per time step

models <- list(
  jss = "edges + mutual + ttriple + transitiveties + ctriple + nodeicov('idegsqrt') + nodeicov('odegsqrt') +
         nodeocov('odegsqrt') + nodeofactor('sex') + nodeifactor('sex') + nodematch('sex') + edgecov(primary) +
         delrecip + memory(type = 'stability')",
  lag2 = "edges + mutual + memory(type = 'autoregression', lag = 2)",
  innovation = "edges + mutual + nodematch('sex') + memory(type = 'innovation')",
  loss = "edges + mutual + delrecip(mutuality = TRUE) + memory(type = 'loss')",
  timecov = "edges + mutual + nodematch('sex') + timecov() + timecov(primaries, minimum = 2, transform = function(t) t^2)",
  offset = "edges + mutual + nodematch('sex') + edgecov(primary) + memory(type = 'stability')"
)
reps <- 1000
results <- list()
for (name in names(models)) {
  f <- as.formula(paste("friendship ~", models[[name]]))
  environment(f) <- environment()
  set.seed(1)
  seconds <- system.time(fit <- btergm(f, R = reps, offset = name == "offset", verbose = FALSE))[["elapsed"]]
  # Replicates with a coefficient that can't be estimated (a resample of one
  # time step, repeated) are left out, as btergm's confint() does.
  complete <- fit@boot$t[complete.cases(fit@boot$t), , drop = FALSE]
  ci <- t(apply(complete, 2, quantile, c(0.025, 0.975)))
  results[[name]] <- list(formula = gsub("\\s+", " ", models[[name]]), offset = name == "offset",
                          names = names(coef(fit)), coef = unname(coef(fit)), boot_mean = unname(colMeans(complete)),
                          boot_sd = unname(apply(complete, 2, sd)), lower = unname(ci[, 1]), upper = unname(ci[, 2]),
                          complete = nrow(complete),
                          time_steps = fit@time.steps, nobs = unname(nobs(fit)[2]), R = reps, seconds = seconds)
  message(sprintf("%-10s %5.1f s  %s", name, seconds, paste(sprintf("%.4f", coef(fit)), collapse = " ")))
}

export <- lapply(friendship, function(nw) list(
  names = network.vertex.names(nw), edges = as.edgelist(nw)[, 1:2] - 1,
  sex = get.vertex.attribute(nw, "sex"), idegsqrt = get.vertex.attribute(nw, "idegsqrt"),
  odegsqrt = get.vertex.attribute(nw, "odegsqrt")))
# boot's percentile interpolation (boot.ci's "perc"), on fixed replicates.
replicates <- sin(1:997) * 3 + (1:997) / 500
percentile <- list(t = replicates, alpha = c(0.025, 0.975, 0.1, 0.5),
                   value = unname(boot:::norm.inter(replicates, c(0.025, 0.975, 0.1, 0.5))[, 2]))

jsonlite::write_json(list(btergm_version = as.character(packageVersion("btergm")), networks = export,
                          primary = unname(primary), primary_names = rownames(primary), models = results,
                          percentile = percentile),
                     file.path("tests", "data", "r_btergm_reference.json"), auto_unbox = TRUE, digits = NA,
                     pretty = TRUE)
