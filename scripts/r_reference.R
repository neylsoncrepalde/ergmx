# Reference results from R's ergm, used by the ergmx test suite.
#
# Exports the networks to tests/data/<network>.graphml, dyadic covariates to
# tests/data/<network>__<name>.csv (graph attributes in the tests), and writes
# tests/data/r_reference.json with, for every model and the checks it is used
# for: the observed statistics ("stats"), the MPLE ("mple"), and the MLE with
# its standard errors, log-likelihood and the time R took ("mle").
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_reference.R

suppressMessages({
  library(ergm)
  library(igraph)
})

out_dir <- file.path("tests", "data")

to_igraph <- function(net) {
  # igraph masks some network functions, so name the package explicitly.
  g <- make_empty_graph(n = network::network.size(net), directed = network::is.directed(net))
  g <- add_edges(g, as.vector(t(network::as.edgelist(net))))
  for (a in setdiff(network::list.vertex.attributes(net), c("na", "vertex.names"))) {
    g <- set_vertex_attr(g, a, value = network::get.vertex.attribute(net, a))
  }
  set_vertex_attr(g, "name", value = network::network.vertex.names(net))
}

data(florentine)
data(samplk)
data(faux.mesa.high)
data(faux.dixon.high)
networks <- list(flomarriage = flomarriage, samplk3 = samplk3,
                 faux.mesa.high = faux.mesa.high, faux.dixon.high = faux.dixon.high)

# Dyadic covariates, stored as network attributes for edgecov("name").
set.seed(2026)
random_matrix <- function(n) {
  x <- matrix(round(runif(n * n), 3), n, n)  # asymmetric on purpose
  diag(x) <- 0
  x
}
covariates <- list(
  flomarriage = list(business = as.matrix(flobusiness)),
  faux.mesa.high = list(asym = random_matrix(network.size(faux.mesa.high))),
  faux.dixon.high = list(asym = random_matrix(network.size(faux.dixon.high)))
)
for (name in names(covariates)) {
  for (cov in names(covariates[[name]])) {
    networks[[name]] %n% cov <- covariates[[name]][[cov]]
    write.table(covariates[[name]][[cov]], file.path(out_dir, sprintf("%s__%s.csv", name, cov)),
                sep = ",", row.names = FALSE, col.names = FALSE)
  }
}
for (name in names(networks)) {
  write_graph(to_igraph(networks[[name]]), file.path(out_dir, paste0(name, ".graphml")),
              format = "graphml")
}

# The same formula strings are parsed by ergmx.
all_checks <- c("stats", "mple", "mle")
models <- list(
  flo_dyadind = list(network = "flomarriage", checks = all_checks,
                     formula = "edges + nodecov('wealth') + absdiff('wealth')"),
  flo_edgecov = list(network = "flomarriage", checks = all_checks,
                     formula = "edges + edgecov('business')"),
  flo_triangle = list(network = "flomarriage", checks = all_checks, formula = "edges + triangle"),
  flo_gwdegree = list(network = "flomarriage", checks = all_checks,
                      formula = "edges + gwdegree(0.25, fixed=TRUE)"),
  samplk_mutual = list(network = "samplk3", checks = all_checks, formula = "edges + mutual"),
  mesa_gwesp = list(network = "faux.mesa.high", checks = all_checks,
                    formula = paste("edges + nodefactor('Sex') + nodematch('Grade') +",
                                    "nodematch('Race') + gwesp(0.5, fixed=TRUE)")),
  mesa_gwdegree = list(network = "faux.mesa.high", checks = all_checks,
                       formula = paste("edges + nodematch('Grade') + nodematch('Race') +",
                                       "gwdegree(0.5, fixed=TRUE) + gwesp(0.5, fixed=TRUE)")),
  mesa_terms = list(network = "faux.mesa.high", checks = "stats",
                    formula = paste("edges + kstar(2:3) + gwdegree(0.5, fixed=TRUE) +",
                                    "gwdsp(0.5, fixed=TRUE) + nodematch('Race', diff=TRUE) +",
                                    "triangle + edgecov('asym')")),
  mesa_mple = list(network = "faux.mesa.high", checks = c("stats", "mple"),
                   formula = paste("edges + kstar(2) + gwdegree(0.5, fixed=TRUE) +",
                                   "gwdsp(0.5, fixed=TRUE) + nodematch('Grade', diff=TRUE) +",
                                   "gwesp(0.5, fixed=TRUE)")),
  dixon_terms = list(network = "faux.dixon.high", checks = "stats",
                     formula = paste("edges + mutual + gwesp(0.5, fixed=TRUE) +",
                                     "gwidegree(0.5, fixed=TRUE) + gwodegree(0.5, fixed=TRUE) +",
                                     "ttriple + ctriple + triangle + istar(2) + ostar(2:3) +",
                                     "nodeifactor('sex') + nodeofactor('race') + nodeicov('grade') +",
                                     "nodeocov('grade') + nodematch('grade', diff=TRUE) + edgecov('asym')")),
  dixon_mple = list(network = "faux.dixon.high", checks = c("stats", "mple"),
                    formula = paste("edges + mutual + ttriple + ctriple + istar(2) + ostar(2) +",
                                    "gwesp(0.5, fixed=TRUE) + gwidegree(0.5, fixed=TRUE) +",
                                    "gwodegree(0.5, fixed=TRUE)")),
  dixon_dyadind = list(network = "faux.dixon.high", checks = all_checks,
                       formula = paste("edges + nodeicov('grade') + nodeocov('grade') +",
                                       "nodeifactor('sex') + nodeofactor('race') +",
                                       "nodematch('grade', diff=TRUE) + edgecov('asym')")),
  samplk_gwesp = list(network = "samplk3", checks = all_checks,
                      formula = "edges + mutual + gwesp(0.5, fixed=TRUE)"),
  dixon_gwesp = list(network = "faux.dixon.high", checks = all_checks,
                     formula = paste("edges + mutual + nodematch('grade') + nodematch('race') +",
                                     "gwesp(0.1, fixed=TRUE)"))
)

named <- function(x) as.list(x)

results <- list()
for (name in names(models)) {
  m <- models[[name]]
  net <- networks[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  result <- list(network = m$network, formula = m$formula, checks = as.list(m$checks),
                 stats = named(summary(f)))
  if ("mple" %in% m$checks) {
    result$mple <- named(coef(ergm(f, estimate = "MPLE")))
  }
  if ("mle" %in% m$checks) {
    seconds <- system.time(fit <- ergm(f, control = control.ergm(seed = 1)))[["elapsed"]]
    s <- summary(fit)$coefficients
    result <- c(result, list(
      dyad_independent = is.dyad.independent(fit),
      mle = named(coef(fit)),
      se = named(s[, "Std. Error"]),
      mcmc_pct = named(if ("MCMC %" %in% colnames(s)) s[, "MCMC %"] else 0 * s[, 1]),
      loglik = as.numeric(logLik(fit)),
      seconds = seconds
    ))
    cat(sprintf("%-14s %6.2f s  %s\n", name, seconds,
                paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
  }
  results[[name]] <- result
}

jsonlite::write_json(
  list(ergm_version = as.character(packageVersion("ergm")),
       r_version = R.version.string, models = results),
  file.path(out_dir, "r_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE
)
