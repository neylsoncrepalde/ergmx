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
  ties <- network::as.edgelist(net)
  missing <- network::as.edgelist(is.na(net))
  g <- add_edges(g, as.vector(t(rbind(ties, missing))))
  # Missing dyads are edges marked na, as in R's network package.
  g <- set_edge_attr(g, "na", value = rep(c(FALSE, TRUE), c(nrow(ties), nrow(missing))))
  for (a in setdiff(network::list.vertex.attributes(net), c("na", "vertex.names"))) {
    g <- set_vertex_attr(g, a, value = network::get.vertex.attribute(net, a))
  }
  set_vertex_attr(g, "name", value = network::network.vertex.names(net))
}

data(florentine)
data(samplk)
data(faux.mesa.high)
data(faux.dixon.high)
# Networks with missing dyads: two nonrespondents in samplk3 (their
# nominations are unknown), and 5% of the dyads of faux.mesa.high at random.
samplk3.nonresponse <- samplk3
samplk3.nonresponse[c(1, 7), ] <- NA
set.seed(2026)
faux.mesa.high.missing <- faux.mesa.high
pairs <- t(combn(network.size(faux.mesa.high), 2))
faux.mesa.high.missing[pairs[sample(nrow(pairs), round(0.05 * nrow(pairs))), ]] <- NA

# multinets' multilevel network, bundled with ergmx.
linked <- read_graph(gzcon(file(file.path("python", "ergmx", "data", "linked_sim.graphml.gz"), "rb")),
                     format = "graphml")
linked_sim <- network::network(as.matrix(as_adjacency_matrix(linked)), directed = FALSE)
for (a in c("type", "level")) network::set.vertex.attribute(linked_sim, a, vertex_attr(linked, a))
network::network.vertex.names(linked_sim) <- vertex_attr(linked, "name")

networks <- list(flomarriage = flomarriage, samplk3 = samplk3, linked_sim = linked_sim,
                 faux.mesa.high = faux.mesa.high, faux.dixon.high = faux.dixon.high,
                 samplk3.nonresponse = samplk3.nonresponse,
                 faux.mesa.high.missing = faux.mesa.high.missing)

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
                                     "gwesp(0.1, fixed=TRUE)")),

  # Terms and operators for structured and multilevel models.
  mesa_terms2 = list(network = "faux.mesa.high", checks = "stats",
                     formula = paste("nodemix('Race') + nodemix('Grade', levels2=TRUE) +",
                                     "nodemix('Race', levels=c('White', 'Hisp'), levels2=TRUE) +",
                                     "degree(0:3) + isolates + esp(0:3) + dsp(0:2) +",
                                     "F(~gwesp(0.5, fixed=TRUE), ~nodematch('Grade')) +",
                                     "F(~edges + triangle, ~!nodematch('Race')) + offset(edges)")),
  dixon_terms2 = list(network = "faux.dixon.high", checks = "stats",
                      formula = paste("nodemix('race') + nodemix('sex', levels2=TRUE) +",
                                      "idegree(0:2) + odegree(0:3) + isolates + esp(0:2) +",
                                      "dsp(0:2) + gwdsp(0.5, fixed=TRUE) +",
                                      "F(~mutual + ttriple, ~nodematch('sex'))")),
  mesa_mple2 = list(network = "faux.mesa.high", checks = c("stats", "mple"),
                    formula = paste("edges + nodemix('Sex') + degree(1) + esp(1) +",
                                    "F(~gwesp(0.5, fixed=TRUE), ~nodematch('Grade'))")),
  dixon_mple2 = list(network = "faux.dixon.high", checks = c("stats", "mple"),
                     formula = paste("edges + mutual + idegree(1) + odegree(0:1) + isolates +",
                                     "esp(1) + dsp(1) + gwdsp(0.5, fixed=TRUE) + nodemix('sex')")),
  mesa_nodemix = list(network = "faux.mesa.high", checks = all_checks,
                      formula = "edges + nodemix('Sex')"),
  flo_offset = list(network = "flomarriage", checks = all_checks,
                    formula = "offset(edges) + nodecov('wealth')", offset_coef = -2.6),
  mesa_offset = list(network = "faux.mesa.high", checks = all_checks, offset_coef = -6.2,
                     formula = "offset(edges) + nodematch('Grade') + gwesp(0.5, fixed=TRUE)"),
  mesa_F = list(network = "faux.mesa.high", checks = all_checks,
                formula = paste("edges + nodematch('Grade') + nodematch('Race') +",
                                "F(~gwesp(0.5, fixed=TRUE), ~nodematch('Grade'))")),
  mesa_esp = list(network = "faux.mesa.high", checks = all_checks,
                  formula = "edges + nodematch('Grade') + esp(1:2)"),

  # Sample space constraints.
  samplk_bd = list(network = "samplk3", checks = c("stats", "mle"), constraints = "bd(maxout=4)",
                   formula = "edges + mutual"),
  mesa_blocks_dyadind = list(network = "faux.mesa.high", checks = all_checks,
                             constraints = "blocks('Grade', levels2=-1)",
                             formula = "edges + nodematch('Race')"),
  mesa_blocks = list(network = "faux.mesa.high", checks = all_checks,
                     constraints = "blocks('Grade', levels2=-c(1, 3, 6, 10, 15, 21))",
                     formula = "edges + nodematch('Race') + gwesp(0.5, fixed=TRUE)"),
  mesa_degrees = list(network = "faux.mesa.high", checks = c("stats", "mle"), constraints = "degrees",
                      formula = "nodematch('Grade') + gwesp(0.5, fixed=TRUE)"),
  samplk_odegrees = list(network = "samplk3", checks = c("stats", "mle"), constraints = "odegrees",
                         formula = "mutual + ttriple"),
  samplk_degrees = list(network = "samplk3", checks = c("stats", "mle"), constraints = "degrees",
                        formula = "mutual + ttriple"),

  # Multilevel models: tie densities by level, closure within levels, and the
  # within-level networks given the affiliations (blocks fixes them).
  linked_mix = list(network = "linked_sim", checks = all_checks,
                    formula = "nodemix('type', levels2=TRUE)"),
  linked_multilevel = list(network = "linked_sim", checks = all_checks,
                           formula = paste("nodemix('level', levels2=TRUE) +",
                                           "F(~gwesp(0.5, fixed=TRUE), ~nodematch('level'))")),
  linked_blocks = list(network = "linked_sim", checks = all_checks,
                       constraints = "blocks('level', levels2=2)",
                       formula = paste("nodemix('level', levels2=c(1, 3)) +",
                                       "F(~gwesp(0.5, fixed=TRUE), ~nodematch('level'))")),

  # Missing dyads. ergm imputes missing dyads at random before its MPLE, so the
  # MPLE of dyad-dependent models with missing dyads is random: not compared.
  samplk_missing = list(network = "samplk3.nonresponse", checks = c("stats", "mle"),
                        formula = "edges + mutual"),
  mesa_missing_dyadind = list(network = "faux.mesa.high.missing", checks = all_checks,
                              formula = "edges + nodematch('Grade') + nodefactor('Sex')"),
  mesa_missing = list(network = "faux.mesa.high.missing", checks = c("stats", "mle"),
                      formula = paste("edges + nodematch('Grade') + nodematch('Race') +",
                                      "gwesp(0.5, fixed=TRUE)"))
)

named <- function(x) as.list(x)

results <- list()
for (name in names(models)) {
  m <- models[[name]]
  net <- networks[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  constraints <- as.formula(paste("~", if (is.null(m$constraints)) "." else m$constraints))
  fit_ergm <- function(...) {
    if (is.null(m$constraints)) ergm(f, offset.coef = m$offset_coef, ...)
    else ergm(f, constraints = constraints, offset.coef = m$offset_coef, ...)
  }
  result <- list(network = m$network, formula = m$formula, checks = as.list(m$checks),
                 constraints = m$constraints, offset_coef = m$offset_coef,
                 stats = named(summary(f)))
  if ("mple" %in% m$checks) {
    result$mple <- named(coef(fit_ergm(estimate = "MPLE")))
  }
  if ("mle" %in% m$checks) {
    seconds <- system.time(fit <- fit_ergm(control = control.ergm(seed = 1)))[["elapsed"]]
    s <- summary(fit)$coefficients
    result <- c(result, list(
      dyad_independent = is.dyad.independent(fit),
      mle = named(coef(fit)),
      se = named(s[, "Std. Error"]),
      mcmc_pct = named(if ("MCMC %" %in% colnames(s)) s[, "MCMC %"] else 0 * s[, 1]),
      loglik = as.numeric(logLik(fit)),
      nobs = nobs(fit),
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
