# Reference results from R's ergm, used by the ergmx test suite.
#
# Exports the networks to tests/data/*.graphml and writes
# tests/data/r_reference.json with, for every model: the observed statistics,
# the MPLE, the MLE with its standard errors, and the time R took.
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
networks <- list(flomarriage = flomarriage, samplk3 = samplk3,
                 faux.mesa.high = faux.mesa.high)
for (name in names(networks)) {
  write_graph(to_igraph(networks[[name]]), file.path(out_dir, paste0(name, ".graphml")),
              format = "graphml")
}

# The same formula strings are parsed by ergmx.
models <- list(
  flo_dyadind = list(network = "flomarriage",
                     formula = "edges + nodecov('wealth') + absdiff('wealth')"),
  flo_triangle = list(network = "flomarriage", formula = "edges + triangle"),
  samplk_mutual = list(network = "samplk3", formula = "edges + mutual"),
  mesa_gwesp = list(network = "faux.mesa.high",
                    formula = paste("edges + nodefactor('Sex') + nodematch('Grade') +",
                                    "nodematch('Race') + gwesp(0.5, fixed=TRUE)"))
)

named <- function(x) as.list(x)

results <- list()
for (name in names(models)) {
  m <- models[[name]]
  net <- networks[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()

  set.seed(1)
  mple <- ergm(f, estimate = "MPLE")
  seconds <- system.time(fit <- ergm(f, control = control.ergm(seed = 1)))[["elapsed"]]
  s <- summary(fit)$coefficients

  results[[name]] <- list(
    network = m$network,
    formula = m$formula,
    dyad_independent = is.dyad.independent(fit),
    stats = named(summary(f)),
    mple = named(coef(mple)),
    mle = named(coef(fit)),
    se = named(s[, "Std. Error"]),
    mcmc_pct = named(if ("MCMC %" %in% colnames(s)) s[, "MCMC %"] else 0 * s[, 1]),
    loglik = as.numeric(logLik(fit)),
    seconds = seconds
  )
  cat(sprintf("%-14s %6.2f s  %s\n", name, seconds,
              paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
}

jsonlite::write_json(
  list(ergm_version = as.character(packageVersion("ergm")),
       r_version = R.version.string, models = results),
  file.path(out_dir, "r_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE
)
