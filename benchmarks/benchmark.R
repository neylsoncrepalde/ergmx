# R side of the benchmark: fits each model with ergm's defaults, without the
# log-likelihood (which ergmx doesn't compute yet), for several seeds.
# Writes benchmarks/data/r_benchmark.json. Run from the repository root.

suppressMessages({
  library(ergm)
  library(igraph)
})

source_network <- function(net, path) {
  g <- make_empty_graph(n = network::network.size(net), directed = network::is.directed(net))
  g <- add_edges(g, as.vector(t(network::as.edgelist(net))))
  for (a in setdiff(network::list.vertex.attributes(net), c("na", "vertex.names"))) {
    g <- set_vertex_attr(g, a, value = network::get.vertex.attribute(net, a))
  }
  write_graph(g, path, format = "graphml")
}

data(faux.mesa.high)
data(faux.magnolia.high)
networks <- list(faux.mesa.high = faux.mesa.high, faux.magnolia.high = faux.magnolia.high)
for (name in names(networks)) {
  source_network(networks[[name]], file.path("benchmarks", "data", paste0(name, ".graphml")))
}

models <- list(
  mesa = list(network = "faux.mesa.high",
              formula = paste("edges + nodefactor('Sex') + nodematch('Grade') +",
                              "nodematch('Race') + gwesp(0.5, fixed=TRUE)")),
  magnolia = list(network = "faux.magnolia.high",
                  formula = paste("edges + nodematch('Grade') + nodematch('Race') +",
                                  "nodematch('Sex') + gwesp(0.25, fixed=TRUE)"))
)

seeds <- 1:3
results <- list()
for (name in names(models)) {
  m <- models[[name]]
  net <- networks[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  runs <- lapply(seeds, function(s) {
    t <- system.time(fit <- ergm(f, eval.loglik = FALSE, control = control.ergm(seed = s)))
    list(seconds = t[["elapsed"]], coef = as.list(coef(fit)),
         se = as.list(summary(fit)$coefficients[, "Std. Error"]))
  })
  results[[name]] <- list(network = m$network, formula = m$formula, n = network.size(net),
                          runs = runs)
  cat(sprintf("%-9s n = %4d  median %.2f s\n", name, network.size(net),
              median(sapply(runs, `[[`, "seconds"))))
}

jsonlite::write_json(list(ergm_version = as.character(packageVersion("ergm")), models = results),
                     file.path("benchmarks", "data", "r_benchmark.json"),
                     auto_unbox = TRUE, digits = NA, pretty = TRUE)
