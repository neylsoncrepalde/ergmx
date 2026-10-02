# R side of the scale benchmark: ergm's MPLE and Monte Carlo MLE (with its
# defaults, including the log-likelihood) of the magnolia model on networks of
# 1,461 to 10,000 vertices, and ergm.multi's of 500 classrooms. Run
# benchmarks/make_scale.py first, then, from the repository root:
#
#     Rscript benchmarks/scale.R
#
# Writes benchmarks/data/r_scale.json.

suppressMessages({
  library(ergm)
  library(ergm.multi)
  library(igraph)
})

as_network <- function(g) {
  net <- network::network.initialize(vcount(g), directed = is_directed(g))
  el <- as_edgelist(g, names = FALSE)
  if (nrow(el)) network::add.edges(net, el[, 1], el[, 2])
  for (a in setdiff(vertex_attr_names(g), "id")) network::set.vertex.attribute(net, a, vertex_attr(g, a))
  net
}
read <- function(name) read_graph(gzcon(file(file.path("benchmarks", "data", paste0(name, ".graphml.gz")), "rb")),
                                   format = "graphml")

magnolia_formula <- "edges + nodematch('Grade') + nodematch('Race') + nodematch('Sex') + gwesp(0.25, fixed=TRUE)"
fit <- function(net, formula, ...) {
  f <- as.formula(paste("net ~", formula))
  environment(f) <- environment()
  mple <- system.time(m <- ergm(f, estimate = "MPLE"))[["elapsed"]]
  mle <- system.time(x <- ergm(f, control = control.ergm(seed = 1)))[["elapsed"]]
  list(n = network::network.size(net), formula = formula, mple_seconds = mple, mle_seconds = mle,
       mple = as.list(coef(m)), coef = as.list(coef(x)), se = as.list(summary(x)$coefficients[, "Std. Error"]))
}

results <- list()
data(faux.magnolia.high)
results$magnolia_1461 <- fit(faux.magnolia.high, magnolia_formula)
cat(sprintf("magnolia_1461: MPLE %.1f s, MLE %.1f s\n", results$magnolia_1461$mple_seconds, results$magnolia_1461$mle_seconds))
for (n in c(2500, 5000, 10000)) {
  name <- paste0("magnolia_", n)
  results[[name]] <- fit(as_network(read(name)), magnolia_formula)
  cat(sprintf("%s: MPLE %.1f s, MLE %.1f s\n", name, results[[name]]$mple_seconds, results[[name]]$mle_seconds))
}
classrooms <- read("classrooms")
rooms <- lapply(sort(unique(vertex_attr(classrooms, "classroom"))), function(k)
  as_network(induced_subgraph(classrooms, which(vertex_attr(classrooms, "classroom") == k))))
results$classrooms <- fit(Networks(rooms), "N(~edges + nodematch('gender') + gwesp(0.5, fixed=TRUE))")
cat(sprintf("classrooms: MPLE %.1f s, MLE %.1f s\n", results$classrooms$mple_seconds, results$classrooms$mle_seconds))

jsonlite::write_json(list(ergm_version = as.character(packageVersion("ergm")),
                          ergm_multi_version = as.character(packageVersion("ergm.multi")), models = results),
                     file.path("benchmarks", "data", "r_scale.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
