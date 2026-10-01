# Exports the networks bundled with ergmx (ergmx.datasets) from R's ergm
# package, as gzipped GraphML in python/ergmx/data/.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/export_datasets.R

suppressMessages({
  library(ergm)
  library(igraph)
})

to_igraph <- function(net) {
  # igraph masks some network functions, so name the package explicitly.
  g <- make_empty_graph(n = network::network.size(net), directed = network::is.directed(net))
  g <- add_edges(g, as.vector(t(network::as.edgelist(net))))
  for (a in setdiff(network::list.vertex.attributes(net), c("na", "vertex.names"))) {
    g <- set_vertex_attr(g, a, value = network::get.vertex.attribute(net, a))
  }
  set_vertex_attr(g, "name", value = as.character(network::network.vertex.names(net)))
}

data(florentine)
data(samplk)
data(faux.mesa.high)
data(faux.dixon.high)
data(faux.magnolia.high)
networks <- list(flomarriage = flomarriage, flobusiness = flobusiness, samplk1 = samplk1,
                 samplk2 = samplk2, samplk3 = samplk3, faux.mesa.high = faux.mesa.high,
                 faux.dixon.high = faux.dixon.high, faux.magnolia.high = faux.magnolia.high)
out_dir <- file.path("python", "ergmx", "data")
for (name in names(networks)) {
  path <- file.path(out_dir, paste0(name, ".graphml"))
  write_graph(to_igraph(networks[[name]]), path, format = "graphml")
  system2("gzip", c("-9", "-f", "-n", path))
  g <- to_igraph(networks[[name]])
  cat(sprintf("%-20s %4d vertices %5d edges  %s  attributes: %s\n", name, vcount(g), ecount(g),
              if (is_directed(g)) "directed  " else "undirected", paste(vertex_attr_names(g), collapse = ", ")))
}
