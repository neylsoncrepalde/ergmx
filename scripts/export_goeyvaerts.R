# Exports ergm.multi's Goeyvaerts household contact networks (318 networks)
# as python/ergmx/data/Goeyvaerts.json.gz (ergmx.datasets.load("Goeyvaerts")):
# for each network, its size, edges (numbered from 0), vertex attributes and
# network attributes.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/export_goeyvaerts.R

suppressMessages(library(ergm.multi))
data(Goeyvaerts)
networks <- lapply(Goeyvaerts, function(net) {
  vertex <- lapply(setNames(nm = c("age", "gender", "role")), function(a) net %v% a)
  vertex$name <- as.character(network.vertex.names(net))
  list(n = network.size(net), edges = unname(as.edgelist(net)) - 1L, vertex = vertex,
       graph = list(included = net %n% "included", weekday = net %n% "weekday"))
})
path <- file.path("python", "ergmx", "data", "Goeyvaerts.json")
jsonlite::write_json(networks, path, auto_unbox = TRUE, digits = NA, matrix = "rowmajor")
system2("gzip", c("-9", "-f", "-n", path))
cat(length(networks), "networks,", sum(sapply(Goeyvaerts, network.size)), "vertices,",
    sum(sapply(Goeyvaerts, network.edgecount)), "edges\n")
