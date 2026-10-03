# Reference results from R's ergm, used by the ergmx test suite.
#
# Exports the networks to tests/data/<network>.graphml, dyadic covariates to
# tests/data/<network>__<name>.csv (graph attributes in the tests), and writes
# tests/data/r_reference.json with, for every model and the checks it is used
# for: the observed statistics ("stats"), the MPLE ("mple"), and the MLE with
# its standard errors, log-likelihood and the time R took ("mle"). Models
# whose R formula or constraints refer to R objects give ergmx's version too
# (py_formula, py_constraints), with the objects as graph attributes.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_reference.R

suppressMessages({
  library(ergm)
  library(ergm.multi)  # Networks() and N()
  library(tergm)       # NetSeries() and the temporal operators
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
  if (network::is.bipartite(net)) {
    # The first mode is FALSE, as igraph's type attribute.
    b1 <- net %n% "bipartite"
    g <- set_vertex_attr(g, "type", value = seq_len(network::network.size(net)) > b1)
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
# And unknown nominations in the earlier waves, for NetSeries' NA.impute.
samplk1.na <- samplk1
samplk1.na[1, 2:5] <- NA
samplk1.na[3, 6:7] <- NA
samplk2.na <- samplk2
samplk2.na[3, 6:9] <- NA
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

# Bipartite networks: Davis's Southern Women, bundled with ergmx, and a
# simulated network with vertex attributes.
davis_graph <- read_graph(gzcon(file(file.path("python", "ergmx", "data", "davis.graphml.gz"), "rb")),
                          format = "graphml")
davis <- network::network(as.matrix(as_adjacency_matrix(davis_graph))[1:18, 19:32],
                          bipartite = 18, directed = FALSE)
network::network.vertex.names(davis) <- vertex_attr(davis_graph, "name")
set.seed(2026)
bipartite_sim <- network::network.initialize(100, bipartite = 60, directed = FALSE)
bipartite_sim %v% "g" <- sample(c("a", "b", "c"), 100, replace = TRUE)
bipartite_sim %v% "x" <- round(runif(100, 0, 10), 1)
bipartite_sim <- simulate(bipartite_sim ~ edges + b1factor("g") + b2cov("x") + gwb1degree(0.5, fixed = TRUE),
                          coef = c(-3.2, 0.4, -0.3, 0.05, -0.5), seed = 2026,
                          control = control.simulate.formula(MCMC.burnin = 100000))

# faux.mesa.high without the ties between grades, its vertices ordered by
# grade, for blockdiag().
by_grade <- order(faux.mesa.high %v% "Grade")
within <- as.matrix(faux.mesa.high)[by_grade, by_grade]
grade <- (faux.mesa.high %v% "Grade")[by_grade]
within[outer(grade, grade, "!=")] <- 0
faux.mesa.within <- network::network(within, directed = FALSE)
for (a in c("Grade", "Race", "Sex")) {
  network::set.vertex.attribute(faux.mesa.within, a, (faux.mesa.high %v% a)[by_grade])
}

# Egos for egocentric(): the seventh graders.
faux.mesa.high %v% "ego" <- (faux.mesa.high %v% "Grade") == 7
faux.dixon.high %v% "ego" <- (faux.dixon.high %v% "grade") == 7

networks <- list(flomarriage = flomarriage, samplk1 = samplk1, samplk2 = samplk2,
                 samplk3 = samplk3, linked_sim = linked_sim,
                 davis = davis, bipartite_sim = bipartite_sim,
                 faux.mesa.high = faux.mesa.high, faux.dixon.high = faux.dixon.high,
                 samplk3.nonresponse = samplk3.nonresponse, samplk1.na = samplk1.na,
                 samplk2.na = samplk2.na,
                 faux.mesa.high.missing = faux.mesa.high.missing, faux.mesa.within = faux.mesa.within)

# Dyadic covariates, stored as network attributes for edgecov("name").
set.seed(2026)
random_matrix <- function(n) {
  x <- matrix(round(runif(n * n), 3), n, n)  # asymmetric on purpose
  diag(x) <- 0
  x
}
random_neighbourhoods <- function(n) {
  x <- matrix(rbinom(n * n, 1, 0.3), n, n)
  x <- 1 * ((x + t(x)) > 0)
  diag(x) <- 0
  x
}
# bd(attribs=): each student's sex as classes, and at most one tie more to
# each sex than the student sends.
sexmat <- 1 * outer(faux.dixon.high %v% "sex", 1:2, "==")
maxsex <- as.matrix(faux.dixon.high) %*% sexmat + 1
covariates <- list(
  flomarriage = list(business = as.matrix(flobusiness)),
  faux.mesa.high = list(asym = random_matrix(network.size(faux.mesa.high))),
  faux.dixon.high = list(asym = random_matrix(network.size(faux.dixon.high)))
)
# Drawn after the others, which keep their values.
covariates$faux.mesa.high$nbhd_mesa <- random_neighbourhoods(network.size(faux.mesa.high))
covariates$faux.dixon.high <- c(covariates$faux.dixon.high, list(sexmat = sexmat, maxsex = maxsex))
covariates$samplk3 <- list(wave2 = as.matrix(samplk2), nbhd_samplk = random_neighbourhoods(network.size(samplk3)))
nbhd_mesa <- covariates$faux.mesa.high$nbhd_mesa
nbhd_samplk <- covariates$samplk3$nbhd_samplk
wave2 <- samplk2
sexmat <- sexmat == 1
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

# Networks combined from several, by name, as the ergmx tests rebuild them
# (tests/conftest.py): Sampson's monks at three times, jointly (Networks) and
# as a series of transitions (NetSeries), and ergm.multi's households whose
# contact diaries were kept on a weekday (bundled with ergmx).
data(Goeyvaerts)
combined <- list(
  samplk123.Networks = Networks(samplk1, samplk2, samplk3),
  samplk123.NetSeries = NetSeries(samplk1, samplk2, samplk3),
  samplk12.NetSeries = NetSeries(samplk1, samplk2),
  samplk123.na.next = NetSeries(samplk1.na, samplk2.na, samplk3.nonresponse, NA.impute = "next"),
  samplk123.na.previous = NetSeries(samplk1.na, samplk2.na, samplk3.nonresponse,
                                    NA.impute = c("previous", "majority")),
  Goeyvaerts.weekday = Networks(Filter(function(g) (g %n% "included") && (g %n% "weekday"), Goeyvaerts)),
  Goeyvaerts.included = Networks(Filter(function(g) g %n% "included", Goeyvaerts))
)

# The same formula strings are parsed by ergmx.
all_checks <- c("stats", "mple", "mle")
# Edge lists for fixedas() and fixallbut() on flomarriage, as R code.
r_edges <- function(m) sprintf("matrix(c(%s), ncol=2, byrow=TRUE)", paste(t(m), collapse = ", "))
flo_ties <- network::as.edgelist(flomarriage)
flo_absent <- which(as.matrix(flomarriage) == 0 & upper.tri(as.matrix(flomarriage)), arr.ind = TRUE)
set.seed(2026)
flo_free <- rbind(flo_ties[1:8, ], flo_absent[sample(nrow(flo_absent), 40), ])
models <- list(
  # More of ergm's vocabulary: statistics.
  mesa_vocab1 = list(network = "faux.mesa.high", checks = "stats",
                     formula = paste("degrange(1:2, 4) + degrange(2, by='Race') + degree(1:2, by='Race') +",
                                     "degree(0:1, by='Sex', homophily=TRUE) + concurrent(by='Sex') +",
                                     "concurrentties + concurrentties(by='Sex') + degree1.5 + isolatededges +",
                                     "density + meandeg + altkstar(2, fixed=TRUE) + degrange(1, 3, by='Race', homophily=TRUE)")),
  mesa_vocab2 = list(network = "faux.mesa.high", checks = "stats",
                     formula = paste("triadcensus(0:3) + balance + threetrail + opentriad + transitiveties +",
                                     "transitiveties(attr='Race') + cyclicalties + kstar(2:3, attr='Sex') +",
                                     "triangle(attr='Race') + triangle(attr='Grade', diff=TRUE) + nodecovrange('Grade') +",
                                     "nodefactordistinct('Race') + gwdegree(0.5, fixed=TRUE, attr='Sex')")),
  mesa_vocab3 = list(network = "faux.mesa.high", checks = "stats",
                     formula = paste("mm('Race') + mm(Race~Sex) + mm(Race~.) + attrcov('Sex', matrix(c(1,2,2,3),2)) +",
                                     "smalldiff('Grade', 2) + diff('Grade', sign.action='abs') +",
                                     "nodematch('Race', diff=TRUE, levels=-1) + nodefactor('Race', levels=c('White','Black')) +",
                                     "sociality(nodes=c(1, 3)) + nodecov('Grade'):nodematch('Sex') +",
                                     "nodefactor('Race'):nodecov('Grade') + dyadcov('asym') + hamming +",
                                     "localtriangle(nbhd_mesa) + nodefactordistinct('Race', levels=c('Hisp','White'))"),
                     py_formula = paste("mm('Race') + mm(Race~Sex) + mm(Race~.) + attrcov('Sex', matrix(c(1,2,2,3),2)) +",
                                        "smalldiff('Grade', 2) + diff('Grade', sign.action='abs') +",
                                        "nodematch('Race', diff=TRUE, levels=-1) + nodefactor('Race', levels=c('White','Black')) +",
                                        "sociality(nodes=c(1, 3)) + nodecov('Grade'):nodematch('Sex') +",
                                        "nodefactor('Race'):nodecov('Grade') + dyadcov('asym') + hamming +",
                                        "localtriangle('nbhd_mesa') + nodefactordistinct('Race', levels=c('Hisp','White'))")),
  dixon_vocab1 = list(network = "faux.dixon.high", checks = "stats",
                      formula = paste("triadcensus + simmelian + simmelianties + nearsimmelian + m2star + ctriad +",
                                      "ttriad + transitiveties + transitiveties(attr='sex') + cyclicalties +",
                                      "cyclicalties(attr='race') + threetrail + balance + triangle(attr='race')")),
  dixon_vocab2 = list(network = "faux.dixon.high", checks = "stats",
                      formula = paste("idegree(0:2, by='sex') + odegrange(c(0,3), c(3, Inf)) +",
                                      "idegrange(1, 3, by='race', homophily=TRUE) + idegree1.5 + odegree1.5 +",
                                      "nodeicovrange('grade') + nodeocovrange('grade') + nodecovrange('grade') +",
                                      "nodeifactor('race', levels=c('W','B')) + sender(nodes=c(2,5)) +",
                                      "receiver(nodes=-(1:240)) + mutual(same='sex') + mutual(by='race') +",
                                      "mutual(same='race', diff=TRUE) + asymmetric(attr='sex') +",
                                      "asymmetric(attr='race', diff=TRUE) + istar(2, attr='sex') + ostar(2:3, attr='race') +",
                                      "desp(0:2) + dgwesp(0.5, fixed=TRUE) + dnsp(1, type='OSP') + mm(race~sex) + mm(.~sex) +",
                                      "nodefactordistinct('race') + nodeofactordistinct('race') + nodeifactordistinct('race') +",
                                      "ttriple(attr='sex', diff=TRUE) + ctriple(attr='race') + dyadcov('asym') +",
                                      "gwodegree(0.5, fixed=TRUE, attr='race') + nodeifactor('race'):nodeofactor('sex') +",
                                      "density + meandeg")),
  samplk_vocab = list(network = "samplk3", checks = "stats",
                      formula = "threetrail + hamming(wave2) + localtriangle(nbhd_samplk) + triadcensus(c('021D','300'))",
                      py_formula = "threetrail + hamming('wave2') + localtriangle('nbhd_samplk') + triadcensus(c('021D','300'))"),
  bip_vocab1 = list(network = "bipartite_sim", checks = "stats",
                    formula = paste("b1twostar('g') + b2twostar('g') + b1starmix(2, 'g') + b2starmix(2, 'g') +",
                                    "b1sociality + b2sociality(nodes=c(1,2)) + b1nodematch('g', diff=TRUE) +",
                                    "b1nodematch('g', byb2attr='g', diff=TRUE) + b1degrange(1:2, 4) + b1degrange(1, by='g') +",
                                    "b2degrange(0, 2, by='g', homophily=TRUE) + b1mindegree(2:3) + b2mindegree(1) +",
                                    "b1covrange('x') + b2covrange('x') + b1factordistinct('g') + b2factordistinct('g') +",
                                    "b1concurrent(by='g') + b2concurrent(by='g') + b1degree(1:2, by='g') +",
                                    "b2degree(1, by='g', levels=-1) + b1factor('g', levels=TRUE) +",
                                    "b2factor('g', levels=c('c','a')) + density + meandeg + isolatededges + diff('x') +",
                                    "b1star(2, attr='g') + b2star(2, attr='g') + gwb1degree(0.5, fixed=TRUE, attr='g')")),
  bip_vocab2 = list(network = "bipartite_sim", checks = "stats",
                    formula = "b1nodematch('g', beta=0.5) + b2nodematch('g', alpha=0.25) + b1starmix(2, 'g', diff=FALSE)"),
  bip_vocab3 = list(network = "bipartite_sim", checks = "stats",
                    formula = "b1nodematch('g', alpha=0.5) + b1nodematch('g', levels=c('a','c'), diff=TRUE)"),
  # Dyad-independent models of the new terms, and the new constraints.
  mesa_mm_dyadind = list(network = "faux.mesa.high", checks = all_checks,
                         formula = "edges + mm(Sex~Grade) + nodecov('Grade'):nodematch('Sex')"),
  mesa_attrcov_dyadind = list(network = "faux.mesa.high", checks = all_checks,
                              formula = paste("edges + attrcov('Race', matrix(c(2,1,1,1,0, 1,2,1,1,0, 1,1,2,1,0,",
                                              "1,1,1,2,0, 0,0,0,0,1), 5)) + smalldiff('Grade', 1) +",
                                              "diff('Grade', sign.action='abs')")),
  dixon_dyadcov_dyadind = list(network = "faux.dixon.high", checks = all_checks,
                               formula = paste("edges + dyadcov('asym') + nodeifactor('race', levels=c('W','B')) +",
                                               "diff('grade', pow=2, sign.action='posonly')")),
  mesa_dyads_fix = list(network = "faux.mesa.high", checks = all_checks,
                        formula = "edges + nodematch('Grade') + nodematch('Race')",
                        constraints = "Dyads(fix=~nodematch('Sex'))"),
  mesa_dyads_vary = list(network = "faux.mesa.high", checks = all_checks, formula = "edges + nodefactor('Sex')",
                         constraints = "Dyads(vary=~nodematch('Grade'))"),
  flo_fixedas = list(network = "flomarriage", checks = all_checks, formula = "edges + nodecov('wealth')",
                     constraints = sprintf("fixedas(present=%s, absent=%s)", r_edges(flo_ties[1:3, ]),
                                           r_edges(flo_absent[1:3, ]))),
  flo_fixallbut = list(network = "flomarriage", checks = all_checks, formula = "edges + nodecov('wealth')",
                       constraints = sprintf("fixallbut(%s)", r_edges(flo_free))),
  mesa_blockdiag = list(network = "faux.mesa.within", checks = all_checks,
                        formula = "edges + nodefactor('Sex') + nodematch('Race')", constraints = "blockdiag('Grade')"),
  # Monte Carlo MLEs with new terms and constraints.
  mesa_vocab_mle = list(network = "faux.mesa.high", checks = c("stats", "mle"),
                        formula = "edges + nodematch('Grade') + degree(1, by='Sex') + gwesp(0.5, fixed=TRUE)"),
  mesa_edges_constraint = list(network = "faux.mesa.high", checks = c("stats", "mle"),
                               formula = "nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)",
                               constraints = "edges"),
  bip_b1degrees = list(network = "bipartite_sim", checks = c("stats", "mle"),
                       formula = "b2star(2) + b2factor('g')", constraints = "b1degrees"),
  dixon_bd_attribs = list(network = "faux.dixon.high", checks = c("stats", "mle"),
                          formula = "edges + mutual + nodematch('race')",
                          constraints = "bd(attribs=sexmat, maxout=maxsex)",
                          py_constraints = "bd(attribs='sexmat', maxout='maxsex')"),
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

  # More terms. ergm 4.12's edgewise RTP statistics are wrong with its
  # shared-partner cache, on by default (see rtp below), so they are left out.
  mesa_terms3 = list(network = "faux.mesa.high", checks = "stats",
                     formula = paste("cycle(3:5) + twopath + concurrent + gwnsp(0.5, fixed=TRUE) +",
                                     "absdiffcat('Grade') + sociality + nsp(0:2)")),
  dixon_terms3 = list(network = "faux.dixon.high", checks = "stats",
                      formula = paste("cycle(2:4) + twopath + asymmetric +",
                                      "gwnsp(0.5, fixed=TRUE) + dsp(0:2, type='RTP') +",
                                      "gwdsp(0.5, fixed=TRUE, type='RTP') +",
                                      "esp(0:3, type='ITP') + gwesp(0.5, fixed=TRUE, type='ITP') +",
                                      "esp(0:3, type='OSP') + dsp(0:2, type='OSP') +",
                                      "gwesp(0.5, fixed=TRUE, type='OSP') + gwnsp(0.5, fixed=TRUE, type='OSP') +",
                                      "esp(0:3, type='ISP') + dsp(0:2, type='ISP') +",
                                      "gwdsp(0.5, fixed=TRUE, type='ISP') + nsp(1, type='ITP')")),
  samplk_vertex = list(network = "samplk3", checks = "stats", formula = "sender + receiver"),
  mesa_mple3 = list(network = "faux.mesa.high", checks = c("stats", "mple"),
                    formula = "edges + concurrent + cycle(4) + nodematch('Grade') + gwnsp(0.5, fixed=TRUE)"),
  dixon_mple3 = list(network = "faux.dixon.high", checks = c("stats", "mple"),
                     formula = paste("edges + mutual + twopath + cycle(3) +",
                                     "gwesp(0.5, fixed=TRUE, type='ITP') + gwdsp(0.5, fixed=TRUE, type='OSP') +",
                                     "esp(1, type='ISP')")),
  mesa_absdiffcat = list(network = "faux.mesa.high", checks = all_checks,
                         formula = "edges + absdiffcat('Grade')"),
  mesa_concurrent = list(network = "faux.mesa.high", checks = c("stats", "mle"),
                         formula = "edges + nodematch('Grade') + concurrent + gwesp(0.5, fixed=TRUE)"),
  samplk_osp = list(network = "samplk3", checks = all_checks,
                    formula = "edges + mutual + gwesp(0.5, fixed=TRUE, type='OSP')"),
  samplk_twopath = list(network = "samplk3", checks = all_checks, formula = "edges + mutual + twopath"),

  # Bipartite networks.
  davis_terms = list(network = "davis", checks = "stats",
                     formula = paste("edges + b1degree(0:3) + b2degree(1:2) + b1star(2:3) + b2star(2) +",
                                     "gwb1degree(0.5, fixed=TRUE) + gwb2degree(0.5, fixed=TRUE) +",
                                     "b1concurrent + b2concurrent + cycle(4) + gwb1dsp(0.5, fixed=TRUE) +",
                                     "gwb2dsp(0.5, fixed=TRUE) + b1dsp(0:2) + b2dsp(1)")),
  bipartite_terms = list(network = "bipartite_sim", checks = "stats",
                         formula = paste("edges + b1factor('g') + b2factor('g') + b1cov('x') +",
                                         "b2cov('x') + b1nodematch('g') + b2nodematch('g') +",
                                         "b1star(2) + b2degree(0:2)")),
  bipartite_dyadind = list(network = "bipartite_sim", checks = all_checks,
                           formula = "edges + b1factor('g') + b2factor('g') + b1cov('x') + b2cov('x')"),
  bipartite_gw = list(network = "bipartite_sim", checks = all_checks,
                      formula = paste("edges + b1factor('g') + gwb1degree(0.5, fixed=TRUE) +",
                                      "gwb2degree(0.5, fixed=TRUE)")),
  bipartite_match = list(network = "bipartite_sim", checks = all_checks,
                         formula = "edges + b1nodematch('g') + b2star(2)"),
  bipartite_dsp = list(network = "bipartite_sim", checks = all_checks,
                       formula = "edges + gwb1dsp(0.5, fixed=TRUE) + b2factor('g')"),
  davis_dsp = list(network = "davis", checks = all_checks, formula = "edges + gwb1dsp(0.5, fixed=TRUE)"),

  # Curved terms (fixed=FALSE). R's default start fails on mesa_curved (the
  # cutoff is exceeded during its MCMC), so R starts it from CD.
  mesa_curved_mple = list(network = "faux.mesa.high", checks = c("stats", "mple"),
                          formula = "edges + nodematch('Grade') + gwesp(0.5)"),
  mesa_curved = list(network = "faux.mesa.high", checks = c("stats", "mle"), init_method = "CD",
                     formula = "edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5)"),
  bipartite_curved = list(network = "bipartite_sim", checks = all_checks,
                          formula = "edges + b1factor('g') + gwb1degree(0.5)"),
  flo_curved = list(network = "flomarriage", checks = c("stats", "mple"), formula = "edges + gwdegree(0.5)"),
  dixon_curved = list(network = "faux.dixon.high", checks = c("stats", "mle"), init_method = "CD",
                      formula = "edges + mutual + nodematch('grade') + nodematch('race') + gwesp(0.1)"),

  # Missing dyads. ergm imputes missing dyads at random before its MPLE, so the
  # MPLE of dyad-dependent models with missing dyads is random: not compared.
  samplk_missing = list(network = "samplk3.nonresponse", checks = c("stats", "mle"),
                        formula = "edges + mutual"),
  mesa_missing_dyadind = list(network = "faux.mesa.high.missing", checks = all_checks,
                              formula = "edges + nodematch('Grade') + nodefactor('Sex')"),
  mesa_missing = list(network = "faux.mesa.high.missing", checks = c("stats", "mle"),
                      formula = paste("edges + nodematch('Grade') + nodematch('Race') +",
                                      "gwesp(0.5, fixed=TRUE)")),

  # S(): terms on subgraphs, such as the levels of a multilevel network and
  # the ties between them (MPNet's within-level and meso-level effects).
  linked_S_stats = list(network = "linked_sim", checks = "stats",
                        formula = paste("S(~edges + kstar(2) + gwesp(0.5, fixed=TRUE) + triangle + isolates, ~level == 'individual') +",
                                        "S(~edges + gwdegree(0.5, fixed=TRUE) + dsp(0:2), ~level == 'organization') +",
                                        "S(~edges + b1star(2) + b2star(2) + gwb1degree(0.5, fixed=TRUE) + cycle(4) +",
                                        "gwb1dsp(0.5, fixed=TRUE) + b2degree(0:2), (level == 'individual') ~ (level == 'organization')) +",
                                        "S(~edges, ~type) + S(~edges, ~!type)")),
  linked_S_dyadind = list(network = "linked_sim", checks = all_checks,
                          formula = paste("S(~edges, ~level == 'individual') + S(~edges, ~level == 'organization') +",
                                          "S(~edges, (level == 'individual') ~ (level == 'organization'))")),
  linked_S = list(network = "linked_sim", checks = all_checks,
                  formula = paste("S(~edges + gwesp(0.5, fixed=TRUE), ~level == 'individual') +",
                                  "S(~edges + gwesp(0.5, fixed=TRUE), ~level == 'organization') +",
                                  "S(~edges + gwb1dsp(0.5, fixed=TRUE), (level == 'individual') ~ (level == 'organization'))")),
  samplk_S = list(network = "samplk3", checks = c("stats", "mple"),
                  formula = "edges + S(~edges + mutual + ttriple, ~group == 'Turks') + S(~mutual, ~cloisterville)"),

  # Samples of networks (ergm.multi).
  multi_stats = list(network = "samplk123.Networks", checks = "stats",
                     formula = paste("N(~edges + mutual) + N(~edges, lm=~I(.NetworkID <= 2)) +",
                                     "N(~gwesp(0.5, fixed=TRUE) + nodematch('group'), lm=~0 + factor(.NetworkID)) +",
                                     "edges + mutual + dsp(0:1) + N(~dsp(0:1))")),
  multi_mutual = list(network = "samplk123.Networks", checks = all_checks, formula = "N(~edges + mutual)"),
  # ergm.multi and tergm fail to fit curved terms inside N() and Form() (the
  # MPLE's gradient is not finite, CD stops with an error): statistics only.
  multi_curved = list(network = "samplk123.Networks", checks = "stats",
                      formula = "N(~edges + mutual + gwesp(0.5, cutoff=10))"),
  goey_dyadind = list(network = "Goeyvaerts.weekday", checks = all_checks,
                      formula = "N(~edges, ~I(n<=3) + I(n>=5)) + N(~nodematch('gender') + absdiff('age'))"),
  goey_weekday = list(network = "Goeyvaerts.weekday", checks = all_checks,
                      formula = paste("N(~edges, ~I(n<=3) + I(n>=5)) +",
                                      "N(~kstar(2) + nodematch('gender') + absdiff('age')) +",
                                      "N(~triangle, ~I(n>=6))")),
  # N()'s subset, offset and label.
  goey_offset = list(network = "Goeyvaerts.weekday", checks = all_checks,
                     formula = "N(~edges + nodematch('gender'), offset=~log(n))"),
  goey_subset = list(network = "Goeyvaerts.weekday", checks = all_checks,
                     formula = paste("N(~edges, subset=~n>=4, lm=~I(n>=5)) +",
                                     "N(~nodematch('gender') + absdiff('age'), lm=~1+offset(I(n<=2)))")),
  goey_subset_label = list(network = "Goeyvaerts.weekday", checks = all_checks,
                           formula = paste("N(~edges + triangle, subset=~n>=4, label='big') +",
                                           "N(~edges, subset=c(TRUE, FALSE), label='odd')")),

  # Series of networks (tergm's CMLE).
  series_stats = list(network = "samplk123.NetSeries", checks = "stats",
                      formula = paste("Form(~edges + mutual + gwesp(0.5, fixed=TRUE) + dsp(0:1)) +",
                                      "Persist(~mutual + ttriple) + Diss(~edges) + Cross(~edges + mutual) +",
                                      "Change(~edges + mutual) + Form(~edges, lm=~.Time) +",
                                      "Cross(~edges, lm=~0 + factor(.TimeID)) + edges + nodematch('group')")),
  series_dyadind = list(network = "samplk123.NetSeries", checks = all_checks,
                        formula = "Form(~edges + nodematch('group')) + Diss(~edges + nodematch('group'))"),
  series_gwesp = list(network = "samplk123.NetSeries", checks = all_checks,
                      formula = "Form(~edges + mutual + gwesp(0.5, fixed=TRUE)) + Persist(~edges + mutual)"),
  series_one = list(network = "samplk12.NetSeries", checks = all_checks,
                    formula = "Form(~edges + mutual) + Diss(~edges + mutual)"),
  series_curved = list(network = "samplk123.NetSeries", checks = "stats",
                       formula = "Form(~edges + mutual + gwesp(0.5, cutoff=10)) + Persist(~edges)"),
  # Missing dyads: imputed in the networks transitioned from, missing in those
  # transitioned to.
  series_na_next_dyadind = list(network = "samplk123.na.next", checks = all_checks,
                                formula = "Form(~edges + nodematch('group')) + Persist(~edges)"),
  series_na_next = list(network = "samplk123.na.next", checks = c("stats", "mle"),
                        formula = "Form(~edges + mutual) + Persist(~edges + mutual)"),
  series_na_previous = list(network = "samplk123.na.previous", checks = all_checks,
                            formula = "Form(~edges + nodematch('group')) + Diss(~edges)"),

  # Degree correlation, triangle percentage and coincidence; the degree
  # distribution and egocentric constraints; N()'s contrasts.
  flo_degcor = list(network = "flomarriage", checks = c("stats", "mple"), formula = "edges + degcor + tripercent"),
  mesa_tripercent = list(network = "faux.mesa.high", checks = c("stats", "mple"),
                         formula = "edges + degcor + tripercent('Grade') + tripercent('Sex', diff=TRUE)"),
  davis_coincidence = list(network = "davis", checks = c("stats", "mple"),
                           formula = "edges + coincidence(levels=c(1, 2, 5, 40))"),
  bip_coincidence = list(network = "bipartite_sim", checks = "stats", formula = "coincidence"),
  mesa_degreedist = list(network = "faux.mesa.high", checks = c("stats", "mle"),
                         formula = "nodematch('Grade') + nodematch('Race') + gwesp(0.5, fixed=TRUE)",
                         constraints = "degreedist"),
  mesa_egocentric = list(network = "faux.mesa.high", checks = all_checks,
                         formula = "edges + nodematch('Grade') + nodefactor('Sex')",
                         constraints = "egocentric('ego')"),
  dixon_egocentric_out = list(network = "faux.dixon.high", checks = all_checks,
                              formula = "edges + nodematch('race') + nodeofactor('sex') + nodeifactor('sex')",
                              constraints = "egocentric('ego', direction='out')"),
  # Fitted to target statistics (simulated annealing, then the MLE).
  flo_target = list(network = "flomarriage", checks = "mle", formula = "edges + nodecov('wealth')",
                    target_stats = c(25, 3000)),
  mesa_target = list(network = "faux.mesa.high", checks = "mle",
                     formula = "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)",
                     target_stats = c(250, 190, 200)),
  goey_contrasts_sum = list(network = "Goeyvaerts.included", checks = all_checks,
                            formula = "N(~edges + nodematch('gender'), lm=~weekday, contrasts=list(weekday='contr.sum'))"),
  goey_contrasts_helmert = list(network = "Goeyvaerts.included", checks = all_checks,
                                formula = paste("N(~edges, lm=~factor(n), subset=~n>=3 & n<=5,",
                                                "contrasts=list(`factor(n)`='contr.helmert'))")),
  goey_contrasts_poly = list(network = "Goeyvaerts.included", checks = all_checks,
                             formula = paste("N(~edges + nodematch('gender'), lm=~factor(n), subset=~n>=3,",
                                             "contrasts=list(`factor(n)`='contr.poly'))"))
)

named <- function(x) as.list(x)

results <- list()
# ERGMX_ONLY=name1,name2 runs only those models, written to r_reference.only.json.
only <- Sys.getenv("ERGMX_ONLY")
if (nzchar(only)) models <- models[strsplit(only, ",")[[1]]]
for (name in names(models)) {
  m <- models[[name]]
  net <- c(networks, combined)[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  constraints <- as.formula(paste("~", if (is.null(m$constraints)) "." else m$constraints))
  fit_ergm <- function(...) {
    if (is.null(m$constraints)) ergm(f, offset.coef = m$offset_coef, target.stats = m$target_stats, ...)
    else ergm(f, constraints = constraints, offset.coef = m$offset_coef, target.stats = m$target_stats, ...)
  }
  fit_control <- if (is.null(m$init_method)) control.ergm(seed = 1)
                 else control.ergm(seed = 1, init.method = m$init_method)
  result <- list(network = m$network, formula = m$formula, checks = as.list(m$checks),
                 constraints = m$constraints, offset_coef = m$offset_coef, target_stats = m$target_stats,
                 py_formula = m$py_formula, py_constraints = m$py_constraints,
                 bipartite = network::is.bipartite(net), stats = named(summary(f)))
  if ("mple" %in% m$checks) {
    result$mple <- named(coef(fit_ergm(estimate = "MPLE")))
  }
  if ("mle" %in% m$checks) {
    seconds <- system.time(fit <- fit_ergm(control = fit_control))[["elapsed"]]
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

# Two places where ergm 4.12's computation differs from its documentation.
#
# 1. Edgewise RTP statistics (esp, gwesp, nsp with type = "RTP"): ergm's
#    shared-partner cache, on by default, reads them with the wrong key when
#    tail > head, so they depend on the order of the vertices. Fixed in ergm's
#    development version (statnet/ergm#656, merged 2026-07-17, not yet on
#    CRAN). Recorded: the statistics from the definition, ergm's with the cache
#    (original and relabeled network) and without it, which match.
d <- faux.dixon.high
A <- as.matrix(d)
M <- A * t(A)  # reciprocated ties
el <- network::as.edgelist(d)
rtp <- sapply(seq_len(nrow(el)), function(e) sum(M[el[e, 1], ] * M[, el[e, 2]]))
r <- 1 - exp(-0.5)
rtp_direct <- list(esp = tabulate(rtp + 1, 4), gwesp = exp(0.5) * sum(1 - r^rtp),
                   summary_original = as.numeric(summary(d ~ esp(0:3, type = "RTP"))),
                   summary_relabeled = {
                     set.seed(1); perm <- sample(network.size(d))
                     as.numeric(summary(network::network(A[perm, perm], directed = TRUE) ~
                                          esp(0:3, type = "RTP")))
                   },
                   uncached = named(summary(d ~ esp(0:3, type = "RTP") +
                                              gwesp(0.5, fixed = TRUE, type = "RTP") +
                                              nsp(0:2, type = "RTP") +
                                              gwnsp(0.5, fixed = TRUE, type = "RTP"),
                                            term.options = list(cache.sp = FALSE))))

# 2. transitive is documented as the number of transitive triads (types 030T,
#    120D, 120U and 300) but computes transitive triples, as ttriple: ergm's
#    value, ttriple, and the triads counted with igraph's triad census.
transitive <- lapply(list(samplk3 = samplk3, faux.dixon.high = faux.dixon.high), function(net) {
  census <- triad_census(graph_from_adjacency_matrix(as.matrix(net), mode = "directed"))
  names(census) <- c("003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C",
                     "201", "120D", "120U", "120C", "210", "300")
  list(ergm_transitive = as.numeric(summary(net ~ transitive)),
       ttriple = as.numeric(summary(net ~ ttriple)),
       triads = sum(census[c("030T", "120D", "120U", "300")]))
})

# 3. dyadcov's utri is documented as the dyads in the upper triangular
#    asymmetric state (only the tie from the lower- to the higher-numbered
#    vertex), but ergm counts those as ltri, and the reverse as utri.
dyadcov_states <- {
  one <- network.initialize(3); one[1, 2] <- 1
  ones <- matrix(1, 3, 3)
  named(summary(one ~ dyadcov(ones)))
}

# 4. intransitive is documented as the number of intransitive triads (types
#    111D, 201, 111U, 021C and 030C) but computes intransitive triples
#    (twopath - ttriple): ergm's value, twopath - ttriple, and the triads.
intransitive <- lapply(list(samplk3 = samplk3, faux.dixon.high = faux.dixon.high), function(net) {
  census <- triad_census(graph_from_adjacency_matrix(as.matrix(net), mode = "directed"))
  names(census) <- c("003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C",
                     "201", "120D", "120U", "120C", "210", "300")
  list(ergm_intransitive = as.numeric(summary(net ~ intransitive)),
       twopath_minus_ttriple = as.numeric(summary(net ~ twopath) - summary(net ~ ttriple)),
       triads = sum(census[c("111D", "201", "111U", "021C", "030C")]))
})

# 5. degcrossprod is documented as the mean over ties of the product of their
#    degrees, but ergm computes half of it; and coincidence(active=) is
#    documented as keeping the pairs with at least active partners in common,
#    but keeps those with more.
cross_products <- {
  deg <- summary(flomarriage ~ sociality(nodes = TRUE))
  el <- network::as.edgelist(flomarriage)
  list(ergm_degcrossprod = as.numeric(summary(flomarriage ~ degcrossprod)),
       mean = mean(deg[el[, 1]] * deg[el[, 2]]))
}
coincidence_active <- {
  counts <- summary(davis ~ coincidence)
  list(active = 3, ergm = names(summary(davis ~ coincidence(active = 3))),
       at_least = names(counts)[counts >= 3])
}

# ergm.multi's gofN() on the households (a dyad-independent fit, whose MLE
# ergmx reproduces exactly): each network's observed statistics, and the mean
# and variance of 2000 simulated ones, for the model's statistics (the
# default) and for other statistics. ergm.multi leaves out the statistics of
# the empty network, so it reports degree0 and isolates minus the network size.
gofn <- if (nzchar(only)) NULL else {
  goey <- combined$Goeyvaerts.weekday
  gofn_formula <- "N(~edges, ~I(n<=3) + I(n>=5)) + N(~nodematch('gender') + absdiff('age'))"
  gofn_fit <- ergm(as.formula(paste("goey ~", gofn_formula)), control = control.ergm(seed = 1))
  per_network <- function(g) lapply(g, function(x) lapply(x[c("observed", "fitted", "var", "var.obs", "pearson")],
                                                         function(v) ifelse(is.na(v), NA, v)))
  set.seed(1)
  default <- suppressMessages(gofN(gofn_fit, control = control.gofN.ergm(nsim = 2000)))
  gof_formula <- "edges + triangle + degree(0:2) + isolates"
  set.seed(2)
  other <- suppressMessages(gofN(gofn_fit, GOF = as.formula(paste("~", gof_formula)),
                                 control = control.gofN.ergm(nsim = 2000)))
  # lm.gofN() of R's gofN: the linear models of the residuals ergmx refits
  # from the same table.
  lm_formula <- "c('edges', 'triangle', 'degree1') ~ n + I(n^2)"
  lms <- lm.gofN(as.formula(lm_formula), data = other)
  lm_results <- lapply(lms, function(l) {
    s <- summary(l)
    list(coef = named(coef(l)), se = named(s$coefficients[, "Std. Error"]), sigma = s$sigma,
         df = s$df[2], r_squared = s$r.squared, adj_r_squared = s$adj.r.squared,
         fstatistic = as.numeric(s$fstatistic), dropped = length(l$na.action))
  })
  list(network = "Goeyvaerts.weekday", formula = gofn_formula, coef = named(coef(gofn_fit)), nsim = 2000,
       default = per_network(default), gof_formula = gof_formula, gof = per_network(other),
       network_size = sapply(subnetwork_templates(goey), network.size),
       lm_formula = lm_formula, lm = lm_results)
}

jsonlite::write_json(
  list(ergm_version = as.character(packageVersion("ergm")),
       r_version = R.version.string, models = results, gofn = gofn,
       ergm_differences = list(rtp = rtp_direct, transitive = transitive, intransitive = intransitive,
                               dyadcov_lower_to_higher_tie = dyadcov_states,
                               degcrossprod = cross_products, coincidence_active = coincidence_active)),
  file.path(out_dir, if (nzchar(only)) "r_reference.only.json" else "r_reference.json"),
  auto_unbox = TRUE, digits = NA, pretty = TRUE
)
