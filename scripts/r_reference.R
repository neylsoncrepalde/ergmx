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

networks <- list(flomarriage = flomarriage, samplk1 = samplk1, samplk2 = samplk2,
                 samplk3 = samplk3, linked_sim = linked_sim,
                 davis = davis, bipartite_sim = bipartite_sim,
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

# Networks combined from several, by name, as the ergmx tests rebuild them
# (tests/conftest.py): Sampson's monks at three times, jointly (Networks) and
# as a series of transitions (NetSeries), and ergm.multi's households whose
# contact diaries were kept on a weekday (bundled with ergmx).
data(Goeyvaerts)
combined <- list(
  samplk123.Networks = Networks(samplk1, samplk2, samplk3),
  samplk123.NetSeries = NetSeries(samplk1, samplk2, samplk3),
  samplk12.NetSeries = NetSeries(samplk1, samplk2),
  Goeyvaerts.weekday = Networks(Filter(function(g) (g %n% "included") && (g %n% "weekday"), Goeyvaerts))
)

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
                       formula = "Form(~edges + mutual + gwesp(0.5, cutoff=10)) + Persist(~edges)")
)

named <- function(x) as.list(x)

results <- list()
for (name in names(models)) {
  m <- models[[name]]
  net <- c(networks, combined)[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  constraints <- as.formula(paste("~", if (is.null(m$constraints)) "." else m$constraints))
  fit_ergm <- function(...) {
    if (is.null(m$constraints)) ergm(f, offset.coef = m$offset_coef, ...)
    else ergm(f, constraints = constraints, offset.coef = m$offset_coef, ...)
  }
  fit_control <- if (is.null(m$init_method)) control.ergm(seed = 1)
                 else control.ergm(seed = 1, init.method = m$init_method)
  result <- list(network = m$network, formula = m$formula, checks = as.list(m$checks),
                 constraints = m$constraints, offset_coef = m$offset_coef,
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

jsonlite::write_json(
  list(ergm_version = as.character(packageVersion("ergm")),
       r_version = R.version.string, models = results,
       ergm_differences = list(rtp = rtp_direct, transitive = transitive)),
  file.path(out_dir, "r_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE
)
