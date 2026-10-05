# Reference results from R's ergm 4.12, ergm.multi 0.3.0 and ergm.count for
# the features of ergmx 0.5.0, used by tests/test_gaps.py: semicycles, interactions in N()'s linear models (and
# R's model.matrix() on a data frame), L() of curved terms and the gw layer
# terms with an estimated decay, bipartite layers (b1dspL, b2dspL), ergm's
# projection operators, and valued models of several networks; statistics,
# names and fits.
#
# ergm.multi 0.3.0's Layer() of bipartite networks misplaces their ties (its
# combined network's edge list has pairs such as (19, 19), and each layer
# loses ties), so only the names of its bipartite layer statistics are used:
# the tests count their values by brute force.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_gaps_reference.R

suppressMessages({
  library(ergm.multi)
  library(ergm.count)
  library(igraph, warn.conflicts = FALSE)
})
data(florentine)
data(samplk)
data(Goeyvaerts)
data(zach)
data(faux.mesa.high)

stats_of <- function(net, formula, ...) {
  s <- summary(as.formula(paste("net ~", formula)), ...)
  list(formula = formula, names = names(s), values = unname(as.numeric(s)))
}

# Davis's Southern Women, bundled with ergmx, and a second layer: the
# attendances whose woman's and event's numbers (from 1, in each mode) don't
# add up to a multiple of 3.
davis_graph <- read_graph(gzcon(file(file.path("python", "ergmx", "data", "davis.graphml.gz"), "rb")),
                          format = "graphml")
attendance <- as.matrix(as_adjacency_matrix(davis_graph))[1:18, 19:32]
davis <- network::network(attendance, bipartite = 18, directed = FALSE)
kept <- attendance * (outer(1:18, 1:14, "+") %% 3 != 0)
davis2 <- network::network(kept, bipartite = 18, directed = FALSE)
davis_layers <- Layer(a = davis, b = davis2)
flo <- Layer(m = flomarriage, b = flobusiness)

# The karate club's counts of contexts, and the counts less one (at least 1).
zach2 <- zach
zach2 %e% "contexts" <- pmax(1, (zach %e% "contexts") - 1)
zachs <- Networks(zach, zach2)

# 80 of the households that Goeyvaerts et al. analysed.
included <- Filter(function(g) g %n% "included", Goeyvaerts)
households <- Networks(included[1:80])

stats <- list(
  semicycles = stats_of(samplk3, "cycle(3:5, semi = TRUE) + cycle(2:4)"),
  lm_interactions = stats_of(households, "N(~edges + triangle, ~n * weekday) + N(~edges, ~log(n):weekday)"),
  curved_L = stats_of(flo, "L(~gwesp(0.5), ~m) + L(~gwdegree(0.25), c(~m, ~b))"),
  curved_gwespL = stats_of(flo, "gwespL(0.5, L.base = ~m, Ls.path = ~b) + gwdspL(0.5, Ls.path = c(~m, ~b))"),
  bipartite_layers = stats_of(davis_layers, paste("L(~edges, ~a & b) + b1dspL(0:2, Ls.path = c(~a, ~b)) +",
                                                  "b2dspL(1:3, Ls.path = c(~a, ~b)) + gwb1dspL(0.5, fixed = TRUE, Ls.path = c(~a, ~b)) +",
                                                  "gwb2dspL(0.25, fixed = TRUE, Ls.path = ~b) + L(~b1degree(1:2) + gwb2degree(0.5, fixed = TRUE), ~a | b)")),
  projections = stats_of(davis, "Proj1(~sum + nonzero) + Proj2(~sum + nonzero + atleast(2)) + Project(~sum(pow = 2) + transitiveweights + CMP, 1)"),
  valued_N = stats_of(zachs, "sum + CMP + N(~sum + nonzero + nodecovar(center = TRUE) + transitiveweights, ~.NetworkID)",
                      response = "contexts")
)

# Names of the curved parameters, from MPLE fits.
param_names <- list(
  curved_L = names(coef(ergm(flo ~ L(~edges, ~m) + L(~gwesp, ~m | b), estimate = "MPLE"))),
  curved_gwespL = names(coef(ergm(flo ~ L(~edges, ~m) + gwespL(L.base = ~m, Ls.path = ~b), estimate = "MPLE")))
)

# R's design matrices, for formulas with interactions.
frame <- data.frame(n = c(3, 4, 2, 5, 3, 6), w = c(TRUE, FALSE, TRUE, TRUE, FALSE, FALSE),
                    g = c("a", "b", "c", "a", "b", "c"), x = c(0.5, 1.5, -1, 2, 0, 1))
designs <- lapply(c("~n:x", "~n*x", "~g:n", "~g*n", "~w*g", "~w:g", "~0 + g:n", "~0 + g*w", "~n + g:w",
                    "~x + n:g + w", "~(n + x):g", "~n*x*w", "~log(n):factor(g)", "~I(n > 3):x"),
                  function(f) {
                    m <- model.matrix(as.formula(f), frame)
                    list(formula = f, names = colnames(m), matrix = unname(m))
                  })

# A layer that is independent of faux.mesa.high: each of its dyads tied with
# probability 0.02.
set.seed(2026)
mesa_random <- network::network(matrix(rbinom(205^2, 1, 0.02), 205, 205) * upper.tri(diag(205)), directed = FALSE)
mesa_layers <- Layer(friends = faux.mesa.high, random = mesa_random)

fits <- list(
  lm_interactions = list(network = "households", formula = "N(~edges, ~n * weekday)", estimate = "MLE"),
  curved_L = list(network = "mesa_layers",
                  formula = "L(~edges + nodematch('Grade') + nodematch('Race') + gwesp(0.5), ~friends) + L(~edges, ~random)",
                  estimate = "MLE", init = "CD"),
  projection = list(network = "davis", formula = "edges + Proj1(~nonzero)", estimate = "MLE"),
  valued_N = list(network = "zachs", formula = "N(~sum + nonzero, ~.NetworkID)", estimate = "MLE",
                  response = "contexts", reference = "Poisson")
)
results <- list()
for (name in names(fits)) {
  m <- fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  control <- if (identical(m$init, "CD")) control.ergm(seed = 1, init.method = "CD") else control.ergm(seed = 1)
  args <- list(f, estimate = m$estimate, control = control)
  if (!is.null(m$response)) args <- c(args, list(response = m$response, reference = ~Poisson))
  seconds <- system.time(fit <- do.call(ergm, args))[["elapsed"]]
  results[[name]] <- c(m, list(names = names(coef(fit)), coef = unname(coef(fit)), se = unname(sqrt(diag(vcov(fit)))),
                              seconds = seconds))
  message(sprintf("%-18s %6.1f s  %s", name, seconds, paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
}

jsonlite::write_json(list(versions = list(ergm = as.character(packageVersion("ergm")),
                                          ergm.multi = as.character(packageVersion("ergm.multi")),
                                          ergm.count = as.character(packageVersion("ergm.count"))),
                          stats = stats, param_names = param_names, frame = frame, designs = designs,
                          mesa_random = network::as.edgelist(mesa_random), fits = results),
                     file.path("tests", "data", "r_gaps_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
