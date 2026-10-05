# Reference results from R's tergm for tie-age (durational) terms, used by
# tests/test_durational.py.
#
# 1. The statistics of tergm's durational terms on networks whose ties have
#    known ages (tergm's `time` and `lasttoggle` network attributes: a tie's
#    age is time + 1 minus the time it formed), undirected and directed.
# 2. Dynamic simulations from an empty network, with durational terms in the
#    model, and the time averages of monitored statistics after a burn-in.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_durational_reference.R

suppressMessages({
  library(tergm)
})

out_file <- file.path("tests", "data", "r_durational_reference.json")

# -- 1. Statistics of networks with tie ages ---------------------------------------------------

set.seed(11)
aged <- function(n, directed, m, time) {
  pairs <- if (directed) which(row(diag(n)) != col(diag(n)), arr.ind = TRUE) else
    which(upper.tri(diag(n)), arr.ind = TRUE)
  pick <- pairs[sample(nrow(pairs), m), , drop = FALSE]
  nw <- network.initialize(n, directed = directed)
  add.edges(nw, pick[, 1], pick[, 2])
  nw %v% "sex" <- sample(c("F", "M"), n, replace = TRUE)
  nw %v% "grade" <- sample(7:9, n, replace = TRUE)
  formed <- sample(0:time, m, replace = TRUE)
  nw %n% "time" <- time
  nw %n% "lasttoggle" <- cbind(pick[, 1], pick[, 2], formed)
  weights <- matrix(round(runif(n * n, 0, 2), 2), n, n)
  if (!directed) weights[lower.tri(weights)] <- t(weights)[lower.tri(weights)]
  diag(weights) <- 0
  nw %n% "w" <- weights
  list(nw = nw, edges = pick - 1, ages = time + 1 - formed, sex = nw %v% "sex", grade = nw %v% "grade",
       w = weights)
}

common <- paste(
  "edge.ages + mean.age + mean.age(log = TRUE) + mean.age(emptyval = 3) +",
  "edges.ageinterval(c(1, 3, 6), c(3, 6, Inf)) + edgecov.ages('w') + edgecov.mean.age('w') +",
  "edgecov.mean.age('w', log = TRUE) + nodefactor.mean.age('sex') + nodefactor.mean.age('grade', levels = -1) +",
  "nodefactor.mean.age('sex', log = TRUE) + nodemix.mean.age('sex') + nodemix.mean.age('grade', levels2 = -1) +",
  "EdgeAges(~edges + nodematch('sex') + edgecov('w'))")
undirected_only <- paste(
  "+ degree.mean.age(1:4) + degree.mean.age(c(1, 2), byarg = 'sex') + degrange.mean.age(1, 3) +",
  "degrange.mean.age(c(1, 3), c(3, Inf), byarg = 'sex', emptyval = 2)")

statistics <- list()
for (kind in c("undirected", "directed")) {
  directed <- kind == "directed"
  a <- aged(if (directed) 15 else 20, directed, if (directed) 45 else 40, 10)
  nw <- a$nw
  formula <- paste(common, if (!directed) undirected_only else "")
  s <- summary(as.formula(paste("nw ~", formula)))
  statistics[[kind]] <- list(directed = directed, n = network.size(nw), edges = a$edges, ages = a$ages,
                             sex = a$sex, grade = a$grade, w = a$w, formula = formula,
                             names = names(s), values = unname(as.numeric(s)))
  message(kind, ": ", length(s), " statistics")
}

# -- 2. Dynamic simulations ---------------------------------------------------------------------

monitor <- ~edges + edges.ageinterval(c(1, 3, 7), c(3, 7, Inf)) + mean.age + degree.mean.age(1:2)
simulations <- list(
  persist_interval = list(n = 40, formula = "Form(~edges) + Persist(~edges + edges.ageinterval(3, 7))",
                          coef = c(-4, 2, -1)),
  persist_edge_ages = list(n = 40, formula = "Form(~edges) + Persist(~edges + edge.ages)",
                           coef = c(-4, 2, -0.2)),
  dissolve_interval = list(n = 40, formula = "Form(~edges) + Diss(~edges + edges.ageinterval(2))",
                           coef = c(-4, -1.5, 1))
)
slices <- 4000
burnin <- 500
results <- list()
for (name in names(simulations)) {
  m <- simulations[[name]]
  nw <- network.initialize(m$n, directed = FALSE)
  seconds <- system.time(s <- simulate(as.formula(paste("nw ~", m$formula)), coef = m$coef, time.slices = slices,
                                       dynamic = TRUE, monitor = monitor, output = "stats", seed = 1,
                                       stats = FALSE))[["elapsed"]]
  s <- as.matrix(s)[-(1:burnin), , drop = FALSE]
  # Batch means: 35 batches of 100 time steps.
  batches <- apply(s, 2, function(x) tapply(x, rep(seq_len(nrow(s) / 100), each = 100), mean))
  results[[name]] <- c(m, list(slices = slices, burnin = burnin, names = colnames(s),
                               mean = unname(colMeans(s)), se = unname(apply(batches, 2, sd) / sqrt(nrow(batches))),
                               seconds = seconds))
  message(sprintf("%-20s %5.1f s  %s", name, seconds, paste(sprintf("%.2f", colMeans(s)), collapse = " ")))
}

jsonlite::write_json(list(tergm_version = as.character(packageVersion("tergm")), statistics = statistics,
                          monitor = deparse(monitor), simulations = results),
                     out_file, auto_unbox = TRUE, digits = NA, pretty = TRUE)
