# Reference results from R's ergm for its operators (Sum, Prod, Log, Exp,
# Symmetrize, Label, Passthrough, I, For, Offset, Curve), used by
# tests/test_operators.py: statistics and names on networks, and fits.
#
# ergm 4.12's Monte Carlo MLE of edges + Symmetrize(~edges) ends far from
# its exact MLE (the reciprocity model reparametrized); the test uses the
# exact one.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_operators_reference.R

suppressMessages(library(ergm))
data(florentine)
data(samplk)
data(faux.mesa.high)

stats_of <- function(net, formula) {
  s <- summary(as.formula(paste("net ~", formula)))
  list(formula = formula, names = names(s), values = unname(as.numeric(s)))
}
undirected <- list(
  "Sum(~edges + triangle, 'x')",
  "Sum(list(~edges, ~triangle), 'y')",
  "Sum(list(2 ~ kstar(1:2), c(1, -1) ~ degree(1:2)), 'w')",
  "Sum(list('sum' ~ degree(1:3)), 'total')",
  "Sum(list('mean' ~ degree(1:3)), I('average'))",
  "Sum(list(matrix(c(1, 2, 3, 4, 5, 6), 2, 3) ~ degree(1:3)), 'm')",
  "Log(~edges + triangle + degree(0))",
  "Log(~degree(0), log0 = -5)",
  "Exp(~triangle + isolates)",
  "Prod(list(~edges, ~triangle), 'p')",
  "Prod(list(2 ~ edges), 'square')",
  "Label(~edges + triangle, 'x.')",
  "Label(~edges + triangle, c('a', 'b'), pos = 'replace')",
  "Label(~edges, '.z', pos = 'append')",
  "Label(~edges + kstar(2), 'L')",
  "Passthrough(~edges + triangle)",
  "I(~edges + triangle)",
  "For(~nodecov(a), a = c('wealth', 'priorates'))",
  "For(~degree(d), d = 1:3)"
)
directed <- list(
  "Symmetrize(~edges + triangle)",
  "Symmetrize(~edges + kstar(2), rule = 'strong')",
  "Symmetrize(~edges + triangle, 'upper')",
  "Symmetrize(~edges + triangle, rule = 'lower')",
  "Symmetrize(~edges + nodematch('group'), 'weak')",
  "Sum(~edges + mutual, 'z')",
  "Log(~mutual + ttriple)"
)
stats <- c(lapply(undirected, function(f) stats_of(flomarriage, f)),
           lapply(directed, function(f) stats_of(samplk3, f)))
names(stats) <- c(paste0("u", seq_along(undirected)), paste0("d", seq_along(directed)))

fits <- list(
  sum_mple = list(network = "flomarriage", formula = "edges + Sum(list(c(1, 2) ~ kstar(2:3)), 'ks')", estimate = "MPLE"),
  log_mle = list(network = "flomarriage", formula = "edges + Log(~kstar(2))", estimate = "MLE"),
  symmetrize_mle = list(network = "samplk3", formula = "edges + Symmetrize(~edges)", estimate = "MLE"),
  offset_mle = list(network = "faux.mesa.high", formula = "edges + Offset(~nodematch('Grade') + nodematch('Sex'), 1, 2)",
                    estimate = "MLE"),
  curve_mle = list(network = "faux.mesa.high", formula = "Curve(~edges + nodematch('Grade') + nodematch('Race'), list(a = 0, b = 0), map = function(x, n, ...) c(x[1], x[2], x[2]), gradient = function(x, n, ...) rbind(c(1, 0, 0), c(0, 1, 1)))",
                   estimate = "MLE")
)
results <- list()
for (name in names(fits)) {
  m <- fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  fit <- ergm(f, estimate = m$estimate, control = control.ergm(seed = 1), verbose = FALSE)
  results[[name]] <- c(m, list(names = names(coef(fit)), coef = unname(coef(fit)),
                              se = unname(sqrt(diag(vcov(fit))))))
  message(sprintf("%-16s %s", name, paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
}
jsonlite::write_json(list(ergm_version = as.character(packageVersion("ergm")), stats = stats, fits = results),
                     file.path("tests", "data", "r_operators_reference.json"), auto_unbox = TRUE, digits = NA,
                     pretty = TRUE)
