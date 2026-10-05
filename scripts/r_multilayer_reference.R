# Reference results from R's ergm.multi for multilayer networks (Layer()),
# used by tests/test_multilayer.py: statistics of L() with Layer Logic, and
# of the layer-aware terms, on the Florentine families' marriages and
# business ties (undirected) and Sampson's monks' liking at three times
# (directed), and fits.
#
# With L.in_order = TRUE, ergm.multi 0.3.0's OSP and ISP shared partners
# (d7, d9, d10) depend on the order the ties are added in: its cache keys
# them by unordered pairs. The tests compare those with the documented
# definition (the first tie at the pair's first vertex) instead.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_multilayer_reference.R

suppressMessages({
  library(ergm.multi)
})
data(florentine)
data(samplk)
flo <- Layer(m = flomarriage, b = flobusiness)
monks <- Layer(samplk1, samplk2, samplk3)

stats_of <- function(net, formula) {
  s <- summary(as.formula(paste("net ~", formula)))
  list(formula = formula, names = names(s), values = unname(as.numeric(s)))
}
undirected <- c(
  "edges + triangle",
  "L(~edges, ~m) + L(~edges, ~b) + L(~edges, ~m & b) + L(~edges, ~m | b) + L(~edges, ~m & !b) + L(~edges, ~xor(m, b))",
  "L(~edges + triangle + kstar(2), ~`1`) + L(~triangle, ~m | b) + L(~edges, c(~m, ~b)) + L(~edges, c(3 ~ m, -1 ~ (b & !m)))",
  "L(~edges, ~(m + b) == 2) + L(~edges, ~m != b) + L(~edges, ~(2 * m - b) > 0) + L(~gwesp(0.5, fixed = TRUE), ~m | b)",
  "L(~nodematch('priorates') + nodecov('wealth'), ~m)",
  "CMBL + CMBL(c(~m, ~b))",
  "twostarL(c(~m, ~b)) + twostarL(c(~m, ~b), distinct = FALSE) + twostarL(c(~m, ~m))",
  "despL(0:2, L.base = ~m, Ls.path = c(~b, ~b)) + despL(1, L.base = ~m, Ls.path = c(~m, ~b), L.in_order = TRUE)",
  "dgwespL(0.5, fixed = TRUE, L.base = ~m, Ls.path = c(~b, ~b)) + ddspL(1:2, Ls.path = c(~m, ~b))",
  "dgwdspL(0.5, fixed = TRUE, Ls.path = c(~m, ~m)) + dnspL(1, L.base = ~m, Ls.path = c(~b, ~b)) + dgwnspL(0.25, fixed = TRUE, L.base = ~b, Ls.path = c(~m, ~b))"
)
directed <- c(
  "edges + mutual",
  "L(~edges, ~`1` & `2`) + L(~edges, ~`1` & !`3`) + L(~mutual, ~`3`) + L(~edges, ~t(`1`) & `2`) + L(~edges, ~(`1` + `2` + `3`) >= 2)",
  "mutualL(Ls = c(~`3`, ~`3`)) + mutualL(Ls = c(~`1`, ~`2`)) + mutualL(same = 'group', Ls = c(~`2`, ~`3`)) + mutualL(same = 'group', diff = TRUE, Ls = c(~`1`, ~`3`))",
  "twostarL(c(~`1`, ~`2`), 'out') + twostarL(c(~`1`, ~`2`), 'in') + twostarL(c(~`1`, ~`2`), 'path') + twostarL(c(~`1`, ~`3`), 'path', distinct = FALSE)",
  "CMBL + CMBL(c(~`1`, ~`3`))",
  "despL(0:2, 'OTP', L.base = ~`3`, Ls.path = c(~`2`, ~`2`)) + despL(1, 'ITP', L.base = ~`3`, Ls.path = c(~`1`, ~`2`), L.in_order = TRUE)",
  "despL(0:1, 'OSP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`)) + despL(0:1, 'ISP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`), L.in_order = TRUE)",
  "dgwespL(0.5, fixed = TRUE, type = 'OTP', L.base = ~`3`, Ls.path = c(~`1`, ~`2`)) + ddspL(1:2, 'OTP', Ls.path = c(~`1`, ~`3`)) + dgwnspL(0.5, fixed = TRUE, type = 'OTP', L.base = ~`1`, Ls.path = c(~`2`, ~`3`))",
  "ddspL(0:2, 'OSP', Ls.path = c(~`1`, ~`3`)) + ddspL(0:2, 'ISP', Ls.path = c(~`1`, ~`3`), L.in_order = TRUE) + dnspL(0:2, 'OSP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`), L.in_order = TRUE) + dnspL(0:2, 'ISP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`))",
  "dgwnspL(0.5, fixed = TRUE, type = 'ISP', L.base = ~`2`, Ls.path = c(~`1`, ~`3`), L.in_order = TRUE) + dgwespL(0.5, fixed = TRUE, type = 'OSP', L.base = ~`1`, Ls.path = c(~`2`, ~`3`), L.in_order = TRUE) + dgwdspL(0.5, fixed = TRUE, type = 'ISP', Ls.path = c(~`1`, ~`2`))"
)
stats <- c(lapply(undirected, function(f) stats_of(flo, f)), lapply(directed, function(f) stats_of(monks, f)))
names(stats) <- c(paste0("u", seq_along(undirected)), paste0("d", seq_along(directed)))
for (s in stats) message(s$formula, " | ", paste(s$names, collapse = ", "), " | ", paste(round(s$values, 4), collapse = " "))

fits <- list(
  flo_layers = list(network = "flo", formula = "L(~edges, ~m) + L(~edges, ~b) + L(~edges, ~m & b)"),
  flo_cmb = list(network = "flo", formula = "L(~edges, ~m) + L(~edges, ~b) + CMBL"),
  monks_layers = list(network = "monks", formula = "edges + L(~edges, ~`2` & `1`) + L(~edges, ~`3` & `2`) + mutualL(Ls = c(~`3`, ~`3`))"),
  monks_twostar = list(network = "monks", formula = "edges + mutual + L(~edges, ~`3` & `2`) + twostarL(c(~`2`, ~`3`), 'out')")
)
results <- list()
for (name in names(fits)) {
  m <- fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  seconds <- system.time(fit <- ergm(f, control = control.ergm(seed = 1)))[["elapsed"]]
  results[[name]] <- c(m, list(names = names(coef(fit)), coef = unname(coef(fit)), se = unname(sqrt(diag(vcov(fit)))),
                              seconds = seconds))
  message(sprintf("%-14s %5.1f s  %s", name, seconds, paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
}
jsonlite::write_json(list(ergm_multi_version = as.character(packageVersion("ergm.multi")), stats = stats, fits = results),
                     file.path("tests", "data", "r_multilayer_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
