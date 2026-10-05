# Reference results from R's bigergm 1.2.6 for tests/test_bigergm.py: the
# MM algorithm's lower bounds, posterior membership probabilities and blocks
# from given starting blocks (with and without covariates, undirected and
# directed), the between- and within-block estimates from given blocks, and
# the means of statistics of networks simulated from a fit's within-block
# model, by ergm. bigergm's toyNet
# (200 vertices in 4 blocks, with covariates x and y) is exported along.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_bigergm_reference.R

suppressMessages({
  library(bigergm)
  library(ergm)
})
data(toyNet)
data(faux.dixon.high)

# Starting blocks: the true ones (toyNet) or the grades (faux.dixon.high), with
# a fifth of the vertices moved to a random block.
perturb <- function(blocks, k) {
  set.seed(1)
  moved <- sample(length(blocks), length(blocks) %/% 5)
  blocks[moved] <- sample(seq_len(k), length(moved), replace = TRUE)
  blocks
}
toy_start <- perturb(toyNet %v% "block", 4)
dixon_start <- perturb(as.integer(factor(faux.dixon.high %v% "grade")), 6)

mm_of <- function(formula, start, k, features) {
  fit <- bigergm(formula, n_blocks = k, initialization = start, clustering_with_features = features,
                 estimate_parameters = FALSE, n_MM_step_max = 100, tol_MM_step = 1e-4)
  list(lower_bound = fit$MM_lower_bound, alpha = unname(as.matrix(fit$alpha)), block = as.integer(fit$block),
       iterations = fit$counter_e_step)
}
mm <- list(
  toy_plain = c(list(network = "toyNet", formula = "edges + triangle", features = FALSE),
                mm_of(toyNet ~ edges + triangle, toy_start, 4, FALSE)),
  toy_features = c(list(network = "toyNet", formula = "edges + nodematch('x') + nodematch('y') + triangle", features = TRUE),
                   mm_of(toyNet ~ edges + nodematch("x") + nodematch("y") + triangle, toy_start, 4, TRUE)),
  dixon_plain = c(list(network = "faux.dixon.high", formula = "edges + mutual", features = FALSE),
                  mm_of(faux.dixon.high ~ edges + mutual, dixon_start, 6, FALSE)),
  dixon_features = c(list(network = "faux.dixon.high", formula = "edges + mutual + nodematch('sex')", features = TRUE),
                     mm_of(faux.dixon.high ~ edges + mutual + nodematch("sex"), dixon_start, 6, TRUE))
)

# Estimates from the true blocks.
fit_of <- function(formula, ...) {
  fit <- bigergm(formula, blocks = toyNet %v% "block", ...)
  within <- fit$est_within
  list(between_names = names(coef(fit$est_between)), between = unname(coef(fit$est_between)),
       between_se = unname(sqrt(diag(fit$est_between$covar))),
       within_names = names(coef(within)), within = unname(coef(within)), within_se = unname(sqrt(diag(vcov(within)))),
       object = fit)
}
toy_formula <- toyNet ~ edges + nodematch("x") + nodematch("y") + triangle
fits <- list(
  mple = fit_of(toy_formula),
  intercepts = fit_of(toy_formula, add_intercepts = TRUE),
  intercepts_plain = fit_of(toy_formula, add_intercepts = TRUE, clustering_with_features = FALSE),
  # With triangle, R's within-block Monte Carlo MLE doesn't mix.
  mle = fit_of(toyNet ~ edges + nodematch("x") + nodematch("y") + gwesp(0.5, fixed = TRUE), method_within = "MLE",
               control_within = control.ergm(seed = 1))
)

# The MPLE fit's within-block model, simulated by ergm, and its between-block
# model's expected ties. bigergm's simulate() doesn't reproduce either: on
# this fit, its networks average 262 ties between blocks (the model's
# expectation is 254) and 1149 within (ergm's simulation of the within-block
# model, 1158).
b <- toyNet %v% "block"
el <- network::as.edgelist(toyNet)
within_net <- network::network.initialize(network.size(toyNet), directed = FALSE)
within_net[el[b[el[, 1]] == b[el[, 2]], ]] <- 1
for (a in c("block", "x", "y")) within_net %v% a <- toyNet %v% a
within_sims <- simulate(within_net ~ edges + nodematch("x") + nodematch("y") + triangle,
                        coef = coef(fits$mple$object$est_within), constraints = ~blockdiag("block"), nsim = 1000,
                        output = "stats", seed = 1,
                        control = control.simulate.formula(MCMC.burnin = 200000, MCMC.interval = 20000))
between_coef <- coef(fits$mple$object$est_between)
same <- function(v) outer(v, v, "==")
u <- between_coef[1] + between_coef[2] * same(toyNet %v% "x") + between_coef[3] * same(toyNet %v% "y")
expected_between <- sum(plogis(u[outer(b, b, "!=") & upper.tri(u)]))
fits <- lapply(fits, function(f) f[names(f) != "object"])

jsonlite::write_json(list(
  bigergm_version = as.character(packageVersion("bigergm")),
  toyNet = list(n = network.size(toyNet), edges = network::as.edgelist(toyNet), block = toyNet %v% "block",
                x = toyNet %v% "x", y = toyNet %v% "y"),
  toy_start = toy_start, dixon_start = dixon_start, mm = mm, fits = fits,
  within_simulated = list(names = colnames(within_sims), mean = unname(colMeans(within_sims)),
                          sd = unname(apply(within_sims, 2, sd))),
  expected_between = expected_between),
  file.path("tests", "data", "r_bigergm_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
