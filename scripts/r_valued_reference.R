# Reference results from R's ergm.count, used by tests/test_valued.py.
#
# Valued statistics of ergm.count's karate club (zach, counts of contexts)
# and of Sampson's monks' liking summed over the three times (counts of 0 to
# 3, a directed network), goodness of fit, and fits of valued models,
# including continuous values (StdNormal and Unif references) of networks
# simulated here, whose values the JSON keeps. Writes
# tests/data/r_valued_reference.json after each fit, since R's valued fits
# are slow.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_valued_reference.R

suppressMessages({
  library(ergm.count)
})

data(zach)
data(samplk)
y <- as.matrix(samplk1) + as.matrix(samplk2) + as.matrix(samplk3)
nominations <- network(y, directed = TRUE, ignore.eval = FALSE, names.eval = "nominations")
nominations %v% "group" <- samplk1 %v% "group"

undirected <- paste(
  "sum + nonzero + nodematch('role') + nodematch('role', form='nonzero') + nodefactor('role') +",
  "nodecov('faction.id') + absdiff('faction.id') + atleast(2) + atmost(c(1,3)) + greaterthan(4) +",
  "smallerthan(2) + equalto(3) + ininterval(1, 3) + ininterval(1,3, open=c(FALSE, TRUE)) +",
  "sum(pow=0.5) + CMP + transitiveweights('min','max','min') +",
  "transitiveweights('geomean','sum','geomean') + cyclicalweights + nodecovar +",
  "nodecovar(center=TRUE, transform='sqrt') + mm('role')")
directed <- paste(
  "sum + nonzero + mutual + mutual(form='nabsdiff') + mutual(form='product') + mutual(form='geometric') +",
  "transitiveweights + cyclicalweights + transitiveweights('geomean','sum','geomean') +",
  "cyclicalweights('geomean','sum','min') + nodeocovar + nodeicovar + nodeocovar(center=TRUE) +",
  "nodeicovar(transform='sqrt') + transitiveties + transitiveties(threshold=1) +",
  "nodematch('group', form='sum') + sender(form='nonzero')")
stats_of <- function(net, formula, response) {
  s <- summary(as.formula(paste("net ~", formula)), response = response)
  list(names = names(s), values = unname(as.numeric(s)))
}
stats <- list(zach = stats_of(zach, undirected, "contexts"),
              nominations = stats_of(nominations, directed, "nominations"))

# The points of gof()'s cdf: of the counts, and of fractional values (with
# fewer points allowed).
fractional <- zach
fractional %e% "scaled" <- (zach %e% "contexts") * 0.35
cdf_points <- list(
  zach = names(summary(zach ~ cdf, response = "contexts")),
  nominations = names(summary(nominations ~ cdf, response = "nominations")),
  fractional = names(summary(fractional ~ cdf, response = "scaled")),
  fractional_nmax = names(summary(fractional ~ cdf(nmax = 5), response = "scaled"))
)

# gof() of a dyad-independent Poisson model, away from its MLE, with nearly
# independent simulated networks.
gof_formula <- "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')"
gof_coef <- c(0.9734, -2.6366, 0.1525, 0.6993, 0.3955)
set.seed(1)
g <- gof(as.formula(paste("zach ~", gof_formula)), coef = gof_coef, response = "contexts",
         reference = ~Poisson,
         control = control.gof.formula(nsim = 1000, MCMC.interval = 20000, MCMC.burnin = 100000))
gof_of <- function(name) list(names = rownames(g[[paste0("summary.", name)]]),
                              obs = unname(g[[paste0("obs.", name)]]),
                              mean = unname(colMeans(g[[paste0("sim.", name)]])),
                              sd = unname(apply(g[[paste0("sim.", name)]], 2, sd)))
gof_reference <- list(formula = gof_formula, coef = gof_coef, nsim = 1000, cdf_points = cdf_points,
                      cdf = gof_of("cdf"), model = gof_of("model"))

# Continuous values: normal ones, higher within two groups (undirected) or
# correlated within pairs (directed), and proportions, higher within groups.
set.seed(42)
group <- rep(c("a", "b"), each = 10)
same <- outer(group, group, "==")
normal <- matrix(rnorm(400, mean = 0.3 + 0.8 * same), 20, 20)
normal[lower.tri(normal)] <- t(normal)[lower.tri(normal)]
a <- matrix(rnorm(225), 15, 15)
b <- matrix(rnorm(225), 15, 15)
reciprocal <- a
reciprocal[lower.tri(reciprocal)] <- (0.4 * t(a) + sqrt(1 - 0.16) * t(b))[lower.tri(reciprocal)]
proportions <- plogis(matrix(rnorm(400, mean = -0.5 + same), 20, 20))
proportions[lower.tri(proportions)] <- t(proportions)[lower.tri(proportions)]
diag(normal) <- diag(reciprocal) <- diag(proportions) <- 0
continuous <- list(group = group, normal = normal, reciprocal = reciprocal, proportions = proportions)
as_valued <- function(y, directed) {
  net <- network(y, directed = directed, ignore.eval = FALSE, names.eval = "value")
  if (nrow(y) == 20) net %v% "group" <- group
  net
}
networks <- list(
  zach = list(net = zach, response = "contexts"),
  nominations = list(net = nominations, response = "nominations"),
  normal = list(net = as_valued(normal, FALSE), response = "value"),
  reciprocal = list(net = as_valued(reciprocal, TRUE), response = "value"),
  proportions = list(net = as_valued(proportions, FALSE), response = "value"),
  empty = list(net = network.copy(zach), response = "contexts")
)
delete.edges(networks$empty$net, seq_along(networks$empty$net$mel))

fits <- list(
  zach_sum = list(network = "zach", reference = "Poisson", formula = "sum"),
  zach_dyadind = list(network = "zach", reference = "Poisson",
                      formula = "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')"),
  nominations_mutual = list(network = "nominations", reference = "Binomial(3)",
                            formula = "sum + nonzero + mutual"),
  zach_transitive = list(network = "zach", reference = "Poisson",
                         formula = "sum + nonzero + transitiveweights('min','max','min')"),
  normal_group = list(network = "normal", reference = "StdNormal",
                      formula = "sum + sum(pow=2) + nodematch('group', form='sum')"),
  reciprocal_product = list(network = "reciprocal", reference = "StdNormal",
                            formula = "sum + sum(pow=2) + mutual(form='product')"),
  proportions_group = list(network = "proportions", reference = "Unif(0,1)",
                           formula = "sum + nodematch('group', form='sum')"),
  zach_target = list(network = "empty", reference = "Poisson", target = "zach",
                     formula = "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')")
)
out_file <- file.path("tests", "data", "r_valued_reference.json")
write_out <- function(results) {
  jsonlite::write_json(list(ergm_count_version = as.character(packageVersion("ergm.count")),
                            stats = stats, gof = gof_reference, continuous = continuous, fits = results),
                       out_file, auto_unbox = TRUE, digits = NA, pretty = TRUE)
}
results <- list()
write_out(results)
for (name in names(fits)) {
  m <- fits[[name]]
  net <- networks[[m$network]]$net
  response <- networks[[m$network]]$response
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  reference <- as.formula(paste("~", m$reference))
  # Fits to target statistics: those of another network, from this one.
  target <- if (is.null(m$target)) NULL else
    summary(as.formula(paste("networks[[m$target]]$net ~", m$formula)), response = networks[[m$target]]$response)
  seconds <- system.time(fit <- try(ergm(f, response = response, reference = reference, target.stats = target,
                                         control = control.ergm(seed = 1)), silent = TRUE))[["elapsed"]]
  if (inherits(fit, "try-error")) {
    message(sprintf("%-20s failed after %.0f s: %s", name, seconds, fit))
    next
  }
  message(sprintf("%-20s %6.1f s  %s", name, seconds,
                  paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
  results[[name]] <- list(network = m$network, reference = m$reference, formula = m$formula, target = m$target,
                          coef = as.list(coef(fit)), se = as.list(sqrt(diag(vcov(fit)))), seconds = seconds)
  write_out(results)
}
