# Reference results from R's ergm.count, used by tests/test_valued.py.
#
# Valued statistics of ergm.count's karate club (zach, counts of contexts)
# and of Sampson's monks' liking summed over the three times (counts of 0 to
# 3, a directed network), and fits of valued models. Writes
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

fits <- list(
  zach_sum = list(network = "zach", reference = "Poisson", formula = "sum"),
  zach_dyadind = list(network = "zach", reference = "Poisson",
                      formula = "sum + nonzero + nodefactor('role', levels=-2) + nodematch('faction.id')"),
  nominations_mutual = list(network = "nominations", reference = "Binomial(3)",
                            formula = "sum + nonzero + mutual"),
  zach_transitive = list(network = "zach", reference = "Poisson",
                         formula = "sum + nonzero + transitiveweights('min','max','min')")
)
out_file <- file.path("tests", "data", "r_valued_reference.json")
write_out <- function(results) {
  jsonlite::write_json(list(ergm_count_version = as.character(packageVersion("ergm.count")),
                            stats = stats, fits = results),
                       out_file, auto_unbox = TRUE, digits = NA, pretty = TRUE)
}
results <- list()
write_out(results)
for (name in names(fits)) {
  m <- fits[[name]]
  net <- if (m$network == "zach") zach else nominations
  response <- if (m$network == "zach") "contexts" else "nominations"
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  reference <- as.formula(paste("~", m$reference))
  seconds <- system.time(fit <- try(ergm(f, response = response, reference = reference,
                                         control = control.ergm(seed = 1)), silent = TRUE))[["elapsed"]]
  if (inherits(fit, "try-error")) {
    message(sprintf("%-20s failed after %.0f s: %s", name, seconds, fit))
    next
  }
  message(sprintf("%-20s %6.1f s  %s", name, seconds,
                  paste(sprintf("%s=%.4f", names(coef(fit)), coef(fit)), collapse = " ")))
  results[[name]] <- list(network = m$network, reference = m$reference, formula = m$formula,
                          coef = as.list(coef(fit)), se = as.list(sqrt(diag(vcov(fit)))), seconds = seconds)
  write_out(results)
}
