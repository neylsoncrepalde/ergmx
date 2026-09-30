# Goodness-of-fit reference results from R's ergm, used by tests/test_gof.py.
#
# For a directed and an undirected model, at R's MLE: the observed
# distributions, R's simulated values and its Monte Carlo p-values. The
# simulated networks are 100,000 proposals apart, so nearly independent.
# Run from the root of the ergmx repository after scripts/r_reference.R:
#   Rscript scripts/r_gof_reference.R

suppressMessages(library(ergm))

reference <- jsonlite::read_json(file.path("tests", "data", "r_reference.json"))$models
data(samplk)
data(faux.mesa.high)
networks <- list(samplk3 = samplk3, faux.mesa.high = faux.mesa.high)

# ergm's names for the statistics, and ergmx's.
parts <- c(deg = "degree", ideg = "idegree", odeg = "odegree", espart = "espartners",
           dist = "distance", model = "model")

results <- list()
for (name in c("samplk_mutual", "mesa_gwesp")) {
  m <- reference[[name]]
  net <- networks[[m$network]]
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  coef <- unlist(m$mle)
  control <- control.gof.formula(nsim = 100, MCMC.burnin = 1e6, MCMC.interval = 1e5, seed = 1)
  gf <- gof(f, coef = coef, control = control)
  tables <- list()
  for (part in names(parts)) {
    if (is.null(gf[[paste0("obs.", part)]])) next
    tables[[parts[[part]]]] <- list(
      observed = as.numeric(gf[[paste0("obs.", part)]]),
      simulated = unname(split(gf[[paste0("sim.", part)]], row(gf[[paste0("sim.", part)]]))),
      pvalue = as.numeric(gf[[paste0("pval.", part)]][, "MC p-value"])
    )
  }
  results[[name]] <- list(network = m$network, formula = m$formula, coef = as.list(coef),
                          tables = tables)
  cat(name, ":", paste(names(tables), collapse = ", "), "\n")
}

jsonlite::write_json(results, file.path("tests", "data", "r_gof_reference.json"),
                     auto_unbox = TRUE, digits = NA)
