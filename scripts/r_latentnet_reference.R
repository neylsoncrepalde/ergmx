# Reference results from R's latentnet 2.12.0 for tests/test_latent.py:
# latent space models (ergmm) of Sampson's monks (samplike), the Gahuku-Gama
# tribes and Davis's Southern Women, with each family, latent effect,
# clusters, random effects and a covariate. For each model:
#   - the model's coefficient names and priors;
#   - the starting values, and a posterior draw with its log-densities (to
#     check the likelihood and priors exactly);
#   - with seeds 1 to 3, posterior summaries invariant to rotations and
#     relabelling: the coefficients' means, standard deviations and
#     quantiles, the latent and random effects' variances, the posterior mean
#     distances and tie probabilities (computed here from the draws: R's
#     predict(type = "post") transposes the covariates of directed
#     networks), the MKL coefficients, the clusters' co-membership
#     probabilities, and the BIC.
#
# Run from the root of the ergmx repository:
#   Rscript scripts/r_latentnet_reference.R

suppressMessages({
  library(latentnet)
  library(igraph, warn.conflicts = FALSE)
})
data(sampson)
data(tribes)
davis_graph <- read_graph(gzcon(file(file.path("python", "ergmx", "data", "davis.graphml.gz"), "rb")),
                          format = "graphml")
davis <- network::network(as.matrix(as_adjacency_matrix(davis_graph))[1:18, 19:32], bipartite = 18,
                          directed = FALSE)

fits <- list(
  sampson_e2 = list(network = "samplike", formula = "euclidean(d = 2)"),
  sampson_e2g3 = list(network = "samplike", formula = "euclidean(d = 2, G = 3)"),
  sampson_rreceiver = list(network = "samplike", formula = "euclidean(d = 2) + rreceiver"),
  sampson_sender_receiver = list(network = "samplike", formula = "euclidean(d = 2) + rsender + rreceiver"),
  sampson_bilinear = list(network = "samplike", formula = "bilinear(d = 2)"),
  sampson_euclidean2 = list(network = "samplike", formula = "euclidean2(d = 2)"),
  sampson_nodematch = list(network = "samplike", formula = "euclidean(d = 2) + nodematch('group')"),
  sampson_poisson = list(network = "samplike", formula = "euclidean(d = 2)", response = "nominations", family = "Poisson"),
  tribes_g3 = list(network = "tribes", formula = "euclidean(d = 2, G = 3)", response = "pos"),
  tribes_sociality = list(network = "tribes", formula = "euclidean(d = 2) + rsociality", response = "pos"),
  tribes_binomial = list(network = "tribes", formula = "euclidean(d = 2)", response = "sign.012", family = "binomial",
                         trials = 2),
  tribes_normal = list(network = "tribes", formula = "euclidean(d = 2)", response = "sign", family = "normal"),
  davis_e2 = list(network = "davis", formula = "euclidean(d = 2)")
)

configuration <- function(par) {
  keep <- c("beta", "Z", "Z.K", "Z.mean", "Z.var", "Z.pK", "sender", "receiver", "sociality", "sender.var",
            "receiver.var", "sociality.var", "dispersion")
  out <- lapply(par[intersect(keep, names(par))], function(x) if (is.matrix(x)) unname(x) else unname(x))
  out
}

summaries <- function(fit) {
  S <- fit$sample
  n <- network.size(fit$model$Yg)
  draws <- length(S$lpY)
  out <- list()
  if (fit$model$p > 0) {
    out$beta_mean <- unname(colMeans(S$beta))
    out$beta_sd <- unname(apply(S$beta, 2, sd))
    out$beta_q <- unname(apply(S$beta, 2, quantile, c(0.025, 0.975)))
    out$mkl_beta <- unname(fit$mkl$beta)
  }
  if (fit$model$d > 0) {
    out$Z_var_mean <- unname(colMeans(as.matrix(S$Z.var)))
    distance <- matrix(0, n, n)
    for (s in seq_len(draws)) distance <- distance + as.matrix(dist(S$Z[s, , ]))
    out$distance_mean <- unname(distance / draws)
    out$mkl_distance <- unname(as.matrix(dist(fit$mkl$Z)))
  }
  for (v in c("sender.var", "receiver.var", "sociality.var", "dispersion"))
    if (length(S[[v]])) out[[sub(".", "_", v, fixed = TRUE)]] <- mean(S[[v]])
  if (fit$model$G > 1) {
    # The probability that two vertices are in the same cluster: invariant to relabelling.
    same <- matrix(0, n, n)
    for (s in seq_len(draws)) same <- same + outer(S$Z.K[s, ], S$Z.K[s, ], "==")
    out$co_cluster <- unname(same / draws)
  }
  ey <- matrix(0, n, n)
  for (s in seq_len(draws)) ey <- ey + latentnet:::ergmm.EY(fit$model, S[[s]], NA.unobserved = FALSE)
  out$tie_probability <- unname(ey / draws)
  out$lpY_mean <- mean(S$lpY)
  b <- bic.ergmm(fit)
  out$bic <- lapply(b, unname)
  out
}

results <- list()
for (name in names(fits)) {
  m <- fits[[name]]
  net <- get(m$network)
  f <- as.formula(paste("net ~", m$formula))
  environment(f) <- environment()
  args <- list(f)
  if (!is.null(m$response)) args$response <- m$response
  if (!is.null(m$family)) args$family <- m$family
  if (identical(m$family, "binomial")) args$fam.par <- list(trials = m$trials)
  if (identical(m$family, "normal")) args$fam.par <- list(prior.var = 1, prior.var.df = 2)
  by_seed <- list()
  for (seed in 1:3) {
    seconds <- system.time(fit <- suppressWarnings(do.call(ergmm, c(args, list(seed = seed)))))[["elapsed"]]
    by_seed[[seed]] <- c(summaries(fit), list(seconds = seconds))
    if (seed == 1) {
      first <- fit
      lp <- sapply(c("lpY", "lpZ", "lpbeta", "lpRE", "lpLV", "lpREV", "lpdispersion"), function(v)
        if (length(fit$sample[[v]])) fit$sample[[v]][1] else 0)
    }
  }
  prior <- first$prior
  results[[name]] <- c(m, list(
    coef_names = first$model$coef.names, directed = network::is.directed(net),
    prior = lapply(prior[setdiff(names(prior), c("adjust.beta.var"))], unname),
    start = configuration(first$start),
    draw = configuration(first$sample[[1]]), draw_lp = as.list(lp),
    seeds = by_seed))
  message(sprintf("%-24s %5.1f s  beta %s", name, by_seed[[1]]$seconds,
                  paste(round(by_seed[[1]]$beta_mean, 3), collapse = " ")))
}

jsonlite::write_json(list(latentnet_version = as.character(packageVersion("latentnet")), fits = results),
                     file.path("tests", "data", "r_latentnet_reference.json"), auto_unbox = TRUE, digits = NA)
