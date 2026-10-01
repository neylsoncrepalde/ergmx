# Reference results for interpreting fitted models, used by the ergmx test
# suite: ergm's predict() (conditional tie probabilities at given
# coefficients), ergMargins' average marginal effects, confint() and texreg's
# tables of exactly fitted (dyad-independent) models, whose numbers ergmx
# reproduces, so that the tables can be compared character by character.
#
# Writes tests/data/r_interpret_reference.json. ergMargins can live in a
# separate library: R_LIBS=/tmp/rlib-margins. Run from the root of the
# ergmx repository:
#   Rscript scripts/r_interpret_reference.R

suppressMessages({
  library(ergm)
  library(ergm.multi)
  library(texreg)
  library(ergMargins)
})

data(florentine)
data(samplk)
data(faux.mesa.high)
data(Goeyvaerts)
samplk3.nonresponse <- samplk3
samplk3.nonresponse[c(1, 7), ] <- NA

predictions <- function(formula, eta) {
  p <- predict(formula, eta = eta)
  list(tail = p$tail - 1L, head = p$head - 1L, p = p$p)
}

# Conditional tie probabilities at given coefficients (canonical ones for the
# curved model), on undirected, directed, missing-dyad, curved and combined
# networks.
households <- Networks(Filter(function(g) (g %n% "included") && (g %n% "weekday"), Goeyvaerts)[1:20])
cases <- list(
  flo = list(network = "flomarriage", formula = "edges + nodecov('wealth')",
             f = flomarriage ~ edges + nodecov("wealth"), coef = c(-2.6, 0.011)),
  samplk = list(network = "samplk3", formula = "edges + mutual + gwesp(0.5, fixed=TRUE)",
                f = samplk3 ~ edges + mutual + gwesp(0.5, fixed = TRUE), coef = c(-2.3, 2.1, 0.2)),
  missing = list(network = "samplk3.nonresponse", formula = "edges + mutual",
                 f = samplk3.nonresponse ~ edges + mutual, coef = c(-2.0, 1.8)),
  curved = list(network = "faux.mesa.high", formula = "edges + nodematch('Grade') + gwesp(0.5)",
                f = faux.mesa.high ~ edges + nodematch("Grade") + gwesp(0.5, cutoff = 30),
                coef = c(-6.3, 2.0, 1.3, 0.6)),
  households = list(network = "Goeyvaerts.weekday20", formula = "N(~edges + triangle + nodematch('gender'))",
                    f = households ~ N(~edges + triangle + nodematch("gender")), coef = c(0.4, 1.2, -0.2))
)
predict_ref <- lapply(cases, function(case) {
  m <- ergm_model(case$f)
  eta <- ergm.eta(case$coef, m$etamap)
  c(list(network = case$network, formula = case$formula, coef = case$coef), predictions(case$f, eta))
})

# Exact fits of dyad-independent models: marginal effects, confidence
# intervals and tables.
f1 <- ergm(flomarriage ~ edges)
f2 <- ergm(flomarriage ~ edges + nodecov("wealth"))
f3 <- ergm(flomarriage ~ edges + nodecov("wealth") + absdiff("wealth") + edgecov(flobusiness))
f4 <- ergm(faux.mesa.high ~ edges + nodematch("Grade") + nodefactor("Sex") + absdiff("Grade"))
fits <- list(flo_null = f1, flo_wealth = f2, flo_full = f3, mesa = f4)
formulas <- list(flo_null = "edges", flo_wealth = "edges + nodecov('wealth')",
                 flo_full = "edges + nodecov('wealth') + absdiff('wealth') + edgecov('business')",
                 mesa = "edges + nodematch('Grade') + nodefactor('Sex') + absdiff('Grade')")
networks <- list(flo_null = "flomarriage", flo_wealth = "flomarriage", flo_full = "flomarriage",
                 mesa = "faux.mesa.high")
exact <- lapply(names(fits), function(name) {
  fit <- fits[[name]]
  terms <- names(coef(fit))
  ame <- lapply(setNames(nm = terms[-1]), function(t) {
    a <- ergm.AME(fit, t)
    list(ame = a[1, "AME"], se = a[1, "Delta SE"])
  })
  ci <- confint(fit)
  list(network = networks[[name]], formula = formulas[[name]], coef = as.list(coef(fit)), ame = ame,
       confint = list(low = setNames(as.list(ci[, 1]), rownames(ci)),
                      high = setNames(as.list(ci[, 2]), rownames(ci))))
})
names(exact) <- names(fits)

tables <- list(
  three = screenreg(list(f1, f2, f3)),
  digits3 = screenreg(list(f2), digits = 3),
  named = screenreg(list(f1, f2), custom.model.names = c("Null", "Wealth")),
  mesa = screenreg(list(f4, f2)),
  latex = texreg(list(f1, f2)),
  html = htmlreg(list(f1, f2))
)

jsonlite::write_json(
  list(ergm_version = as.character(packageVersion("ergm")),
       ergMargins_version = as.character(packageVersion("ergMargins")),
       texreg_version = as.character(packageVersion("texreg")),
       predict = predict_ref, exact = exact, tables = tables),
  file.path("tests", "data", "r_interpret_reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE
)
