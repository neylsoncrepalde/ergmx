---
file_format: mystnb
kernelspec:
  name: python3
---

# A complete multilevel ERGM analysis

A multilevel network has vertices at two levels, such as researchers and the
laboratories they belong to, with ties within each level and affiliation
ties between them ([Lazega et al. 2008](https://doi.org/10.1016/j.socnet.2008.02.001)). This page fits the multilevel ERGMs
of [Wang, Robins, Pattison and Lazega (2013)](https://doi.org/10.1016/j.socnet.2013.01.004), as their program MPNet does,
to a network simulated from a known model, so we can check that the
estimates recover the true values. It draws the network with
[multinets](https://github.com/neylsoncrepalde/multinets-py), then
describes, fits, checks, compares and reports the models with `ergmx`.

## The data

`labs_sim` has 120 researchers and 30 laboratories. Its vertex attribute
`type` is multinets' convention for the levels (`True` for laboratories),
and `level` names them, for `ergmx`'s formulas. `nodemix` counts the ties
of each kind:

```{code-cell} ipython3
import random

import igraph as ig
import matplotlib.pyplot as plt
import multinets as mn

import ergmx
from ergmx import datasets

labs = datasets.load("labs_sim")  # an igraph.Graph
ergmx.summary_stats(labs, "nodemix('level', levels2=TRUE)")
```

The 120 researchers have 372 ties among them, and the 30 laboratories 61.
The 150 affiliations are memberships: every researcher belongs to a
laboratory, and a quarter of them to two.

## Drawing the network

multinets lays out each level on its own band, the laboratories above the
researchers, and colours and shapes the vertices and ties by level:

```{code-cell} ipython3
random.seed(1)  # igraph's layouts use Python's random numbers
layout = mn.layout_multilevel(labs, layout="kk")
styled = mn.set_shape_multilevel(mn.set_color_multilevel(labs))

fig, ax = plt.subplots(figsize=(8, 7))
ig.plot(styled, target=ax, layout=layout, vertex_size=7, edge_width=0.6);
```

The laboratories are the blue squares, the researchers the red circles, and
the affiliations the grey ties between them.

## The model behind the data

The memberships were drawn first. Then the ties within each level were
simulated given them ([the script](https://github.com/neylsoncrepalde/ergmx/blob/main/scripts/simulate_labs.py)), from a model with these
effects:

| Term | Effect | True value |
|---|---|---|
| `S(~edges, ~level == 'researcher')` | the researchers' baseline log-odds of a tie | -3.6 |
| `S(~gwesp(0.693147, fixed=TRUE), ~level == 'researcher')` | triadic closure among researchers | 0.3 |
| `S(~edges, ~level == 'laboratory')` | the laboratories' baseline log-odds of a tie | -2.8 |
| `txbx('level')` | researchers of the same laboratory are tied | 1.5 |
| `txax('level')` | laboratories that share a researcher are tied | 1.5 |
| `c4axb('level')` | members of tied laboratories are tied: the levels are aligned | 0.4 |

The operator {func}`~ergmx.S` evaluates terms on the network within a set
of vertices ([Multilevel networks](../multilevel.md)). The last three terms
are MPNet's configurations that join ties of different kinds. Their first
argument is the level attribute; its values in sorted order are level A,
the laboratories, and level B, the researchers. So `txbx` counts the ties
between researchers who share a laboratory, and `txax` those between
laboratories that share a researcher. `c4axb` counts the four-cycles of a
tie between researchers, a tie between laboratories and the two
affiliations that join them.

As the memberships are given, the models are fitted given them, with the
constraint `blocks('level', levels2=2)`. It fixes the dyads of the second
mixing type, in `nodemix`'s order: (laboratory, laboratory), (laboratory,
researcher), (researcher, researcher) ([Constraints](../constraints.md)).

## A model of each level

A first model has each level's density and the closure among researchers,
but nothing that joins the levels:

```{code-cell} ipython3
given = "blocks('level', levels2=2)"
levels = (
    "S(~edges + gwesp(0.693147, fixed=TRUE), ~level == 'researcher')"
    " + S(~edges, ~level == 'laboratory')"
)
separate = ergmx.ergm(labs, levels, constraints=given, seed=1)
separate.summary()
```

It overestimates the closure among researchers (0.39, against a true 0.3):
without the cross-level terms, the model takes the ties among members of
the same laboratory, who often share partners, for triadic closure. The
laboratories' `edges` coefficient is their overall log-odds of a tie, far
above the baseline of the true model.

## Adding the cross-level effects

```{code-cell} ipython3
cross = " + txbx('level') + txax('level') + c4axb('level')"
multilevel = ergmx.ergm(labs, levels + cross, constraints=given, seed=1, interval=4096)
multilevel.summary()
```

`interval=4096` spaces the networks of the MCMC sample by 4096 proposals,
four times the default, for better mixing (see the next section). Sharing
a laboratory multiplies the odds of a tie between two researchers by about
exp(1.54) = 4.7, and sharing a researcher those of two laboratories by
about exp(1.80) = 6.

## Are the true values recovered?

```{code-cell} ipython3
truth = {
    'S(level=="researcher")~edges': -3.6,
    'S(level=="researcher")~gwesp.fixed.0.693147': 0.3,
    'S(level=="laboratory")~edges': -2.8,
    "TXBX.level": 1.5,
    "TXAX.level": 1.5,
    "C4AXB.level": 0.4,
}
intervals = multilevel.confint().to_dict()
print(f"{'':<45}{'true':>6}{'estimate':>10}   95% interval")
for name, value in truth.items():
    low, high = intervals[name].values()
    print(f"{name:<45}{value:>6.1f}{multilevel.coef[name]:>10.2f}   [{low:.2f}, {high:.2f}]")
```

Every true value is inside its 95% interval.

## Did the estimation converge?

As for any Monte Carlo MLE, the simulated networks should reproduce the
observed statistics, and the chains agree and be stationary
([Convergence and MCMC diagnostics](../diagnostics.md)):

```{code-cell} ipython3
multilevel.mcmc_diagnostics()
```

The mean deviations are below a tenth of an SD, R-hat is close to 1, and no
Geweke z-score is beyond ±2. Hotelling's test flags the deviations as
significant only because the effective sizes are large: they are
negligible. With the default interval of 1024 proposals, 4 of the 24 Geweke
z-scores were beyond ±2: the chains hadn't mixed as well.

## Does the model fit each level?

{meth}`~ergmx.ErgmFit.gof` with `by="level"` compares each level's network
with those of 100 networks simulated from the model (given the memberships,
as the fit): its degrees, edgewise shared partners and distances, as MPNet's
goodness of fit does ([Goodness of fit](../goodness-of-fit.md)):

```{code-cell} ipython3
multilevel.gof(by="level", stats=["degree", "espartners", "distance"], seed=1).plot();
```

The model reproduces both levels' distributions (the black lines are the
observed ones, the boxplots the simulated ones), as the true model should.
The laboratories' degrees are noisier, as there are only 30 of them: one
laboratory with 4 ties and four with 8 are at the edge of the simulations,
deviations that happen by chance with so few vertices.

## Which model is better?

```{code-cell} ipython3
ergmx.compare(separate, multilevel)
```

The cross-level effects improve the model enormously: the AIC is about 200
lower, and the likelihood-ratio test is highly significant. The levels are
not independent: who collaborates with whom among researchers depends on
their laboratories, and the other way round.

## Reporting the results

{func}`ergmx.table` puts the models side by side, with `rename=` for
readable labels:

```{code-cell} ipython3
ergmx.table(
    separate,
    multilevel,
    names=["Each level", "Multilevel"],
    rename={
        'S(level=="researcher")~edges': "Researchers: edges",
        'S(level=="researcher")~gwesp.fixed.0.693147': "Researchers: GWESP",
        'S(level=="laboratory")~edges': "Laboratories: edges",
        "TXBX.level": "Same laboratory (TXBX)",
        "TXAX.level": "Shared researcher (TXAX)",
        "C4AXB.level": "Alignment (C4AXB)",
    },
)
```

[Multilevel networks](../multilevel.md) has every MPNet configuration, how
each MPNet effect is written in `ergmx`, and other ways to model multilevel
networks.

Next, [A complete valued ERGM analysis](valued.md) models the counts on
the ties of a network.
