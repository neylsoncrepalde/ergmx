---
file_format: mystnb
kernelspec:
  name: python3
---

# Formulas

A model is a list of terms, each contributing one or more network statistics.
`ergmx` takes it in two forms.

**A string in R syntax**, so models can be pasted from R:

```{code-cell} ipython3
import ergmx
from ergmx import datasets

mesa = datasets.load("faux.mesa.high")
ergmx.summary_stats(mesa, "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)")
```

`summary_stats` is R's `summary(net ~ formula)`: the statistics of a network.
The string follows R's conventions: `TRUE`/`FALSE` (or `True`/`False`),
`c(2, 3)` and `2:3` for vectors, and a left-hand side such as
`"mesa ~ edges + triangle"` is ignored. Nothing in it is evaluated: arguments
must be literals.

**Terms combined with `+`**, which a program can build:

```{code-cell} ipython3
from ergmx import edges, gwesp, nodematch

formula = edges() + nodematch("Grade") + gwesp(0.5, fixed=True)
ergmx.summary_stats(mesa, formula)
```

Both forms can be mixed: `edges() + "triangle"`.

## Statistic names

Names follow ergm, so tables can be compared with R's output line by line.
Terms with several statistics name each of them:

```{code-cell} ipython3
ergmx.summary_stats(mesa, "kstar(2:3) + nodefactor('Race') + nodematch('Sex', diff=TRUE)")
```

## Errors

Unknown terms, bad arguments and terms that don't apply to the network fail
before any fitting, with a message that says why:

```{code-cell} ipython3
try:
    ergmx.summary_stats(mesa, "edges + mutual")   # mutual needs a directed network
except ValueError as error:
    print(error)
```

The [term reference](../terms.md) lists every term, its statistic and the
networks it applies to.
