---
file_format: mystnb
kernelspec:
  name: python3
---

# Networks

`ergmx` models binary networks, directed or undirected, given as an
[igraph](https://python.igraph.org) or a [networkx](https://networkx.org)
graph. There is no network class of its own: pass your graph, and the
results (simulated networks, for example) come back as graphs of the same
kind.

## The bundled networks

{mod}`ergmx.datasets` has the networks of R's ergm documentation:

```{code-cell} ipython3
from ergmx import datasets

for name in datasets.names():
    g = datasets.load(name)
    kind = "directed" if g.is_directed() else "undirected"
    print(f"{name:20} {g.vcount():5} vertices {g.ecount():5} edges  {kind}")
```

```{code-cell} ipython3
print(datasets.describe("faux.mesa.high"))
```

`load()` returns an {class}`igraph.Graph`; `load(name, backend="networkx")`
returns a {class}`networkx.Graph` or {class}`networkx.DiGraph` with the same
vertices, in the same order.

## Vertex attributes

Terms such as `nodematch('Grade')` read vertex attributes. In igraph they are
`g.vs["Grade"]`; in networkx, node data (`G.nodes[v]["Grade"]`):

```{code-cell} ipython3
import ergmx

mesa = datasets.load("faux.mesa.high")
mesa_nx = datasets.load("faux.mesa.high", backend="networkx")

formula = "edges + nodematch('Grade') + nodefactor('Race')"
ergmx.summary_stats(mesa, formula) == ergmx.summary_stats(mesa_nx, formula)
```

Categorical attributes can hold any sortable values (strings, integers...).
Their levels are sorted, and terms with one statistic per level, like
`nodefactor`, drop the first level, as ergm does. Numeric terms, like
`nodecov`, need numbers.

## Graph attributes

Dyadic covariates for `edgecov` are n x n matrices stored as graph attributes,
`g["name"]` in igraph and `G.graph["name"]` in networkx. ergm's `flobusiness`
network can be a covariate of `flomarriage`:

```{code-cell} ipython3
import numpy as np

flomarriage = datasets.load("flomarriage")
business = datasets.load("flobusiness")
flomarriage["business"] = np.array(business.get_adjacency().data)

ergmx.summary_stats(flomarriage, "edges + edgecov('business')")
```

`edgecov` also accepts the matrix itself, or a graph on the same vertices:
`edgecov(business)`.

## What is not supported

A network must not have multiple edges between the same vertices (use
`igraph.Graph.simplify()`, or `nx.Graph` rather than `nx.MultiGraph`) or
self-loops. Edge attributes such as weights are ignored, as ERGMs model whether
ties exist, not their values, except one: an edge with a true `na` attribute
marks a dyad whose value is unknown (see [](missing-data.md)).
