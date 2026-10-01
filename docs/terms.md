# Term reference

Every term has ergm's definition and statistic names, and was checked
against R's `summary()` on directed and undirected networks. Below, $y_{ij}$
is 1 if there is a tie from $i$ to $j$ (or between them, if undirected),
$d_i$ is the degree of $i$ (in- and out-degrees $d^{in}_i$, $d^{out}_i$ if
directed), $x_i$ a vertex attribute and $D$ the set of dyads: pairs $i < j$
if undirected, ordered pairs $i \neq j$ if directed.

The *dyad-independent* terms are marked with {octicon}`dot-fill`: a model of
only those is fitted exactly, by logistic regression.

## Dyadic terms

`edges()` {octicon}`dot-fill`
: Number of ties, $\sum_{D} y_{ij}$. Name: `edges`.

`mutual()`, directed
: Number of reciprocated pairs, $\sum_{i<j} y_{ij} y_{ji}$. Name: `mutual`.

`edgecov(x)` {octicon}`dot-fill`
: Sum of a dyadic covariate over ties, $\sum_D y_{ij} x_{ij}$. `x` is the name
  of a graph attribute holding an $n \times n$ matrix (in a formula string:
  `edgecov('trade')`), the matrix itself, or a graph on the same vertices.
  Undirected networks use the upper triangle, $x_{\min(i,j)\max(i,j)}$.
  Name: `edgecov.<attribute>`, or `edgecov` for a matrix.

## Degree terms

`kstar(k)`, undirected
: Number of $k$-stars, $\sum_i \binom{d_i}{k}$, for one or more $k$
  (`kstar(2:3)`). Names: `kstar2`, `kstar3`...

`istar(k)`, `ostar(k)`, directed
: In- and out-$k$-stars, $\sum_i \binom{d^{in}_i}{k}$ and
  $\sum_i \binom{d^{out}_i}{k}$. Names: `istar2`, `ostar2`...

`gwdegree(decay, fixed=TRUE)`, undirected
: Geometrically weighted degree distribution,
  $e^{\alpha} \sum_i \left[1 - (1 - e^{-\alpha})^{d_i}\right]$ with
  $\alpha$ the decay. A positive coefficient favors spreading ties evenly, a
  negative one hubs and isolates. Name: `gwdeg.fixed.<decay>`.

`gwidegree(decay, fixed=TRUE)`, `gwodegree(decay, fixed=TRUE)`, directed
: The same with in- and out-degrees. Names: `gwideg.fixed.<decay>`,
  `gwodeg.fixed.<decay>`.

## Triad terms

`triangle()`
: Number of triangles. In directed networks, ergm counts transitive plus
  cyclic triples, `ttriple + ctriple`. Name: `triangle`.

`ttriple()`, directed
: Number of transitive triples: $i \to j$, $j \to k$ and $i \to k$. Name:
  `ttriple`.

`ctriple()`, directed
: Number of cyclic triples: $i \to j \to k \to i$. Name: `ctriple`.

`gwesp(decay, fixed=TRUE)`
: Geometrically weighted edgewise shared partners,
  $e^{\alpha} \sum_D y_{ij} \left[1 - (1 - e^{-\alpha})^{s_{ij}}\right]$, with
  $s_{ij}$ the number of vertices tied to both $i$ and $j$. A positive
  coefficient favors ties that close triangles, with diminishing returns as
  $\alpha$ gets smaller. In directed networks, shared partners are outgoing
  two-paths, $i \to k \to j$ (ergm's default `type = "OTP"`). Names:
  `gwesp.fixed.<decay>`, `gwesp.OTP.fixed.<decay>` if directed.

`gwdsp(decay, fixed=TRUE)`, undirected
: Geometrically weighted dyadwise shared partners: as gwesp, but over all
  pairs of vertices, tied or not. Name: `gwdsp.fixed.<decay>`.

The geometrically weighted terms need `fixed=TRUE`: estimating the decay
(a curved ERGM) is not supported yet.

## Attribute terms

`nodematch(attr, diff=FALSE)` {octicon}`dot-fill`
: Number of ties between vertices with the same value of `attr`,
  $\sum_D y_{ij} [x_i = x_j]$. With `diff=TRUE`, one statistic per value.
  Names: `nodematch.<attr>`, or `nodematch.<attr>.<value>`.

`nodefactor(attr)` {octicon}`dot-fill`
: For each level $\ell$ of `attr` but the first, the number of tie endpoints
  at that level, $\sum_D y_{ij} ([x_i = \ell] + [x_j = \ell])$. Names:
  `nodefactor.<attr>.<level>`.

`nodeifactor(attr)`, `nodeofactor(attr)` {octicon}`dot-fill`, directed
: The same, counting only receivers ($[x_j = \ell]$) or senders
  ($[x_i = \ell]$). Names: `nodeifactor.<attr>.<level>`,
  `nodeofactor.<attr>.<level>`.

`nodecov(attr)` {octicon}`dot-fill`
: Sum of a numeric attribute over tie endpoints, $\sum_D y_{ij} (x_i + x_j)$.
  Name: `nodecov.<attr>`.

`nodeicov(attr)`, `nodeocov(attr)` {octicon}`dot-fill`, directed
: The same for receivers ($x_j$) or senders ($x_i$) only. Names:
  `nodeicov.<attr>`, `nodeocov.<attr>`.

`absdiff(attr)` {octicon}`dot-fill`
: Sum over ties of the absolute difference in a numeric attribute,
  $\sum_D y_{ij} |x_i - x_j|$. Name: `absdiff.<attr>`.

## Proposals

Models with `triangle`, `ttriple`, `ctriple`, `gwesp` or `gwdsp` mix half
tie/no-tie (TNT) MCMC proposals with *triadic* proposals, which pick a
vertex, one of its neighbors and one of that neighbor's neighbors, and toggle
the tie that would close or open the triangle. Like ergm's default for these
models, this explores clustered networks much faster. `triadic_weight` in
{class}`ergmx.Control` changes the share.
