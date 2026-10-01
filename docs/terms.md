# Term reference

Every term has ergm's definition and statistic names, and was checked
against R's `summary()` on directed and undirected networks. Repeated names
are made unique as in R: `mix.Race.White.White.1`. Below, $y_{ij}$
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

`degree(d)`, undirected
: Number of vertices with degree exactly $d$, $\sum_i [d_i = d]$, for one or
  more $d$ (`degree(0:3)`). Names: `degree0`, `degree1`...

`idegree(d)`, `odegree(d)`, directed
: Number of vertices with in- or out-degree exactly $d$. Names: `idegree0`,
  `odegree0`...

`isolates()`
: Number of vertices without ties (in either direction, if directed). Name:
  `isolates`.

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

`gwdsp(decay, fixed=TRUE)`
: Geometrically weighted dyadwise shared partners: as gwesp, but over all
  pairs of vertices, tied or not (ordered pairs and OTP shared partners if
  directed). Names: `gwdsp.fixed.<decay>`, `gwdsp.OTP.fixed.<decay>`.

`esp(d)`
: Number of ties with exactly $d$ edgewise shared partners, for one or more
  $d$, $\sum_D y_{ij} [s_{ij} = d]$ (OTP shared partners if directed). Names:
  `esp0`, `esp1`..., or `esp.OTP0`...

`dsp(d)`
: Number of pairs of vertices, tied or not, with exactly $d$ shared partners
  (ordered pairs and OTP if directed). Names: `dsp0`..., or `dsp.OTP0`...

The geometrically weighted terms need `fixed=TRUE`: estimating the decay
(a curved ERGM) is not supported yet.

## Attribute terms

`nodematch(attr, diff=FALSE)` {octicon}`dot-fill`
: Number of ties between vertices with the same value of `attr`,
  $\sum_D y_{ij} [x_i = x_j]$. With `diff=TRUE`, one statistic per value.
  Names: `nodematch.<attr>`, or `nodematch.<attr>.<value>`.

`nodemix(attr, levels=None, levels2=-1)` {octicon}`dot-fill`
: Number of ties for each *mixing type*: each pair of levels of `attr`, from
  sender to receiver if directed. Types are ordered as in ergm, column by column
  of the levels' mixing matrix: (1, 1), (1, 2), (2, 2), (1, 3)... if
  undirected, (1, 1), (2, 1), (3, 1)... if directed. `levels2` selects types:
  `-1` (the default) all but the first, `TRUE` all, 1-based indices such as
  `c(1, 3)`, negative indices to leave out (`-c(1, 3)`), or a logical matrix.
  `levels` selects levels, by name or index. Names:
  `mix.<attr>.<level>.<level>`.

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

## Operators

`offset(term)`
: Fixes the term's coefficients at the values given to `ergm()` as
  `offset_coef`, in formula order, instead of estimating them. A coefficient
  of `-inf` forbids the ties the term counts (dyad-independent terms only).
  Names: `offset(<name>)`.

`F(formula, filter)`
: Evaluates the terms of `formula` on the network of the ties that pass
  `filter`: a dyad-independent term with one statistic, which a tie passes
  if adding it would change the statistic. `~!filter` keeps the other ties.
  In a formula string, both are one-sided R formulas:
  `F(~gwesp(0.5, fixed=TRUE), ~nodematch('Grade'))`; in Python,
  `F(gwesp(0.5, fixed=True), nodematch("Grade"), negate=False)`. Names:
  `F(<filter>)~<name>`, with the filter as R prints it, such as
  `F(nodematch("Grade"))~gwesp.fixed.0.5`.

## Proposals

Models with `triangle`, `ttriple`, `ctriple`, `gwesp`, `gwdsp`, `esp` or
`dsp` (also inside `F()`) mix half tie/no-tie (TNT) MCMC proposals with
*triadic* proposals, which pick a vertex, one of its neighbors and one of
that neighbor's neighbors, and toggle the tie that would close or open the
triangle. Like ergm's default for these models, this explores clustered
networks much faster. `triadic_weight` in {class}`ergmx.Control` changes the
share. Degree-preserving [constraints](user-guide/constraints.md) use their
own moves.
