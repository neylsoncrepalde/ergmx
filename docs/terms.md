# Term reference

Every term of ergm has ergm's definition and statistic names, and was
checked against R's `summary()` on directed and undirected networks. MPNet's
[multilevel terms](#multilevel-terms), which no R package has, follow [Wang
et al. (2013)](https://doi.org/10.1016/j.socnet.2013.01.004), and were checked against their definitions. Repeated names
are made unique as in R: `mix.Race.White.White.1`. Below, $y_{ij}$
is 1 if there is a tie from $i$ to $j$ (or between them, if undirected),
$d_i$ is the degree of $i$ (in- and out-degrees $d^{in}_i$, $d^{out}_i$ if
directed), $x_i$ a vertex attribute and $D$ the set of dyads: pairs $i < j$
if undirected, ordered pairs $i \neq j$ if directed.

The *dyad-independent* terms are marked with {octicon}`dot-fill`: a model of
only those is fitted exactly, by logistic regression.

Arguments that select levels of an attribute (`levels`, and the older
`keep`, `base` and `nodes`) take ergm's specifications: `NULL` or `TRUE` for
all, values (`c('White', 'Black')`, in that order), 1-based indices into the
sorted values (`2:3`), or negative indices to leave some out (`-1`, the
default of `nodefactor`). Attributes with missing values (None or NaN) are
refused, as in ergm: recode them as a level of their own.

Interactions of dyad-independent terms are written as in ergm:
`nodecov('Grade'):nodematch('Sex')` is the sum over ties of the product of
the two terms' change statistics, named `nodecov.Grade:nodematch.Sex` (with
several statistics on a side, one per pair, the first side's varying
fastest), and `a*b` is `a + b + a:b`.

## Dyadic terms

`edges()` {octicon}`dot-fill`
: Number of ties, $\sum_{D} y_{ij}$. Name: `edges`.

`mutual(same=, by=, diff=FALSE, levels=)`, directed
: Number of reciprocated pairs, $\sum_{i<j} y_{ij} y_{ji}$. With `same`, only
  those between vertices with the same value of that attribute, in total or
  by value (`diff=TRUE`); with `by`, for each value, the vertices with it in
  reciprocated pairs. Names: `mutual`, `mutual.<attr>`,
  `mutual.same.<attr>.<value>`, `mutual.by.<attr>.<value>`.

`asymmetric(attr=, diff=FALSE, levels=)`, directed
: Number of pairs with a tie in one direction only; with `attr`, only pairs
  of vertices with the same value, in total or by value. Names: `asymmetric`,
  `asymmetric.<attr>`, `asymmetric.<attr>.<value>`.

`sender(nodes=-1)`, `receiver(nodes=-1)` {octicon}`dot-fill`, directed
: Each vertex's out-degree (in-degree), one statistic per vertex of `nodes`,
  by default all but the first. Names: `sender2`, `sender3`... by vertex
  position.

`sociality(attr=, levels=, nodes=-1)` {octicon}`dot-fill`, undirected
: Each vertex's degree, one statistic per vertex of `nodes`; with `attr`,
  only its ties to vertices with the same value. Names: `sociality2`...,
  `sociality2.<attr>`.

`density()`, `meandeg()` {octicon}`dot-fill`
: The number of ties over the number of dyads (the first mode's times the
  second's, if bipartite), and the mean degree, $2|y|/n$ ($|y|/n$ if
  directed). Names: `density`, `meandeg`.

`dyadcov(x)` {octicon}`dot-fill`
: In directed networks, a dyadic covariate summed by dyad state: over mutual
  dyads, over those with only the tie from the lower- to the higher-numbered
  vertex (in the upper triangle of the adjacency matrix), and over the
  reverse; of `x`, its upper triangle, as in ergm. This is how ergm
  documents `dyadcov`, but ergm 4.12 swaps the last two: using `dyadcov` on
  a directed network warns with {class}`ergmx.ErgmDifferenceWarning`. In
  undirected networks, `edgecov`. Names: `dyadcov.<attribute>.mutual`,
  `.utri`, `.ltri`.

`hamming(x=None, cov=None)` {octicon}`dot-fill`
: The Hamming distance to a reference network: the number of dyads whose
  value differs from `x`'s (the observed network by default; a graph
  attribute holding an adjacency matrix, the matrix or a graph), each
  weighted by the covariate `cov` if given. Names: `hamming`,
  `hamming.<attribute>`.

`attrcov(attr, mat)` {octicon}`dot-fill`
: Sum over ties of the entry of `mat` (levels by levels of `attr`, sorted) of
  the pair of their vertices' levels. Name: `attrcov.<attr>`.

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

`kstar(k, attr=)`, `istar(k, attr=)`, `ostar(k, attr=)` with an attribute
: Only the stars whose vertices all have the same value of `attr`. Names:
  `kstar2.<attr>`...

`degree(d, by=, homophily=FALSE, levels=)`, undirected
: Number of vertices with degree exactly $d$, $\sum_i [d_i = d]$, for one or
  more $d$ (`degree(0:3)`). With `by`, one count per value of that attribute
  (of `levels`): `deg1.<attr>.<value>`. With `homophily=TRUE`, degrees only
  count the ties between vertices with the same value (as ergm, every
  vertex is counted, and the values left out of `levels` form one more
  value): `deg1.homophily.<attr>`. Names: `degree0`, `degree1`...

`idegree(d, by=, homophily=, levels=)`, `odegree(d, ...)`, directed
: Number of vertices with in- or out-degree exactly $d$. Names: `idegree0`,
  `odegree0`..., and `ideg1.<attr>.<value>`... with `by`.

`degrange(from, to=Inf, by=, homophily=FALSE, levels=)`, undirected
: Number of vertices with degree in $[\text{from}, \text{to})$, for each
  pair (either can have length 1, recycled). `by`, `homophily` and `levels`
  as for `degree`. Names: `deg1to4`, `deg2+` (`to=Inf`),
  `deg1to4.<attr><value>`, `deg1to4.homophily.<attr>`.

`idegrange(...)`, `odegrange(...)`, directed
: The same with in- and out-degrees. Names: `ideg1to4`, `odeg3+`...

`degree1.5()`, `idegree1.5()`, `odegree1.5()`
: Sum over vertices of their degree (in-, out-degree) to the power 3/2. In
  Python, `degree1_5()`; in formula strings, both names. Names:
  `degree1.5`...

`concurrentties(by=, levels=)`, undirected
: Sum over vertices of their ties beyond the first, $\sum_i \max(d_i - 1,
  0)$, by value of `by` if given. Names: `concurrentties`,
  `concurrentties.<attr><value>`.

`isolatededges()`, undirected
: Number of ties whose two vertices have no other tie. Name: `isolatededges`.

`altkstar(lambda, fixed=TRUE)`, undirected
: Alternating $k$-stars (Snijders et al. 2006),
  $\lambda^2 \sum_i \left[(1 - 1/\lambda)^{d_i} - 1 + d_i / \lambda\right]$.
  Only with `fixed=TRUE`: ergm's version with an estimated lambda is not the
  same statistic, and ergm recommends `gwdegree`, with `edges` the same
  model. Name: `altkstar.<lambda>`.

`isolates()`
: Number of vertices without ties (in either direction, if directed). Name:
  `isolates`.

`concurrent(by=, levels=)`, undirected
: Number of vertices with degree 2 or more, by value of `by` if given.
  Names: `concurrent`, `concurrent.<attr><value>`.

`twopath()`
: Number of 2-paths: $i \to j \to k$ with $i \neq k$ if directed;
  `kstar(2)` if undirected. Name: `twopath`.

`gwdegree(decay, fixed=TRUE)`, undirected
: Geometrically weighted degree distribution,
  $e^{\alpha} \sum_i \left[1 - (1 - e^{-\alpha})^{d_i}\right]$ with
  $\alpha$ the decay. A positive coefficient favors spreading ties evenly, a
  negative one hubs and isolates. Name: `gwdeg.fixed.<decay>`.

`gwidegree(decay, fixed=TRUE)`, `gwodegree(decay, fixed=TRUE)`, directed
: The same with in- and out-degrees. Names: `gwideg.fixed.<decay>`,
  `gwodeg.fixed.<decay>`.

`gwdegree(decay, fixed=TRUE, attr=, levels=)` with an attribute (also `gwidegree`, `gwodegree`)
: One statistic per value of `attr`: the geometrically weighted degrees of
  its vertices. As in ergm, the decay must be fixed. Names:
  `gwdeg<decay>.<attr>.<value>`.

## Triad terms

`triangle(attr=, diff=FALSE, levels=)`
: Number of triangles. In directed networks, ergm counts transitive plus
  cyclic triples, `ttriple + ctriple`. With `attr`, only the triangles whose
  vertices all have the same value, in total or by value; `ttriple` and
  `ctriple` take the same arguments. Names: `triangle`, `triangle.<attr>`,
  `triangle.<attr>.<value>`. Also `triangles`.

`ttriple()`, directed
: Number of transitive triples: $i \to j$, $j \to k$ and $i \to k$. Name:
  `ttriple`. Also `ttriad`.

`ctriple()`, directed
: Number of cyclic triples: $i \to j \to k \to i$. Name: `ctriple`. Also
  `ctriad`.

`transitive()`, directed
: Number of transitive triads: triads of types 030T, 120D, 120U and 300 in
  [Davis and Leinhardt's (1972)](https://scholar.google.com/scholar?q=%22The+structure+of+positive+interpersonal+relations+in+small+groups%22+Davis+Leinhardt) census, those with at least one transitive
  triple and no intransitive two-path. This is how ergm documents its
  `transitive` term, but ergm 4.12 computes transitive triples instead, the
  same as `ttriple`: use `ttriple` to reproduce ergm's results. Using
  `transitive` warns with {class}`ergmx.ErgmDifferenceWarning`. Name:
  `transitive`.

`cycle(k)`
: Number of cycles of length $k$, for one or more $k$: 3 or more if
  undirected, 2 or more if directed (`cycle(2)` is `mutual`). Names: `cycle3`...

`triadcensus(levels=)`
: The number of triads of each type of Davis and Leinhardt's census, in
  ergm's order: 003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201,
  120D, 120U, 120C, 210, 300 (codes 0 to 15), by default all but 003; in
  undirected networks, triads with 0, 1, 2 or 3 ties, by default 1 to 3.
  `levels` selects types by code or name (`c('021D', '300')`). Names:
  `triadcensus.021D`..., `triadcensus.1`...

`balance()`
: Number of balanced triads: types 102 and 300 (in undirected networks,
  triads with one or three ties). Name: `balance`.

`intransitive()`, directed
: Number of intransitive triads: types 111D, 201, 111U, 021C and 030C. This
  is how ergm documents its `intransitive` term, but ergm 4.12 computes
  intransitive triples instead (two-paths $i \to j \to k$ without $i \to k$),
  `twopath` minus `ttriple`: using `intransitive` warns with
  {class}`ergmx.ErgmDifferenceWarning`. Name: `intransitive`.

`simmelian()`, `nearsimmelian()`, `simmelianties()`, directed
: Simmelian triads (Krackhardt and Handcock 2007), complete ones (type
  300); near-Simmelian triads, one tie short (type 210); and the ties in at
  least one Simmelian triad. Names: `simmelian`, `nearsimmelian`,
  `simmelianties`.

`transitiveties(attr=, levels=)`, `cyclicalties(attr=, levels=)`
: Number of ties $i \to j$ with a two-path $i \to k \to j$ (transitive) or
  $j \to k \to i$ (cyclical); in undirected networks, both are the ties with
  a shared partner. With `attr`, only ties and two-paths whose three vertices
  have the same value. Names: `transitiveties`, `transitiveties.<attr>`...

`threetrail(levels=)`
: Number of 3-trails: walks along three distinct ties, a triangle counting
  three, $\sum_{ij} y_{ij} (d_i - 1)(d_j - 1)$ if undirected. In directed
  networks, four statistics by the directions of the outer steps around the
  middle one: RRR ($i \to j \to k \to l$), RRL, LRR and LRL; `levels`
  selects some. Names: `threetrail`, `threetrail.RRR`... Also `threepath`.

`opentriad()`, undirected
: Number of 2-stars minus three times the number of triangles. Name:
  `opentriad`.

`localtriangle(x)`
: Number of triangles (transitive plus cyclic triples, if directed) whose
  three pairs of vertices are neighbours in `x`: a graph attribute holding a
  symmetric adjacency matrix, the matrix or a graph. Name:
  `localtriangle.<attribute>`.

`m2star()`, directed
: Number of mixed 2-stars, $i \to j \to k$ with $i \neq k$: `twopath`.
  Name: `m2star`.

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
  pairs of vertices, tied or not (ordered pairs if directed). Names:
  `gwdsp.fixed.<decay>`, `gwdsp.OTP.fixed.<decay>`.

`gwnsp(decay, fixed=TRUE)`
: Geometrically weighted non-edgewise shared partners: as gwesp, over the
  pairs without a tie; `gwdsp` minus `gwesp`. Names: `gwnsp.fixed.<decay>`...

`esp(d)`
: Number of ties with exactly $d$ edgewise shared partners, for one or more
  $d$, $\sum_D y_{ij} [s_{ij} = d]$ (OTP shared partners if directed). Names:
  `esp0`, `esp1`..., or `esp.OTP0`...

`dsp(d)`
: Number of pairs of vertices, tied or not, with exactly $d$ shared partners
  (ordered pairs if directed). Names: `dsp0`..., or `dsp.OTP0`...

`nsp(d)`
: Number of pairs without a tie with exactly $d$ shared partners. Names:
  `nsp0`..., or `nsp.OTP0`...

In directed networks, every shared partner term takes a `type`, as in ergm:
a shared partner $k$ of the pair $(i, j)$ is

| `type` | $k$ is a shared partner if |
|---|---|
| `"OTP"` (default), outgoing two-path | $i \to k \to j$ |
| `"ITP"`, incoming two-path | $j \to k \to i$ |
| `"RTP"`, reciprocated two-path | $i \leftrightarrow k \leftrightarrow j$ |
| `"OSP"`, outgoing shared partner | $i \to k$ and $j \to k$ |
| `"ISP"`, incoming shared partner | $k \to i$ and $k \to j$ |

and the type is part of the names: `gwesp.ITP.fixed.0.5`, `esp.OSP1`.
`desp`, `ddsp`, `dnsp`, `dgwesp`, `dgwdsp` and `dgwnsp` are the same terms
for directed networks only, as in ergm. In
ergm 4.12.0, the edgewise statistics of type RTP (`esp`, `gwesp`, `nsp`) are
wrong unless its shared-partner cache is turned off
(`term.options = list(cache.sp = FALSE)`), a bug fixed in its development
version; ergmx's match ergm's with the cache off.

With `fixed=FALSE` (ergm's default), the geometrically weighted terms
estimate their decay: see [curved terms](#curved-terms).

## Attribute terms

`nodematch(attr, diff=FALSE, levels=)` {octicon}`dot-fill`
: Number of ties between vertices with the same value of `attr`,
  $\sum_D y_{ij} [x_i = x_j]$, for the values of `levels` (all by default).
  With `diff=TRUE`, one statistic per value. Names: `nodematch.<attr>`, or
  `nodematch.<attr>.<value>`.

`nodemix(attr, levels=None, levels2=-1)` {octicon}`dot-fill`
: Number of ties for each *mixing type*: each pair of levels of `attr`, from
  sender to receiver if directed. Types are ordered as in ergm, column by column
  of the levels' mixing matrix: (1, 1), (1, 2), (2, 2), (1, 3)... if
  undirected, (1, 1), (2, 1), (3, 1)... if directed. `levels2` selects types:
  `-1` (the default) all but the first, `TRUE` all, 1-based indices such as
  `c(1, 3)`, negative indices to leave out (`-c(1, 3)`), or a logical matrix.
  `levels` selects levels, by name or index. Names:
  `mix.<attr>.<level>.<level>`.

`nodefactor(attr, levels=-1)` {octicon}`dot-fill`
: For each level $\ell$ of `levels`, by default all but the first, the
  number of tie endpoints at that level, $\sum_D y_{ij} ([x_i = \ell] +
  [x_j = \ell])$. Names: `nodefactor.<attr>.<level>`.

`nodeifactor(attr)`, `nodeofactor(attr)` {octicon}`dot-fill`, directed
: The same, counting only receivers ($[x_j = \ell]$) or senders
  ($[x_i = \ell]$). Names: `nodeifactor.<attr>.<level>`,
  `nodeofactor.<attr>.<level>`.

`nodecov(attr)` {octicon}`dot-fill`
: Sum of a numeric attribute over tie endpoints, $\sum_D y_{ij} (x_i + x_j)$.
  Name: `nodecov.<attr>`. Also `nodemain`.

`nodeicov(attr)`, `nodeocov(attr)` {octicon}`dot-fill`, directed
: The same for receivers ($x_j$) or senders ($x_i$) only. Names:
  `nodeicov.<attr>`, `nodeocov.<attr>`.

`absdiff(attr)` {octicon}`dot-fill`
: Sum over ties of the absolute difference in a numeric attribute,
  $\sum_D y_{ij} |x_i - x_j|$. Name: `absdiff.<attr>`.

`absdiffcat(attr)` {octicon}`dot-fill`
: For each distinct nonzero absolute difference $\delta$ of a numeric
  attribute, the number of ties with $|x_i - x_j| = \delta$. Names:
  `absdiff.<attr>.<difference>`.

`mm(attrs, levels=, levels2=-1)` {octicon}`dot-fill`
: The cells of a mixing matrix, as ergm's `mm`: `mm('A')` (or `mm(~A)`) is
  attribute A with itself, `mm(A~B)` rows of A and columns of B (from
  senders to receivers if directed, both orientations of each tie if
  undirected), `mm(A~.)` and `mm(.~B)` its margins. Cells are ordered column
  by column, only those at or above the diagonal for an attribute with
  itself in an undirected network; `levels2` selects them (all but the first
  by default). Names: `mm[A=a,B=b]`, `mm[A=a,.]`.

`diff(attr, pow=1, dir="t-h", sign.action="identity")` {octicon}`dot-fill`
: Sum over ties of a function of the difference of the vertices' values:
  tail minus head (`dir="t-h"`, `"b1-b2"`) or head minus tail (`"h-t"`,
  `"b2-b1"`), transformed by `sign.action` (`"abs"`, `"posonly"`,
  `"negonly"`) and raised to `pow` (its sign, for `pow=0`). Undirected ties
  go from the lower- to the higher-numbered vertex, as in ergm. In Python,
  `sign_action`. Names: `diff.t-h.<attr>`, `diff.abs.<attr>`,
  `diff2.posonly.h-t.<attr>`.

`smalldiff(attr, cutoff)` {octicon}`dot-fill`
: Number of ties whose vertices' values differ by at most `cutoff`, as
  ergm's code computes it (its documentation says less than). Name:
  `smalldiff.<attr><cutoff>`.

`nodecovrange(attr)`, `nodeicovrange(attr)`, `nodeocovrange(attr)`
: Sum over vertices of the range of `attr` over their neighbours (Hoffman,
  Block and Snijders 2023): over in- or out-neighbours, or in directed
  networks for `nodecovrange` over the out-neighbours plus over the
  in-neighbours. Names: `nodecovrange.<attr>`...

`nodefactordistinct(attr, levels=TRUE)` (also `nodeofactordistinct`, `nodeifactordistinct`)
: Sum over vertices of the number of distinct values of `attr` among their
  neighbours (in either direction, if directed; or among out- or
  in-neighbours). Names: `nodefactordistinct.<attr>`...

## Bipartite terms

For [bipartite networks](user-guide/bipartite.md). `b1` terms are about the
first mode, `b2` terms the second; each has a `b2` (or `b1`) twin.

`b1star(k, attr=)`
: Number of $k$-stars centred on first-mode vertices; with `attr`, only those
  whose vertices all have the same value. Names: `b1star2`, `b1star2.<attr>`.

`b1degree(d, by=, levels=)`
: Number of first-mode vertices with degree exactly $d$, by value of `by` if
  given. Names: `b1degree0`..., `b1deg1.<attr>.<value>`.

`b1degrange(from, to=Inf, by=, homophily=, levels=)`, `b1mindegree(d)`
: First-mode vertices with degree in $[\text{from}, \text{to})$, as
  `degrange`, and with degree at least $d$. Names: `b1deg1to4`, `b1mindeg2`.

`gwb1degree(decay, fixed=TRUE)`
: Geometrically weighted degree distribution of the first mode. Name:
  `gwb1deg.fixed.<decay>`.

`b1concurrent(by=, levels=)`
: Number of first-mode vertices with degree 2 or more, by value of `by` if
  given. Names: `b1concurrent`, `b1concurrent.<attr><value>`.

`b1factor(attr, levels=-1)` {octicon}`dot-fill`
: For each level of `levels` among first-mode vertices, by default all but
  the first, the number of their ties. Names: `b1factor.<attr>.<level>`.

`b1sociality(nodes=-1)` {octicon}`dot-fill`
: Each first-mode vertex's degree, one statistic per vertex of `nodes`
  (indices among the first mode's vertices). Names: `b1sociality2`..., by
  vertex number.

`b1cov(attr)` {octicon}`dot-fill`
: Sum over ties of the first-mode endpoint's value of a numeric attribute.
  Name: `b1cov.<attr>`.

`b1nodematch(attr, diff=FALSE, alpha=1, beta=1, byb2attr=, levels=)`
: Number of 2-stars centred on second-mode vertices whose two first-mode
  ends have the same value of `attr` (Bomiriya et al. 2023): by value with
  `diff=TRUE`, by value of the centres' attribute `byb2attr`, and
  discounted with `beta` < 1 (for each tie, half its number of such
  2-stars to the power beta) or `alpha` < 1 (for each pair of matching
  ends, their shared partners to the power alpha). Names:
  `b1nodematch.<attr>`, `b1nodematch.<attr>.<value>`... (`b2nodematch`
  takes `byb1attr`).

`b1starmix(k, attr, base=, diff=TRUE)`
: Number of $k$-stars centred on first-mode vertices whose ends all have the
  same value of `attr`, by the value of the centre and (with `diff=TRUE`) of
  the ends. Names: `b1starmix.2.<attr>.<centre>.<ends>`.

`b1twostar(b1attr, b2attr=, ...)`
: Number of 2-stars centred on first-mode vertices, by the value of `b1attr`
  of the centre and the (unordered) values of `b2attr` of the two ends.
  Names: `b1twostar.<b1attr>.<value>.<b2attr>.<value>.<value>`.

`b1covrange(attr)`, `b1factordistinct(attr, levels=TRUE)`
: Sum over first-mode vertices of the range of `attr` over their
  neighbours, and of the number of distinct values among them. Names:
  `b1covrange.<attr>`, `b1factordistinct.<attr>`.

`b1dsp(d)`, `gwb1dsp(decay, fixed=TRUE)`
: Pairs of first-mode vertices with exactly $d$ shared partners, and their
  geometrically weighted distribution. Names: `b1dsp0`...,
  `gwb1dsp.fixed.<decay>`.

`edges`, `edgecov` (with a first-mode by second-mode matrix, as in ergm),
`cycle(4)`, `isolates`, `degree`, `nodematch`, `density`, `meandeg`, `diff`
(from the first mode to the second) and the other terms that don't require a
unipartite network also apply. `gwb1degree` and `gwb2degree` also take an
attribute (`attr=`, with a fixed decay), as `gwdegree` does.

## Curved terms

With `fixed=FALSE`, ergm's default, the decay of `gwesp`, `gwdsp`, `gwnsp`,
`gwdegree`, `gwidegree`, `gwodegree`, `gwb1degree`, `gwb2degree`, `gwb1dsp`
and `gwb2dsp` is estimated along with the other parameters, starting from
the `decay` argument (0.5 by default). The model is then a *curved*
exponential family: two parameters, $\theta$ and the decay $\alpha$,
weight the histogram the term summarizes, such as the counts $e_k$ of ties
with $k$ edgewise shared partners for gwesp:

$$
\theta e^{\alpha} \sum_{k \geq 1} \left[1 - (1 - e^{-\alpha})^k\right] e_k.
$$

The statistics are the counts up to `cutoff` (30 by default, as in ergm, or
the largest possible count if smaller): `esp#1`, `esp#2`... If the cutoff is
below the largest possible count, a last statistic counts everything above it
(`esp#>30`), weighted by the limit $\theta e^{\alpha}$; ergm instead stops
with an error when a network exceeds the cutoff. The parameters are named as
in ergm: `gwesp` and `gwesp.decay`, `gwesp.OTP` and `gwesp.OTP.decay` if
directed, `gwdegree` and `gwdegree.decay`...

## Multilevel terms

MPNet's configurations of two-level networks ([Wang, Robins, Pattison and
Lazega 2013](https://doi.org/10.1016/j.socnet.2013.01.004)), for undirected networks, and below, those of directed networks
from [MPNet's manual](https://static1.squarespace.com/static/57a1436215d5dbbcd2031828/t/5ec68365a155792a0fa03b8f/1590068122536/MPNetManual.pdf). Their first argument is the vertex
attribute with the levels, and `levels=(A, B)` its two values (by default,
the attribute's two values, sorted; vertices with other values are left
out). For a vertex $v$ of level A, $a_v$ is its number of A-ties (ties to A
vertices) and $x_v$ its number of X-ties (to B vertices), and $b_v$, $x_v$
likewise for B; $s^B_{uv}$ is the number of B vertices tied to both $u$ and
$v$, and $g(d) = e^{\alpha}(1 - (1 - e^{-\alpha})^d)$ the geometric weight
of the alternating terms, with `decay` $\alpha$ (MPNet's $\lambda =
e^{\alpha}$; the default `decay=log(2)` is MPNet's $\lambda = 2$). The
decay is fixed by default, as in MPNet; with `fixed=FALSE`, it is
estimated, as for ergm's curved terms (below), for the terms with one
alternating part: the statistics are then the counts of the histogram the
term weights, such as `AXS1A.<attr>#1`, `#2`... up to `cutoff`, and the
parameters `AXS1A.<attr>` and `AXS1A.<attr>.decay`. Each `a` term has a `b`
twin with the levels swapped. See [](user-guide/multilevel.md).

`star2ax(attr)`, `star2bx(attr)`
: $\sum_{v \in A} a_v x_v$: 2-stars of an A-tie and an X-tie. Name:
  `Star2AX.<attr>`.

`axs1a(attr, decay)`, `aas1x(attr, decay)`, `aaaxs(attr, decay)`
: $\sum_{v \in A} a_v\, g(x_v)$, $\sum_{v \in A} g(a_v)\, x_v$ and
  $\sum_{v \in A} g(a_v)\, g(x_v)$: alternating X-stars with one A-tie,
  alternating A-stars with one X-tie, and both alternating. Names:
  `AXS1A.<attr>.<decay>`, `AAS1X...`, `AAAXS...`; the `b` twins are
  `axs1b`, `abs1x` and `abaxs`.

`txax(attr)`, `atxax(attr, decay)`
: $\sum_{\text{A-ties } uv} s^B_{uv}$ and $\sum_{\text{A-ties } uv}
  g(s^B_{uv})$: triangles of an A-tie and two X-ties to a common B vertex,
  and their alternating version (gwesp with the partners in B). Names:
  `TXAX.<attr>`, `ATXAX.<attr>.<decay>`; twins `txbx`, `atxbx`.

`l3xax(attr)`
: $\sum_{\text{A-ties } uv} x_u x_v$: three-paths of an X-tie, an A-tie
  and an X-tie, closed ones (the TXAX triangles) included, as Wang et al.
  (2013) count them ("the TXAX configuration is also part of L3XAX"). Name:
  `L3XAX.<attr>`; twin `l3xbx`.

`l3axb(attr)`
: $\sum_{\text{X-ties } uv,\, u \in A,\, v \in B} a_u b_v$: three-paths of
  an A-tie, an X-tie and a B-tie. Name: `L3AXB.<attr>`.

`c4axb(attr)`
: The 4-cycles of an A-tie, a B-tie and the two X-ties joining their ends,
  $\tfrac12 \operatorname{tr}(A X B X^\top)$. Name: `C4AXB.<attr>`.

`exta(attr)`, `extb(attr)`
: $\sum_{v \in A} t_v x_v$, with $t_v$ the number of A-triangles of $v$: an
  A-triangle with an X-tie at one of its vertices. Names: `EXTA.<attr>`,
  `EXTB.<attr>`.

`asaxasb(attr, decay)`
: $\sum_{\text{X-ties } uv} g(a_u)\, g(b_v)$: alternating A-stars and
  alternating B-stars joined by an X-tie. Name: `ASAXASB.<attr>.<decay>`.

### Directed two-level networks

In a directed network, A-ties and B-ties are the arcs within each level,
and X-ties the arcs from an A vertex to a B vertex: affiliations go from
level A to level B (choose them with `levels=`), and the terms refuse a
network with arcs from B to A. As those dyads carry no affiliation, fix them
with `blocks('level', levels2=2)` (the second mixing type, from B to A). For
$v$ at level S, $\text{in}_v$ and $\text{out}_v$ are its in- and out-degrees
within S and $x_v$ its X-ties; $s_{uv}$ the vertices of the other level that
both $u$ and $v$ are X-tied to. Each A term has a B twin.

| Term | Statistic | MPNet |
|---|---|---|
| `in2starax`, `out2starax` | $\sum_{v \in A} \text{in}_v x_v$, $\sum \text{out}_v x_v$ | In2StarAX, Out2StarAX |
| `axs1ain`, `axs1aout` | $\sum_{v \in A} \text{in}_v\, g(x_v)$, $\sum \text{out}_v\, g(x_v)$ | AXS1Ain, AXS1Aout |
| `aains1x`, `aaouts1x` | $\sum_{v \in A} g(\text{in}_v)\, x_v$, $\sum g(\text{out}_v)\, x_v$ | AAinS1X, AAoutS1X |
| `txaxarc`, `txaxreciprocity` | over A-arcs (reciprocated pairs) $uv$, $s_{uv}$ | TXAXarc, TXAXreciprocity |
| `atxaxarc`, `atxaxreciprocity` | the same with $g(s_{uv})$ | ATXAXarc, ATXAXreciprocity |
| `l3xax`, `l3xaxreciprocity` | over A-arcs (reciprocated pairs) $uv$, $x_u x_v$ | L3XAX, L3XAXreciprocity |
| `l3axbin`, `l3axbout`, `l3axbpath`, `l3bxapath` | over X-ties $a \to b$, $\text{in}_a \text{in}_b$, $\text{out}_a \text{out}_b$, $\text{in}_a \text{out}_b$, $\text{out}_a \text{in}_b$ | L3AXBin, L3AXBout, L3AXBpath, L3BXApath |
| `c4axbentrainment` | 4-cycles of an A-arc $u \to v$, a B-arc $w \to z$ and the X-ties $u \to w$, $v \to z$ | C4AXBentrainment |
| `c4axbexchange` | the same with the X-ties $u \to z$, $v \to w$ | C4AXBexchange |
| `c4axbexchangeareciprocity`, `c4axbexchangebreciprocity`, `c4axbreciprocity` | 4-cycles of a reciprocated A pair and a B-arc, an A-arc and a reciprocated B pair, and both reciprocated | C4AXBexchangeAreciprocity... |
| `ainasxainbs`, `aoutasxaoutbs`, `ainasxaoutbs`, `aoutasxainbs` | over X-ties $a \to b$, $g(\text{in}_a)\, g(\text{in}_b)$ and the other combinations | AinASXAinBS... |

The names are MPNet's labels with the attribute (and decay): `TXAXarc.<attr>`,
`ATXAXarc.<attr>.<decay>`. The manual's drawings of
C4AXBexchangeBreciprocity repeat those of C4AXBreciprocity; ergmx counts
what the labels describe, an A-arc with a reciprocated pair of B-arcs.
MPNet's AC4AXB (alternating four-cycles) is not available.

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

`S(formula, attrs)`
: ergm's subgraph operator: evaluates `formula` on the subgraph induced by
  the vertices that the one-sided R formula `attrs` selects
  (`~level == 'individual'`, `~type`, `~!type`), or, with a two-sided one,
  on the undirected bipartite network of the ties between two disjoint sets
  (`(level == 'A') ~ (level == 'B')`), whose first mode is the left-hand set.
  Each side is an R expression of the vertex attributes, logical or 1-based
  indices (negative ones leave vertices out). Names: `S(<attrs>)~<name>`, the
  attributes as ergm prints them without spaces, such as
  `S(level=="individual")~edges` and
  `S((level=="A"),(level=="B"))~b1star2`.

### Several networks

These operators evaluate their terms on each network of networks combined
with {func}`ergmx.Networks` or {func}`ergmx.NetSeries`, and combine them
through a linear model `lm` of network-level attributes (by default `~1`,
which sums them). Each statistic $g$ of the formula gives, for each column
$c$ of the linear model's design matrix $X$, the statistic
$\sum_k X_{kc}\, g(y_k)$, named `<operator>(<column>)~<name>`, such as
`N(1)~edges` or `N(log(n))~edges`. With curved terms, the statistics are each
network's instead, named `N#<k>~<name>`, as in ergm.multi and tergm. See
[](user-guide/multiple-networks.md) and [](user-guide/temporal.md).

All of them take ergm.multi's `subset` (the networks to use: an expression
of their attributes, logical values or indices), `offset` (an amount added
to every coefficient in each network, an expression or numbers; also
`offset()` terms in `lm`), which adds the statistics `offset1`,
`offset2`... with coefficients fixed at 1, and `label` (a name for the
operator, `N(<label>,<column>)~<name>`, or a function of the statistic's name
and the column).

`N(formula, lm=~1, subset=, offset=, label=)`
: ergm.multi's operator: `formula` on each network.

`Form(formula, lm=~1)`
: tergm's formation: `formula` on the union of the previous and the current
  network of each transition of a {func}`~ergmx.NetSeries`.

`Persist(formula, lm=~1)`
: tergm's persistence: `formula` on the intersection of the previous and the
  current network.

`Diss(formula, lm=~1)`
: tergm's dissolution: `Persist()` with its statistics negated.

`Cross(formula, lm=~1)`
: tergm's cross-section: `formula` on the current network.

`Change(formula, lm=~1)`
: tergm's change: `formula` on the network of the dyads that changed.

## Statistics of tie ages

tergm's durational statistics describe a network together with the ages of
its ties (1 in the time step a tie formed): they are targets of the EGMME
(`tergm(estimate="EGMME", targets=...)`) and monitors of dynamic simulations
(`simulate_dynamic(monitor=...)`), not terms of a model to fit. See
[](user-guide/temporal.md).

`edge.ages`
: Sum over ties of their ages.

`mean.age(emptyval=0, log=FALSE)`
: Mean age of the ties (of their logarithms, `mean.log.age`, with
  `log=TRUE`); `emptyval` without ties.

`edges.ageinterval(from, to=Inf)`
: Number of ties with age in [`from`, `to`), for one or more intervals.

`edgecov.ages(x)`
: Sum over ties of a dyadic covariate times their age.

`nodefactor.mean.age(attr, levels=, emptyval=0, log=FALSE)`
: For each level of `attr`, the mean age of the ties of its vertices (a tie
  between two of them counts twice).

## Proposals

Models with `triangle`, `ttriple`, `ctriple`, `transitive`, `cycle`, or a
shared partner term (also inside `F()` or `N()`) mix half tie/no-tie (TNT) MCMC proposals with
*triadic* proposals, which pick a vertex, one of its neighbors and one of
that neighbor's neighbors, and toggle the tie that would close or open the
triangle. Like ergm's default for these models, this explores clustered
networks much faster. `triadic_weight` in {class}`ergmx.Control` changes the
share. Degree-preserving [constraints](user-guide/constraints.md) use their
own moves, and bipartite networks only tie/no-tie proposals between the modes.
Models of a series of networks (with tergm's operators) also mix in, half
the time, toggles of a dyad that differs from the previous network, as
tergm's discordTNT proposal does.
