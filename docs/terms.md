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

## Dyadic terms

`edges()` {octicon}`dot-fill`
: Number of ties, $\sum_{D} y_{ij}$. Name: `edges`.

`mutual()`, directed
: Number of reciprocated pairs, $\sum_{i<j} y_{ij} y_{ji}$. Name: `mutual`.

`asymmetric()`, directed
: Number of pairs with a tie in one direction only. Name: `asymmetric`.

`sender()`, `receiver()` {octicon}`dot-fill`, directed
: Each vertex's out-degree (in-degree), one statistic per vertex but the first.
  Names: `sender2`, `sender3`... by vertex position.

`sociality()` {octicon}`dot-fill`, undirected
: Each vertex's degree, one statistic per vertex but the first. Names:
  `sociality2`...

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

`concurrent()`, undirected
: Number of vertices with degree 2 or more. Name: `concurrent`.

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

## Triad terms

`triangle()`
: Number of triangles. In directed networks, ergm counts transitive plus
  cyclic triples, `ttriple + ctriple`. Name: `triangle`.

`ttriple()`, directed
: Number of transitive triples: $i \to j$, $j \to k$ and $i \to k$. Name:
  `ttriple`.

`ctriple()`, directed
: Number of cyclic triples: $i \to j \to k \to i$. Name: `ctriple`.

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

and the type is part of the names: `gwesp.ITP.fixed.0.5`, `esp.OSP1`. In
ergm 4.12.0, the edgewise statistics of type RTP (`esp`, `gwesp`, `nsp`) are
wrong unless its shared-partner cache is turned off
(`term.options = list(cache.sp = FALSE)`), a bug fixed in its development
version; ergmx's match ergm's with the cache off.

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

`absdiffcat(attr)` {octicon}`dot-fill`
: For each distinct nonzero absolute difference $\delta$ of a numeric
  attribute, the number of ties with $|x_i - x_j| = \delta$. Names:
  `absdiff.<attr>.<difference>`.

## Bipartite terms

For [bipartite networks](user-guide/bipartite.md). `b1` terms are about the
first mode, `b2` terms the second; each has a `b2` (or `b1`) twin.

`b1star(k)`
: Number of $k$-stars centred on first-mode vertices. Names: `b1star2`...

`b1degree(d)`
: Number of first-mode vertices with degree exactly $d$. Names: `b1degree0`...

`gwb1degree(decay, fixed=TRUE)`
: Geometrically weighted degree distribution of the first mode. Name:
  `gwb1deg.fixed.<decay>`.

`b1concurrent()`
: Number of first-mode vertices with degree 2 or more. Name: `b1concurrent`.

`b1factor(attr)` {octicon}`dot-fill`
: For each level of `attr` among first-mode vertices but the first, the
  number of their ties. Names: `b1factor.<attr>.<level>`.

`b1cov(attr)` {octicon}`dot-fill`
: Sum over ties of the first-mode endpoint's value of a numeric attribute.
  Name: `b1cov.<attr>`.

`b1nodematch(attr)`
: Number of 2-stars centred on second-mode vertices whose two first-mode
  ends have the same value of `attr` (ergm's default `alpha = beta = 1`).
  Name: `b1nodematch.<attr>`.

`b1dsp(d)`, `gwb1dsp(decay, fixed=TRUE)`
: Pairs of first-mode vertices with exactly $d$ shared partners, and their
  geometrically weighted distribution. Names: `b1dsp0`...,
  `gwb1dsp.fixed.<decay>`.

`edges`, `edgecov` (with a first-mode by second-mode matrix, as in ergm),
`cycle(4)`, `isolates`, `degree`, `nodematch` and the other terms that don't
require a unipartite network also apply.

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
Lazega 2013](https://doi.org/10.1016/j.socnet.2013.01.004)), for undirected networks. Their first argument is the vertex
attribute with the levels, and `levels=(A, B)` its two values (by default,
the attribute's two values, sorted; vertices with other values are left
out). For a vertex $v$ of level A, $a_v$ is its number of A-ties (ties to A
vertices) and $x_v$ its number of X-ties (to B vertices), and $b_v$, $x_v$
likewise for B; $s^B_{uv}$ is the number of B vertices tied to both $u$ and
$v$, and $g(d) = e^{\alpha}(1 - (1 - e^{-\alpha})^d)$ the geometric weight
of the alternating terms, with `decay` $\alpha$ (MPNet's $\lambda =
e^{\alpha}$; the default `decay=log(2)` is MPNet's $\lambda = 2$). Each
`a` term has a `b` twin with the levels swapped. See
[](user-guide/multilevel.md).

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
  and an X-tie, closed ones (the TXAX triangles) included. Name:
  `L3XAX.<attr>`; twin `l3xbx`.

`l3axb(attr)`
: $\sum_{\text{X-ties } uv,\, u \in A,\, v \in B} a_u b_v$: three-paths of
  an A-tie, an X-tie and a B-tie. Name: `L3AXB.<attr>`.

`c4axb(attr)`
: The 4-cycles of an A-tie, a B-tie and the two X-ties joining their ends,
  $\tfrac12 \operatorname{tr}(A X B X^\top)$. Name: `C4AXB.<attr>`.

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

`N(formula, lm=~1)`
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
