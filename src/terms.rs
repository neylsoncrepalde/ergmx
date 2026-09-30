//! Model terms and their change statistics.
//!
//! A term contributes one or more statistics. `change` adds to `out` how its
//! statistics change when the tie i -> j (or i -- j) is toggled in `net`;
//! `sign` is +1 if the tie is being added and -1 if it is being removed.
//! Definitions follow the R package ergm.

use crate::network::{Network, count_common, for_each_common};

pub trait Term: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]);
}

/// Which endpoints of a tie a vertex-level term looks at.
#[derive(Clone, Copy)]
enum Ends {
    /// Both endpoints (undirected networks, or both sides of a directed tie).
    Both,
    /// The sender i of i -> j, with out-degrees.
    Tail,
    /// The receiver j of i -> j, with in-degrees.
    Head,
}

impl Ends {
    /// The endpoints of the tie, with their current degrees on this side.
    fn degrees(self, net: &Network, i: u32, j: u32) -> ([(u32, usize); 2], usize) {
        match self {
            Ends::Both => ([(i, net.neighbours(i).len()), (j, net.neighbours(j).len())], 2),
            Ends::Tail => ([(i, net.out_neighbours(i).len()), (0, 0)], 1),
            Ends::Head => ([(j, net.in_neighbours(j).len()), (0, 0)], 1),
        }
    }

    fn vertices(self, i: u32, j: u32) -> ([u32; 2], usize) {
        match self {
            Ends::Both => ([i, j], 2),
            Ends::Tail => ([i, 0], 1),
            Ends::Head => ([j, 0], 1),
        }
    }
}

/// Degree of a vertex before the tie existed: its current degree if the tie
/// is being added, one less if it is being removed.
fn degree_without(degree: usize, sign: f64) -> f64 {
    if sign > 0.0 { degree as f64 } else { degree as f64 - 1.0 }
}

fn binomial(n: f64, k: u32) -> f64 {
    (0..k).fold(1.0, |acc, t| acc * (n - t as f64) / (t as f64 + 1.0))
}

// -- Dyadic and reciprocity terms ------------------------------------------------

struct Edges;

impl Term for Edges {
    fn change(&self, _: &Network, _: u32, _: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign;
    }
}

/// Number of reciprocated pairs of ties (directed networks).
struct Mutual;

impl Term for Mutual {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        if net.has_edge(j, i) {
            out[0] += sign;
        }
    }
}

// -- Degree terms ------------------------------------------------------------------

/// Number of k-stars, for each k: the sum over vertices of C(degree, k), with
/// degrees (kstar), in-degrees (istar) or out-degrees (ostar).
struct Stars {
    ks: Vec<u32>,
    ends: Ends,
}

impl Term for Stars {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for (stat, &k) in out.iter_mut().zip(&self.ks) {
            for &(_, degree) in &degrees[..count] {
                // A vertex of degree d gains C(d, k - 1) k-stars with a new tie.
                *stat += sign * binomial(degree_without(degree, sign), k - 1);
            }
        }
    }
}

/// Geometrically weighted degree with a fixed decay:
///   exp(decay) * sum over vertices of 1 - r^degree, with r = 1 - exp(-decay).
/// A new tie on a vertex of degree d adds exp(decay) * r^d * (1 - r) = r^d.
struct GwDegree {
    r: f64,
    ends: Ends,
}

impl Term for GwDegree {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for &(_, degree) in &degrees[..count] {
            out[0] += sign * self.r.powf(degree_without(degree, sign));
        }
    }
}

// -- Triad terms --------------------------------------------------------------------

/// Number of triangles (undirected networks).
struct Triangle;

impl Term for Triangle {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * count_common(net.neighbours(i), net.neighbours(j)) as f64;
    }
}

/// Number of transitive triples a -> b -> c with a -> c (directed networks).
/// The tie i -> j can be the shortcut a -> c, the first step a -> b or the
/// second step b -> c.
fn transitive_triples(net: &Network, i: u32, j: u32) -> u32 {
    let (out_i, in_i) = (net.out_neighbours(i), net.in_neighbours(i));
    let (out_j, in_j) = (net.out_neighbours(j), net.in_neighbours(j));
    count_common(out_i, in_j) + count_common(out_i, out_j) + count_common(in_i, in_j)
}

/// Number of cyclic triples a -> b -> c -> a (directed networks).
fn cyclic_triples(net: &Network, i: u32, j: u32) -> u32 {
    count_common(net.out_neighbours(j), net.in_neighbours(i))
}

struct TTriple;

impl Term for TTriple {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * transitive_triples(net, i, j) as f64;
    }
}

struct CTriple;

impl Term for CTriple {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * cyclic_triples(net, i, j) as f64;
    }
}

/// In directed networks, ergm counts triangles as transitive plus cyclic triples.
struct DirectedTriangle;

impl Term for DirectedTriangle {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * (transitive_triples(net, i, j) + cyclic_triples(net, i, j)) as f64;
    }
}

/// Geometrically weighted edgewise shared partners with a fixed decay
/// (undirected networks):
///
///   gwesp = exp(decay) * sum over edges (a, b) of 1 - r^sp(a, b),
///
/// with r = 1 - exp(-decay) and sp(a, b) the number of shared partners of a
/// and b. Toggling (i, j) changes the term for (i, j) itself and, for every
/// shared partner u, adds or removes a shared partner of (i, u) and (j, u);
/// each of those contributes r^sp, with sp counted without the tie (i, j).
struct Gwesp {
    exp_decay: f64,
    r: f64,
}

impl Term for Gwesp {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (ni, nj) = (net.neighbours(i), net.neighbours(j));
        // When removing, j is a shared partner of (i, u) and i one of (j, u).
        let own = if sign < 0.0 { 1 } else { 0 };
        let mut total = self.exp_decay * (1.0 - self.r.powi(count_common(ni, nj) as i32));
        for_each_common(ni, nj, |u| {
            let nu = net.neighbours(u);
            total += self.r.powi((count_common(ni, nu) - own) as i32);
            total += self.r.powi((count_common(nj, nu) - own) as i32);
        });
        out[0] += sign * total;
    }
}

/// gwesp for directed networks, with ergm's default outgoing two-path (OTP)
/// shared partners: k is a shared partner of the tie a -> b if a -> k -> b.
///
/// Toggling i -> j changes the term for i -> j itself; adds or removes the
/// partner j of every tie i -> v with j -> v; and the partner i of every tie
/// u -> j with u -> i.
struct GwespOtp {
    exp_decay: f64,
    r: f64,
}

impl Term for GwespOtp {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (out_i, in_i) = (net.out_neighbours(i), net.in_neighbours(i));
        let (out_j, in_j) = (net.out_neighbours(j), net.in_neighbours(j));
        let own = if sign < 0.0 { 1 } else { 0 };
        let mut total = self.exp_decay * (1.0 - self.r.powi(count_common(out_i, in_j) as i32));
        for_each_common(out_i, out_j, |v| {
            total += self.r.powi((count_common(out_i, net.in_neighbours(v)) - own) as i32);
        });
        for_each_common(in_i, in_j, |u| {
            total += self.r.powi((count_common(net.out_neighbours(u), in_j) - own) as i32);
        });
        out[0] += sign * total;
    }
}

/// Geometrically weighted dyadwise shared partners with a fixed decay
/// (undirected networks): like gwesp, but over all pairs of vertices, tied or
/// not. Toggling (i, j) adds or removes the partner j of every pair (i, u)
/// with u a neighbour of j, and the partner i of every pair (j, u) with u a
/// neighbour of i.
struct Gwdsp {
    r: f64,
}

impl Term for Gwdsp {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (ni, nj) = (net.neighbours(i), net.neighbours(j));
        let own = if sign < 0.0 { 1 } else { 0 };
        let mut total = 0.0;
        // Pairs (a, u) gaining or losing the partner b, for (a, b) = (i, j) and (j, i).
        for (a, na, nb) in [(i, ni, nj), (j, nj, ni)] {
            for &u in nb.iter().filter(|&&u| u != a) {
                total += self.r.powi((count_common(na, net.neighbours(u)) - own) as i32);
            }
        }
        out[0] += sign * total;
    }
}

// -- Vertex and dyad attribute terms -----------------------------------------------------

/// Number of ties between vertices with the same attribute value; with
/// `diff`, one statistic per value.
struct NodeMatch {
    codes: Vec<i64>,
    n_levels: usize,
    diff: bool,
}

impl Term for NodeMatch {
    fn n_stats(&self) -> usize {
        if self.diff { self.n_levels } else { 1 }
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (a, b) = (self.codes[i as usize], self.codes[j as usize]);
        if a == b {
            out[if self.diff { a as usize } else { 0 }] += sign;
        }
    }
}

/// For each attribute level, the number of tie endpoints at that level.
/// Vertices with a negative code (the base level) are not counted.
struct NodeFactor {
    codes: Vec<i64>,
    n_levels: usize,
    ends: Ends,
}

impl Term for NodeFactor {
    fn n_stats(&self) -> usize {
        self.n_levels
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (vertices, count) = self.ends.vertices(i, j);
        for &v in &vertices[..count] {
            if let Ok(level) = usize::try_from(self.codes[v as usize]) {
                out[level] += sign;
            }
        }
    }
}

/// Sum over ties of the attribute values of the endpoints.
struct NodeCov {
    x: Vec<f64>,
    ends: Ends,
}

impl Term for NodeCov {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (vertices, count) = self.ends.vertices(i, j);
        out[0] += sign * vertices[..count].iter().map(|&v| self.x[v as usize]).sum::<f64>();
    }
}

/// Sum over ties of the absolute difference of the endpoints' attribute values.
struct AbsDiff {
    x: Vec<f64>,
}

impl Term for AbsDiff {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * (self.x[i as usize] - self.x[j as usize]).abs();
    }
}

/// Sum over ties of a dyadic covariate x[i][j]. Undirected networks use the
/// upper triangle, x[min(i, j)][max(i, j)], as ergm does.
struct EdgeCov {
    x: Vec<f64>,
    n: usize,
    directed: bool,
}

impl Term for EdgeCov {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (a, b) = if self.directed || i < j { (i, j) } else { (j, i) };
        out[0] += sign * self.x[a as usize * self.n + b as usize];
    }
}

// -- Building a model from the Python specification ----------------------------------------------

/// A term as sent from Python: its name, real parameters and integer parameters.
pub type TermSpec = (String, Vec<f64>, Vec<i64>);

fn build_term(n: usize, directed: bool, spec: &TermSpec) -> Result<Box<dyn Term>, String> {
    let (name, reals, ints) = spec;
    let expect = |len: usize, what: &str, want: usize| {
        if len == want { Ok(()) } else { Err(format!("{name}: expected {want} {what}, got {len}")) }
    };
    let only = |needs_directed: bool| {
        if directed == needs_directed {
            Ok(())
        } else {
            let kind = if needs_directed { "directed" } else { "undirected" };
            Err(format!("{name} is only implemented for {kind} networks"))
        }
    };
    let decay = || reals.first().copied().ok_or_else(|| format!("{name}: missing decay"));
    let r = |decay: f64| 1.0 - (-decay).exp();
    let levels = || (ints.iter().copied().max().unwrap_or(-1).max(-1) + 1) as usize;
    let ks = || -> Result<Vec<u32>, String> {
        if ints.is_empty() || ints.iter().any(|&k| k < 1) {
            return Err(format!("{name}: k must be one or more integers >= 1"));
        }
        Ok(ints.iter().map(|&k| k as u32).collect())
    };
    Ok(match name.as_str() {
        "edges" => Box::new(Edges),
        "mutual" => {
            only(true)?;
            Box::new(Mutual)
        }
        "kstar" => {
            only(false)?;
            Box::new(Stars { ks: ks()?, ends: Ends::Both })
        }
        "istar" | "ostar" => {
            only(true)?;
            let ends = if name == "istar" { Ends::Head } else { Ends::Tail };
            Box::new(Stars { ks: ks()?, ends })
        }
        "gwdegree" => {
            only(false)?;
            Box::new(GwDegree { r: r(decay()?), ends: Ends::Both })
        }
        "gwidegree" | "gwodegree" => {
            only(true)?;
            let ends = if name == "gwidegree" { Ends::Head } else { Ends::Tail };
            Box::new(GwDegree { r: r(decay()?), ends })
        }
        "triangle" if directed => Box::new(DirectedTriangle),
        "triangle" => Box::new(Triangle),
        "ttriple" => {
            only(true)?;
            Box::new(TTriple)
        }
        "ctriple" => {
            only(true)?;
            Box::new(CTriple)
        }
        "gwesp" => {
            let d = decay()?;
            if directed {
                Box::new(GwespOtp { exp_decay: d.exp(), r: r(d) })
            } else {
                Box::new(Gwesp { exp_decay: d.exp(), r: r(d) })
            }
        }
        "gwdsp" => {
            only(false)?;
            Box::new(Gwdsp { r: r(decay()?) })
        }
        "nodematch" | "nodematchdiff" => {
            expect(ints.len(), "vertex values", n)?;
            Box::new(NodeMatch { codes: ints.clone(), n_levels: levels(), diff: name == "nodematchdiff" })
        }
        "nodefactor" | "nodeifactor" | "nodeofactor" => {
            expect(ints.len(), "vertex values", n)?;
            let ends = match name.as_str() {
                "nodeifactor" => Ends::Head,
                "nodeofactor" => Ends::Tail,
                _ => Ends::Both,
            };
            if !matches!(ends, Ends::Both) {
                only(true)?;
            }
            Box::new(NodeFactor { codes: ints.clone(), n_levels: levels(), ends })
        }
        "nodecov" | "nodeicov" | "nodeocov" => {
            expect(reals.len(), "vertex values", n)?;
            let ends = match name.as_str() {
                "nodeicov" => Ends::Head,
                "nodeocov" => Ends::Tail,
                _ => Ends::Both,
            };
            if !matches!(ends, Ends::Both) {
                only(true)?;
            }
            Box::new(NodeCov { x: reals.clone(), ends })
        }
        "absdiff" => {
            expect(reals.len(), "vertex values", n)?;
            Box::new(AbsDiff { x: reals.clone() })
        }
        "edgecov" => {
            expect(reals.len(), "matrix entries", n * n)?;
            Box::new(EdgeCov { x: reals.clone(), n, directed })
        }
        other => return Err(format!("unknown term: {other}")),
    })
}

/// The terms of a model, with the position of each term's statistics.
pub struct Model {
    terms: Vec<Box<dyn Term>>,
    offsets: Vec<usize>,
    n_stats: usize,
}

impl Model {
    pub fn new(n: u32, directed: bool, specs: &[TermSpec]) -> Result<Self, String> {
        let terms = specs
            .iter()
            .map(|spec| build_term(n as usize, directed, spec))
            .collect::<Result<Vec<_>, _>>()?;
        let mut offsets = Vec::with_capacity(terms.len());
        let mut n_stats = 0;
        for term in &terms {
            offsets.push(n_stats);
            n_stats += term.n_stats();
        }
        Ok(Self { terms, offsets, n_stats })
    }

    pub fn n_stats(&self) -> usize {
        self.n_stats
    }

    /// Writes to `out` the change in statistics from toggling (i, j).
    pub fn change(&self, net: &Network, i: u32, j: u32, out: &mut [f64]) {
        out.fill(0.0);
        let sign = if net.has_edge(i, j) { -1.0 } else { 1.0 };
        for (term, &offset) in self.terms.iter().zip(&self.offsets) {
            term.change(net, i, j, sign, &mut out[offset..offset + term.n_stats()]);
        }
    }

    /// Statistics of `net`: the sum of the changes from adding its edges one by one.
    pub fn summary(&self, net: &Network) -> Vec<f64> {
        let mut build = Network::new(net.n(), net.directed());
        let mut stats = vec![0.0; self.n_stats];
        let mut delta = vec![0.0; self.n_stats];
        for &(i, j) in net.edges() {
            self.change(&build, i, j, &mut delta);
            build.toggle(i, j);
            stats.iter_mut().zip(&delta).for_each(|(s, d)| *s += d);
        }
        stats
    }

    /// Change statistics for adding each dyad given the rest of `net`, and
    /// whether the dyad is a tie: the data of the MPLE logistic regression.
    pub fn mple_data(&self, net: &Network) -> (Vec<f64>, Vec<f64>) {
        let n = net.n();
        let dyads = net.n_dyads() as usize;
        let mut x = Vec::with_capacity(dyads * self.n_stats);
        let mut y = Vec::with_capacity(dyads);
        let mut delta = vec![0.0; self.n_stats];
        for i in 0..n {
            let first = if net.directed() { 0 } else { i + 1 };
            for j in (first..n).filter(|&j| j != i) {
                self.change(net, i, j, &mut delta);
                let tie = net.has_edge(i, j);
                if tie {
                    delta.iter_mut().for_each(|d| *d = -*d);
                }
                x.extend_from_slice(&delta);
                y.push(tie as u8 as f64);
            }
        }
        (x, y)
    }
}
