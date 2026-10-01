//! Model terms and their change statistics.
//!
//! A term contributes one or more statistics. `change` adds to `out` how its
//! statistics change when the tie i -> j (or i -- j) is toggled in `net`;
//! `sign` is +1 if the tie is being added and -1 if it is being removed.
//! Definitions follow the R package ergm.

use crate::network::{Network, count_common};
use crate::partners::{Bins, Scope, SharedPartners, SpType, Weight};
use crate::space::Space;

pub trait Term: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]);

    /// Adds to `out` the statistics of the empty network (0 for most terms).
    fn empty(&self, _n: u32, _directed: bool, _out: &mut [f64]) {}
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

/// Whether a vertex is counted: every vertex, or those of one mode of a
/// bipartite network.
fn counted(mask: &Option<Vec<bool>>, v: u32) -> bool {
    mask.as_ref().is_none_or(|m| m[v as usize])
}

/// Number of k-stars, for each k: the sum over vertices of C(degree, k), with
/// degrees (kstar), in-degrees (istar) or out-degrees (ostar).
struct Stars {
    ks: Vec<u32>,
    ends: Ends,
    mask: Option<Vec<bool>>,
}

impl Term for Stars {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for &(v, degree) in &degrees[..count] {
            if !counted(&self.mask, v) {
                continue;
            }
            for (stat, &k) in out.iter_mut().zip(&self.ks) {
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
    mask: Option<Vec<bool>>,
}

impl Term for GwDegree {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for &(v, degree) in &degrees[..count] {
            if counted(&self.mask, v) {
                out[0] += sign * self.r.powf(degree_without(degree, sign));
            }
        }
    }
}

/// Number of vertices with each degree in `bins`: degrees (degree),
/// in-degrees (idegree) or out-degrees (odegree), with an overflow bin for
/// curved gwdegree.
struct DegreeCount {
    bins: Bins,
    ends: Ends,
    mask: Option<Vec<bool>>,
}

impl Term for DegreeCount {
    fn n_stats(&self) -> usize {
        self.bins.len()
    }

    fn empty(&self, n: u32, _: bool, out: &mut [f64]) {
        if let Some(s) = self.bins.index(0) {
            out[s] += (0..n).filter(|&v| counted(&self.mask, v)).count() as f64;
        }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for &(v, degree) in &degrees[..count] {
            if counted(&self.mask, v) {
                // Adding: degree -> degree + 1; removing: degree - 1 -> degree, reversed.
                let base = if sign > 0.0 { degree } else { degree - 1 };
                self.bins.shift(base as u32, sign, out);
            }
        }
    }
}

/// Number of vertices with degree 2 or more.
struct Concurrent {
    mask: Option<Vec<bool>>,
}

impl Term for Concurrent {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        for v in [i, j] {
            let degree = net.neighbours(v).len();
            if counted(&self.mask, v) && degree == if sign > 0.0 { 1 } else { 2 } {
                out[0] += sign;
            }
        }
    }
}

/// Number of vertices without ties (in either direction, if directed).
struct Isolates;

impl Term for Isolates {
    fn empty(&self, n: u32, _: bool, out: &mut [f64]) {
        out[0] += n as f64;
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        // Toggling i -> j leaves i and j neighbours if j -> i exists.
        if net.directed() && net.has_edge(j, i) {
            return;
        }
        for v in [i, j] {
            let neighbours = net.neighbours(v).len();
            if sign > 0.0 && neighbours == 0 {
                out[0] -= 1.0;
            } else if sign < 0.0 && neighbours == 1 {
                out[0] += 1.0;
            }
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

/// Number of transitive triads (directed networks): triads of types 030T,
/// 120D, 120U and 300 in Davis and Leinhardt's (1972) census, those with at
/// least one transitive triple and no open two-path. (R's ergm 4.12 documents
/// its `transitive` term so, but computes transitive triples, as `ttriple`.)
///
/// Toggling i -> j only changes the triads {i, j, k} with k tied to i or j;
/// each is classified from its six possible ties with a lookup table.
struct TransitiveTriads {
    /// Whether a triad is transitive, by its ties as bits: i->j, j->i, i->k,
    /// k->i, j->k, k->j.
    table: [bool; 64],
}

impl TransitiveTriads {
    fn new() -> Self {
        // tie(a, b) for the vertices 0 = i, 1 = j, 2 = k.
        let bit = |a: usize, b: usize| match (a, b) {
            (0, 1) => 0,
            (1, 0) => 1,
            (0, 2) => 2,
            (2, 0) => 3,
            (1, 2) => 4,
            (2, 1) => 5,
            _ => unreachable!(),
        };
        let mut table = [false; 64];
        for (mask, transitive) in table.iter_mut().enumerate() {
            let tie = |a, b| mask >> bit(a, b) & 1 == 1;
            let (mut closed, mut open) = (false, false);
            for (x, y, z) in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)] {
                if tie(x, y) && tie(y, z) {
                    if tie(x, z) { closed = true } else { open = true }
                }
            }
            *transitive = closed && !open;
        }
        Self { table }
    }
}

impl Term for TransitiveTriads {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let tied = (net.has_edge(i, j) as usize) | (net.has_edge(j, i) as usize) << 1;
        let mut total = 0i64;
        let mut visit = |k: u32| {
            let rest = (net.has_edge(i, k) as usize) << 2
                | (net.has_edge(k, i) as usize) << 3
                | (net.has_edge(j, k) as usize) << 4
                | (net.has_edge(k, j) as usize) << 5;
            let (with, without) = (rest | tied | 1, rest | (tied & !1));
            total += self.table[with] as i64 - self.table[without] as i64;
        };
        // The vertices tied to i or j, in either direction: a merge of two sorted lists.
        let (ni, nj) = (net.neighbours(i), net.neighbours(j));
        let (mut a, mut b) = (0, 0);
        while a < ni.len() || b < nj.len() {
            let k = match (ni.get(a), nj.get(b)) {
                (Some(&x), Some(&y)) if x == y => {
                    a += 1;
                    b += 1;
                    x
                }
                (Some(&x), Some(&y)) if x < y => {
                    a += 1;
                    x
                }
                (Some(_), Some(&y)) => {
                    b += 1;
                    y
                }
                (Some(&x), None) => {
                    a += 1;
                    x
                }
                (None, Some(&y)) => {
                    b += 1;
                    y
                }
                (None, None) => unreachable!(),
            };
            if k != i && k != j {
                visit(k);
            }
        }
        // `total` is the change from adding i -> j; removing it reverses it.
        out[0] += if sign > 0.0 { total as f64 } else { -(total as f64) };
    }
}

/// Number of 2-paths i -> j -> k with i != k (directed networks; kstar(2) if undirected).
struct TwoPath;

impl Term for TwoPath {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let back = net.has_edge(j, i) as usize;
        let paths = net.out_neighbours(j).len() - back + net.in_neighbours(i).len() - back;
        out[0] += sign * paths as f64;
    }
}

/// Number of pairs with a tie in one direction only (directed networks).
struct Asymmetric;

impl Term for Asymmetric {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += if net.has_edge(j, i) { -sign } else { sign };
    }
}

/// Number of cycles of each length in `ks`: the k-cycles through the tie
/// i -> j (or i -- j) are the simple paths of length k - 1 from j back to i
/// (following ties' directions, if directed).
struct Cycle {
    ks: Vec<u32>,
    longest: u32,
}

impl Cycle {
    fn count_paths(&self, net: &Network, i: u32, path: &mut Vec<u32>, counts: &mut [f64]) {
        let v = *path.last().unwrap();
        let length = path.len() as u32; // edges so far + 1 after the next step
        for &w in net.out_neighbours(v) {
            if w == i {
                counts[length as usize] += 1.0;
            } else if length < self.longest - 1 && !path.contains(&w) {
                path.push(w);
                self.count_paths(net, i, path, counts);
                path.pop();
            }
        }
    }
}

impl Term for Cycle {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        // counts[l]: simple paths of length l from j to i.
        let mut counts = vec![0.0; self.longest as usize];
        self.count_paths(net, i, &mut vec![j], &mut counts);
        for (stat, &k) in out.iter_mut().zip(&self.ks) {
            *stat += sign * counts[k as usize - 1];
        }
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

/// Number of ties by mixing type: for each selected pair of attribute levels
/// (ordered from sender to receiver if directed), the ties between them.
struct NodeMix {
    /// Each vertex's level, or a negative value for levels left out.
    codes: Vec<i64>,
    n_levels: usize,
    /// Statistic of each pair of levels (row-major), or -1 if not counted.
    map: Vec<i64>,
    n_stats: usize,
}

impl Term for NodeMix {
    fn n_stats(&self) -> usize {
        self.n_stats
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (a, b) = (self.codes[i as usize], self.codes[j as usize]);
        if a < 0 || b < 0 {
            return;
        }
        if let Ok(stat) = usize::try_from(self.map[a as usize * self.n_levels + b as usize]) {
            out[stat] += sign;
        }
    }
}

/// For each distinct nonzero absolute difference of a numeric attribute, the
/// number of ties with that difference.
struct AbsDiffCat {
    x: Vec<f64>,
    values: Vec<f64>,
}

impl Term for AbsDiffCat {
    fn n_stats(&self) -> usize {
        self.values.len()
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let d = (self.x[i as usize] - self.x[j as usize]).abs();
        if let Some(s) = self.values.iter().position(|&v| v == d) {
            out[s] += sign;
        }
    }
}

/// b1nodematch (b2nodematch): number of 2-stars centred on second-mode
/// (first-mode) vertices whose two ends have the same attribute value.
struct BipartiteMatch {
    /// Level of each end vertex (-1 for centres).
    codes: Vec<i64>,
}

impl Term for BipartiteMatch {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (end, centre) = if self.codes[i as usize] >= 0 { (i, j) } else { (j, i) };
        let level = self.codes[end as usize];
        let matches = net
            .neighbours(centre)
            .iter()
            .filter(|&&u| u != end && self.codes[u as usize] == level)
            .count();
        out[0] += sign * matches as f64;
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

/// A term as sent from Python: its name, real parameters, integer parameters
/// and, for operators such as F(), the terms it applies to.
#[derive(pyo3::FromPyObject, Clone, Debug)]
pub struct TermSpec(pub String, pub Vec<f64>, pub Vec<i64>, pub Vec<TermSpec>);

/// Splits integer parameters sent as chunks: a length, then that many values.
fn chunks(name: &str, ints: &[i64]) -> Result<Vec<Vec<i64>>, String> {
    let (mut out, mut rest) = (Vec::new(), ints);
    while let Some((&len, tail)) = rest.split_first() {
        let len = usize::try_from(len).map_err(|_| format!("{name}: bad parameters"))?;
        if tail.len() < len {
            return Err(format!("{name}: bad parameters"));
        }
        out.push(tail[..len].to_vec());
        rest = &tail[len..];
    }
    Ok(out)
}

fn build_term(n: usize, directed: bool, spec: &TermSpec) -> Result<Box<dyn Term>, String> {
    let TermSpec(name, reals, ints, _) = spec;
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
    // Chunked parameters: counts, flags and vertex masks.
    let parts = || chunks(name, ints);
    let counts = |values: &[i64], min: i64| -> Result<Vec<u32>, String> {
        if values.is_empty() || values.iter().any(|&k| k < min) {
            return Err(format!("{name}: k must be one or more integers >= {min}"));
        }
        Ok(values.iter().map(|&k| k as u32).collect())
    };
    let mask = |values: &[i64]| -> Result<Option<Vec<bool>>, String> {
        if values.is_empty() {
            return Ok(None);
        }
        expect(values.len(), "vertex mask values", n)?;
        Ok(Some(values.iter().map(|&x| x != 0).collect()))
    };
    let part = |p: &[Vec<i64>], k: usize| -> Result<Vec<i64>, String> {
        p.get(k).cloned().ok_or_else(|| format!("{name}: missing parameters"))
    };
    let sp = |scope: Scope| -> Result<Box<dyn Term>, String> {
        // gw*: reals [decay], chunks [type], [mode mask];
        // histograms: chunks [type], [ks], [overflow], [mode mask].
        let p = parts()?;
        let kind = SpType::from_code(part(&p, 0)?.first().copied().unwrap_or(1), directed)?;
        let (weight, mode) = if reals.is_empty() {
            let bins = Bins::new(&counts(&part(&p, 1)?, 0)?, part(&p, 2)?.first() == Some(&1));
            (Weight::Histogram(bins), mask(&part(&p, 3)?)?)
        } else {
            let d = decay()?;
            (Weight::Geometric { r: r(d), exp_decay: d.exp() }, mask(&part(&p, 1)?)?)
        };
        Ok(Box::new(SharedPartners { kind, scope, weight, mode }))
    };
    let ends_of = |both: &str, head: &str| -> Result<Ends, String> {
        if name == both {
            only(false)?;
            Ok(Ends::Both)
        } else {
            only(true)?;
            Ok(if name == head { Ends::Head } else { Ends::Tail })
        }
    };
    Ok(match name.as_str() {
        "edges" => Box::new(Edges),
        "mutual" => {
            only(true)?;
            Box::new(Mutual)
        }
        // chunks: [ks], [mask]
        "kstar" | "istar" | "ostar" => {
            let ends = ends_of("kstar", "istar")?;
            let p = parts()?;
            Box::new(Stars { ks: counts(&part(&p, 0)?, 1)?, ends, mask: mask(&part(&p, 1)?)? })
        }
        // chunks: [ks], [overflow], [mask]
        "degree" | "idegree" | "odegree" => {
            let ends = ends_of("degree", "idegree")?;
            let p = parts()?;
            let bins = Bins::new(&counts(&part(&p, 0)?, 0)?, part(&p, 1)?.first() == Some(&1));
            Box::new(DegreeCount { bins, ends, mask: mask(&part(&p, 2)?)? })
        }
        "isolates" => Box::new(Isolates),
        // reals [decay]; chunks: [mask]
        "gwdegree" | "gwidegree" | "gwodegree" => {
            let ends = ends_of("gwdegree", "gwidegree")?;
            let p = parts()?;
            Box::new(GwDegree { r: r(decay()?), ends, mask: mask(&part(&p, 0)?)? })
        }
        "concurrent" => {
            only(false)?;
            Box::new(Concurrent { mask: mask(&part(&parts()?, 0)?)? })
        }
        "triangle" if directed => Box::new(DirectedTriangle),
        "triangle" => Box::new(Triangle),
        "ttriple" => {
            only(true)?;
            Box::new(TTriple)
        }
        "transitive" => {
            only(true)?;
            Box::new(TransitiveTriads::new())
        }
        "ctriple" => {
            only(true)?;
            Box::new(CTriple)
        }
        "twopath" => {
            only(true)?;
            Box::new(TwoPath)
        }
        "asymmetric" => {
            only(true)?;
            Box::new(Asymmetric)
        }
        "cycle" => {
            let ks = counts(ints, if directed { 2 } else { 3 })?;
            let longest = ks.iter().copied().max().unwrap();
            Box::new(Cycle { ks, longest })
        }
        "esp" | "gwesp" => sp(Scope::Edgewise)?,
        "dsp" | "gwdsp" => sp(Scope::Dyadwise)?,
        "nsp" | "gwnsp" => sp(Scope::NonEdgewise)?,
        "nodematch" | "nodematchdiff" => {
            expect(ints.len(), "vertex values", n)?;
            Box::new(NodeMatch { codes: ints.clone(), n_levels: levels(), diff: name == "nodematchdiff" })
        }
        "bipartitematch" => {
            only(false)?;
            expect(ints.len(), "vertex values", n)?;
            Box::new(BipartiteMatch { codes: ints.clone() })
        }
        "nodemix" => {
            // ints: the n vertex levels, the number of levels L, then the L x L map.
            if ints.len() < n + 1 {
                return Err("nodemix: expected vertex levels".into());
            }
            let n_levels = ints[n] as usize;
            expect(ints.len(), "values", n + 1 + n_levels * n_levels)?;
            let map = ints[n + 1..].to_vec();
            let n_stats = (map.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize;
            Box::new(NodeMix { codes: ints[..n].to_vec(), n_levels, map, n_stats })
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
        // reals: the n vertex values, then the distinct differences counted
        "absdiffcat" => {
            if reals.len() <= n {
                return Err("absdiffcat: expected vertex values and differences".into());
            }
            Box::new(AbsDiffCat { x: reals[..n].to_vec(), values: reals[n..].to_vec() })
        }
        "edgecov" => {
            expect(reals.len(), "matrix entries", n * n)?;
            Box::new(EdgeCov { x: reals.clone(), n, directed })
        }
        "F" => return Err("F() can't be nested in F()".into()),
        other => return Err(format!("unknown term: {other}")),
    })
}

/// The filter of an F() term: a dyad-independent term with one statistic,
/// whose change statistic for adding a dyad is nonzero (zero, if negated) for
/// the dyads the filtered terms see.
struct Filter {
    term: Box<dyn Term>,
    negate: bool,
}

impl Filter {
    fn passes(&self, net: &Network, i: u32, j: u32) -> bool {
        let mut f = [0.0];
        // Dyad-independent: the change does not depend on the network.
        self.term.change(net, i, j, 1.0, &mut f);
        (f[0] != 0.0) != self.negate
    }
}

enum Entry {
    Plain(Box<dyn Term>),
    /// Terms of F(formula, filter), evaluated on the network of the ties that
    /// pass filter `filter` (the auxiliary network of that index).
    Filtered { terms: Vec<Box<dyn Term>>, filter: usize, n_stats: usize },
}

impl Entry {
    fn n_stats(&self) -> usize {
        match self {
            Entry::Plain(term) => term.n_stats(),
            Entry::Filtered { n_stats, .. } => *n_stats,
        }
    }
}

/// A network with the auxiliary networks that F() terms need: for each filter,
/// the ties that pass it.
#[derive(Clone)]
pub struct State {
    pub net: Network,
    aux: Vec<Network>,
}

/// The terms of a model, with the position of each term's statistics.
pub struct Model {
    entries: Vec<Entry>,
    offsets: Vec<usize>,
    filters: Vec<Filter>,
    n_stats: usize,
}

impl Model {
    pub fn new(n: u32, directed: bool, specs: &[TermSpec]) -> Result<Self, String> {
        let (n_usize, mut entries, mut filters) = (n as usize, Vec::new(), Vec::new());
        for spec in specs {
            if spec.0 == "F" {
                let children = &spec.3;
                let Some((filter, inner)) = children.split_last() else {
                    return Err("F() needs terms and a filter".into());
                };
                let filter_term = build_term(n_usize, directed, filter)?;
                if filter_term.n_stats() != 1 {
                    return Err("the filter of F() must have exactly one statistic".into());
                }
                let terms = inner
                    .iter()
                    .map(|t| build_term(n_usize, directed, t))
                    .collect::<Result<Vec<_>, _>>()?;
                let n_stats = terms.iter().map(|t| t.n_stats()).sum();
                filters.push(Filter { term: filter_term, negate: spec.2.first() == Some(&1) });
                entries.push(Entry::Filtered { terms, filter: filters.len() - 1, n_stats });
            } else {
                entries.push(Entry::Plain(build_term(n_usize, directed, spec)?));
            }
        }
        let mut offsets = Vec::with_capacity(entries.len());
        let mut n_stats = 0;
        for entry in &entries {
            offsets.push(n_stats);
            n_stats += entry.n_stats();
        }
        Ok(Self { entries, offsets, filters, n_stats })
    }

    pub fn n_stats(&self) -> usize {
        self.n_stats
    }

    /// The state of a network: the network and the auxiliary networks of its filters.
    pub fn state(&self, net: Network) -> State {
        let aux = self
            .filters
            .iter()
            .map(|filter| {
                let mut filtered = Network::new(net.n(), net.directed());
                for &(i, j) in net.edges() {
                    if filter.passes(&net, i, j) {
                        filtered.toggle(i, j);
                    }
                }
                filtered
            })
            .collect();
        State { net, aux }
    }

    /// Toggles (i, j) in the network and in the auxiliary networks it belongs to.
    pub fn toggle(&self, state: &mut State, i: u32, j: u32) {
        for (filter, aux) in self.filters.iter().zip(&mut state.aux) {
            if filter.passes(&state.net, i, j) {
                aux.toggle(i, j);
            }
        }
        state.net.toggle(i, j);
    }

    /// Writes to `out` the change in statistics from toggling (i, j).
    pub fn change(&self, state: &State, i: u32, j: u32, out: &mut [f64]) {
        out.fill(0.0);
        let sign = if state.net.has_edge(i, j) { -1.0 } else { 1.0 };
        for (entry, &offset) in self.entries.iter().zip(&self.offsets) {
            let out = &mut out[offset..offset + entry.n_stats()];
            match entry {
                Entry::Plain(term) => term.change(&state.net, i, j, sign, out),
                Entry::Filtered { terms, filter, .. } => {
                    if self.filters[*filter].passes(&state.net, i, j) {
                        let aux = &state.aux[*filter];
                        let mut start = 0;
                        for term in terms {
                            term.change(aux, i, j, sign, &mut out[start..start + term.n_stats()]);
                            start += term.n_stats();
                        }
                    }
                }
            }
        }
    }

    /// Statistics of `net`: those of the empty network, plus the changes from
    /// adding its edges one by one.
    pub fn summary(&self, net: &Network) -> Vec<f64> {
        let mut build = self.state(Network::new(net.n(), net.directed()));
        let mut stats = vec![0.0; self.n_stats];
        for (entry, &offset) in self.entries.iter().zip(&self.offsets) {
            let out = &mut stats[offset..offset + entry.n_stats()];
            match entry {
                Entry::Plain(term) => term.empty(net.n(), net.directed(), out),
                Entry::Filtered { terms, .. } => {
                    let mut start = 0;
                    for term in terms {
                        term.empty(net.n(), net.directed(), &mut out[start..start + term.n_stats()]);
                        start += term.n_stats();
                    }
                }
            }
        }
        let mut delta = vec![0.0; self.n_stats];
        for &(i, j) in net.edges() {
            self.change(&build, i, j, &mut delta);
            self.toggle(&mut build, i, j);
            stats.iter_mut().zip(&delta).for_each(|(s, d)| *s += d);
        }
        stats
    }

    /// Change statistics for adding each free dyad given the rest of `net`,
    /// and whether the dyad is a tie: the data of the MPLE logistic regression.
    pub fn mple_data(&self, net: Network, space: &Space) -> (Vec<f64>, Vec<f64>) {
        let n = net.n();
        let directed = net.directed();
        let state = self.state(net);
        let dyads = space.n_free() as usize;
        let mut x = Vec::with_capacity(dyads * self.n_stats);
        let mut y = Vec::with_capacity(dyads);
        let mut delta = vec![0.0; self.n_stats];
        for i in 0..n {
            let first = if directed { 0 } else { i + 1 };
            for j in (first..n).filter(|&j| j != i && space.is_free(i, j)) {
                self.change(&state, i, j, &mut delta);
                let tie = state.net.has_edge(i, j);
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
