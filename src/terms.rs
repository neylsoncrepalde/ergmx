//! Model terms and their change statistics.
//!
//! A term contributes one or more statistics. `change` adds to `out` how its
//! statistics change when the tie i -> j (or i -- j) is toggled in `net`;
//! `sign` is +1 if the tie is being added and -1 if it is being removed.
//! Definitions follow the R package ergm.

use rustc_hash::FxHashMap;

use crate::network::{Network, Partners, count_common};
use crate::partners::{Bins, Scope, SharedPartners, SpType, Weight};
use crate::space::Space;

pub trait Term: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]);

    /// Adds to `out` the statistics of the empty network (0 for most terms).
    fn empty(&self, _n: u32, _directed: bool, _out: &mut [f64]) {}

    /// The shared partner counts the term reads, which networks can keep.
    fn partners(&self) -> Option<Partners> {
        None
    }
}

/// Which endpoints of a tie a vertex-level term looks at.
#[derive(Clone, Copy)]
pub(crate) enum Ends {
    /// Both endpoints (undirected networks, or both sides of a directed tie).
    Both,
    /// The sender i of i -> j, with out-degrees.
    Tail,
    /// The receiver j of i -> j, with in-degrees.
    Head,
}

impl Ends {
    /// The endpoints of the tie, with their current degrees on this side.
    pub(crate) fn degrees(self, net: &Network, i: u32, j: u32) -> ([(u32, usize); 2], usize) {
        match self {
            Ends::Both => ([(i, net.neighbours(i).len()), (j, net.neighbours(j).len())], 2),
            Ends::Tail => ([(i, net.out_neighbours(i).len()), (0, 0)], 1),
            Ends::Head => ([(j, net.in_neighbours(j).len()), (0, 0)], 1),
        }
    }

    pub(crate) fn vertices(self, i: u32, j: u32) -> ([u32; 2], usize) {
        match self {
            Ends::Both => ([i, j], 2),
            Ends::Tail => ([i, 0], 1),
            Ends::Head => ([j, 0], 1),
        }
    }
}

/// Degree of a vertex before the tie existed: its current degree if the tie
/// is being added, one less if it is being removed.
pub(crate) fn degree_without(degree: usize, sign: f64) -> f64 {
    if sign > 0.0 { degree as f64 } else { degree as f64 - 1.0 }
}

pub(crate) fn binomial(n: f64, k: u32) -> f64 {
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
pub(crate) fn counted(mask: &Option<Vec<bool>>, v: u32) -> bool {
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
        if a == b && a >= 0 {
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

// -- Multilevel terms (MPNet's) ----------------------------------------------------------------

/// MPNet's configurations of two-level networks (Wang, Robins, Pattison and
/// Lazega 2013), in one undirected network whose vertices are at level A,
/// level B or neither: A-ties within A, B-ties within B, X-ties between them.
/// For a level S (A or B, the term's side) and the other level O, a vertex v
/// of S has within-level degree w(v) and cross-level degree x(v); g(d) =
/// exp(decay) (1 - r^d), with r = 1 - exp(-decay) (MPNet's lambda = exp(decay)),
/// is the geometric weighting of alternating statistics.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    /// Star2SX: sum over v in S of w(v) x(v).
    Star2X,
    /// SXS1S (AXS1A): sum of w(v) g(x(v)), alternating X-stars with one S-tie.
    XStarsWithin,
    /// SSS1X (AAS1X): sum of g(w(v)) x(v), alternating S-stars with one X-tie.
    WithinStarsX,
    /// SSSXS (AAAXS): sum of g(w(v)) g(x(v)).
    AlternatingStars,
    /// TXSX: over S-ties, the number of shared partners at level O.
    TriangleX,
    /// ATXSX: over S-ties, g(shared partners at level O).
    AltTriangleX,
    /// L3XSX: over S-ties (v, u), x(v) x(u) (paths X-S-X, closed ones included).
    PathXSX,
    /// L3AXB: over X-ties (a, b), w(a) w(b).
    PathAXB,
    /// C4AXB: the 4-cycles of an A-tie, a B-tie and two X-ties.
    CycleAXB,
}

struct Multilevel {
    kind: Level,
    /// 0 for level A, 1 for B, -1 for neither.
    level: Vec<i8>,
    /// The term's side: 0 (A) or 1 (B).
    side: i8,
    r: f64,
    exp_decay: f64,
}

impl Multilevel {
    #[inline]
    fn at(&self, v: u32) -> i8 {
        self.level[v as usize]
    }

    /// Neighbours of v at level `l`.
    fn count(&self, net: &Network, v: u32, l: i8) -> usize {
        net.neighbours(v).iter().filter(|&&u| self.at(u) == l).count()
    }

    fn g(&self, d: usize) -> f64 {
        self.exp_decay * (1.0 - self.r.powi(d as i32))
    }

    /// Partners of v and u at level `l`.
    fn shared(&self, net: &Network, v: u32, u: u32, l: i8) -> usize {
        let mut c = 0;
        crate::network::for_each_common(net.neighbours(v), net.neighbours(u), |k| {
            if self.at(k) == l {
                c += 1
            }
        });
        c
    }

    /// The change from adding the tie (i, j) (with `removing`, from adding it
    /// back after its removal: the change of removing it, reversed).
    fn added(&self, net: &Network, i: u32, j: u32, removing: bool) -> f64 {
        let (li, lj) = (self.at(i), self.at(j));
        if li < 0 || lj < 0 {
            return 0.0;
        }
        let s = self.side;
        let o = 1 - s;
        // Degrees and counts without the tie (i, j).
        let without = |d: usize| if removing { d - 1 } else { d };
        match self.kind {
            Level::PathAXB | Level::CycleAXB => self.cross_level(net, i, j),
            _ if li == s && lj == s => {
                // An S-tie.
                let (xi, xj) = (self.count(net, i, o), self.count(net, j, o));
                match self.kind {
                    Level::Star2X => (xi + xj) as f64,
                    Level::XStarsWithin => self.g(xi) + self.g(xj),
                    Level::WithinStarsX | Level::AlternatingStars => [(i, xi), (j, xj)]
                        .iter()
                        .map(|&(v, x)| {
                            let weight = self.r.powi(without(self.count(net, v, s)) as i32);
                            weight * if self.kind == Level::WithinStarsX { x as f64 } else { self.g(x) }
                        })
                        .sum(),
                    Level::TriangleX => self.shared(net, i, j, o) as f64,
                    Level::AltTriangleX => self.g(self.shared(net, i, j, o)),
                    Level::PathXSX => (xi * xj) as f64,
                    Level::PathAXB | Level::CycleAXB => unreachable!(),
                }
            }
            _ if li != lj => {
                // An X-tie: a at level S, b at level O.
                let (a, b) = if li == s { (i, j) } else { (j, i) };
                let w = self.count(net, a, s);
                match self.kind {
                    Level::Star2X => w as f64,
                    Level::XStarsWithin => w as f64 * self.r.powi(without(self.count(net, a, o)) as i32),
                    Level::WithinStarsX => self.g(w),
                    Level::AlternatingStars => self.g(w) * self.r.powi(without(self.count(net, a, o)) as i32),
                    Level::TriangleX => self.shared(net, a, b, s) as f64,
                    Level::AltTriangleX => {
                        // S-ties (a, u) with u also tied to b gain a partner.
                        let mut total = 0.0;
                        crate::network::for_each_common(net.neighbours(a), net.neighbours(b), |u| {
                            if self.at(u) == s {
                                total += self.r.powi(without(self.shared(net, a, u, o)) as i32);
                            }
                        });
                        total
                    }
                    Level::PathXSX => net
                        .neighbours(a)
                        .iter()
                        .filter(|&&u| self.at(u) == s)
                        .map(|&u| self.count(net, u, o))
                        .sum::<usize>() as f64,
                    Level::PathAXB | Level::CycleAXB => unreachable!(),
                }
            }
            _ => 0.0,
        }
    }

    /// L3AXB and C4AXB, which involve ties of both levels (none of their
    /// counts depends on the tie (i, j) itself).
    fn cross_level(&self, net: &Network, i: u32, j: u32) -> f64 {
        let (li, lj) = (self.at(i), self.at(j));
        if li != lj {
            // An X-tie (a, b), a in A, b in B.
            let (a, b) = if li == 0 { (i, j) } else { (j, i) };
            return match self.kind {
                Level::PathAXB => (self.count(net, a, 0) * self.count(net, b, 1)) as f64,
                // A-ties (a, a') and B-ties (b, b') with a' tied to b'.
                _ => net
                    .neighbours(a)
                    .iter()
                    .filter(|&&u| self.at(u) == 0)
                    .map(|&u| {
                        let mut c = 0;
                        crate::network::for_each_common(net.neighbours(u), net.neighbours(b), |k| {
                            if self.at(k) == 1 {
                                c += 1
                            }
                        });
                        c
                    })
                    .sum::<usize>() as f64,
            };
        }
        // A tie within a level: its endpoints' partners at the other level m.
        let m = 1 - li;
        match self.kind {
            // Each X-tie (v, k) of an endpoint v gains one in w(v).
            Level::PathAXB => [i, j]
                .iter()
                .map(|&v| {
                    net.neighbours(v).iter().filter(|&&k| self.at(k) == m).map(|&k| self.count(net, k, m)).sum::<usize>()
                })
                .sum::<usize>() as f64,
            // m-ties (k, k') with k tied to i and k' tied to j.
            _ => net
                .neighbours(i)
                .iter()
                .filter(|&&k| self.at(k) == m)
                .map(|&k| {
                    let mut c = 0;
                    crate::network::for_each_common(net.neighbours(k), net.neighbours(j), |u| {
                        if self.at(u) == m {
                            c += 1
                        }
                    });
                    c
                })
                .sum::<usize>() as f64,
        }
    }
}

impl Term for Multilevel {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        // Removing a tie reverses adding it to the network without it, whose
        // counts the change computes with the tie discounted.
        out[0] += sign * self.added(net, i, j, sign < 0.0);
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

pub(crate) fn build_term(n: usize, directed: bool, spec: &TermSpec) -> Result<Box<dyn Term>, String> {
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
            (Weight::geometric(d, r(d), n), mask(&part(&p, 1)?)?)
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
        // ints: each vertex's level (0 A, 1 B, -1 neither); reals: [decay] if alternating.
        "star2ax" | "star2bx" | "axs1a" | "axs1b" | "aas1x" | "abs1x" | "aaaxs" | "abaxs" | "txax" | "txbx"
        | "atxax" | "atxbx" | "l3xax" | "l3xbx" | "l3axb" | "c4axb" => {
            only(false)?;
            expect(ints.len(), "vertex levels", n)?;
            let (kind, side) = match name.as_str() {
                "star2ax" => (Level::Star2X, 0),
                "star2bx" => (Level::Star2X, 1),
                "axs1a" => (Level::XStarsWithin, 0),
                "axs1b" => (Level::XStarsWithin, 1),
                "aas1x" => (Level::WithinStarsX, 0),
                "abs1x" => (Level::WithinStarsX, 1),
                "aaaxs" => (Level::AlternatingStars, 0),
                "abaxs" => (Level::AlternatingStars, 1),
                "txax" => (Level::TriangleX, 0),
                "txbx" => (Level::TriangleX, 1),
                "atxax" => (Level::AltTriangleX, 0),
                "atxbx" => (Level::AltTriangleX, 1),
                "l3xax" => (Level::PathXSX, 0),
                "l3xbx" => (Level::PathXSX, 1),
                "l3axb" => (Level::PathAXB, 0),
                _ => (Level::CycleAXB, 0),
            };
            let alternating = matches!(kind, Level::XStarsWithin | Level::WithinStarsX | Level::AlternatingStars | Level::AltTriangleX);
            let d = if alternating { decay()? } else { 0.0 };
            if ints.iter().any(|&l| !(-1..=1).contains(&l)) {
                return Err(format!("{name}: levels must be 0, 1 or -1"));
            }
            Box::new(Multilevel { kind, level: ints.iter().map(|&l| l as i8).collect(), side, r: r(d), exp_decay: d.exp() })
        }
        "F" => return Err("F() can't be nested in F()".into()),
        other => {
            let child = |s: &TermSpec| build_term(n, directed, s);
            if let Some(term) = crate::vocab::build(n, directed, spec, &child) {
                return term;
            }
            return crate::multilevel::build(n, directed, other, reals, ints)
                .unwrap_or_else(|| Err(format!("unknown term: {other}")));
        }
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

/// How an operator on the blocks of a combined network turns a block's
/// network y into the network its terms see, given the block's previous
/// network p (tergm's operators): y itself (ergm.multi's N(), Cross()), the
/// union y | p (Form()), the intersection y & p (Persist(), Diss()), or the
/// dyads that changed, y ^ p (Change()).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum View {
    Same,
    Union,
    Intersection,
    Change,
}

impl View {
    fn from_code(code: i64) -> Result<Self, String> {
        Ok(match code {
            0 => View::Same,
            1 => View::Union,
            2 => View::Intersection,
            3 => View::Change,
            _ => return Err(format!("unknown network view {code}")),
        })
    }

    /// Whether toggling the dyad (i, j) of y toggles it in the view.
    #[inline]
    fn sees(self, prev: Option<&Network>, i: u32, j: u32) -> bool {
        match self {
            View::Same | View::Change => true,
            View::Union => prev.is_some_and(|p| !p.has_edge(i, j)),
            View::Intersection => prev.is_some_and(|p| p.has_edge(i, j)),
        }
    }

    /// The view of y.
    fn of(self, y: &Network, prev: Option<&Network>) -> Network {
        let empty = Network::new(y.n(), y.directed());
        let prev = prev.unwrap_or(&empty);
        match self {
            View::Same => y.clone(),
            View::Union | View::Change => {
                let mut view = prev.clone();
                for &(i, j) in y.edges() {
                    // Union: add y's ties; change: toggle them, leaving y ^ p.
                    if self == View::Change || !view.has_edge(i, j) {
                        view.toggle(i, j);
                    }
                }
                view
            }
            View::Intersection => {
                let mut view = empty.clone();
                for &(i, j) in y.edges().iter().filter(|&&(i, j)| prev.has_edge(i, j)) {
                    view.toggle(i, j);
                }
                view
            }
        }
    }
}

/// The blocks of a combined network (ergm.multi's Networks(), tergm's
/// NetSeries()): the vertices start..start + size of each, with no ties
/// between blocks.
pub struct Layout {
    starts: Vec<u32>,
    sizes: Vec<u32>,
}

impl Layout {
    pub fn new(blocks: &[(u32, u32)], n: u32) -> Result<Self, String> {
        let mut end = 0;
        for &(start, size) in blocks {
            if start < end || size == 0 || start as u64 + size as u64 > n as u64 {
                return Err("blocks must be disjoint, ordered ranges of the vertices".into());
            }
            end = start + size;
        }
        Ok(Self { starts: blocks.iter().map(|b| b.0).collect(), sizes: blocks.iter().map(|b| b.1).collect() })
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn size(&self, k: usize) -> u32 {
        self.sizes[k]
    }

    /// The block of the dyad (i, j) and its vertices' numbers in the block, if
    /// both are in one block.
    #[inline]
    fn locate(&self, i: u32, j: u32) -> Option<(usize, u32, u32)> {
        let k = self.starts.partition_point(|&s| s <= i).checked_sub(1)?;
        let (s, n) = (self.starts[k], self.sizes[k]);
        (i < s + n && j >= s && j < s + n).then(|| (k, i - s, j - s))
    }

    /// Each block's network in `net`, with the vertices numbered within the block.
    pub fn split(&self, net: &Network) -> Vec<Network> {
        let mut parts: Vec<Network> = self.sizes.iter().map(|&n| Network::new(n, net.directed())).collect();
        for &(i, j) in net.edges() {
            if let Some((k, a, b)) = self.locate(i, j) {
                parts[k].toggle(a, b);
            }
        }
        parts
    }
}

/// The terms of a block operator for one block: a model of the block's
/// network, with the block's row of the operator's linear model.
struct SubBlock {
    model: Model,
    design: Vec<f64>,
    /// Position of the block's statistics, when they are kept apart.
    offset: usize,
}

enum Entry {
    Plain(Box<dyn Term>),
    /// Terms of F(formula, filter), evaluated on the network of the ties that
    /// pass filter `filter` (the auxiliary network of that index).
    Filtered { terms: Vec<Box<dyn Term>>, filter: usize, n_stats: usize },
    /// Terms evaluated on a view of each block of a combined network, as
    /// ergm.multi's N() and tergm's Form(), Persist(), Diss(), Cross() and
    /// Change(). The statistics are, with `compact`, sums over the blocks of
    /// each statistic times each column of the operator's linear model (q
    /// columns, statistic-major), or else each block's statistics in turn
    /// (for curved terms, whose coefficients differ between blocks). `scale`
    /// is -1 for Diss(), which negates Persist().
    Blocks { view: View, scale: f64, slot: usize, blocks: Vec<SubBlock>, compact: bool, q: usize, n_stats: usize },
    /// Terms of ergm's S(formula, attrs), evaluated on the subgraph induced by
    /// some vertices, or on the bipartite subgraph between two disjoint sets
    /// of vertices: `local` numbers the subgraph's vertices (u32::MAX for the
    /// others) and, if bipartite, `head` marks those of the second set. In a
    /// directed network (`oriented`), the bipartite subgraph has the arcs from
    /// the first set to the second, as an undirected network (`sub_directed` false).
    Subgraph { slot: usize, local: Vec<u32>, head: Option<Vec<bool>>, oriented: bool, sub_directed: bool, model: Box<Model> },
}

impl Entry {
    fn n_stats(&self) -> usize {
        match self {
            Entry::Plain(term) => term.n_stats(),
            Entry::Filtered { n_stats, .. } | Entry::Blocks { n_stats, .. } => *n_stats,
            Entry::Subgraph { model, .. } => model.n_stats(),
        }
    }
}

/// Adds a block's statistics `inner` to an operator's statistics `out`.
fn place(inner: &[f64], block: &SubBlock, compact: bool, q: usize, scale: f64, out: &mut [f64]) {
    if compact {
        for (s, &g) in inner.iter().enumerate() {
            for (r, &x) in block.design.iter().enumerate() {
                out[s * q + r] += scale * g * x;
            }
        }
    } else {
        out[block.offset..block.offset + inner.len()].iter_mut().zip(inner).for_each(|(o, g)| *o += scale * g);
    }
}

impl Entry {
    /// For S(): the dyad (i, j) in the subgraph's numbering, if it is in it.
    #[inline]
    fn subgraph_dyad(local: &[u32], head: &Option<Vec<bool>>, oriented: bool, i: u32, j: u32) -> Option<(u32, u32)> {
        let (a, b) = (local[i as usize], local[j as usize]);
        if a == u32::MAX || b == u32::MAX {
            return None;
        }
        match head {
            Some(h) if h[i as usize] == h[j as usize] => None,
            Some(h) if oriented && h[i as usize] => None, // an arc from the second set to the first
            _ => Some((a, b)),
        }
    }
}

/// A network with the auxiliary networks that its terms need: for each
/// filter of F(), the ties that pass it; with blocks, each block's previous
/// network and, for each block operator (and S()), the state of each block's
/// view (of the subgraph).
#[derive(Clone)]
pub struct State {
    pub net: Network,
    aux: Vec<Network>,
    prev: Vec<Network>,
    subs: Vec<Vec<State>>,
}

/// The terms of a model, with the position of each term's statistics.
pub struct Model {
    entries: Vec<Entry>,
    offsets: Vec<usize>,
    filters: Vec<Filter>,
    n_stats: usize,
    layout: Option<Layout>,
    /// Each block's previous network, for tergm's operators (NetSeries()).
    prev: Vec<Network>,
    /// Number of block operators.
    n_slots: usize,
    n_vertices: u32,
}

impl Model {
    /// A model for networks with `n` vertices; `layout` and `prev` give the
    /// blocks of a combined network and their previous networks.
    pub fn new(
        n: u32,
        directed: bool,
        specs: &[TermSpec],
        layout: Option<Layout>,
        prev: Vec<Network>,
    ) -> Result<Self, String> {
        let (n_usize, mut entries, mut filters, mut n_slots) = (n as usize, Vec::new(), Vec::new(), 0);
        if let Some(layout) = &layout
            && !prev.is_empty()
            && (prev.len() != layout.len() || prev.iter().enumerate().any(|(k, p)| p.n() != layout.size(k)))
        {
            return Err("need one previous network per block, of the block's size".into());
        }
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
            } else if spec.0 == "subgraph" {
                // ints: [bipartite, k, the k vertices of the first set, then (if
                // bipartite) the vertices of the second]; children: the terms,
                // for a network of the subgraph's vertices in that order.
                let ints = &spec.2;
                let (bipartite, k) = match ints.as_slice() {
                    [b, k, ..] if *k >= 0 => (*b == 1, *k as usize),
                    _ => return Err("S(): bad parameters".into()),
                };
                let vertices = &ints[2..];
                if vertices.len() < k || (!bipartite && vertices.len() != k) || vertices.iter().any(|&v| v < 0 || v >= n as i64) {
                    return Err("S(): bad vertices".into());
                }
                let mut local = vec![u32::MAX; n_usize];
                for (position, &v) in vertices.iter().enumerate() {
                    if local[v as usize] != u32::MAX {
                        return Err("S(): a vertex can't be in both sets".into());
                    }
                    local[v as usize] = position as u32;
                }
                let head = bipartite.then(|| {
                    let mut h = vec![false; n_usize];
                    vertices[k..].iter().for_each(|&v| h[v as usize] = true);
                    h
                });
                let size = vertices.len() as u32;
                if size < 2 {
                    return Err("S(): the subgraph needs at least 2 vertices".into());
                }
                let sub_directed = directed && !bipartite;
                let model = Box::new(Model::new(size, sub_directed, &spec.3, None, Vec::new())?);
                let oriented = directed && bipartite;
                entries.push(Entry::Subgraph { slot: n_slots, local, head, oriented, sub_directed, model });
                n_slots += 1;
            } else if spec.0 == "blocks" {
                // ints: [view, negate, compact, q]; children: one "block" per block,
                // with its row of the linear model as reals and its terms as children.
                let Some(layout) = &layout else {
                    return Err("N() and tergm's operators need networks combined with Networks() or \
                                NetSeries(), and can't be nested"
                        .into());
                };
                let &[view, negate, compact, q] = spec.2.as_slice() else {
                    return Err("block operator: bad parameters".into());
                };
                let view = View::from_code(view)?;
                if view != View::Same && prev.is_empty() {
                    return Err("Form(), Persist(), Diss() and Change() need the previous networks of a NetSeries()".into());
                }
                let (compact, q) = (compact == 1, q as usize);
                if spec.3.len() != layout.len() {
                    return Err(format!("block operator: {} blocks for {}", spec.3.len(), layout.len()));
                }
                let mut blocks = Vec::with_capacity(spec.3.len());
                let mut offset = 0;
                for (k, block) in spec.3.iter().enumerate() {
                    let model = Model::new(layout.size(k), directed, &block.3, None, Vec::new())?;
                    if compact && block.1.len() != q {
                        return Err("block operator: each block needs a row of the linear model".into());
                    }
                    let size = model.n_stats();
                    blocks.push(SubBlock { model, design: block.1.clone(), offset });
                    offset += size;
                }
                let n_stats = if compact {
                    let p = blocks.first().map_or(0, |b| b.model.n_stats());
                    if blocks.iter().any(|b| b.model.n_stats() != p) {
                        return Err("block operator: the blocks have different numbers of statistics".into());
                    }
                    p * q
                } else {
                    offset
                };
                let scale = if negate == 1 { -1.0 } else { 1.0 };
                entries.push(Entry::Blocks { view, scale, slot: n_slots, blocks, compact, q, n_stats });
                n_slots += 1;
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
        Ok(Self { entries, offsets, filters, n_stats, layout, prev, n_slots, n_vertices: n })
    }

    pub fn n_stats(&self) -> usize {
        self.n_stats
    }

    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// For models of transitions (tergm's operators), the previous networks
    /// of the blocks, as one network with the model's vertices.
    pub fn previous_network(&self, directed: bool) -> Option<Network> {
        let layout = self.layout.as_ref().filter(|_| !self.prev.is_empty())?;
        let n = layout.starts.last().zip(layout.sizes.last()).map_or(0, |(s, z)| s + z);
        let mut joined = Network::new(n.max(self.n_vertices), directed);
        for (k, prev) in self.prev.iter().enumerate() {
            for &(i, j) in prev.edges() {
                joined.toggle(i + layout.starts[k], j + layout.starts[k]);
            }
        }
        Some(joined)
    }

    /// The state of a network: the network and the auxiliary networks of its filters.
    pub fn state(&self, net: Network) -> State {
        self.state_with(net, self.prev.clone())
    }

    /// The shared partner counts the terms read, on the network and (for
    /// the terms inside F()) on the filtered networks.
    fn partner_kinds(&self) -> (Vec<Partners>, Vec<Partners>) {
        let (mut own, mut filtered) = (Vec::new(), Vec::new());
        for entry in &self.entries {
            match entry {
                Entry::Plain(term) => own.extend(term.partners()),
                Entry::Filtered { terms, .. } => filtered.extend(terms.iter().filter_map(|t| t.partners())),
                _ => {}
            }
        }
        own.dedup();
        filtered.dedup();
        (own, filtered)
    }

    /// The state of a network whose blocks have the previous networks `prev`.
    pub fn state_with(&self, mut net: Network, prev: Vec<Network>) -> State {
        let (own, filtered_kinds) = self.partner_kinds();
        for &kind in &own {
            net.keep_partners(kind);
        }
        let aux = self
            .filters
            .iter()
            .map(|filter| {
                let mut filtered = Network::new(net.n(), net.directed());
                for &kind in &filtered_kinds {
                    filtered.keep_partners(kind);
                }
                for &(i, j) in net.edges() {
                    if filter.passes(&net, i, j) {
                        filtered.toggle(i, j);
                    }
                }
                filtered
            })
            .collect();
        let mut subs = Vec::with_capacity(self.n_slots);
        let parts = self.layout.as_ref().filter(|_| self.n_slots > 0).map(|l| l.split(&net));
        for entry in &self.entries {
            match entry {
                Entry::Blocks { view, blocks, .. } => {
                    let parts = parts.as_ref().expect("block operators need a layout");
                    let states = blocks
                        .iter()
                        .zip(parts)
                        .enumerate()
                        .map(|(k, (block, y))| block.model.state(view.of(y, prev.get(k))))
                        .collect();
                    subs.push(states);
                }
                Entry::Subgraph { local, head, oriented, sub_directed, model, .. } => {
                    let mut sub = Network::new(model.n_vertices, *sub_directed);
                    for &(i, j) in net.edges() {
                        if let Some((a, b)) = Entry::subgraph_dyad(local, head, *oriented, i, j) {
                            sub.toggle(a, b);
                        }
                    }
                    subs.push(vec![model.state(sub)]);
                }
                _ => {}
            }
        }
        State { net, aux, prev, subs }
    }

    /// Toggles (i, j) in the network and in the auxiliary networks it belongs to.
    pub fn toggle(&self, state: &mut State, i: u32, j: u32) {
        for (filter, aux) in self.filters.iter().zip(&mut state.aux) {
            if filter.passes(&state.net, i, j) {
                aux.toggle(i, j);
            }
        }
        if self.n_slots > 0 {
            let located = self.layout.as_ref().and_then(|l| l.locate(i, j));
            for entry in &self.entries {
                match entry {
                    Entry::Blocks { view, slot, blocks, .. } => {
                        if let Some((k, a, b)) = located
                            && view.sees(state.prev.get(k), a, b)
                        {
                            blocks[k].model.toggle(&mut state.subs[*slot][k], a, b);
                        }
                    }
                    Entry::Subgraph { slot, local, head, oriented, model, .. } => {
                        if let Some((a, b)) = Entry::subgraph_dyad(local, head, *oriented, i, j) {
                            model.toggle(&mut state.subs[*slot][0], a, b);
                        }
                    }
                    _ => {}
                }
            }
        }
        state.net.toggle(i, j);
    }

    /// Writes to `out` the change in statistics from toggling (i, j).
    pub fn change(&self, state: &State, i: u32, j: u32, out: &mut [f64]) {
        out.fill(0.0);
        let sign = if state.net.has_edge(i, j) { -1.0 } else { 1.0 };
        let located = if self.n_slots > 0 { self.layout.as_ref().and_then(|l| l.locate(i, j)) } else { None };
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
                Entry::Subgraph { slot, local, head, oriented, model, .. } => {
                    if let Some((a, b)) = Entry::subgraph_dyad(local, head, *oriented, i, j) {
                        model.change(&state.subs[*slot][0], a, b, out);
                    }
                }
                Entry::Blocks { view, scale, slot, blocks, compact, q, .. } => {
                    let Some((k, a, b)) = located else { continue };
                    if !view.sees(state.prev.get(k), a, b) {
                        continue;
                    }
                    let (block, sub) = (&blocks[k], &state.subs[*slot][k]);
                    let p = block.model.n_stats();
                    if *compact {
                        // The block's changes in out[..p], then spread over the
                        // q columns from the last, which never overwrites an
                        // unread change.
                        block.model.change(sub, a, b, &mut out[..p]);
                        for s in (0..p).rev() {
                            let g = scale * out[s];
                            for r in (0..*q).rev() {
                                out[s * q + r] = g * block.design[r];
                            }
                        }
                    } else {
                        let out = &mut out[block.offset..block.offset + p];
                        block.model.change(sub, a, b, out);
                        if *scale != 1.0 {
                            out.iter_mut().for_each(|x| *x *= scale);
                        }
                    }
                }
            }
        }
    }

    /// Statistics of `net`: those of the empty network, plus the changes from
    /// adding its edges one by one.
    pub fn summary(&self, net: &Network) -> Vec<f64> {
        self.summary_with(net, &self.prev)
    }

    /// Statistics of `net`, whose blocks have the previous networks `prev`.
    pub fn summary_with(&self, net: &Network, prev: &[Network]) -> Vec<f64> {
        let mut build = self.state_with(Network::new(net.n(), net.directed()), prev.to_vec());
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
                // The statistics of each block's view of the empty network: of
                // the previous network itself, for Form() and Change().
                Entry::Blocks { scale, slot, blocks, compact, q, .. } => {
                    for (k, block) in blocks.iter().enumerate() {
                        let inner = block.model.summary(&build.subs[*slot][k].net);
                        place(&inner, block, *compact, *q, *scale, out);
                    }
                }
                Entry::Subgraph { slot, model, .. } => {
                    out.copy_from_slice(&model.summary(&build.subs[*slot][0].net));
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

    /// The MPLE data compressed, as ergm compresses it: each distinct row of
    /// change statistics (for adding the dyad's tie) with the number of free
    /// dyads that have it, by whether they are ties. Returns the rows (row
    /// major), their response (0 or 1) and counts, sorted. The rows of
    /// vertices are computed in parallel, and the dyad x statistics matrix
    /// never exists.
    pub fn mple_table(&self, net: Network, space: &Space) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        use rayon::prelude::*;
        let n = net.n();
        let p = self.n_stats;
        let state = self.state(net);
        // Distinct rows, as the bit patterns of the statistics (with -0 as 0),
        // with the number of non-ties and ties.
        type Table = FxHashMap<Vec<u64>, [u64; 2]>;
        let add = |table: &mut Table, delta: &mut Vec<f64>, key: &mut Vec<u64>, i: u32, j: u32| {
            self.change(&state, i, j, delta);
            let tie = state.net.has_edge(i, j);
            for (k, &d) in key.iter_mut().zip(delta.iter()) {
                let d = if tie { -d } else { d };
                *k = if d == 0.0 { 0 } else { d.to_bits() };
            }
            match table.get_mut(key) {
                Some(counts) => counts[tie as usize] += 1,
                None => {
                    let mut counts = [0; 2];
                    counts[tie as usize] = 1;
                    table.insert(key.clone(), counts);
                }
            }
        };
        let start = || (Table::default(), vec![0.0; p], vec![0u64; p]);
        let tables: Vec<Table> = match space.list() {
            Some(list) => list
                .par_chunks(4096)
                .fold(start, |(mut table, mut delta, mut key), chunk| {
                    chunk.iter().for_each(|&(i, j)| add(&mut table, &mut delta, &mut key, i, j));
                    (table, delta, key)
                })
                .map(|(table, _, _)| table)
                .collect(),
            None => (0..n)
                .into_par_iter()
                .fold(start, |(mut table, mut delta, mut key), i| {
                    space.for_each_free_of(i, |j| add(&mut table, &mut delta, &mut key, i, j));
                    (table, delta, key)
                })
                .map(|(table, _, _)| table)
                .collect(),
        };
        let merged = tables
            .into_iter()
            .reduce(|mut a, b| {
                for (key, counts) in b {
                    let entry = a.entry(key).or_insert([0; 2]);
                    entry[0] += counts[0];
                    entry[1] += counts[1];
                }
                a
            })
            .unwrap_or_default();
        let mut rows: Vec<(Vec<u64>, [u64; 2])> = merged.into_iter().collect();
        rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let (mut x, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for (key, counts) in rows {
            for (response, &count) in counts.iter().enumerate() {
                if count > 0 {
                    x.extend(key.iter().map(|&bits| f64::from_bits(bits)));
                    y.push(response as f64);
                    w.push(count as f64);
                }
            }
        }
        (x, y, w)
    }

    /// Change statistics for adding each free dyad given the rest of `net`,
    /// whether the dyad is a tie, and the dyads (i < j if undirected), in
    /// the order of the space's free dyads.
    pub fn mple_data(&self, net: Network, space: &Space) -> (Vec<f64>, Vec<f64>, Vec<(u32, u32)>) {
        let state = self.state(net);
        let dyads = space.n_free() as usize;
        let mut x = Vec::with_capacity(dyads * self.n_stats);
        let mut y = Vec::with_capacity(dyads);
        let mut pairs = Vec::with_capacity(dyads);
        let mut delta = vec![0.0; self.n_stats];
        space.for_each_free(|i, j| {
            self.change(&state, i, j, &mut delta);
            let tie = state.net.has_edge(i, j);
            if tie {
                delta.iter_mut().for_each(|d| *d = -*d);
            }
            x.extend_from_slice(&delta);
            y.push(tie as u8 as f64);
            pairs.push((i, j));
        });
        (x, y, pairs)
    }
}
