//! More of ergm's vocabulary: degree ranges and degrees by attribute, the
//! triad census and the terms built on it, trails, Simmelian ties, covariate
//! ranges and distinct neighbour types, terms restricted to vertices that
//! match on an attribute, more bipartite terms, and interactions of
//! dyad-independent terms. Definitions follow R's ergm, and its C code where
//! the documentation leaves details open.

use crate::network::{Network, for_each_common};
use crate::terms::{Ends, Term, TermSpec, binomial, counted};

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

/// A vertex's ties on one side: all (undirected), out-ties or in-ties.
#[inline]
fn ties(net: &Network, ends: Ends, v: u32) -> &[u32] {
    match ends {
        Ends::Both => net.neighbours(v),
        Ends::Tail => net.out_neighbours(v),
        Ends::Head => net.in_neighbours(v),
    }
}

fn levels_of(codes: &[i64]) -> usize {
    (codes.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize
}

// -- Degrees ------------------------------------------------------------------------------------

/// Number of vertices with degree in [from, to), for each range (degree,
/// degrange, their in- and out- versions and bipartite modes, concurrent and
/// mindegree): with `codes` (ergm's by=), one set of ranges per level; with
/// `homophily`, degrees only count ties between vertices of the same level.
struct DegreeRange {
    ranges: Vec<(u32, u32)>,
    ends: Ends,
    mask: Option<Vec<bool>>,
    codes: Option<Vec<i64>>,
    homophily: bool,
    n_levels: usize,
}

impl DegreeRange {
    fn degree(&self, net: &Network, v: u32) -> u32 {
        let list = ties(net, self.ends, v);
        match (&self.codes, self.homophily) {
            (Some(codes), true) => list.iter().filter(|&&u| codes[u as usize] == codes[v as usize]).count() as u32,
            _ => list.len() as u32,
        }
    }

    /// The statistics' level of v, or None if it isn't counted.
    fn level(&self, v: u32) -> Option<usize> {
        if !counted(&self.mask, v) {
            return None;
        }
        match &self.codes {
            None => Some(0),
            Some(codes) => {
                let c = usize::try_from(codes[v as usize]).ok()?;
                Some(if self.homophily { 0 } else { c })
            }
        }
    }
}

impl Term for DegreeRange {
    fn n_stats(&self) -> usize {
        self.ranges.len() * self.n_levels
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        if self.homophily
            && let Some(codes) = &self.codes
            && codes[i as usize] != codes[j as usize]
        {
            return;
        }
        let (vertices, count) = self.ends.vertices(i, j);
        for &v in &vertices[..count] {
            let Some(level) = self.level(v) else { continue };
            let before = self.degree(net, v);
            let after = if sign > 0.0 { before + 1 } else { before - 1 };
            for (r, &(from, to)) in self.ranges.iter().enumerate() {
                let inside = |d: u32| (d >= from && d < to) as i32 as f64;
                out[level * self.ranges.len() + r] += inside(after) - inside(before);
            }
        }
    }

    fn empty(&self, n: u32, _: bool, out: &mut [f64]) {
        for v in 0..n {
            if let Some(level) = self.level(v) {
                for (r, &(from, _)) in self.ranges.iter().enumerate() {
                    if from == 0 {
                        out[level * self.ranges.len() + r] += 1.0;
                    }
                }
            }
        }
    }
}

/// Sum over vertices of a power of their degree: degree1.5 and its in- and out- versions.
struct DegreePower {
    power: f64,
    ends: Ends,
}

impl Term for DegreePower {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (vertices, count) = self.ends.vertices(i, j);
        for &v in &vertices[..count] {
            let before = ties(net, self.ends, v).len() as f64;
            out[0] += (before + sign).powf(self.power) - before.powf(self.power);
        }
    }
}

/// concurrentties: the ties of each vertex beyond its first, by level of `codes`.
struct ConcurrentTies {
    mask: Option<Vec<bool>>,
    codes: Option<Vec<i64>>,
    n_levels: usize,
}

impl Term for ConcurrentTies {
    fn n_stats(&self) -> usize {
        self.n_levels
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        for v in [i, j] {
            if !counted(&self.mask, v) {
                continue;
            }
            let level = match &self.codes {
                None => 0,
                Some(c) => match usize::try_from(c[v as usize]) {
                    Ok(l) => l,
                    Err(_) => continue,
                },
            };
            let before = net.neighbours(v).len() as f64;
            let excess = |d: f64| (d - 1.0).max(0.0);
            out[level] += excess(before + sign) - excess(before);
        }
    }
}

/// edges times a constant: density and meandeg.
struct ScaledEdges(f64);

impl Term for ScaledEdges {
    fn change(&self, _: &Network, _: u32, _: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * self.0;
    }
}

/// Number of ties whose two vertices have no other tie (undirected).
struct IsolatedEdges;

impl Term for IsolatedEdges {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let degree = |v: u32| net.neighbours(v).len();
        let (di, dj) = (degree(i), degree(j));
        let mut change = 0.0;
        if sign < 0.0 {
            if di == 1 && dj == 1 {
                change -= 1.0;
            }
            // A vertex left with one tie makes it isolated if its other end has no other.
            for (v, d, other) in [(i, di, j), (j, dj, i)] {
                if d == 2 {
                    change += net.neighbours(v).iter().filter(|&&u| u != other && degree(u) == 1).count() as f64;
                }
            }
        } else {
            if di == 0 && dj == 0 {
                change += 1.0;
            }
            for (v, d) in [(i, di), (j, dj)] {
                if d == 1 {
                    change -= net.neighbours(v).iter().filter(|&&u| degree(u) == 1).count() as f64;
                }
            }
        }
        out[0] += change;
    }
}

// -- Dyadic covariates -------------------------------------------------------------------------

/// dyadcov in directed networks: the covariate of mutual dyads, of dyads with
/// only the tie from the lower to the higher vertex, and the reverse.
struct DyadCov {
    x: Vec<f64>,
    n: usize,
}

impl Term for DyadCov {
    fn n_stats(&self) -> usize {
        3
    }

    fn change(&self, net: &Network, i: u32, j: u32, _: f64, out: &mut [f64]) {
        let (lo, hi) = (i.min(j), i.max(j));
        let w = self.x[lo as usize * self.n + hi as usize];
        // The dyad's state: 0 mutual, 1 lower -> higher only, 2 higher -> lower only.
        let state = |up: bool, down: bool| match (up, down) {
            (true, true) => Some(0),
            (true, false) => Some(1),
            (false, true) => Some(2),
            _ => None,
        };
        let (up, down) = (net.has_edge(lo, hi), net.has_edge(hi, lo));
        let before = state(up, down);
        let after = if i == lo { state(!up, down) } else { state(up, !down) };
        if let Some(s) = before {
            out[s] -= w;
        }
        if let Some(s) = after {
            out[s] += w;
        }
    }
}

/// Hamming distance to a reference network, weighted by `weights` if given.
struct Hamming {
    reference: Vec<bool>,
    weights: Option<Vec<f64>>,
    n: usize,
    directed: bool,
}

impl Term for Hamming {
    fn change(&self, net: &Network, i: u32, j: u32, _: f64, out: &mut [f64]) {
        let (a, b) = if self.directed || i < j { (i, j) } else { (j, i) };
        let k = a as usize * self.n + b as usize;
        let w = self.weights.as_ref().map_or(1.0, |w| w[k]);
        // Toggling makes the dyad agree with the reference if it disagreed, and the reverse.
        out[0] += if net.has_edge(i, j) == self.reference[k] { w } else { -w };
    }

    fn empty(&self, n: u32, directed: bool, out: &mut [f64]) {
        for a in 0..n as usize {
            for b in 0..n as usize {
                let k = a * self.n + b;
                if a != b && (directed || a < b) && self.reference[k] {
                    out[0] += self.weights.as_ref().map_or(1.0, |w| w[k]);
                }
            }
        }
    }
}

/// Sum over ties of a covariate of the levels of their vertices (attrcov).
struct AttrCov {
    codes: Vec<i64>,
    mat: Vec<f64>,
    n_levels: usize,
    directed: bool,
}

impl Term for AttrCov {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (a, b) = if self.directed || i < j { (i, j) } else { (j, i) };
        let (ca, cb) = (self.codes[a as usize], self.codes[b as usize]);
        if ca >= 0 && cb >= 0 {
            out[0] += sign * self.mat[ca as usize * self.n_levels + cb as usize];
        }
    }
}

/// The cells of a mixing matrix of a row and a column attribute (ergm's mm):
/// the ties by (row level of the tail, column level of the head), or in
/// undirected networks by both orientations of each tie, unless the matrix
/// is of one attribute with itself (then symmetric, each tie counted once).
struct MixMatrix {
    rows: Vec<i64>,
    cols: Vec<i64>,
    n_cols: usize,
    /// Statistic of each cell (row-major), or -1.
    map: Vec<i64>,
    both: bool,
}

impl MixMatrix {
    #[inline]
    fn cell(&self, a: u32, b: u32, sign: f64, out: &mut [f64]) {
        let (r, c) = (self.rows[a as usize], self.cols[b as usize]);
        if r >= 0
            && c >= 0
            && let Ok(s) = usize::try_from(self.map[r as usize * self.n_cols + c as usize])
        {
            out[s] += sign;
        }
    }
}

impl Term for MixMatrix {
    fn n_stats(&self) -> usize {
        levels_of(&self.map)
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        self.cell(i, j, sign, out);
        if self.both {
            self.cell(j, i, sign, out);
        }
    }
}

// -- Covariate ranges and distinct neighbour types -----------------------------------------------

/// Sum over vertices of the range of a covariate over their neighbours
/// (nodecovrange; out- or in-neighbours, or both in turn in directed
/// networks; b1covrange and b2covrange with a mode mask).
struct CovRange {
    x: Vec<f64>,
    /// The sides counted: the tail's out-neighbours, the head's in-neighbours
    /// (in undirected networks, both vertices' neighbours).
    tail: bool,
    head: bool,
    mask: Option<Vec<bool>>,
}

impl CovRange {
    /// The change in the range over `list` from adding (or removing) `other`.
    fn delta(&self, list: &[u32], other: u32, adding: bool) -> f64 {
        let (mut old, mut new) = ((f64::INFINITY, f64::NEG_INFINITY), (f64::INFINITY, f64::NEG_INFINITY));
        for &u in list {
            let v = self.x[u as usize];
            old = (old.0.min(v), old.1.max(v));
            if adding || u != other {
                new = (new.0.min(v), new.1.max(v));
            }
        }
        if adding {
            let v = self.x[other as usize];
            new = (new.0.min(v), new.1.max(v));
        }
        let range = |r: (f64, f64)| if r.1.is_finite() { r.1 - r.0 } else { 0.0 };
        range(new) - range(old)
    }
}

impl Term for CovRange {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let adding = sign > 0.0;
        if !net.directed() {
            for (v, other) in [(i, j), (j, i)] {
                if counted(&self.mask, v) {
                    out[0] += self.delta(net.neighbours(v), other, adding);
                }
            }
            return;
        }
        if self.tail && counted(&self.mask, i) {
            out[0] += self.delta(net.out_neighbours(i), j, adding);
        }
        if self.head && counted(&self.mask, j) {
            out[0] += self.delta(net.in_neighbours(j), i, adding);
        }
    }
}

/// Sum over vertices of the number of distinct levels among their
/// neighbours (nodefactordistinct), as ergm counts them: in directed
/// networks, a vertex's levels are those of its out-neighbours (`tail`), its
/// in-neighbours (`head`), or both together; a mask restricts the vertices
/// counted (bipartite modes).
struct FactorDistinct {
    codes: Vec<i64>,
    tail: bool,
    head: bool,
    mask: Option<Vec<bool>>,
}

impl FactorDistinct {
    /// How many of v's neighbours, on the counted sides, have the level `level`.
    fn frequency(&self, net: &Network, v: u32, level: i64) -> usize {
        let count = |list: &[u32]| list.iter().filter(|&&u| self.codes[u as usize] == level).count();
        if !net.directed() {
            return count(net.neighbours(v));
        }
        (if self.tail { count(net.out_neighbours(v)) } else { 0 }) + (if self.head { count(net.in_neighbours(v)) } else { 0 })
    }
}

impl Term for FactorDistinct {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let undirected = !net.directed();
        // The tail sees the head's level among its out-neighbours, the head the tail's.
        for (v, other, active) in [(i, j, self.tail || undirected), (j, i, self.head || undirected)] {
            let level = self.codes[other as usize];
            if !active || level < 0 || !counted(&self.mask, v) {
                continue;
            }
            let before = self.frequency(net, v, level) as f64;
            out[0] += ((before + sign) != 0.0) as i32 as f64 - (before != 0.0) as i32 as f64;
        }
    }
}

// -- Attribute differences -------------------------------------------------------------------------

/// diff: sum over ties of f(x[tail] - x[head]) (or head - tail), where f is
/// the sign action then the power (the signum, for power 0). Undirected ties
/// go from the lower to the higher vertex, bipartite ties from the first mode.
struct Diff {
    x: Vec<f64>,
    pow: f64,
    head_minus_tail: bool,
    action: u8,
    orient: u8,
    first: Vec<bool>,
}

impl Term for Diff {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (t, h) = match self.orient {
            0 => (i, j),
            1 => (i.min(j), i.max(j)),
            _ => {
                if self.first[i as usize] {
                    (i, j)
                } else {
                    (j, i)
                }
            }
        };
        let mut d = self.x[t as usize] - self.x[h as usize];
        if self.head_minus_tail {
            d = -d;
        }
        d = match self.action {
            1 => d.abs(),
            2 => d.max(0.0),
            3 => d.min(0.0),
            _ => d,
        };
        let value = if self.pow == 0.0 { d.signum() * (d != 0.0) as i32 as f64 } else { d.powf(self.pow) };
        out[0] += sign * value;
    }
}

/// Number of ties whose vertices' values differ by at most `cutoff` (ergm's
/// C code; its documentation says less than).
struct SmallDiff {
    x: Vec<f64>,
    cutoff: f64,
}

impl Term for SmallDiff {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        if (self.x[i as usize] - self.x[j as usize]).abs() <= self.cutoff {
            out[0] += sign;
        }
    }
}

/// Alternating k-stars (Snijders et al. 2006) with a fixed lambda: a new tie
/// on a vertex of degree d adds lambda (1 - (1 - 1/lambda)^d).
struct AltKStar {
    lambda: f64,
}

impl Term for AltKStar {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let r = 1.0 - 1.0 / self.lambda;
        for v in [i, j] {
            let d = net.neighbours(v).len() as i32 - (sign < 0.0) as i32;
            out[0] += sign * self.lambda * (1.0 - r.powi(d));
        }
    }
}

/// Per-vertex statistics counting only ties between vertices of the same
/// level of `codes` (sociality's attr=): `stat` is each vertex's statistic or -1.
struct FactorMatch {
    stat: Vec<i64>,
    codes: Vec<i64>,
    n_stats: usize,
}

impl Term for FactorMatch {
    fn n_stats(&self) -> usize {
        self.n_stats
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        if self.codes[i as usize] != self.codes[j as usize] {
            return;
        }
        for v in [i, j] {
            if let Ok(s) = usize::try_from(self.stat[v as usize]) {
                out[s] += sign;
            }
        }
    }
}

// -- Triads ------------------------------------------------------------------------------------------

/// The 16 triad types of Davis and Leinhardt (1972), in ergm's order: 003,
/// 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201, 120D, 120U, 120C,
/// 210, 300 (sna's definitions: 021D A<-B->C, 111D A<->B<-C, 120D A<-B->C
/// with A<->C, and so on).
fn triad_type(tie: impl Fn(usize, usize) -> bool) -> usize {
    let pairs = [(0, 1), (0, 2), (1, 2)];
    let (mut mutual, mut asym) = (Vec::new(), Vec::new());
    for &(a, b) in &pairs {
        match (tie(a, b), tie(b, a)) {
            (true, true) => mutual.push((a, b)),
            (true, false) => asym.push((a, b)),
            (false, true) => asym.push((b, a)),
            _ => {}
        }
    }
    let out_degree = |v: usize| asym.iter().filter(|&&(a, _)| a == v).count();
    let in_degree = |v: usize| asym.iter().filter(|&&(_, b)| b == v).count();
    match (mutual.len(), asym.len()) {
        (0, 0) => 0,
        (0, 1) => 1,
        (1, 0) => 2,
        (0, 2) => {
            if (0..3).any(|v| out_degree(v) == 2) {
                3 // 021D
            } else if (0..3).any(|v| in_degree(v) == 2) {
                4 // 021U
            } else {
                5 // 021C
            }
        }
        (1, 1) => {
            // The asymmetric tie points into the mutual dyad (111D) or out of it (111U).
            let (p, q) = mutual[0];
            let (_, b) = asym[0];
            if b == p || b == q { 6 } else { 7 }
        }
        (0, 3) => {
            if (0..3).any(|v| out_degree(v) == 2) { 8 } else { 9 }
        }
        (2, 0) => 10,
        (1, 2) => {
            let (p, q) = mutual[0];
            let r = 3 - p - q;
            if out_degree(r) == 2 {
                11 // 120D
            } else if in_degree(r) == 2 {
                12 // 120U
            } else {
                13 // 120C
            }
        }
        (2, 1) => 14,
        _ => 15,
    }
}

/// Counts of triads by type, each mapped to a statistic (or none): the
/// triad census, balance, intransitive, simmelian and nearsimmelian. The
/// undirected types are the number of ties, 0 to 3.
struct TriadCensus {
    /// Statistic of each type, or -1.
    map: Vec<i64>,
    /// Type of the triad {i, j, k} by its ties as bits: i->j, j->i, i->k, k->i, j->k, k->j.
    table: [u8; 64],
    n_stats: usize,
}

impl TriadCensus {
    fn new(map: Vec<i64>, directed: bool) -> Self {
        let mut table = [0u8; 64];
        for (bits, t) in table.iter_mut().enumerate() {
            let bit = |a: usize, b: usize| {
                let k = match (a, b) {
                    (0, 1) => 0,
                    (1, 0) => 1,
                    (0, 2) => 2,
                    (2, 0) => 3,
                    (1, 2) => 4,
                    _ => 5,
                };
                bits >> k & 1 == 1
            };
            *t = if directed {
                triad_type(bit) as u8
            } else {
                (bit(0, 1) as u8) + (bit(0, 2) as u8) + (bit(1, 2) as u8)
            };
        }
        let n_stats = levels_of(&map);
        Self { map, table, n_stats }
    }

    #[inline]
    fn add(&self, t: u8, by: f64, out: &mut [f64]) {
        if let Ok(s) = usize::try_from(self.map[t as usize]) {
            out[s] += by;
        }
    }
}

impl Term for TriadCensus {
    fn n_stats(&self) -> usize {
        self.n_stats
    }

    fn change(&self, net: &Network, i: u32, j: u32, _: f64, out: &mut [f64]) {
        let directed = net.directed();
        let tied = net.has_edge(i, j) as usize | ((directed && net.has_edge(j, i)) as usize) << 1;
        // The toggled tie i -> j is bit 0 (in undirected networks, the only bit of the pair).
        let flip = |bits: usize| bits ^ 1;
        let mut seen = 0u64;
        let mut visit = |k: u32| {
            let rest = if directed {
                (net.has_edge(i, k) as usize) << 2
                    | (net.has_edge(k, i) as usize) << 3
                    | (net.has_edge(j, k) as usize) << 4
                    | (net.has_edge(k, j) as usize) << 5
            } else {
                (net.has_edge(i, k) as usize) << 2 | (net.has_edge(j, k) as usize) << 4
            };
            let before = rest | tied;
            self.add(self.table[before], -1.0, out);
            self.add(self.table[flip(before)], 1.0, out);
            seen += 1;
        };
        // The vertices tied to i or j, each once.
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
        // The other vertices form triads with only the pair (i, j) tied.
        let others = (net.n() as u64 - 2 - seen) as f64;
        if others > 0.0 {
            self.add(self.table[tied], -others, out);
            self.add(self.table[flip(tied)], others, out);
        }
    }

    fn empty(&self, n: u32, _: bool, out: &mut [f64]) {
        let n = n as f64;
        self.add(0, n * (n - 1.0) * (n - 2.0) / 6.0, out);
    }
}

/// Number of ties in at least one Simmelian triad (directed), as ergm.
struct SimmelianTies;

impl Term for SimmelianTies {
    fn change(&self, net: &Network, tail: u32, head: u32, sign: f64, out: &mut [f64]) {
        if !net.has_edge(head, tail) {
            return;
        }
        let has = |a: u32, b: u32| net.has_edge(a, b);
        let (mut change, mut in_triad) = (0.0, false);
        for &k in net.out_neighbours(head) {
            if k == tail || !(has(k, tail) && has(tail, k) && has(k, head)) {
                continue;
            }
            in_triad = true;
            // Whether (tail, k) and (head, k) are in another Simmelian triad.
            for (v, other) in [(tail, head), (head, tail)] {
                let alone = !net
                    .out_neighbours(v)
                    .iter()
                    .any(|&w| w != other && w != k && has(w, v) && has(w, k) && has(k, w));
                if alone {
                    change += 1.0;
                }
            }
        }
        change += in_triad as i32 as f64;
        out[0] += sign * 2.0 * change;
    }
}

/// Number of 3-trails: in undirected networks, sum over ties of (d_j - 1)
/// (d_k - 1); in directed networks, the trails i - j -> k - l by the
/// directions of their outer steps (RRR, RRL, LRR, LRL), as ergm's C code.
struct ThreeTrail {
    /// The directed types reported, 1 to 4, in order.
    types: Vec<usize>,
}

impl Term for ThreeTrail {
    fn n_stats(&self) -> usize {
        self.types.len().max(1)
    }

    fn change(&self, net: &Network, tail: u32, head: u32, sign: f64, out: &mut [f64]) {
        if !net.directed() {
            // Adding (a, b) to degrees d without it: d_a d_b, plus d_k - 1 for each other tie (a, k), (b, k).
            let removing = sign < 0.0;
            let d = |v: u32| net.neighbours(v).len() as f64;
            let (da, db) = (d(tail) - removing as i32 as f64, d(head) - removing as i32 as f64);
            let mut change = da * db;
            for (v, other) in [(tail, head), (head, tail)] {
                change += net.neighbours(v).iter().filter(|&&k| k != other).map(|&k| d(k) - 1.0).sum::<f64>();
            }
            out[0] += sign * change;
            return;
        }
        let state = net.has_edge(tail, head) as i64;
        let (ind, outd) = (|v: u32| net.in_neighbours(v).len() as i64, |v: u32| net.out_neighbours(v).len() as i64);
        let mut dc = [
            ind(tail) * outd(head),
            ind(tail) * (ind(head) - state),
            (outd(tail) - state) * outd(head),
            (outd(tail) - state) * (ind(head) - state),
        ];
        for &k in net.out_neighbours(head) {
            dc[1] += ind(k) - 1;
            dc[0] += outd(k);
        }
        for &k in net.in_neighbours(head).iter().filter(|&&k| k != tail) {
            dc[3] += outd(k) - 1;
            dc[1] += ind(k);
        }
        for &k in net.in_neighbours(tail) {
            dc[2] += outd(k) - 1;
            dc[0] += ind(k);
        }
        for &k in net.out_neighbours(tail).iter().filter(|&&k| k != head) {
            dc[3] += ind(k) - 1;
            dc[2] += outd(k);
        }
        dc[0] -= net.has_edge(head, tail) as i64 * (1 + 2 * state);
        for (s, &t) in self.types.iter().enumerate() {
            out[s] += sign * dc[t - 1] as f64;
        }
    }
}

/// Number of ties with at least one two-path between their ends:
/// transitiveties (i -> k -> j for the tie i -> j) or cyclicalties
/// (j -> k -> i); in undirected networks, ties with a shared partner. With
/// `codes`, only ties and two-paths of vertices of one level count.
struct SupportedTies {
    cyclical: bool,
    codes: Option<Vec<i64>>,
}

impl SupportedTies {
    #[inline]
    fn level_ok(&self, v: u32, level: i64) -> bool {
        self.codes.as_ref().is_none_or(|c| c[v as usize] == level)
    }

    /// Two-paths supporting the tie a -> b: a -> k -> b (transitive), b -> k
    /// -> a (cyclical), or shared partners if undirected.
    fn support(&self, net: &Network, a: u32, b: u32, level: i64) -> usize {
        let (x, y) = if !net.directed() {
            (net.neighbours(a), net.neighbours(b))
        } else if self.cyclical {
            (net.in_neighbours(a), net.out_neighbours(b))
        } else {
            (net.out_neighbours(a), net.in_neighbours(b))
        };
        let mut count = 0;
        for_each_common(x, y, |k| {
            if self.level_ok(k, level) {
                count += 1
            }
        });
        count
    }
}

impl Term for SupportedTies {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let level = self.codes.as_ref().map_or(0, |c| c[i as usize]);
        if self.codes.is_some() && (level < 0 || !self.level_ok(j, level)) {
            return;
        }
        let present = net.has_edge(i, j) as usize;
        let mut change = (self.support(net, i, j, level) > 0) as i32 as f64;
        // Other ties for which the toggled tie is one step of a supporting two-path.
        let mut affected = |a: u32, b: u32| {
            if self.level_ok(a, level) && self.level_ok(b, level) && self.support(net, a, b, level) == present {
                change += 1.0;
            }
        };
        if !net.directed() {
            for &v in net.neighbours(j).iter().filter(|&&v| v != i) {
                if net.has_edge(i, v) {
                    affected(i, v);
                }
            }
            for &u in net.neighbours(i).iter().filter(|&&u| u != j) {
                if net.has_edge(j, u) {
                    affected(j, u);
                }
            }
        } else if !self.cyclical {
            // i -> j -> v supports i -> v; u -> i -> j supports u -> j.
            for &v in net.out_neighbours(j).iter().filter(|&&v| v != i) {
                if net.has_edge(i, v) {
                    affected(i, v);
                }
            }
            for &u in net.in_neighbours(i).iter().filter(|&&u| u != j) {
                if net.has_edge(u, j) {
                    affected(u, j);
                }
            }
        } else {
            // i -> j -> a supports a -> i; b -> i -> j supports j -> b.
            for &a in net.out_neighbours(j).iter().filter(|&&a| a != i) {
                if net.has_edge(a, i) {
                    affected(a, i);
                }
            }
            for &b in net.in_neighbours(i).iter().filter(|&&b| b != j) {
                if net.has_edge(j, b) {
                    affected(j, b);
                }
            }
        }
        out[0] += sign * change;
    }
}

/// Number of 2-stars minus three times the number of triangles (undirected).
struct OpenTriad;

impl Term for OpenTriad {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let removing = (sign < 0.0) as i32 as f64;
        let stars = net.neighbours(i).len() as f64 + net.neighbours(j).len() as f64 - 2.0 * removing;
        let triangles = crate::network::count_common(net.neighbours(i), net.neighbours(j)) as f64;
        out[0] += sign * (stars - 3.0 * triangles);
    }
}

/// Triangles whose three pairs are all neighbours in `x` (localtriangle).
struct LocalTriangle {
    x: Vec<bool>,
    n: usize,
}

impl Term for LocalTriangle {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let near = |a: u32, b: u32| self.x[a as usize * self.n + b as usize];
        if !near(i, j) {
            return;
        }
        let mut change = 0.0;
        if net.directed() {
            let arcs = |a: u32, b: u32| net.has_edge(a, b) as i32 as f64 + net.has_edge(b, a) as i32 as f64;
            for &k in net.neighbours(j) {
                if k != i && near(k, i) && near(k, j) {
                    change += arcs(k, j) * arcs(k, i);
                }
            }
        } else {
            for_each_common(net.neighbours(i), net.neighbours(j), |k| {
                if near(k, i) && near(k, j) {
                    change += 1.0;
                }
            });
        }
        out[0] += sign * change;
    }
}

// -- Terms restricted to vertices that match on an attribute ---------------------------------------

/// k-stars (in-, out-) whose vertices all have the same level.
struct StarsMatch {
    ks: Vec<u32>,
    ends: Ends,
    codes: Vec<i64>,
    mask: Option<Vec<bool>>,
}

impl Term for StarsMatch {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let level = self.codes[i as usize];
        if level < 0 || self.codes[j as usize] != level {
            return;
        }
        let (centres, count) = self.ends.vertices(i, j);
        for &v in &centres[..count] {
            if !counted(&self.mask, v) {
                continue;
            }
            let other = if v == i { j } else { i };
            let same = ties(net, self.ends, v).iter().filter(|&&u| u != other && self.codes[u as usize] == level).count();
            for (stat, &k) in out.iter_mut().zip(&self.ks) {
                *stat += sign * binomial(same as f64, k - 1);
            }
        }
    }
}

/// Triangles (triangle, ttriple, ctriple) whose vertices all have the same
/// level, in total or (`diff`) by level.
struct TrianglesMatch {
    codes: Vec<i64>,
    diff: bool,
    n_levels: usize,
    /// 0 triangle, 1 transitive triples, 2 cyclic triples.
    kind: u8,
}

impl Term for TrianglesMatch {
    fn n_stats(&self) -> usize {
        if self.diff { self.n_levels } else { 1 }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let level = self.codes[i as usize];
        if level < 0 || self.codes[j as usize] != level {
            return;
        }
        let mut count = 0;
        let mut common = |a: &[u32], b: &[u32]| {
            for_each_common(a, b, |k| {
                if self.codes[k as usize] == level {
                    count += 1
                }
            })
        };
        if !net.directed() {
            common(net.neighbours(i), net.neighbours(j));
        } else {
            let (out_i, in_i, out_j, in_j) =
                (net.out_neighbours(i), net.in_neighbours(i), net.out_neighbours(j), net.in_neighbours(j));
            if self.kind != 2 {
                common(out_i, in_j);
                common(out_i, out_j);
                common(in_i, in_j);
            }
            if self.kind != 1 {
                common(out_j, in_i);
            }
        }
        out[if self.diff { level as usize } else { 0 }] += sign * count as f64;
    }
}

/// Mutual dyads whose vertices match on an attribute (mutual's same=), in
/// total or by level; or (`by`) the vertices of each level in mutual dyads.
struct MutualMatch {
    codes: Vec<i64>,
    diff: bool,
    by: bool,
    n_levels: usize,
}

impl Term for MutualMatch {
    fn n_stats(&self) -> usize {
        if self.diff || self.by { self.n_levels } else { 1 }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        if !net.has_edge(j, i) {
            return;
        }
        let (a, b) = (self.codes[i as usize], self.codes[j as usize]);
        if self.by {
            for c in [a, b] {
                if let Ok(s) = usize::try_from(c) {
                    out[s] += sign;
                }
            }
        } else if a >= 0 && a == b {
            out[if self.diff { a as usize } else { 0 }] += sign;
        }
    }
}

/// Asymmetric dyads whose vertices match on an attribute, in total or by level.
struct AsymmetricMatch {
    codes: Vec<i64>,
    diff: bool,
    n_levels: usize,
}

impl Term for AsymmetricMatch {
    fn n_stats(&self) -> usize {
        if self.diff { self.n_levels } else { 1 }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let a = self.codes[i as usize];
        if a < 0 || self.codes[j as usize] != a {
            return;
        }
        let change = if net.has_edge(j, i) { -sign } else { sign };
        out[if self.diff { a as usize } else { 0 }] += change;
    }
}

/// Geometrically weighted degree by level (gwdegree's attr=, with a fixed decay).
struct GwDegreeMatch {
    r: f64,
    ends: Ends,
    codes: Vec<i64>,
    mask: Option<Vec<bool>>,
    n_levels: usize,
}

impl Term for GwDegreeMatch {
    fn n_stats(&self) -> usize {
        self.n_levels
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for &(v, degree) in &degrees[..count] {
            if let Ok(level) = usize::try_from(self.codes[v as usize])
                && counted(&self.mask, v)
            {
                out[level] += sign * self.r.powf(crate::terms::degree_without(degree, sign));
            }
        }
    }
}

// -- Bipartite terms ----------------------------------------------------------------------------------

/// b1nodematch and b2nodematch with their options (Bomiriya et al. 2023),
/// as ergm's C code: two-stars centred on the other mode whose ends match,
/// discounted by `beta` (for each tie, half the number of such two-stars it
/// is in, to the power beta) or `alpha` (for each pair of matching ends, the
/// number of their shared partners to the power alpha); by level of the
/// ends (`diff`) and of the centres (`by`).
struct NodeMatchBipartite {
    /// The vertices of the matched mode, the ends of the two-stars.
    ends: Vec<bool>,
    /// Level of each end vertex (-1 for levels left out).
    codes: Vec<i64>,
    alpha: f64,
    beta: f64,
    diff: bool,
    n_levels: usize,
    /// Level of each centre (byb2attr=), if any.
    by: Option<Vec<i64>>,
    n_by: usize,
}

impl Term for NodeMatchBipartite {
    fn n_stats(&self) -> usize {
        (if self.diff { self.n_levels } else { 1 }) * self.by.as_ref().map_or(1, |_| self.n_by)
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        // The end (of the matched mode) and the centre.
        let (end, centre) = if self.ends[i as usize] { (i, j) } else { (j, i) };
        let level = self.codes[end as usize];
        if level < 0 {
            return;
        }
        let by_level = self.by.as_ref().map(|b| b[centre as usize]);
        if by_level.is_some_and(|b| b < 0) {
            return;
        }
        let alpha_type = self.beta >= 1.0 && self.alpha < 1.0;
        let exponent = if alpha_type { self.alpha } else { self.beta };
        let mut count = 0usize;
        let mut change = 0.0;
        for &other in net.neighbours(centre) {
            if other == end || self.codes[other as usize] != level {
                continue;
            }
            count += 1;
            if alpha_type {
                // Two-paths between end and other through centres other than this one.
                let mut shared = 0usize;
                for &c in net.neighbours(end) {
                    if c != centre
                        && by_level.is_none_or(|b| self.by.as_ref().unwrap()[c as usize] == b)
                        && net.has_edge(other, c)
                    {
                        shared += 1;
                    }
                }
                change += if shared == 0 { 1.0 } else { (shared as f64 + 1.0).powf(exponent) - (shared as f64).powf(exponent) };
            }
        }
        if !alpha_type {
            change = 0.0;
            if count > 0 {
                let c = count as f64;
                let lower = if exponent == 0.0 { if count == 1 { 0.0 } else { 1.0 } } else { (c - 1.0).powf(exponent) };
                change = 0.5 * (c + 1.0) * c.powf(exponent) - 0.5 * c * lower;
            }
        }
        let column = by_level.map_or(0, |b| b as usize);
        let width = self.by.as_ref().map_or(1, |_| self.n_by);
        let row = if self.diff { level as usize } else { 0 };
        out[row * width + column] += sign * change;
    }
}

/// k-stars centred on one mode whose ends all have the same level, by the
/// levels of the centre and the ends (b1starmix, b2starmix).
struct StarMix {
    k: u32,
    centre: Vec<bool>,
    centre_codes: Vec<i64>,
    leaf_codes: Vec<i64>,
    n_leaf: usize,
    /// Statistic of each (centre level, leaf level), row-major, or -1.
    map: Vec<i64>,
}

impl Term for StarMix {
    fn n_stats(&self) -> usize {
        levels_of(&self.map)
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (c, leaf) = if self.centre[i as usize] { (i, j) } else { (j, i) };
        let (a, b) = (self.centre_codes[c as usize], self.leaf_codes[leaf as usize]);
        if a < 0 || b < 0 {
            return;
        }
        let Ok(s) = usize::try_from(self.map[a as usize * self.n_leaf + b as usize]) else { return };
        let same = net.neighbours(c).iter().filter(|&&u| u != leaf && self.leaf_codes[u as usize] == b).count();
        out[s] += sign * binomial(same as f64, self.k - 1);
    }
}

/// Two-stars centred on one mode by the level of the centre and the
/// (unordered) levels of the two ends (b1twostar, b2twostar).
struct TwoStarMix {
    centre: Vec<bool>,
    centre_codes: Vec<i64>,
    leaf_codes: Vec<i64>,
    n_leaf: usize,
    /// Statistic of each (centre, lower leaf, higher leaf) level, row-major, or -1.
    map: Vec<i64>,
}

impl Term for TwoStarMix {
    fn n_stats(&self) -> usize {
        levels_of(&self.map)
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (c, leaf) = if self.centre[i as usize] { (i, j) } else { (j, i) };
        let (a, b) = (self.centre_codes[c as usize], self.leaf_codes[leaf as usize]);
        if a < 0 || b < 0 {
            return;
        }
        for &u in net.neighbours(c) {
            let d = self.leaf_codes[u as usize];
            if u == leaf || d < 0 {
                continue;
            }
            let (lo, hi) = (b.min(d) as usize, b.max(d) as usize);
            let k = (a as usize * self.n_leaf + lo) * self.n_leaf + hi;
            if let Ok(s) = usize::try_from(self.map[k]) {
                out[s] += sign;
            }
        }
    }
}

// -- Interactions ---------------------------------------------------------------------------------------

/// Products of the change statistics of two sets of dyad-independent terms
/// (ergm's `:`): for each pair of statistics, the first varying fastest.
struct Interaction {
    left: Vec<Box<dyn Term>>,
    right: Vec<Box<dyn Term>>,
    n_left: usize,
    n_right: usize,
}

impl Interaction {
    fn adding(terms: &[Box<dyn Term>], width: usize, net: &Network, i: u32, j: u32) -> Vec<f64> {
        let mut values = vec![0.0; width];
        let mut start = 0;
        for t in terms {
            t.change(net, i, j, 1.0, &mut values[start..start + t.n_stats()]);
            start += t.n_stats();
        }
        values
    }
}

impl Term for Interaction {
    fn n_stats(&self) -> usize {
        self.n_left * self.n_right
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let a = Self::adding(&self.left, self.n_left, net, i, j);
        let b = Self::adding(&self.right, self.n_right, net, i, j);
        for (r, &y) in b.iter().enumerate() {
            for (l, &x) in a.iter().enumerate() {
                out[r * self.n_left + l] += sign * x * y;
            }
        }
    }
}

// -- Building ---------------------------------------------------------------------------------------------

/// Builds a term from its specification (for the terms of an interaction).
pub type BuildTerm<'a> = dyn Fn(&TermSpec) -> Result<Box<dyn Term>, String> + 'a;

// -- Degree correlation, triangle percentage, coincidence -------------------------------------

/// `constant + scale * the sum over ties of the product of their vertices'
/// degrees` (undirected): ergm's degcor and degcrossprod, linearized at the
/// observed network as ergm's change statistics are (the constant and scale
/// make it the observed network's correlation, or mean, there).
struct DegreeCross {
    scale: f64,
    constant: f64,
}

impl Term for DegreeCross {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let degree = |v: u32| net.neighbours(v).len() as f64;
        // The degrees of the other ends of v's ties but the toggled one: each
        // such tie's product changes by that degree.
        let others = |v: u32, skip: u32| net.neighbours(v).iter().filter(|&&k| k != skip).map(|&k| degree(k)).sum::<f64>();
        let (di, dj) = (degree(i), degree(j));
        let change = if sign > 0.0 { (di + 1.0) * (dj + 1.0) } else { di * dj } + others(i, j) + others(j, i);
        out[0] += self.scale * sign * change;
    }

    fn empty(&self, _: u32, _: bool, out: &mut [f64]) {
        out[0] += self.constant;
    }
}

/// 100 times the triangles over the triangles and the two-paths not in a
/// triangle (ergm's tripercent; undirected), counting only ties between
/// vertices with the same code (`codes`), within one level each (`levels`).
struct TriPercent {
    codes: Option<Vec<i64>>,
    /// The level of each statistic; empty: one statistic over all vertices.
    levels: Vec<i64>,
}

impl TriPercent {
    /// Whether the tie u -- v counts towards the statistic of `level`.
    fn counts(&self, level: Option<i64>, u: u32, v: u32) -> bool {
        match &self.codes {
            None => true,
            Some(c) => c[u as usize] == c[v as usize] && level.is_none_or(|l| c[u as usize] == l),
        }
    }

    /// Triangles and two-stars (two-paths, triangles' included) of the ties that count.
    fn totals(&self, net: &Network, level: Option<i64>) -> (f64, f64) {
        let (mut triangles, mut stars) = (0.0, 0.0);
        for u in 0..net.n() {
            let mut degree = 0.0;
            for &v in net.neighbours(u) {
                if !self.counts(level, u, v) {
                    continue;
                }
                degree += 1.0;
                if v > u {
                    for_each_common(net.neighbours(u), net.neighbours(v), |w| {
                        if w > v && self.counts(level, u, w) {
                            triangles += 1.0;
                        }
                    });
                }
            }
            stars += degree * (degree - 1.0) / 2.0;
        }
        (triangles, stars)
    }
}

fn tripercent_ratio(triangles: f64, stars: f64) -> f64 {
    if triangles == 0.0 { 0.0 } else { triangles / (stars - 2.0 * triangles) }
}

impl Term for TriPercent {
    fn n_stats(&self) -> usize {
        self.levels.len().max(1)
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        for (k, stat) in out.iter_mut().enumerate() {
            let level = self.levels.get(k).copied();
            if !self.counts(level, i, j) {
                continue;
            }
            // With and without the tie: the network now, and the change.
            let (triangles, stars) = self.totals(net, level);
            let mut common = 0.0;
            for_each_common(net.neighbours(i), net.neighbours(j), |w| {
                if self.counts(level, i, w) {
                    common += 1.0;
                }
            });
            let degree = |v: u32| net.neighbours(v).iter().filter(|&&u| u != i && u != j && self.counts(level, v, u)).count() as f64;
            // Two-stars the tie centres on i or j: its other ties there.
            let (with, without) = if sign > 0.0 {
                ((triangles + common, stars + degree(i) + degree(j)), (triangles, stars))
            } else {
                ((triangles, stars), (triangles - common, stars - degree(i) - degree(j)))
            };
            *stat += sign * 100.0 * (tripercent_ratio(with.0, with.1) - tripercent_ratio(without.0, without.1));
        }
    }
}

/// For each pair of the second mode's vertices, the number of first-mode
/// vertices tied to both (ergm's coincidence; bipartite networks).
struct Coincidence {
    /// Each vertex's position in the second mode, or -1 (first mode).
    position: Vec<i64>,
    m: usize,
    /// The statistic of each pair (a < b) of positions, at a * m + b; -1 if not counted.
    stat_of: Vec<i64>,
    n_stats: usize,
}

impl Term for Coincidence {
    fn n_stats(&self) -> usize {
        self.n_stats
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (actor, event) = if self.position[i as usize] < 0 { (i, j) } else { (j, i) };
        let a = self.position[event as usize] as usize;
        for &other in net.neighbours(actor) {
            if other == event {
                continue;
            }
            let b = self.position[other as usize] as usize;
            let stat = self.stat_of[a.min(b) * self.m + a.max(b)];
            if stat >= 0 {
                out[stat as usize] += sign;
            }
        }
    }
}

/// The terms of this module by name, or None if `name` isn't one of them.
pub fn build(n: usize, directed: bool, spec: &TermSpec, build_child: &BuildTerm) -> Option<Result<Box<dyn Term>, String>> {
    let TermSpec(name, reals, ints, children) = spec;
    let parts = || chunks(name, ints);
    let part = |p: &[Vec<i64>], k: usize| -> Vec<i64> { p.get(k).cloned().unwrap_or_default() };
    let mask = |values: Vec<i64>| -> Result<Option<Vec<bool>>, String> {
        if values.is_empty() {
            return Ok(None);
        }
        if values.len() != n {
            return Err(format!("{name}: expected {n} vertex mask values"));
        }
        Ok(Some(values.iter().map(|&x| x != 0).collect()))
    };
    let vertex_values = |values: Vec<i64>| -> Result<Vec<i64>, String> {
        if values.len() == n { Ok(values) } else { Err(format!("{name}: expected {n} vertex values")) }
    };
    let optional_codes = |values: Vec<i64>| -> Result<Option<Vec<i64>>, String> {
        if values.is_empty() { Ok(None) } else { vertex_values(values).map(Some) }
    };
    let only = |needs_directed: bool| -> Result<(), String> {
        if directed == needs_directed {
            Ok(())
        } else {
            Err(format!("{name} is only implemented for {} networks", if needs_directed { "directed" } else { "undirected" }))
        }
    };
    let ends = |head: &str, tail: &str| match name.as_str() {
        s if s == head => Ends::Head,
        s if s == tail => Ends::Tail,
        _ => Ends::Both,
    };
    let real_matrix = |values: &[f64]| -> Result<Vec<f64>, String> {
        if values.len() == n * n { Ok(values.to_vec()) } else { Err(format!("{name}: expected an {n} x {n} matrix")) }
    };
    let result = (|| -> Result<Box<dyn Term>, String> {
        Ok(match name.as_str() {
            // chunks: [from], [to] (-1: infinity), [mask], [codes], [homophily]
            "degrange" | "idegrange" | "odegrange" => {
                let p = parts()?;
                let (from, to) = (part(&p, 0), part(&p, 1));
                if from.len() != to.len() || from.is_empty() {
                    return Err(format!("{name}: from and to must have the same length"));
                }
                let ranges = from.iter().zip(&to).map(|(&f, &t)| (f as u32, if t < 0 { u32::MAX } else { t as u32 })).collect();
                let codes = optional_codes(part(&p, 3))?;
                let homophily = part(&p, 4).first() == Some(&1);
                let n_levels = match (&codes, homophily) {
                    (Some(c), false) => levels_of(c),
                    _ => 1,
                };
                Box::new(DegreeRange { ranges, ends: ends("idegrange", "odegrange"), mask: mask(part(&p, 2))?, codes, homophily, n_levels })
            }
            // reals: [power]
            "degreepower" | "idegreepower" | "odegreepower" => {
                let power = *reals.first().ok_or_else(|| format!("{name}: missing power"))?;
                Box::new(DegreePower { power, ends: ends("idegreepower", "odegreepower") })
            }
            // chunks: [mask], [codes]
            "concurrentties" => {
                only(false)?;
                let p = parts()?;
                let codes = optional_codes(part(&p, 1))?;
                let n_levels = codes.as_ref().map_or(1, |c| levels_of(c));
                Box::new(ConcurrentTies { mask: mask(part(&p, 0))?, codes, n_levels })
            }
            "scalededges" => Box::new(ScaledEdges(*reals.first().ok_or("scalededges: missing factor")?)),
            "isolatededges" => {
                only(false)?;
                Box::new(IsolatedEdges)
            }
            // reals: the n x n covariate, symmetric
            "dyadcov" => {
                only(true)?;
                Box::new(DyadCov { x: real_matrix(reals)?, n })
            }
            // reals: the reference network's n x n adjacency matrix, then the weights if any
            "hamming" => {
                if reals.len() != n * n && reals.len() != 2 * n * n {
                    return Err("hamming: expected a reference matrix and optional weights".into());
                }
                let reference = reals[..n * n].iter().map(|&x| x != 0.0).collect();
                let weights = (reals.len() == 2 * n * n).then(|| reals[n * n..].to_vec());
                Box::new(Hamming { reference, weights, n, directed })
            }
            // ints: the vertices' levels; reals: the levels x levels matrix
            "attrcov" => {
                let codes = vertex_values(ints.clone())?;
                let n_levels = levels_of(&codes);
                if reals.len() != n_levels * n_levels {
                    return Err(format!("attrcov: expected a {n_levels} x {n_levels} matrix"));
                }
                Box::new(AttrCov { codes, mat: reals.clone(), n_levels, directed })
            }
            // chunks: [row levels], [column levels], [number of columns], [both], [map]
            "mixmatrix" => {
                let p = parts()?;
                let n_cols = part(&p, 2).first().copied().unwrap_or(0) as usize;
                let map = part(&p, 4);
                if n_cols == 0 || !map.len().is_multiple_of(n_cols) {
                    return Err("mixmatrix: bad map".into());
                }
                let both = part(&p, 3).first() == Some(&1);
                Box::new(MixMatrix { rows: vertex_values(part(&p, 0))?, cols: vertex_values(part(&p, 1))?, n_cols, map, both })
            }
            // reals: x; chunks: [tail, head], [mask]
            "covrange" => {
                let p = parts()?;
                let sides = part(&p, 0);
                if reals.len() != n || sides.len() != 2 {
                    return Err("covrange: bad parameters".into());
                }
                Box::new(CovRange { x: reals.clone(), tail: sides[0] == 1, head: sides[1] == 1, mask: mask(part(&p, 1))? })
            }
            // chunks: [codes], [tail, head], [mask]
            "factordistinct" => {
                let p = parts()?;
                let sides = part(&p, 1);
                if sides.len() != 2 {
                    return Err("factordistinct: bad parameters".into());
                }
                let codes = vertex_values(part(&p, 0))?;
                Box::new(FactorDistinct { codes, tail: sides[0] == 1, head: sides[1] == 1, mask: mask(part(&p, 2))? })
            }
            // reals: x, then pow; ints: [head minus tail, action, orientation], then the first-mode mask
            "diff" => {
                if reals.len() != n + 1 || ints.len() < 3 {
                    return Err("diff: bad parameters".into());
                }
                let first = if ints[2] == 2 { ints[3..].iter().map(|&x| x != 0).collect() } else { Vec::new() };
                if ints[2] == 2 && first.len() != n {
                    return Err("diff: expected a mode mask".into());
                }
                Box::new(Diff { x: reals[..n].to_vec(), pow: reals[n], head_minus_tail: ints[0] == 1, action: ints[1] as u8, orient: ints[2] as u8, first })
            }
            // reals: x, then the cutoff
            "smalldiff" => {
                if reals.len() != n + 1 {
                    return Err("smalldiff: bad parameters".into());
                }
                Box::new(SmallDiff { x: reals[..n].to_vec(), cutoff: reals[n] })
            }
            "altkstar" => {
                only(false)?;
                let lambda = *reals.first().ok_or("altkstar: missing lambda")?;
                Box::new(AltKStar { lambda })
            }
            // chunks: [each vertex's statistic], [codes]
            "factormatch" => {
                let p = parts()?;
                let stat = vertex_values(part(&p, 0))?;
                let n_stats = levels_of(&stat);
                Box::new(FactorMatch { stat, codes: vertex_values(part(&p, 1))?, n_stats })
            }
            // ints: the statistic of each triad type (16, or 4 if undirected), or -1
            "triadcensus" => {
                if ints.len() != if directed { 16 } else { 4 } {
                    return Err("triadcensus: bad map".into());
                }
                Box::new(TriadCensus::new(ints.clone(), directed))
            }
            "simmelianties" => {
                only(true)?;
                Box::new(SimmelianTies)
            }
            // ints: the directed types reported (1 RRR, 2 RRL, 3 LRR, 4 LRL)
            "threetrail" => {
                let types: Vec<usize> = ints.iter().map(|&t| t as usize).collect();
                if directed && (types.is_empty() || types.iter().any(|&t| !(1..=4).contains(&t))) {
                    return Err("threetrail: types must be 1 to 4".into());
                }
                Box::new(ThreeTrail { types: if directed { types } else { vec![1] } })
            }
            // chunks: [cyclical], [codes]
            "supportedties" => {
                let p = parts()?;
                let codes = optional_codes(part(&p, 1))?;
                Box::new(SupportedTies { cyclical: part(&p, 0).first() == Some(&1), codes })
            }
            "opentriad" => {
                only(false)?;
                Box::new(OpenTriad)
            }
            // reals: the n x n neighbourhood matrix
            "localtriangle" => Box::new(LocalTriangle { x: real_matrix(reals)?.iter().map(|&v| v != 0.0).collect(), n }),
            // chunks: [ks], [codes], [mask]
            "kstarmatch" | "istarmatch" | "ostarmatch" => {
                let p = parts()?;
                let ks: Vec<u32> = part(&p, 0).iter().map(|&k| k as u32).collect();
                if ks.is_empty() || ks.contains(&0) {
                    return Err(format!("{name}: k must be 1 or more"));
                }
                Box::new(StarsMatch { ks, ends: ends("istarmatch", "ostarmatch"), codes: vertex_values(part(&p, 1))?, mask: mask(part(&p, 2))? })
            }
            // chunks: [codes], [diff, kind]
            "trianglesmatch" => {
                let p = parts()?;
                let codes = vertex_values(part(&p, 0))?;
                let flags = part(&p, 1);
                let n_levels = levels_of(&codes);
                Box::new(TrianglesMatch { codes, diff: flags.first() == Some(&1), n_levels, kind: flags.get(1).copied().unwrap_or(0) as u8 })
            }
            // chunks: [codes], [diff, by]
            "mutualmatch" => {
                only(true)?;
                let p = parts()?;
                let codes = vertex_values(part(&p, 0))?;
                let flags = part(&p, 1);
                let n_levels = levels_of(&codes);
                Box::new(MutualMatch { codes, diff: flags.first() == Some(&1), by: flags.get(1) == Some(&1), n_levels })
            }
            // chunks: [codes], [diff]
            "asymmetricmatch" => {
                only(true)?;
                let p = parts()?;
                let codes = vertex_values(part(&p, 0))?;
                let n_levels = levels_of(&codes);
                Box::new(AsymmetricMatch { codes, diff: part(&p, 1).first() == Some(&1), n_levels })
            }
            // reals: [decay]; chunks: [codes], [mask]
            "gwdegreematch" | "gwidegreematch" | "gwodegreematch" => {
                let p = parts()?;
                let decay = *reals.first().ok_or_else(|| format!("{name}: missing decay"))?;
                let codes = vertex_values(part(&p, 0))?;
                let n_levels = levels_of(&codes);
                let ends = ends("gwidegreematch", "gwodegreematch");
                Box::new(GwDegreeMatch { r: 1.0 - (-decay).exp(), ends, codes, mask: mask(part(&p, 1))?, n_levels })
            }
            // reals: [alpha, beta]; chunks: [end mask], [end codes], [diff], [centre codes]
            "nodematchbipartite" => {
                only(false)?;
                let p = parts()?;
                let ends = vertex_values(part(&p, 0))?.iter().map(|&x| x != 0).collect();
                let codes = vertex_values(part(&p, 1))?;
                let by = optional_codes(part(&p, 3))?;
                let (alpha, beta) = (reals.first().copied().unwrap_or(1.0), reals.get(1).copied().unwrap_or(1.0));
                let n_levels = levels_of(&codes);
                let n_by = by.as_ref().map_or(1, |b| levels_of(b));
                Box::new(NodeMatchBipartite { ends, codes, alpha, beta, diff: part(&p, 2).first() == Some(&1), n_levels, by, n_by })
            }
            // chunks: [k], [centre mask], [centre codes], [leaf codes], [n_leaf], [map]
            "starmix" => {
                only(false)?;
                let p = parts()?;
                let k = part(&p, 0).first().copied().unwrap_or(0);
                if k < 1 {
                    return Err("starmix: k must be 1 or more".into());
                }
                let n_leaf = part(&p, 4).first().copied().unwrap_or(0) as usize;
                Box::new(StarMix {
                    k: k as u32,
                    centre: part(&p, 1).iter().map(|&x| x != 0).collect(),
                    centre_codes: vertex_values(part(&p, 2))?,
                    leaf_codes: vertex_values(part(&p, 3))?,
                    n_leaf,
                    map: part(&p, 5),
                })
            }
            // chunks: [centre mask], [centre codes], [leaf codes], [n_leaf], [map]
            "twostarmix" => {
                only(false)?;
                let p = parts()?;
                let n_leaf = part(&p, 3).first().copied().unwrap_or(0) as usize;
                Box::new(TwoStarMix {
                    centre: part(&p, 0).iter().map(|&x| x != 0).collect(),
                    centre_codes: vertex_values(part(&p, 1))?,
                    leaf_codes: vertex_values(part(&p, 2))?,
                    n_leaf,
                    map: part(&p, 4),
                })
            }
            // ints: [terms on the left]; children: the terms of both sides
            "interact" => {
                let left_count = ints.first().copied().unwrap_or(0) as usize;
                if left_count == 0 || left_count >= children.len() {
                    return Err("interact: needs terms on both sides".into());
                }
                let built = children.iter().map(build_child).collect::<Result<Vec<_>, _>>()?;
                let mut built = built.into_iter();
                let left: Vec<_> = built.by_ref().take(left_count).collect();
                let right: Vec<_> = built.collect();
                let n_left = left.iter().map(|t| t.n_stats()).sum();
                let n_right = right.iter().map(|t| t.n_stats()).sum();
                Box::new(Interaction { left, right, n_left, n_right })
            }
            // reals: [scale, constant]
            "degreecross" => {
                only(false)?;
                let [scale, constant] = reals[..] else { return Err("degreecross: expected the scale and constant".into()) };
                Box::new(DegreeCross { scale, constant })
            }
            // chunks: [codes], [levels]
            "tripercent" => {
                only(false)?;
                let p = parts()?;
                Box::new(TriPercent { codes: optional_codes(part(&p, 0))?, levels: part(&p, 1) })
            }
            // chunks: [each vertex's position in the second mode, -1 for the first], [the pairs' statistics, m x m]
            "coincidence" => {
                only(false)?;
                let p = parts()?;
                let position = vertex_values(part(&p, 0))?;
                let m = position.iter().filter(|&&x| x >= 0).count();
                let stat_of = part(&p, 1);
                if stat_of.len() != m * m {
                    return Err(format!("coincidence: expected {} pair values", m * m));
                }
                let n_stats = (stat_of.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize;
                Box::new(Coincidence { position, m, stat_of, n_stats })
            }
            _ => return Err(String::new()),
        })
    })();
    match result {
        Err(e) if e.is_empty() => None,
        other => Some(other),
    }
}

#[cfg(test)]
mod tests {
    use super::triad_type;

    #[test]
    fn triad_types_cover_the_sixteen() {
        let mut seen = [0usize; 16];
        for bits in 0..64usize {
            let tie = |a: usize, b: usize| {
                let k = [[9, 0, 2], [1, 9, 4], [3, 5, 9]][a][b];
                bits >> k & 1 == 1
            };
            seen[triad_type(tie)] += 1;
        }
        // Isomorphism classes of labelled triads: their sizes sum to 64.
        assert_eq!(seen, [1, 6, 3, 3, 3, 6, 6, 6, 6, 2, 3, 3, 3, 6, 6, 1]);
    }
}
