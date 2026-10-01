//! Model terms and their change statistics.
//!
//! A term contributes one or more statistics. `change` adds to `out` how its
//! statistics change when the tie i -> j (or i -- j) is toggled in `net`;
//! `sign` is +1 if the tie is being added and -1 if it is being removed.
//! Definitions follow the R package ergm.

use crate::network::{Network, count_common, for_each_common};
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

/// Shared partner counts change by one for some ties or pairs when (i, j) is
/// toggled: `base` is a count without the tie (i, j). Adds the change in the
/// number of ties (or pairs) with exactly k shared partners, for each k.
#[inline]
fn shift_histogram(ks: &[u32], base: u32, sign: f64, out: &mut [f64]) {
    for (stat, &k) in out.iter_mut().zip(ks) {
        if k == base + 1 {
            *stat += sign;
        } else if k == base {
            *stat -= sign;
        }
    }
}

/// Number of ties with exactly k edgewise shared partners, for each k
/// (undirected networks). See Gwesp for which ties change.
struct Esp {
    ks: Vec<u32>,
}

impl Term for Esp {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (ni, nj) = (net.neighbours(i), net.neighbours(j));
        let own = if sign < 0.0 { 1 } else { 0 };
        let shared = count_common(ni, nj);
        for (stat, &k) in out.iter_mut().zip(&self.ks) {
            if k == shared {
                *stat += sign;
            }
        }
        let mut bases = Vec::new();
        for_each_common(ni, nj, |u| {
            let nu = net.neighbours(u);
            bases.push(count_common(ni, nu) - own);
            bases.push(count_common(nj, nu) - own);
        });
        for base in bases {
            shift_histogram(&self.ks, base, sign, out);
        }
    }
}

/// esp for directed networks, with outgoing two-path (OTP) shared partners.
/// See GwespOtp for which ties change.
struct EspOtp {
    ks: Vec<u32>,
}

impl Term for EspOtp {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (out_i, in_i) = (net.out_neighbours(i), net.in_neighbours(i));
        let (out_j, in_j) = (net.out_neighbours(j), net.in_neighbours(j));
        let own = if sign < 0.0 { 1 } else { 0 };
        let shared = count_common(out_i, in_j);
        for (stat, &k) in out.iter_mut().zip(&self.ks) {
            if k == shared {
                *stat += sign;
            }
        }
        let mut bases = Vec::new();
        for_each_common(out_i, out_j, |v| bases.push(count_common(out_i, net.in_neighbours(v)) - own));
        for_each_common(in_i, in_j, |u| bases.push(count_common(net.out_neighbours(u), in_j) - own));
        for base in bases {
            shift_histogram(&self.ks, base, sign, out);
        }
    }
}

/// Calls `f` with the shared partner count, without the tie (i, j), of every
/// pair of vertices that gains or loses a shared partner when (i, j) is toggled.
/// Undirected: the pairs (i, u) with u a neighbour of j, and (j, u) with u a
/// neighbour of i. Directed (OTP): the pairs (i, v) with j -> v, and (u, j)
/// with u -> i.
fn for_each_dyad_partner_change(net: &Network, i: u32, j: u32, removing: bool, mut f: impl FnMut(u32)) {
    let own = removing as u32;
    if net.directed() {
        let out_i = net.out_neighbours(i);
        for &v in net.out_neighbours(j).iter().filter(|&&v| v != i) {
            f(count_common(out_i, net.in_neighbours(v)) - own);
        }
        let in_j = net.in_neighbours(j);
        for &u in net.in_neighbours(i).iter().filter(|&&u| u != j) {
            f(count_common(net.out_neighbours(u), in_j) - own);
        }
    } else {
        let (ni, nj) = (net.neighbours(i), net.neighbours(j));
        for (a, na, nb) in [(i, ni, nj), (j, nj, ni)] {
            for &u in nb.iter().filter(|&&u| u != a) {
                f(count_common(na, net.neighbours(u)) - own);
            }
        }
    }
}

/// Number of pairs of vertices with exactly k shared partners, for each k
/// (OTP shared partners and ordered pairs if directed).
struct Dsp {
    ks: Vec<u32>,
}

impl Term for Dsp {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn empty(&self, n: u32, directed: bool, out: &mut [f64]) {
        let pairs = n as f64 * (n as f64 - 1.0) / if directed { 1.0 } else { 2.0 };
        for (stat, &k) in out.iter_mut().zip(&self.ks) {
            if k == 0 {
                *stat += pairs;
            }
        }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        for_each_dyad_partner_change(net, i, j, sign < 0.0, |base| {
            shift_histogram(&self.ks, base, sign, out)
        });
    }
}

/// gwdsp for directed networks, with OTP shared partners over ordered pairs.
struct GwdspOtp {
    r: f64,
}

impl Term for GwdspOtp {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let mut total = 0.0;
        for_each_dyad_partner_change(net, i, j, sign < 0.0, |base| total += self.r.powi(base as i32));
        out[0] += sign * total;
    }
}

/// Number of vertices with degree exactly k, for each k: degrees (degree),
/// in-degrees (idegree) or out-degrees (odegree).
struct DegreeCount {
    ks: Vec<u32>,
    ends: Ends,
}

impl Term for DegreeCount {
    fn n_stats(&self) -> usize {
        self.ks.len()
    }

    fn empty(&self, n: u32, _: bool, out: &mut [f64]) {
        for (stat, &k) in out.iter_mut().zip(&self.ks) {
            if k == 0 {
                *stat += n as f64;
            }
        }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (degrees, count) = self.ends.degrees(net, i, j);
        for &(_, degree) in &degrees[..count] {
            let old = degree as i64;
            let new = old + sign as i64;
            for (stat, &k) in out.iter_mut().zip(&self.ks) {
                *stat += (new == k as i64) as u8 as f64 - (old == k as i64) as u8 as f64;
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
    let counts = |min: i64| -> Result<Vec<u32>, String> {
        if ints.is_empty() || ints.iter().any(|&k| k < min) {
            return Err(format!("{name}: k must be one or more integers >= {min}"));
        }
        Ok(ints.iter().map(|&k| k as u32).collect())
    };
    let ks = || counts(1);
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
        "degree" => {
            only(false)?;
            Box::new(DegreeCount { ks: counts(0)?, ends: Ends::Both })
        }
        "idegree" | "odegree" => {
            only(true)?;
            let ends = if name == "idegree" { Ends::Head } else { Ends::Tail };
            Box::new(DegreeCount { ks: counts(0)?, ends })
        }
        "isolates" => Box::new(Isolates),
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
        "esp" if directed => Box::new(EspOtp { ks: counts(0)? }),
        "esp" => Box::new(Esp { ks: counts(0)? }),
        "gwdsp" if directed => Box::new(GwdspOtp { r: r(decay()?) }),
        "gwdsp" => Box::new(Gwdsp { r: r(decay()?) }),
        "dsp" => Box::new(Dsp { ks: counts(0)? }),
        "nodematch" | "nodematchdiff" => {
            expect(ints.len(), "vertex values", n)?;
            Box::new(NodeMatch { codes: ints.clone(), n_levels: levels(), diff: name == "nodematchdiff" })
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
