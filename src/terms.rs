//! Model terms and their change statistics.
//!
//! A term contributes one or more statistics. `change` adds to `out` how its
//! statistics change when the dyad (i, j) is toggled in `net`; `sign` is +1 if
//! the tie is being added and -1 if it is being removed. Definitions follow
//! the R package ergm.

use crate::network::{Network, count_common, for_each_common};

pub trait Term: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]);
}

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

/// Number of triangles (undirected networks).
struct Triangle;

impl Term for Triangle {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * count_common(net.neighbours(i), net.neighbours(j)) as f64;
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

/// Number of ties between vertices with the same attribute value.
struct NodeMatch {
    codes: Vec<i64>,
}

impl Term for NodeMatch {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        if self.codes[i as usize] == self.codes[j as usize] {
            out[0] += sign;
        }
    }
}

/// For each attribute level, the number of tie endpoints at that level.
/// Vertices with a negative code (the base level) are not counted.
struct NodeFactor {
    codes: Vec<i64>,
    n_levels: usize,
}

impl Term for NodeFactor {
    fn n_stats(&self) -> usize {
        self.n_levels
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        for v in [i, j] {
            if let Ok(level) = usize::try_from(self.codes[v as usize]) {
                out[level] += sign;
            }
        }
    }
}

/// Sum over ties of the attribute values of both endpoints.
struct NodeCov {
    x: Vec<f64>,
}

impl Term for NodeCov {
    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        out[0] += sign * (self.x[i as usize] + self.x[j as usize]);
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

/// A term as sent from Python: its name, real parameters and integer parameters.
pub type TermSpec = (String, Vec<f64>, Vec<i64>);

fn build_term(n: usize, directed: bool, spec: &TermSpec) -> Result<Box<dyn Term>, String> {
    let (name, reals, ints) = spec;
    let per_vertex = |len: usize| {
        if len == n {
            Ok(())
        } else {
            Err(format!("{name}: expected {n} vertex values, got {len}"))
        }
    };
    let only = |needs_directed: bool| {
        if directed == needs_directed {
            Ok(())
        } else {
            let kind = if needs_directed { "directed" } else { "undirected" };
            Err(format!("{name} is only implemented for {kind} networks"))
        }
    };
    Ok(match name.as_str() {
        "edges" => Box::new(Edges),
        "mutual" => {
            only(true)?;
            Box::new(Mutual)
        }
        "triangle" => {
            only(false)?;
            Box::new(Triangle)
        }
        "gwesp" => {
            only(false)?;
            let decay = *reals.first().ok_or("gwesp: missing decay")?;
            Box::new(Gwesp { exp_decay: decay.exp(), r: 1.0 - (-decay).exp() })
        }
        "nodematch" => {
            per_vertex(ints.len())?;
            Box::new(NodeMatch { codes: ints.clone() })
        }
        "nodefactor" => {
            per_vertex(ints.len())?;
            let n_levels = ints.iter().copied().max().unwrap_or(-1).max(-1) + 1;
            Box::new(NodeFactor { codes: ints.clone(), n_levels: n_levels as usize })
        }
        "nodecov" => {
            per_vertex(reals.len())?;
            Box::new(NodeCov { x: reals.clone() })
        }
        "absdiff" => {
            per_vertex(reals.len())?;
            Box::new(AbsDiff { x: reals.clone() })
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
