//! Shared partner terms: esp, dsp, nsp and their geometrically weighted
//! versions, for undirected networks and the five types of shared partner of
//! directed networks, as in ergm.
//!
//! A shared partner of the pair (a, b) is a vertex k with, by type:
//!
//! * undirected: k -- a and k -- b;
//! * OTP (outgoing two-path): a -> k -> b;
//! * ITP (incoming two-path): b -> k -> a;
//! * RTP (reciprocated two-path): a <-> k <-> b;
//! * OSP (outgoing shared partner): a -> k and b -> k;
//! * ISP (incoming shared partner): k -> a and k -> b.
//!
//! esp counts the pairs that are ties (a -> b), dsp all pairs (ordered pairs
//! if directed), nsp the pairs that are not ties. Toggling the tie i -> j adds
//! or removes the partner i or j of some pairs; `for_each_affected` lists
//! them, and each pair's count without the toggled tie is its *base*.

use crate::network::{Network, Partners, count_common, for_each_common};
use crate::terms::Term;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpType {
    Undirected,
    Otp,
    Itp,
    Rtp,
    Osp,
    Isp,
}

impl SpType {
    pub fn from_code(code: i64, directed: bool) -> Result<Self, String> {
        Ok(match (code, directed) {
            (_, false) => SpType::Undirected,
            (1, true) => SpType::Otp,
            (2, true) => SpType::Itp,
            (3, true) => SpType::Rtp,
            (4, true) => SpType::Osp,
            (5, true) => SpType::Isp,
            _ => return Err(format!("unknown shared partner type {code}")),
        })
    }

    /// The kind of shared partner counts a network can keep for this type.
    pub fn kept(self) -> Option<Partners> {
        match self {
            SpType::Undirected => Some(Partners::Undirected),
            SpType::Otp | SpType::Itp => Some(Partners::TwoPaths),
            SpType::Osp => Some(Partners::OutShared),
            SpType::Isp => Some(Partners::InShared),
            SpType::Rtp => None,
        }
    }
}

/// Which pairs a term counts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Pairs that are ties: esp.
    Edgewise,
    /// All pairs: dsp.
    Dyadwise,
    /// Pairs that are not ties: nsp.
    NonEdgewise,
}

/// Shared partners of the ordered pair (a, b): kept by the network, or counted.
#[inline]
fn count(net: &Network, t: SpType, a: u32, b: u32) -> u32 {
    let kept = match t {
        SpType::Itp => net.kept_partners(Partners::TwoPaths, b, a),
        _ => t.kept().and_then(|kind| net.kept_partners(kind, a, b)),
    };
    if let Some(c) = kept {
        return c;
    }
    let (out, inn) = (|v| net.out_neighbours(v), |v| net.in_neighbours(v));
    match t {
        SpType::Undirected => count_common(net.neighbours(a), net.neighbours(b)),
        SpType::Otp => count_common(out(a), inn(b)),
        SpType::Itp => count_common(inn(a), out(b)),
        SpType::Osp => count_common(out(a), out(b)),
        SpType::Isp => count_common(inn(a), inn(b)),
        SpType::Rtp => {
            let mut c = 0;
            for_each_common(out(a), out(b), |k| {
                if net.has_edge(k, a) && net.has_edge(k, b) {
                    c += 1;
                }
            });
            c
        }
    }
}

/// Calls `f(x, v)` (or `f(v, x)` with `flip`) for each v in `from`, also in
/// `ties` if `edgewise`, other than `skip`.
#[inline(always)]
fn visit<F: FnMut(u32, u32)>(from: &[u32], ties: &[u32], skip: u32, flip: bool, x: u32, edgewise: bool, f: &mut F) {
    let mut call = |v: u32| {
        if v != skip {
            if flip { f(v, x) } else { f(x, v) }
        }
    };
    if edgewise {
        for_each_common(from, ties, &mut call);
    } else {
        from.iter().for_each(|&v| call(v));
    }
}

/// Calls `f(a, b)` for every pair whose shared partner count changes when the
/// tie i -> j (or i -- j) is toggled; with `edgewise`, only those that are ties.
#[inline(always)]
fn for_each_affected<F: FnMut(u32, u32)>(net: &Network, t: SpType, i: u32, j: u32, edgewise: bool, mut f: F) {
    let (out, inn) = (|v| net.out_neighbours(v), |v| net.in_neighbours(v));
    let e = edgewise;
    match t {
        SpType::Undirected => {
            let (ni, nj) = (net.neighbours(i), net.neighbours(j));
            if edgewise {
                // The ties (i, u) and (j, u) of the common neighbours u: one merge.
                for_each_common(ni, nj, |u| {
                    f(i, u);
                    f(j, u);
                });
            } else {
                // Partner j of (i, u) for u -- j; partner i of (j, u) for u -- i.
                visit(nj, ni, i, false, i, e, &mut f);
                visit(ni, nj, j, false, j, e, &mut f);
            }
        }
        SpType::Otp => {
            visit(out(j), out(i), i, false, i, e, &mut f); // i -> j -> v
            visit(inn(i), inn(j), j, true, j, e, &mut f); // u -> i -> j
        }
        SpType::Itp => {
            visit(inn(i), out(j), j, false, j, e, &mut f); // (j, b): b -> i -> j
            visit(out(j), inn(i), i, true, i, e, &mut f); // (a, i): i -> j -> a
        }
        SpType::Osp => {
            visit(inn(j), out(i), i, false, i, e, &mut f); // (i, b): i -> j, b -> j
            visit(inn(j), inn(i), i, true, i, e, &mut f); // (a, i): a -> j, i -> j
        }
        SpType::Isp => {
            visit(out(i), out(j), j, false, j, e, &mut f); // (j, b): i -> j, i -> b
            visit(out(i), inn(j), j, true, j, e, &mut f); // (a, j): i -> a, i -> j
        }
        SpType::Rtp => {
            // Only a reciprocated pair i <-> j can be a reciprocated two-path.
            if !net.has_edge(j, i) {
                return;
            }
            for (k, other) in [(j, i), (i, j)] {
                let mut mutual = Vec::new();
                for_each_common(out(k), inn(k), |v| mutual.push(v));
                for &b in mutual.iter().filter(|&&b| b != other) {
                    if !edgewise || net.has_edge(other, b) {
                        f(other, b);
                    }
                    if !edgewise || net.has_edge(b, other) {
                        f(b, other);
                    }
                }
            }
        }
    }
}

/// Maps counts (of shared partners, or degrees) to histogram statistics.
pub struct Bins {
    /// Statistic of each count from 0 to the largest listed, or -1.
    table: Vec<i32>,
    /// Statistic of every count above the largest listed, if any.
    overflow: Option<usize>,
    n: usize,
}

impl Bins {
    /// `ks` are the counts with a statistic each; with `overflow`, one more
    /// statistic counts everything above the largest.
    pub fn new(ks: &[u32], overflow: bool) -> Self {
        let max = ks.iter().copied().max().unwrap_or(0) as usize;
        let mut table = vec![-1; max + 1];
        for (stat, &k) in ks.iter().enumerate() {
            table[k as usize] = stat as i32;
        }
        let overflow = overflow.then_some(ks.len());
        Self { table, overflow, n: ks.len() + overflow.is_some() as usize }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn index(&self, count: u32) -> Option<usize> {
        match self.table.get(count as usize) {
            Some(&s) if s >= 0 => Some(s as usize),
            Some(_) => None,
            None => self.overflow,
        }
    }

    /// Adds the change from one more partner (a count going from base to
    /// base + 1, if sign > 0), or one less.
    #[inline]
    pub fn shift(&self, base: u32, sign: f64, out: &mut [f64]) {
        let (from, to) = (self.index(base), self.index(base + 1));
        if from != to {
            if let Some(s) = from {
                out[s] -= sign;
            }
            if let Some(s) = to {
                out[s] += sign;
            }
        }
    }
}

pub enum Weight {
    /// exp(decay) * sum of 1 - r^count, with r = 1 - exp(-decay), and the
    /// powers r^k for counts up to the network size.
    Geometric { r: f64, exp_decay: f64, powers: Vec<f64> },
    Histogram(Bins),
}

impl Weight {
    pub fn geometric(decay: f64, r: f64, n: usize) -> Self {
        let powers = (0..=n).scan(1.0, |p, _| {
            let now = *p;
            *p *= r;
            Some(now)
        });
        Weight::Geometric { r, exp_decay: decay.exp(), powers: powers.collect() }
    }
}

#[inline]
fn power(powers: &[f64], r: f64, k: u32) -> f64 {
    powers.get(k as usize).copied().unwrap_or_else(|| r.powi(k as i32))
}

pub struct SharedPartners {
    pub kind: SpType,
    pub scope: Scope,
    pub weight: Weight,
    /// For bipartite networks: only pairs of vertices of this mode (dyadwise).
    pub mode: Option<Vec<bool>>,
}

impl SharedPartners {
    /// The change in one scope, from toggling (i, j).
    fn add_change(&self, net: &Network, i: u32, j: u32, sign: f64, scope: Scope, out: &mut [f64]) {
        let own = (sign < 0.0) as u32;
        let t = self.kind;
        let counted = |a: u32| self.mode.as_ref().is_none_or(|m| m[a as usize]);
        match &self.weight {
            Weight::Geometric { r, exp_decay, powers } => {
                let mut total = 0.0;
                if scope == Scope::Edgewise {
                    total += exp_decay * (1.0 - power(powers, *r, count(net, t, i, j)));
                }
                for_each_affected(net, t, i, j, scope == Scope::Edgewise, |a, b| {
                    if counted(a) {
                        total += power(powers, *r, count(net, t, a, b) - own);
                    }
                });
                out[0] += sign * total;
            }
            Weight::Histogram(bins) => {
                if scope == Scope::Edgewise
                    && let Some(s) = bins.index(count(net, t, i, j))
                {
                    out[s] += sign;
                }
                for_each_affected(net, t, i, j, scope == Scope::Edgewise, |a, b| {
                    if counted(a) {
                        bins.shift(count(net, t, a, b) - own, sign, out);
                    }
                });
            }
        }
    }
}

impl Term for SharedPartners {
    fn partners(&self) -> Option<Partners> {
        self.kind.kept()
    }

    fn n_stats(&self) -> usize {
        match &self.weight {
            Weight::Geometric { .. } => 1,
            Weight::Histogram(bins) => bins.len(),
        }
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        match self.scope {
            Scope::NonEdgewise => {
                // Pairs that are not ties: all pairs minus the ties.
                self.add_change(net, i, j, sign, Scope::Dyadwise, out);
                let mut edgewise = vec![0.0; out.len()];
                self.add_change(net, i, j, sign, Scope::Edgewise, &mut edgewise);
                out.iter_mut().zip(&edgewise).for_each(|(o, e)| *o -= e);
            }
            scope => self.add_change(net, i, j, sign, scope, out),
        }
    }

    fn empty(&self, n: u32, directed: bool, out: &mut [f64]) {
        // In the empty network, every pair has 0 shared partners.
        if self.scope == Scope::Edgewise {
            return;
        }
        let vertices = self.mode.as_ref().map_or(n as f64, |m| m.iter().filter(|&&x| x).count() as f64);
        let pairs = vertices * (vertices - 1.0) / if directed { 1.0 } else { 2.0 };
        if let Weight::Histogram(bins) = &self.weight
            && let Some(s) = bins.index(0)
        {
            out[s] += pairs;
        }
    }
}
