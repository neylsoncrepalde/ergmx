//! Metropolis-Hastings sampling of networks.
//!
//! Proposals mix two moves:
//!
//! * TNT (tie / no tie; Morris, Handcock and Hunter 2008, ergm's MH_TNT):
//!   toggle a random existing tie with probability 1/2, a random dyad otherwise;
//! * triadic: pick a random vertex i, a random neighbour k of i and a random
//!   neighbour j != i of k, and toggle (i, j) (in directed networks, neighbours
//!   in either direction, and i -> j or j -> i with probability 1/2). This
//!   closes or opens a triangle, which is what models with triangle or gwesp
//!   terms need to mix well.
//!
//! The acceptance ratio uses the exact probability that the mixture proposes
//! the dyad, before and after the toggle.

use crate::network::{Network, for_each_common};
use crate::rng::Rng;
use crate::terms::Model;

/// Probability that a TNT move toggles a random existing tie.
const TIE_PROB: f64 = 0.5;

pub struct Chain {
    /// Sampled statistics, one row of `n_stats` values per sample.
    pub stats: Vec<f64>,
    /// Edges of each sampled network, if requested.
    pub networks: Vec<Vec<(u32, u32)>>,
    /// The network at the end of the chain.
    pub last: Network,
}

/// Probability that a TNT move proposes one given dyad.
fn tnt_prob(edges: f64, dyads: f64, tie: bool) -> f64 {
    if tie {
        TIE_PROB / edges + (1.0 - TIE_PROB) / dyads
    } else if edges == 0.0 {
        1.0 / dyads
    } else {
        (1.0 - TIE_PROB) / dyads
    }
}

/// Probability that a triadic move proposes the dyad (i, j), given the degrees
/// of i and j and `s`, the sum of 1 / (degree - 1) over their shared partners.
/// Degrees count neighbours in either direction; `directions` is 2 in directed
/// networks, where each walk picks one of the two ties between i and j.
fn triadic_prob(n: f64, directions: f64, degree_i: f64, degree_j: f64, s: f64) -> f64 {
    if s == 0.0 { 0.0 } else { (1.0 / degree_i + 1.0 / degree_j) * s / (n * directions) }
}

fn triadic_draw(net: &Network, rng: &mut Rng) -> Option<(u32, u32)> {
    let i = rng.below(net.n() as u64) as u32;
    let ni = net.neighbours(i);
    if ni.is_empty() {
        return None;
    }
    let k = ni[rng.below(ni.len() as u64) as usize];
    let nk = net.neighbours(k);
    if nk.len() < 2 {
        return None;
    }
    // A random neighbour of k other than i.
    let skip = nk.binary_search(&i).unwrap();
    let mut r = rng.below(nk.len() as u64 - 1) as usize;
    if r >= skip {
        r += 1;
    }
    let j = nk[r];
    Some(if net.directed() && rng.unif() < 0.5 { (j, i) } else { (i, j) })
}

pub struct Proposal {
    /// Share of triadic moves; 0 gives plain TNT.
    pub triadic_weight: f64,
}

impl Proposal {
    /// Draws a dyad to toggle and returns it with log q(y' -> y) / q(y -> y'),
    /// or None if the triadic move found no dyad (the chain stays put).
    fn propose(&self, net: &Network, rng: &mut Rng) -> Option<(u32, u32, f64)> {
        let w = self.triadic_weight;
        let edges = net.n_edges() as f64;
        let dyads = net.n_dyads() as f64;
        let (i, j) = if w > 0.0 && rng.unif() < w {
            triadic_draw(net, rng)?
        } else if edges > 0.0 && rng.unif() < TIE_PROB {
            net.random_edge(rng)
        } else {
            net.random_dyad(rng)
        };
        let tie = net.has_edge(i, j);
        if w == 0.0 {
            let after = edges + if tie { -1.0 } else { 1.0 };
            let ratio = tnt_prob(after, dyads, !tie) / tnt_prob(edges, dyads, tie);
            return Some((i, j, ratio.ln()));
        }
        let (ni, nj) = (net.neighbours(i), net.neighbours(j));
        let mut s = 0.0;
        for_each_common(ni, nj, |k| s += 1.0 / (net.neighbours(k).len() - 1) as f64);
        let (di, dj, n) = (ni.len() as f64, nj.len() as f64, net.n() as f64);
        let directions = if net.directed() { 2.0 } else { 1.0 };
        let change = if tie { -1.0 } else { 1.0 };
        // i and j stay neighbours if the opposite tie j -> i exists.
        let degree_change = if net.directed() && net.has_edge(j, i) { 0.0 } else { change };
        let forward = w * triadic_prob(n, directions, di, dj, s)
            + (1.0 - w) * tnt_prob(edges, dyads, tie);
        let reverse = w * triadic_prob(n, directions, di + degree_change, dj + degree_change, s)
            + (1.0 - w) * tnt_prob(edges + change, dyads, !tie);
        Some((i, j, (reverse / forward).ln()))
    }
}

/// Runs `burnin` steps, then records `samplesize` samples `interval` steps apart.
#[allow(clippy::too_many_arguments)]
pub fn run_chain(
    model: &Model,
    proposal: &Proposal,
    mut net: Network,
    theta: &[f64],
    burnin: u64,
    interval: u64,
    samplesize: usize,
    rng: &mut Rng,
    keep_networks: bool,
) -> Chain {
    let mut stats = model.summary(&net);
    let mut delta = vec![0.0; stats.len()];
    let mut sample = Vec::with_capacity(samplesize * stats.len());
    let mut networks = Vec::new();
    for step in 1..=burnin + interval * samplesize as u64 {
        if let Some((i, j, log_q)) = proposal.propose(&net, rng) {
            model.change(&net, i, j, &mut delta);
            let log_accept = log_q + theta.iter().zip(&delta).map(|(t, d)| t * d).sum::<f64>();
            if log_accept >= 0.0 || rng.unif() < log_accept.exp() {
                net.toggle(i, j);
                stats.iter_mut().zip(&delta).for_each(|(s, d)| *s += d);
            }
        }
        if step > burnin && (step - burnin).is_multiple_of(interval) {
            sample.extend_from_slice(&stats);
            if keep_networks {
                networks.push(net.edges().to_vec());
            }
        }
    }
    Chain { stats: sample, networks, last: net }
}
