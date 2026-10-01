//! Metropolis-Hastings sampling of networks.
//!
//! Without degree constraints, proposals mix two moves:
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
//! the dyad, before and after the toggle. Only free dyads are proposed, and
//! toggles that break degree bounds are rejected.
//!
//! Degree-preserving constraints use moves that keep the degrees:
//!
//! * `degrees`: swap the endpoints of two ties (a -- b, c -- d become
//!   a -- d, c -- b), and, in directed networks, also reverse cyclic triples
//!   (a -> b -> c -> a becomes a -> c -> b -> a), without which some networks
//!   with the same in- and out-degrees can't reach each other (Rao, Jana and
//!   Bandyopadhyay 1996);
//! * `odegrees` (`idegrees`): move the head (tail) of a tie to another vertex.

use crate::network::{Network, count_common, for_each_common};
use crate::rng::Rng;
use crate::space::{EdgeIndex, Preserve, Space};
use crate::terms::{Model, State};

/// Probability that a TNT move toggles a random existing tie.
const TIE_PROB: f64 = 0.5;
/// Share of cyclic triple reversals among degree-preserving moves (directed).
const CYCLE_PROB: f64 = 0.2;

pub struct Chain {
    /// Sampled statistics, one row of `n_stats` values per sample.
    pub stats: Vec<f64>,
    /// Edges of each sampled network, if requested.
    pub networks: Vec<Vec<(u32, u32)>>,
    /// The network at the end of the chain.
    pub last: Network,
    /// Whether the chain stopped early because a network exceeded `max_edges`.
    pub exceeded: bool,
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

/// A proposed change: up to six toggles, applied in order, and the log ratio
/// of the reverse and forward proposal probabilities (without its reverse
/// part, for cyclic triple reversals, which is known only after the toggles).
pub struct Move {
    toggles: [(u32, u32); 6],
    len: usize,
    log_q: f64,
    /// A cyclic triple reversal: (a, b, c) for a -> b -> c -> a.
    cycle: Option<(u32, u32, u32)>,
}

impl Move {
    fn single(i: u32, j: u32, log_q: f64) -> Self {
        let mut toggles = [(0, 0); 6];
        toggles[0] = (i, j);
        Self { toggles, len: 1, log_q, cycle: None }
    }

    fn of(list: &[(u32, u32)], log_q: f64) -> Self {
        let mut toggles = [(0, 0); 6];
        toggles[..list.len()].copy_from_slice(list);
        Self { toggles, len: list.len(), log_q, cycle: None }
    }
}

/// Number of ways a cyclic triple reversal picks the cycle through the tie
/// x -> y: the third vertices w with y -> w -> x.
fn cycle_choices(net: &Network, x: u32, y: u32) -> f64 {
    count_common(net.out_neighbours(y), net.in_neighbours(x)) as f64
}

/// Sum over the cycle's ties of the probability of choosing the cycle from
/// that tie (up to the common factor 1 / ties).
fn cycle_weight(net: &Network, a: u32, b: u32, c: u32) -> f64 {
    1.0 / cycle_choices(net, a, b) + 1.0 / cycle_choices(net, b, c) + 1.0 / cycle_choices(net, c, a)
}

pub struct Proposal<'a> {
    /// Share of triadic moves; 0 gives plain TNT.
    pub triadic_weight: f64,
    /// Stop the chain if a network has more edges (a sign of degeneracy).
    pub max_edges: usize,
    pub space: &'a Space,
}

impl Proposal<'_> {
    /// Draws a move, or None if the draw found nothing to change.
    fn propose(&self, net: &Network, ties: &EdgeIndex, rng: &mut Rng) -> Option<Move> {
        match self.space.preserve {
            Preserve::Nothing => self.toggle(net, ties, rng),
            Preserve::Degrees if net.directed() && rng.unif() < CYCLE_PROB => self.reverse_cycle(net, ties, rng),
            Preserve::Degrees => self.swap(net, ties, rng),
            Preserve::OutDegrees | Preserve::InDegrees => self.move_end(net, ties, rng),
        }
    }

    /// A TNT or triadic toggle of one free dyad.
    fn toggle(&self, net: &Network, ties: &EdgeIndex, rng: &mut Rng) -> Option<Move> {
        let space = self.space;
        let w = self.triadic_weight;
        let edges = ties.count(net) as f64;
        let dyads = space.n_free() as f64;
        let (i, j) = if w > 0.0 && rng.unif() < w {
            let (i, j) = triadic_draw(net, rng)?;
            if !space.is_free(i, j) {
                return None;
            }
            (i, j)
        } else if edges > 0.0 && rng.unif() < TIE_PROB {
            ties.random(net, rng)
        } else {
            space.random_dyad(net, rng)?
        };
        let tie = net.has_edge(i, j);
        if w == 0.0 {
            let after = edges + if tie { -1.0 } else { 1.0 };
            let ratio = tnt_prob(after, dyads, !tie) / tnt_prob(edges, dyads, tie);
            return Some(Move::single(i, j, ratio.ln()));
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
        Some(Move::single(i, j, (reverse / forward).ln()))
    }

    /// Whether the dyad may become a tie: not one already, and free.
    fn open(&self, net: &Network, i: u32, j: u32) -> bool {
        i != j && !net.has_edge(i, j) && self.space.is_free(i, j)
    }

    /// Swaps the endpoints of two random ties; symmetric, so log q = 0.
    fn swap(&self, net: &Network, ties: &EdgeIndex, rng: &mut Rng) -> Option<Move> {
        let m = ties.count(net);
        if m < 2 {
            return None;
        }
        let (a, b) = ties.random(net, rng);
        let (mut c, mut d) = ties.random(net, rng);
        if (a, b) == (c, d) {
            return None;
        }
        if !net.directed() && rng.unif() < 0.5 {
            (c, d) = (d, c);
        }
        if !self.open(net, a, d) || !self.open(net, c, b) || (a, d) == (c, b) {
            return None;
        }
        Some(Move::of(&[(a, b), (c, d), (a, d), (c, b)], 0.0))
    }

    /// Reverses a random cyclic triple a -> b -> c -> a (directed networks).
    fn reverse_cycle(&self, net: &Network, ties: &EdgeIndex, rng: &mut Rng) -> Option<Move> {
        if ties.count(net) == 0 {
            return None;
        }
        let (a, b) = ties.random(net, rng);
        let (out_b, in_a) = (net.out_neighbours(b), net.in_neighbours(a));
        let choices = count_common(out_b, in_a);
        if choices == 0 {
            return None;
        }
        let mut pick = rng.below(choices as u64);
        let mut c = 0;
        for_each_common(out_b, in_a, |w| {
            if pick == 0 {
                c = w;
            }
            pick = pick.wrapping_sub(1);
        });
        let space = self.space;
        if !space.is_free(b, c) || !space.is_free(c, a) {
            return None;
        }
        if !self.open(net, b, a) || !self.open(net, c, b) || !self.open(net, a, c) {
            return None;
        }
        let forward = cycle_weight(net, a, b, c);
        let mut mv = Move::of(&[(a, b), (b, c), (c, a), (b, a), (c, b), (a, c)], -forward.ln());
        mv.cycle = Some((a, b, c));
        Some(mv)
    }

    /// Moves the head (OutDegrees) or tail (InDegrees) of a random tie to a
    /// random other vertex; symmetric, so log q = 0.
    fn move_end(&self, net: &Network, ties: &EdgeIndex, rng: &mut Rng) -> Option<Move> {
        if ties.count(net) == 0 {
            return None;
        }
        let (a, b) = ties.random(net, rng);
        let n = net.n() as u64;
        let keep = if self.space.preserve == Preserve::OutDegrees { a } else { b };
        let mut v = rng.below(n - 1) as u32;
        if v >= keep {
            v += 1;
        }
        let (i, j) = if self.space.preserve == Preserve::OutDegrees { (a, v) } else { (v, b) };
        if !self.open(net, i, j) {
            return None;
        }
        Some(Move::of(&[(a, b), (i, j)], 0.0))
    }
}

fn log_weight(theta: &[f64], delta: &[f64]) -> f64 {
    // Skipping zero changes lets coefficients be -Inf (offsets that forbid ties).
    theta.iter().zip(delta).filter(|(_, d)| **d != 0.0).map(|(t, d)| t * d).sum()
}

struct Sampler<'a> {
    model: &'a Model,
    proposal: &'a Proposal<'a>,
    theta: &'a [f64],
    state: State,
    ties: EdgeIndex,
    stats: Vec<f64>,
    delta: Vec<f64>,
    scratch: Vec<f64>,
}

impl Sampler<'_> {
    fn apply(&mut self, i: u32, j: u32) {
        let removed = self.state.net.has_edge(i, j);
        self.model.toggle(&mut self.state, i, j);
        self.ties.toggle(self.state.net.directed(), i, j, removed);
    }

    /// One Metropolis-Hastings step.
    fn step(&mut self, rng: &mut Rng) {
        let Some(mv) = self.proposal.propose(&self.state.net, &self.ties, rng) else { return };
        let space = self.proposal.space;
        if mv.len == 1 {
            let (i, j) = mv.toggles[0];
            if let Some(bounds) = &space.bounds
                && !bounds.allows(&self.state.net, i, j)
            {
                return;
            }
            self.model.change(&self.state, i, j, &mut self.delta);
            let log_accept = mv.log_q + log_weight(self.theta, &self.delta);
            if log_accept >= 0.0 || rng.unif() < log_accept.exp() {
                self.apply(i, j);
                self.stats.iter_mut().zip(&self.delta).for_each(|(s, d)| *s += d);
            }
            return;
        }
        // Several toggles: apply them in order, then accept or undo.
        self.delta.fill(0.0);
        for &(i, j) in &mv.toggles[..mv.len] {
            self.model.change(&self.state, i, j, &mut self.scratch);
            self.delta.iter_mut().zip(&self.scratch).for_each(|(d, s)| *d += s);
            self.apply(i, j);
        }
        let mut log_q = mv.log_q;
        if let Some((a, b, c)) = mv.cycle {
            // The reversed cycle a -> c -> b -> a, in the new network.
            log_q += cycle_weight(&self.state.net, a, c, b).ln();
        }
        let within = match &space.bounds {
            None => true,
            Some(bounds) => mv.toggles[..mv.len]
                .iter()
                .all(|&(i, j)| bounds.satisfied(&self.state.net, i) && bounds.satisfied(&self.state.net, j)),
        };
        let log_accept = log_q + log_weight(self.theta, &self.delta);
        if within && (log_accept >= 0.0 || rng.unif() < log_accept.exp()) {
            self.stats.iter_mut().zip(&self.delta).for_each(|(s, d)| *s += d);
        } else {
            for &(i, j) in mv.toggles[..mv.len].iter().rev() {
                self.apply(i, j);
            }
        }
    }
}

/// Runs `burnin` steps, then records `samplesize` samples `interval` steps apart.
#[allow(clippy::too_many_arguments)]
pub fn run_chain(
    model: &Model,
    proposal: &Proposal,
    net: Network,
    theta: &[f64],
    burnin: u64,
    interval: u64,
    samplesize: usize,
    rng: &mut Rng,
    keep_networks: bool,
) -> Chain {
    let stats = model.summary(&net);
    let p = stats.len();
    let ties = EdgeIndex::new(proposal.space, &net);
    let mut s = Sampler {
        model,
        proposal,
        theta,
        state: model.state(net),
        ties,
        stats,
        delta: vec![0.0; p],
        scratch: vec![0.0; p],
    };
    let mut sample = Vec::with_capacity(samplesize * p);
    let mut networks = Vec::new();
    for step in 1..=burnin + interval * samplesize as u64 {
        s.step(rng);
        if s.state.net.n_edges() > proposal.max_edges {
            return Chain { stats: sample, networks, last: s.state.net, exceeded: true };
        }
        if step > burnin && (step - burnin).is_multiple_of(interval) {
            sample.extend_from_slice(&s.stats);
            if keep_networks {
                networks.push(s.state.net.edges().to_vec());
            }
        }
    }
    Chain { stats: sample, networks, last: s.state.net, exceeded: false }
}
