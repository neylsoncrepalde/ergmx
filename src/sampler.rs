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
//! Models of transitions between networks (tergm's) also mix in, with
//! probability 1/2, toggles of a random *discordant* dyad, one that differs
//! from the previous network (tergm's discordTNT): they undo formations and
//! dissolutions, which keeps proposals of persisting ties from being rejected
//! nearly always.
//!
//! Degree-preserving constraints use moves that keep the degrees:
//!
//! * `degrees`: swap the endpoints of two ties (a -- b, c -- d become
//!   a -- d, c -- b), and, in directed networks, also reverse cyclic triples
//!   (a -> b -> c -> a becomes a -> c -> b -> a), without which some networks
//!   with the same in- and out-degrees can't reach each other (Rao, Jana and
//!   Bandyopadhyay 1996);
//! * `odegrees` (`idegrees`): move the head (tail) of a tie to another vertex;
//! * `b1degrees` (`b2degrees`): move the end of a tie in the other mode to
//!   another vertex of that mode;
//!
//! and `edges`, which preserves the number of edges, swaps a random tie for a
//! random non-tie.
//!
//! Dynamic simulation (tergm's, of a model conditional on the previous
//! network) runs one chain per time step, from the previous network, until the
//! number of dyads that differ from it stops growing (`run_series`).

use rustc_hash::FxHashMap;

use crate::network::{Network, count_common, for_each_common};
use crate::rng::Rng;
use crate::space::{EdgeIndex, Preserve, Space};
use crate::terms::{Model, State};

/// Probability that a TNT move toggles a random existing tie.
const TIE_PROB: f64 = 0.5;
/// Share of cyclic triple reversals among degree-preserving moves (directed).
const CYCLE_PROB: f64 = 0.2;
/// Share of toggles of discordant dyads, when there are some (tergm's default).
const DISCORD_PROB: f64 = 0.5;

/// The dyads where a network differs from a reference network (the previous
/// one, in models of transitions), with O(1) random draws.
pub struct Discord {
    directed: bool,
    dyads: Vec<(u32, u32)>,
    position: FxHashMap<u64, usize>,
}

impl Discord {
    pub fn new(reference: &Network, net: &Network) -> Self {
        let mut d = Self { directed: net.directed(), dyads: Vec::new(), position: FxHashMap::default() };
        for &(i, j) in net.edges().iter().filter(|&&(i, j)| !reference.has_edge(i, j)) {
            d.toggle(i, j);
        }
        for &(i, j) in reference.edges().iter().filter(|&&(i, j)| !net.has_edge(i, j)) {
            d.toggle(i, j);
        }
        d
    }

    fn key(&self, i: u32, j: u32) -> (u64, (u32, u32)) {
        let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
        (((a as u64) << 32) | b as u64, (a, b))
    }

    pub fn len(&self) -> usize {
        self.dyads.len()
    }

    fn contains(&self, i: u32, j: u32) -> bool {
        self.position.contains_key(&self.key(i, j).0)
    }

    fn random(&self, rng: &mut Rng) -> (u32, u32) {
        self.dyads[rng.below(self.dyads.len() as u64) as usize]
    }

    /// Records the toggle of (i, j): discordant dyads become concordant and
    /// the other way round.
    fn toggle(&mut self, i: u32, j: u32) {
        let (key, dyad) = self.key(i, j);
        if let Some(p) = self.position.remove(&key) {
            self.dyads.swap_remove(p);
            if p < self.dyads.len() {
                let moved = self.key(self.dyads[p].0, self.dyads[p].1).0;
                self.position.insert(moved, p);
            }
        } else {
            self.position.insert(key, self.dyads.len());
            self.dyads.push(dyad);
        }
    }
}

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
    fn propose(&self, net: &Network, ties: &EdgeIndex, discord: Option<&Discord>, rng: &mut Rng) -> Option<Move> {
        match self.space.preserve {
            Preserve::Nothing => self.toggle(net, ties, discord, rng),
            Preserve::Degrees if net.directed() && rng.unif() < CYCLE_PROB => self.reverse_cycle(net, ties, rng),
            Preserve::Degrees => self.swap(net, ties, rng),
            Preserve::OutDegrees | Preserve::InDegrees | Preserve::FirstModeDegrees | Preserve::SecondModeDegrees => {
                self.move_end(net, ties, rng)
            }
            Preserve::Edges => self.swap_tie(net, ties, rng),
        }
    }

    /// A TNT, triadic or discordant toggle of one free dyad.
    fn toggle(&self, net: &Network, ties: &EdgeIndex, discord: Option<&Discord>, rng: &mut Rng) -> Option<Move> {
        let space = self.space;
        let w = self.triadic_weight;
        let edges = ties.count(net) as f64;
        let dyads = space.n_free() as f64;
        let share = |d: &Discord| if d.len() > 0 { DISCORD_PROB } else { 0.0 };
        let (i, j) = match discord {
            Some(d) if rng.unif() < share(d) => d.random(rng),
            _ if w > 0.0 && rng.unif() < w => triadic_draw(net, rng)?,
            _ if edges > 0.0 && rng.unif() < TIE_PROB => ties.random(net, rng),
            _ => space.random_dyad(net, rng)?,
        };
        if !space.is_free(i, j) {
            return None;
        }
        let tie = net.has_edge(i, j);
        let change = if tie { -1.0 } else { 1.0 };
        // Probabilities that the TNT and triadic moves propose (i, j), now and
        // after the toggle.
        let (mut forward, mut reverse) = (tnt_prob(edges, dyads, tie), tnt_prob(edges + change, dyads, !tie));
        if w > 0.0 {
            let (ni, nj) = (net.neighbours(i), net.neighbours(j));
            let mut s = 0.0;
            for_each_common(ni, nj, |k| s += 1.0 / (net.neighbours(k).len() - 1) as f64);
            let (di, dj, n) = (ni.len() as f64, nj.len() as f64, net.n() as f64);
            let directions = if net.directed() { 2.0 } else { 1.0 };
            // i and j stay neighbours if the opposite tie j -> i exists.
            let degree_change = if net.directed() && net.has_edge(j, i) { 0.0 } else { change };
            forward = w * triadic_prob(n, directions, di, dj, s) + (1.0 - w) * forward;
            reverse = w * triadic_prob(n, directions, di + degree_change, dj + degree_change, s) + (1.0 - w) * reverse;
        }
        if let Some(d) = discord {
            // The toggle makes a discordant dyad concordant, or the reverse.
            let (inside, size) = (d.contains(i, j), d.len() as f64);
            let after = if inside { size - 1.0 } else { size + 1.0 };
            let (now, then) = (if size > 0.0 { DISCORD_PROB } else { 0.0 }, if after > 0.0 { DISCORD_PROB } else { 0.0 });
            forward = now * if inside { 1.0 / size } else { 0.0 } + (1.0 - now) * forward;
            reverse = then * if inside { 0.0 } else { 1.0 / after } + (1.0 - then) * reverse;
        }
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
        let keep = match self.space.preserve {
            Preserve::OutDegrees => a,
            Preserve::InDegrees => b,
            // The end in the mode whose degrees are kept; within-mode dyads aren't free.
            mode => {
                let first = self.space.first_mode.as_ref().expect("mode degrees need the modes");
                if first[a as usize] == (mode == Preserve::FirstModeDegrees) { a } else { b }
            }
        };
        let mut v = rng.below(n - 1) as u32;
        if v >= keep {
            v += 1;
        }
        let (i, j) = if self.space.preserve == Preserve::InDegrees { (v, b) } else { (keep, v) };
        if !self.open(net, i, j) {
            return None;
        }
        Some(Move::of(&[(a, b), (i, j)], 0.0))
    }

    /// Swaps a random tie for a random non-tie, keeping the number of edges;
    /// symmetric, so log q = 0.
    fn swap_tie(&self, net: &Network, ties: &EdgeIndex, rng: &mut Rng) -> Option<Move> {
        let m = ties.count(net) as u64;
        if m == 0 || m >= self.space.n_free() {
            return None;
        }
        let (a, b) = ties.random(net, rng);
        loop {
            let (c, d) = self.space.random_dyad(net, rng)?;
            if !net.has_edge(c, d) {
                return Some(Move::of(&[(a, b), (c, d)], 0.0));
            }
        }
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
    /// For models of transitions: the dyads that differ from the previous
    /// network (in dynamic simulation, the network the time step started from).
    discord: Option<Discord>,
}

impl<'a> Sampler<'a> {
    fn new(model: &'a Model, proposal: &'a Proposal<'a>, theta: &'a [f64], state: State, stats: Vec<f64>) -> Self {
        let p = stats.len();
        let ties = EdgeIndex::new(proposal.space, &state.net);
        Self {
            model,
            proposal,
            theta,
            state,
            ties,
            stats,
            delta: vec![0.0; p],
            scratch: vec![0.0; p],
            discord: None,
        }
    }

    fn apply(&mut self, i: u32, j: u32) {
        let removed = self.state.net.has_edge(i, j);
        self.model.toggle(&mut self.state, i, j);
        self.ties.toggle(self.state.net.directed(), i, j, removed);
        if let Some(discord) = &mut self.discord {
            discord.toggle(i, j);
        }
    }

    /// Number of dyads that differ from the reference network.
    fn hamming(&self) -> i64 {
        self.discord.as_ref().map_or(0, |d| d.len() as i64)
    }

    /// One Metropolis-Hastings step.
    fn step(&mut self, rng: &mut Rng) {
        let Some(mv) = self.proposal.propose(&self.state.net, &self.ties, self.discord.as_ref(), rng) else { return };
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
    let previous = model.previous_network(net.directed());
    let mut s = Sampler::new(model, proposal, theta, model.state(net), stats);
    s.discord = previous.map(|prev| Discord::new(&prev, &s.state.net));
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

/// When a time step of a dynamic simulation ends, as tergm's
/// MCMC.burnin.min, .max, .pval and .add.
pub struct StepRule {
    pub min: u64,
    pub max: u64,
    pub pval: f64,
    pub add: f64,
}

/// P(Z > z) for a standard normal Z.
fn normal_upper(z: f64) -> f64 {
    // erfc by Chebyshev fitting (Numerical Recipes' erfcc), relative error < 1.2e-7.
    let x = z / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let poly = -x * x - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let erfc = t * poly.exp();
    0.5 * if x >= 0.0 { erfc } else { 2.0 - erfc }
}

/// Runs the chain of one time step, as tergm does: the per-step increments in
/// the number of dyads that differ from the start (the Hamming distance) are
/// averaged with exponentially decaying weights (decay 1 - 1/min); from `min`
/// steps on, once a z-test no longer finds them positive (p-value above
/// `pval`), the chain runs `add` times as many steps again, and stops.
/// Returns the number of steps, at most `max`.
fn run_until_stable(s: &mut Sampler, rule: &StepRule, rng: &mut Rng) -> u64 {
    let decay = 1.0 - 1.0 / rule.min.max(1) as f64;
    let (mut si, mut si2, mut sw, mut sw2) = (0.0, 0.0, 0.0, 0.0);
    let mut stop_at: Option<u64> = None;
    let mut step = 0;
    while step < rule.max && stop_at.is_none_or(|end| step < end) {
        let before = s.hamming();
        s.step(rng);
        step += 1;
        if s.state.net.n_edges() > s.proposal.max_edges {
            break;
        }
        let i = (s.hamming() - before) as f64;
        sw = sw * decay + 1.0;
        si = si * decay + i;
        sw2 = sw2 * decay * decay + 1.0;
        si2 = si2 * decay + i * i;
        if step >= rule.min && stop_at.is_none() {
            let (mi, mi2) = (si / sw, si2 / sw);
            let variance = mi2 - mi * mi;
            // No change at all: nothing left to wait for.
            let p = if variance <= 0.0 { 1.0 } else { normal_upper(mi / (variance * sw2 / (sw * sw)).sqrt()) };
            if p > rule.pval {
                let extra = (step as f64 * rule.add + rng.unif()).round() as u64;
                stop_at = Some(step + extra);
            }
        }
    }
    step
}

/// A dynamic simulation: the network after each time step, the model's
/// statistics then (each step's model is conditional on the previous network),
/// and the MCMC steps each took.
pub struct Series {
    pub networks: Vec<Vec<(u32, u32)>>,
    pub stats: Vec<f64>,
    pub steps: Vec<u64>,
    pub exceeded: bool,
}

/// Runs `slices` time steps of a model with tergm's operators from `start`:
/// at each, the blocks' previous networks are their networks at the start.
#[allow(clippy::too_many_arguments)]
pub fn run_series(
    model: &Model,
    proposal: &Proposal,
    start: Network,
    theta: &[f64],
    slices: usize,
    rule: &StepRule,
    rng: &mut Rng,
) -> Series {
    let layout = model.layout().expect("dynamic simulation needs a block layout");
    let mut out = Series { networks: Vec::new(), stats: Vec::new(), steps: Vec::new(), exceeded: false };
    let mut net = start;
    for _ in 0..slices {
        let prev = layout.split(&net);
        let stats = model.summary_with(&net, &prev);
        let mut s = Sampler::new(model, proposal, theta, model.state_with(net.clone(), prev), stats);
        s.discord = Some(Discord::new(&net, &net));
        let steps = run_until_stable(&mut s, rule, rng);
        out.exceeded = s.state.net.n_edges() > proposal.max_edges;
        out.networks.push(s.state.net.edges().to_vec());
        out.stats.extend_from_slice(&s.stats);
        out.steps.push(steps);
        net = s.state.net;
        if out.exceeded {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::normal_upper;

    #[test]
    fn normal_tail() {
        for (z, p) in [(0.0, 0.5), (1.0, 0.15865525393145707), (-2.0, 0.9772498680518208), (3.0, 0.0013498980316301)] {
            assert!((normal_upper(z) - p).abs() < 2e-7 * p.max(1e-3), "{z}");
        }
    }
}
