//! Valued networks, as R's ergm with ergm.count: each dyad has a value,
//! drawn from a reference measure (Poisson, geometric, binomial, discrete
//! uniform; continuous uniform, standard normal) tilted by the model's
//! statistics.
//!
//! A valued term reports how its statistics change when a dyad's value goes
//! from `old` to `new`. Dyad-independent binary terms become valued ones
//! through ergm's `form`: `"sum"` weighs each dyad's change statistic by its
//! value, `"nonzero"` by whether it is nonzero.

use rustc_hash::FxHashMap;

use crate::network::Network;
use crate::rng::Rng;
use crate::sampler::SanSettings;
use crate::space::{EdgeIndex, Space};
use crate::terms::{Term, TermSpec};

/// A valued network: the nonzero values, with each vertex's neighbours
/// through nonzero dyads, and the totals some terms need.
#[derive(Clone)]
pub struct WtNetwork {
    n: u32,
    directed: bool,
    values: FxHashMap<u64, f64>,
    out: Vec<Vec<u32>>,
    inn: Vec<Vec<u32>>,
    nonzero: Vec<(u32, u32)>,
    position: FxHashMap<u64, usize>,
    /// Sums over the nonzero dyads of the values, and of their square roots.
    pub total: f64,
    pub total_sqrt: f64,
}

fn insert_sorted(list: &mut Vec<u32>, v: u32) {
    if let Err(p) = list.binary_search(&v) {
        list.insert(p, v);
    }
}

fn remove_sorted(list: &mut Vec<u32>, v: u32) {
    if let Ok(p) = list.binary_search(&v) {
        list.remove(p);
    }
}

impl WtNetwork {
    pub fn new(n: u32, directed: bool) -> Self {
        Self {
            n,
            directed,
            values: FxHashMap::default(),
            out: vec![Vec::new(); n as usize],
            inn: vec![Vec::new(); if directed { n as usize } else { 0 }],
            nonzero: Vec::new(),
            position: FxHashMap::default(),
            total: 0.0,
            total_sqrt: 0.0,
        }
    }

    pub fn from_values(n: u32, directed: bool, values: &[(u32, u32, f64)]) -> Result<Self, String> {
        let mut net = Self::new(n, directed);
        for &(i, j, v) in values {
            if i >= n || j >= n || i == j {
                return Err(format!("bad dyad ({i}, {j}) for {n} vertices"));
            }
            if !v.is_finite() {
                return Err(format!("dyad values must be finite, not {v}"));
            }
            net.set(i, j, v);
        }
        Ok(net)
    }

    pub fn n(&self) -> u32 {
        self.n
    }

    pub fn directed(&self) -> bool {
        self.directed
    }

    fn key(&self, i: u32, j: u32) -> u64 {
        let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
        ((a as u64) << 32) | b as u64
    }

    /// The value of the dyad i -> j (or i -- j).
    #[inline]
    pub fn get(&self, i: u32, j: u32) -> f64 {
        self.values.get(&self.key(i, j)).copied().unwrap_or(0.0)
    }

    /// Vertices k with a nonzero value i -> k (all of i's, if undirected).
    pub fn out_neighbours(&self, i: u32) -> &[u32] {
        &self.out[i as usize]
    }

    /// Vertices k with a nonzero value k -> i (all of i's, if undirected).
    pub fn in_neighbours(&self, i: u32) -> &[u32] {
        if self.directed { &self.inn[i as usize] } else { &self.out[i as usize] }
    }

    pub fn n_nonzero(&self) -> usize {
        self.nonzero.len()
    }

    /// A nonzero dyad, uniformly.
    pub fn random_nonzero(&self, rng: &mut Rng) -> (u32, u32) {
        self.nonzero[rng.below(self.nonzero.len() as u64) as usize]
    }

    /// The nonzero dyads (i < j if undirected).
    pub fn nonzero(&self) -> &[(u32, u32)] {
        &self.nonzero
    }

    /// The nonzero dyads (i < j if undirected) with their values.
    pub fn triples(&self) -> Vec<(u32, u32, f64)> {
        self.nonzero.iter().map(|&(i, j)| (i, j, self.get(i, j))).collect()
    }

    pub fn set(&mut self, i: u32, j: u32, v: f64) {
        let key = self.key(i, j);
        let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
        let old = self.values.get(&key).copied().unwrap_or(0.0);
        self.total += v - old;
        self.total_sqrt += v.sqrt() - old.sqrt();
        if v == 0.0 {
            if self.values.remove(&key).is_some() {
                if self.directed {
                    remove_sorted(&mut self.out[a as usize], b);
                    remove_sorted(&mut self.inn[b as usize], a);
                } else {
                    remove_sorted(&mut self.out[a as usize], b);
                    remove_sorted(&mut self.out[b as usize], a);
                }
                let p = self.position.remove(&key).unwrap();
                self.nonzero.swap_remove(p);
                if p < self.nonzero.len() {
                    let (c, d) = self.nonzero[p];
                    let moved = self.key(c, d);
                    self.position.insert(moved, p);
                }
            }
        } else if self.values.insert(key, v).is_none() {
            if self.directed {
                insert_sorted(&mut self.out[a as usize], b);
                insert_sorted(&mut self.inn[b as usize], a);
            } else {
                insert_sorted(&mut self.out[a as usize], b);
                insert_sorted(&mut self.out[b as usize], a);
            }
            self.position.insert(key, self.nonzero.len());
            self.nonzero.push((a, b));
        }
    }
}

/// The natural log of the gamma function (Lanczos' approximation, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (k, c) in C.iter().enumerate().skip(1) {
        a += c / (x + k as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// log(y!), exactly for small integers.
pub fn ln_factorial(y: f64) -> f64 {
    if y < 0.0 {
        return f64::NEG_INFINITY;
    }
    if y == y.floor() && y <= 50.0 {
        return (2..=y as u64).map(|k| (k as f64).ln()).sum();
    }
    ln_gamma(y + 1.0)
}

fn ln_choose(n: f64, k: f64) -> f64 {
    if k < 0.0 || k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

// -- Terms --------------------------------------------------------------------------------------

pub trait WtTerm: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    /// Adds to `out` how the statistics change when the dyad i -> j goes from `old` to `new`.
    fn change(&self, net: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]);

    /// Adds the statistics of the network with every value 0.
    fn empty(&self, _out: &mut [f64]) {}
}

/// A dyad-independent binary term, weighted by the dyad's value (`sum`) or by
/// whether it is nonzero.
struct Dyadic {
    inner: Box<dyn Term>,
    nonzero: bool,
    blank: Network,
}

impl WtTerm for Dyadic {
    fn n_stats(&self) -> usize {
        self.inner.n_stats()
    }

    fn change(&self, _: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]) {
        let factor = if self.nonzero { (new != 0.0) as u8 as f64 - (old != 0.0) as u8 as f64 } else { new - old };
        if factor == 0.0 {
            return;
        }
        // The slice is this term's only, zero before.
        self.inner.change(&self.blank, i, j, 1.0, out);
        out.iter_mut().for_each(|x| *x *= factor);
    }
}

/// The sum of the values to a power.
struct SumPow(f64);

impl WtTerm for SumPow {
    fn change(&self, _: &WtNetwork, _: u32, _: u32, old: f64, new: f64, out: &mut [f64]) {
        out[0] += if self.0 == 1.0 { new - old } else { new.powf(self.0) - old.powf(self.0) };
    }
}

/// Dyads whose value meets each threshold (atleast, atmost, greaterthan,
/// smallerthan), or is in an interval.
struct Threshold {
    test: Test,
    dyads: f64,
}

enum Test {
    AtLeast(Vec<f64>),
    AtMost(Vec<f64>),
    Greater(Vec<f64>),
    Smaller(Vec<f64>),
    Interval { lower: f64, upper: f64, open_lower: bool, open_upper: bool },
}

impl Test {
    fn holds(&self, k: usize, y: f64) -> bool {
        match self {
            Test::AtLeast(t) => y >= t[k],
            Test::AtMost(t) => y <= t[k],
            Test::Greater(t) => y > t[k],
            Test::Smaller(t) => y < t[k],
            Test::Interval { lower, upper, open_lower, open_upper } => {
                (if *open_lower { y > *lower } else { y >= *lower }) && (if *open_upper { y < *upper } else { y <= *upper })
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Test::AtLeast(t) | Test::AtMost(t) | Test::Greater(t) | Test::Smaller(t) => t.len(),
            Test::Interval { .. } => 1,
        }
    }
}

impl WtTerm for Threshold {
    fn n_stats(&self) -> usize {
        self.test.len()
    }

    fn change(&self, _: &WtNetwork, _: u32, _: u32, old: f64, new: f64, out: &mut [f64]) {
        for (k, stat) in out.iter_mut().enumerate() {
            *stat += self.test.holds(k, new) as u8 as f64 - self.test.holds(k, old) as u8 as f64;
        }
    }

    fn empty(&self, out: &mut [f64]) {
        for (k, stat) in out.iter_mut().enumerate() {
            if self.test.holds(k, 0.0) {
                *stat += self.dyads;
            }
        }
    }
}

/// Reciprocity of values (directed): the minimum of y_ij and y_ji, minus
/// their absolute difference, their product or their geometric mean.
enum MutualForm {
    Min,
    NAbsDiff,
    Product,
    GeoMean,
}

struct WtMutual(MutualForm);

impl WtTerm for WtMutual {
    fn change(&self, net: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]) {
        let back = net.get(j, i);
        out[0] += match self.0 {
            MutualForm::Min => new.min(back) - old.min(back),
            MutualForm::NAbsDiff => -((new - back).abs() - (old - back).abs()),
            MutualForm::Product => (new - old) * back,
            MutualForm::GeoMean => (new * back).sqrt() - (old * back).sqrt(),
        };
    }
}

/// Transitive (or cyclical) weights: for each nonzero dyad t -> h, the
/// combination (max or sum) over third vertices k of the two-paths t -> k
/// -> h (cyclical: h -> k -> t), each the minimum or geometric mean of its
/// two values, compared (min or geometric mean) with y_th. Undirected
/// networks: every pair once, every third vertex.
struct Weights {
    geomean_path: bool,
    sum: bool,
    geomean_affect: bool,
    cyclical: bool,
}

impl Weights {
    fn path(&self, a: f64, b: f64) -> f64 {
        if self.geomean_path { (a * b).sqrt() } else { a.min(b) }
    }

    fn affect(&self, paths: f64, y: f64) -> f64 {
        if self.geomean_affect { (paths * y).sqrt() } else { paths.min(y) }
    }

}

impl WtTerm for Weights {
    fn change(&self, net: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]) {
        // The ties whose contribution may change: i -> j itself, and those
        // with i -> j as a link of a two-path. Every tie touching i or j.
        let directed = net.directed();
        let canonical = |a: u32, b: u32| if !directed && a > b { (b, a) } else { (a, b) };
        let mut ties: Vec<(u32, u32)> = vec![canonical(i, j)];
        for &v in &[i, j] {
            for &k in net.out_neighbours(v) {
                ties.push(canonical(v, k));
            }
            if directed {
                for &k in net.in_neighbours(v) {
                    ties.push((k, v));
                }
            }
        }
        ties.sort_unstable();
        ties.dedup();
        let (a, b) = canonical(i, j);
        let before = |x: u32, y: u32| if canonical(x, y) == (a, b) { old } else { net.get(x, y) };
        let after = |x: u32, y: u32| if canonical(x, y) == (a, b) { new } else { net.get(x, y) };
        let mut change = 0.0;
        for &(t, h) in &ties {
            change += self.contribution_with(net, t, h, &after, (a, b, new)) - self.contribution_with(net, t, h, &before, (a, b, old));
        }
        out[0] += change;
    }
}

impl Weights {
    /// The contribution of the tie t -> h: its value compared with its
    /// two-paths' (the values read through `value`), with the changing dyad
    /// (a, b), of value v, among the second links considered.
    fn contribution_with(&self, net: &WtNetwork, t: u32, h: u32, value: &dyn Fn(u32, u32) -> f64, dyad: (u32, u32, f64)) -> f64 {
        let y = value(t, h);
        if y == 0.0 {
            return 0.0;
        }
        let directed = net.directed();
        let mut ks: Vec<u32> = if self.cyclical { net.out_neighbours(h).to_vec() } else { net.in_neighbours(h).to_vec() };
        // The changing dyad as a second link: k -> h (transitive) or h -> k
        // (cyclical), which `net`'s neighbours miss if it is 0 there.
        let (a, b, v) = dyad;
        if v != 0.0 {
            if !directed {
                if a == h {
                    ks.push(b);
                } else if b == h {
                    ks.push(a);
                }
            } else if self.cyclical {
                if a == h {
                    ks.push(b);
                }
            } else if b == h {
                ks.push(a);
            }
        }
        ks.sort_unstable();
        ks.dedup();
        let mut paths = 0.0;
        for k in ks {
            if k == t || k == h {
                continue;
            }
            let p = if self.cyclical { self.path(value(h, k), value(k, t)) } else { self.path(value(t, k), value(k, h)) };
            paths = if self.sum { paths + p } else { paths.max(p) };
        }
        self.affect(paths, y)
    }
}

/// Covariance of the values of the dyads of each vertex (ergm's nodecovar,
/// nodeocovar, nodeicovar), of the values or their square roots, centered
/// or not.
struct NodeCovar {
    sqrt: bool,
    center: bool,
    side: Side,
}

#[derive(Clone, Copy)]
enum Side {
    All,
    Out,
    In,
}

impl WtTerm for NodeCovar {
    fn change(&self, net: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]) {
        let t = |y: f64| if self.sqrt { y.sqrt() } else { y };
        let n = net.n() as f64;
        let diff = (t(new) - t(old)) / (n - 2.0);
        let mut change = 0.0;
        match self.side {
            Side::All => {
                for &k in net.out_neighbours(i) {
                    if k != j {
                        change += diff * t(net.get(i, k));
                    }
                }
                for &k in net.out_neighbours(j) {
                    if k != i {
                        change += diff * t(net.get(j, k));
                    }
                }
            }
            Side::Out => {
                for &k in net.out_neighbours(i) {
                    if k != j {
                        change += 2.0 * diff * t(net.get(i, k));
                    }
                }
            }
            Side::In => {
                for &k in net.in_neighbours(j) {
                    if k != i {
                        change += 2.0 * diff * t(net.get(k, j));
                    }
                }
            }
        }
        if self.center {
            let sum = if self.sqrt { net.total_sqrt } else { net.total };
            let new_sum = sum + t(new) - t(old);
            let dyads = n * (n - 1.0) / if net.directed() { 1.0 } else { 2.0 };
            change += (sum * sum - new_sum * new_sum) / dyads;
        }
        out[0] += change;
    }
}

/// Conway-Maxwell-Poisson dispersion: the sum over dyads of log(y!).
struct Cmp;

impl WtTerm for Cmp {
    fn change(&self, _: &WtNetwork, _: u32, _: u32, old: f64, new: f64, out: &mut [f64]) {
        out[0] += ln_factorial(new) - ln_factorial(old);
    }
}

/// Valued transitive (cyclical) ties: the nonzero dyads above `threshold`
/// with a two-path above it, recomputed around the changing dyad.
struct TiesAbove {
    threshold: f64,
    cyclical: bool,
}

impl TiesAbove {
    fn supported(&self, net: &WtNetwork, t: u32, h: u32, value: &dyn Fn(u32, u32) -> f64) -> bool {
        if value(t, h) <= self.threshold {
            return false;
        }
        let n = net.n();
        (0..n).any(|k| {
            k != t && k != h && if self.cyclical {
                value(h, k) > self.threshold && value(k, t) > self.threshold
            } else {
                value(t, k) > self.threshold && value(k, h) > self.threshold
            }
        })
    }
}

impl WtTerm for TiesAbove {
    fn change(&self, net: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]) {
        let directed = net.directed();
        let canonical = |a: u32, b: u32| if !directed && a > b { (b, a) } else { (a, b) };
        let (a, b) = canonical(i, j);
        let before = |x: u32, y: u32| if canonical(x, y) == (a, b) { old } else { net.get(x, y) };
        let after = |x: u32, y: u32| if canonical(x, y) == (a, b) { new } else { net.get(x, y) };
        let mut ties: Vec<(u32, u32)> = vec![(a, b)];
        for &v in &[i, j] {
            for &k in net.out_neighbours(v) {
                ties.push(canonical(v, k));
            }
            if directed {
                for &k in net.in_neighbours(v) {
                    ties.push((k, v));
                }
            }
        }
        ties.sort_unstable();
        ties.dedup();
        let mut change = 0.0;
        for &(t, h) in &ties {
            change += self.supported(net, t, h, &after) as u8 as f64 - self.supported(net, t, h, &before) as u8 as f64;
        }
        out[0] += change;
    }
}

// -- Model --------------------------------------------------------------------------------------

pub struct WtModel {
    entries: Vec<WtEntry>,
    offsets: Vec<usize>,
    n_stats: usize,
    n_slots: usize,
}

/// A valued model's term, or ergm.multi's N() of valued terms on each
/// network of a combined network.
enum WtEntry {
    Term(Box<dyn WtTerm>),
    Blocks(WtBlocks),
}

/// N() of valued terms: each block's statistics (`models[k]`, on the block's
/// own network, in the state's `locals[slot][k]`) times each of the q
/// columns of the linear model's row of the block, summed over the blocks.
struct WtBlocks {
    slot: usize,
    starts: Vec<u32>,
    block_of: Vec<usize>,
    models: Vec<WtModel>,
    design: Vec<Vec<f64>>,
    p: usize,
    q: usize,
}

impl WtBlocks {
    fn locate(&self, i: u32, j: u32) -> (usize, u32, u32) {
        let k = self.block_of[i as usize];
        (k, i - self.starts[k], j - self.starts[k])
    }
}

/// A chain's state of a valued model: the networks of the blocks of its N()
/// operators (none without them).
#[derive(Clone, Default)]
pub struct WtState {
    locals: Vec<Vec<WtNetwork>>,
}

impl WtModel {
    pub fn new(n: u32, directed: bool, specs: &[TermSpec], dyads: f64) -> Result<Self, String> {
        let mut entries = Vec::new();
        let mut n_slots = 0;
        for spec in specs {
            if spec.0 == "wtblocks" {
                entries.push(WtEntry::Blocks(blocks(n, directed, spec, n_slots)?));
                n_slots += 1;
            } else {
                entries.push(WtEntry::Term(build(n, directed, spec, dyads)?));
            }
        }
        let mut offsets = Vec::with_capacity(entries.len());
        let mut total = 0;
        for e in &entries {
            offsets.push(total);
            total += match e {
                WtEntry::Term(t) => t.n_stats(),
                WtEntry::Blocks(b) => b.p * b.q,
            };
        }
        Ok(Self { entries, offsets, n_stats: total, n_slots })
    }

    pub fn n_stats(&self) -> usize {
        self.n_stats
    }

    /// Whether the model has N() operators, whose blocks need a state.
    pub fn has_blocks(&self) -> bool {
        self.n_slots > 0
    }

    /// The state of a network: its blocks' networks, for the N() operators.
    pub fn state(&self, net: &WtNetwork) -> WtState {
        let mut locals = vec![Vec::new(); self.n_slots];
        for e in &self.entries {
            if let WtEntry::Blocks(b) = e {
                let mut nets: Vec<WtNetwork> =
                    (0..b.models.len()).map(|k| WtNetwork::new(b.size(k, net.n()), net.directed())).collect();
                for (i, j, v) in net.triples() {
                    let (k, a, c) = b.locate(i, j);
                    nets[k].set(a, c, v);
                }
                locals[b.slot] = nets;
            }
        }
        WtState { locals }
    }

    /// The change of every statistic when the dyad goes from `old` to `new` (`out` is overwritten).
    #[allow(clippy::too_many_arguments)]
    pub fn change(&self, state: &WtState, net: &WtNetwork, i: u32, j: u32, old: f64, new: f64, out: &mut [f64]) {
        out.fill(0.0);
        for (e, &offset) in self.entries.iter().zip(&self.offsets) {
            match e {
                WtEntry::Term(term) => term.change(net, i, j, old, new, &mut out[offset..offset + term.n_stats()]),
                WtEntry::Blocks(b) => {
                    let (k, a, c) = b.locate(i, j);
                    let mut inner = vec![0.0; b.p];
                    b.models[k].change(&WtState::default(), &state.locals[b.slot][k], a, c, old, new, &mut inner);
                    for (s, &g) in inner.iter().enumerate() {
                        for (r, &x) in b.design[k].iter().enumerate() {
                            out[offset + s * b.q + r] += g * x;
                        }
                    }
                }
            }
        }
    }

    /// Records in `state` that the dyad i -> j is now `value`.
    pub fn apply(&self, state: &mut WtState, i: u32, j: u32, value: f64) {
        for e in &self.entries {
            if let WtEntry::Blocks(b) = e {
                let (k, a, c) = b.locate(i, j);
                state.locals[b.slot][k].set(a, c, value);
            }
        }
    }

    pub fn summary(&self, net: &WtNetwork) -> Vec<f64> {
        let mut stats = vec![0.0; self.n_stats];
        for (e, &offset) in self.entries.iter().zip(&self.offsets) {
            match e {
                WtEntry::Term(term) => term.empty(&mut stats[offset..offset + term.n_stats()]),
                WtEntry::Blocks(b) => {
                    for (k, model) in b.models.iter().enumerate() {
                        let inner = model.summary(&WtNetwork::new(b.size(k, net.n()), net.directed()));
                        for (s, &g) in inner.iter().enumerate() {
                            for (r, &x) in b.design[k].iter().enumerate() {
                                stats[offset + s * b.q + r] += g * x;
                            }
                        }
                    }
                }
            }
        }
        let mut build = WtNetwork::new(net.n(), net.directed());
        let mut state = self.state(&build);
        let mut delta = vec![0.0; self.n_stats];
        for (i, j, v) in net.triples() {
            self.change(&state, &build, i, j, 0.0, v, &mut delta);
            stats.iter_mut().zip(&delta).for_each(|(s, d)| *s += d);
            build.set(i, j, v);
            self.apply(&mut state, i, j, v);
        }
        stats
    }
}

impl WtBlocks {
    fn size(&self, k: usize, n: u32) -> u32 {
        self.starts.get(k + 1).copied().unwrap_or(n) - self.starts[k]
    }
}

/// N() of valued terms, from its spec: ints [q, number of blocks, each
/// block's first vertex], then one child per block, ("block", its row of
/// the linear model, [], its valued terms' specs).
fn blocks(n: u32, directed: bool, spec: &TermSpec, slot: usize) -> Result<WtBlocks, String> {
    let TermSpec(_, _, ints, children) = spec;
    let (q, count) = match ints.as_slice() {
        [q, count, ..] => (*q as usize, *count as usize),
        _ => return Err("N(): bad parameters".into()),
    };
    let starts: Vec<u32> = ints.get(2..2 + count).ok_or("N(): one start per network")?.iter().map(|&s| s as u32).collect();
    if children.len() != count {
        return Err("N(): one child per network".into());
    }
    let mut block_of = vec![0; n as usize];
    let mut models = Vec::with_capacity(count);
    let mut design = Vec::with_capacity(count);
    for (k, child) in children.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(n);
        block_of[starts[k] as usize..end as usize].iter_mut().for_each(|b| *b = k);
        let size = end - starts[k];
        let dyads = size as f64 * (size as f64 - 1.0) / if directed { 1.0 } else { 2.0 };
        let model = WtModel::new(size, directed, &child.3, dyads)?;
        if model.n_slots > 0 {
            return Err("N() inside N() is not supported".into());
        }
        if child.1.len() != q {
            return Err("N(): one row of the linear model per network".into());
        }
        models.push(model);
        design.push(child.1.clone());
    }
    let p = models.first().map_or(0, |m| m.n_stats());
    if models.iter().any(|m| m.n_stats() != p) {
        return Err("N(): the networks' terms have different numbers of statistics".into());
    }
    Ok(WtBlocks { slot, starts, block_of, models, design, p, q })
}

fn build(n: u32, directed: bool, spec: &TermSpec, dyads: f64) -> Result<Box<dyn WtTerm>, String> {
    let TermSpec(name, reals, ints, children) = spec;
    let real = |k: usize| reals.get(k).copied().ok_or_else(|| format!("{name}: missing parameter"));
    let flag = |k: usize| ints.get(k).copied().unwrap_or(0) != 0;
    let only_directed = || if directed { Ok(()) } else { Err(format!("{name} is only for directed networks")) };
    Ok(match name.as_str() {
        // ints: [nonzero]; children: the binary term
        "dyadic" => {
            let child = children.first().ok_or("dyadic: needs a term")?;
            let inner = crate::terms::build_term(n as usize, directed, child)?;
            Box::new(Dyadic { inner, nonzero: flag(0), blank: Network::new(n, directed) })
        }
        "sum" => Box::new(SumPow(reals.first().copied().unwrap_or(1.0))),
        "atleast" => Box::new(Threshold { test: Test::AtLeast(reals.clone()), dyads }),
        "atmost" => Box::new(Threshold { test: Test::AtMost(reals.clone()), dyads }),
        "greaterthan" => Box::new(Threshold { test: Test::Greater(reals.clone()), dyads }),
        "smallerthan" => Box::new(Threshold { test: Test::Smaller(reals.clone()), dyads }),
        // reals: [lower, upper]; ints: [open lower, open upper]
        "ininterval" => Box::new(Threshold {
            test: Test::Interval { lower: real(0)?, upper: real(1)?, open_lower: flag(0), open_upper: flag(1) },
            dyads,
        }),
        // ints: [form: 0 min, 1 nabsdiff, 2 product, 3 geomean]
        "mutual" => {
            only_directed()?;
            let form = match ints.first().copied().unwrap_or(0) {
                0 => MutualForm::Min,
                1 => MutualForm::NAbsDiff,
                2 => MutualForm::Product,
                _ => MutualForm::GeoMean,
            };
            Box::new(WtMutual(form))
        }
        // ints: [geomean path, sum combine, geomean affect]
        "transitiveweights" | "cyclicalweights" => Box::new(Weights {
            geomean_path: flag(0),
            sum: flag(1),
            geomean_affect: flag(2),
            cyclical: name == "cyclicalweights",
        }),
        // ints: [sqrt, center]
        "nodecovar" | "nodeocovar" | "nodeicovar" => {
            let side = match name.as_str() {
                "nodeocovar" => Side::Out,
                "nodeicovar" => Side::In,
                _ => Side::All,
            };
            if matches!(side, Side::All) == directed {
                return Err(format!("{name} is for {} networks", if directed { "undirected" } else { "directed" }));
            }
            Box::new(NodeCovar { sqrt: flag(0), center: flag(1), side })
        }
        "CMP" => Box::new(Cmp),
        // reals: [threshold]
        "transitiveties" | "cyclicalties" => Box::new(TiesAbove { threshold: real(0)?, cyclical: name == "cyclicalties" }),
        other => return Err(format!("unknown valued term: {other}")),
    })
}

// -- The reference measures and the sampler -----------------------------------------------------

#[derive(Clone, Copy)]
pub enum Reference {
    Poisson,
    Geometric,
    Binomial(f64),
    DiscUnif(f64, f64),
    /// Continuous uniform on [a, b], proposing values uniformly (ergm's Unif).
    Unif(f64, f64),
    /// Standard normal, proposing a dyad's value plus a normal step with this
    /// standard deviation (ergm's StdNormal, whose default is 0.2).
    StdNormal(f64),
}

const FUDGE: f64 = 0.5;

fn rpois(mu: f64, rng: &mut Rng) -> f64 {
    if mu < 30.0 {
        // Inversion.
        let mut k = 0.0;
        let mut p = (-mu).exp();
        let mut cdf = p;
        let u = rng.unif();
        while u > cdf {
            k += 1.0;
            p *= mu / k;
            cdf += p;
            if p < 1e-300 && cdf >= 1.0 - 1e-15 {
                break;
            }
        }
        return k;
    }
    // Hörmann's PTRS (transformed rejection with squeeze).
    let (b, a) = (0.931 + 2.53 * mu.sqrt(), -0.059 + 0.02483 * (0.931 + 2.53 * mu.sqrt()));
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.unif() - 0.5;
        let v = rng.unif();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + mu + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if (v * inv_alpha / (a / (us * us) + b)).ln() <= -mu + k * mu.ln() - ln_factorial(k) {
            return k;
        }
    }
}

fn ldpois(y: f64, mu: f64) -> f64 {
    y * mu.ln() - mu - ln_factorial(y)
}

fn rgeom(p: f64, rng: &mut Rng) -> f64 {
    // Failures before the first success.
    let u = 1.0 - rng.unif();
    (u.ln() / (1.0 - p).ln()).floor()
}

fn ldgeom(y: f64, p: f64) -> f64 {
    p.ln() + y * (1.0 - p).ln()
}

fn rbinom(n: f64, p: f64, rng: &mut Rng) -> f64 {
    (0..n as u64).filter(|_| rng.unif() < p).count() as f64
}

fn ldbinom(y: f64, n: f64, p: f64) -> f64 {
    ln_choose(n, y) + y * p.ln() + (n - y) * (1.0 - p).ln()
}

impl Reference {
    /// A new value, different from `from`, as ergm.count's jumps.
    fn jump(self, from: f64, rng: &mut Rng) -> f64 {
        loop {
            let to = match self {
                Reference::Poisson => rpois(from + FUDGE, rng),
                Reference::Geometric => rgeom(1.0 / (from + 1.0 + FUDGE), rng),
                Reference::Binomial(n) => rbinom(n, (from + FUDGE) / (n + 2.0 * FUDGE), rng),
                Reference::DiscUnif(a, b) => (a + rng.unif() * (b - a + 1.0)).floor().min(b),
                Reference::Unif(a, b) => a + rng.unif() * (b - a),
                Reference::StdNormal(sd) => from + sd * rng.normal(),
            };
            if to != from {
                return to;
            }
        }
    }

    /// log q(to | from) of `jump`.
    fn log_jump(self, from: f64, to: f64) -> f64 {
        match self {
            Reference::Poisson => {
                let mu = from + FUDGE;
                ldpois(to, mu) - (-ldpois(from, mu).exp()).ln_1p()
            }
            Reference::Geometric => {
                let p = 1.0 / (from + 1.0 + FUDGE);
                ldgeom(to, p) - (-ldgeom(from, p).exp()).ln_1p()
            }
            Reference::Binomial(n) => {
                let p = (from + FUDGE) / (n + 2.0 * FUDGE);
                ldbinom(to, n, p) - (-ldbinom(from, n, p).exp()).ln_1p()
            }
            Reference::DiscUnif(a, b) => -(b - a).ln(),
            // Symmetric proposals.
            Reference::Unif(..) | Reference::StdNormal(_) => 0.0,
        }
    }

    /// Whether the values are continuous (proposed without DiscTNT's jumps to 0).
    pub fn continuous(self) -> bool {
        matches!(self, Reference::Unif(..) | Reference::StdNormal(_))
    }

    /// log h(y), the reference measure.
    pub fn log_h(self, y: f64) -> f64 {
        match self {
            Reference::Poisson => -ln_factorial(y),
            Reference::Geometric | Reference::DiscUnif(..) | Reference::Unif(..) => 0.0,
            Reference::Binomial(n) => ln_choose(n, y),
            Reference::StdNormal(_) => -0.5 * y * y,
        }
    }
}

pub struct WtChain {
    pub stats: Vec<f64>,
    pub networks: Vec<Vec<(u32, u32, f64)>>,
    pub last: WtNetwork,
    pub exceeded: bool,
}

/// The proposal: ergm.count's DiscTNT (with `p0`, the probability of
/// proposing a nonzero dyad's jump to 0) or Disc (without), over the free
/// dyads of `space`; continuous references take no `p0`.
pub struct WtProposal<'a> {
    pub reference: Reference,
    pub p0: Option<f64>,
    pub space: &'a Space,
    pub max_nonzero: usize,
}

/// A proposal's change of one dyad: (i, j) from `from` to `to`, and the
/// log-ratio of the proposal's probabilities and the reference measure's.
struct WtMove {
    i: u32,
    j: u32,
    from: f64,
    to: f64,
    log_ratio: f64,
}

/// The proposal's state in a chain: the free dyads' count and the nonzero
/// free dyads (ergm's E), indexed if some dyads are fixed.
struct Proposer<'a> {
    proposal: &'a WtProposal<'a>,
    p0: Option<f64>,
    dyads: f64,
    blank: Network,
    free_nonzero: EdgeIndex,
}

impl<'a> Proposer<'a> {
    fn new(proposal: &'a WtProposal<'a>, net: &WtNetwork) -> Self {
        let p0 = if proposal.reference.continuous() { None } else { proposal.p0 };
        Self {
            proposal,
            p0,
            dyads: proposal.space.n_free() as f64,
            blank: Network::new(net.n(), net.directed()),
            free_nonzero: EdgeIndex::of_pairs(proposal.space, net.nonzero()),
        }
    }

    /// A move, or None if the proposal can't be made or changes nothing.
    fn propose(&self, net: &WtNetwork, rng: &mut Rng) -> Option<WtMove> {
        let reference = self.proposal.reference;
        let e = self.free_nonzero.free_count().unwrap_or_else(|| net.n_nonzero());
        let (i, j, to) = match self.p0 {
            Some(p0) if e > 0 && rng.unif() < p0 => {
                let (i, j) = self.free_nonzero.random_free(rng).unwrap_or_else(|| net.random_nonzero(rng));
                (i, j, 0.0)
            }
            _ => {
                let (i, j) = self.proposal.space.random_dyad(&self.blank, rng)?;
                (i, j, reference.jump(net.get(i, j), rng))
            }
        };
        let from = net.get(i, j);
        if to == from {
            return None;
        }
        if let Reference::DiscUnif(a, b) = reference
            && (to < a || to > b)
        {
            return None;
        }
        let forward = reference.log_jump(from, to);
        let backward = reference.log_jump(to, from);
        let log_ratio = match self.p0 {
            None => backward - forward,
            Some(p0) => {
                let q = 1.0 - p0;
                let d_over = p0 * self.dyads / q;
                let e = e as f64;
                if from == 0.0 {
                    (backward.exp() + d_over / (e + 1.0)).ln() - forward + if e == 0.0 { q.ln() } else { 0.0 }
                } else if to == 0.0 {
                    backward - (forward.exp() + d_over / e).ln() - if e == 1.0 { q.ln() } else { 0.0 }
                } else {
                    backward - forward
                }
            }
        };
        Some(WtMove { i, j, from, to, log_ratio: log_ratio + reference.log_h(to) - reference.log_h(from) })
    }

    /// Makes the move.
    fn apply(&mut self, net: &mut WtNetwork, mv: &WtMove) {
        net.set(mv.i, mv.j, mv.to);
        if (mv.from == 0.0) != (mv.to == 0.0) {
            self.free_nonzero.toggle(net.directed(), mv.i, mv.j, mv.to == 0.0);
        }
    }
}

/// Runs `burnin` steps, then records `samplesize` samples `interval` steps apart.
#[allow(clippy::too_many_arguments)]
pub fn run_wt_chain(
    model: &WtModel,
    proposal: &WtProposal,
    mut net: WtNetwork,
    theta: &[f64],
    burnin: u64,
    interval: u64,
    samplesize: usize,
    rng: &mut Rng,
    keep_networks: bool,
) -> WtChain {
    let mut stats = model.summary(&net);
    let p = stats.len();
    let mut delta = vec![0.0; p];
    let mut state = model.state(&net);
    let mut proposer = Proposer::new(proposal, &net);
    let mut sample = Vec::with_capacity(samplesize * p);
    let mut networks = Vec::new();
    for step in 1..=burnin + interval * samplesize as u64 {
        // A proposal that can't be made, or changes nothing, still counts as a step.
        if let Some(mv) = proposer.propose(&net, rng) {
            model.change(&state, &net, mv.i, mv.j, mv.from, mv.to, &mut delta);
            let log_accept = mv.log_ratio
                + theta.iter().zip(&delta).filter(|(_, d)| **d != 0.0).map(|(t, d)| t * d).sum::<f64>();
            if log_accept >= 0.0 || rng.unif() < log_accept.exp() {
                proposer.apply(&mut net, &mv);
                model.apply(&mut state, mv.i, mv.j, mv.to);
                stats.iter_mut().zip(&delta).for_each(|(s, d)| *s += d);
            }
            if net.n_nonzero() > proposal.max_nonzero {
                return WtChain { stats: sample, networks, last: net, exceeded: true };
            }
        }
        if step > burnin && (step - burnin).is_multiple_of(interval) {
            sample.extend_from_slice(&stats);
            if keep_networks {
                networks.push(net.triples());
            }
        }
    }
    WtChain { stats: sample, networks, last: net, exceeded: false }
}

/// The result of a valued simulated annealing run, as `sampler::San`.
pub struct WtSan {
    pub last: WtNetwork,
    pub deviations: Vec<f64>,
    pub proposed: Vec<f64>,
}

/// Simulated annealing of a valued network (ergm's SAN): as the binary one
/// (`sampler::run_san`), with the valued proposals, whose probabilities and
/// reference measure are ignored, as in ergm.
pub fn run_wt_san(model: &WtModel, proposal: &WtProposal, mut net: WtNetwork, settings: &SanSettings, rng: &mut Rng) -> WtSan {
    let stats = model.summary(&net);
    let q = settings.targeted.len();
    let mut delta = vec![0.0; stats.len()];
    let mut deviations: Vec<f64> = settings.targeted.iter().zip(settings.target).map(|(&k, t)| stats[k] - t).collect();
    let mut state = model.state(&net);
    let mut proposer = Proposer::new(proposal, &net);
    let samples = settings.samplesize.max(1) as u64;
    let interval = (settings.nsteps / samples).max(1);
    let burnin = settings.nsteps.saturating_sub((samples - 1) * interval);
    let (mut out, mut proposed_rows) = (Vec::new(), Vec::new());
    let mut finished = deviations.iter().all(|&d| d == 0.0);
    for sample in 0..samples {
        let mut proposed = vec![0.0; q];
        let steps = if sample == 0 { burnin } else { interval };
        for _ in 0..steps {
            if finished {
                break;
            }
            let Some(mv) = proposer.propose(&net, rng) else { continue };
            model.change(&state, &net, mv.i, mv.j, mv.from, mv.to, &mut delta);
            let mut energy = 0.0; // the change of (s - target)' W (s - target)
            for (a, &k) in settings.targeted.iter().enumerate() {
                proposed[a] += delta[k];
                let weighted: f64 = settings.targeted.iter().enumerate().map(|(b, &l)| delta[l] * settings.weights[a * q + b]).sum();
                energy += weighted * (delta[k] + 2.0 * deviations[a]);
            }
            let offset: f64 = settings.offsets.iter().filter(|&&(k, _)| delta[k] != 0.0).map(|&(k, eta)| eta * delta[k]).sum();
            let accept = if settings.tau == 0.0 { energy - offset <= 0.0 } else { energy / settings.tau - offset <= -rng.unif().ln() };
            if accept {
                proposer.apply(&mut net, &mv);
                model.apply(&mut state, mv.i, mv.j, mv.to);
                for (a, &k) in settings.targeted.iter().enumerate() {
                    deviations[a] += delta[k];
                }
                finished = deviations.iter().all(|&d| d == 0.0);
            }
        }
        out.extend_from_slice(&deviations);
        proposed_rows.extend_from_slice(&proposed);
        if finished {
            break;
        }
    }
    WtSan { last: net, deviations: out, proposed: proposed_rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_factorials_and_gamma() {
        assert!((ln_factorial(5.0) - 120f64.ln()).abs() < 1e-12);
        // Python's math.lgamma.
        for (x, expected) in [
            (0.5, 0.572_364_942_924_700_4),
            (3.7, 1.428_072_326_665_388_5),
            (10.0, 12.801_827_480_081_467),
            (60.5, 186.578_917_833_337_84),
            (1234.5, 7_550.550_901_077_895),
        ] {
            assert!((ln_gamma(x) - expected).abs() < 1e-10 * expected.abs().max(1.0), "ln_gamma({x})");
        }
    }

    #[test]
    fn summaries_add_up_changes() {
        let spec = |name: &str, reals: Vec<f64>, ints: Vec<i64>| TermSpec(name.into(), reals, ints, vec![]);
        let model = WtModel::new(
            4,
            true,
            &[
                spec("sum", vec![1.0], vec![]),
                spec("mutual", vec![], vec![0]),
                spec("transitiveweights", vec![], vec![0, 0, 0]),
                spec("CMP", vec![], vec![]),
                spec("atleast", vec![2.0], vec![]),
            ],
            12.0,
        )
        .unwrap();
        let net = WtNetwork::from_values(4, true, &[(0, 1, 3.0), (1, 0, 2.0), (1, 2, 2.0), (0, 2, 1.0)]).unwrap();
        let s = model.summary(&net);
        // sum 8; mutual: min(3, 2) = 2; transitive weights: 0 -> 2 via 1,
        // min(min(3, 2), 1) = 1, and 1 -> 2 via 0, min(min(2, 1), 2) = 1; CMP
        // log(3! 2! 2! 1!); three values at least 2.
        assert_eq!(s[0], 8.0);
        assert_eq!(s[1], 2.0);
        assert_eq!(s[2], 2.0);
        assert!((s[3] - (6f64 * 2.0 * 2.0).ln()).abs() < 1e-12);
        assert_eq!(s[4], 3.0);
    }

    #[test]
    fn normal_draws_have_the_standard_moments() {
        let mut rng = Rng::new(1);
        let n = 200_000;
        let draws: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let kurtosis = draws.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n as f64 / (var * var);
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
        assert!((kurtosis - 3.0).abs() < 0.05, "{kurtosis}");
    }
}
