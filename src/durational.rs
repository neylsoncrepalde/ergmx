//! tergm's durational terms: statistics of the ages of the ties, for models
//! of networks evolving in time steps. A tie's age is 1 in the step it forms
//! and grows by one each step it persists. As tergm's change statistics,
//! computed before the time step's tick, a step's model sees the ties' ages
//! at its start: a tie of the previous network has the age it had then, a
//! tie formed in the step age 1. The statistics reported after a step, as
//! tergm's, count the ages at its end (`TieAges::ticked`).
//!
//! A durational term may keep sums over the ties (its storage, a vector of
//! numbers), updated at each toggle, so that its change statistics are O(1).

use rustc_hash::FxHashMap;

use crate::network::Network;
use crate::terms::{Term, TermSpec, build_term};

/// The ages of the ties of a previous network, at its time step.
#[derive(Clone)]
pub struct TieAges {
    prev: Network,
    ages: FxHashMap<u64, u32>,
}

fn key(directed: bool, i: u32, j: u32) -> u64 {
    let (a, b) = if !directed && i > j { (j, i) } else { (i, j) };
    ((a as u64) << 32) | b as u64
}

impl TieAges {
    /// The previous network with its ties' ages (ties not given have age 1).
    pub fn new(prev: Network, ages: FxHashMap<u64, u32>) -> Self {
        Self { prev, ages }
    }

    /// The ages, keyed as `key`, from (i, j, age) rows.
    pub fn map(directed: bool, rows: &[(u32, u32, u32)]) -> FxHashMap<u64, u32> {
        rows.iter().map(|&(i, j, a)| (key(directed, i, j), a)).collect()
    }

    /// The age of (i, j) in the previous network (1 if not given).
    #[inline]
    pub fn previous(&self, i: u32, j: u32) -> u32 {
        self.ages.get(&key(self.prev.directed(), i, j)).copied().unwrap_or(1)
    }

    /// The age (i, j) has as a tie in the step's model: its age in the
    /// previous network if it was a tie there, else 1.
    #[inline]
    pub fn now(&self, i: u32, j: u32) -> f64 {
        if self.prev.has_edge(i, j) { self.previous(i, j) as f64 } else { 1.0 }
    }

    /// The ages of `net`'s ties at the end of the step, as rows (i, j, age):
    /// the previous network's ages at the next step.
    pub fn advance(&self, net: &Network) -> Vec<(u32, u32, u32)> {
        let age = |i, j| if self.prev.has_edge(i, j) { self.previous(i, j) + 1 } else { 1 };
        net.edges().iter().map(|&(i, j)| (i, j, age(i, j))).collect()
    }

    /// The previous network's ages one step later, those its ties have at the
    /// end of the step if they persist: for the statistics after the tick.
    pub fn ticked(prev: &Network, ages: &FxHashMap<u64, u32>) -> FxHashMap<u64, u32> {
        let local = TieAges::new(prev.clone(), ages.clone());
        local.advance(prev).iter().map(|&(i, j, a)| (key(prev.directed(), i, j), a)).collect()
    }
}

/// A term of the ages of the ties.
pub trait AgedTerm: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    /// The storage of the network `net`, whose ties have these ages.
    fn init(&self, _net: &Network, _ages: &TieAges) -> Vec<f64> {
        Vec::new()
    }

    /// Writes to `out` the change from toggling (i, j) (adding it if `sign`
    /// is 1), whose age as a tie is `age`.
    #[allow(clippy::too_many_arguments)]
    fn change(&self, net: &Network, store: &[f64], i: u32, j: u32, age: f64, sign: f64, out: &mut [f64]);

    /// Updates the storage for the toggle, before it changes `net`.
    fn update(&self, _net: &Network, _store: &mut [f64], _i: u32, _j: u32, _age: f64, _sign: f64) {}

    /// Adds to `out` the statistics of the network without ties.
    fn empty(&self, _out: &mut [f64]) {}
}

#[inline]
fn transform(age: f64, log: bool) -> f64 {
    if log { age.ln() } else { age }
}

/// A mean, or `emptyval` without anything to average.
#[inline]
fn mean(sum: f64, count: f64, emptyval: f64) -> f64 {
    if count > 0.0 { sum / count } else { emptyval }
}

/// edges.ageinterval: the ties with age in [from, to), for each interval.
struct EdgesAgeInterval(Vec<(f64, f64)>);

impl AgedTerm for EdgesAgeInterval {
    fn n_stats(&self) -> usize {
        self.0.len()
    }

    fn change(&self, _: &Network, _: &[f64], _: u32, _: u32, age: f64, sign: f64, out: &mut [f64]) {
        for (o, &(from, to)) in out.iter_mut().zip(&self.0) {
            if from <= age && age < to {
                *o = sign;
            }
        }
    }
}

/// edge.ages and edgecov.ages: the sum over ties of their ages (times a covariate).
struct EdgeAges(Option<Vec<f64>>);

impl AgedTerm for EdgeAges {
    fn change(&self, net: &Network, _: &[f64], i: u32, j: u32, age: f64, sign: f64, out: &mut [f64]) {
        out[0] = sign * age * self.0.as_ref().map_or(1.0, |x| x[dyad_index(net, i, j)]);
    }
}

#[inline]
fn dyad_index(net: &Network, i: u32, j: u32) -> usize {
    let (a, b) = if !net.directed() && i > j { (j, i) } else { (i, j) };
    a as usize * net.n() as usize + b as usize
}

/// mean.age and edgecov.mean.age: the (weighted) mean of the ties' ages, or of
/// their logs. Storage: the weighted sum and the total weight.
struct MeanAge {
    weights: Option<Vec<f64>>,
    emptyval: f64,
    log: bool,
}

impl MeanAge {
    fn weight(&self, net: &Network, i: u32, j: u32) -> f64 {
        self.weights.as_ref().map_or(1.0, |x| x[dyad_index(net, i, j)])
    }
}

impl AgedTerm for MeanAge {
    fn init(&self, net: &Network, ages: &TieAges) -> Vec<f64> {
        let mut store = vec![0.0; 2];
        for &(i, j) in net.edges() {
            let w = self.weight(net, i, j);
            store[0] += w * transform(ages.now(i, j), self.log);
            store[1] += w;
        }
        store
    }

    fn change(&self, net: &Network, store: &[f64], i: u32, j: u32, age: f64, sign: f64, out: &mut [f64]) {
        let w = self.weight(net, i, j);
        let (sum, total) = (store[0] + sign * w * transform(age, self.log), store[1] + sign * w);
        out[0] = mean(sum, total, self.emptyval) - mean(store[0], store[1], self.emptyval);
    }

    fn update(&self, net: &Network, store: &mut [f64], i: u32, j: u32, age: f64, sign: f64) {
        let w = self.weight(net, i, j);
        store[0] += sign * w * transform(age, self.log);
        store[1] += sign * w;
    }

    fn empty(&self, out: &mut [f64]) {
        out[0] += self.emptyval;
    }
}

/// nodefactor.mean.age and nodemix.mean.age: for each group of ties' ends
/// (the levels of the ends' vertices) or of ties (their mixing types), the
/// mean of their ages. `groups` gives, for a tie (i, j), the statistics it
/// counts in, as many times as it does. Storage: the sums, then the counts.
struct GroupMeanAge {
    codes: Vec<i64>,
    /// nodemix: levels x levels to the statistic (or -1); nodefactor: None.
    cells: Option<(usize, Vec<i64>)>,
    emptyval: Vec<f64>,
    log: bool,
}

impl GroupMeanAge {
    fn groups(&self, net: &Network, i: u32, j: u32) -> [Option<usize>; 2] {
        let (i, j) = if !net.directed() && i > j { (j, i) } else { (i, j) };
        let (a, b) = (self.codes[i as usize], self.codes[j as usize]);
        match &self.cells {
            None => [(a >= 0).then_some(a as usize), (b >= 0).then_some(b as usize)],
            Some((levels, map)) => {
                let cell = if a >= 0 && b >= 0 { map[a as usize * levels + b as usize] } else { -1 };
                [(cell >= 0).then_some(cell as usize), None]
            }
        }
    }
}

impl AgedTerm for GroupMeanAge {
    fn n_stats(&self) -> usize {
        self.emptyval.len()
    }

    fn init(&self, net: &Network, ages: &TieAges) -> Vec<f64> {
        let k = self.n_stats();
        let mut store = vec![0.0; 2 * k];
        for &(i, j) in net.edges() {
            let f = transform(ages.now(i, j), self.log);
            for g in self.groups(net, i, j).into_iter().flatten() {
                store[g] += f;
                store[k + g] += 1.0;
            }
        }
        store
    }

    fn change(&self, net: &Network, store: &[f64], i: u32, j: u32, age: f64, sign: f64, out: &mut [f64]) {
        let k = self.n_stats();
        let f = transform(age, self.log);
        let groups = self.groups(net, i, j);
        for (position, g) in groups.iter().enumerate() {
            let Some(g) = *g else { continue };
            if groups[..position].contains(&Some(g)) {
                continue; // a tie whose two ends are in the group: done with both
            }
            let times = groups.iter().filter(|h| **h == Some(g)).count() as f64;
            let (sum, count) = (store[g] + sign * times * f, store[k + g] + sign * times);
            out[g] = mean(sum, count, self.emptyval[g]) - mean(store[g], store[k + g], self.emptyval[g]);
        }
    }

    fn update(&self, net: &Network, store: &mut [f64], i: u32, j: u32, age: f64, sign: f64) {
        let k = self.n_stats();
        let f = transform(age, self.log);
        for g in self.groups(net, i, j).into_iter().flatten() {
            store[g] += sign * f;
            store[k + g] += sign;
        }
    }

    fn empty(&self, out: &mut [f64]) {
        out.iter_mut().zip(&self.emptyval).for_each(|(o, e)| *o += e);
    }
}

/// degree.mean.age and degrange.mean.age (undirected): for each degree range
/// [from, to) (and level of an attribute), the mean age of the ties at
/// vertices with degrees in it, a tie counted once per such end. Storage:
/// each vertex's sum over its ties, then the statistics' sums and counts.
struct DegreeMeanAge {
    /// (from, to, level, or -1 for any)
    cells: Vec<(f64, f64, i64)>,
    codes: Vec<i64>,
    emptyval: Vec<f64>,
    log: bool,
}

impl DegreeMeanAge {
    #[inline]
    fn counts(&self, k: usize, v: u32, degree: f64) -> bool {
        let (from, to, level) = self.cells[k];
        from <= degree && degree < to && (level < 0 || self.codes[v as usize] == level)
    }

    /// The statistics' sums and counts after the toggle, from `store`.
    fn after(&self, net: &Network, store: &[f64], i: u32, j: u32, f: f64, sign: f64) -> Vec<f64> {
        let (n, k) = (net.n() as usize, self.cells.len());
        let mut sums = store[n..n + 2 * k].to_vec();
        for v in [i, j] {
            let degree = net.neighbours(v).len() as f64;
            let (vsum, vsum_after) = (store[v as usize], store[v as usize] + sign * f);
            for c in 0..k {
                if self.counts(c, v, degree) {
                    sums[c] -= vsum;
                    sums[k + c] -= degree;
                }
                if self.counts(c, v, degree + sign) {
                    sums[c] += vsum_after;
                    sums[k + c] += degree + sign;
                }
            }
        }
        sums
    }
}

impl AgedTerm for DegreeMeanAge {
    fn n_stats(&self) -> usize {
        self.cells.len()
    }

    fn init(&self, net: &Network, ages: &TieAges) -> Vec<f64> {
        let (n, k) = (net.n() as usize, self.cells.len());
        let mut store = vec![0.0; n + 2 * k];
        for &(i, j) in net.edges() {
            let f = transform(ages.now(i, j), self.log);
            store[i as usize] += f;
            store[j as usize] += f;
        }
        for v in 0..net.n() {
            let degree = net.neighbours(v).len() as f64;
            for c in 0..k {
                if self.counts(c, v, degree) {
                    store[n + c] += store[v as usize];
                    store[n + k + c] += degree;
                }
            }
        }
        store
    }

    fn change(&self, net: &Network, store: &[f64], i: u32, j: u32, age: f64, sign: f64, out: &mut [f64]) {
        let (n, k) = (net.n() as usize, self.cells.len());
        let after = self.after(net, store, i, j, transform(age, self.log), sign);
        for c in 0..k {
            out[c] = mean(after[c], after[k + c], self.emptyval[c])
                - mean(store[n + c], store[n + k + c], self.emptyval[c]);
        }
    }

    fn update(&self, net: &Network, store: &mut [f64], i: u32, j: u32, age: f64, sign: f64) {
        let (n, k) = (net.n() as usize, self.cells.len());
        let f = transform(age, self.log);
        let after = self.after(net, store, i, j, f, sign);
        store[n..n + 2 * k].copy_from_slice(&after);
        store[i as usize] += sign * f;
        store[j as usize] += sign * f;
    }

    fn empty(&self, out: &mut [f64]) {
        out.iter_mut().zip(&self.emptyval).for_each(|(o, e)| *o += e);
    }
}

/// EdgeAges(formula): for each statistic of a dyad-independent formula, the
/// sum over ties of their ages times the statistic's change on adding them.
struct EdgeAgesOf {
    terms: Vec<Box<dyn Term>>,
    n_stats: usize,
}

impl AgedTerm for EdgeAgesOf {
    fn n_stats(&self) -> usize {
        self.n_stats
    }

    fn change(&self, net: &Network, _: &[f64], i: u32, j: u32, age: f64, sign: f64, out: &mut [f64]) {
        let mut start = 0;
        for term in &self.terms {
            let part = &mut out[start..start + term.n_stats()];
            term.change(net, i, j, 1.0, part); // the change on adding (i, j), dyad-independent
            part.iter_mut().for_each(|x| *x *= sign * age);
            start += term.n_stats();
        }
    }
}

/// Whether a term spec is durational ("aged." names).
pub fn is_aged(spec: &TermSpec) -> bool {
    spec.0.starts_with("aged.")
}

/// A durational term from its spec: `aged.<name>`, with its parameters.
pub fn build_aged(n: usize, directed: bool, spec: &TermSpec) -> Result<Box<dyn AgedTerm>, String> {
    let TermSpec(name, reals, ints, children) = spec;
    let matrix = |values: &[f64]| -> Result<Vec<f64>, String> {
        if values.len() != n * n {
            return Err(format!("{name}: a covariate of {n} x {n} values"));
        }
        Ok(values.to_vec())
    };
    // reals: [emptyval..., log] for the means.
    let log = || reals.last().is_some_and(|&x| x != 0.0);
    let emptyvals = |k: usize| -> Result<Vec<f64>, String> {
        let values = &reals[..reals.len().saturating_sub(1)];
        match values.len() {
            1 => Ok(vec![values[0]; k]),
            m if m == k => Ok(values.to_vec()),
            _ => Err(format!("{name}: emptyval must have 1 value or {k}")),
        }
    };
    Ok(match name.as_str() {
        "aged.edges_ageinterval" => {
            if reals.is_empty() || reals.len() % 2 != 0 {
                return Err(format!("{name}: intervals as (from, to) pairs"));
            }
            Box::new(EdgesAgeInterval(reals.chunks(2).map(|c| (c[0], c[1])).collect()))
        }
        "aged.edge_ages" => Box::new(EdgeAges(None)),
        "aged.edgecov_ages" => Box::new(EdgeAges(Some(matrix(reals)?))),
        "aged.mean_age" => Box::new(MeanAge { weights: None, emptyval: emptyvals(1)?[0], log: log() }),
        "aged.edgecov_mean_age" => {
            // reals: [emptyval, log, the covariate...]
            if reals.len() < 2 {
                return Err(format!("{name}: missing parameters"));
            }
            Box::new(MeanAge { weights: Some(matrix(&reals[2..])?), emptyval: reals[0], log: reals[1] != 0.0 })
        }
        "aged.nodefactor_mean_age" | "aged.nodemix_mean_age" => {
            // ints: the vertices' codes, then (nodemix) the levels and the levels x levels map.
            if ints.len() < n {
                return Err(format!("{name}: one code per vertex"));
            }
            let codes = ints[..n].to_vec();
            let (cells, k) = if name == "aged.nodemix_mean_age" {
                let levels = *ints.get(n).ok_or_else(|| format!("{name}: missing the levels"))? as usize;
                let map = ints[n + 1..].to_vec();
                if map.len() != levels * levels {
                    return Err(format!("{name}: a {levels} x {levels} map"));
                }
                let k = map.iter().copied().max().unwrap_or(-1).max(-1) as usize + 1;
                let k = if map.iter().all(|&m| m < 0) { 0 } else { k };
                (Some((levels, map)), k)
            } else {
                let k = codes.iter().copied().max().unwrap_or(-1).max(-1);
                (None, (k + 1) as usize)
            };
            Box::new(GroupMeanAge { codes, cells, emptyval: emptyvals(k)?, log: log() })
        }
        "aged.degree_mean_age" => {
            // ints: the vertices' codes (or none), then (from, to, level) triples as reals' prefix.
            if directed {
                return Err(format!("{name} is for undirected networks"));
            }
            let k = ints.len().saturating_sub(n) / 3;
            let cells = (0..k).map(|c| {
                let t = &ints[n + 3 * c..n + 3 * c + 3];
                (t[0] as f64, if t[1] < 0 { f64::INFINITY } else { t[1] as f64 }, t[2])
            });
            let cells: Vec<_> = cells.collect();
            if ints.len() != n + 3 * k || k == 0 {
                return Err(format!("{name}: one code per vertex, then (from, to, level) triples"));
            }
            Box::new(DegreeMeanAge { cells, codes: ints[..n].to_vec(), emptyval: emptyvals(k)?, log: log() })
        }
        "aged.EdgeAges" => {
            let terms = children.iter().map(|t| build_term(n, directed, t)).collect::<Result<Vec<_>, _>>()?;
            let n_stats = terms.iter().map(|t| t.n_stats()).sum();
            Box::new(EdgeAgesOf { terms, n_stats })
        }
        _ => return Err(format!("unknown durational term {name:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::terms::Model;

    fn spec(name: &str, reals: Vec<f64>, ints: Vec<i64>, children: Vec<TermSpec>) -> TermSpec {
        TermSpec(name.into(), reals, ints, children)
    }

    /// Statistics tracked change by change, over random toggles, against
    /// those recomputed from the network, for every durational term.
    #[test]
    fn tracked_statistics_match_summaries() {
        for directed in [false, true] {
            let n = 12u32;
            let mut rng = Rng::new(7);
            let codes: Vec<i64> = (0..n).map(|v| (v % 3) as i64).collect();
            let sex: Vec<i64> = (0..n).map(|v| (v % 2) as i64).collect();
            let x: Vec<f64> = (0..n * n).map(|k| (k % 5) as f64 / 2.0).collect();
            let mut specs = vec![
                spec("aged.edges_ageinterval", vec![1.0, 3.0, 3.0, f64::INFINITY], vec![], vec![]),
                spec("aged.edge_ages", vec![], vec![], vec![]),
                spec("aged.mean_age", vec![2.5, 0.0], vec![], vec![]),
                spec("aged.mean_age", vec![0.0, 1.0], vec![], vec![]),
                spec("aged.edgecov_ages", x.clone(), vec![], vec![]),
                spec("aged.edgecov_mean_age", [vec![1.0, 0.0], x.clone()].concat(), vec![], vec![]),
                spec("aged.nodefactor_mean_age", vec![0.5, 0.0], codes.clone(), vec![]),
                spec("aged.nodemix_mean_age", vec![0.0, 1.0], [codes.clone(), vec![3], (0..9).map(|k| if k == 0 { -1 } else { k - 1 }).collect()].concat(), vec![]),
                spec("aged.EdgeAges", vec![], vec![], vec![spec("edges", vec![], vec![], vec![])]),
            ];
            if !directed {
                // degree 1, 2 and [3, Inf), by sex for the last two
                let triples = [vec![1, 2, -1], vec![2, 3, 0], vec![3, -1, 1]].concat();
                specs.push(spec("aged.degree_mean_age", vec![0.0, 0.0], [sex.clone(), triples].concat(), vec![]));
            }
            let model = Model::new(n, directed, &specs, None, Vec::new()).unwrap();
            let mut prev = Network::new(n, directed);
            let mut rows = Vec::new();
            for _ in 0..30 {
                let (i, j) = (rng.below(n as u64) as u32, rng.below(n as u64) as u32);
                if i != j && !prev.has_edge(i, j) {
                    prev.toggle(i, j);
                    rows.push((i, j, 1 + rng.below(6) as u32));
                }
            }
            let ages = TieAges::new(prev.clone(), TieAges::map(directed, &rows));
            let mut state = model.state_aged(prev.clone(), Vec::new(), &[], Some(ages.clone()));
            let mut stats = model.summary_aged(&prev, &[], &[], Some(ages.clone()));
            let mut delta = vec![0.0; model.n_stats()];
            for step in 1..=3000 {
                let (i, j) = (rng.below(n as u64) as u32, rng.below(n as u64) as u32);
                if i == j {
                    continue;
                }
                model.change(&state, i, j, &mut delta);
                model.toggle(&mut state, i, j);
                stats.iter_mut().zip(&delta).for_each(|(s, d)| *s += d);
                if step % 100 == 0 {
                    let fresh = model.summary_aged(&state.net, &[], &[], Some(ages.clone()));
                    for (k, (a, b)) in stats.iter().zip(&fresh).enumerate() {
                        assert!((a - b).abs() < 1e-8, "directed {directed}, step {step}, statistic {k}: {a} vs {b}");
                    }
                }
            }
        }
    }
}
