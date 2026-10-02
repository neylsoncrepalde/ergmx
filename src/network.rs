//! A binary network without self-loops, stored as sorted neighbour lists.
//!
//! An edge list with a position index gives O(1) random edges, which the TNT
//! proposal needs. Undirected edges are stored once, as (min, max). Directed
//! networks also keep, for each vertex, its neighbours in either direction.
//! A network can also keep the number of shared partners of every pair that
//! has any (as ergm's shared partner cache), which its toggles update.

use rustc_hash::FxHashMap;

use crate::rng::Rng;

#[derive(Clone)]
pub struct Network {
    n: u32,
    directed: bool,
    /// Out-neighbours, or all neighbours if undirected. Sorted.
    out: Vec<Vec<u32>>,
    /// In-neighbours. Sorted, and empty if undirected.
    inn: Vec<Vec<u32>>,
    /// Neighbours in either direction. Sorted, and empty if undirected.
    both: Vec<Vec<u32>>,
    edges: Vec<(u32, u32)>,
    position: FxHashMap<u64, usize>,
    /// Shared partner counts, if kept.
    partners: Option<Box<PartnerCache>>,
}

/// The kinds of shared partner a cache can keep: of undirected networks, and
/// of directed ones the two-paths a -> k -> b (OTP, and ITP read backwards),
/// the common out-neighbours (OSP) and the common in-neighbours (ISP).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Partners {
    Undirected,
    TwoPaths,
    OutShared,
    InShared,
}

/// Shared partner counts of the pairs that have any, by kind: keyed by
/// (a, b), ordered for two-paths and as (min, max) otherwise.
#[derive(Clone, Default)]
pub struct PartnerCache {
    counts: [Option<FxHashMap<u64, u32>>; 4],
}

/// Above this many pairs of neighbours (a bound on the cache's size), a
/// network doesn't keep the counts of a kind: they are computed instead.
const CACHE_LIMIT: u64 = 20_000_000;
/// Below this many pairs of neighbours per vertex, neither: merging short
/// neighbour lists is faster than updating counts (8% faster without the
/// cache on faux.magnolia.high, 3.8 pairs per vertex; 10% slower on
/// faux.mesa.high, 8.4, and faux.dixon.high, 34).
const CACHE_MIN_PAIRS: f64 = 6.0;

#[inline]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

#[inline]
fn bump(map: &mut FxHashMap<u64, u32>, key: u64, add: bool) {
    if add {
        *map.entry(key).or_insert(0) += 1;
    } else if let Some(c) = map.get_mut(&key) {
        *c -= 1;
        if *c == 0 {
            map.remove(&key);
        }
    }
}

impl Network {
    pub fn new(n: u32, directed: bool) -> Self {
        let lists = || vec![Vec::new(); n as usize];
        Self {
            n,
            directed,
            out: lists(),
            inn: if directed { lists() } else { Vec::new() },
            both: if directed { lists() } else { Vec::new() },
            edges: Vec::new(),
            position: FxHashMap::default(),
            partners: None,
        }
    }

    /// Keeps the shared partner counts of this kind from now on, unless
    /// there would be too many, or the network is too sparse for them to
    /// help: returns whether it does.
    pub fn keep_partners(&mut self, kind: Partners) -> bool {
        self.keep_partners_if(kind, CACHE_MIN_PAIRS)
    }

    #[cfg(test)]
    fn keep_partners_always(&mut self, kind: Partners) -> bool {
        self.keep_partners_if(kind, 0.0)
    }

    fn keep_partners_if(&mut self, kind: Partners, min_pairs: f64) -> bool {
        if self.partners.as_ref().is_some_and(|c| c.counts[kind as usize].is_some()) {
            return true;
        }
        if (kind != Partners::Undirected) != self.directed {
            return false;
        }
        let pairs = |lists: &[Vec<u32>]| lists.iter().map(|l| (l.len() as u64).pow(2)).sum::<u64>();
        let size = match kind {
            Partners::OutShared => pairs(&self.inn),
            Partners::Undirected | Partners::InShared => pairs(&self.out),
            Partners::TwoPaths => (0..self.n as usize).map(|k| self.inn[k].len() as u64 * self.out[k].len() as u64).sum(),
        };
        // In sparse networks, updating the counts costs more than counting.
        if size > CACHE_LIMIT || (size as f64) < min_pairs * self.n as f64 {
            return false;
        }
        let mut map = FxHashMap::default();
        for k in 0..self.n as usize {
            match kind {
                // Pairs of neighbours of k, with k as a shared partner.
                Partners::Undirected | Partners::OutShared | Partners::InShared => {
                    let list = if kind == Partners::OutShared { &self.inn[k] } else { &self.out[k] };
                    for (x, &a) in list.iter().enumerate() {
                        for &b in &list[x + 1..] {
                            *map.entry(pair_key(a, b)).or_insert(0) += 1;
                        }
                    }
                }
                Partners::TwoPaths => {
                    for &a in &self.inn[k] {
                        for &b in self.out[k].iter().filter(|&&b| b != a) {
                            *map.entry(pair_key(a, b)).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        self.partners.get_or_insert_with(Default::default).counts[kind as usize] = Some(map);
        true
    }

    /// The number of shared partners of this kind of (a, b), if kept.
    #[inline]
    pub fn kept_partners(&self, kind: Partners, a: u32, b: u32) -> Option<u32> {
        let map = self.partners.as_ref()?.counts[kind as usize].as_ref()?;
        let key = match kind {
            Partners::TwoPaths => pair_key(a, b),
            _ => pair_key(a.min(b), a.max(b)),
        };
        Some(map.get(&key).copied().unwrap_or(0))
    }

    /// Updates the kept counts for the toggle of (i, j), before it.
    fn update_partners(&mut self, i: u32, j: u32, add: bool) {
        let Some(cache) = self.partners.as_mut() else { return };
        let sym = |a: u32, b: u32| pair_key(a.min(b), a.max(b));
        let (iu, ju) = (i as usize, j as usize);
        let [undirected, two_paths, out_shared, in_shared] = &mut cache.counts;
        if let Some(map) = undirected {
            // i gains (or loses) the partner j with j's neighbours, and j with i's.
            for &u in self.out[ju].iter().filter(|&&u| u != i) {
                bump(map, sym(i, u), add);
            }
            for &u in self.out[iu].iter().filter(|&&u| u != j) {
                bump(map, sym(j, u), add);
            }
        }
        if let Some(map) = two_paths {
            for &v in self.out[ju].iter().filter(|&&v| v != i) {
                bump(map, pair_key(i, v), add); // i -> j -> v
            }
            for &u in self.inn[iu].iter().filter(|&&u| u != j) {
                bump(map, pair_key(u, j), add); // u -> i -> j
            }
        }
        if let Some(map) = out_shared {
            for &b in self.inn[ju].iter().filter(|&&b| b != i) {
                bump(map, sym(i, b), add); // i -> j <- b
            }
        }
        if let Some(map) = in_shared {
            for &b in self.out[iu].iter().filter(|&&b| b != j) {
                bump(map, sym(j, b), add); // b <- i -> j
            }
        }
    }

    pub fn from_edges(n: u32, directed: bool, edges: &[(u32, u32)]) -> Result<Self, String> {
        let mut net = Self::new(n, directed);
        for &(i, j) in edges {
            if i >= n || j >= n {
                return Err(format!("edge ({i}, {j}) refers to a vertex >= {n}"));
            }
            if i == j {
                return Err(format!("self-loop on vertex {i}"));
            }
            if net.has_edge(i, j) {
                return Err(format!("duplicated edge ({i}, {j})"));
            }
            net.toggle(i, j);
        }
        Ok(net)
    }

    pub fn n(&self) -> u32 {
        self.n
    }

    pub fn directed(&self) -> bool {
        self.directed
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    /// Neighbours of `i`, in either direction if the network is directed.
    pub fn neighbours(&self, i: u32) -> &[u32] {
        if self.directed { &self.both[i as usize] } else { &self.out[i as usize] }
    }

    /// Vertices `i` sends a tie to (all neighbours if undirected).
    pub fn out_neighbours(&self, i: u32) -> &[u32] {
        &self.out[i as usize]
    }

    /// Vertices that send a tie to `i` (all neighbours if undirected).
    pub fn in_neighbours(&self, i: u32) -> &[u32] {
        if self.directed { &self.inn[i as usize] } else { &self.out[i as usize] }
    }

    #[inline]
    pub fn has_edge(&self, i: u32, j: u32) -> bool {
        self.out[i as usize].binary_search(&j).is_ok()
    }

    fn key(&self, i: u32, j: u32) -> u64 {
        let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
        ((a as u64) << 32) | b as u64
    }

    /// Adds the edge (i, j) if it is absent, removes it otherwise.
    pub fn toggle(&mut self, i: u32, j: u32) {
        let key = self.key(i, j);
        if self.partners.is_some() {
            let add = !self.position.contains_key(&key);
            self.update_partners(i, j, add);
        }
        if let Some(p) = self.position.remove(&key) {
            self.edges.swap_remove(p);
            if p < self.edges.len() {
                let (a, b) = self.edges[p];
                let moved = self.key(a, b);
                self.position.insert(moved, p);
            }
            remove_sorted(&mut self.out[i as usize], j);
            if self.directed {
                remove_sorted(&mut self.inn[j as usize], i);
                if !self.has_edge(j, i) {
                    remove_sorted(&mut self.both[i as usize], j);
                    remove_sorted(&mut self.both[j as usize], i);
                }
            } else {
                remove_sorted(&mut self.out[j as usize], i);
            }
        } else {
            self.position.insert(key, self.edges.len());
            let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
            self.edges.push((a, b));
            insert_sorted(&mut self.out[i as usize], j);
            if self.directed {
                insert_sorted(&mut self.inn[j as usize], i);
                insert_sorted(&mut self.both[i as usize], j);
                insert_sorted(&mut self.both[j as usize], i);
            } else {
                insert_sorted(&mut self.out[j as usize], i);
            }
        }
    }

    pub fn random_edge(&self, rng: &mut Rng) -> (u32, u32) {
        self.edges[rng.below(self.edges.len() as u64) as usize]
    }

    pub fn random_dyad(&self, rng: &mut Rng) -> (u32, u32) {
        let n = self.n as u64;
        let i = rng.below(n) as u32;
        let mut j = rng.below(n - 1) as u32;
        if j >= i {
            j += 1;
        }
        (i, j)
    }
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

/// Number of values in both sorted slices.
pub fn count_common(a: &[u32], b: &[u32]) -> u32 {
    let mut count = 0;
    for_each_common(a, b, |_| count += 1);
    count
}

/// Calls `f` for every value in both sorted slices.
#[inline]
pub fn for_each_common(a: &[u32], b: &[u32], mut f: impl FnMut(u32)) {
    let (mut x, mut y) = (0, 0);
    while x < a.len() && y < b.len() {
        match a[x].cmp(&b[y]) {
            std::cmp::Ordering::Less => x += 1,
            std::cmp::Ordering::Greater => y += 1,
            std::cmp::Ordering::Equal => {
                f(a[x]);
                x += 1;
                y += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn random_toggles_keep_every_structure_consistent() {
        for directed in [false, true] {
            let (n, mut rng) = (30, Rng::new(1));
            let mut net = Network::new(n, directed);
            let mut expected = HashSet::new();
            for _ in 0..20_000 {
                let (i, j) = if net.n_edges() > 0 && rng.unif() < 0.5 {
                    net.random_edge(&mut rng)
                } else {
                    net.random_dyad(&mut rng)
                };
                let key = if directed { (i, j) } else { (i.min(j), i.max(j)) };
                if !expected.remove(&key) {
                    expected.insert(key);
                }
                net.toggle(i, j);
            }
            assert_eq!(net.n_edges(), expected.len());
            let listed: HashSet<_> = net.edges().iter().copied().collect();
            assert_eq!(listed, expected);
            for i in 0..n {
                for j in (0..n).filter(|&j| j != i) {
                    let key = if directed { (i, j) } else { (i.min(j), i.max(j)) };
                    assert_eq!(net.has_edge(i, j), expected.contains(&key));
                }
                assert!(net.out[i as usize].windows(2).all(|w| w[0] < w[1]));
                let either: Vec<u32> = (0..n)
                    .filter(|&j| j != i && (net.has_edge(i, j) || net.has_edge(j, i)))
                    .collect();
                assert_eq!(net.neighbours(i), either.as_slice());
                let sources: Vec<u32> = (0..n).filter(|&j| j != i && net.has_edge(j, i)).collect();
                assert_eq!(net.in_neighbours(i), sources.as_slice());
            }
        }
    }

    #[test]
    fn kept_partner_counts_follow_the_toggles() {
        for directed in [false, true] {
            let (n, mut rng) = (12, Rng::new(3));
            let mut net = Network::new(n, directed);
            for _ in 0..40 {
                let (i, j) = net.random_dyad(&mut rng);
                net.toggle(i, j);
            }
            let kinds: &[Partners] =
                if directed { &[Partners::TwoPaths, Partners::OutShared, Partners::InShared] } else { &[Partners::Undirected] };
            for &kind in kinds {
                assert!(net.keep_partners_always(kind));
            }
            assert!(!net.keep_partners_always(if directed { Partners::Undirected } else { Partners::TwoPaths }));
            for step in 0..3000 {
                let (i, j) = if net.n_edges() > 0 && rng.unif() < 0.5 { net.random_edge(&mut rng) } else { net.random_dyad(&mut rng) };
                net.toggle(i, j);
                if step % 100 != 0 {
                    continue;
                }
                for a in 0..n {
                    for b in (0..n).filter(|&b| b != a) {
                        for &kind in kinds {
                            let direct = match kind {
                                Partners::Undirected => count_common(net.neighbours(a), net.neighbours(b)),
                                Partners::TwoPaths => count_common(net.out_neighbours(a), net.in_neighbours(b)),
                                Partners::OutShared => count_common(net.out_neighbours(a), net.out_neighbours(b)),
                                Partners::InShared => count_common(net.in_neighbours(a), net.in_neighbours(b)),
                            };
                            assert_eq!(net.kept_partners(kind, a, b), Some(direct), "{kind:?} ({a}, {b})");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn common_neighbours() {
        assert_eq!(count_common(&[1, 3, 5, 7], &[2, 3, 4, 7, 9]), 2);
        assert_eq!(count_common(&[], &[1, 2]), 0);
    }
}
