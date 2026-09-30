//! A binary network without self-loops, stored as sorted neighbour lists.
//!
//! An edge list with a position index gives O(1) random edges, which the TNT
//! proposal needs. Undirected edges are stored once, as (min, max).

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
    edges: Vec<(u32, u32)>,
    position: FxHashMap<u64, usize>,
}

impl Network {
    pub fn new(n: u32, directed: bool) -> Self {
        let lists = || vec![Vec::new(); n as usize];
        Self {
            n,
            directed,
            out: lists(),
            inn: if directed { lists() } else { Vec::new() },
            edges: Vec::new(),
            position: FxHashMap::default(),
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

    pub fn n_dyads(&self) -> u64 {
        let n = self.n as u64;
        if self.directed { n * (n - 1) } else { n * (n - 1) / 2 }
    }

    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    /// Neighbours of `i` in an undirected network, or its out-neighbours.
    pub fn neighbours(&self, i: u32) -> &[u32] {
        &self.out[i as usize]
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
            }
        }
    }

    #[test]
    fn common_neighbours() {
        assert_eq!(count_common(&[1, 3, 5, 7], &[2, 3, 4, 7, 9]), 2);
        assert_eq!(count_common(&[], &[1, 2]), 0);
    }
}
