//! The sample space of a model, as set by ergm's constraints: which dyads may
//! change (`blocks`, observed dyads when sampling missing ones), bounds on
//! degrees (`bd`), and whether degrees are preserved (`degrees`, `odegrees`,
//! `idegrees`).

use rustc_hash::FxHashMap;

use crate::network::Network;
use crate::rng::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preserve {
    Nothing,
    /// Degrees (in- and out-degrees if directed).
    Degrees,
    OutDegrees,
    InDegrees,
}

/// Bounds on degrees; undirected networks use the `out` bounds for degrees.
pub struct Bounds {
    pub min_out: Vec<u32>,
    pub max_out: Vec<u32>,
    pub min_in: Vec<u32>,
    pub max_in: Vec<u32>,
}

impl Bounds {
    fn out_degree(net: &Network, v: u32) -> usize {
        if net.directed() { net.out_neighbours(v).len() } else { net.neighbours(v).len() }
    }

    /// Whether toggling (i, j) keeps the degrees of i and j within bounds.
    pub fn allows(&self, net: &Network, i: u32, j: u32) -> bool {
        let adding = !net.has_edge(i, j);
        let ok = |degree: usize, min: u32, max: u32| {
            if adding { degree < max as usize } else { degree > min as usize }
        };
        let (iu, ju) = (i as usize, j as usize);
        if net.directed() {
            ok(net.out_neighbours(i).len(), self.min_out[iu], self.max_out[iu])
                && ok(net.in_neighbours(j).len(), self.min_in[ju], self.max_in[ju])
        } else {
            ok(net.neighbours(i).len(), self.min_out[iu], self.max_out[iu])
                && ok(net.neighbours(j).len(), self.min_out[ju], self.max_out[ju])
        }
    }

    /// Whether the degrees of `v` are within bounds.
    pub fn satisfied(&self, net: &Network, v: u32) -> bool {
        let u = v as usize;
        let out = Self::out_degree(net, v);
        let within = |d: usize, min: u32, max: u32| d >= min as usize && d <= max as usize;
        within(out, self.min_out[u], self.max_out[u])
            && (!net.directed() || within(net.in_neighbours(v).len(), self.min_in[u], self.max_in[u]))
    }
}

pub struct Space {
    n: u32,
    directed: bool,
    /// One bit per dyad, set if the dyad may change; None if every dyad may.
    free: Option<Vec<u64>>,
    /// The free dyads, listed when they are a small share of all dyads.
    list: Option<Vec<(u32, u32)>>,
    n_free: u64,
    pub bounds: Option<Bounds>,
    pub preserve: Preserve,
}

/// Below this share of free dyads, random free dyads come from a list rather
/// than from rejection sampling.
const LIST_BELOW: f64 = 0.125;

impl Space {
    pub fn unconstrained(n: u32, directed: bool) -> Self {
        let dyads = n as u64 * (n as u64 - 1) / if directed { 1 } else { 2 };
        Self { n, directed, free: None, list: None, n_free: dyads, bounds: None, preserve: Preserve::Nothing }
    }

    /// `free` is an n x n row-major mask of the dyads that may change (the
    /// upper triangle, i < j, if undirected).
    pub fn new(
        n: u32,
        directed: bool,
        free: Option<&[bool]>,
        bounds: Option<Bounds>,
        preserve: Preserve,
    ) -> Result<Self, String> {
        let mut space = Self::unconstrained(n, directed);
        space.bounds = bounds;
        space.preserve = preserve;
        if let Some(mask) = free {
            let size = n as usize * n as usize;
            if mask.len() != size {
                return Err(format!("the free dyad mask must have {size} values"));
            }
            let mut bits = vec![0u64; size.div_ceil(64)];
            let mut count = 0u64;
            for i in 0..n {
                for j in 0..n {
                    let index = (i * n + j) as usize;
                    if i != j && (directed || i < j) && mask[index] {
                        bits[index / 64] |= 1 << (index % 64);
                        count += 1;
                    }
                }
            }
            if (count as f64) < LIST_BELOW * space.n_free as f64 {
                let mut list = Vec::with_capacity(count as usize);
                for i in 0..n {
                    for j in 0..n {
                        let index = (i * n + j) as usize;
                        if bits[index / 64] >> (index % 64) & 1 == 1 {
                            list.push((i, j));
                        }
                    }
                }
                space.list = Some(list);
            }
            space.free = Some(bits);
            space.n_free = count;
        }
        Ok(space)
    }

    pub fn restricted(&self) -> bool {
        self.free.is_some()
    }

    pub fn n_free(&self) -> u64 {
        self.n_free
    }

    #[inline]
    fn index(&self, i: u32, j: u32) -> usize {
        let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
        (a * self.n + b) as usize
    }

    #[inline]
    pub fn is_free(&self, i: u32, j: u32) -> bool {
        match &self.free {
            None => true,
            Some(bits) => {
                let k = self.index(i, j);
                bits[k / 64] >> (k % 64) & 1 == 1
            }
        }
    }

    /// A uniformly random free dyad, or None if there is none.
    pub fn random_dyad(&self, net: &Network, rng: &mut Rng) -> Option<(u32, u32)> {
        if self.n_free == 0 {
            return None;
        }
        if let Some(list) = &self.list {
            return Some(list[rng.below(list.len() as u64) as usize]);
        }
        loop {
            let (i, j) = net.random_dyad(rng);
            if self.is_free(i, j) {
                return Some((i, j));
            }
        }
    }
}

/// Ties, with the position of each in the list.
type IndexedTies = (Vec<(u32, u32)>, FxHashMap<u64, usize>);

/// The ties a chain can toggle off: every tie, or those on free dyads, with
/// O(1) random draws.
pub struct EdgeIndex {
    /// Ties on free dyads, if some dyads are fixed.
    free: Option<IndexedTies>,
}

fn key(i: u32, j: u32) -> u64 {
    ((i as u64) << 32) | j as u64
}

impl EdgeIndex {
    pub fn new(space: &Space, net: &Network) -> Self {
        if !space.restricted() {
            return Self { free: None };
        }
        let edges: Vec<(u32, u32)> = net.edges().iter().copied().filter(|&(i, j)| space.is_free(i, j)).collect();
        let position = edges.iter().enumerate().map(|(p, &(i, j))| (key(i, j), p)).collect();
        Self { free: Some((edges, position)) }
    }

    pub fn count(&self, net: &Network) -> usize {
        match &self.free {
            None => net.n_edges(),
            Some((edges, _)) => edges.len(),
        }
    }

    pub fn random(&self, net: &Network, rng: &mut Rng) -> (u32, u32) {
        match &self.free {
            None => net.random_edge(rng),
            Some((edges, _)) => edges[rng.below(edges.len() as u64) as usize],
        }
    }

    /// Records the toggle of the free dyad (i, j), which was a tie if `removed`.
    pub fn toggle(&mut self, directed: bool, i: u32, j: u32, removed: bool) {
        let Some((edges, position)) = &mut self.free else { return };
        let (a, b) = if !directed && i > j { (j, i) } else { (i, j) };
        if removed {
            if let Some(p) = position.remove(&key(a, b)) {
                edges.swap_remove(p);
                if p < edges.len() {
                    let (x, y) = edges[p];
                    position.insert(key(x, y), p);
                }
            }
        } else {
            position.insert(key(a, b), edges.len());
            edges.push((a, b));
        }
    }
}
