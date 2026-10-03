//! The sample space of a model, as set by ergm's constraints: which dyads may
//! change (`blocks`, `Dyads`, `fixedas`..., observed dyads when sampling
//! missing ones), bounds on degrees (`bd`, also by the alters' attributes),
//! and what is preserved (`edges`, `degrees`, `odegrees`, `idegrees`,
//! `b1degrees`, `b2degrees`, and the degree distributions: `degreedist`,
//! `odegreedist`, `idegreedist`).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::network::Network;
use crate::rng::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preserve {
    Nothing,
    /// Degrees (in- and out-degrees if directed).
    Degrees,
    OutDegrees,
    InDegrees,
    /// The number of edges.
    Edges,
    /// The degrees of the first (or second) mode of a bipartite network.
    FirstModeDegrees,
    SecondModeDegrees,
    /// The degree distribution (of in- and out-degrees, if directed).
    DegreeDist,
    OutDegreeDist,
    InDegreeDist,
}

/// Bounds on degrees; undirected networks use the `out` bounds for degrees.
pub struct Bounds {
    pub min_out: Vec<u32>,
    pub max_out: Vec<u32>,
    pub min_in: Vec<u32>,
    pub max_in: Vec<u32>,
    /// Bounds on the ties to alters of each class, as bd(attribs=).
    pub classes: Option<ClassBounds>,
}

/// bd(attribs=...): `attribs` (vertices x classes, row-major) marks each
/// vertex's classes, and the bounds (same shape) limit each vertex's ties to
/// alters of each class.
pub struct ClassBounds {
    pub k: usize,
    pub attribs: Vec<bool>,
    pub min_out: Vec<u32>,
    pub max_out: Vec<u32>,
    pub min_in: Vec<u32>,
    pub max_in: Vec<u32>,
}

impl ClassBounds {
    /// v's ties (out-ties, or in-ties) to alters of class c.
    fn count(&self, list: &[u32], c: usize) -> u32 {
        list.iter().filter(|&&u| self.attribs[u as usize * self.k + c]).count() as u32
    }

    fn allows(&self, net: &Network, i: u32, j: u32) -> bool {
        let adding = !net.has_edge(i, j);
        let ok = |count: u32, min: u32, max: u32| if adding { count < max } else { count > min };
        let check = |v: u32, other: u32, list: &[u32], min: &[u32], max: &[u32]| {
            (0..self.k).filter(|&c| self.attribs[other as usize * self.k + c]).all(|c| {
                let at = v as usize * self.k + c;
                ok(self.count(list, c), min[at], max[at])
            })
        };
        if net.directed() {
            check(i, j, net.out_neighbours(i), &self.min_out, &self.max_out)
                && check(j, i, net.in_neighbours(j), &self.min_in, &self.max_in)
        } else {
            check(i, j, net.neighbours(i), &self.min_out, &self.max_out)
                && check(j, i, net.neighbours(j), &self.min_out, &self.max_out)
        }
    }

    fn satisfied(&self, net: &Network, v: u32) -> bool {
        (0..self.k).all(|c| {
            let at = v as usize * self.k + c;
            let out = self.count(if net.directed() { net.out_neighbours(v) } else { net.neighbours(v) }, c);
            let within = |d: u32, min: u32, max: u32| d >= min && d <= max;
            within(out, self.min_out[at], self.max_out[at])
                && (!net.directed() || within(self.count(net.in_neighbours(v), c), self.min_in[at], self.max_in[at]))
        })
    }
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
        let plain = if net.directed() {
            ok(net.out_neighbours(i).len(), self.min_out[iu], self.max_out[iu])
                && ok(net.in_neighbours(j).len(), self.min_in[ju], self.max_in[ju])
        } else {
            ok(net.neighbours(i).len(), self.min_out[iu], self.max_out[iu])
                && ok(net.neighbours(j).len(), self.min_out[ju], self.max_out[ju])
        };
        plain && self.classes.as_ref().is_none_or(|c| c.allows(net, i, j))
    }

    /// Whether the degrees of `v` are within bounds.
    pub fn satisfied(&self, net: &Network, v: u32) -> bool {
        let u = v as usize;
        let out = Self::out_degree(net, v);
        let within = |d: usize, min: u32, max: u32| d >= min as usize && d <= max as usize;
        within(out, self.min_out[u], self.max_out[u])
            && (!net.directed() || within(net.in_neighbours(v).len(), self.min_in[u], self.max_in[u]))
            && self.classes.as_ref().is_none_or(|c| c.satisfied(net, v))
    }
}

/// Groups of vertices, the only dyads that may change being within a group
/// (the networks of a combined network, blockdiag()), and in bipartite
/// networks between the modes: each group's members, the first mode's first.
struct Groups {
    /// Each vertex's group (u32::MAX: in none).
    of: Vec<u32>,
    members: Vec<u32>,
    /// Group g's members are members[starts[g]..starts[g + 1]], its first
    /// mode's (bipartite) members[starts[g]..split[g]].
    starts: Vec<usize>,
    split: Vec<usize>,
    /// Cumulative numbers of the groups' dyads, to draw a dyad uniformly.
    cumulative: Vec<u64>,
    /// Bipartite: whether each vertex is of the first mode.
    first: Option<Vec<bool>>,
}

impl Groups {
    fn new(n: u32, directed: bool, of: Option<&[i64]>, first: Option<&[bool]>) -> Self {
        let of: Vec<u32> = match of {
            Some(g) => g.iter().map(|&x| if x < 0 { u32::MAX } else { x as u32 }).collect(),
            None => vec![0; n as usize],
        };
        let k = of.iter().filter(|&&g| g != u32::MAX).map(|&g| g as usize + 1).max().unwrap_or(0);
        let mut lists: Vec<(Vec<u32>, Vec<u32>)> = vec![(Vec::new(), Vec::new()); k];
        for (v, &g) in of.iter().enumerate() {
            if g != u32::MAX {
                let list = &mut lists[g as usize];
                if first.is_none_or(|f| f[v]) { list.0.push(v as u32) } else { list.1.push(v as u32) }
            }
        }
        let (mut members, mut starts, mut split, mut cumulative, mut total) = (Vec::new(), vec![0], Vec::new(), Vec::new(), 0u64);
        for (a, b) in lists {
            let dyads = if first.is_some() {
                a.len() as u64 * b.len() as u64
            } else {
                let s = a.len() as u64;
                s * s.saturating_sub(1) / if directed { 1 } else { 2 }
            };
            members.extend_from_slice(&a);
            split.push(members.len());
            members.extend_from_slice(&b);
            starts.push(members.len());
            total += dyads;
            cumulative.push(total);
        }
        Self { of, members, starts, split, cumulative, first: first.map(<[bool]>::to_vec) }
    }

    fn total(&self) -> u64 {
        self.cumulative.last().copied().unwrap_or(0)
    }

    #[inline]
    fn allows(&self, i: u32, j: u32) -> bool {
        let (g, h) = (self.of[i as usize], self.of[j as usize]);
        if g != h || g == u32::MAX {
            return false;
        }
        self.first.as_ref().is_none_or(|first| first[i as usize] != first[j as usize])
    }

    /// A uniformly random dyad within a group.
    fn draw(&self, rng: &mut Rng) -> (u32, u32) {
        let r = rng.below(self.total());
        let g = self.cumulative.partition_point(|&c| c <= r);
        let (start, split, stop) = (self.starts[g], self.split[g], self.starts[g + 1]);
        if self.first.is_some() {
            let a = self.members[start + rng.below((split - start) as u64) as usize];
            let b = self.members[split + rng.below((stop - split) as u64) as usize];
            return (a, b);
        }
        let s = (stop - start) as u64;
        let x = rng.below(s);
        let mut y = rng.below(s - 1);
        if y >= x {
            y += 1;
        }
        (self.members[start + x as usize], self.members[start + y as usize])
    }

    /// The vertices j that i may share a dyad with (j > i if undirected).
    fn partners(&self, i: u32, directed: bool, mut f: impl FnMut(u32)) {
        let g = self.of[i as usize];
        if g == u32::MAX {
            return;
        }
        let (start, split, stop) = (self.starts[g as usize], self.split[g as usize], self.starts[g as usize + 1]);
        let range = match &self.first {
            None => start..stop,
            Some(first) if first[i as usize] => split..stop,
            Some(_) => start..split,
        };
        for &j in &self.members[range] {
            if j != i && (directed || j > i) {
                f(j);
            }
        }
    }
}

/// What restricts the dyads of a sample space, besides degree bounds.
#[derive(Default)]
pub struct Restriction<'a> {
    /// Each vertex's group (negative: in none): dyads only within a group.
    pub groups: Option<&'a [i64]>,
    /// The first mode's vertices of a bipartite network: dyads only between modes.
    pub modes: Option<&'a [bool]>,
    /// An n x n row-major mask of fixed dyads.
    pub fixed_mask: Option<&'a [bool]>,
    /// Fixed dyads.
    pub fixed: &'a [(u32, u32)],
    /// If given, the only dyads that may change (with the restrictions above).
    pub only: Option<&'a [(u32, u32)]>,
}

pub struct Space {
    n: u32,
    directed: bool,
    /// Dyads only within groups (and between modes); None: any dyad.
    groups: Option<Groups>,
    /// One bit per dyad, set if the dyad is fixed.
    fixed_bits: Option<Vec<u64>>,
    fixed: FxHashSet<u64>,
    /// The free dyads, listed: those of `only`, or when they are a small share.
    list: Option<Vec<(u32, u32)>>,
    only: Option<FxHashSet<u64>>,
    n_free: u64,
    pub bounds: Option<Bounds>,
    pub preserve: Preserve,
    /// The vertices of the first mode, for FirstModeDegrees and SecondModeDegrees.
    pub first_mode: Option<Vec<bool>>,
}

/// Below this share of free dyads, random free dyads come from a list rather
/// than from rejection sampling.
const LIST_BELOW: f64 = 0.125;

impl Space {
    pub fn unconstrained(n: u32, directed: bool) -> Self {
        let dyads = n as u64 * (n as u64).saturating_sub(1) / if directed { 1 } else { 2 };
        Self {
            n,
            directed,
            groups: None,
            fixed_bits: None,
            fixed: FxHashSet::default(),
            list: None,
            only: None,
            n_free: dyads,
            bounds: None,
            preserve: Preserve::Nothing,
            first_mode: None,
        }
    }

    pub fn new(
        n: u32,
        directed: bool,
        restriction: Restriction,
        bounds: Option<Bounds>,
        preserve: Preserve,
    ) -> Result<Self, String> {
        let mut space = Self::unconstrained(n, directed);
        space.bounds = bounds;
        space.preserve = preserve;
        let size = n as usize * n as usize;
        if restriction.groups.is_some_and(|g| g.len() != n as usize) || restriction.modes.is_some_and(|m| m.len() != n as usize) {
            return Err("groups and modes need one value per vertex".into());
        }
        if restriction.groups.is_some() || restriction.modes.is_some() {
            let groups = Groups::new(n, directed, restriction.groups, restriction.modes);
            space.n_free = groups.total();
            space.groups = Some(groups);
        }
        let structural = space.n_free;
        if let Some(mask) = restriction.fixed_mask {
            if mask.len() != size {
                return Err(format!("the fixed dyad mask must have {size} values"));
            }
            let mut bits = vec![0u64; size.div_ceil(64)];
            for (index, _) in mask.iter().enumerate().filter(|&(_, &m)| m) {
                let (i, j) = ((index / n as usize) as u32, (index % n as usize) as u32);
                if i != j {
                    let k = space.index(i, j);
                    bits[k / 64] |= 1 << (k % 64);
                }
            }
            space.fixed_bits = Some(bits);
        }
        for &(i, j) in restriction.fixed {
            if i >= n || j >= n {
                return Err(format!("fixed dyad ({i}, {j}) refers to a vertex >= {n}"));
            }
            if i != j {
                space.fixed.insert(space.key(i, j));
            }
        }
        if let Some(only) = restriction.only {
            let mut seen = FxHashSet::default();
            let mut list = Vec::with_capacity(only.len());
            for &(i, j) in only {
                if i >= n || j >= n {
                    return Err(format!("dyad ({i}, {j}) refers to a vertex >= {n}"));
                }
                if i != j && space.allowed(i, j) && seen.insert(space.key(i, j)) {
                    list.push(space.canonical(i, j));
                }
            }
            space.n_free = list.len() as u64;
            space.only = Some(seen);
            space.list = Some(list);
            return Ok(space);
        }
        if space.fixed_bits.is_none() && space.fixed.is_empty() {
            return Ok(space);
        }
        // Count the fixed dyads among those the groups allow; list the free
        // ones if they are few.
        let mut fixed = 0u64;
        if let Some(bits) = &space.fixed_bits {
            for (word, &w) in bits.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let k = word * 64 + w.trailing_zeros() as usize;
                    w &= w - 1;
                    let (i, j) = ((k / n as usize) as u32, (k % n as usize) as u32);
                    if space.groups.as_ref().is_none_or(|g| g.allows(i, j)) {
                        fixed += 1;
                    }
                }
            }
        }
        for &key in &space.fixed {
            let (i, j) = ((key >> 32) as u32, key as u32);
            if space.groups.as_ref().is_none_or(|g| g.allows(i, j)) && !space.masked(i, j) {
                fixed += 1;
            }
        }
        space.n_free = structural - fixed;
        if (space.n_free as f64) < LIST_BELOW * structural as f64 {
            let mut list = Vec::with_capacity(space.n_free as usize);
            space.for_each_free(|i, j| list.push((i, j)));
            space.list = Some(list);
        }
        Ok(space)
    }

    pub fn restricted(&self) -> bool {
        self.groups.is_some() || self.fixed_bits.is_some() || !self.fixed.is_empty() || self.only.is_some()
    }

    pub fn n_free(&self) -> u64 {
        self.n_free
    }

    #[inline]
    fn canonical(&self, i: u32, j: u32) -> (u32, u32) {
        if !self.directed && i > j { (j, i) } else { (i, j) }
    }

    #[inline]
    fn key(&self, i: u32, j: u32) -> u64 {
        let (a, b) = self.canonical(i, j);
        ((a as u64) << 32) | b as u64
    }

    #[inline]
    fn index(&self, i: u32, j: u32) -> usize {
        let (a, b) = self.canonical(i, j);
        a as usize * self.n as usize + b as usize
    }

    #[inline]
    fn masked(&self, i: u32, j: u32) -> bool {
        self.fixed_bits.as_ref().is_some_and(|bits| {
            let k = self.index(i, j);
            bits[k / 64] >> (k % 64) & 1 == 1
        })
    }

    /// Whether the groups and the fixed dyads let (i, j) change.
    #[inline]
    fn allowed(&self, i: u32, j: u32) -> bool {
        self.groups.as_ref().is_none_or(|g| g.allows(i, j))
            && !self.masked(i, j)
            && (self.fixed.is_empty() || !self.fixed.contains(&self.key(i, j)))
    }

    #[inline]
    pub fn is_free(&self, i: u32, j: u32) -> bool {
        if i == j {
            return false;
        }
        if !self.restricted() {
            return true;
        }
        match &self.only {
            Some(only) => only.contains(&self.key(i, j)),
            None => self.allowed(i, j),
        }
    }

    /// Calls `f` for each free dyad (i < j if undirected).
    pub fn for_each_free(&self, mut f: impl FnMut(u32, u32)) {
        if let Some(list) = &self.list {
            list.iter().for_each(|&(i, j)| f(i, j));
            return;
        }
        for i in 0..self.n {
            self.for_each_free_of(i, |j| f(i, j));
        }
    }

    /// Calls `f` for each j such that (i, j) is free (j > i if undirected).
    /// With a list of the free dyads, use `for_each_free` instead.
    pub fn for_each_free_of(&self, i: u32, mut f: impl FnMut(u32)) {
        match &self.groups {
            Some(groups) => groups.partners(i, self.directed, |j| {
                if self.allowed(i, j) {
                    f(j)
                }
            }),
            None => {
                let first = if self.directed { 0 } else { i + 1 };
                for j in (first..self.n).filter(|&j| j != i && self.allowed(i, j)) {
                    f(j);
                }
            }
        }
    }

    /// The free dyads, if they are listed.
    pub fn list(&self) -> Option<&[(u32, u32)]> {
        self.list.as_deref()
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
            let (i, j) = match &self.groups {
                Some(groups) => groups.draw(rng),
                None => net.random_dyad(rng),
            };
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
