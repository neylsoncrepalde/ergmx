//! MPNet's configurations of directed two-level networks (the MPNet manual,
//! Wang et al.'s program), and two undirected ones: EXTA/EXTB and ASAXASB.
//!
//! The vertices are at level A, level B or neither. In a directed network,
//! A-ties and B-ties are arcs within a level, and X-ties (affiliations) are
//! the arcs from an A vertex to a B vertex; arcs from B to A are not X-ties,
//! and no configuration counts them. For a vertex v at level S, in(v) and
//! out(v) are its in- and out-degrees within S, and x(v) its number of
//! X-ties; g(d) = exp(decay) (1 - r^d), r = 1 - exp(-decay), weights the
//! alternating statistics, as gwesp's (MPNet's lambda is exp(decay)).
//!
//! Each change statistic is computed in the network without the toggled tie
//! (`Without`), as the change from adding it; removing it reverses that.

use crate::network::Network;
use crate::partners::Bins;
use crate::terms::Term;

/// A network seen without one of its ties (in both orientations, if undirected).
struct Without<'a> {
    net: &'a Network,
    skip: Option<(u32, u32)>,
    directed: bool,
}

impl Without<'_> {
    #[inline]
    fn skipped(&self, a: u32, b: u32) -> bool {
        match self.skip {
            None => false,
            Some((x, y)) => (a, b) == (x, y) || (!self.directed && (a, b) == (y, x)),
        }
    }

    #[inline]
    fn has(&self, a: u32, b: u32) -> bool {
        !self.skipped(a, b) && self.net.has_edge(a, b)
    }

    fn out(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        self.net.out_neighbours(v).iter().copied().filter(move |&w| !self.skipped(v, w))
    }

    fn inn(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        self.net.in_neighbours(v).iter().copied().filter(move |&w| !self.skipped(w, v))
    }

    /// Neighbours in either direction, once each.
    fn both(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        let net = self.net;
        net.neighbours(v).iter().copied().filter(move |&w| self.has(v, w) || self.has(w, v))
    }
}

/// The directed configurations, for the term's side S (A or B; O is the other level).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Directed {
    /// In2StarSX, Out2StarSX: sum over v in S of in(v) x(v), out(v) x(v).
    StarIn,
    StarOut,
    /// SXS1Sin, SXS1Sout (AXS1Ain...): in(v) g(x(v)), out(v) g(x(v)).
    XStarsIn,
    XStarsOut,
    /// SSinS1X, SSoutS1X (AAinS1X...): g(in(v)) x(v), g(out(v)) x(v).
    InStarsX,
    OutStarsX,
    /// TXSXarc, TXSXreciprocity: over S-arcs (mutual pairs), their shared O partners.
    TriangleArc,
    TriangleMutual,
    /// ATXSXarc, ATXSXreciprocity: g(shared O partners).
    AltTriangleArc,
    AltTriangleMutual,
    /// L3XSX, L3XSXreciprocity: over S-arcs (mutual pairs) u, v, x(u) x(v).
    PathArc,
    PathMutual,
    /// L3AXBin, L3AXBout, L3AXBpath, L3BXApath: over X-ties a -> b, in_A(a)
    /// in_B(b), out_A(a) out_B(b), in_A(a) out_B(b), out_A(a) in_B(b).
    PathIn,
    PathOut,
    PathAB,
    PathBA,
    /// C4AXB with directed A and B ties: entrainment (A-arc u -> v, B-arc w
    /// -> z, X-ties u -> w and v -> z), exchange (X-ties u -> z and v -> w),
    /// A reciprocated with a B-arc, an A-arc with B reciprocated, and both reciprocated.
    CycleEntrainment,
    CycleExchange,
    CycleMutualA,
    CycleMutualB,
    CycleMutual,
    /// AinASXAinBS and its three variants: over X-ties a -> b, g(in or
    /// out_A(a)) g(in or out_B(b)).
    StarsStars { a_in: bool, b_in: bool },
}

/// An undirected configuration not in `crate::terms::Level`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Undirected {
    /// EXTA, EXTB: sum over v in S of its triangles within S times x(v).
    TriangleX,
    /// ASAXASB: over X-ties (a, b), g(A-degree of a) g(B-degree of b).
    StarsStars,
}

pub struct MultilevelExtra {
    pub directed: Option<Directed>,
    pub undirected: Option<Undirected>,
    /// 0 for level A, 1 for B, -1 for neither.
    pub level: Vec<i8>,
    /// The term's side: 0 (A) or 1 (B).
    pub side: i8,
    pub r: f64,
    pub exp_decay: f64,
}

impl MultilevelExtra {
    #[inline]
    fn at(&self, v: u32) -> i8 {
        self.level[v as usize]
    }

    fn g(&self, d: usize) -> f64 {
        self.exp_decay * (1.0 - self.r.powi(d as i32))
    }

    fn step(&self, d: usize) -> f64 {
        // g(d + 1) - g(d)
        self.r.powi(d as i32)
    }

    // -- Directed helpers, in the network without the toggled tie ---------------------------------

    /// X-ties of v: its out-arcs to B if v is at A, its in-arcs from A if at B.
    fn x_ties<'a>(&'a self, w: &'a Without, v: u32) -> Box<dyn Iterator<Item = u32> + 'a> {
        match self.at(v) {
            0 => Box::new(w.out(v).filter(move |&u| self.at(u) == 1)),
            1 => Box::new(w.inn(v).filter(move |&u| self.at(u) == 0)),
            _ => Box::new(std::iter::empty()),
        }
    }

    fn x(&self, w: &Without, v: u32) -> usize {
        self.x_ties(w, v).count()
    }

    fn in_level(&self, w: &Without, v: u32) -> usize {
        let l = self.at(v);
        w.inn(v).filter(|&u| self.at(u) == l).count()
    }

    fn out_level(&self, w: &Without, v: u32) -> usize {
        let l = self.at(v);
        w.out(v).filter(|&u| self.at(u) == l).count()
    }

    /// v's partners at its own level with ties both ways.
    fn mutual_partners<'a>(&'a self, w: &'a Without, v: u32) -> impl Iterator<Item = u32> + 'a {
        let l = self.at(v);
        w.out(v).filter(move |&u| self.at(u) == l && w.has(u, v))
    }

    /// Whether a -> b is an X-tie (a at A, b at B).
    #[inline]
    fn xtie(&self, w: &Without, a: u32, b: u32) -> bool {
        self.at(a) == 0 && self.at(b) == 1 && w.has(a, b)
    }

    /// Shared partners at the other level of two vertices of the side's level.
    fn shared(&self, w: &Without, u: u32, v: u32) -> usize {
        self.x_ties(w, u).filter(|&o| self.x_ties(w, v).any(|p| p == o)).count()
    }

    fn sum_x_over(&self, w: &Without, items: impl Iterator<Item = u32>) -> usize {
        items.map(|u| self.x(w, u)).sum()
    }

    /// The change from adding the arc i -> j to the network without it.
    fn added_directed(&self, kind: Directed, w: &Without, i: u32, j: u32) -> f64 {
        let (li, lj) = (self.at(i), self.at(j));
        if li < 0 || lj < 0 || (li == 1 && lj == 0) {
            return 0.0; // not at the levels, or an arc from B to A
        }
        let s = self.side;
        let within = li == lj; // an S- or O-arc; otherwise an X-tie i -> j
        use Directed::*;
        match kind {
            StarIn | StarOut | XStarsIn | XStarsOut | InStarsX | OutStarsX => {
                if within {
                    if li != s {
                        return 0.0;
                    }
                    // The head gains an in-tie, the tail an out-tie.
                    let incoming = matches!(kind, StarIn | XStarsIn | InStarsX);
                    let v = if incoming { j } else { i };
                    match kind {
                        StarIn | StarOut => self.x(w, v) as f64,
                        XStarsIn | XStarsOut => self.g(self.x(w, v)),
                        InStarsX => self.step(self.in_level(w, v)) * self.x(w, v) as f64,
                        _ => self.step(self.out_level(w, v)) * self.x(w, v) as f64,
                    }
                } else {
                    // The side's end gains an X-tie.
                    let v = if s == 0 { i } else { j };
                    match kind {
                        StarIn => self.in_level(w, v) as f64,
                        StarOut => self.out_level(w, v) as f64,
                        XStarsIn => self.in_level(w, v) as f64 * self.step(self.x(w, v)),
                        XStarsOut => self.out_level(w, v) as f64 * self.step(self.x(w, v)),
                        InStarsX => self.g(self.in_level(w, v)),
                        _ => self.g(self.out_level(w, v)),
                    }
                }
            }
            TriangleArc | TriangleMutual | AltTriangleArc | AltTriangleMutual => {
                let mutual = matches!(kind, TriangleMutual | AltTriangleMutual);
                let alternating = matches!(kind, AltTriangleArc | AltTriangleMutual);
                let weight = |shared: usize| if alternating { self.g(shared) } else { shared as f64 };
                let step = |shared: usize| if alternating { self.step(shared) } else { 1.0 };
                if within {
                    if li != s || (mutual && !w.has(j, i)) {
                        return 0.0;
                    }
                    return weight(self.shared(w, i, j));
                }
                // An X-tie: the side's end v gains the partner p; each S-arc (or
                // mutual pair) of v with another vertex u tied to p gains a shared partner.
                let (v, p) = if s == 0 { (i, j) } else { (j, i) };
                let mut total = 0.0;
                for u in self.x_ties(w, p).collect::<Vec<_>>() {
                    if u == v {
                        continue;
                    }
                    let arcs = if mutual {
                        (w.has(v, u) && w.has(u, v)) as usize
                    } else {
                        w.has(v, u) as usize + w.has(u, v) as usize
                    };
                    if arcs > 0 {
                        total += arcs as f64 * step(self.shared(w, v, u));
                    }
                }
                total
            }
            PathArc | PathMutual => {
                let mutual = kind == PathMutual;
                if within {
                    if li != s || (mutual && !w.has(j, i)) {
                        return 0.0;
                    }
                    return (self.x(w, i) * self.x(w, j)) as f64;
                }
                let v = if s == 0 { i } else { j };
                if mutual {
                    self.sum_x_over(w, self.mutual_partners(w, v).collect::<Vec<_>>().into_iter()) as f64
                } else {
                    let l = self.at(v);
                    let outs: Vec<u32> = w.out(v).filter(|&u| self.at(u) == l).collect();
                    let ins: Vec<u32> = w.inn(v).filter(|&u| self.at(u) == l).collect();
                    (self.sum_x_over(w, outs.into_iter()) + self.sum_x_over(w, ins.into_iter())) as f64
                }
            }
            PathIn | PathOut | PathAB | PathBA => {
                // The degree each end contributes: (A end's, B end's), true = in.
                let (a_in, b_in) = match kind {
                    PathIn => (true, true),
                    PathOut => (false, false),
                    PathAB => (true, false),
                    _ => (false, true),
                };
                let degree = |v: u32, incoming: bool| if incoming { self.in_level(w, v) } else { self.out_level(w, v) };
                if !within {
                    return (degree(i, a_in) * degree(j, b_in)) as f64;
                }
                // A within-level arc i -> j raises in(j) and out(i): the X-ties of
                // the vertex whose counted degree rises gain the other end's degree.
                let counted_in = if li == 0 { a_in } else { b_in };
                let v = if counted_in { j } else { i };
                let other_in = if li == 0 { b_in } else { a_in };
                self.x_ties(w, v).collect::<Vec<_>>().into_iter().map(|u| degree(u, other_in)).sum::<usize>() as f64
            }
            CycleEntrainment | CycleExchange | CycleMutualA | CycleMutualB | CycleMutual => {
                self.cycle(kind, w, i, j, within, li)
            }
            StarsStars { a_in, b_in } => {
                let degree = |v: u32, incoming: bool| if incoming { self.in_level(w, v) } else { self.out_level(w, v) };
                if !within {
                    return self.g(degree(i, a_in)) * self.g(degree(j, b_in));
                }
                let counted_in = if li == 0 { a_in } else { b_in };
                let v = if counted_in { j } else { i };
                let other_in = if li == 0 { b_in } else { a_in };
                let k = degree(v, counted_in);
                self.step(k) * self.x_ties(w, v).collect::<Vec<_>>().into_iter().map(|u| self.g(degree(u, other_in))).sum::<f64>()
            }
        }
    }

    /// The four-cycles of an A-tie, a B-tie and two X-ties, by direction.
    fn cycle(&self, kind: Directed, w: &Without, i: u32, j: u32, within: bool, level: i8) -> f64 {
        use Directed::*;
        let collect = |it: Box<dyn Iterator<Item = u32> + '_>| it.collect::<Vec<u32>>();
        let mutual = |a: u32, b: u32| w.has(a, b) && w.has(b, a);
        if !within {
            // An X-tie a -> b, in either role of the cycle.
            let (a, b) = (i, j);
            let l_a: Vec<u32> = w.out(a).filter(|&u| self.at(u) == 0).collect(); // a -> v
            let l_ai: Vec<u32> = w.inn(a).filter(|&u| self.at(u) == 0).collect(); // u -> a
            let l_b: Vec<u32> = w.out(b).filter(|&u| self.at(u) == 1).collect(); // b -> z
            let l_bi: Vec<u32> = w.inn(b).filter(|&u| self.at(u) == 1).collect(); // w -> b
            let m_a: Vec<u32> = self.mutual_partners(w, a).collect();
            let m_b: Vec<u32> = self.mutual_partners(w, b).collect();
            let count = |xs: &[u32], zs: &[u32]| {
                xs.iter().map(|&v| zs.iter().filter(|&&z| self.xtie(w, v, z)).count()).sum::<usize>() as f64
            };
            return match kind {
                // As the tie of the sources (u = a, w = b), then of the targets.
                CycleEntrainment => count(&l_a, &l_b) + count(&l_ai, &l_bi),
                // As u -> z (u = a, z = b), then as v -> w (v = a, w = b).
                CycleExchange => count(&l_a, &l_bi) + count(&l_ai, &l_b),
                CycleMutualA => count(&m_a, &l_b) + count(&m_a, &l_bi),
                CycleMutualB => count(&l_a, &m_b) + count(&l_ai, &m_b),
                _ => count(&m_a, &m_b),
            };
        }
        // An A-arc (level 0) or a B-arc (level 1) i -> j.
        let reciprocated = w.has(j, i);
        let at_a = level == 0;
        let needs_mutual = match kind {
            CycleMutualA => at_a,
            CycleMutualB => !at_a,
            CycleMutual => true,
            _ => false,
        };
        if needs_mutual && !reciprocated {
            return 0.0;
        }
        // The X-partners of each end, at the other level.
        let (xi, xj) = (collect(self.x_ties(w, i)), collect(self.x_ties(w, j)));
        // Ties of the other level between partners p of i and q of j: arcs
        // p -> q (`forward`), q -> p, or mutual pairs.
        let pairs = |want: &dyn Fn(u32, u32) -> bool| {
            xi.iter().map(|&p| xj.iter().filter(|&&q| p != q && want(p, q)).count()).sum::<usize>() as f64
        };
        let other_mutual = match kind {
            CycleMutualA => !at_a,
            CycleMutualB => at_a,
            CycleMutual => true,
            _ => false,
        };
        if other_mutual {
            return pairs(&|p, q| mutual(p, q));
        }
        match kind {
            // The A-arc u -> v with partners w of u and z of v: entrainment needs w -> z.
            CycleEntrainment => pairs(&|p, q| w.has(p, q)),
            // Exchange: B-arc w -> z with w tied to v (the head) and z to u (the tail).
            CycleExchange => pairs(&|p, q| w.has(q, p)),
            // One side reciprocated: the other side's arc in either direction
            // (each cycle once: its partners are assigned by the ends).
            _ => pairs(&|p, q| w.has(p, q)) + pairs(&|p, q| w.has(q, p)),
        }
    }

    // -- Undirected extras ------------------------------------------------------------------------

    fn degree_at(&self, w: &Without, v: u32, l: i8) -> usize {
        w.both(v).filter(|&u| self.at(u) == l).count()
    }

    fn triangles(&self, w: &Without, v: u32) -> usize {
        let l = self.at(v);
        let mine: Vec<u32> = w.both(v).filter(|&u| self.at(u) == l).collect();
        let mut t = 0;
        for (k, &a) in mine.iter().enumerate() {
            for &b in &mine[k + 1..] {
                if w.has(a, b) || w.has(b, a) {
                    t += 1;
                }
            }
        }
        t
    }

    fn added_undirected(&self, kind: Undirected, w: &Without, i: u32, j: u32) -> f64 {
        let (li, lj) = (self.at(i), self.at(j));
        if li < 0 || lj < 0 {
            return 0.0;
        }
        let s = self.side;
        match kind {
            Undirected::TriangleX => {
                if li == lj {
                    if li != s {
                        return 0.0;
                    }
                    // The common neighbours k at S each make a new triangle i, j, k.
                    let ni: Vec<u32> = w.both(i).filter(|&u| self.at(u) == s).collect();
                    let common: Vec<u32> = w.both(j).filter(|&u| self.at(u) == s && ni.contains(&u)).collect();
                    let c = common.len() as f64;
                    let o = 1 - s;
                    let x = |v: u32| self.degree_at(w, v, o) as f64;
                    c * (x(i) + x(j)) + common.iter().map(|&k| x(k)).sum::<f64>()
                } else {
                    let v = if li == s { i } else { j };
                    self.triangles(w, v) as f64
                }
            }
            Undirected::StarsStars => {
                if li != lj {
                    let (a, b) = if li == 0 { (i, j) } else { (j, i) };
                    return self.g(self.degree_at(w, a, 0)) * self.g(self.degree_at(w, b, 1));
                }
                let o = 1 - li;
                let mut total = 0.0;
                for v in [i, j] {
                    let k = self.degree_at(w, v, li);
                    let partners: f64 = w.both(v).filter(|&u| self.at(u) == o).map(|u| self.g(self.degree_at(w, u, o))).sum();
                    total += self.step(k) * partners;
                }
                total
            }
        }
    }
}

impl Term for MultilevelExtra {
    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let skip = (sign < 0.0).then_some((i, j));
        let w = Without { net, skip, directed: net.directed() };
        let added = match (self.directed, self.undirected) {
            (Some(kind), _) => self.added_directed(kind, &w, i, j),
            (_, Some(kind)) => self.added_undirected(kind, &w, i, j),
            _ => 0.0,
        };
        out[0] += sign * added;
    }
}

/// The histograms of the alternating configurations with an estimated
/// decay (curved, as ergm's gwesp with fixed=FALSE): the statistic is
/// sum_k g(k) H_k, and H_k counts
///
/// * X-stars with one S-tie (AXS1A, AXS1Ain, AXS1Aout): the S-ties (in, out)
///   of the vertices with k X-ties;
/// * S-stars with one X-tie (AAS1X, AAinS1X, AAoutS1X): the X-ties of the
///   vertices with k S-ties (in, out);
/// * alternating triangles (ATXAX, ATXAXarc, ATXAXreciprocity): the S-ties
///   (arcs, reciprocated pairs) with k shared partners at the other level.
pub struct MultilevelHistogram {
    kind: Histogram,
    /// The levels and side, with the helpers that count ties.
    base: MultilevelExtra,
    bins: Bins,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Histogram {
    XStars(Degree),
    StarsX(Degree),
    Triangles(Pairs),
}

/// Which within-level degree: all ties (undirected), in- or out-ties.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Degree {
    All,
    In,
    Out,
}

/// Which within-level pairs: ties (undirected), arcs or reciprocated pairs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pairs {
    Ties,
    Arcs,
    Mutual,
}

impl MultilevelHistogram {
    /// The vertex's X-ties (undirected: its neighbours at the other level).
    fn x(&self, h: &MultilevelExtra, w: &Without, v: u32) -> usize {
        if w.directed { h.x(w, v) } else { h.degree_at(w, v, 1 - h.at(v)) }
    }

    fn within(&self, h: &MultilevelExtra, w: &Without, v: u32, d: Degree) -> usize {
        match d {
            Degree::All => h.degree_at(w, v, h.at(v)),
            Degree::In => h.in_level(w, v),
            Degree::Out => h.out_level(w, v),
        }
    }

    /// The other-level partners of v (X-ties).
    fn partners(&self, h: &MultilevelExtra, w: &Without, v: u32) -> Vec<u32> {
        if w.directed {
            h.x_ties(w, v).collect()
        } else {
            let o = 1 - h.at(v);
            w.both(v).filter(|&u| h.at(u) == o).collect()
        }
    }

    fn shared(&self, h: &MultilevelExtra, w: &Without, u: u32, v: u32) -> usize {
        let pu = self.partners(h, w, u);
        self.partners(h, w, v).iter().filter(|p| pu.contains(p)).count()
    }

    fn add(&self, k: usize, by: f64, out: &mut [f64]) {
        if let Some(s) = self.bins.index(k as u32) {
            out[s] += by;
        }
    }
}

impl Term for MultilevelHistogram {
    fn n_stats(&self) -> usize {
        self.bins.len()
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let h = &self.base;
        let w = Without { net, skip: (sign < 0.0).then_some((i, j)), directed: net.directed() };
        let (li, lj) = (h.at(i), h.at(j));
        if li < 0 || lj < 0 || (w.directed && li == 1 && lj == 0) {
            return;
        }
        let s = h.side;
        // The change from adding the tie, times sign.
        if li == lj {
            if li != s {
                return;
            }
            match self.kind {
                Histogram::XStars(d) => {
                    // The ends whose counted degree rises add one at their X-degree.
                    for v in [i, j] {
                        let counted = match d {
                            Degree::All => true,
                            Degree::In => v == j,
                            Degree::Out => v == i,
                        };
                        if counted {
                            self.add(self.x(h, &w, v), sign, out);
                        }
                    }
                }
                Histogram::StarsX(d) => {
                    for v in [i, j] {
                        let counted = match d {
                            Degree::All => true,
                            Degree::In => v == j,
                            Degree::Out => v == i,
                        };
                        if counted {
                            let k = self.within(h, &w, v, d);
                            let x = self.x(h, &w, v) as f64;
                            self.bins.shift(k as u32, sign * x, out);
                        }
                    }
                }
                Histogram::Triangles(pairs) => {
                    if pairs == Pairs::Mutual && !w.has(j, i) {
                        return;
                    }
                    self.add(self.shared(h, &w, i, j), sign, out);
                }
            }
            return;
        }
        // An X-tie: the side's end v gains the partner p.
        let (v, p) = if li == s { (i, j) } else { (j, i) };
        match self.kind {
            Histogram::XStars(d) => {
                let k = self.x(h, &w, v);
                let weight = self.within(h, &w, v, d) as f64;
                self.bins.shift(k as u32, sign * weight, out);
            }
            Histogram::StarsX(d) => self.add(self.within(h, &w, v, d), sign, out),
            Histogram::Triangles(pairs) => {
                for u in self.partners(h, &w, p) {
                    if u == v {
                        continue;
                    }
                    let ties = match pairs {
                        Pairs::Ties => (w.has(v, u) || w.has(u, v)) as usize,
                        Pairs::Arcs => w.has(v, u) as usize + w.has(u, v) as usize,
                        Pairs::Mutual => (w.has(v, u) && w.has(u, v)) as usize,
                    };
                    if ties > 0 {
                        self.bins.shift(self.shared(h, &w, v, u) as u32, sign * ties as f64, out);
                    }
                }
            }
        }
    }
}

/// ints: chunks [kind, side], [each vertex's level], [counts], [overflow].
fn build_histogram(n: usize, directed: bool, ints: &[i64]) -> Result<Box<dyn Term>, String> {
    let mut parts = Vec::new();
    let mut rest = ints;
    while let Some((&len, tail)) = rest.split_first() {
        let len = len as usize;
        if tail.len() < len {
            return Err("multilevel histogram: bad parameters".into());
        }
        parts.push(tail[..len].to_vec());
        rest = &tail[len..];
    }
    let [head, levels, ks, overflow] = parts.as_slice() else {
        return Err("multilevel histogram: bad parameters".into());
    };
    let (&[kind, side], true) = (head.as_slice().try_into().unwrap_or(&[-1, -1]), levels.len() == n) else {
        return Err("multilevel histogram: bad parameters".into());
    };
    let kind = match (kind, directed) {
        (0, false) => Histogram::XStars(Degree::All),
        (1, true) => Histogram::XStars(Degree::In),
        (2, true) => Histogram::XStars(Degree::Out),
        (3, false) => Histogram::StarsX(Degree::All),
        (4, true) => Histogram::StarsX(Degree::In),
        (5, true) => Histogram::StarsX(Degree::Out),
        (6, false) => Histogram::Triangles(Pairs::Ties),
        (7, true) => Histogram::Triangles(Pairs::Arcs),
        (8, true) => Histogram::Triangles(Pairs::Mutual),
        _ => return Err("multilevel histogram: not for this kind of network".into()),
    };
    let ks: Vec<u32> = ks.iter().map(|&k| k as u32).collect();
    let level = levels.iter().map(|&l| l as i8).collect();
    let base = MultilevelExtra { directed: None, undirected: None, level, side: side as i8, r: 0.0, exp_decay: 0.0 };
    Ok(Box::new(MultilevelHistogram { kind, base, bins: Bins::new(&ks, overflow.first() == Some(&1)) }))
}

/// The terms of this module by name, or None. ints: each vertex's level (0
/// A, 1 B, -1 neither); reals: [decay] for the alternating ones.
pub fn build(n: usize, directed: bool, name: &str, reals: &[f64], ints: &[i64]) -> Option<Result<Box<dyn Term>, String>> {
    use Directed::*;
    if name == "multilevelhistogram" {
        return Some(build_histogram(n, directed, ints));
    }
    let (kind, side, alternating) = match name {
        "in2starax" => (Ok(StarIn), 0, false),
        "in2starbx" => (Ok(StarIn), 1, false),
        "out2starax" => (Ok(StarOut), 0, false),
        "out2starbx" => (Ok(StarOut), 1, false),
        "axs1ain" => (Ok(XStarsIn), 0, true),
        "axs1bin" => (Ok(XStarsIn), 1, true),
        "axs1aout" => (Ok(XStarsOut), 0, true),
        "axs1bout" => (Ok(XStarsOut), 1, true),
        "aains1x" => (Ok(InStarsX), 0, true),
        "abins1x" => (Ok(InStarsX), 1, true),
        "aaouts1x" => (Ok(OutStarsX), 0, true),
        "abouts1x" => (Ok(OutStarsX), 1, true),
        "txaxarc" => (Ok(TriangleArc), 0, false),
        "txbxarc" => (Ok(TriangleArc), 1, false),
        "txaxreciprocity" => (Ok(TriangleMutual), 0, false),
        "txbxreciprocity" => (Ok(TriangleMutual), 1, false),
        "atxaxarc" => (Ok(AltTriangleArc), 0, true),
        "atxbxarc" => (Ok(AltTriangleArc), 1, true),
        "atxaxreciprocity" => (Ok(AltTriangleMutual), 0, true),
        "atxbxreciprocity" => (Ok(AltTriangleMutual), 1, true),
        "l3xaxarc" => (Ok(PathArc), 0, false),
        "l3xbxarc" => (Ok(PathArc), 1, false),
        "l3xaxreciprocity" => (Ok(PathMutual), 0, false),
        "l3xbxreciprocity" => (Ok(PathMutual), 1, false),
        "l3axbin" => (Ok(PathIn), 0, false),
        "l3axbout" => (Ok(PathOut), 0, false),
        "l3axbpath" => (Ok(PathAB), 0, false),
        "l3bxapath" => (Ok(PathBA), 0, false),
        "c4axbentrainment" => (Ok(CycleEntrainment), 0, false),
        "c4axbexchange" => (Ok(CycleExchange), 0, false),
        "c4axbexchangeareciprocity" => (Ok(CycleMutualA), 0, false),
        "c4axbexchangebreciprocity" => (Ok(CycleMutualB), 0, false),
        "c4axbreciprocity" => (Ok(CycleMutual), 0, false),
        "ainasxainbs" => (Ok(StarsStars { a_in: true, b_in: true }), 0, true),
        "aoutasxaoutbs" => (Ok(StarsStars { a_in: false, b_in: false }), 0, true),
        "ainasxaoutbs" => (Ok(StarsStars { a_in: true, b_in: false }), 0, true),
        "aoutasxainbs" => (Ok(StarsStars { a_in: false, b_in: true }), 0, true),
        "exta" => (Err(Undirected::TriangleX), 0, false),
        "extb" => (Err(Undirected::TriangleX), 1, false),
        "asaxasb" => (Err(Undirected::StarsStars), 0, true),
        _ => return None,
    };
    Some((|| {
        let needs_directed = kind.is_ok();
        if directed != needs_directed {
            return Err(format!("{name} is only implemented for {} networks", if needs_directed { "directed" } else { "undirected" }));
        }
        if ints.len() != n || ints.iter().any(|&l| !(-1..=1).contains(&l)) {
            return Err(format!("{name}: expected each vertex's level (0, 1 or -1)"));
        }
        let decay = if alternating { *reals.first().ok_or_else(|| format!("{name}: missing decay"))? } else { 0.0 };
        let (directed_kind, undirected_kind) = match kind {
            Ok(d) => (Some(d), None),
            Err(u) => (None, Some(u)),
        };
        Ok(Box::new(MultilevelExtra {
            directed: directed_kind,
            undirected: undirected_kind,
            level: ints.iter().map(|&l| l as i8).collect(),
            side,
            r: 1.0 - (-decay).exp(),
            exp_decay: decay.exp(),
        }) as Box<dyn Term>)
    })())
}
