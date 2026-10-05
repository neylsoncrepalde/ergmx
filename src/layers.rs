//! ergm.multi's Layer Logic: a logical layer's tie at a dyad is a function
//! of the observed layers' ties there (and, in directed networks, of their
//! reverse ties, by t()), in integer arithmetic, as ergm.multi's.
//!
//! The function is a postfix program of (code, argument) pairs: (1, k) the
//! tie of observed layer k (1-based), (2, k) its reverse tie, (3, v) the
//! number v, (4, op) an operator of `Op`.

/// The operators, by code.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Not = 1,
    Neg = 2,
    Abs = 3,
    Sign = 4,
    Round = 5,
    And = 10,
    Or = 11,
    Xor = 12,
    Eq = 13,
    Ne = 14,
    Lt = 15,
    Gt = 16,
    Le = 17,
    Ge = 18,
    Add = 19,
    Sub = 20,
    Mul = 21,
    Div = 22,
    Mod = 23,
    Pow = 24,
    Round2 = 25,
}

impl Op {
    fn from_code(code: i64) -> Option<Self> {
        Some(match code {
            1 => Op::Not,
            2 => Op::Neg,
            3 => Op::Abs,
            4 => Op::Sign,
            5 => Op::Round,
            10 => Op::And,
            11 => Op::Or,
            12 => Op::Xor,
            13 => Op::Eq,
            14 => Op::Ne,
            15 => Op::Lt,
            16 => Op::Gt,
            17 => Op::Le,
            18 => Op::Ge,
            19 => Op::Add,
            20 => Op::Sub,
            21 => Op::Mul,
            22 => Op::Div,
            23 => Op::Mod,
            24 => Op::Pow,
            25 => Op::Round2,
            _ => return None,
        })
    }

    fn arity(self) -> usize {
        if (self as i64) < 10 { 1 } else { 2 }
    }
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Layer(usize),
    Reverse(usize),
    Const(i64),
    Op(Op),
}

/// A logical layer's function of the observed layers.
#[derive(Clone, Debug)]
pub struct Logic {
    steps: Vec<Step>,
    /// The observed layers it reads (0-based), directly and reversed.
    direct: Vec<bool>,
    reversed: Vec<bool>,
}

/// R's ^ for integers (0^0 = 1; negative powers truncate, as ergm.multi's C).
fn power(x: i64, y: i64) -> i64 {
    if y < 0 {
        return if x == 1 { 1 } else if x == -1 { if y % 2 == 0 { 1 } else { -1 } } else { 0 };
    }
    x.saturating_pow(y.min(u32::MAX as i64) as u32)
}

/// R's round(x, digits) (Rmath's fround) for integers: to the nearest
/// multiple of 10^-digits, halves to even (a no-op unless digits < 0).
fn round_to(x: i64, digits: i64) -> i64 {
    if digits >= 0 {
        return x;
    }
    let unit = 10i64.saturating_pow((-digits) as u32);
    let (rest, base) = (x.abs() % unit, x.abs() - x.abs() % unit);
    let up = 2 * rest > unit || (2 * rest == unit && (base / unit) % 2 == 1);
    let r = if up { base + unit } else { base };
    if x < 0 { -r } else { r }
}

impl Logic {
    /// From the postfix program, for `layers` observed layers.
    pub fn new(program: &[i64], layers: usize) -> Result<Self, String> {
        if !program.len().is_multiple_of(2) {
            return Err("layer logic: the program is (code, argument) pairs".into());
        }
        let (mut steps, mut direct, mut reversed, mut depth) = (Vec::new(), vec![false; layers], vec![false; layers], 0i64);
        for pair in program.chunks(2) {
            let step = match pair[0] {
                1 | 2 => {
                    let k = pair[1];
                    if k < 1 || k as usize > layers {
                        return Err(format!("layer logic: no layer {k} of {layers}"));
                    }
                    let k = k as usize - 1;
                    if pair[0] == 1 {
                        direct[k] = true;
                        Step::Layer(k)
                    } else {
                        reversed[k] = true;
                        Step::Reverse(k)
                    }
                }
                3 => Step::Const(pair[1]),
                4 => Step::Op(Op::from_code(pair[1]).ok_or_else(|| format!("layer logic: no operator {}", pair[1]))?),
                code => return Err(format!("layer logic: bad code {code}")),
            };
            depth += match step {
                Step::Op(op) => 1 - op.arity() as i64,
                _ => 1,
            };
            if depth < 1 {
                return Err("layer logic: an operator lacks operands".into());
            }
            steps.push(step);
        }
        if depth != 1 {
            return Err("layer logic: the program doesn't leave one value".into());
        }
        Ok(Self { steps, direct, reversed })
    }

    /// Whether a tie of observed layer k (0-based) can change the logical
    /// layer: at the same dyad (`direct`), or at the reverse one.
    pub fn reads(&self, k: usize) -> (bool, bool) {
        (self.direct[k], self.reversed[k])
    }

    /// The value at the dyad (a, b), from `tie(k, a, b)`, the observed layer
    /// k's tie at (a, b).
    pub fn value(&self, tie: &dyn Fn(usize, u32, u32) -> bool, a: u32, b: u32) -> i64 {
        let mut stack: Vec<i64> = Vec::with_capacity(8);
        for step in &self.steps {
            match *step {
                Step::Layer(k) => stack.push(tie(k, a, b) as i64),
                Step::Reverse(k) => stack.push(tie(k, b, a) as i64),
                Step::Const(v) => stack.push(v),
                Step::Op(op) if op.arity() == 1 => {
                    let x = stack.pop().unwrap();
                    stack.push(match op {
                        Op::Not => (x == 0) as i64,
                        Op::Neg => -x,
                        Op::Abs => x.abs(),
                        Op::Sign => x.signum(),
                        _ => x, // round(x): integers already
                    });
                }
                Step::Op(op) => {
                    let y = stack.pop().unwrap();
                    let x = stack.pop().unwrap();
                    stack.push(match op {
                        Op::And => (x != 0 && y != 0) as i64,
                        Op::Or => (x != 0 || y != 0) as i64,
                        Op::Xor => ((x != 0) != (y != 0)) as i64,
                        Op::Eq => (x == y) as i64,
                        Op::Ne => (x != y) as i64,
                        Op::Lt => (x < y) as i64,
                        Op::Gt => (x > y) as i64,
                        Op::Le => (x <= y) as i64,
                        Op::Ge => (x >= y) as i64,
                        Op::Add => x + y,
                        Op::Sub => x - y,
                        Op::Mul => x * y,
                        // Integer division and remainder, as ergm.multi's C (0 by 0 is 0).
                        Op::Div => if y == 0 { 0 } else { x / y },
                        Op::Mod => if y == 0 { 0 } else { x % y },
                        Op::Pow => power(x, y),
                        _ => round_to(x, y),
                    });
                }
            }
        }
        stack[0]
    }

    /// Whether the logical layer has a tie at (a, b).
    #[inline]
    pub fn tie(&self, tie: &dyn Fn(usize, u32, u32) -> bool, a: u32, b: u32) -> bool {
        self.value(tie, a, b) != 0
    }
}

use crate::network::Network;
use crate::valued::ln_factorial;

/// Where the layers are in the network of layers: the first vertex of each,
/// and their (common) size.
#[derive(Clone, Debug)]
pub struct LayerLayout {
    pub starts: Vec<u32>,
    pub size: u32,
}

impl LayerLayout {
    /// The layer of (i, j) and its vertices' numbers in it, if both are in one layer.
    pub fn locate(&self, i: u32, j: u32) -> Option<(usize, u32, u32)> {
        let k = self.starts.partition_point(|&s| s <= i).checked_sub(1)?;
        let s = self.starts[k];
        (i < s + self.size && j >= s && j < s + self.size).then(|| (k, i - s, j - s))
    }

    /// Layer k's tie (a, b) in the network of layers.
    #[inline]
    pub fn tie(&self, net: &Network, k: usize, a: u32, b: u32) -> bool {
        net.has_edge(self.starts[k] + a, self.starts[k] + b)
    }

    /// A logical layer's network, from the network of layers.
    pub fn view(&self, net: &Network, logic: &Logic) -> Network {
        let directed = net.directed();
        let mut out = Network::new(self.size, directed);
        let tie = |l: usize, a: u32, b: u32| self.tie(net, l, a, b);
        // Its ties are among the observed layers' ties (and their reverses):
        // it has none where none of them has one.
        let mut seen = rustc_hash::FxHashSet::default();
        for (k, &start) in self.starts.iter().enumerate() {
            let (direct, reversed) = logic.reads(k);
            if !direct && !reversed {
                continue;
            }
            for &(i, j) in net.edges() {
                let Some((l, a, b)) = self.locate(i, j) else { continue };
                if l != k || i < start {
                    continue;
                }
                let candidates = if directed { [(a, b), (b, a)] } else { [(a.min(b), a.max(b)); 2] };
                for (x, y) in candidates {
                    if seen.insert((x, y)) && logic.tie(&tie, x, y) && !out.has_edge(x, y) {
                        out.toggle(x, y);
                    }
                }
            }
        }
        out
    }
}

/// The dyads of logical layers (view, a, b; a < b if undirected) whose ties
/// toggling (i, j) in the network of layers flips.
pub fn flips(views: &[&Logic], layout: &LayerLayout, net: &Network, i: u32, j: u32) -> Vec<(usize, u32, u32)> {
    let Some((k, a, b)) = layout.locate(i, j) else { return Vec::new() };
    let directed = net.directed();
    let before = |l: usize, x: u32, y: u32| layout.tie(net, l, x, y);
    let toggled = |l: usize, x: u32, y: u32| {
        let same = if directed { (x, y) == (a, b) } else { (x.min(y), x.max(y)) == (a.min(b), a.max(b)) };
        layout.tie(net, l, x, y) ^ (l == k && same)
    };
    let mut out = Vec::new();
    for (v, logic) in views.iter().enumerate() {
        let (direct, reversed) = logic.reads(k);
        let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(2);
        if directed {
            if direct {
                candidates.push((a, b));
            }
            if reversed {
                candidates.push((b, a));
            }
        } else if direct || reversed {
            candidates.push((a.min(b), a.max(b)));
        }
        for (x, y) in candidates {
            if logic.tie(&before, x, y) != logic.tie(&toggled, x, y) {
                out.push((v, x, y));
            }
        }
    }
    out
}

/// The logical layers' networks with some dyads flipped (`after`) or not.
pub struct Overlay<'a> {
    pub views: &'a [Network],
    pub flips: &'a [(usize, u32, u32)],
    pub after: bool,
}

impl Overlay<'_> {
    fn directed(&self) -> bool {
        self.views[0].directed()
    }

    fn flipped(&self, v: usize, x: u32, y: u32) -> bool {
        let directed = self.directed();
        self.after
            && self.flips.iter().any(|&(w, a, b)| w == v && if directed { (a, b) == (x, y) } else { (a, b) == (x.min(y), x.max(y)) })
    }

    /// View v's tie x -> y (x -- y).
    pub fn has(&self, v: usize, x: u32, y: u32) -> bool {
        x != y && (self.views[v].has_edge(x, y) ^ self.flipped(v, x, y))
    }

    fn adjust(&self, v: usize, x: u32, mut list: Vec<u32>, outgoing: bool) -> Vec<u32> {
        if !self.after {
            return list;
        }
        let directed = self.directed();
        for &(w, a, b) in self.flips {
            if w != v {
                continue;
            }
            let other = if !directed {
                if a == x { Some(b) } else if b == x { Some(a) } else { None }
            } else if outgoing {
                (a == x).then_some(b)
            } else {
                (b == x).then_some(a)
            };
            if let Some(o) = other {
                match list.iter().position(|&z| z == o) {
                    Some(p) => {
                        list.swap_remove(p);
                    }
                    None => list.push(o),
                }
            }
        }
        list
    }

    /// View v's out-neighbours of x (neighbours, if undirected).
    pub fn out(&self, v: usize, x: u32) -> Vec<u32> {
        self.adjust(v, x, self.views[v].out_neighbours(x).to_vec(), true)
    }

    /// View v's in-neighbours of x (neighbours, if undirected).
    pub fn inn(&self, v: usize, x: u32) -> Vec<u32> {
        self.adjust(v, x, self.views[v].in_neighbours(x).to_vec(), false)
    }
}

/// A statistic of logical layers.
pub trait LayerStat: Send + Sync {
    fn n_stats(&self) -> usize {
        1
    }

    /// Writes to `out` the change from flipping these dyads of the views together.
    fn change(&self, views: &[Network], flips: &[(usize, u32, u32)], out: &mut [f64]);

    /// Adds the statistics of empty views of `n` vertices.
    fn empty(&self, _n: u32, _directed: bool, _out: &mut [f64]) {}
}

fn unique<T: PartialEq + Copy>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for x in items {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}

/// CMBL: the sum over dyads of log(E! (R - E)! / R!), E the number of the R
/// views with a tie.
struct Cmb;

impl LayerStat for Cmb {
    fn change(&self, views: &[Network], flips: &[(usize, u32, u32)], out: &mut [f64]) {
        let r = views.len() as f64;
        let (before, after) = (Overlay { views, flips, after: false }, Overlay { views, flips, after: true });
        let term = |e: f64| ln_factorial(e) + ln_factorial(r - e);
        for (x, y) in unique(flips.iter().map(|&(_, a, b)| (a, b))) {
            let count = |o: &Overlay| (0..views.len()).filter(|&v| o.has(v, x, y)).count() as f64;
            out[0] += term(count(&after)) - term(count(&before));
        }
    }
}

/// twostarL: two-stars with a tie in each of two views, at a vertex.
struct Twostar {
    /// 0 undirected, 1 out, 2 in, 3 path (in from the first, out in the second).
    kind: u8,
    distinct: bool,
}

impl Twostar {
    fn at(&self, o: &Overlay, c: u32) -> f64 {
        let (first, second) = match self.kind {
            0 | 1 => (o.out(0, c), o.out(1, c)),
            2 => (o.inn(0, c), o.inn(1, c)),
            _ => (o.inn(0, c), o.out(1, c)),
        };
        let common = if self.distinct { first.iter().filter(|x| second.contains(x)).count() } else { 0 };
        (first.len() * second.len() - common) as f64
    }
}

impl LayerStat for Twostar {
    fn change(&self, views: &[Network], flips: &[(usize, u32, u32)], out: &mut [f64]) {
        let (before, after) = (Overlay { views, flips, after: false }, Overlay { views, flips, after: true });
        for c in unique(flips.iter().flat_map(|&(_, a, b)| [a, b])) {
            out[0] += self.at(&after, c) - self.at(&before, c);
        }
    }
}

/// mutualL: ordered pairs (i, j) with i -> j in the first view and j -> i in
/// the second; mode 0 all, 1 matching codes, 2 by matching level, 3 by each
/// end's level.
struct Mutual {
    mode: u8,
    codes: Vec<i64>,
    levels: usize,
}

impl LayerStat for Mutual {
    fn n_stats(&self) -> usize {
        if self.mode >= 2 { self.levels } else { 1 }
    }

    fn change(&self, views: &[Network], flips: &[(usize, u32, u32)], out: &mut [f64]) {
        let (before, after) = (Overlay { views, flips, after: false }, Overlay { views, flips, after: true });
        let pairs = unique(flips.iter().flat_map(|&(_, a, b)| [(a, b), (b, a)]));
        for (x, y) in pairs {
            let mutual = |o: &Overlay| (o.has(0, x, y) && o.has(1, y, x)) as i64 as f64;
            let delta = mutual(&after) - mutual(&before);
            if delta == 0.0 {
                continue;
            }
            let (cx, cy) = (self.codes.get(x as usize).copied().unwrap_or(-1), self.codes.get(y as usize).copied().unwrap_or(-1));
            match self.mode {
                0 => out[0] += delta,
                1 if cx == cy => out[0] += delta,
                2 if cx == cy && cx >= 0 => out[cx as usize] += delta,
                3 => {
                    if cx >= 0 {
                        out[cx as usize] += delta;
                    }
                    if cy >= 0 {
                        out[cy as usize] += delta;
                    }
                }
                _ => {}
            }
        }
    }
}

/// The layer-aware shared partner terms: views 0 and 1 are the two-paths'
/// ties, view 2 (esp, nsp) the base.
struct SharedPartners {
    /// 0 esp, 1 dsp, 2 nsp.
    kind: u8,
    /// 0 UTP (undirected), 1 OTP, 2 ITP, 4 OSP, 5 ISP.
    sp_type: u8,
    any_order: bool,
    cutoff: u32,
    /// The counts of shared partners, each exactly d (d >= 0) or at least -d (an overflow, d < 0).
    ds: Vec<i64>,
    /// gw: exp(decay) (1 - (1 - exp(-decay))^sp), for sp up to the cutoff.
    decay: Option<f64>,
    /// Bipartite (b1dsp, b2dsp): the focal pairs are those of vertices of this mode.
    mode: Option<(Vec<i64>, i64)>,
}

impl SharedPartners {
    /// Whether the focal dyads are unordered (only in undirected networks:
    /// OSP and ISP partners are of ordered pairs, as in ergm).
    fn unordered(&self) -> bool {
        self.sp_type == 0
    }

    fn two(&self, o: &Overlay, a1: u32, b1: u32, a2: u32, b2: u32) -> bool {
        (o.has(0, a1, b1) && o.has(1, a2, b2)) || (self.any_order && o.has(1, a1, b1) && o.has(0, a2, b2))
    }

    /// The shared partners of the focal dyad (i, j): with the order, the
    /// first tie (the one at i, for OSP and ISP) is in the first path layer.
    fn sp(&self, o: &Overlay, i: u32, j: u32) -> u32 {
        let candidates = |x: u32, out: bool| {
            let mut c = if out { o.out(0, x) } else { o.inn(0, x) };
            c.extend(if out { o.out(1, x) } else { o.inn(1, x) });
            unique(c)
        };
        match self.sp_type {
            0 | 1 => candidates(i, true).into_iter().filter(|&k| k != j && self.two(o, i, k, k, j)).count() as u32,
            2 => candidates(j, true).into_iter().filter(|&k| k != i && self.two(o, j, k, k, i)).count() as u32,
            4 => candidates(i, true).into_iter().filter(|&k| k != j && self.two(o, i, k, j, k)).count() as u32,
            _ => candidates(i, false).into_iter().filter(|&k| k != j && self.two(o, k, i, k, j)).count() as u32,
        }
    }

    fn weight(&self, sp: u32, out: &mut [f64], sign: f64) {
        match self.decay {
            Some(alpha) => {
                if sp >= 1 && sp <= self.cutoff {
                    out[0] += sign * alpha.exp() * (1.0 - (1.0 - (-alpha).exp()).powi(sp as i32));
                }
            }
            None => {
                for (o, &d) in out.iter_mut().zip(&self.ds) {
                    if (d >= 0 && sp as i64 == d) || (d < 0 && sp as i64 >= -d) {
                        *o += sign;
                    }
                }
            }
        }
    }

    fn focal(&self, o: &Overlay, i: u32, j: u32) -> bool {
        if let Some((modes, target)) = &self.mode
            && (modes[i as usize] != *target || modes[j as usize] != *target)
        {
            return false;
        }
        match self.kind {
            0 => o.has(2, i, j),
            1 => true,
            _ => !o.has(2, i, j) && !(self.unordered() && o.has(2, j, i)),
        }
    }
}

impl LayerStat for SharedPartners {
    fn n_stats(&self) -> usize {
        if self.decay.is_some() { 1 } else { self.ds.len() }
    }

    fn change(&self, views: &[Network], flips: &[(usize, u32, u32)], out: &mut [f64]) {
        let (before, after) = (Overlay { views, flips, after: false }, Overlay { views, flips, after: true });
        let directed = views[0].directed();
        // The focal dyads whose shared partners or base tie may change: the
        // flipped dyads, and those of their ends with the ends' neighbours.
        let mut ends = Vec::new();
        for &(_, a, b) in flips {
            ends.push(a);
            ends.push(b);
        }
        let ends = unique(ends);
        let mut near = ends.clone();
        for &x in &ends {
            for o in [&before, &after] {
                for v in 0..2 {
                    near.extend(o.out(v, x));
                    near.extend(o.inn(v, x));
                }
            }
        }
        let near = unique(near);
        let mut focal = Vec::new();
        for &x in &ends {
            for &w in &near {
                if w != x {
                    focal.push((x, w));
                    focal.push((w, x));
                }
            }
        }
        let canonical = |(i, j): (u32, u32)| if !directed || self.unordered() { (i.min(j), i.max(j)) } else { (i, j) };
        for (i, j) in unique(focal.into_iter().map(canonical)) {
            for (o, sign) in [(&after, 1.0), (&before, -1.0)] {
                if self.focal(o, i, j) {
                    self.weight(self.sp(o, i, j), out, sign);
                }
            }
        }
    }

    fn empty(&self, n: u32, directed: bool, out: &mut [f64]) {
        // Dyads without shared partners: every focal dyad of dsp and nsp.
        if self.kind == 0 || self.decay.is_some() {
            return;
        }
        let m = match &self.mode {
            Some((modes, target)) => modes.iter().filter(|&&c| c == *target).count() as f64,
            None => n as f64,
        };
        let dyads = m * (m - 1.0) / if !directed || self.unordered() { 2.0 } else { 1.0 };
        for (o, &d) in out.iter_mut().zip(&self.ds) {
            if d == 0 {
                *o += dyads;
            }
        }
    }
}

/// The programs of `count` views, each its length then its (code, argument) pairs.
pub fn programs(ints: &[i64], count: usize, layers: usize) -> Result<(Vec<Logic>, usize), String> {
    let (mut out, mut at) = (Vec::new(), 0);
    for _ in 0..count {
        let len = *ints.get(at).ok_or("layer logic: missing a program")? as usize;
        let program = ints.get(at + 1..at + 1 + len).ok_or("layer logic: a program is cut short")?;
        out.push(Logic::new(program, layers)?);
        at += 1 + len;
    }
    Ok((out, at))
}

/// A layer-aware statistic from its spec, with its views.
pub fn build_stat(name: &str, reals: &[f64], ints: &[i64], layers: usize, size: u32) -> Result<(Box<dyn LayerStat>, Vec<Logic>), String> {
    let int = |k: usize| ints.get(k).copied().ok_or_else(|| format!("{name}: missing parameters"));
    Ok(match name {
        "layerCMB" => {
            let (views, _) = programs(&ints[1..], int(0)? as usize, layers)?;
            (Box::new(Cmb), views)
        }
        "layerTwostar" => {
            let (views, _) = programs(&ints[2..], 2, layers)?;
            (Box::new(Twostar { kind: int(0)? as u8, distinct: int(1)? != 0 }), views)
        }
        "layerMutual" => {
            let (mode, levels) = (int(0)? as u8, int(1)? as usize);
            let n_codes = if mode == 0 { 0 } else { size as usize };
            let codes = ints.get(2..2 + n_codes).ok_or("layerMutual: one code per vertex")?.to_vec();
            let (views, _) = programs(&ints[2 + n_codes..], 2, layers)?;
            (Box::new(Mutual { mode, codes, levels }), views)
        }
        "layerSP" => {
            let (kind, sp_type, any_order, cutoff, nd) = (int(0)? as u8, int(1)? as u8, int(2)? != 0, int(3)? as u32, int(4)? as usize);
            let ds = ints.get(5..5 + nd).ok_or("layerSP: missing the degrees")?.to_vec();
            // Bipartite: the mode of the focal pairs, then each vertex's (none: 0).
            let target = *ints.get(5 + nd).ok_or("layerSP: missing the mode")?;
            let at = 6 + nd + if target > 0 { size as usize } else { 0 };
            let mode = if target > 0 {
                Some((ints.get(6 + nd..at).ok_or("layerSP: one mode per vertex")?.to_vec(), target))
            } else {
                None
            };
            let count = *ints.get(at).ok_or("layerSP: missing the views")? as usize;
            let (views, _) = programs(&ints[at + 1..], count, layers)?;
            let decay = reals.first().copied();
            (Box::new(SharedPartners { kind, sp_type, any_order, cutoff, ds, decay, mode }), views)
        }
        _ => return Err(format!("unknown layer term {name:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_logic_evaluates_as_r() {
        // Layers 1 and 2 at a dyad where 1 has a tie and 2 doesn't, and 2 has the reverse tie.
        let tie = |k: usize, a: u32, b: u32| matches!((k, a, b), (0, 0, 1) | (1, 1, 0));
        let check = |program: &[i64], want: i64| {
            assert_eq!(Logic::new(program, 2).unwrap().value(&tie, 0, 1), want, "{program:?}");
        };
        check(&[1, 1, 1, 2, 4, 10], 0); // `1` & `2`
        check(&[1, 1, 1, 2, 4, 11], 1); // `1` | `2`
        check(&[1, 1, 1, 2, 4, 1, 4, 10], 1); // `1` & !`2`
        check(&[1, 1, 2, 2, 4, 10], 1); // `1` & t(`2`)
        check(&[3, 2, 1, 1, 4, 21, 1, 2, 4, 20], 2); // 2 * `1` - `2`
        check(&[1, 1, 1, 2, 4, 19, 3, 2, 4, 13], 0); // (`1` + `2`) == 2
        check(&[3, 7, 3, 2, 4, 22], 3); // 7 / 2, integer
        check(&[3, -7, 3, 2, 4, 23], -1); // -7 %% 2, as C's
        check(&[3, 1234, 3, -2, 4, 25], 1200); // round(1234, -2)
        check(&[3, 1250, 3, -2, 4, 25], 1200); // halves to even
        check(&[3, 1350, 3, -2, 4, 25], 1400);
        assert!(Logic::new(&[1, 1, 4, 10], 2).is_err());
        assert!(Logic::new(&[1, 3], 2).is_err());
    }
}
