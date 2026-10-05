//! Latent space models of networks, as R's latentnet (`ergmm()`; Hoff,
//! Raftery and Handcock 2002; Handcock, Raftery and Tantrum 2007; Krivitsky
//! et al. 2009): the dyads are independent given each vertex's position Z_i
//! in a latent space (and, optionally, its random sender, receiver or
//! sociality effect), with
//!
//!   eta_ij = L(Z_i, Z_j) + sum_k beta_k X_k[i, j] + s_i + r_j,
//!
//! L the negative Euclidean distance, its square or the inner product, and
//! the positions normal around their cluster's mean (a mixture of G
//! spherical normals) or around 0. The MCMC is latentnet's: each vertex's
//! position and random effects proposed jointly, the clusters and the
//! variances by Gibbs steps, and a joint proposal of the coefficients, the
//! scale of the positions and shifts of the random effects.

use crate::rng::Rng;
use crate::valued::ln_gamma;

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

#[derive(Clone, Copy, PartialEq)]
pub enum Latent {
    None,
    Euclidean,
    Bilinear,
    Euclidean2,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Family {
    Bernoulli,
    Binomial,
    Poisson,
    Normal,
}

/// The model's priors, as latentnet's: beta_k ~ N(mean_k, var_k); with
/// clusters, Z_i ~ N(mu_K, s2_K I), mu_g ~ N(0, z_mean_var I),
/// s2_g ~ SIchi2(z_var_df, z_var), pi ~ Dirichlet(z_pk); without, Z_i ~
/// N(0, s2 I) with s2 ~ SIchi2(z_var_df, z_var); random effects normal
/// around 0 with SIchi2 variances; the normal family's variance SIchi2.
#[derive(Clone)]
pub struct Prior {
    pub beta_mean: Vec<f64>,
    pub beta_var: Vec<f64>,
    pub z_var: f64,
    pub z_mean_var: f64,
    pub z_var_df: f64,
    pub z_pk: f64,
    pub sender_var: f64,
    pub sender_var_df: f64,
    pub receiver_var: f64,
    pub receiver_var_df: f64,
    pub dispersion: f64,
    pub dispersion_df: f64,
}

/// The model: the network's values `y` (n x n, row major) on the observed
/// dyads (`observed`: directed i != j, undirected i > j), the covariates
/// `x` (p x n x n), and the shifts of the random effects that the joint
/// proposal makes (`eff_sender`: k_s x n; `eff_receiver`: k_r x n).
pub struct Model {
    pub n: usize,
    pub d: usize,
    pub g: usize,
    pub p: usize,
    pub latent: Latent,
    pub family: Family,
    pub directed: bool,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub trials: Vec<f64>,
    pub observed: Vec<bool>,
    pub sender: bool,
    pub receiver: bool,
    /// One effect s_i for both ends: eta_ij gets s_i + s_j (in `Par::sender`).
    pub sociality: bool,
    pub eff_sender: Vec<f64>,
    pub eff_receiver: Vec<f64>,
    pub prior: Prior,
}

/// One configuration of the parameters; with sociality, `sender` holds it.
#[derive(Clone)]
pub struct Par {
    pub beta: Vec<f64>,
    pub z: Vec<f64>,
    pub z_mean: Vec<f64>,
    pub z_var: Vec<f64>,
    pub z_k: Vec<usize>,
    pub z_pk: Vec<f64>,
    pub sender: Vec<f64>,
    pub receiver: Vec<f64>,
    pub sender_var: f64,
    pub receiver_var: f64,
    pub dispersion: f64,
}

/// The log-densities of a configuration (latentnet's lpY, lpZ, lpbeta...).
#[derive(Clone, Copy, Default)]
pub struct Lp {
    pub y: f64,
    pub z: f64,
    pub beta: f64,
    pub re: f64,
    pub lv: f64,
    pub rev: f64,
    pub dispersion: f64,
}

impl Lp {
    /// The posterior's log-density, given the clusters (latentnet's posterior mode criterion).
    pub fn total(&self) -> f64 {
        self.y + self.z + self.lv + self.beta + self.re + self.rev + self.dispersion
    }
}

fn ln_norm(x: f64, mean: f64, var: f64) -> f64 {
    let dx = x - mean;
    -LN_SQRT_2PI - 0.5 * var.ln() - dx * dx / (2.0 * var)
}

/// log density of the scaled inverse chi-squared with `df` and `scale`.
pub fn ln_sclinvchisq(x: f64, df: f64, scale: f64) -> f64 {
    0.5 * df * (0.5 * df * scale).ln() - ln_gamma(0.5 * df) - (0.5 * df + 1.0) * x.ln() - df * scale / (2.0 * x)
}

/// log(1 + e^x), without overflow.
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

impl Model {
    #[inline]
    fn obs(&self, i: usize, j: usize) -> bool {
        self.observed[i * self.n + j]
    }

    pub fn eta(&self, par: &Par, i: usize, j: usize) -> f64 {
        let d = self.d;
        let mut eta = match self.latent {
            Latent::None => 0.0,
            Latent::Bilinear => (0..d).map(|k| par.z[i * d + k] * par.z[j * d + k]).sum(),
            Latent::Euclidean | Latent::Euclidean2 => {
                let s: f64 = (0..d).map(|k| (par.z[i * d + k] - par.z[j * d + k]).powi(2)).sum();
                if self.latent == Latent::Euclidean { -s.sqrt() } else { -s }
            }
        };
        let nn = self.n * self.n;
        for k in 0..self.p {
            eta += par.beta[k] * self.x[k * nn + i * self.n + j];
        }
        if self.sociality {
            eta += par.sender[i] + par.sender[j];
        } else {
            if self.sender {
                eta += par.sender[i];
            }
            if self.receiver {
                eta += par.receiver[j];
            }
        }
        eta
    }

    /// The dyad's log-likelihood, without the family's constants.
    pub fn lp_edge(&self, par: &Par, i: usize, j: usize) -> f64 {
        let eta = self.eta(par, i, j);
        let y = self.y[i * self.n + j];
        match self.family {
            Family::Bernoulli => y * eta - softplus(eta),
            Family::Binomial => y * eta - self.trials[i * self.n + j] * softplus(eta),
            Family::Poisson => y * eta - eta.exp(),
            Family::Normal => -(y - eta).powi(2) / par.dispersion / 2.0 - par.dispersion.ln() / 2.0,
        }
    }

    /// The constants of the log-likelihood (binomial coefficients, -log y!, -log sqrt(2 pi)).
    pub fn lp_y_const(&self) -> f64 {
        let mut c = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                if !self.obs(i, j) {
                    continue;
                }
                let y = self.y[i * self.n + j];
                c += match self.family {
                    Family::Bernoulli => 0.0,
                    Family::Binomial => {
                        let m = self.trials[i * self.n + j];
                        ln_gamma(m + 1.0) - ln_gamma(y + 1.0) - ln_gamma(m - y + 1.0)
                    }
                    Family::Poisson => -ln_gamma(y + 1.0),
                    Family::Normal => -LN_SQRT_2PI,
                };
            }
        }
        c
    }

    fn lp_z_vertex(&self, par: &Par, i: usize) -> f64 {
        let d = self.d;
        if self.g > 0 {
            let k = par.z_k[i];
            (0..d).map(|c| ln_norm(par.z[i * d + c], par.z_mean[k * d + c], par.z_var[k])).sum()
        } else {
            (0..d).map(|c| ln_norm(par.z[i * d + c], 0.0, par.z_var[0])).sum()
        }
    }

    fn lp_re_vertex(&self, par: &Par, i: usize) -> f64 {
        let mut lp = 0.0;
        if self.sender || self.sociality {
            lp += ln_norm(par.sender[i], 0.0, par.sender_var);
        }
        if self.receiver && !self.sociality {
            lp += ln_norm(par.receiver[i], 0.0, par.receiver_var);
        }
        lp
    }

    /// The log-densities of a configuration, the likelihood without its constants.
    pub fn lp(&self, par: &Par, lpedge: &mut [f64]) -> Lp {
        let n = self.n;
        let mut lp = Lp::default();
        for i in 0..n {
            for j in 0..n {
                if self.obs(i, j) {
                    let v = self.lp_edge(par, i, j);
                    lpedge[i * n + j] = v;
                    lp.y += v;
                }
            }
        }
        if self.d > 0 {
            lp.z = (0..n).map(|i| self.lp_z_vertex(par, i)).sum();
            lp.lv = self.lp_lv(par);
        }
        lp.beta = (0..self.p).map(|k| ln_norm(par.beta[k], self.prior.beta_mean[k], self.prior.beta_var[k])).sum();
        lp.re = (0..n).map(|i| self.lp_re_vertex(par, i)).sum();
        lp.rev = self.lp_rev(par);
        if self.family == Family::Normal {
            lp.dispersion = ln_sclinvchisq(par.dispersion, self.prior.dispersion_df, self.prior.dispersion);
        }
        lp
    }

    fn lp_lv(&self, par: &Par) -> f64 {
        let pr = &self.prior;
        if self.g > 0 {
            let mut lp = 0.0;
            for k in 0..self.g {
                for c in 0..self.d {
                    lp += ln_norm(par.z_mean[k * self.d + c], 0.0, pr.z_mean_var);
                }
                lp += ln_sclinvchisq(par.z_var[k], pr.z_var_df, pr.z_var);
            }
            lp
        } else {
            ln_sclinvchisq(par.z_var[0], pr.z_var_df, pr.z_var)
        }
    }

    fn lp_rev(&self, par: &Par) -> f64 {
        let pr = &self.prior;
        let mut lp = 0.0;
        if self.sender || self.sociality {
            lp += ln_sclinvchisq(par.sender_var, pr.sender_var_df, pr.sender_var);
        }
        if self.receiver && !self.sociality {
            lp += ln_sclinvchisq(par.receiver_var, pr.receiver_var_df, pr.receiver_var);
        }
        lp
    }

    /// The dimension of the joint proposal: coefficients, the positions'
    /// log-scale, the random effects' shifts and the log-variance.
    pub fn group_dim(&self) -> usize {
        self.p + (self.d > 0) as usize + self.ks() + self.kr() + (self.family == Family::Normal) as usize
    }

    fn ks(&self) -> usize {
        if self.sender || self.sociality { self.eff_sender.len() / self.n } else { 0 }
    }

    fn kr(&self) -> usize {
        if self.receiver && !self.sociality { self.eff_receiver.len() / self.n } else { 0 }
    }
}

// -- Random variates ----------------------------------------------------------------------------

/// Gamma(shape, 1), by Marsaglia and Tsang's method.
pub fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.unif();
        return gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = rng.normal();
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v * v * v);
            }
        };
        let u = rng.unif();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn chisq(rng: &mut Rng, df: f64) -> f64 {
    2.0 * gamma(rng, df / 2.0)
}

/// A draw of the scaled inverse chi-squared: scale df / chi2(df).
fn sclinvchisq(rng: &mut Rng, df: f64, scale: f64) -> f64 {
    scale * df / chisq(rng, df)
}

// -- The sampler --------------------------------------------------------------------------------

/// The proposals' sizes: the vertices' position and random effects' steps,
/// and the joint proposal's matrix U (m x m, row major; the step is U' eps).
pub struct Deltas {
    pub z: f64,
    pub re: f64,
    pub group: Vec<f64>,
}

/// A chain's draws (each draw's values in turn), its last state, and the
/// configurations of the highest likelihood and posterior density seen.
pub struct Draws {
    pub beta: Vec<f64>,
    pub z: Vec<f64>,
    pub z_k: Vec<usize>,
    pub z_mean: Vec<f64>,
    pub z_var: Vec<f64>,
    pub z_pk: Vec<f64>,
    pub sender: Vec<f64>,
    pub receiver: Vec<f64>,
    pub sender_var: Vec<f64>,
    pub receiver_var: Vec<f64>,
    pub dispersion: Vec<f64>,
    pub lp: Vec<Lp>,
    pub z_rate: Vec<f64>,
    pub group_rate: Vec<f64>,
    pub last: Par,
    pub best_y: (Par, Lp),
    pub best_post: (Par, Lp),
}

struct State<'m> {
    model: &'m Model,
    par: Par,
    lp: Lp,
    lpedge: Vec<f64>,
}

impl<'m> State<'m> {
    fn new(model: &'m Model, par: Par) -> Self {
        let mut lpedge = vec![0.0; model.n * model.n];
        let lp = model.lp(&par, &mut lpedge);
        Self { model, par, lp, lpedge }
    }

    /// The observed dyads with vertex i at an end.
    fn dyads_of(&self, i: usize, out: &mut Vec<(usize, usize)>) {
        let m = self.model;
        out.clear();
        for j in 0..m.n {
            if m.directed {
                if m.obs(i, j) {
                    out.push((i, j));
                }
                if j != i && m.obs(j, i) {
                    out.push((j, i));
                }
            } else {
                let (a, b) = if i > j { (i, j) } else { (j, i) };
                if m.obs(a, b) {
                    out.push((a, b));
                }
            }
        }
    }

    /// Step A: each vertex's position and random effects, jointly, in a random order.
    fn vertex_moves(&mut self, deltas: &Deltas, rng: &mut Rng, dyads: &mut Vec<(usize, usize)>, buf: &mut Vec<f64>) -> usize {
        let m = self.model;
        let (n, d) = (m.n, m.d);
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..n.saturating_sub(1) {
            let k = i + rng.below((n - i) as u64) as usize;
            order.swap(i, k);
        }
        let (has_s, has_r) = (m.sender || m.sociality, m.receiver && !m.sociality);
        let mut accepted = 0;
        for &i in &order {
            let old_z: Vec<f64> = self.par.z[i * d..(i + 1) * d].to_vec();
            let (old_s, old_r) = (
                if has_s { self.par.sender[i] } else { 0.0 },
                if has_r { self.par.receiver[i] } else { 0.0 },
            );
            let before = if d > 0 { m.lp_z_vertex(&self.par, i) } else { 0.0 } + m.lp_re_vertex(&self.par, i);
            for c in 0..d {
                self.par.z[i * d + c] += deltas.z * rng.normal();
            }
            if has_s {
                self.par.sender[i] += deltas.re * rng.normal();
            }
            if has_r {
                self.par.receiver[i] += deltas.re * rng.normal();
            }
            self.dyads_of(i, dyads);
            buf.clear();
            let mut diff = 0.0;
            for &(a, b) in dyads.iter() {
                let v = m.lp_edge(&self.par, a, b);
                diff += v - self.lpedge[a * n + b];
                buf.push(v);
            }
            let after = if d > 0 { m.lp_z_vertex(&self.par, i) } else { 0.0 } + m.lp_re_vertex(&self.par, i);
            let lr = diff + after - before;
            if rng.unif() < lr.exp() {
                for (&(a, b), &v) in dyads.iter().zip(buf.iter()) {
                    self.lpedge[a * n + b] = v;
                }
                self.lp.y += diff;
                accepted += 1;
            } else {
                self.par.z[i * d..(i + 1) * d].copy_from_slice(&old_z);
                if has_s {
                    self.par.sender[i] = old_s;
                }
                if has_r {
                    self.par.receiver[i] = old_r;
                }
            }
        }
        if d > 0 {
            self.lp.z = (0..n).map(|i| m.lp_z_vertex(&self.par, i)).sum();
        }
        self.lp.re = (0..n).map(|i| m.lp_re_vertex(&self.par, i)).sum();
        accepted
    }

    /// Step B: the clusters, their probabilities, variances and means (or
    /// the positions' variance), by Gibbs steps.
    #[allow(clippy::needless_range_loop)] // the clusters' arrays, side by side
    fn latent_gibbs(&mut self, rng: &mut Rng) {
        let m = self.model;
        let (n, d, g) = (m.n, m.d, m.g);
        let pr = &m.prior;
        let par = &mut self.par;
        if g > 0 {
            // Labels, from the current means, variances and probabilities (in logs: no underflow).
            let mut w = vec![0.0; g];
            for i in 0..n {
                for k in 0..g {
                    w[k] = par.z_pk[k].ln()
                        + (0..d).map(|c| ln_norm(par.z[i * d + c], par.z_mean[k * d + c], par.z_var[k])).sum::<f64>();
                }
                let top = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = w.iter().map(|x| (x - top).exp()).sum();
                let u = rng.unif() * total;
                let mut acc = 0.0;
                par.z_k[i] = g - 1;
                for (k, x) in w.iter().enumerate() {
                    acc += (x - top).exp();
                    if u < acc {
                        par.z_k[i] = k;
                        break;
                    }
                }
            }
            let mut counts = vec![0usize; g];
            for &k in &par.z_k {
                counts[k] += 1;
            }
            let draws: Vec<f64> = (0..g).map(|k| gamma(rng, counts[k] as f64 + pr.z_pk)).collect();
            let sum: f64 = draws.iter().sum();
            par.z_pk = draws.iter().map(|x| x / sum).collect();
            // Variances, given the old means; then the means, given the new variances.
            for k in 0..g {
                let mut s = 0.0;
                for i in 0..n {
                    if par.z_k[i] == k {
                        s += (0..d).map(|c| (par.z[i * d + c] - par.z_mean[k * d + c]).powi(2)).sum::<f64>();
                    }
                }
                let df = counts[k] as f64 * d as f64 + pr.z_var_df;
                par.z_var[k] = sclinvchisq(rng, df, (pr.z_var * pr.z_var_df + s) / df);
            }
            for k in 0..g {
                let nk = counts[k] as f64;
                for c in 0..d {
                    let mean = if counts[k] > 0 {
                        (0..n).filter(|&i| par.z_k[i] == k).map(|i| par.z[i * d + c]).sum::<f64>() / nk
                    } else {
                        0.0
                    };
                    let shrink = nk + par.z_var[k] / pr.z_mean_var;
                    par.z_mean[k * d + c] = nk * mean / shrink + (par.z_var[k] / shrink).sqrt() * rng.normal();
                }
            }
        } else {
            let s: f64 = par.z.iter().map(|x| x * x).sum();
            let df = (n * d) as f64 + pr.z_var_df;
            par.z_var[0] = sclinvchisq(rng, df, (pr.z_var * pr.z_var_df + s) / df);
        }
        self.lp.z = (0..n).map(|i| m.lp_z_vertex(&self.par, i)).sum();
        self.lp.lv = m.lp_lv(&self.par);
    }

    /// Step C: the coefficients, the positions' scale (with the clusters'
    /// means and variances) and shifts of the random effects, jointly.
    fn group_move(&mut self, deltas: &Deltas, rng: &mut Rng, fresh: &mut [f64]) -> bool {
        let m = self.model;
        let dim = m.group_dim();
        let (n, d, g, p) = (m.n, m.d, m.g, m.p);
        let eps: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        let u: Vec<f64> = (0..dim).map(|j| (0..dim).map(|i| deltas.group[i * dim + j] * eps[i]).sum()).collect();
        let mut prop = self.par.clone();
        let mut jacobian = 0.0;
        let mut at = 0;
        for k in 0..p {
            prop.beta[k] += u[at];
            at += 1;
        }
        if d > 0 {
            let logh = u[at];
            at += 1;
            let h = logh.exp();
            prop.z.iter_mut().for_each(|z| *z *= h);
            jacobian += (n * d) as f64 * logh;
            if g > 0 {
                prop.z_mean.iter_mut().for_each(|z| *z *= h);
                prop.z_var.iter_mut().for_each(|v| *v *= h * h);
                jacobian += (g * d) as f64 * logh + (2 * g) as f64 * logh;
            } else {
                prop.z_var[0] *= h * h;
                jacobian += 2.0 * logh;
            }
        }
        for k in 0..m.ks() {
            for i in 0..n {
                prop.sender[i] += u[at] * m.eff_sender[k * n + i];
            }
            at += 1;
        }
        for k in 0..m.kr() {
            for i in 0..n {
                prop.receiver[i] += u[at] * m.eff_receiver[k * n + i];
            }
            at += 1;
        }
        if m.family == Family::Normal {
            prop.dispersion *= (2.0 * u[at]).exp();
            jacobian += 2.0 * u[at];
        }
        let new = m.lp(&prop, fresh);
        let lr = (new.y - self.lp.y) + (new.beta - self.lp.beta) + (new.z - self.lp.z) + (new.lv - self.lp.lv)
            + (new.re - self.lp.re) + (new.dispersion - self.lp.dispersion) + jacobian;
        if rng.unif() < lr.exp() {
            self.par = prop;
            // The random effects' variances, and so lp.rev, are unchanged.
            self.lp = Lp { rev: self.lp.rev, ..new };
            self.lpedge.copy_from_slice(fresh);
            true
        } else {
            false
        }
    }

    /// Step D: the random effects' variances, by Gibbs steps.
    fn re_gibbs(&mut self, rng: &mut Rng) {
        let m = self.model;
        let pr = &m.prior;
        let n = m.n as f64;
        if m.sender || m.sociality {
            let s: f64 = self.par.sender.iter().map(|x| x * x).sum();
            self.par.sender_var = sclinvchisq(rng, n + pr.sender_var_df, (pr.sender_var * pr.sender_var_df + s) / (n + pr.sender_var_df));
        }
        if m.receiver && !m.sociality {
            let s: f64 = self.par.receiver.iter().map(|x| x * x).sum();
            self.par.receiver_var =
                sclinvchisq(rng, n + pr.receiver_var_df, (pr.receiver_var * pr.receiver_var_df + s) / (n + pr.receiver_var_df));
        }
        self.lp.re = (0..m.n).map(|i| m.lp_re_vertex(&self.par, i)).sum();
        self.lp.rev = m.lp_rev(&self.par);
    }
}

/// `samples` draws, `interval` iterations apart, of a chain from `start`.
pub fn run(model: &Model, start: Par, deltas: &Deltas, samples: usize, interval: usize, seed: u64) -> Draws {
    let mut rng = Rng::new(seed);
    let mut state = State::new(model, start);
    let mut dyads = Vec::with_capacity(2 * model.n);
    let mut buf = Vec::with_capacity(2 * model.n);
    let mut fresh = vec![0.0; model.n * model.n];
    let mut draws = Draws {
        beta: Vec::new(),
        z: Vec::new(),
        z_k: Vec::new(),
        z_mean: Vec::new(),
        z_var: Vec::new(),
        z_pk: Vec::new(),
        sender: Vec::new(),
        receiver: Vec::new(),
        sender_var: Vec::new(),
        receiver_var: Vec::new(),
        dispersion: Vec::new(),
        lp: Vec::new(),
        z_rate: Vec::new(),
        group_rate: Vec::new(),
        last: state.par.clone(),
        best_y: (state.par.clone(), state.lp),
        best_post: (state.par.clone(), state.lp),
    };
    let has_effects = model.d > 0 || model.sender || model.receiver || model.sociality;
    let (mut z_accepted, mut group_accepted) = (0usize, 0usize);
    for iteration in 1..=samples * interval {
        if has_effects {
            z_accepted += state.vertex_moves(deltas, &mut rng, &mut dyads, &mut buf);
        }
        if model.d > 0 {
            state.latent_gibbs(&mut rng);
        }
        if model.group_dim() > 0 && state.group_move(deltas, &mut rng, &mut fresh) {
            group_accepted += 1;
        }
        if model.sender || model.receiver || model.sociality {
            state.re_gibbs(&mut rng);
        }
        if state.lp.y > draws.best_y.1.y {
            draws.best_y = (state.par.clone(), state.lp);
        }
        if state.lp.total() > draws.best_post.1.total() {
            draws.best_post = (state.par.clone(), state.lp);
        }
        if iteration % interval == 0 {
            let par = &state.par;
            draws.beta.extend_from_slice(&par.beta);
            draws.z.extend_from_slice(&par.z);
            draws.z_k.extend_from_slice(&par.z_k);
            draws.z_mean.extend_from_slice(&par.z_mean);
            draws.z_var.extend_from_slice(&par.z_var);
            draws.z_pk.extend_from_slice(&par.z_pk);
            draws.sender.extend_from_slice(&par.sender);
            draws.receiver.extend_from_slice(&par.receiver);
            draws.sender_var.push(par.sender_var);
            draws.receiver_var.push(par.receiver_var);
            draws.dispersion.push(par.dispersion);
            draws.lp.push(state.lp);
            draws.z_rate.push(z_accepted as f64 / (interval * model.n) as f64);
            draws.group_rate.push(group_accepted as f64 / interval as f64);
            z_accepted = 0;
            group_accepted = 0;
        }
    }
    draws.last = state.par;
    draws
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_draws_have_the_right_moments() {
        let mut rng = Rng::new(1);
        for shape in [0.5, 1.0, 3.7] {
            let draws: Vec<f64> = (0..200_000).map(|_| gamma(&mut rng, shape)).collect();
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
            assert!((mean - shape).abs() < 0.02 * shape.max(1.0), "{shape}: mean {mean}");
            assert!((var - shape).abs() < 0.05 * shape.max(1.0), "{shape}: var {var}");
        }
    }

    #[test]
    fn scaled_inverse_chi_squared_density_integrates_to_one() {
        let (df, scale) = (4.5, 0.8);
        let (mut total, h) = (0.0, 1e-4);
        let mut x = h / 2.0;
        while x < 200.0 {
            total += ln_sclinvchisq(x, df, scale).exp() * h;
            x += h;
        }
        assert!((total - 1.0).abs() < 1e-3, "{total}");
    }
}
