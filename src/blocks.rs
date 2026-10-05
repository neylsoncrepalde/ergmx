//! The membership step of bigergm's MM algorithm (Babkin et al. 2020): each
//! vertex's posterior block probabilities tau_i maximize, over the simplex,
//! the surrogate sum_k s_ik tau_ik - m_ik tau_ik^2, solved row by row as
//! bigergm's solveQP does, by its active-set iterations.

/// The new memberships (n x k, row major) from the quadratic coefficients `m`
/// and the linear ones `s`; `tau` holds the current ones, kept for the rows
/// with nothing to solve.
pub fn solve_qp(m: &[f64], s: &[f64], tau: &mut [f64], k: usize, precision: f64) {
    let n = m.len().checked_div(k).unwrap_or(0);
    let (mut in_k, mut at_zero) = (vec![false; k], vec![false; k]);
    let (mut free, mut zero, mut one) = (vec![false; k], vec![false; k], vec![false; k]);
    for row in 0..n {
        let (m, s, tau) = (&m[row * k..(row + 1) * k], &s[row * k..(row + 1) * k], &mut tau[row * k..(row + 1) * k]);
        in_k.iter_mut().for_each(|x| *x = true);
        at_zero.iter_mut().for_each(|x| *x = false);
        let mut count = 0;
        loop {
            count += 1;
            // The multiplier of the active set's constraint sum tau = 1.
            let (mut v1, mut v2) = (0.0, 0.0);
            for j in 0..k {
                if in_k[j] {
                    v1 += 1.0 / m[j];
                    v2 += s[j] / m[j];
                }
            }
            let lambda = (v2 - 2.0) / v1;
            for j in 0..k {
                (free[j], zero[j], one[j]) = if !in_k[j] {
                    (false, false, false)
                } else if lambda >= s[j] {
                    (false, true, false)
                } else if lambda > -2.0 * m[j] + s[j] {
                    (true, false, false)
                } else {
                    (false, false, true)
                };
            }
            let (mut delta, mut v3, mut v4, mut none_free) = (0.0, 0.0, 0.0, true);
            for j in 0..k {
                delta += one[j] as u8 as f64;
                if free[j] {
                    v3 += s[j] / m[j];
                    v4 += 1.0 / m[j];
                    none_free = false;
                }
            }
            delta += v3 / 2.0 - lambda * v4 / 2.0 - 1.0;
            if delta.abs() < precision || none_free || count >= k {
                for j in 0..k {
                    tau[j] = if at_zero[j] || zero[j] {
                        0.0
                    } else if one[j] {
                        1.0
                    } else {
                        (s[j] - lambda) / (2.0 * m[j])
                    };
                }
                break;
            } else if delta > 0.0 {
                for j in 0..k {
                    if zero[j] {
                        at_zero[j] = true;
                        in_k[j] = false;
                    }
                }
            } else {
                for j in 0..k {
                    tau[j] = if one[j] { 1.0 } else { 0.0 };
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_solution_maximizes_the_surrogate_on_the_simplex() {
        // max s.t - m.t^2 over the simplex, against a grid.
        let (m, s) = (vec![2.0, 1.0, 3.0], vec![1.0, 0.5, 2.5]);
        let mut tau = vec![1.0 / 3.0; 3];
        solve_qp(&m, &s, &mut tau, 3, 1e-10);
        let value = |t: &[f64]| t.iter().zip(&m).zip(&s).map(|((t, m), s)| s * t - m * t * t).sum::<f64>();
        let best = value(&tau);
        assert!((tau.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for a in 0..=100 {
            for b in 0..=(100 - a) {
                let t = [a as f64 / 100.0, b as f64 / 100.0, (100 - a - b) as f64 / 100.0];
                assert!(value(&t) <= best + 1e-9, "{t:?}");
            }
        }
    }
}
