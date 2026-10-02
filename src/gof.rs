//! The distributions of goodness of fit (ergm's gof()): edgewise and dyadwise
//! shared partners and geodesic distances, with memory linear in the network
//! size (breadth-first searches and two-path counters, no n x n matrices).

use crate::network::{Network, count_common};

/// Edgewise shared partners: for each tie i -> j, the vertices k with i - k
/// and k - j (i -> k -> j if directed, ergm's OTP). Counts 0 to n - 2.
pub fn espartners(net: &Network) -> Vec<u64> {
    let n = net.n() as usize;
    let mut out = vec![0u64; n.saturating_sub(1).max(1)];
    for &(i, j) in net.edges() {
        let k = count_common(net.out_neighbours(i), net.in_neighbours(j)) as usize;
        out[k] += 1;
    }
    out.truncate(n.saturating_sub(1));
    out
}

/// Dyadwise shared partners of every pair (ordered pairs and OTP two-paths
/// if directed). Counts 0 to n - 2.
pub fn dspartners(net: &Network) -> Vec<u64> {
    let n = net.n() as usize;
    let directed = net.directed();
    let mut out = vec![0u64; n.saturating_sub(1).max(1)];
    let mut count = vec![0u32; n];
    let mut touched = Vec::new();
    let mut nonzero = 0u64;
    for i in 0..n as u32 {
        for &k in net.out_neighbours(i) {
            for &j in net.out_neighbours(k) {
                if j == i || (!directed && j < i) {
                    continue;
                }
                if count[j as usize] == 0 {
                    touched.push(j);
                }
                count[j as usize] += 1;
            }
        }
        for &j in &touched {
            out[count[j as usize] as usize] += 1;
            count[j as usize] = 0;
        }
        nonzero += touched.len() as u64;
        touched.clear();
    }
    let pairs = n as u64 * (n as u64).saturating_sub(1) / if directed { 1 } else { 2 };
    out[0] += pairs - nonzero;
    out.truncate(n.saturating_sub(1));
    out
}

/// Geodesic distances of every pair (ordered if directed, along the ties'
/// directions): counts of distances 1 to n - 1, then of unreachable pairs.
pub fn distances(net: &Network) -> Vec<u64> {
    let n = net.n() as usize;
    let directed = net.directed();
    let mut out = vec![0u64; n.max(1)];
    let mut depth = vec![u32::MAX; n];
    let mut queue = Vec::with_capacity(n);
    for s in 0..n as u32 {
        depth[s as usize] = 0;
        queue.clear();
        queue.push(s);
        let mut head = 0;
        let mut reached = 0u64;
        while head < queue.len() {
            let v = queue[head];
            head += 1;
            let d = depth[v as usize] + 1;
            for &w in net.out_neighbours(v) {
                if depth[w as usize] == u32::MAX {
                    depth[w as usize] = d;
                    queue.push(w);
                    if directed || w > s {
                        out[d as usize - 1] += 1;
                        reached += 1;
                    }
                }
            }
        }
        let others = if directed { n as u64 - 1 } else { (n - 1 - s as usize) as u64 };
        out[n - 1] += others - reached;
        for &v in &queue {
            depth[v as usize] = u32::MAX;
        }
    }
    if n == 0 {
        out.clear();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_and_a_triangle() {
        // 0 - 1 - 2 - 0 (a triangle), 2 - 3, and 4 alone.
        let net = Network::from_edges(5, false, &[(0, 1), (1, 2), (0, 2), (2, 3)]).unwrap();
        assert_eq!(espartners(&net), vec![1, 3, 0, 0]);
        // Pairs with 1 shared partner: 01 (2), 02 (1), 12 (0), 03 (2), 13 (2).
        assert_eq!(dspartners(&net), vec![5, 5, 0, 0]);
        // Distance 1: 4 ties; 2: 03, 13; unreachable: the 4 pairs with 4.
        assert_eq!(distances(&net), vec![4, 2, 0, 0, 4]);
        let directed = Network::from_edges(3, true, &[(0, 1), (1, 2)]).unwrap();
        assert_eq!(distances(&directed), vec![2, 1, 3]);
        assert_eq!(dspartners(&directed), vec![5, 1]);
    }
}
