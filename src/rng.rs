//! xoshiro256** pseudo-random numbers, seeded with splitmix64.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(mut seed: u64) -> Self {
        let mut s = [0u64; 4];
        for x in &mut s {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *x = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n). The modulo bias is below n / 2^53.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 11) % n
    }
}
