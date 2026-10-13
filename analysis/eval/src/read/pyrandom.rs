//! Python's `random.seed(s)` for a string `s` and `random.random()`, as
//! CPython implements them (MT19937, seed version 2). LegalBench-RAG
//! picks its mini subset by sorting on one such draw per test, so the
//! subset can only be reproduced with the same generator.
//!
//! The seed: `int.from_bytes(s_utf8 + sha512(s_utf8), "big")`, split into
//! 32-bit words least significant first, with high zero words dropped,
//! and passed to MT19937's `init_by_array`. A draw takes two outputs,
//! `a >> 5` and `b >> 6`, and returns `(a * 2^26 + b) / 2^53`.

use turbo_goldens::sha::sha512;

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    i: usize,
}

impl PyRandom {
    /// `random.seed(s)` for a `str` `s`.
    pub fn from_str_seed(s: &str) -> PyRandom {
        let mut big = s.as_bytes().to_vec();
        big.extend_from_slice(&sha512(s.as_bytes()));
        // Big endian bytes to 32-bit words, least significant word first.
        let mut key: Vec<u32> = big
            .rchunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[4 - c.len()..].copy_from_slice(c);
                u32::from_be_bytes(w)
            })
            .collect();
        while key.len() > 1 && key.last() == Some(&0) {
            key.pop();
        }
        let mut r = PyRandom { mt: [0; N], i: N };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.i = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn next_u32(&mut self) -> u32 {
        if self.i >= N {
            for k in 0..N {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7fff_ffff);
                let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
                self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ mag;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.random()`: a float in `[0, 1)` with 53 random bits.
    pub fn random(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values CPython 3.13 printed for `random.seed(s)` followed by
    /// two `random.random()` calls.
    #[test]
    fn matches_cpython() {
        for (s, a, b) in [
            ("cuad/a.txt", 0.724149806000887, 0.04887750428019866),
            ("maud/Merger Agreement é.txt", 0.25459445736940867, 0.7043235853059594),
            ("", 0.9602256525641875, 0.595411957851699),
        ] {
            let mut r = PyRandom::from_str_seed(s);
            assert_eq!(r.random(), a, "{s:?}");
            assert_eq!(r.random(), b, "{s:?}");
        }
    }
}
