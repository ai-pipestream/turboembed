//! The SHA-256 block function. On x86-64 with the SHA extensions and on
//! AArch64 with the SHA-2 crypto extension it runs on those instructions,
//! chosen once per call by a cached CPU feature check; elsewhere it runs
//! the plain FIPS 180-4 rounds. Every path gives the same state.

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The round constants: the first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

/// Runs the compression function over each of `blocks` in order.
pub(crate) fn blocks(state: &mut [u32; 8], blocks: &[[u8; 64]]) {
    if blocks.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if x86::available() {
        // SAFETY: `available` saw every feature `x86::blocks` enables.
        unsafe { x86::blocks(state, blocks) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if arm::available() {
        // SAFETY: `available` saw every feature `arm::blocks` enables.
        unsafe { arm::blocks(state, blocks) };
        return;
    }
    portable(state, blocks);
}

/// The plain rounds over each of `blocks`.
fn portable(state: &mut [u32; 8], blocks: &[[u8; 64]]) {
    for block in blocks {
        portable_block(state, block);
    }
}

/// One block in plain integer code, for a CPU without SHA instructions.
fn portable_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (wi, bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) {
        *wi = u32::from_be_bytes(*bytes);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes that are not a short period, so a misordered word shows.
    fn noise(n: usize) -> Vec<u8> {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn the_cpu_path_is_the_portable_rounds() {
        #[cfg(target_arch = "x86_64")]
        eprintln!("x86 SHA extensions: {}", x86::available());
        #[cfg(target_arch = "aarch64")]
        eprintln!("AArch64 SHA-2 extension: {}", arm::available());
        let data = noise(64 * 37);
        let (all, _) = data.as_chunks::<64>();
        for n in [1, 2, 3, 37] {
            let (mut fast, mut slow) = ([0x6a09e667u32, 1, 2, 3, 4, 5, 6, 7], [0x6a09e667u32, 1, 2, 3, 4, 5, 6, 7]);
            blocks(&mut fast, &all[..n]);
            portable(&mut slow, &all[..n]);
            assert_eq!(fast, slow, "{n} blocks");
        }
    }
}
