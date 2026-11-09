//! SHA-512 (FIPS 180-4): 128-byte blocks, 64-bit words. The zoo and the
//! goldens record a file's SHA-512 beside its SHA-256, and the evaluation
//! sets seed Python's `random` with it.

/// The round constants: the first 64 bits of the fractional parts of the
/// cube roots of the first 80 primes.
const K512: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// The initial hash value: the first 64 bits of the fractional parts of
/// the square roots of the first eight primes.
const H512: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// The SHA-512 digest of `data`.
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finish()
}

/// An incremental SHA-512: [`update`](Sha512::update) any number of times,
/// then [`finish`](Sha512::finish).
#[derive(Clone, Debug)]
pub struct Sha512 {
    state: [u64; 8],
    block: [u8; 128],
    filled: usize,
    total: u64,
}

impl Default for Sha512 {
    fn default() -> Self {
        Sha512::new()
    }
}

impl Sha512 {
    /// A SHA-512 state at the FIPS 180-4 initial value.
    pub const fn new() -> Sha512 {
        Sha512 { state: H512, block: [0; 128], filled: 0, total: 0 }
    }

    /// Feeds `data` into the state.
    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let n = (128 - self.filled).min(data.len());
            self.block[self.filled..self.filled + n].copy_from_slice(&data[..n]);
            self.filled += n;
            data = &data[n..];
            if self.filled < 128 {
                return self;
            }
            let block = self.block;
            compress512(&mut self.state, &block);
            self.filled = 0;
        }
        let (blocks, rest) = data.as_chunks::<128>();
        for b in blocks {
            compress512(&mut self.state, b);
        }
        self.block[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
        self
    }

    /// Pads the message and returns its 64-byte digest.
    pub fn finish(mut self) -> [u8; 64] {
        let bits = (self.total as u128).wrapping_mul(8);
        let mut pad = [0u8; 144];
        pad[0] = 0x80;
        let zeros = (111usize.wrapping_sub(self.filled)) % 128;
        let n = 1 + zeros;
        pad[n..n + 16].copy_from_slice(&bits.to_be_bytes());
        self.update(&pad[..n + 16]);
        debug_assert_eq!(self.filled, 0);
        let mut out = [0u8; 64];
        for (o, w) in out.as_chunks_mut::<8>().0.iter_mut().zip(self.state) {
            *o = w.to_be_bytes();
        }
        out
    }
}

fn compress512(state: &mut [u64; 8], block: &[u8; 128]) {
    // The message schedule as a ring of sixteen words, each replaced by
    // `W[t]` just before round `t` reads it.
    let mut w = [0u64; 16];
    for (wi, word) in w.iter_mut().zip(block.as_chunks::<8>().0) {
        *wi = u64::from_be_bytes(*word);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    // Round t writes what the next round reads as `a` into `h` and its
    // `e` into `d`, so with the names turned by one each round no word
    // moves.
    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $base:expr, $j:expr) => {
            if $base > 0 {
                let w15 = w[($j + 1) & 15];
                let w2 = w[($j + 14) & 15];
                let s0 = w15.rotate_right(1) ^ w15.rotate_right(8) ^ (w15 >> 7);
                let s1 = w2.rotate_right(19) ^ w2.rotate_right(61) ^ (w2 >> 6);
                w[$j] = w[$j].wrapping_add(s0).wrapping_add(w[($j + 9) & 15]).wrapping_add(s1);
            }
            let s1 = $e.rotate_right(14) ^ $e.rotate_right(18) ^ $e.rotate_right(41);
            let ch = $g ^ ($e & ($f ^ $g));
            let t1 = $h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K512[$base + $j]).wrapping_add(w[$j]);
            let s0 = $a.rotate_right(28) ^ $a.rotate_right(34) ^ $a.rotate_right(39);
            let maj = ($a & $b) | ($c & ($a | $b));
            $d = $d.wrapping_add(t1);
            $h = t1.wrapping_add(s0.wrapping_add(maj));
        };
    }
    for base in (0..80).step_by(16) {
        round!(a, b, c, d, e, f, g, h, base, 0);
        round!(h, a, b, c, d, e, f, g, base, 1);
        round!(g, h, a, b, c, d, e, f, base, 2);
        round!(f, g, h, a, b, c, d, e, base, 3);
        round!(e, f, g, h, a, b, c, d, base, 4);
        round!(d, e, f, g, h, a, b, c, base, 5);
        round!(c, d, e, f, g, h, a, b, base, 6);
        round!(b, c, d, e, f, g, h, a, base, 7);
        round!(a, b, c, d, e, f, g, h, base, 8);
        round!(h, a, b, c, d, e, f, g, base, 9);
        round!(g, h, a, b, c, d, e, f, base, 10);
        round!(f, g, h, a, b, c, d, e, base, 11);
        round!(e, f, g, h, a, b, c, d, base, 12);
        round!(d, e, f, g, h, a, b, c, base, 13);
        round!(c, d, e, f, g, h, a, b, base, 14);
        round!(b, c, d, e, f, g, h, a, base, 15);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}
