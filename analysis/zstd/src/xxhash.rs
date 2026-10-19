//! XXH64, the hash whose low 32 bits are a frame's content checksum, as
//! the xxHash specification (`doc/xxhash_spec.md` in Cyan4973/xxHash)
//! defines it.

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

fn u64_at(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}

fn u32_at(b: &[u8]) -> u64 {
    u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

fn merge(acc: u64, v: u64) -> u64 {
    (acc ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4)
}

/// The four accumulators' starting values.
fn start(seed: u64) -> [u64; 4] {
    [seed.wrapping_add(P1).wrapping_add(P2), seed.wrapping_add(P2), seed, seed.wrapping_sub(P1)]
}

#[inline(always)]
fn stripe(v: &mut [u64; 4], s: &[u8; 32]) {
    for (i, v) in v.iter_mut().enumerate() {
        *v = round(*v, u64_at(&s[8 * i..]));
    }
}

/// The accumulators folded into one value, for inputs of at least one
/// stripe.
fn converge(v: &[u64; 4]) -> u64 {
    let h = v[0].rotate_left(1).wrapping_add(v[1].rotate_left(7)).wrapping_add(v[2].rotate_left(12));
    let h = h.wrapping_add(v[3].rotate_left(18));
    v.iter().fold(h, |h, &v| merge(h, v))
}

/// The hash from `h`, which counts the input's length, and the bytes past
/// its last stripe.
fn tail(mut h: u64, mut rest: &[u8]) -> u64 {
    while rest.len() >= 8 {
        h = (h ^ round(0, u64_at(rest))).rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        rest = &rest[8..];
    }
    if rest.len() >= 4 {
        h = (h ^ u32_at(rest).wrapping_mul(P1)).rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        rest = &rest[4..];
    }
    for &b in rest {
        h = (h ^ u64::from(b).wrapping_mul(P5)).rotate_left(11).wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

/// The XXH64 hash of `data` with `seed`, in one call; the decoder uses
/// [`Xxh64`], which the tests check against this.
#[cfg(test)]
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    let (stripes, rest) = data.as_chunks::<32>();
    let h = if stripes.is_empty() {
        seed.wrapping_add(P5)
    } else {
        let mut v = start(seed);
        for s in stripes {
            stripe(&mut v, s);
        }
        converge(&v)
    };
    tail(h.wrapping_add(data.len() as u64), rest)
}

/// XXH64 of input given in pieces, so a frame's content is hashed block
/// by block as it is decoded: the hash of the pieces joined.
pub struct Xxh64 {
    seed: u64,
    v: [u64; 4],
    /// The bytes given since the last whole stripe, fewer than 32.
    buf: [u8; 32],
    buf_len: usize,
    len: u64,
}

impl Xxh64 {
    pub fn new(seed: u64) -> Xxh64 {
        Xxh64 { seed, v: start(seed), buf: [0; 32], buf_len: 0, len: 0 }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if self.buf_len > 0 {
            let n = (32 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + n].copy_from_slice(&data[..n]);
            self.buf_len += n;
            data = &data[n..];
            if self.buf_len < 32 {
                return;
            }
            stripe(&mut self.v, &self.buf);
            self.buf_len = 0;
        }
        let (stripes, rest) = data.as_chunks::<32>();
        for s in stripes {
            stripe(&mut self.v, s);
        }
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    pub fn finish(&self) -> u64 {
        let h = if self.len >= 32 { converge(&self.v) } else { self.seed.wrapping_add(P5) };
        tail(h.wrapping_add(self.len), &self.buf[..self.buf_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test buffer of xxHash's own sanity checks: each byte the top
    /// byte of a generator that starts at 2654435761 and is multiplied by
    /// 11400714785074694797 after each byte.
    fn sanity_buffer(len: usize) -> Vec<u8> {
        let mut next: u64 = 2_654_435_761;
        (0..len)
            .map(|_| {
                let b = (next >> 56) as u8;
                next = next.wrapping_mul(11_400_714_785_074_694_797);
                b
            })
            .collect()
    }

    /// The hash of the input given in two or three pieces, split at every
    /// point of the first 100 bytes and at stripe edges further on, is the
    /// hash of the whole.
    #[test]
    fn pieces_hash_as_the_whole() {
        let buf = sanity_buffer(300);
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 100, 222, 300] {
            let data = &buf[..len];
            let whole = xxh64(data, 7);
            let pieces = |cuts: &[usize]| {
                let mut x = Xxh64::new(7);
                let mut at = 0;
                for &c in cuts {
                    x.update(&data[at..c]);
                    at = c;
                }
                x.update(&data[at..]);
                x.finish()
            };
            assert_eq!(pieces(&[]), whole, "{len} bytes");
            for a in 0..=len.min(100) {
                assert_eq!(pieces(&[a]), whole, "{len} bytes cut at {a}");
                for b in [a, a + 1, a + 31, a + 32, a + 33, len].into_iter().filter(|&b| b <= len && b >= a) {
                    assert_eq!(pieces(&[a, b]), whole, "{len} bytes cut at {a} and {b}");
                }
            }
        }
        // One byte at a time.
        let mut x = Xxh64::new(0);
        for b in &buf[..222] {
            x.update(std::slice::from_ref(b));
        }
        assert_eq!(x.finish(), 0xB641_AE8C_B691_C174);
    }

    #[test]
    fn published_vectors() {
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"a", 0), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
        assert_eq!(xxh64(b"xxhash", 0), 0x32DD_3895_2C4B_C720);
        assert_eq!(xxh64(b"xxhash", 20_141_025), 0xB559_B98D_844E_0635);
    }

    #[test]
    fn xxhash_sanity_vectors() {
        // From xxHash's sanity check, seeds 0 and 2654435761, lengths
        // that reach every tail: none, bytes, a 4-byte word, 8-byte words
        // and 32-byte stripes.
        let buf = sanity_buffer(222);
        let seed = 2_654_435_761;
        assert_eq!(xxh64(&buf[..0], seed), 0xAC75_FDA2_929B_17EF);
        assert_eq!(xxh64(&buf[..1], 0), 0xE934_A84A_DB05_2768);
        assert_eq!(xxh64(&buf[..1], seed), 0x5014_6076_43A9_B4C3);
        assert_eq!(xxh64(&buf[..4], 0), 0x9136_A0DC_A574_57EE);
        assert_eq!(xxh64(&buf[..14], 0), 0x8282_DCC4_994E_35C8);
        assert_eq!(xxh64(&buf[..14], seed), 0xC3BD_6BF6_3DEB_6DF0);
        assert_eq!(xxh64(&buf[..222], 0), 0xB641_AE8C_B691_C174);
        assert_eq!(xxh64(&buf[..222], seed), 0x20CB_8AB7_AE10_C14A);
    }
}
