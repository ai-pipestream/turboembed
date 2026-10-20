//! The hashes behind a sketch: words, word sequences, shingles, and the
//! MinHash and SimHash of a set of shingles. Every constant is fixed, so
//! a sketch is the same on every machine and in every process, and two
//! documents sketched apart can be compared.

/// MinHash values per item.
pub const LANES: usize = 64;
/// Lanes per band in the banding that finds candidate pairs: 16 bands of
/// 4 lanes. Two items of Jaccard similarity `j` share a band with
/// probability `1 - (1 - j^4)^16`: 0.51 at 0.4, 0.88 at 0.6, 0.996 at
/// 0.8.
pub const ROWS: usize = 4;
pub const BANDS: usize = LANES / ROWS;
const _: () = assert!(BANDS * ROWS == LANES);

/// The hash every number word becomes in a shape (see `Words::shape`).
pub const NUMBER: u64 = 0x6e75_6d62_6572_0001;

const P1: u64 = 0x9e37_79b9_7f4a_7c15;
const P2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const P3: u64 = 0x1656_67b1_9e37_79f9;

/// The splitmix64 step: a bijection on u64 with full avalanche.
pub const fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(P1);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The odd multipliers of the MinHash lanes: lane `i` hashes a shingle
/// `s` to the high 32 bits of `s * MULT[i]`, a multiply-shift hash.
pub const MULT: [u64; LANES] = {
    let mut m = [0u64; LANES];
    let mut i = 0;
    while i < LANES {
        m[i] = splitmix(0x7475_7262_6f67_7270 ^ (i as u64).wrapping_mul(P2)) | 1;
        i += 1;
    }
    m
};

/// A word's bytes to 64 bits, eight bytes at a time.
#[inline]
pub fn word(bytes: &[u8]) -> u64 {
    let mut h = (bytes.len() as u64).wrapping_mul(P3);
    let (chunks, rest) = bytes.as_chunks::<8>();
    for c in chunks {
        h = (h ^ u64::from_le_bytes(*c)).wrapping_mul(P1).rotate_left(31);
    }
    if !rest.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        h = (h ^ u64::from_le_bytes(tail)).wrapping_mul(P2).rotate_left(29);
    }
    splitmix(h)
}

/// One step of an order-sensitive fold of hashes.
#[inline]
pub fn fold(h: u64, x: u64, p: u64) -> u64 {
    (h ^ x).wrapping_mul(p).rotate_left(27)
}

/// The 128-bit hash of a sequence of word hashes: two folds with
/// different multipliers.
pub fn sequence(words: &[u64]) -> u128 {
    let (mut a, mut b) = (P3, P2);
    for &w in words {
        a = fold(a, w, P1);
        b = fold(b, w, P2);
    }
    let n = words.len() as u64;
    (u128::from(splitmix(a ^ n)) << 64) | u128::from(splitmix(b ^ n.rotate_left(32)))
}

/// The 64-bit hash of a sequence of word hashes.
pub fn sequence64(words: &[u64]) -> u64 {
    splitmix(words.iter().fold(P3 ^ words.len() as u64, |h, &w| fold(h, w, P1)))
}

const SEP: u8 = 0;
const ALNUM: u8 = 1;
const HIGH: u8 = 2;

/// Each byte's class: an ASCII letter or digit, a byte of a character
/// past ASCII, or anything else.
const CLASS: [u8; 256] = {
    let mut t = [SEP; 256];
    let mut b = 0;
    while b < 256 {
        let c = b as u8;
        t[b] = if c >= 0x80 {
            HIGH
        } else if c.is_ascii_alphanumeric() {
            ALNUM
        } else {
            SEP
        };
        b += 1;
    }
    t
};

/// Calls `f` with each word of `text` in order: a maximal run of
/// characters `char::is_alphanumeric` accepts, so whitespace and
/// punctuation separate words and are not part of any. ASCII is
/// classified byte by byte from a table; other characters are decoded.
#[inline]
pub fn words(text: &str, mut f: impl FnMut(&str)) {
    let b = text.as_bytes();
    let n = b.len();
    // Whether the character at `i`, which starts past ASCII, is a word
    // character, and its length.
    let high = |i: usize| {
        let c = text[i..].chars().next().expect("a character at a char boundary");
        (c.is_alphanumeric(), c.len_utf8())
    };
    let mut i = 0;
    while i < n {
        match CLASS[b[i] as usize] {
            SEP => {
                i += 1;
                continue;
            }
            HIGH => {
                let (alnum, len) = high(i);
                if !alnum {
                    i += len;
                    continue;
                }
            }
            _ => {}
        }
        let start = i;
        loop {
            while i < n && CLASS[b[i] as usize] == ALNUM {
                i += 1;
            }
            if i < n && CLASS[b[i] as usize] == HIGH {
                let (alnum, len) = high(i);
                if alnum {
                    i += len;
                    continue;
                }
            }
            break;
        }
        f(&text[start..i]);
    }
}

/// Whether a word is a number: every character numeric.
#[inline]
pub fn is_number(w: &str) -> bool {
    if w.is_ascii() { w.bytes().all(|b| b.is_ascii_digit()) } else { w.chars().all(char::is_numeric) }
}

/// The shingles of a sequence of word hashes: each run of `width`
/// consecutive words, or the whole sequence as one shingle when it is
/// shorter. None for no words.
pub fn shingles(words: &[u64], width: usize, mut f: impl FnMut(u64)) {
    if words.is_empty() {
        return;
    }
    if words.len() <= width {
        f(sequence64(words));
        return;
    }
    for w in words.windows(width) {
        f(sequence64(w));
    }
}

/// The MinHash of a set of shingles: lane by lane, the least hash. Lane
/// `i` hashes shingle `s` to the high half of `s * MULT[i]`, 64 hashes
/// per shingle. The arithmetic is the same on every machine; on x86-64
/// with AVX-512 or AVX2 the lanes run as vectors, chosen once per item.
pub fn minhash(shingles: &[u64]) -> [u32; LANES] {
    let mut lanes = [u32::MAX; LANES];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512dq") {
            // SAFETY: the CPU has the features the function is built for.
            unsafe { mins_avx512(&mut lanes, shingles) };
            return lanes;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: as above.
            unsafe { mins_avx2(&mut lanes, shingles) };
            return lanes;
        }
    }
    mins(&mut lanes, shingles);
    lanes
}

#[inline(always)]
fn mins(lanes: &mut [u32; LANES], shingles: &[u64]) {
    for &s in shingles {
        for (m, &k) in lanes.iter_mut().zip(&MULT) {
            *m = (*m).min((s.wrapping_mul(k) >> 32) as u32);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
fn mins_avx512(lanes: &mut [u32; LANES], shingles: &[u64]) {
    mins(lanes, shingles)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn mins_avx2(lanes: &mut [u32; LANES], shingles: &[u64]) {
    mins(lanes, shingles)
}

/// The SimHash of a bag of shingles: bit b is set when more shingles have
/// it set than clear. Bits are counted eight at a time, a byte of the
/// shingle at once into eight byte-wide counters, flushed every 255
/// shingles.
pub struct SimHash {
    counts: [u32; 64],
    bytes: [u64; 8],
    pending: u32,
    n: u32,
}

impl Default for SimHash {
    fn default() -> SimHash {
        SimHash { counts: [0; 64], bytes: [0; 8], pending: 0, n: 0 }
    }
}

/// Byte k of `SPREAD[x]` is bit k of x.
const SPREAD: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut x = 0;
    while x < 256 {
        let mut k = 0;
        while k < 8 {
            t[x] |= ((x as u64 >> k) & 1) << (8 * k);
            k += 1;
        }
        x += 1;
    }
    t
};

impl SimHash {
    #[inline]
    pub fn add(&mut self, s: u64) {
        for (j, b) in self.bytes.iter_mut().enumerate() {
            *b += SPREAD[(s >> (8 * j)) as usize & 0xff];
        }
        self.pending += 1;
        self.n += 1;
        if self.pending == 255 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        for (j, b) in self.bytes.iter_mut().enumerate() {
            for k in 0..8 {
                self.counts[8 * j + k] += (*b >> (8 * k)) as u32 & 0xff;
            }
            *b = 0;
        }
        self.pending = 0;
    }

    pub fn finish(&mut self) -> u64 {
        self.flush();
        let n = u64::from(self.n);
        self.counts.iter().enumerate().fold(0, |h, (b, &c)| h | (u64::from(2 * u64::from(c) > n) << b))
    }
}

/// The estimated Jaccard similarity of two MinHash rows: the share of
/// lanes that agree.
pub fn jaccard(a: &[u32], b: &[u32]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

/// The key of one band of a MinHash row.
pub fn band(row: &[u32], b: usize) -> u64 {
    let lanes = &row[b * ROWS..(b + 1) * ROWS];
    splitmix(lanes.iter().fold(b as u64, |h, &x| fold(h, u64::from(x), P2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_words(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        words(text, |w| out.push(w.to_owned()));
        out
    }

    #[test]
    fn words_are_alphanumeric_runs_in_any_script() {
        assert_eq!(
            all_words("Hello, world! v2.0 naïve 日本語 x"),
            ["Hello", "world", "v2", "0", "naïve", "日本語", "x"]
        );
        assert_eq!(all_words("  ...  "), Vec::<String>::new());
        assert_eq!(all_words("end"), ["end"]);
        assert_eq!(all_words("é…a·b¹²"), ["é", "a", "b¹²"]);
    }

    #[test]
    fn numbers_are_digit_runs() {
        assert!(is_number("2024") && is_number("٣") && !is_number("v2") && !is_number("x"));
    }

    #[test]
    fn word_hashes_see_every_byte_and_the_length() {
        assert_ne!(word(b"abcdefgh"), word(b"abcdefgi"));
        assert_ne!(word(b"abcdefghi"), word(b"abcdefghj"));
        assert_ne!(word(b"a"), word(b"a\0"));
        assert_eq!(word(b"same"), word(b"same"));
    }

    #[test]
    fn sequences_are_order_sensitive() {
        let (a, b) = (word(b"a"), word(b"b"));
        assert_ne!(sequence(&[a, b]), sequence(&[b, a]));
        assert_ne!(sequence(&[a]), sequence(&[a, a]));
        assert_ne!(sequence64(&[a, b]), sequence64(&[b, a]));
    }

    #[test]
    fn short_sequences_make_one_shingle() {
        let ws = [1, 2, 3, 4].map(splitmix);
        let mut n = 0;
        shingles(&ws, 3, |_| n += 1);
        assert_eq!(n, 2);
        n = 0;
        shingles(&ws[..2], 3, |_| n += 1);
        assert_eq!(n, 1);
        n = 0;
        shingles(&[], 3, |_| n += 1);
        assert_eq!(n, 0);
    }

    fn sketch(range: std::ops::Range<u64>) -> [u32; LANES] {
        minhash(&range.map(splitmix).collect::<Vec<_>>())
    }

    #[test]
    fn minhash_estimates_jaccard_of_overlapping_sets() {
        // The true Jaccard similarities are 500 / 1500 and 900 / 1100.
        let (a, b, c) = (sketch(0..1000), sketch(500..1500), sketch(100..1100));
        let (ab, ac) = (jaccard(&a, &b), jaccard(&a, &c));
        assert!((ab - 1.0 / 3.0).abs() < 0.15, "{ab}");
        assert!((ac - 9.0 / 11.0).abs() < 0.12, "{ac}");
        assert_eq!(jaccard(&a, &a), 1.0);
        assert_eq!(sketch(0..0), [u32::MAX; LANES]);
    }

    #[test]
    fn every_lane_path_computes_the_same_values() {
        let shingles: Vec<u64> = (0..300).map(splitmix).collect();
        let mut plain = [u32::MAX; LANES];
        mins(&mut plain, &shingles);
        assert_eq!(minhash(&shingles), plain);
        #[cfg(target_arch = "x86_64")]
        {
            let mut v = [u32::MAX; LANES];
            if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512dq") {
                // SAFETY: the CPU has the features.
                unsafe { mins_avx512(&mut v, &shingles) };
                assert_eq!(v, plain, "AVX-512");
            }
            v = [u32::MAX; LANES];
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: the CPU has the feature.
                unsafe { mins_avx2(&mut v, &shingles) };
                assert_eq!(v, plain, "AVX2");
            }
        }
    }

    #[test]
    fn the_lane_multipliers_are_odd_and_distinct() {
        let mut m = MULT.to_vec();
        assert!(m.iter().all(|x| x & 1 == 1));
        m.sort_unstable();
        m.dedup();
        assert_eq!(m.len(), LANES);
    }

    #[test]
    fn simhash_follows_the_majority_across_flushes() {
        let mut s = SimHash::default();
        s.add(0b101);
        s.add(0b100);
        s.add(0b001);
        assert_eq!(s.finish(), 0b101);
        // 300 shingles with bit 3, 200 without: the counters flush.
        let mut s = SimHash::default();
        (0..300).for_each(|_| s.add(1 << 3));
        (0..200).for_each(|_| s.add(1 << 9));
        assert_eq!(s.finish(), 1 << 3);
    }
}
