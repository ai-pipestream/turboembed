//! What the decoder's tests share: the originals the fixtures in
//! `tests/data` were compressed from, rebuilt from seeded generators (the
//! `originals` example writes them out for the `zstd` tool), and the
//! fixtures themselves.

#![allow(dead_code)]

pub mod build;

use std::path::Path;

/// A small linear congruential generator, so the originals are the same
/// on every run.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Made-up words: letters, a few with accents or in Greek and Cyrillic
/// so the text has multibyte characters, and some digits.
fn vocabulary(rng: &mut Lcg, n: usize) -> Vec<String> {
    const LETTERS: &[&str] = &[
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
        "w", "x", "y", "z", "e", "a", "o", "i", "n", "t", "s", "r",
    ];
    const RARE: &[&str] = &["é", "ü", "ß", "ñ", "ø", "α", "λ", "ж", "я", "ç"];
    (0..n)
        .map(|i| {
            let len = 1 + rng.below(9);
            let mut w = String::new();
            for _ in 0..len {
                if rng.below(40) == 0 {
                    w.push_str(RARE[rng.below(RARE.len())]);
                } else {
                    w.push_str(LETTERS[rng.below(LETTERS.len())]);
                }
            }
            if i % 97 == 0 {
                w = format!("{}", rng.below(100_000));
            }
            w
        })
        .collect()
}

/// About `len` bytes of text-like data: sentences of words drawn from a
/// vocabulary of 3000, the common ones far more often, with punctuation
/// and paragraphs.
pub fn text(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let words = vocabulary(&mut rng, 3000);
    let mut out = String::with_capacity(len + 100);
    while out.len() < len {
        let n = 4 + rng.below(16);
        for i in 0..n {
            // Close to Zipf's law: the index's logarithm is uniform.
            let r = rng.below(1 << 20) as f64 / (1 << 20) as f64;
            let w = &words[((words.len() as f64).powf(r) as usize).saturating_sub(1)];
            if i == 0 {
                let mut c = w.chars();
                out.extend(c.next().map(|c| c.to_ascii_uppercase()));
                out.push_str(c.as_str());
            } else {
                out.push_str(w);
            }
            if i + 1 < n {
                out.push_str(if rng.below(8) == 0 { ", " } else { " " });
            }
        }
        out.push_str([". ", "? ", "! ", ".\n\n"][rng.below(4)]);
    }
    out.truncate(out.floor_char_boundary(len));
    out.into_bytes()
}

/// `len` bytes from the generator, which no compressor shortens.
pub fn random(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.next() as u8).collect()
}

/// Long runs and text whose only new bytes are spaces: 300,000 zeros
/// (RLE blocks), runs of each letter of growing length, then a text
/// twice, the second time with spaces put in, so the second copy's
/// literals are all spaces (RLE literals).
pub fn runs() -> Vec<u8> {
    let mut out = vec![0u8; 300_000];
    for (i, c) in (b'a'..=b'z').enumerate() {
        out.extend(std::iter::repeat_n(c, 1000 * (i + 1)));
    }
    let t = text(20_000, 7);
    out.extend(&t);
    let mut rng = Lcg(8);
    for (i, &b) in t.iter().enumerate() {
        out.push(b);
        if i % 50 == 0 && rng.below(2) == 0 {
            out.extend(b"   ");
        }
    }
    out
}

/// A text, 3 MB of zeros, and the same text again: the second copy is
/// further back than a normal window reaches, which long distance
/// matching finds.
pub fn long() -> Vec<u8> {
    let t = text(100_000, 11);
    let mut out = t.clone();
    out.extend(std::iter::repeat_n(0u8, 3 << 20));
    out.extend(&t);
    out
}

/// Every original a fixture was made from, by file name.
pub fn originals() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", Vec::new()),
        ("byte", b"a".to_vec()),
        ("text", text(150_000, 1)),
        ("small", text(40_000, 2)),
        ("random", random(20_000, 3)),
        ("runs", runs()),
        ("long", long()),
    ]
}

/// The original named `name`.
pub fn original(name: &str) -> Vec<u8> {
    originals().into_iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("no original {name}")).1
}

/// The fixture `name` in `tests/data`.
pub fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}
