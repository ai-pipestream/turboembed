//! Hashed embeddings of a word's text (`hash_embed` of a head): a word's
//! identity as the head reads it beside the encoder's states, with no
//! vocabulary to ship and something for every word the head never saw.
//!
//! After spaCy's `MultiHashEmbed` (MIT, read at v3.7.5): four features of
//! the word's text, each looked up in its own table by [`SEEDS`] hashes
//! of the feature and the rows summed (Bloom embeddings, so two features
//! rarely share all their rows), the four sums joined in [`FEATURES`]
//! order. The features:
//!
//! - NORM: the word lowercased (`char::to_lowercase`);
//! - PREFIX: its first character;
//! - SUFFIX: its last three characters (all of a shorter word);
//! - SHAPE: each letter as `X` (uppercase) or `x`, each numeric character
//!   as `d`, any other character as itself, a run of one shape character
//!   kept to its first four, and `LONG` for a word of 100 characters or
//!   more.
//!
//! A feature's key is the FNV-1a 64 hash of its UTF-8 bytes; hash k of
//! the key is the SplitMix64 finalizer of the key XOR `SEED_BASE` times
//! k + 1, taken modulo the table's rows. Lowercasing and the character
//! classes are the standard library's, so they follow the Unicode version
//! of the toolchain that builds the crate.

use crate::manifest::HashEmbed;
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result};

/// The features, in table and output order.
pub const FEATURES: [&str; 4] = ["norm", "prefix", "suffix", "shape"];
/// Hashes (and rows summed) per feature.
pub const SEEDS: usize = 4;
const SEED_BASE: u64 = 0x9e37_79b9_7f4a_7c15;

/// The tensor suffix of feature table k: `hash.<feature>`.
pub fn suffix(k: usize) -> String {
    format!("hash.{}", FEATURES[k])
}

/// The four features of `form`, in [`FEATURES`] order.
pub fn features(form: &str) -> [String; 4] {
    let chars: Vec<char> = form.chars().collect();
    let prefix = chars.first().map(|c| c.to_string()).unwrap_or_default();
    let suffix = chars[chars.len().saturating_sub(3)..].iter().collect();
    [form.to_lowercase(), prefix, suffix, shape(&chars)]
}

fn shape(chars: &[char]) -> String {
    if chars.len() >= 100 {
        return "LONG".into();
    }
    let mut out = String::with_capacity(chars.len());
    let (mut last, mut run) = (None, 0);
    for &c in chars {
        let s = if c.is_alphabetic() {
            if c.is_uppercase() { 'X' } else { 'x' }
        } else if c.is_numeric() {
            'd'
        } else {
            c
        };
        run = if Some(s) == last { run + 1 } else { 0 };
        last = Some(s);
        if run < 4 {
            out.push(s);
        }
    }
    out
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The rows of each feature table `form` reads, [`SEEDS`] per feature.
pub fn rows(h: &HashEmbed, form: &str) -> [[u32; SEEDS]; 4] {
    let f = features(form);
    std::array::from_fn(|k| {
        let key = fnv1a(&f[k]);
        std::array::from_fn(|s| (mix(key ^ SEED_BASE.wrapping_mul(s as u64 + 1)) % h.rows[k] as u64) as u32)
    })
}

/// Loaded tables: `tables[k]` is feature k's [rows[k], dim].
#[derive(Debug, Clone)]
pub struct Tables {
    spec: HashEmbed,
    tables: [Vec<f32>; 4],
}

impl Tables {
    /// The tables of `spec`, refused unless each holds rows[k] x dim
    /// finite values.
    pub fn new(spec: HashEmbed, tables: [Vec<f32>; 4]) -> Result<Tables> {
        for (k, t) in tables.iter().enumerate() {
            if t.len() as u64 != spec.rows[k] as u64 * spec.dim as u64 {
                return Err(Error::new(
                    INVALID_SHAPE,
                    format!("{} has {} values, expected {} x {}", suffix(k), t.len(), spec.rows[k], spec.dim),
                ));
            }
            if let Some(at) = t.iter().position(|v| !v.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, format!("{}[{at}] is {}, not finite", suffix(k), t[at])));
            }
        }
        Ok(Tables { spec, tables })
    }

    /// Values a word gets: four features of `dim`.
    pub fn width(&self) -> usize {
        4 * self.spec.dim as usize
    }

    /// The embedding of `form` into `out` (`width` values).
    pub fn embed(&self, form: &str, out: &mut [f32]) {
        let d = self.spec.dim as usize;
        out.fill(0.0);
        for (k, seeds) in rows(&self.spec, form).iter().enumerate() {
            let dst = &mut out[k * d..][..d];
            for &r in seeds {
                let row = &self.tables[k][r as usize * d..][..d];
                dst.iter_mut().zip(row).for_each(|(o, v)| *o += v);
            }
        }
    }
}
