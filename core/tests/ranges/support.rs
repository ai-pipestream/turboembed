//! Helpers the range tests share: a session taking `max_ranges` ranges,
//! the option shortcuts, the texts, tokenization through the bundle, and
//! the runs.

use std::path::Path;

use crate::common::*;
use turbo::*;

/// Three texts of different lengths, the shortest still five tokens, as
/// the range cases ask of their rows.
pub const RANGES_TEXTS: [&str; 3] =
    ["The quick brown fox jumps over the lazy dog. It was a sunny day.", "how do I reset a password", "a b c"];

/// A session of the model's size that takes `max_ranges` ranges.
pub fn session_of(l: &Loaded, max_ranges: u32) -> Session {
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = max_ranges;
    Session::create(l.m, Some(&d)).unwrap_or_else(|e| panic!("{e:?}"))
}

pub fn normalized(normalize: u32) -> turbo_embed_options {
    opts(|o| o.normalize = normalize)
}

/// `texts` tokenized with the bundle in `dir` and padded on the right, and
/// each row's length.
pub fn tokens_of(dir: &Path, texts: &[&str]) -> (Tokens, Vec<usize>) {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = texts.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let lens = rows.iter().map(Vec::len).collect();
    (Tokens::new(&rows, 0), lens)
}

/// Row `r` of `t`: its ids, its mask and its types (zero when `t` has none).
pub fn row_of(t: &Tokens, r: usize) -> (&[i32], &[i32], Vec<i32>) {
    let seq = t.seq as usize;
    let at = r * seq..(r + 1) * seq;
    let types = t.types.as_ref().map_or(vec![0; seq], |v| v[at.clone()].to_vec());
    (&t.ids[at.clone()], &t.mask[at], types)
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

pub fn run_ranges(s: &Session, t: &Tokens, ranges: &[turbo_token_range], o: &turbo_embed_options) -> Vec<Vec<f32>> {
    s.write_ranges(&t.batch(), ranges, Some(o)).unwrap_or_else(|e| panic!("{e:?}"));
    s.run().unwrap_or_else(|e| panic!("{e:?}")).rows()
}
