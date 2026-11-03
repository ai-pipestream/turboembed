//! Each word's features from its subword states, pooled as the core's
//! word reader pools them (`turbo::head::words`): the word's first row,
//! or the mean of its rows, under a word map that names each row's word
//! or [`NO_WORD`]; and the gradient back to the rows.

use turbo::head::tagger::{NO_WORD, word_count};
use turbo::manifest::WordPooling;
use turbo::status::Result;

use crate::real::Real;

/// features [n, hidden] from states [rows, hidden] under `words` [rows],
/// with each word's row count kept in `count` for backward; returns n.
/// Refuses the word maps the core refuses (`word_count`).
pub fn forward<R: Real>(
    states: &[R],
    words: &[u32],
    hidden: usize,
    pooling: WordPooling,
    features: &mut Vec<R>,
    count: &mut Vec<u32>,
) -> Result<usize> {
    debug_assert_eq!(states.len(), words.len() * hidden);
    let n = word_count(words)?;
    features.clear();
    features.resize(n * hidden, R::ZERO);
    count.clear();
    count.resize(n, 0);
    for (r, &word) in words.iter().enumerate() {
        if word == NO_WORD {
            continue;
        }
        let i = word as usize;
        let (row, dst) = (&states[r * hidden..][..hidden], &mut features[i * hidden..][..hidden]);
        match pooling {
            WordPooling::First if count[i] == 0 => dst.copy_from_slice(row),
            WordPooling::First => {}
            WordPooling::Mean => dst.iter_mut().zip(row).for_each(|(d, &s)| *d += s),
        }
        count[i] += 1;
    }
    if pooling == WordPooling::Mean {
        for (i, &c) in count.iter().enumerate() {
            let inv = R::ONE / R::from_f64(c as f64);
            features[i * hidden..][..hidden].iter_mut().for_each(|v| *v *= inv);
        }
    }
    Ok(n)
}

/// dstates [rows, hidden] += the gradient of `forward` from dfeatures
/// [n, hidden], under the same `words` and the `count` forward kept.
pub fn backward<R: Real>(
    dfeatures: &[R],
    words: &[u32],
    hidden: usize,
    pooling: WordPooling,
    count: &[u32],
    dstates: &mut [R],
) {
    debug_assert_eq!(dstates.len(), words.len() * hidden);
    let mut seen = vec![false; count.len()];
    for (r, &word) in words.iter().enumerate() {
        if word == NO_WORD {
            continue;
        }
        let i = word as usize;
        let (src, dst) = (&dfeatures[i * hidden..][..hidden], &mut dstates[r * hidden..][..hidden]);
        match pooling {
            WordPooling::First if !seen[i] => dst.iter_mut().zip(src).for_each(|(d, &s)| *d += s),
            WordPooling::First => {}
            WordPooling::Mean => {
                let inv = R::ONE / R::from_f64(count[i] as f64);
                dst.iter_mut().zip(src).for_each(|(d, &s)| *d += s * inv);
            }
        }
        seen[i] = true;
    }
}
