//! What every word head reads first, whatever it scores: each word's
//! subword states pooled (its first row, or their mean), then the
//! optional BiLSTM adapter over the words, each layer reading the one
//! before and joining its two directions. The token tagger (tagger.rs)
//! and the biaffine parser (parser.rs) share it, so the same spec gives
//! the same word vectors to both.

use crate::cpu::lstm::{Lstm, Weights, Work as LstmWork};
use crate::manifest::{HeadSpec, WordPooling};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

/// A subword row that belongs to no word (a special token), in a word map.
pub const NO_WORD: u32 = u32::MAX;

/// The adapter's tensors as suffixes and shapes, and the width of the
/// word vectors it gives (`hidden` without an adapter).
pub(crate) fn tensors(spec: &HeadSpec, hidden: usize) -> (Vec<(String, Vec<u64>)>, u64) {
    let mut out = Vec::new();
    let mut width = hidden as u64;
    if let Some(a) = &spec.adapter {
        let h = a.bilstm.hidden as u64;
        for k in 0..a.bilstm.layers {
            for dir in ["fwd", "bwd"] {
                out.push((format!("lstm.{k}.{dir}.w"), vec![4 * h, width]));
                out.push((format!("lstm.{k}.{dir}.u"), vec![4 * h, h]));
                out.push((format!("lstm.{k}.{dir}.b"), vec![4 * h]));
            }
            width = 2 * h;
        }
    }
    (out, width)
}

/// Each tensor's value count, checked against overflow, so a width past
/// what memory could hold is refused before anything is read.
pub(crate) fn sizes(spec: &HeadSpec, shapes: &[(String, Vec<u64>)]) -> Result<Vec<(String, usize)>> {
    shapes
        .iter()
        .map(|(suffix, dims)| {
            let n = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).and_then(|n| usize::try_from(n).ok());
            n.map(|n| (suffix.clone(), n)).ok_or_else(|| {
                Error::new(INVALID_SHAPE, format!("head {:?}: {suffix} {dims:?} is too large", spec.name))
            })
        })
        .collect()
}

/// Gives each tensor by suffix, checked to hold the count `sizes` gave.
pub(crate) struct Take<'a> {
    pub name: &'a str,
    pub sizes: Vec<(String, usize)>,
    pub get: &'a dyn Fn(&str) -> Option<Vec<f32>>,
}

impl Take<'_> {
    pub(crate) fn size(&self, suffix: &str) -> usize {
        self.sizes.iter().find(|(s, _)| s == suffix).map_or(0, |(_, n)| *n)
    }

    pub(crate) fn take(&self, suffix: &str) -> Result<Vec<f32>> {
        let len = self.size(suffix);
        let v = (self.get)(suffix).ok_or_else(|| invalid(format!("head {:?}: no tensor {suffix}", self.name)))?;
        if v.len() != len {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("head {:?}: {suffix} has {} values, expected {len}", self.name, v.len()),
            ));
        }
        Ok(v)
    }
}

struct AdapterLayer(Lstm);

impl std::fmt::Debug for AdapterLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lstm").field("input", &self.0.input()).field("hidden", &self.0.hidden()).finish()
    }
}

/// Pools words and runs the adapter.
#[derive(Debug)]
pub(crate) struct WordReader {
    pooling: WordPooling,
    hidden: usize,
    adapter: Vec<AdapterLayer>,
    width: usize,
}

/// What a word reader computes in, kept across calls.
#[derive(Default)]
pub struct Work {
    pub(crate) pooled: Vec<f32>,
    adapted: Vec<f32>,
    count: Vec<u32>,
    lstm: LstmWork,
}

impl WordReader {
    pub(crate) fn new(spec: &HeadSpec, hidden: usize, t: &Take<'_>) -> Result<WordReader> {
        let mut adapter = Vec::new();
        let mut width = hidden;
        if let Some(a) = &spec.adapter {
            let h = a.bilstm.hidden as usize;
            for k in 0..a.bilstm.layers {
                let part = |dir: &str, x: &str| t.take(&format!("lstm.{k}.{dir}.{x}"));
                let (fw, fu, fb) = (part("fwd", "w")?, part("fwd", "u")?, part("fwd", "b")?);
                let (bw, bu, bb) = (part("bwd", "w")?, part("bwd", "u")?, part("bwd", "b")?);
                let dirs = [
                    Weights { w: &fw, u: &fu, b: &fb, reverse: false },
                    Weights { w: &bw, u: &bu, b: &bb, reverse: true },
                ];
                adapter.push(AdapterLayer(Lstm::new(width, h, &dirs)?));
                width = 2 * h;
            }
        }
        Ok(WordReader { pooling: spec.word_pooling, hidden, adapter, width })
    }

    /// The width of the word vectors `read` gives.
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// The word vectors [n, width] of one sequence into `w.pooled`, from
    /// its subword states `states` [rows, hidden] and `words` [rows], each
    /// row's word index or [`NO_WORD`]. Returns n.
    ///
    /// Refused: states or map of the wrong size (INVALID_SHAPE); a word
    /// map out of order or a non-finite state (INVALID_ARGUMENT).
    pub(crate) fn read(&self, states: &[f32], words: &[u32], w: &mut Work) -> Result<usize> {
        let rows = words.len();
        if Some(states.len()) != rows.checked_mul(self.hidden) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("{} state values for {rows} rows of {}", states.len(), self.hidden),
            ));
        }
        if !states.iter().all(|v| v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a hidden state is not finite"));
        }
        let n = word_count(words)?;
        if n == 0 {
            w.pooled.clear();
            return Ok(0);
        }
        self.pool(states, words, n, w);
        for layer in &self.adapter {
            let width = layer.0.width();
            w.adapted.clear();
            w.adapted.resize(n * width, 0.0);
            layer.0.run(&w.pooled, &[n], &mut w.adapted, &mut w.lstm)?;
            std::mem::swap(&mut w.pooled, &mut w.adapted);
        }
        w.pooled.truncate(n * self.width);
        Ok(n)
    }

    fn pool(&self, states: &[f32], words: &[u32], n: usize, w: &mut Work) {
        let h = self.hidden;
        w.pooled.clear();
        w.pooled.resize(n * h, 0.0);
        let count = &mut w.count;
        count.clear();
        count.resize(n, 0);
        for (r, &word) in words.iter().enumerate() {
            if word == NO_WORD {
                continue;
            }
            let i = word as usize;
            let (row, dst) = (&states[r * h..][..h], &mut w.pooled[i * h..][..h]);
            match self.pooling {
                WordPooling::First if count[i] == 0 => dst.copy_from_slice(row),
                WordPooling::First => {}
                WordPooling::Mean => dst.iter_mut().zip(row).for_each(|(d, s)| *d += s),
            }
            count[i] += 1;
        }
        if self.pooling == WordPooling::Mean {
            for (i, &c) in w.count.iter().enumerate() {
                let inv = 1.0 / c as f32;
                w.pooled[i * h..][..h].iter_mut().for_each(|v| *v *= inv);
            }
        }
    }
}

/// The number of words a word map names, checking it starts at 0 and
/// rises by at most 1 from row to row, skipping rows of no word, and that
/// no row of no word falls between two rows of one word.
pub fn word_count(words: &[u32]) -> Result<usize> {
    let mut next = 0u32;
    // Whether a NO_WORD row came after the last word's rows: that word
    // may not go on, since a word's rows are adjacent.
    let mut gap = false;
    for (r, &word) in words.iter().enumerate() {
        if word == NO_WORD {
            gap = next > 0;
            continue;
        }
        if word == next {
            next += 1;
        } else if next == 0 || word != next - 1 || gap {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!(
                    "word map row {r}: word {word} after word {}{}",
                    next as i64 - 1,
                    if gap { " and a gap" } else { "" }
                ),
            ));
        }
        gap = false;
    }
    Ok(next as usize)
}
