//! What every word head reads first, whatever it scores: each word's
//! subword states pooled (its first row, or their mean), with a
//! `hash_embed` the hashed embedding of the word's text joined after
//! them (hash_embed.rs), then the optional BiLSTM adapter over the words, each layer reading the one
//! before and joining its two directions. The token tagger (tagger.rs)
//! and the biaffine parser (parser.rs) share it, so the same spec gives
//! the same word vectors to both.

use super::hash_embed::{self, Tables};
use crate::cpu::lstm::{Lstm, Weights, Work as LstmWork};
use crate::manifest::{HeadSpec, WordPooling};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

/// A subword row that belongs to no word (a special token), in a word map.
pub const NO_WORD: u32 = u32::MAX;

/// The adapter's tensors as suffixes and shapes, then the hashed
/// feature tables', and the width of the word vectors the reader gives
/// (`hidden`, and the hashed width, without an adapter).
pub(crate) fn tensors(spec: &HeadSpec, hidden: usize) -> (Vec<(String, Vec<u64>)>, u64) {
    let mut out = Vec::new();
    let mut width = hidden as u64 + spec.hash_embed.map_or(0, |h| 4 * h.dim as u64);
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
    if let Some(h) = &spec.hash_embed {
        for k in 0..4 {
            out.push((hash_embed::suffix(k), vec![h.rows[k] as u64, h.dim as u64]));
        }
    }
    (out, width)
}

/// Appends `mix.weight` `[layers]` and `mix.gamma` `[1]` when `spec` mixes
/// encoder layers (`spec.layer_mix`).
pub(crate) fn push_mix(spec: &HeadSpec, out: &mut Vec<(String, Vec<u64>)>) {
    if let Some(m) = &spec.layer_mix {
        out.push(("mix.weight".into(), vec![m.layers as u64]));
        out.push(("mix.gamma".into(), vec![1]));
    }
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
    /// `gamma * softmax(mix.weight)` per tapped layer, when `spec.layer_mix` is set.
    mix: Option<Vec<f32>>,
    hash: Option<Tables>,
    adapter: Vec<AdapterLayer>,
    width: usize,
}

/// What a word reader computes in, kept across calls.
#[derive(Default)]
pub struct Work {
    pub(crate) pooled: Vec<f32>,
    tap: Vec<f32>,
    adapted: Vec<f32>,
    count: Vec<u32>,
    lstm: LstmWork,
}

impl WordReader {
    pub(crate) fn new(spec: &HeadSpec, hidden: usize, t: &Take<'_>) -> Result<WordReader> {
        let mix = match &spec.layer_mix {
            Some(_) => {
                let w = t.take("mix.weight")?;
                let g = t.take("mix.gamma")?;
                if !w.iter().chain(&g).all(|x| x.is_finite()) {
                    return Err(invalid(format!("head {:?}: layer mix weights are not finite", spec.name)));
                }
                let max = w.iter().map(|&v| v as f64).fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = w.iter().map(|&v| (v as f64 - max).exp()).collect();
                let z: f64 = e.iter().sum();
                let gamma = g[0];
                Some(e.into_iter().map(|v| gamma * (v / z) as f32).collect())
            }
            None => None,
        };
        let hash = match spec.hash_embed {
            Some(h) => {
                let take = |k| t.take(&hash_embed::suffix(k));
                Some(Tables::new(h, [take(0)?, take(1)?, take(2)?, take(3)?])?)
            }
            None => None,
        };
        let mut adapter = Vec::new();
        let mut width = hidden + hash.as_ref().map_or(0, Tables::width);
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
        Ok(WordReader { pooling: spec.word_pooling, hidden, mix, hash, adapter, width })
    }

    /// The width of the word vectors `read` gives.
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Encoder layers the reader mixes (1 when `layer_mix` is absent).
    pub(crate) fn taps(&self) -> usize {
        self.mix.as_ref().map_or(1, Vec::len)
    }

    /// Whether the reader needs each word's text (a `hash_embed`).
    pub(crate) fn reads_forms(&self) -> bool {
        self.hash.is_some()
    }

    /// The word vectors [n, width] of one sequence into `w.pooled`, from
    /// its subword states `states` ([rows, hidden], or [taps, rows, hidden]
    /// with `layer_mix`) and `words` [rows], each
    /// row's word index or [`NO_WORD`], and with a `hash_embed` each
    /// word's text `forms` (ignored without one). Returns n.
    ///
    /// Refused: states, map or forms of the wrong size (INVALID_SHAPE); a
    /// word map out of order, a non-finite state, or no forms for a head
    /// that reads them (INVALID_ARGUMENT).
    pub(crate) fn read(&self, states: &[f32], words: &[u32], forms: Option<&[&str]>, w: &mut Work) -> Result<usize> {
        let rows = words.len();
        let taps = self.taps();
        let want = taps.checked_mul(rows).and_then(|r| r.checked_mul(self.hidden));
        if Some(states.len()) != want {
            return Err(Error::new(
                INVALID_SHAPE,
                if taps == 1 {
                    format!("{} state values for {rows} rows of {}", states.len(), self.hidden)
                } else {
                    format!("{} state values for {taps} layers x {rows} rows of {}", states.len(), self.hidden)
                },
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
        if let Some(t) = &self.hash {
            let forms = forms.ok_or_else(|| Error::new(INVALID_ARGUMENT, "the head reads each word's text"))?;
            if forms.len() != n {
                return Err(Error::new(INVALID_SHAPE, format!("{} forms for {n} words", forms.len())));
            }
            let (h, width) = (self.hidden, self.hidden + t.width());
            w.adapted.clear();
            w.adapted.resize(n * width, 0.0);
            for (i, form) in forms.iter().enumerate() {
                let row = &mut w.adapted[i * width..][..width];
                row[..h].copy_from_slice(&w.pooled[i * h..][..h]);
                t.embed(form, &mut row[h..]);
            }
            std::mem::swap(&mut w.pooled, &mut w.adapted);
        }
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
        let rows = words.len();
        if let Some(scales) = &self.mix {
            w.pooled.clear();
            w.pooled.resize(n * h, 0.0);
            let block = rows * h;
            for (t, &scale) in scales.iter().enumerate() {
                pool_tap(self.pooling, h, &states[t * block..(t + 1) * block], words, n, &mut w.tap, &mut w.count);
                w.pooled.iter_mut().zip(&w.tap).for_each(|(dst, &v)| *dst += scale * v);
            }
        } else {
            pool_tap(self.pooling, h, states, words, n, &mut w.pooled, &mut w.count);
        }
    }

    pub(crate) fn pool_into(
        &self,
        states: &[f32],
        words: &[u32],
        n: usize,
        pooled: &mut Vec<f32>,
        count: &mut Vec<u32>,
    ) {
        let h = self.hidden;
        if let Some(scales) = &self.mix {
            let rows = words.len();
            pooled.clear();
            pooled.resize(n * h, 0.0);
            let block = rows * h;
            let mut tap = Vec::with_capacity(n * h);
            for (t, &scale) in scales.iter().enumerate() {
                pool_tap(self.pooling, h, &states[t * block..(t + 1) * block], words, n, &mut tap, count);
                pooled.iter_mut().zip(&tap).for_each(|(dst, &v)| *dst += scale * v);
            }
        } else {
            pool_tap(self.pooling, h, states, words, n, pooled, count);
        }
    }
}

fn pool_tap(
    pooling: WordPooling,
    h: usize,
    states: &[f32],
    words: &[u32],
    n: usize,
    out: &mut Vec<f32>,
    count: &mut Vec<u32>,
) {
    out.clear();
    out.resize(n * h, 0.0);
    count.clear();
    count.resize(n, 0);
    for (r, &word) in words.iter().enumerate() {
        if word == NO_WORD {
            continue;
        }
        let i = word as usize;
        let (row, dst) = (&states[r * h..][..h], &mut out[i * h..][..h]);
        match pooling {
            WordPooling::First if count[i] == 0 => dst.copy_from_slice(row),
            WordPooling::First => {}
            WordPooling::Mean => dst.iter_mut().zip(row).for_each(|(d, s)| *d += s),
        }
        count[i] += 1;
    }
    if pooling == WordPooling::Mean {
        for (i, &c) in count.iter().enumerate() {
            let inv = 1.0 / c as f32;
            out[i * h..][..h].iter_mut().for_each(|v| *v *= inv);
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
