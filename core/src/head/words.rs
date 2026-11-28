//! What every word head reads first, whatever it scores: each word's
//! subword states pooled (its first row, or their mean), with a
//! `hash_embed` the hashed embedding of the word's text joined after
//! them (hash_embed.rs), then the optional BiLSTM adapter over the words, each layer reading the one
//! before and joining its two directions. The token tagger (tagger.rs)
//! and the biaffine parser (parser.rs) share it, so the same spec gives
//! the same word vectors to both.

use super::adapter::Adapter;
use super::hash_embed::{self, Tables};
use crate::device::lstm::Work as LstmWork;
use crate::device::{Device, Precision};
use crate::manifest::{HeadSpec, WordPooling};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

/// A subword row that belongs to no word (a special token), in a word map.
pub const NO_WORD: u32 = u32::MAX;

/// The adapter's tensors as suffixes and shapes, then the hashed
/// feature tables', and the width of the word vectors the reader gives
/// (`hidden`, and the hashed width, without an adapter).
pub(crate) fn tensors(spec: &HeadSpec, hidden: usize) -> (Vec<(String, Vec<u64>)>, u64) {
    let mut out = Vec::new();
    if let Some(language) = &spec.language {
        out.push(("language.embedding".into(), vec![language.languages.len() as u64, language.dim as u64]));
    }
    let language_width = spec.language.as_ref().map_or(0, |language| language.dim as u64);
    let mut width = hidden as u64 + language_width + spec.hash_embed.map_or(0, |h| 4 * h.dim as u64);
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
/// encoder layers (`spec.layer_mix` without a single `layer` override).
pub(crate) fn push_mix(spec: &HeadSpec, out: &mut Vec<(String, Vec<u64>)>) {
    if let Some(m) = &spec.layer_mix
        && m.layer.is_none()
    {
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

#[derive(Debug)]
struct Language {
    names: Vec<String>,
    table: Vec<f32>,
    dim: usize,
}

/// Pools words and runs the adapter.
#[derive(Debug)]
pub(crate) struct WordReader {
    pooling: WordPooling,
    hidden: usize,
    taps: usize,
    /// `gamma * softmax(mix.weight)` per tapped layer, when `spec.layer_mix` mixes layers.
    mix: Option<Vec<f32>>,
    language: Option<Language>,
    hash: Option<Tables>,
    adapter: Adapter,
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
        let (taps, mix) = match &spec.layer_mix {
            Some(m) if m.layer.is_some() => (m.layers as usize, None),
            Some(m) => {
                let w = t.take("mix.weight")?;
                let g = t.take("mix.gamma")?;
                if !w.iter().chain(&g).all(|x| x.is_finite()) {
                    return Err(invalid(format!("head {:?}: layer mix weights are not finite", spec.name)));
                }
                let max = w.iter().map(|&v| v as f64).fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = w.iter().map(|&v| (v as f64 - max).exp()).collect();
                let z: f64 = e.iter().sum();
                let gamma = g[0];
                (m.layers as usize, Some(e.into_iter().map(|v| gamma * (v / z) as f32).collect()))
            }
            None => (1, None),
        };
        let language = match &spec.language {
            Some(language) => {
                let table = t.take("language.embedding")?;
                if table.iter().any(|value| !value.is_finite()) {
                    return Err(invalid(format!(
                        "head {:?}: language.embedding holds a value that is not finite",
                        spec.name
                    )));
                }
                Some(Language { names: language.languages.clone(), table, dim: language.dim as usize })
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
        let language_width = language.as_ref().map_or(0, |language| language.dim);
        let in_width = hidden + language_width + hash.as_ref().map_or(0, Tables::width);
        let (adapter, width) = Adapter::new(spec, in_width, t)?;
        Ok(WordReader { pooling: spec.word_pooling, hidden, taps, mix, language, hash, adapter, width })
    }

    pub(crate) fn on_device_at(&mut self, device: Device, precision: Precision) -> Result<()> {
        self.adapter.on_device_at(device, precision)
    }

    pub(crate) fn lstm_device(&self) -> Option<Device> {
        self.adapter.device()
    }

    pub(crate) fn lstm_precision(&self) -> Option<Precision> {
        self.adapter.precision()
    }

    /// The width of the word vectors `read` gives.
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Encoder layers the reader taps from the top of the encoder (1 when `layer_mix` is absent).
    pub(crate) fn taps(&self) -> usize {
        self.taps
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
        self.read_language(states, words, forms, None, w)
    }

    /// As [`read`](Self::read), selecting the language embedding by name.
    /// A one-language head defaults to its only row; a multilingual head
    /// requires a name and refuses names its manifest does not list.
    pub(crate) fn read_language(
        &self,
        states: &[f32],
        words: &[u32],
        forms: Option<&[&str]>,
        language: Option<&str>,
        w: &mut Work,
    ) -> Result<usize> {
        let language = self.language_index(language)?;
        let n = self.prepare_sequence(states, words, forms, language, w)?;
        if n > 0 {
            self.adapter.run(&[n], n, &mut w.pooled, &mut w.adapted, &mut w.lstm)?;
            w.pooled.truncate(n * self.width);
        }
        Ok(n)
    }

    /// Reads `features.iter().zip(forms)` as one batched adapter pass, writing
    /// all sequences' `[n_s, width]` vectors in input order into `out` and
    /// returning each sequence's word count.
    pub(crate) fn read_sequences(
        &self,
        features: &[Vec<f32>],
        forms: &[Vec<&str>],
        language: Option<&str>,
        out: &mut Vec<f32>,
        w: &mut Work,
    ) -> Result<Vec<usize>> {
        let count = features.len().min(forms.len());
        let stride = self.taps() * self.hidden;
        if count <= 1 || self.adapter.is_empty() {
            out.clear();
            let mut lens = Vec::with_capacity(count);
            for (f, fs) in features[..count].iter().zip(&forms[..count]) {
                let words: Vec<u32> = (0..(f.len() / stride) as u32).collect();
                let n = self.read_language(f, &words, Some(fs), language, w)?;
                out.extend_from_slice(&w.pooled[..n * self.width]);
                lens.push(n);
            }
            return Ok(lens);
        }
        let lang = self.language_index(language)?;
        let in_width =
            self.hidden + self.language.as_ref().map_or(0, |l| l.dim) + self.hash.as_ref().map_or(0, Tables::width);
        let mut prepared: Vec<Vec<f32>> = Vec::with_capacity(count);
        let mut lens = Vec::with_capacity(count);
        for (f, fs) in features[..count].iter().zip(&forms[..count]) {
            let words: Vec<u32> = (0..(f.len() / stride) as u32).collect();
            let n = self.prepare_sequence(f, &words, Some(fs), lang, w)?;
            prepared.push(w.pooled[..n * in_width].to_vec());
            lens.push(n);
        }
        let mut order: Vec<usize> = (0..count).filter(|&i| lens[i] > 0).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
        let sorted_lens: Vec<usize> = order.iter().map(|&i| lens[i]).collect();
        let rows: usize = sorted_lens.iter().sum();
        out.clear();
        if rows == 0 {
            return Ok(lens);
        }
        w.pooled.clear();
        w.pooled.reserve(rows * in_width);
        for &i in &order {
            w.pooled.extend_from_slice(&prepared[i]);
        }
        self.adapter.run(&sorted_lens, rows, &mut w.pooled, &mut w.adapted, &mut w.lstm)?;
        let mut offsets = vec![0usize; count];
        let mut acc = 0usize;
        for (i, &len) in lens.iter().enumerate() {
            offsets[i] = acc;
            acc += len;
        }
        out.resize(rows * self.width, 0.0);
        let mut at = 0usize;
        for (&i, &len) in order.iter().zip(&sorted_lens) {
            out[offsets[i] * self.width..][..len * self.width]
                .copy_from_slice(&w.pooled[at * self.width..][..len * self.width]);
            at += len;
        }
        Ok(lens)
    }

    fn prepare_sequence(
        &self,
        states: &[f32],
        words: &[u32],
        forms: Option<&[&str]>,
        language: Option<usize>,
        w: &mut Work,
    ) -> Result<usize> {
        let rows = words.len();
        let taps = self.taps();
        let one = rows.checked_mul(self.hidden);
        let want = taps.checked_mul(rows).and_then(|r| r.checked_mul(self.hidden));
        let ok = if self.mix.is_some() {
            Some(states.len()) == want
        } else {
            // A single-layer head (`layer: Some(k)`) reads the first tap of
            // the trailing `taps` encoder slice at runtime (`want`), or the
            // pre-extracted layer `k` buffer (`one`) in cached evaluation.
            Some(states.len()) == want || Some(states.len()) == one
        };
        if !ok {
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
        if let (Some(table), Some(index)) = (&self.language, language) {
            let (h, d, width) = (self.hidden, table.dim, self.hidden + table.dim);
            w.adapted.clear();
            w.adapted.resize(n * width, 0.0);
            for i in 0..n {
                let row = &mut w.adapted[i * width..][..width];
                row[..h].copy_from_slice(&w.pooled[i * h..][..h]);
                row[h..].copy_from_slice(&table.table[index * d..][..d]);
            }
            std::mem::swap(&mut w.pooled, &mut w.adapted);
        }
        if let Some(t) = &self.hash {
            let forms = forms.ok_or_else(|| Error::new(INVALID_ARGUMENT, "the head reads each word's text"))?;
            if forms.len() != n {
                return Err(Error::new(INVALID_SHAPE, format!("{} forms for {n} words", forms.len())));
            }
            let h = self.hidden + self.language.as_ref().map_or(0, |language| language.dim);
            let width = h + t.width();
            w.adapted.clear();
            w.adapted.resize(n * width, 0.0);
            for (i, form) in forms.iter().enumerate() {
                let row = &mut w.adapted[i * width..][..width];
                row[..h].copy_from_slice(&w.pooled[i * h..][..h]);
                t.embed(form, &mut row[h..]);
            }
            std::mem::swap(&mut w.pooled, &mut w.adapted);
        }
        Ok(n)
    }

    fn language_index(&self, name: Option<&str>) -> Result<Option<usize>> {
        let Some(language) = &self.language else {
            return match name {
                Some(name) => Err(invalid(format!("language {name:?} given to a head without a language input"))),
                None => Ok(None),
            };
        };
        let name = match name {
            Some(name) => name,
            None if language.names.len() == 1 => &language.names[0],
            None => return Err(invalid("a multilingual head needs a language")),
        };
        language
            .names
            .iter()
            .position(|candidate| candidate == name)
            .map(Some)
            .ok_or_else(|| invalid(format!("unknown language {name:?}: one of {}", language.names.join(", "))))
    }

    fn pool(&self, states: &[f32], words: &[u32], n: usize, w: &mut Work) {
        self.pool_with_tap(states, words, n, &mut w.pooled, &mut w.tap, &mut w.count);
    }

    pub(crate) fn pool_into(
        &self,
        states: &[f32],
        words: &[u32],
        n: usize,
        pooled: &mut Vec<f32>,
        count: &mut Vec<u32>,
    ) {
        let mut tap = Vec::new();
        self.pool_with_tap(states, words, n, pooled, &mut tap, count);
    }

    fn pool_with_tap(
        &self,
        states: &[f32],
        words: &[u32],
        n: usize,
        pooled: &mut Vec<f32>,
        tap: &mut Vec<f32>,
        count: &mut Vec<u32>,
    ) {
        let h = self.hidden;
        if let Some(scales) = &self.mix {
            let block = words.len() * h;
            pooled.clear();
            pooled.resize(n * h, 0.0);
            for (t, &scale) in scales.iter().enumerate() {
                pool_tap(self.pooling, h, &states[t * block..(t + 1) * block], words, n, tap, count);
                pooled.iter_mut().zip(&*tap).for_each(|(dst, &v)| *dst += scale * v);
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
