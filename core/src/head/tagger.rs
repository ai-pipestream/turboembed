//! A token tagger head (`heads[]` of kind HEAD_TOKEN_TAGGER): one label
//! per word, read on the host from the encoder's final-layer subword
//! states, whichever device ran the encoder.
//!
//! The steps, for one sequence of words (a sentence):
//! 1. pool each word's subword rows (its first row, or their mean);
//! 2. run the optional BiLSTM adapter over the words, each layer reading
//!    the one before and joining its two directions;
//! 3. score every label of every word with one linear layer;
//! 4. decode: the best label of each word, or the best sequence under a
//!    first-order CRF with the transitions the scheme forbids removed.
//!
//! The weights are named by the head's `tensor_prefix` and the suffixes
//! [`tensors`] lists (docs/bundle.md, "Heads"). The same suffixes and
//! shapes are what a trainer writes, so [`TokenTagger::from_tensors`]
//! builds a tagger from weights in memory as `load` builds one from a
//! bundle.

use crate::bundle::Bundle;
use crate::cpu::lstm::{Lstm, Weights, Work as LstmWork};
use crate::manifest::{Decode, HeadKind, HeadSpec, WordPooling};
use crate::model::{Expected, place};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::crf::{self, Allowed, Crf, WordSpan};
use super::{Linear, read_f32};

/// A subword row that belongs to no word (a special token), in a
/// [`TokenTagger::scores`] word map.
pub const NO_WORD: u32 = u32::MAX;

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let labels = spec.labels.len() as u64;
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
    out.push(("out.weight".into(), vec![labels, width]));
    out.push(("out.bias".into(), vec![labels]));
    if spec.decode == Decode::Crf {
        out.push(("crf.transitions".into(), vec![labels, labels]));
        out.push(("crf.start".into(), vec![labels]));
        out.push(("crf.end".into(), vec![labels]));
    }
    out
}

/// A loaded token tagger. Read-only after loading, so one tagger serves
/// many threads, each with its own [`Work`].
#[derive(Debug)]
pub struct TokenTagger {
    spec: HeadSpec,
    hidden: usize,
    adapter: Vec<AdapterLayer>,
    out: Linear,
    crf: Option<Crf>,
    allowed: Allowed,
    types: Vec<String>,
}

struct AdapterLayer(Lstm);

impl std::fmt::Debug for AdapterLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lstm").field("input", &self.0.input()).field("hidden", &self.0.hidden()).finish()
    }
}

/// What a tagger computes in, kept across calls so that a call of the
/// same or a smaller size allocates nothing.
#[derive(Default)]
pub struct Work {
    pooled: Vec<f32>,
    adapted: Vec<f32>,
    count: Vec<u32>,
    lstm: LstmWork,
}

impl TokenTagger {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<TokenTagger> {
        let m = &bundle.manifest;
        let spec = m
            .heads
            .iter()
            .find(|h| h.name == name)
            .ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("the bundle has no head {name:?}")))?;
        let a = m.architecture.as_ref().ok_or_else(|| invalid("manifest.json: heads: requires an architecture"))?;
        let hidden = a.hidden as usize;
        let art = m
            .artifacts
            .iter()
            .find(|a| a.name == spec.artifact)
            .ok_or_else(|| invalid(format!("head {:?}: no artifact {:?}", spec.name, spec.artifact)))?;
        let wanted = tensors(spec, hidden);
        let expected: Vec<Expected> = wanted
            .iter()
            .map(|(suffix, shape)| Expected {
                name: format!("{}.{suffix}", spec.tensor_prefix),
                shape: shape.clone(),
                what: format!("head {:?} {suffix}", spec.name),
            })
            .collect();
        let files = art.files.iter().map(|f| bundle.read_verified_aligned(f)).collect::<Result<Vec<_>>>()?;
        let (placed, _) = place(art, &files, &expected)?;
        let values: Vec<Vec<f32>> = placed.iter().map(|p| read_f32(&files, p)).collect();
        let get = |suffix: &str| wanted.iter().position(|(s, _)| s == suffix).map(|i| values[i].clone());
        Self::from_tensors(spec, hidden, &get)
    }

    /// A tagger for `spec` over an encoder `hidden` wide, from `get`,
    /// which gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::TokenTagger {
            return Err(invalid(format!("head {:?} is not a token tagger", spec.name)));
        }
        spec.validate("head")?;
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        let take = |suffix: &str, len: u64| -> Result<Vec<f32>> {
            let v = get(suffix).ok_or_else(|| invalid(format!("head {:?}: no tensor {suffix}", spec.name)))?;
            if v.len() as u64 != len {
                return Err(Error::new(
                    INVALID_SHAPE,
                    format!("head {:?}: {suffix} has {} values, expected {len}", spec.name, v.len()),
                ));
            }
            Ok(v)
        };
        // Each tensor's size, checked, so a width past what memory could
        // hold is refused before anything is read.
        let shapes = tensors(spec, hidden);
        let mut sizes = Vec::with_capacity(shapes.len());
        for (suffix, dims) in &shapes {
            let n = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).filter(|&n| usize::try_from(n).is_ok());
            let n = n.ok_or_else(|| {
                Error::new(INVALID_SHAPE, format!("head {:?}: {suffix} {dims:?} is too large", spec.name))
            })?;
            sizes.push((suffix.as_str(), n));
        }
        let shape = |suffix: &str| sizes.iter().find(|(s, _)| *s == suffix).expect("listed").1;
        let mut adapter = Vec::new();
        let mut width = hidden;
        if let Some(a) = &spec.adapter {
            let h = a.bilstm.hidden as usize;
            for k in 0..a.bilstm.layers {
                let part = |dir: &str, t: &str| {
                    let suffix = format!("lstm.{k}.{dir}.{t}");
                    take(&suffix, shape(&suffix))
                };
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
        let labels = spec.labels.len();
        let out =
            Linear::new(width, labels, take("out.weight", shape("out.weight"))?, take("out.bias", labels as u64)?)?;
        let crf = match spec.decode {
            Decode::Argmax => None,
            Decode::Crf => Some(Crf::new(
                labels,
                take("crf.transitions", (labels * labels) as u64)?,
                take("crf.start", labels as u64)?,
                take("crf.end", labels as u64)?,
            )?),
        };
        let (types, allowed) = crf::allowed(spec.scheme, &spec.labels)?;
        Ok(TokenTagger { spec: spec.clone(), hidden, adapter, out, crf, allowed, types })
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    /// The span types the labels name, in the order they first appear;
    /// empty under SCHEME_TAGS.
    pub fn types(&self) -> &[String] {
        &self.types
    }

    /// The encoder width the tagger reads.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The label scores [words, labels] of one sequence into `out`, from
    /// its subword states `states` [rows, hidden] and `words` [rows], each
    /// row's word index or [`NO_WORD`]. Word indices must start at 0 and
    /// rise by 0 or 1 from row to row, and a word's rows are adjacent: no
    /// NO_WORD row between two rows of one word. Returns the number of
    /// words.
    ///
    /// Refused: states or map of the wrong size (INVALID_SHAPE); a word
    /// map out of order or a non-finite state (INVALID_ARGUMENT).
    pub fn scores(&self, states: &[f32], words: &[u32], out: &mut Vec<f32>, w: &mut Work) -> Result<usize> {
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
        out.clear();
        if n == 0 {
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
        out.resize(n * self.out.out_dim, 0.0);
        self.out.apply_rows(&w.pooled[..n * self.out.in_dim], n, out);
        Ok(n)
    }

    /// Each word's pooled state into `w.pooled` [n, hidden].
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
            match self.spec.word_pooling {
                WordPooling::First if count[i] == 0 => dst.copy_from_slice(row),
                WordPooling::First => {}
                WordPooling::Mean => dst.iter_mut().zip(row).for_each(|(d, s)| *d += s),
            }
            count[i] += 1;
        }
        if self.spec.word_pooling == WordPooling::Mean {
            for (i, &c) in w.count.iter().enumerate() {
                let inv = 1.0 / c as f32;
                w.pooled[i * h..][..h].iter_mut().for_each(|v| *v *= inv);
            }
        }
    }

    /// The label index of each of `words` words from their `scores`
    /// [words, labels]: argmax, or CRF Viterbi under the scheme.
    pub fn decode(&self, scores: &[f32], words: usize, out: &mut Vec<u32>) -> Result<()> {
        match &self.crf {
            None => crf::argmax(scores, words, self.spec.labels.len(), out),
            Some(c) => crf::viterbi(c, &self.allowed, scores, words, out),
        }
    }

    /// The word spans decoded labels mark (BIO and BILOU; none under
    /// SCHEME_TAGS). See [`crf::spans`] for how an ill-formed run reads.
    pub fn spans(&self, tags: &[u32]) -> Result<Vec<WordSpan>> {
        crf::spans(self.spec.scheme, &self.spec.labels, tags)
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
