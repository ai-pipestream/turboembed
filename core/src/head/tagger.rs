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
use crate::manifest::{Decode, HeadKind, HeadSpec};
use crate::model::{Expected, place};
use crate::status::{Error, INVALID_ARGUMENT, Result, invalid};

use super::crf::{self, Allowed, Crf, WordSpan};
use super::words::{self, Take, WordReader};
pub use super::words::{NO_WORD, Work, word_count};
use super::{Linear, read_f32};

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let labels = spec.labels.len() as u64;
    let (mut out, width) = words::tensors(spec, hidden);
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
    reader: WordReader,
    out: Linear,
    crf: Option<Crf>,
    allowed: Allowed,
    types: Vec<String>,
}

impl TokenTagger {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<TokenTagger> {
        let l = load_tensors(bundle, name, HeadKind::TokenTagger, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
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
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let labels = spec.labels.len();
        let out = Linear::new(reader.width(), labels, t.take("out.weight")?, t.take("out.bias")?)?;
        let crf = match spec.decode {
            Decode::Crf => {
                Some(Crf::new(labels, t.take("crf.transitions")?, t.take("crf.start")?, t.take("crf.end")?)?)
            }
            _ => None,
        };
        let (types, allowed) = crf::allowed(spec.scheme, &spec.labels)?;
        Ok(TokenTagger { spec: spec.clone(), hidden, reader, out, crf, allowed, types })
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
        out.clear();
        let n = self.reader.read(states, words, w)?;
        out.resize(n * self.out.out_dim, 0.0);
        if n > 0 {
            self.out.apply_rows(&w.pooled, n, out);
        }
        Ok(n)
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

/// Tensor suffixes and shapes, as `tensors` gives them.
pub(crate) type Shapes = Vec<(String, Vec<u64>)>;

/// A head's spec, its encoder width and its tensors by suffix.
pub(crate) struct Loaded {
    pub spec: HeadSpec,
    pub hidden: usize,
    values: Vec<(String, Vec<f32>)>,
}

impl Loaded {
    pub(crate) fn get(&self, suffix: &str) -> Option<Vec<f32>> {
        self.values.iter().find(|(s, _)| s == suffix).map(|(_, v)| v.clone())
    }
}

/// The head `name` of `bundle`, its encoder width, and its tensors by
/// suffix as `tensors` lists them, read and checked from its artifact; a
/// head of another kind than `kind` is refused before anything is read.
pub(crate) fn load_tensors(
    bundle: &Bundle,
    name: &str,
    kind: HeadKind,
    tensors: fn(&HeadSpec, usize) -> Shapes,
) -> Result<Loaded> {
    let m = &bundle.manifest;
    let spec = m
        .heads
        .iter()
        .find(|h| h.name == name)
        .ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("the bundle has no head {name:?}")))?;
    if spec.kind != kind {
        return Err(invalid(format!("head {name:?} is {:?}, not {kind:?}", spec.kind)));
    }
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
    let values = wanted.into_iter().map(|(s, _)| s).zip(values).collect();
    Ok(Loaded { spec: spec.clone(), hidden, values })
}
