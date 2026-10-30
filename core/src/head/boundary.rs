//! A sentence boundary head (`heads[]` of kind HEAD_SENTENCE_BOUNDARY):
//! whether each word ends its sentence, read from the same word vectors
//! every word head of the bundle reads, so a multi-task bundle finds its
//! sentences in the encoder pass that also tags and parses.
//!
//! SaT scores a boundary after every subword piece of raw text. This head
//! scores one after every word of a tokenized document instead, the unit
//! UD treebanks mark sentences in, so it trains on the same words as the
//! bundle's other heads. Its tensors are a token tagger's (tagger.rs)
//! with the two labels `inside` and `last`; a CRF decode is allowed.
//!
//! The last word of a sequence always ends a sentence, whatever its
//! label, so the sentences cover every word.

use std::ops::Range;

use crate::bundle::Bundle;
use crate::manifest::{HeadKind, HeadSpec};
use crate::status::{Error, INVALID_SHAPE, Result, invalid};

use super::tagger::{self, TokenTagger, load_tensors};
use super::words::Work;

/// The label index of `last`, the word that ends a sentence.
pub const LAST: u32 = 1;

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape:
/// a token tagger's.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    tagger::tensors(spec, hidden)
}

/// A loaded sentence boundary head. Read-only after loading, so one head
/// serves many threads, each with its own [`Work`].
#[derive(Debug)]
pub struct SentenceBoundary {
    scorer: TokenTagger,
}

impl SentenceBoundary {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<SentenceBoundary> {
        let l = load_tensors(bundle, name, HeadKind::SentenceBoundary, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A head for `spec` over an encoder `hidden` wide, from `get`, which
    /// gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::SentenceBoundary {
            return Err(invalid(format!("head {:?} is not a sentence boundary head", spec.name)));
        }
        Ok(SentenceBoundary { scorer: TokenTagger::scoring_words(spec, hidden, get)? })
    }

    pub fn spec(&self) -> &HeadSpec {
        self.scorer.spec()
    }

    /// The encoder width the head reads.
    pub fn hidden(&self) -> usize {
        self.scorer.hidden()
    }

    /// Whether the head reads each word's text (a `hash_embed`).
    pub fn reads_forms(&self) -> bool {
        self.scorer.reads_forms()
    }

    /// The scores [words, 2] of one sequence, as
    /// [`TokenTagger::scores_forms`] gives a tagger's. Returns the number
    /// of words.
    pub fn scores(
        &self,
        states: &[f32],
        words: &[u32],
        forms: Option<&[&str]>,
        out: &mut Vec<f32>,
        w: &mut Work,
    ) -> Result<usize> {
        self.scorer.scores_forms(states, words, forms, out, w)
    }

    /// Whether each of `words` words ends a sentence, from its `scores`
    /// [words, 2]: its decoded label is `last`, or it is the last word.
    pub fn decode(&self, scores: &[f32], words: usize, out: &mut Vec<bool>) -> Result<()> {
        let mut labels = Vec::with_capacity(words);
        self.scorer.decode(scores, words, &mut labels)?;
        out.clear();
        out.extend(labels.iter().map(|&l| l == LAST));
        if let Some(end) = out.last_mut() {
            *end = true;
        }
        Ok(())
    }
}

/// The sentences `ends` marks, as word ranges in order: each runs from
/// the word after the last end to the next end. Refused unless the last
/// word ends one, as [`SentenceBoundary::decode`] gives them.
pub fn sentences(ends: &[bool]) -> Result<Vec<Range<usize>>> {
    if ends.last() == Some(&false) {
        return Err(Error::new(INVALID_SHAPE, "the last word does not end a sentence"));
    }
    let mut out = Vec::new();
    let mut start = 0;
    for (i, _) in ends.iter().enumerate().filter(|(_, e)| **e) {
        out.push(start..i + 1);
        start = i + 1;
    }
    Ok(out)
}
