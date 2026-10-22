//! GLiREL's input: `[CLS]`, then `[REL] <label>` for each relation label,
//! `[SEP]`, the text's words and `[SEP]`. Each label and each word is
//! tokenized on its own, as the reference's fast tokenizer does with
//! `is_split_into_words`, and what the heads read is the encoder output at
//! each element's first subword.

use turbo::sentencepiece::{Scratch, SentencePiece};

use super::words::{Word, split};

/// The special tokens and their ids in the checkpoint's vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Specials {
    pub cls: u32,
    pub sep: u32,
    pub rel: u32,
}

/// One relation request as the encoder sees it.
#[derive(Debug)]
pub struct Prompt {
    pub ids: Vec<u32>,
    pub words: Vec<Word>,
    /// Each word's first subword position in `ids`; `None` for a word
    /// the tokenizer gives no subwords (one normalization removes), which
    /// the reference reads as a zero vector.
    pub word_first: Vec<Option<usize>>,
    /// Each label's `[REL]` position and its first subword's.
    pub labels: Vec<[usize; 2]>,
}

/// The input for `labels` over `text`. A label with no subwords (an empty
/// or all-whitespace one) has nothing to read and is `None`.
pub fn build(sp: &SentencePiece, specials: &Specials, text: &str, labels: &[&str]) -> Option<Prompt> {
    let mut scratch = Scratch::default();
    let mut ids = vec![specials.cls];
    let mut marks = Vec::with_capacity(labels.len());
    for label in labels {
        let rel = ids.len();
        ids.push(specials.rel);
        let first = ids.len();
        sp.encode_ids(label, &mut scratch, &mut ids);
        if ids.len() == first {
            return None;
        }
        marks.push([rel, first]);
    }
    ids.push(specials.sep);
    let words = split(text);
    let mut word_first = Vec::with_capacity(words.len());
    for w in &words {
        let at = ids.len();
        sp.encode_ids(&text[w.start..w.end], &mut scratch, &mut ids);
        word_first.push((ids.len() > at).then_some(at));
    }
    ids.push(specials.sep);
    Some(Prompt { ids, words, word_first, labels: marks })
}
