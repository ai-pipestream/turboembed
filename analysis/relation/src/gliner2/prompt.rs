//! GLiNER2's input for relation extraction: one schema per relation label,
//! then the text's words, each element tokenized on its own as the
//! reference tokenizes it, with no `[CLS]` or `[SEP]` around the whole.
//!
//! A relation's schema is the elements
//! `( [P] <label> ( [R] head [R] tail ) )`, schemas are joined by
//! `[SEP_STRUCT]`, and `[SEP_TEXT]` comes before the words. The label is
//! one element, as given; the words are lower-cased. The text is the
//! caller's with a `.` added when it does not end in `.`, `!` or `?`,
//! as the reference normalizes it.
//!
//! What the heads read back is the encoder output at three places per
//! relation (its `[P]` and its two `[R]`) and at the first subword of each
//! word; a word with no subwords takes the position the next element
//! starts at.

use std::borrow::Cow;

use turbo::sentencepiece::{Scratch, SentencePiece};

use super::words::{Word, split};

/// The special tokens and their ids in the checkpoint's vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Specials {
    pub sep_struct: u32,
    pub sep_text: u32,
    pub p: u32,
    pub r: u32,
}

/// One relation request as the encoder sees it.
#[derive(Debug)]
pub struct Prompt<'a> {
    pub ids: Vec<u32>,
    /// The normalized text, which the word offsets index.
    pub text: Cow<'a, str>,
    pub words: Vec<Word<'a>>,
    /// Each word's first subword position in `ids`.
    pub word_first: Vec<usize>,
    /// Each relation's `[P]`, head `[R]` and tail `[R]` positions.
    pub markers: Vec<[usize; 3]>,
}

/// `text` as the reference normalizes it before splitting. Borrows the
/// text when nothing changes; copies only when a `.` must be appended.
pub fn normalize(text: &str) -> Cow<'_, str> {
    if text.is_empty() {
        Cow::Borrowed(".")
    } else if text.ends_with(['.', '!', '?']) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(format!("{text}."))
    }
}

/// Builds the input for `labels` over `text`.
pub fn build<'a>(sp: &SentencePiece, specials: &Specials, text: &'a str, labels: &[&str]) -> Prompt<'a> {
    build_words(sp, specials, text, labels, split(text))
}

pub fn build_words<'a>(
    sp: &SentencePiece,
    specials: &Specials,
    text: &'a str,
    labels: &[&str],
    mut words: Vec<Word<'a>>,
) -> Prompt<'a> {
    let mut scratch = Scratch::default();
    let mut ids = Vec::new();
    let mut piece = |s: &str, ids: &mut Vec<u32>| sp.encode_ids(s, &mut scratch, ids);
    let mut markers = Vec::with_capacity(labels.len());
    for (k, label) in labels.iter().enumerate() {
        if k > 0 {
            ids.push(specials.sep_struct);
        }
        piece("(", &mut ids);
        let p = ids.len();
        ids.push(specials.p);
        piece(label, &mut ids);
        piece("(", &mut ids);
        let head = ids.len();
        ids.push(specials.r);
        piece("head", &mut ids);
        let tail = ids.len();
        ids.push(specials.r);
        piece("tail", &mut ids);
        piece(")", &mut ids);
        piece(")", &mut ids);
        markers.push([p, head, tail]);
    }
    ids.push(specials.sep_text);
    let normalized = normalize(text);
    super::words::append_period(text, &mut words);
    let mut word_first = Vec::with_capacity(words.len());
    for w in &mut words {
        word_first.push(ids.len());
        let at = ids.len();
        piece(&w.text, &mut ids);
        w.subword_len = ids.len() - at;
    }
    Prompt { ids, text: normalized, words, word_first, markers }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_borrows_when_nothing_changes() {
        assert!(matches!(normalize("ends here."), Cow::Borrowed(_)));
        assert!(matches!(normalize(""), Cow::Borrowed(_)));
        assert!(matches!(normalize("no period"), Cow::Owned(_)));
        assert_eq!(normalize("no period").as_ref(), "no period.");
        assert_eq!(normalize("ends!").as_ref(), "ends!");
    }
}
