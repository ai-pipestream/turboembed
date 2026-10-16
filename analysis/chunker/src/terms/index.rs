//! Tokens to terms: the vocabulary of a document and the term of each
//! token.

use turbo_analysis::text::Edit;
use turbo_analysis::{Annotator, Error, Ix, Refs, Span, View};
use turbo_analysis_layers::{Strs, Tokens};

use super::layer::{Terms, Vocabulary};

/// Reads a text view and the tokens; writes the document's `Vocabulary`
/// and the term of each token (`Terms`).
///
/// A term is the text of a token in the view read, byte for byte. Nothing
/// is lowercased or folded here: to match terms without case, read a case
/// folded view (`NormalizeStep::new(CaseFold::full())` in turbo-normalize,
/// bound with `.read(0, "folded")`), and normalize the query the same
/// way. The tokens are either in that view or in its parent, the view it
/// was made from, so they can be found on the source text and their terms
/// read from the folded one; a token's span is mapped through the view's
/// alignment. Its edits must not cross a token edge: an edge strictly inside
/// an edit (one edit "don't" to "do not" over the tokens "do" and "n't")
/// would give both tokens the whole replacement, so such a document is
/// refused. Case folding replaces one character per edit, so it never
/// does.
///
/// A token is a term when it holds at least one character that
/// `char::is_alphanumeric` accepts (an alphabetic character, or a number
/// of category Nd, Nl or No); a token of punctuation or symbols alone,
/// such as a period, is not, and counts toward no length. Lucene's
/// `StandardTokenizer` decides by UAX #29 word boundaries instead: it
/// keeps emoji as tokens, which are no terms here, and makes no token of a
/// lone No character such as "½", which is a term here. Its splits also
/// differ from the word tokenizer's, so the terms here are not Lucene's.
pub struct TermIndexer;

/// The linguistic column to index. Normalize queries with the same profile
/// and column as the documents before constructing a BM25 [`super::Query`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TermForm {
    Surface,
    Folded,
    Stem,
    Lemma,
}

/// Converts language-aware [`turbo_analysis_layers::Terms`] to retrieval
/// IDs. Selects exactly one column; requesting a missing stem or lemma
/// column is an error. Empty and punctuation-only forms have no term ID.
/// The original linguistic layer remains available in the document.
pub struct NormalizedTermIndexer {
    pub form: TermForm,
}

impl Annotator for NormalizedTermIndexer {
    const ID: &'static str = "turbo:normalized-term-indexer";
    type Reads = (turbo_analysis_layers::Terms, Tokens);
    type Writes = (Vocabulary, Terms);

    fn annotate(&self, (terms, tokens): Refs<'_, Self>) -> Result<(Vocabulary, Terms), Error> {
        if terms.surface.len() != tokens.spans.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "term forms are not aligned to tokens".into(),
            });
        }
        let column = match self.form {
            TermForm::Surface => Some(&terms.surface),
            TermForm::Folded => Some(&terms.folded),
            TermForm::Stem => terms.stem.as_ref(),
            TermForm::Lemma => terms.lemma.as_ref(),
        }
        .ok_or_else(|| Error::Annotator {
            annotator: Self::ID,
            detail: format!("requested {:?} term column is missing", self.form),
        })?;
        let words: Vec<_> = column.iter().map(|w| Some(w).filter(|w| w.chars().any(char::is_alphanumeric))).collect();
        let (vocabulary, term) = intern(&words);
        Ok((vocabulary, Terms { term }))
    }

    fn config_hash(&self) -> u64 {
        match self.form {
            TermForm::Surface => 1,
            TermForm::Folded => 2,
            TermForm::Stem => 3,
            TermForm::Lemma => 4,
        }
    }
}

impl Annotator for TermIndexer {
    const ID: &'static str = "turbo:term-indexer";
    type Reads = (View, Tokens);
    type Writes = (Vocabulary, Terms);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Vocabulary, Terms), Error> {
        let in_view = tokens.spans.view() == view.id();
        if !in_view && view.parent() != Some(tokens.spans.view()) {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, neither the terms' view {} nor its parent",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        if !in_view {
            let edits = view.alignment().edits();
            for (i, s) in tokens.spans.iter() {
                if let Some(e) = [s.start, s.end].into_iter().find_map(|x| inside(edits, x)) {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!(
                            "token {} at {}..{} has an edge inside the edit of {}..{} that made the terms' view; \
                             the view's edits must not cross a token edge",
                            i.get(),
                            s.start,
                            s.end,
                            e.parent.start,
                            e.parent.end
                        ),
                    });
                }
            }
        }
        let text = view.text();
        let span = |s: Span| if in_view { s } else { view.alignment().span_to_child(s) };
        let words: Vec<Option<&str>> = tokens
            .spans
            .iter()
            .map(|(_, s)| Some(&text[span(s).range()]).filter(|w| w.chars().any(char::is_alphanumeric)))
            .collect();
        let (vocabulary, term) = intern(&words);
        Ok((vocabulary, Terms { term }))
    }
}

/// The edit whose parent bytes hold offset `x` strictly inside them, if
/// any. Mapping a span edge there would widen it to the whole edit, so
/// two tokens in one edit would both read its whole replacement. An
/// insertion (an edit of no parent bytes) holds no offset strictly inside
/// it. A binary search: the edits are in text order and do not overlap.
fn inside(edits: &[Edit], x: u32) -> Option<Edit> {
    let k = edits.partition_point(|e| e.parent.end <= x);
    edits.get(k).filter(|e| e.parent.start < x).copied()
}

/// The distinct words in byte order, and each word's id among them. Sorts
/// the token indexes by their text, so equal words are adjacent, and gives
/// each run one id.
pub fn intern(words: &[Option<&str>]) -> (Vocabulary, Vec<Option<Ix<Vocabulary>>>) {
    let mut order: Vec<u32> = (0..words.len() as u32).filter(|&i| words[i as usize].is_some()).collect();
    order.sort_unstable_by_key(|&i| words[i as usize]);
    let mut text = Strs::default();
    let mut ids = vec![None; words.len()];
    let mut last: Option<&str> = None;
    for i in order {
        let w = words[i as usize];
        if w != last {
            text.push(w.expect("only words are sorted"));
            last = w;
        }
        ids[i as usize] = Some(Ix::new(text.len() as u32 - 1));
    }
    (Vocabulary { text }, ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_get_ids_in_byte_order_whatever_order_they_come_in() {
        let (v, ids) = intern(&[Some("nap"), Some("cats"), None, Some("nap"), Some("Cats")]);
        let terms: Vec<&str> = v.text.iter().collect();
        assert_eq!(terms, ["Cats", "cats", "nap"]);
        let ids: Vec<Option<usize>> = ids.iter().map(|t| t.map(|t| t.get())).collect();
        assert_eq!(ids, [Some(2), Some(1), None, Some(2), Some(0)]);
        assert_eq!(v.lookup("nap"), Some(Ix::new(2)));
        assert_eq!(v.lookup("dog"), None);
        assert_eq!(v.lookup(""), None);
        assert!(<Vocabulary as turbo_analysis::Layer>::check(&v).is_ok());
    }

    #[test]
    fn no_words_make_an_empty_vocabulary() {
        let (v, ids) = intern(&[None, None]);
        assert!(v.is_empty());
        assert_eq!(ids, [None, None]);
        assert_eq!(v.lookup("a"), None);
    }
}
