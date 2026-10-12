//! Tokens to terms: the vocabulary of a document and the term of each
//! token.

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
/// alignment.
///
/// A token with no letter or digit (`char::is_alphanumeric`) is not a
/// term: punctuation would otherwise be in nearly every unit and count
/// toward every length, where Lucene's standard tokenizer drops it.
pub struct TermIndexer;

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
