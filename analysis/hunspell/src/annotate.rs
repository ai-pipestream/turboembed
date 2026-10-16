//! The Hunspell stemmer as an analysis step, as OpenNLP's
//! `StemmerAnnotator` runs `HunspellStemmer`.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::Tokens;
use turbo_stem_layers::Stems;

use crate::dictionary::Dictionary;
use crate::stemmer::Stemmer;

/// Reads a text view and its `opennlp:tokens` and writes `turbo:stems`:
/// each token's first stem, or the token itself when the dictionary does
/// not read it, as OpenNLP's `HunspellStemmer.stem` returns.
///
/// Each distinct token text is stemmed once per document.
pub struct HunspellAnnotator {
    dictionary: Arc<Dictionary>,
}

impl HunspellAnnotator {
    pub fn new(dictionary: Arc<Dictionary>) -> HunspellAnnotator {
        HunspellAnnotator { dictionary }
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }
}

impl Annotator for HunspellAnnotator {
    const ID: &'static str = "turbo-hunspell:stemmer";
    type Reads = (View, Tokens);
    type Writes = (Stems,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Stems,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are in another view".into() });
        }
        let text = view.text();
        let stemmer = Stemmer::new(&self.dictionary);
        let (starts, ends) = tokens.spans.columns();
        let mut stems = Stems::default();
        // The item that holds the stem of each word stemmed so far.
        let mut seen: HashMap<&str, usize> = HashMap::with_capacity(starts.len());
        let mut repeated = String::new();
        for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
            let Some(token) = text.get(start as usize..end as usize) else {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("token span {start}..{end} is out of bounds or not on UTF-8 character boundaries"),
                });
            };
            if let Some(&first) = seen.get(token) {
                // the column cannot lend an item to itself while it grows
                repeated.clear();
                repeated.push_str(stems.stem.get(first));
                stems.stem.push(&repeated);
            } else {
                match stemmer.stem(token) {
                    Some(stem) => stems.stem.push(&stem),
                    None => stems.stem.push(token),
                }
                seen.insert(token, i);
            }
        }
        Ok((stems,))
    }

    fn config_hash(&self) -> u64 {
        self.dictionary.fingerprint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoadMode;
    use turbo_analysis::{Document, Span, Spans};

    fn sample_annotator() -> HunspellAnnotator {
        let aff = b"SET UTF-8\nSFX S Y 1\nSFX S 0 s .\n";
        let dic = b"1\nwalk/S\n";
        let dict = Dictionary::from_bytes(aff, dic, LoadMode::Strict).expect("valid dictionary");
        HunspellAnnotator::new(Arc::new(dict))
    }

    #[test]
    fn stems_tokens_and_reuses_repeated_words() {
        let annotator = sample_annotator();
        let doc = Document::new("walks walks unknown");
        let view = doc.get::<View>().expect("source view");
        let mut spans = Spans::new(view.id());
        spans.push(Span::new(0, 5));
        spans.push(Span::new(6, 11));
        spans.push(Span::new(12, 19));
        let tokens = Tokens { spans, sentence: None };
        let (stems,) = annotator.annotate((view, &tokens)).expect("valid annotation");
        let collected: Vec<&str> = (0..3).map(|i| stems.stem.get(i)).collect();
        assert_eq!(collected, ["walk", "walk", "unknown"]);
    }

    #[test]
    fn rejects_invalid_token_spans() {
        let annotator = sample_annotator();
        let doc = Document::new("café");
        let view = doc.get::<View>().expect("source view");
        for bad in [Span::new(0, 99), Span::new(0, 4), Span::new(4, 5)] {
            let mut spans = Spans::new(view.id());
            spans.push(bad);
            let tokens = Tokens { spans, sentence: None };
            let err = annotator.annotate((view, &tokens)).expect_err("invalid span must fail");
            assert!(err.to_string().contains("token span"), "unexpected error: {err}");
        }
    }
}
