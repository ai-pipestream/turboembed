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
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
            let token = &text[start as usize..end as usize];
            if let Some(&first) = seen.get(token) {
                let stem = stems.stem.get(first).to_string();
                stems.stem.push(&stem);
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
