//! The stemmer as an analysis step: OpenNLP's `StemmerAnnotator`.

use std::collections::HashMap;

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::Tokens;
use turbo_stem_layers::Stems;

use crate::algorithm::Algorithm;
use crate::stemmer::Stemmer;

/// Reads a text view and its `opennlp:tokens` and writes `turbo:stems`,
/// the stem of each token's text.
///
/// Each distinct token text is stemmed once per document; a repeat
/// copies the first stem.
pub struct StemAnnotator {
    algorithm: Algorithm,
}

impl StemAnnotator {
    pub fn new(algorithm: Algorithm) -> StemAnnotator {
        StemAnnotator { algorithm }
    }

    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }
}

impl Annotator for StemAnnotator {
    const ID: &'static str = "turbo-stem:stemmer";
    type Reads = (View, Tokens);
    type Writes = (Stems,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Stems,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are in another view".into() });
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let mut stems = Stems::default();
        // The item that holds the stem of each word stemmed so far.
        let mut seen: HashMap<&str, usize> = HashMap::new();
        let (mut word, mut out, mut stem) = (Vec::new(), Vec::new(), String::new());
        for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
            let token = &text[start as usize..end as usize];
            stem.clear();
            if let Some(&first) = seen.get(token) {
                stem.push_str(stems.stem.get(first));
            } else {
                word.clear();
                word.extend(token.encode_utf16());
                self.algorithm.stem_utf16(&word, &mut out);
                stem.extend(char::decode_utf16(out.iter().copied()).map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER)));
                seen.insert(token, i);
            }
            stems.stem.push(&stem);
        }
        Ok((stems,))
    }

    fn config_hash(&self) -> u64 {
        self.algorithm.config_hash()
    }
}
