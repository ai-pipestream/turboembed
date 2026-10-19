//! TokenizerME as an annotator: reads the view and, when the document has
//! them, the sentences; writes the tokens and a confidence per token.

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Confidence, Sentences, Tokens};

use super::model::COMPONENT_NAME;
use super::tokenizer::{Scratch, TokenizerMe};
use crate::unit::tokenize_units_with;

impl Annotator for TokenizerMe {
    const ID: &'static str = "turbo-tokenize:maxent";
    const VERSION: &'static str = "1";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, Confidence<Tokens>);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens, Confidence<Tokens>), Error> {
        let mut scratch = Scratch::new();
        let mut confidence = Vec::with_capacity(view.text().len() / 6);
        let tokens = tokenize_units_with(
            Self::ID,
            view,
            sentences,
            |text, _, out| self.tokenize_with(text, &mut scratch, out),
            |p: f64| confidence.push(p),
        )?;
        Ok((tokens, Confidence::new(confidence)))
    }

    /// The model file: `TokenizerME` and its language as the id, the
    /// OpenNLP version that trained it as the revision, and the file's
    /// SHA-256.
    fn model(&self) -> Option<ModelRef> {
        let m = &self.model;
        Some(ModelRef {
            id: format!("{COMPONENT_NAME}:{}", m.language()),
            revision: m.opennlp_version().to_string(),
            sha256: m.sha256().to_hex(),
        })
    }

    fn config_hash(&self) -> u64 {
        self.options.config_hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenizerOptions;
    use crate::me::TokenizerModel;
    use turbo_analysis::{Document, Span, Spans, ViewId};

    /// The tokenizer over the published English model, or `None` when it
    /// is not downloaded.
    fn english() -> Option<TokenizerMe> {
        let path = turbo_goldens::published_model("opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin")?;
        Some(TokenizerMe::new(TokenizerModel::read(path).unwrap(), TokenizerOptions::default()))
    }

    /// The tokenizer over the checked in fixture model, for what does not
    /// depend on the model's decisions.
    fn fixture() -> TokenizerMe {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../goldens/data/fixtures/tokenizer-abbreviations.bin");
        TokenizerMe::new(TokenizerModel::read(path).unwrap(), TokenizerOptions::default())
    }

    #[test]
    fn whitespace_only_text_has_no_tokens() {
        let doc = Document::new(" \n\t\u{a0} ");
        let (tokens, confidence) = fixture().annotate((doc.get::<View>().unwrap(), None)).unwrap();
        assert!(tokens.spans.is_empty());
        assert!(tokens.sentence.is_none());
        assert!(confidence.value.is_empty());
    }

    #[test]
    fn text_outside_the_sentences_is_not_tokenized() {
        let doc = Document::new("skip Hello. skip");
        let view = doc.get::<View>().unwrap();
        let mut spans = Spans::new(view.id());
        spans.push(Span::new(5, 11));
        let Some(t) = english() else { return };
        let (tokens, confidence) = t.annotate((view, Some(&Sentences { spans }))).unwrap();
        let got: Vec<(u32, u32)> = tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(got, [(5, 10), (10, 11)]);
        assert_eq!(confidence.value[1], 1.0);
        assert_eq!(tokens.sentence.unwrap().iter().map(|i| i.get()).collect::<Vec<_>>(), [0, 0]);
    }

    #[test]
    fn sentences_in_another_view_are_refused() {
        let doc = Document::new("ab");
        let sentences = Sentences { spans: Spans::new(ViewId(3)) };
        let e = fixture().annotate((doc.get::<View>().unwrap(), Some(&sentences))).err().unwrap();
        assert_eq!(
            e,
            Error::Annotator {
                annotator: "turbo-tokenize:maxent",
                detail: "the sentences are in view 3, not in view 0 that is tokenized".into()
            }
        );
    }
}
