//! TokenizerME as an annotator: reads the view and, when the document has
//! them, the sentences; writes the tokens and a confidence per token.

use std::any::Any;

use turbo_analysis::annotate::{AnchorUse, Step};
use turbo_analysis::{Annotator, Bound, Document, Error, Ix, KeyId, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Confidence, Sentences, Tokens};

use super::model::COMPONENT_NAME;
use super::tokenizer::{Scratch, TokenizerMe};

impl Annotator for TokenizerMe {
    const ID: &'static str = "turbo-tokenize:maxent";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, Confidence<Tokens>);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens, Confidence<Tokens>), Error> {
        let (tokens, confidence) = tokenize_units(self, view, sentences)?;
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

impl TokenizerMe {
    /// The tokenizer as a pipeline step, for `Analyzer::builder().boxed`.
    ///
    /// `Analyzer::build` refuses a step that writes an aligned layer whose
    /// anchor it does not read, and the confidence is aligned to the
    /// tokens this same step writes. The step returned here is the
    /// annotator bound to the default instances, reporting the confidence
    /// as aligned to its own tokens output, so the pipeline accepts it.
    /// Everything else (reads, provenance, failures) is the bound
    /// annotator's.
    pub fn into_step(self) -> Box<dyn Step> {
        Box::new(OwnTokens(Bound::new(self)))
    }
}

/// A bound `TokenizerMe` whose confidence output is anchored to its own
/// tokens output.
struct OwnTokens(Bound<TokenizerMe>);

impl Step for OwnTokens {
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn reads(&self) -> Vec<KeyId> {
        self.0.reads()
    }
    fn required(&self) -> Vec<bool> {
        self.0.required()
    }
    fn writes(&self) -> Vec<KeyId> {
        self.0.writes()
    }
    fn anchors(&self) -> Vec<AnchorUse> {
        let writes = self.0.writes();
        self.0
            .anchors()
            .into_iter()
            .map(|a| match a {
                AnchorUse::Unread(base) => writes.iter().find(|w| w.layer == base).map_or(a, |&w| AnchorUse::To(w)),
                other => other,
            })
            .collect()
    }
    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        self.0.compute(docs)
    }
    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        self.0.store(out, docs, status)
    }
}

/// Tokenizes each sentence on its own, or the whole view when there are
/// no sentences, and fills the tokens with their sentence column and the
/// confidences. To be replaced by the crate's shared sentence handling
/// (`unit.rs`), which carries no per-token values yet.
fn tokenize_units(t: &TokenizerMe, view: &View, sentences: Option<&Sentences>) -> Result<(Tokens, Vec<f32>), Error> {
    let text = view.text();
    let mut scratch = Scratch::new();
    let mut found: Vec<(Span, f64)> = Vec::new();
    let mut spans = Spans::<Tokens>::with_capacity(view.id(), text.len() / 5);
    let mut confidence = Vec::with_capacity(text.len() / 5);
    let Some(sentences) = sentences else {
        t.tokenize_with(text, &mut scratch, &mut found);
        for &(s, p) in &found {
            spans.push(s);
            confidence.push(p as f32);
        }
        return Ok((Tokens { spans, sentence: None }, confidence));
    };
    if sentences.spans.view() != view.id() {
        return Err(Error::Annotator {
            annotator: TokenizerMe::ID,
            detail: format!(
                "the sentences are in view {}, not in view {} that is tokenized",
                sentences.spans.view().0,
                view.id().0
            ),
        });
    }
    let mut column: Vec<Ix<Sentences>> = Vec::with_capacity(confidence.capacity());
    for (ix, sentence) in sentences.spans.iter() {
        found.clear();
        t.tokenize_with(&text[sentence.range()], &mut scratch, &mut found);
        for &(s, p) in &found {
            spans.push(Span::new(s.start + sentence.start, s.end + sentence.start));
            confidence.push(p as f32);
            column.push(ix);
        }
    }
    Ok((Tokens { spans, sentence: Some(column) }, confidence))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::me::{TokenizerMeOptions, TokenizerModel};
    use turbo_analysis::ViewId;

    fn english() -> TokenizerMe {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/models/opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin");
        TokenizerMe::new(TokenizerModel::read(path).unwrap(), TokenizerMeOptions::default())
    }

    #[test]
    fn whitespace_only_text_has_no_tokens() {
        let doc = Document::new(" \n\t\u{a0} ");
        let (tokens, confidence) = english().annotate((doc.get::<View>().unwrap(), None)).unwrap();
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
        let (tokens, confidence) = english().annotate((view, Some(&Sentences { spans }))).unwrap();
        let got: Vec<(u32, u32)> = tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(got, [(5, 10), (10, 11)]);
        assert_eq!(confidence.value[1], 1.0);
        assert_eq!(tokens.sentence.unwrap().iter().map(|i| i.get()).collect::<Vec<_>>(), [0, 0]);
    }

    #[test]
    fn sentences_in_another_view_are_refused() {
        let doc = Document::new("ab");
        let sentences = Sentences { spans: Spans::new(ViewId(3)) };
        let e = english().annotate((doc.get::<View>().unwrap(), Some(&sentences))).err().unwrap();
        assert_eq!(
            e,
            Error::Annotator {
                annotator: "turbo-tokenize:maxent",
                detail: "the sentences are in view 3, not in view 0 that is tokenized".into()
            }
        );
    }
}
