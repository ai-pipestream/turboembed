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
