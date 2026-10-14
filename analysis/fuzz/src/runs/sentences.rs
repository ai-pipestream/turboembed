//! The sentence detector runs: the maxent detector with a published or
//! fixture model in either whitespace mode, and the newline detector.

use std::sync::Arc;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::{Confidence, Sentences};
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_maxent::ModelFile;
use turbo_sentdetect::{NewlineDetector, SentenceDetector, SentenceModel, WhitespaceMode};

use super::{Outcome, Run, RustError, analyze, has_cut, outcome};
use crate::resources::{Resources, Unavailable};

/// The models that are checked in fixtures rather than downloads.
const FIXTURE_MODELS: [&str; 3] = ["en-abbrev", "en-plain", "en-plain-rust"];

/// The probabilities are compared bit for bit, as
/// `analysis/sentdetect/tests/golden.rs` compares them with `to_bits`.
fn confidence_tolerance() -> Tolerance {
    Tolerance::exact()
}

/// Whitespace mode `legacy` when the run's options name it.
pub(crate) fn whitespace_mode(options: &[&str]) -> Result<WhitespaceMode, Unavailable> {
    match options {
        [] => Ok(WhitespaceMode::Unicode),
        ["legacy"] => Ok(WhitespaceMode::Legacy),
        other => Err(Unavailable::NotCovered(format!("the options {other:?}"))),
    }
}

pub(crate) struct MaxentSentences {
    analyzer: Analyzer,
}

impl MaxentSentences {
    pub(crate) fn open(model: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let mode = whitespace_mode(options)?;
        let path = if FIXTURE_MODELS.contains(&model) {
            res.fixture(&format!("{model}.bin"))?
        } else {
            res.model(&format!("{model}.bin"))?
        };
        let file = ModelFile::read(&path).map_err(|e| Unavailable::Broken(format!("{}: {e}", path.display())))?;
        let model = SentenceModel::open(file).map_err(|e| Unavailable::Broken(format!("{}: {e}", path.display())))?;
        let detector = SentenceDetector::new(Arc::new(model), mode);
        let analyzer = Analyzer::builder().add(detector).build().map_err(|e| Unavailable::Broken(e.to_string()))?;
        Ok(Box::new(MaxentSentences { analyzer }))
    }
}

impl Run for MaxentSentences {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        compare_sentences(doc, golden, true)
    }
}

pub(crate) struct NewlineSentences {
    analyzer: Analyzer,
}

impl NewlineSentences {
    pub(crate) fn open(options: &[&str]) -> Result<Box<dyn Run>, Unavailable> {
        let mode = whitespace_mode(options)?;
        let analyzer = Analyzer::builder()
            .add(NewlineDetector::new(mode))
            .build()
            .map_err(|e| Unavailable::Broken(e.to_string()))?;
        Ok(Box::new(NewlineSentences { analyzer }))
    }
}

impl Run for NewlineSentences {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        compare_sentences(doc, golden, false)
    }
}

/// The sentence spans exactly and, for the maxent detector, the
/// probabilities within the tolerance. A JVM sentence that starts or ends
/// between the halves of a pair has no byte offset, and the golden tests
/// have no such golden, so it is skipped and counted.
fn compare_sentences(doc: &Document, golden: &Golden, confidence: bool) -> Outcome {
    if has_cut(golden, "opennlp:sentences") {
        return Outcome::Skipped("a JVM sentence starts or ends inside a surrogate pair".into());
    }
    let spans = outcome(matches_golden::<Sentences>(doc, golden, Tolerance::exact()));
    if spans != Outcome::Match || !confidence {
        return spans;
    }
    outcome(matches_golden::<Confidence<Sentences>>(doc, golden, confidence_tolerance()))
}
