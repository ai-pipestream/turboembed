//! The detector as a pipeline step.
//!
//! The detector writes `Sentences` and `Confidence<Sentences>`, a layer
//! aligned to the sentences it writes in the same step. The analyzer
//! refuses an aligned output whose anchor the step does not read, and it
//! sees only reads, so a plain `Analyzer::builder().add(detector)` is
//! refused. [`DetectorStep`] is the detector bound to the default
//! instances, declaring that its confidence output is aligned to its own
//! sentences output. The document then anchors the confidence to the
//! default `Sentences` instance, which is the one the step wrote.

use std::any::Any;

use turbo_analysis::annotate::{AnchorUse, Step};
use turbo_analysis::{Bound, Document, Error, KeyId};

use crate::me::detector::SentenceDetector;

/// A [`SentenceDetector`] as a step whose confidence output is aligned to
/// its own sentences output.
pub struct DetectorStep(Bound<SentenceDetector>);

impl SentenceDetector {
    /// This detector as a pipeline step, for `Analyzer::builder().step(..)`.
    pub fn into_step(self) -> DetectorStep {
        DetectorStep(Bound::new(self))
    }
}

impl Step for DetectorStep {
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

#[cfg(test)]
mod tests {
    use turbo_analysis::Analyzer;
    use turbo_analysis_layers::{Confidence, Sentences};
    use turbo_maxent::ModelFile;
    use turbo_unicode::WhitespaceMode;

    use super::*;
    use crate::me::model::SentenceModel;

    fn detector() -> SentenceDetector {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/en-plain.bin");
        SentenceDetector::new(SentenceModel::open(ModelFile::read(path).unwrap()).unwrap(), WhitespaceMode::Unicode)
    }

    #[test]
    fn the_confidence_is_aligned_to_the_own_sentences() {
        // Bound alone declares the anchor unread, which the analyzer refuses.
        let bound = Bound::new(detector());
        assert!(matches!(bound.anchors()[..], [AnchorUse::None, AnchorUse::Unread(_)]));
        assert!(Analyzer::builder().add(detector()).build().is_err());

        let step = detector().into_step();
        let sentences = turbo_analysis::Key::<Sentences>::DEFAULT.id();
        assert_eq!(step.writes(), [sentences, turbo_analysis::Key::<Confidence<Sentences>>::DEFAULT.id()]);
        assert_eq!(step.anchors(), [AnchorUse::None, AnchorUse::To(sentences)]);
        assert!(Analyzer::builder().step(step).build().is_ok());
    }
}
