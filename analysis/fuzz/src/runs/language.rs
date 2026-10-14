//! The language detector run with the published model.

use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::Language;
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_langdetect::{ContextGenerator, LanguageDetector, LanguageModel};
use turbo_maxent::ModelFile;

use super::{Outcome, Run, RustError, analyze, outcome};
use crate::resources::{Resources, Unavailable};

pub(crate) struct Detector {
    analyzer: Analyzer,
}

impl Detector {
    pub(crate) fn open(model: &str, res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = res.model(&format!("{model}.bin"))?;
        let broken = |e: &dyn std::fmt::Display| Unavailable::Broken(format!("{}: {e}", path.display()));
        let file = ModelFile::read(&path).map_err(|e| broken(&e))?;
        let model = LanguageModel::open(file).map_err(|e| broken(&e))?;
        let analyzer = Analyzer::builder()
            .add(LanguageDetector::new(model))
            .build()
            .map_err(|e| Unavailable::Broken(e.to_string()))?;
        Ok(Box::new(Detector { analyzer }))
    }
}

impl Run for Detector {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    /// The ranking exactly. On a text without language evidence (empty,
    /// or nothing but digits, URLs and emoji) the JVM ranks every language
    /// at the prior and the port declines to write a layer, which
    /// `analysis/langdetect/tests/goldens.rs` records.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if doc.get::<Language>().is_none() {
            let status = doc.status(Key::<Language>::DEFAULT.id()).map(|s| format!("{s:?}")).unwrap_or_default();
            let all_equal = golden
                .layer("opennlp:language")
                .and_then(|l| l.f64s("confidence"))
                .is_some_and(|scores| scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-15));
            return if all_equal {
                Outcome::Documented(format!(
                    "the JVM ranks every language at the prior and the port declines ({status}; analysis/langdetect/tests/goldens.rs)"
                ))
            } else {
                Outcome::Divergence(format!("the port wrote no language layer ({status}) and the JVM ranked"))
            };
        }
        let exact = matches_golden::<Language>(doc, golden, Tolerance::exact());
        if exact.is_ok() {
            return Outcome::Match;
        }
        // The n-grams go to the model in the iteration order of the JVM's
        // hash maps, which the port reproduces except where a bucket the
        // JDK turned into a tree held n-grams with equal hash codes, whose
        // order depends on identity hash codes (analysis/langdetect/src/context.rs,
        // `exact_order`). Then the same n-grams are summed in another order
        // and the probabilities differ in their last bits.
        let mut generator = ContextGenerator::default();
        generator.generate(doc.source());
        if !generator.exact_order() && matches_golden::<Language>(doc, golden, Tolerance::abs(1e-12)).is_ok() {
            return Outcome::Documented(
                "a tree bucket of the JVM's n-gram map held n-grams with equal hash codes, which the JDK orders by identity hash code; the probabilities differ in the last bits (analysis/langdetect/src/context.rs, exact_order)".into(),
            );
        }
        outcome(exact)
    }
}
