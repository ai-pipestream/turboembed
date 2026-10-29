//! The language detector runs: the published model and the fixture the
//! Rust trainer wrote, both single-pass (`predictLanguages`) and
//! chunked probing (`probingPredictLanguages`).

use turbo_analysis::{Analyzer, Document, Key, Provenance};
use turbo_analysis_layers::{Language, Strs};
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_langdetect::{ContextGenerator, LanguageDetector, LanguageModel, ProbingConfig, Ranking, Scratch};
use turbo_maxent::ModelFile;

use super::{Outcome, Run, RustError, analyze, outcome};
use crate::resources::{Resources, Unavailable};

/// The checked in model the Rust trainer wrote from
/// `inputs/langdetect/samples.txt`.
const FIXTURE: &str = "langdetect-fixture-rust";

const CHUNK_50: ProbingConfig =
    ProbingConfig { max_length: 250, chunk_size: 50, min_consec_improvements: 2, min_diff: 0.20 };

pub(crate) struct Detector {
    analyzer: Analyzer,
    detector: LanguageDetector,
    probing: Option<ProbingConfig>,
}

impl Detector {
    pub(crate) fn open(model: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let probing = match options {
            [] => None,
            ["probing"] => Some(ProbingConfig::default()),
            ["probing-chunk50"] => Some(CHUNK_50),
            _ => return Err(Unavailable::NotCovered(format!("langdetect/{model}+{}", options.join("+")))),
        };
        let name = format!("{model}.bin");
        let path = if model == FIXTURE { res.fixture(&name)? } else { res.model(&name)? };
        let broken = |e: &dyn std::fmt::Display| Unavailable::Broken(format!("{}: {e}", path.display()));
        let file = ModelFile::read(&path).map_err(|e| broken(&e))?;
        let loaded = LanguageModel::open(file).map_err(|e| broken(&e))?;
        let file2 = ModelFile::read(&path).map_err(|e| broken(&e))?;
        let loaded2 = LanguageModel::open(file2).map_err(|e| broken(&e))?;
        let analyzer = Analyzer::builder()
            .add(LanguageDetector::new(loaded))
            .build()
            .map_err(|e| Unavailable::Broken(e.to_string()))?;
        Ok(Box::new(Detector { analyzer, detector: LanguageDetector::new(loaded2), probing }))
    }

    fn compare_probing(&self, text: &str, config: ProbingConfig, golden: &Golden) -> Outcome {
        let mut scratch = Scratch::new();
        let mut ranking = Ranking::new();
        let length = self.detector.probing_predict_languages_with(text, config, &mut scratch, &mut ranking);
        let Some(expected_len) =
            golden.layer("turbo:probing-length").and_then(|l| l.u32s("length")).and_then(|s| s.first()).copied()
        else {
            return Outcome::Divergence("golden is missing turbo:probing-length".into());
        };
        if length != expected_len as usize {
            return Outcome::Divergence(format!("probing length is {length} in Rust and {expected_len} on the JVM"));
        }
        let mut code = Strs::default();
        for &o in ranking.outcomes() {
            code.push(self.detector.model().code(o as usize));
        }
        let mut doc = Document::new(text);
        doc.add(
            &Key::<Language>::DEFAULT,
            Language { code, score: ranking.probabilities().to_vec() },
            Provenance::by("turbo-langdetect:probing"),
        )
        .expect("document has no language layer yet");
        let exact = matches_golden::<Language>(&doc, golden, Tolerance::exact());
        if exact.is_ok() {
            return Outcome::Match;
        }
        if !scratch.probing_context().exact_order()
            && matches_golden::<Language>(&doc, golden, Tolerance::abs(1e-12)).is_ok()
        {
            return Outcome::Documented(
                "a tree bucket of the JVM's n-gram map held n-grams with equal hash codes, which the JDK orders by identity hash code; the probabilities differ in the last bits (analysis/langdetect/src/probing.rs, exact_order)".into(),
            );
        }
        outcome(exact)
    }
}

impl Run for Detector {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        if self.probing.is_some() {
            return Ok(Document::new(text));
        }
        analyze(&self.analyzer, text)
    }

    /// The ranking exactly. On a text without language evidence (empty,
    /// or nothing but digits, URLs and emoji) the JVM ranks every language
    /// at the prior and the port declines to write a layer, which
    /// `analysis/langdetect/tests/goldens.rs` records.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if let Some(config) = self.probing {
            return self.compare_probing(doc.source(), config, golden);
        }
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
