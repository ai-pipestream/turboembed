//! Tests for the Viterbi code-switching `SpanDetector` and `CalibratedSpanDetector`
//! over a native `NgramModel`.

use turbo_analysis::{Analyzer, Document, ModelRef};
use turbo_analysis_layers::{Language, LanguageSpans};
use turbo_langdetect::NgramModel;
use turbo_langspan::{Settings, SpanDetector, SpanScratch};
use turbo_unicode::Script;

fn synthetic_ngram_model() -> NgramModel {
    // 3 languages: eng (Latin), deu (Latin), ell (Greek)
    let buckets = 64u32;
    let mut weights = vec![0.0f32; 3 * buckets as usize];
    // Boost 't', 'h', 'e' n-grams for eng (0)
    for (b, c) in turbo_langdetect::ngram::features("the hello world", (1, 3), buckets) {
        weights[b as usize] += 2.0 * c;
    }
    // Boost 'd', 'e', 'r', 'u', 'n', 'd' n-grams for deu (1)
    for (b, c) in turbo_langdetect::ngram::features("der und morgen", (1, 3), buckets) {
        weights[buckets as usize + b as usize] += 2.0 * c;
    }
    // Boost Greek for ell (2)
    for (b, c) in turbo_langdetect::ngram::features("καλημέρα κόσμε", (1, 3), buckets) {
        weights[2 * buckets as usize + b as usize] += 2.0 * c;
    }
    NgramModel::new(
        vec!["eng".into(), "deu".into(), "ell".into()],
        weights,
        vec![0.0, 0.0, 0.0],
        (1, 3),
        buckets,
        ModelRef { id: "test:langid".into(), revision: "1".into(), sha256: "0".into() },
    )
    .unwrap()
}

#[test]
fn viterbi_span_detector_segments_code_switched_runs_and_populates_document_layers() {
    let model = synthetic_ngram_model();
    let detector = SpanDetector::new(model, Settings::default());
    let text = "the hello world. der und morgen! καλημέρα κόσμε.";

    let mut scratch = SpanScratch::new();
    let mut runs = Vec::new();
    detector.detect(text, None, None, &mut scratch, &mut runs);
    assert_eq!(runs.len(), 3, "{runs:?}");
    assert_eq!(detector.code(&runs[0]), "eng");
    assert_eq!(runs[0].script, Script::Latin);
    assert_eq!(&text[runs[0].span.range()], "the hello world. ");
    assert_eq!(detector.code(&runs[1]), "deu");
    assert_eq!(runs[1].script, Script::Latin);
    assert_eq!(&text[runs[1].span.range()], "der und morgen! ");
    assert_eq!(detector.code(&runs[2]), "ell");
    assert_eq!(runs[2].script, Script::Greek);
    assert_eq!(&text[runs[2].span.range()], "καλημέρα κόσμε.");

    // Also run as an Annotator on Document
    let analyzer = Analyzer::builder().add(detector).build().unwrap();
    let mut doc = Document::new(text);
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);

    let spans = doc.get::<LanguageSpans>().expect("LanguageSpans layer present");
    assert_eq!(spans.spans.len(), 3);
    let codes: Vec<&str> = spans.language.iter().map(|&t| spans.languages.label(t)).collect();
    assert_eq!(codes, ["eng", "deu", "ell"]);

    let lang = doc.get::<Language>().expect("Language summary layer present");
    assert_eq!(lang.code.len(), 3);
}
