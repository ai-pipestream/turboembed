use turbo_analysis::{Annotator, ModelRef};
use turbo_langdetect::NgramModel;
use turbo_langspan::NgramDetector;
use turbo_query::QueryAnalysisProfile;
use turbo_unicode::Script;

#[test]
fn default_language_evidence_uses_scripts_without_a_model() {
    assert!(turbo_query::default_span_detector().model().is_none());
    let profile = QueryAnalysisProfile::new();
    for (text, script) in [
        ("hello world", Script::Latin),
        ("Привет мир", Script::Cyrillic),
        ("中文文本", Script::Han),
        ("مرحبا بالعالم", Script::Arabic),
    ] {
        let analysis = profile.analyze(text).unwrap();
        let evidence = analysis.language.unwrap();
        assert_eq!(evidence.best_code, "und", "script cannot identify a language: {text}");
        assert!(!evidence.spans.is_empty(), "{text}");
        for span in evidence.spans {
            assert_eq!(span.script, script, "{text}");
            assert_eq!(span.code, "und", "{text}");
            assert!(!text[span.span.range()].is_empty());
        }
    }
}

#[test]
fn query_uses_trained_language_evidence() {
    let model = NgramModel::new(
        vec!["eng".into()],
        vec![0.0; 16],
        vec![0.0],
        (1, 3),
        16,
        ModelRef { id: "test-langid".into(), revision: "1".into(), sha256: "0".into() },
    )
    .unwrap();
    let profile = QueryAnalysisProfile::builder().with_ngram_detector(NgramDetector::new(model)).build().unwrap();
    let evidence = profile.analyze("hello world").unwrap().language.unwrap();
    assert_eq!(evidence.best_code, "eng");
    assert_eq!(evidence.spans.len(), 1);
    assert_eq!(evidence.spans[0].code, "eng");
    assert!(evidence.spans[0].confidence > 0.99);
}
