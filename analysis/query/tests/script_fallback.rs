use turbo_analysis::Annotator;
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
