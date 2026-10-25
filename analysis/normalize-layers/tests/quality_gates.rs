use turbo_analysis::{Analyzer, Bound, ConditionalStep, Document, Key, LayerStatus, ViewId};
use turbo_analysis_layers::{
    QualityFlags as F, Sentences, TextQuality, lexical_quality_gate, quality_gate, quality_gate_on,
};
use turbo_normalize_layers::TextQualityAnnotator;
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::SimpleTokenizer;

const PROSE: &str = "A careful reader can use this passage to learn about clear writing. Each sentence presents a useful idea and gives enough context for the reader to understand the entire explanation.";

#[test]
fn actual_conditional_sentence_steps_skip_or_run_and_report_their_output_statuses() {
    let score_step = ConditionalStep::new(
        "quality",
        quality_gate(ViewId::SOURCE, 0.6, F::NAV_BOILERPLATE | F::MOJIBAKE_SUSPECT),
        Bound::new(RuleSentenceDetector::new("en")).write(0, "after-quality"),
    )
    .require(Key::<TextQuality>::DEFAULT);
    let lexical_step = ConditionalStep::new(
        "lexical",
        lexical_quality_gate(ViewId::SOURCE, 0.3, 8.0),
        Bound::new(RuleSentenceDetector::new("en")).write(0, "after-lexical"),
    )
    .require(Key::<TextQuality>::DEFAULT);
    let pipeline = Analyzer::builder()
        .add(RuleSentenceDetector::new("en"))
        .add(SimpleTokenizer::default())
        .add(TextQualityAnnotator)
        .step(score_step)
        .step(lexical_step)
        .build()
        .unwrap();
    let mut docs = [
        Document::new(PROSE),
        Document::new("Home | About | Contact | Privacy"),
        Document::new("\u{fffd} \u{fffd} \u{fffd}"),
    ];
    let reports = pipeline.run_with_report(&mut docs);
    for (i, (result, report)) in reports.iter().enumerate() {
        assert_eq!(result, &Ok(()));
        for name in ["after-quality", "after-lexical"] {
            let key = Key::<Sentences>::named(name);
            let recorded = report.steps.iter().flat_map(|s| &s.outputs).find(|o| o.key == key.id()).unwrap();
            let expected = docs[i].status(key.id()).unwrap();
            assert_eq!(recorded.status.as_ref(), Some(expected));
            if i == 0 {
                assert_eq!(expected, &LayerStatus::Complete);
                assert!(!docs[i].get_key(&key).unwrap().spans.is_empty());
            } else {
                assert!(matches!(expected, LayerStatus::Skipped(_)));
                assert!(docs[i].get_key(&key).is_none());
            }
        }
    }
}

#[test]
fn predicates_fail_closed_for_missing_wrong_named_view_and_invalid_thresholds() {
    let mut docs = [Document::new(PROSE)];
    assert!(!quality_gate(ViewId::SOURCE, 0.0, F::CLEAN)(&docs[0]));
    let pipeline = Analyzer::builder()
        .add(RuleSentenceDetector::new("en"))
        .add(SimpleTokenizer::default())
        .add(TextQualityAnnotator)
        .build()
        .unwrap();
    pipeline.run(&mut docs)[0].as_ref().unwrap();
    let doc = &docs[0];
    let q = doc.get::<TextQuality>().unwrap();
    assert!(quality_gate(ViewId::SOURCE, q.metrics.composite_score, F::CLEAN)(doc));
    assert!(lexical_quality_gate(ViewId::SOURCE, q.metrics.information_density, q.metrics.mtld_lexical_diversity)(doc));
    assert!(!quality_gate(ViewId(999), 0.0, F::CLEAN)(doc));
    assert!(!quality_gate_on(Key::named("missing"), ViewId::SOURCE, 0.0, F::CLEAN)(doc));
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(!quality_gate(ViewId::SOURCE, threshold, F::CLEAN)(doc));
        assert!(!lexical_quality_gate(ViewId::SOURCE, threshold, 0.0)(doc));
    }
    assert!(!lexical_quality_gate(ViewId::SOURCE, 0.0, f32::NAN)(doc));
    assert!(!quality_gate(ViewId::SOURCE, 0.0, F(0x8000))(doc));
}
