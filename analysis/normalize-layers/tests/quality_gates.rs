use turbo_analysis::{Analyzer, Bound, ConditionalStep, DocPredicate, Document, Key, LayerStatus, ViewId};
use turbo_analysis_layers::{
    QualityFlags as F, Sentences, TextQuality, lexical_quality_gate, quality_gate, quality_gate_on,
};
use turbo_normalize_layers::TextQualityAnnotator;
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::SimpleTokenizer;

const PROSE: &str = "A careful reader can use this passage to learn about clear writing. Each sentence presents a useful idea and gives enough context for the reader to understand the entire explanation.";

#[test]
fn actual_conditional_sentence_steps_skip_or_run_and_report_their_output_statuses() {
    let score_step = ConditionalStep::from_predicate(
        "quality",
        quality_gate(ViewId::SOURCE, 0.6, F::NAV_BOILERPLATE | F::MOJIBAKE_SUSPECT),
        Bound::new(RuleSentenceDetector::new("en")).write(0, "after-quality"),
    );
    let lexical_step = ConditionalStep::from_predicate(
        "lexical",
        lexical_quality_gate(ViewId::SOURCE, 0.3, 8.0),
        Bound::new(RuleSentenceDetector::new("en")).write(0, "after-lexical"),
    );
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
    assert!(!quality_gate(ViewId::SOURCE, 0.0, F::CLEAN).test(&docs[0]));
    let pipeline = Analyzer::builder()
        .add(RuleSentenceDetector::new("en"))
        .add(SimpleTokenizer::default())
        .add(TextQualityAnnotator)
        .build()
        .unwrap();
    pipeline.run(&mut docs)[0].as_ref().unwrap();
    let doc = &docs[0];
    let q = doc.get::<TextQuality>().unwrap();
    assert!(quality_gate(ViewId::SOURCE, q.metrics.composite_score, F::CLEAN).test(doc));
    assert!(
        lexical_quality_gate(ViewId::SOURCE, q.metrics.information_density, q.metrics.mtld_lexical_diversity).test(doc)
    );
    assert!(!quality_gate(ViewId(999), 0.0, F::CLEAN).test(doc));
    assert!(!quality_gate_on(Key::named("missing"), ViewId::SOURCE, 0.0, F::CLEAN).test(doc));
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(!quality_gate(ViewId::SOURCE, threshold, F::CLEAN).test(doc));
        assert!(!lexical_quality_gate(ViewId::SOURCE, threshold, 0.0).test(doc));
    }
    assert!(!lexical_quality_gate(ViewId::SOURCE, 0.0, f32::NAN).test(doc));
    assert!(!quality_gate(ViewId::SOURCE, 0.0, F(0x8000)).test(doc));
}

#[test]
fn typed_gate_parameters_and_input_instances_change_conditional_provenance() {
    let hash = |predicate| {
        ConditionalStep::from_predicate(
            "quality",
            predicate,
            Bound::new(RuleSentenceDetector::new("en")).write(0, "after-quality"),
        )
        .config_hash()
    };
    let baseline = hash(quality_gate(ViewId::SOURCE, 0.5, F::CLEAN));
    for predicate in [
        quality_gate(ViewId::SOURCE, 0.6, F::CLEAN),
        quality_gate(ViewId::SOURCE, 0.5, F::NAV_BOILERPLATE),
        quality_gate(ViewId(1), 0.5, F::CLEAN),
        quality_gate_on(Key::named("clean"), ViewId::SOURCE, 0.5, F::CLEAN),
        lexical_quality_gate(ViewId::SOURCE, 0.5, 0.0),
        lexical_quality_gate(ViewId::SOURCE, 0.5, 10.0),
    ] {
        assert_ne!(baseline, hash(predicate));
    }
    assert_eq!(baseline, hash(quality_gate(ViewId::SOURCE, 0.5, F::CLEAN)));
}

#[test]
fn named_quality_dependencies_are_bound_and_failed_quality_is_reported_before_downstream_skip() {
    use turbo_analysis::{Provenance, View, ViewBuilder};
    let pipeline = Analyzer::builder()
        .add(RuleSentenceDetector::new("en"))
        .add(SimpleTokenizer::default())
        .step(Bound::new(TextQualityAnnotator).write(0, "named"))
        .step(ConditionalStep::from_predicate(
            "named-quality",
            quality_gate_on(Key::named("named"), ViewId::SOURCE, 0.5, F::CLEAN),
            Bound::new(RuleSentenceDetector::new("en")).write(0, "after-named"),
        ))
        .build()
        .unwrap();
    let mut docs = [Document::new(PROSE)];
    pipeline.run_with_report(&mut docs)[0].0.as_ref().unwrap();
    assert!(docs[0].get_key(&Key::<Sentences>::named("after-named")).is_some());
    assert!(
        docs[0]
            .provenance(Key::<Sentences>::named("after-named").id())
            .unwrap()
            .reads
            .contains(&Key::<TextQuality>::named("named").id())
    );

    let mut doc = Document::new(PROSE);
    let (text, alignment) = ViewBuilder::new(PROSE).finish();
    let other = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("other"), other, Provenance::by("test")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::named("other"))
        .add(RuleSentenceDetector::new("en"))
        .add(SimpleTokenizer::default())
        .step(Bound::new(TextQualityAnnotator).read(0, "other"))
        .step(ConditionalStep::from_predicate(
            "quality",
            quality_gate(ViewId::SOURCE, 0.0, F::CLEAN),
            Bound::new(RuleSentenceDetector::new("en")).write(0, "after-failure"),
        ))
        .build()
        .unwrap();
    let mut docs = [doc];
    let reports = pipeline.run_with_report(&mut docs);
    assert!(reports[0].0.is_err());
    let outputs: Vec<_> = reports[0].1.steps.iter().flat_map(|s| &s.outputs).collect();
    assert!(matches!(
        outputs.iter().find(|o| o.key == Key::<TextQuality>::DEFAULT.id()).unwrap().status,
        Some(LayerStatus::Failed(_))
    ));
    assert!(matches!(
        outputs.iter().find(|o| o.key == Key::<Sentences>::named("after-failure").id()).unwrap().status,
        Some(LayerStatus::Skipped(_))
    ));
}
