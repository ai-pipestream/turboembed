use super::*;
use turbo_analysis::{Analyzer, Document, Ix, IxRange, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::{Incivility, Sentences, Tokens};

const COMMENTS: [&str; 2] =
    ["You’re an idiot and a liar! 😢 Stop lying about the report.", "Thanks for sharing the update on the schedule."];

fn sample_units(texts: &[&str]) -> Vec<Unit> {
    texts
        .iter()
        .enumerate()
        .map(|(i, s)| Unit {
            id: format!("c{i}"),
            parent_id: None,
            text: (*s).to_owned(),
            targets: IncivilityScores::default(),
            subgroups: Vec::new(),
        })
        .collect()
}

#[test]
fn comment_offsets_preserve_utf8_and_project_across_coordinate_systems() {
    let units = sample_units(&COMMENTS);
    let p = prepare(&units).unwrap();
    // "You’re": curly apostrophe occupies UTF-8 bytes 3..6, so token 0 is 0..3, token 1 is 3..6, token 2 is 6..8.
    assert_eq!(p.token_layer.spans.columns().0[1], 3);
    assert_eq!(p.token_layer.spans.columns().1[1], 6);
    let mut token = 0u32;
    let mut base = 0u32;
    for (i, u) in units.iter().enumerate() {
        let doc = Document::new(u.text.as_str());
        let mut expected = Vec::new();
        turbo_tokenize::simple_spans(&u.text, turbo_tokenize::TokenizerOptions::default(), &mut expected);
        assert_eq!(p.sentence_layer.spans.columns().0[i], base);
        assert_eq!(p.sentence_layer.spans.columns().1[i], base + u.text.len() as u32);
        for original in expected {
            let range = IxRange::new(token, token + 1);
            let actual = p.source_span(i, range).unwrap();
            assert_eq!(actual, Span::new(original.start, original.end));
            let utf16 = doc.view_span_to_utf16(ViewId::SOURCE, actual).unwrap();
            let points = doc.view_span_to_code_point(ViewId::SOURCE, actual).unwrap();
            assert_eq!(doc.view_span_from_utf16(ViewId::SOURCE, utf16).unwrap(), actual);
            assert_eq!(doc.view_span_from_code_point(ViewId::SOURCE, points).unwrap(), actual);
            assert_eq!(p.token_layer.spans.columns().0[token as usize], base + actual.start);
            assert_eq!(p.token_layer.spans.columns().1[token as usize], base + actual.end);
            token += 1;
        }
        base += u.text.len() as u32;
    }
}

#[test]
fn empty_comments_add_no_synthetic_tokens_and_preserve_following_offsets() {
    let units = sample_units(&["", " \t\n", "😀", COMMENTS[0]]);
    let p = prepare(&units).unwrap();
    assert!(p.tokens[0].is_empty() && p.tokens[1].is_empty());
    assert_eq!(p.token_layer.spans.columns().0[0], 3);
    assert_eq!(p.token_layer.spans.columns().1[0], 7);
    let scored = score_split(&IncivilityAnnotator::new(), &p, &[], None, 0).unwrap();
    assert!(scored.evidence_of(Ix::new(0)).next().is_none());
    assert!(scored.evidence_of(Ix::new(1)).next().is_none());
    assert!(p.source_span(0, IxRange::new(0, 1)).is_err());
    assert!(p.source_span(3, IxRange::new(0, 1)).is_err());
    let doc = Document::new("😀");
    let emoji = p.source_span(2, IxRange::new(0, 1)).unwrap();
    let utf16 = doc.view_span_to_utf16(ViewId::SOURCE, emoji).unwrap();
    assert_eq!(doc.view_span_from_utf16(ViewId::SOURCE, utf16).unwrap(), emoji);
}

#[test]
fn corpus_evidence_matches_production_annotator_on_original_comments() {
    let units = sample_units(&[COMMENTS[0], COMMENTS[1], "You are ſtupid.", "You are STUPID."]);
    let p = prepare(&units).unwrap();
    let a = IncivilityAnnotator::new();
    let corpus = score_split(&a, &p, &[], None, 0).unwrap();
    let mut token_offset = 0u32;
    for (i, u) in units.iter().enumerate() {
        let mut doc = Document::new(u.text.as_str());
        let single = prepare(std::slice::from_ref(u)).unwrap();
        doc.add(&Key::DEFAULT, single.sentence_layer, Provenance::by("original-comment")).unwrap();
        doc.add(&Key::DEFAULT, single.token_layer, Provenance::by("original-comment")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .add(a.clone())
            .build()
            .unwrap();
        for result in analyzer.run(std::slice::from_mut(&mut doc)) {
            result.unwrap();
        }
        let actual = doc.get::<Incivility>().unwrap();
        assert_eq!(corpus.scores(Ix::new(i as u32)), actual.scores(Ix::new(0)));
        let expected: Vec<_> = actual
            .evidence_of(Ix::new(0))
            .map(|(l, r)| (l, IxRange::new(r.start.get() as u32 + token_offset, r.end.get() as u32 + token_offset)))
            .collect();
        assert_eq!(corpus.evidence_of(Ix::new(i as u32)).collect::<Vec<_>>(), expected);
        for (_, range) in expected {
            assert!(!u.text[p.source_span(i, range).unwrap().range()].is_empty());
        }
        token_offset += p.tokens[i].len() as u32;
    }
}

#[test]
#[ignore = "release timing receipt for #963"]
fn score_unit_interleaved_timing_receipt() {
    let head = IncivilityVectorHead::new(
        4,
        vec![
            0.8, -0.2, 0.4, 0.1, // Insult
            0.3, 0.9, -0.1, 0.2, // Profanity
            0.2, 0.1, 0.95, -0.3, // Threat
            0.7, -0.4, 0.2, 0.6, // IdentityAttack
            0.5, 0.3, -0.2, 0.8, // Intolerance
            0.4, 0.4, 0.5, 0.3, // SevereToxicity
        ],
        [-1.0, -1.2, -1.5, -1.1, -1.3, -2.0],
    )
    .unwrap();
    let scorer: Arc<dyn IncivilityVectorScorer> = Arc::new(head);
    let baseline = IncivilityAnnotator::new()
        .with_vector_scorer(Arc::clone(&scorer))
        .with_fusion_gate(0.0)
        .with_severe_toxicity_rule(true);
    let tuned = IncivilityAnnotator::new()
        .with_vector_scorer(Arc::clone(&scorer))
        .with_fusion_gate(0.25)
        .with_severe_toxicity_rule(true);
    let units = sample_units(&COMMENTS);
    let p = prepare(&units).unwrap();
    let vec_a = [0.6f32, 0.2, 0.7, 0.1];
    let vec_b = [-0.3f32, 0.1, -0.2, 0.0];
    let mut base_us = Vec::with_capacity(7);
    let mut tuned_us = Vec::with_capacity(7);
    let run_batch = |ann: &IncivilityAnnotator| -> u128 {
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for i in 0..10_000usize {
            let row = i & 1;
            let v = if row == 0 { &vec_a[..] } else { &vec_b[..] };
            let (s, ev) = ann.score_unit(&p.tokens[row], 0, p.tokens[row].len(), None, Some(v)).unwrap();
            acc += s.insult + ev.len() as f64;
        }
        std::hint::black_box(acc);
        t0.elapsed().as_micros()
    };
    let _ = run_batch(&baseline);
    let _ = run_batch(&tuned);
    for round in 0..7 {
        if round % 2 == 0 {
            base_us.push(run_batch(&baseline));
            tuned_us.push(run_batch(&tuned));
        } else {
            tuned_us.push(run_batch(&tuned));
            base_us.push(run_batch(&baseline));
        }
    }
    base_us.sort_unstable();
    tuned_us.sort_unstable();
    eprintln!(
        "baseline_us (gate=0.0): min={} med={} max={} | tuned_us (gate=0.25): min={} med={} max={}",
        base_us[0], base_us[3], base_us[6], tuned_us[0], tuned_us[3], tuned_us[6]
    );
}

#[test]
#[ignore = "requires the actual published Civil Comments MiniLM head"]
fn tuned_dev_metrics_match_the_selected_configuration_on_published_head() {
    let path = std::env::var("TURBO_INCIVILITY_HEAD").expect("download the published MiniLM TIVH head");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(Sha256::of(&bytes).to_string(), "26c1147d5431901e41ca6847ee482d0534c63b0ddcf63e9e99f2b22e472997ba");
    let head = IncivilityVectorHead::from_bytes(&bytes).unwrap();
    let annotator = IncivilityAnnotator::new().with_vector_scorer(Arc::new(head));
    let mut units = sample_units(&["You are an idiot."]);
    units[0].targets.severe_toxicity = 1.0;
    let prepared = prepare(&units).unwrap();
    // Finite adversarial input to the published head, not a corpus accuracy claim.
    let vectors = vec![5.0_f32; 384];
    // Evidence evaluation must score with its supplied thresholds as well.
    let evidence_units = sample_units(&["Uncharacteristically, you are an idiot."]);
    let evidence = prepare(&evidence_units).unwrap();
    let doc = Document::new(evidence_units[0].text.as_str());
    let views = [doc.get::<turbo_analysis::View>().unwrap().clone()];
    let tokens = std::slice::from_ref(&evidence.token_layer);
    let gold = [turbo_doccat::GoldToxicSpanUnit { utf8_spans: vec![], token_ranges: vec![] }];
    let evaluate = |ann: &IncivilityAnnotator| {
        spans_eval::eval_subset(ann, &views, tokens, &evidence.tokens, &gold, &vectors, 384, &[0.05; 6]).unwrap()
    };
    assert_eq!(evaluate(&annotator), evaluate(&annotator.clone().with_threshold(0.05)));

    let frozen = score_split(&annotator, &prepared, &vectors, None, 384).unwrap();
    let candidate = score_split(&annotator.clone().with_threshold(0.05), &prepared, &vectors, None, 384).unwrap();
    assert!(frozen.scores(Ix::new(0)).severe_toxicity < 0.05);
    assert!(candidate.scores(Ix::new(0)).severe_toxicity > 0.4);
    let (uniform, uniform_micro, per_label, macro_f1, micro_f1) =
        tune_thresholds(&annotator, &prepared, &vectors, None, 384).unwrap();
    let selected = score_split(&annotator.clone().with_threshold(uniform), &prepared, &vectors, None, 384).unwrap();
    assert_eq!(uniform_micro, evaluate_incivility_corpus(&prepared.units, &selected, uniform).micro_f1);
    let selected = score_split(&annotator.with_thresholds(per_label), &prepared, &vectors, None, 384).unwrap();
    let metrics = evaluate_incivility_corpus_with_thresholds(&prepared.units, &selected, &per_label);
    assert_eq!((macro_f1, micro_f1), (metrics.macro_f1, metrics.micro_f1));
}

#[test]
#[ignore = "requires the actual pinned WILDS CSV"]
fn training_source_matches_the_verified_wilds_file() {
    let dir = std::env::var("TURBO_CIVIL_COMMENTS").expect("TURBO_CIVIL_COMMENTS names the CSV directory");
    let record = source::verify(std::path::Path::new(&dir)).unwrap();
    assert_eq!(record["zoo_id"], "civil-comments-wilds");
    assert_eq!(record["size_bytes"], 272_542_670_u64);
    assert_eq!(record["revision"], "3fbfeca80bad0f3aec37e72fa07eff222b6e752f");
}
