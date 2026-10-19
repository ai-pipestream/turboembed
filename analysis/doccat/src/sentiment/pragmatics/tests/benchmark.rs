use super::*;

#[test]
fn pragmatics_benchmark_f1_and_evidence_invariants() {
    // Project-authored evaluation set (Apache-2.0, this repository) covering:
    // 1. Riloff positive-sentiment vs negative-situation incongruity
    // 2. Text vs emoji polarity mismatch
    // 3. Embedding-head-only implicit irony (no explicit contrast tokens)
    // 4. Negative controls (literal positive, literal negative, neutral, and
    //    adversative 'but'/'however' mixed-sentiment sentences).
    #[derive(Copy, Clone)]
    enum ExpectedKind {
        ExplicitContrast,
        HeadOnly,
        NotSarcastic,
    }

    let cases: &[(&str, [f32; 2], ExpectedKind)] = &[
        // Riloff incongruity positives (10)
        ("I just love waiting in line at the DMV for three hours.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
        ("Oh great, the server crashed right before our demo.", [0.9, 0.6], ExpectedKind::ExplicitContrast),
        ("So thrilled to be stuck in traffic on a Friday afternoon.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
        ("Nothing like a flat tire in the pouring rain.", [0.7, 0.5], ExpectedKind::ExplicitContrast),
        ("My flight was cancelled and they lost my luggage, wonderful.", [0.8, 0.4], ExpectedKind::ExplicitContrast),
        ("Just what I needed today, a root canal and a migraine.", [0.9, 0.5], ExpectedKind::ExplicitContrast),
        ("I love working all weekend on unpaid overtime.", [0.8, 0.6], ExpectedKind::ExplicitContrast),
        ("Oh fantastic, the wifi is down and the build failed.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
        ("Love being ignored by customer support after waiting on hold.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
        (
            "Yeah right, because a merge conflict at midnight is so much fun.",
            [0.9, 0.6],
            ExpectedKind::ExplicitContrast,
        ),
        // Text-against-emoji mismatch positives (4)
        ("This new user interface is truly brilliant 🙄.", [0.6, 0.4], ExpectedKind::ExplicitContrast),
        ("Another wonderful Monday morning meeting 😭.", [0.6, 0.4], ExpectedKind::ExplicitContrast),
        ("The production database was deleted and the server crashed 😂.", [0.7, 0.5], ExpectedKind::ExplicitContrast),
        ("I got food poisoning and missed my flight 🎉.", [0.7, 0.5], ExpectedKind::ExplicitContrast),
        // Embedding-head-only implicit irony positives (4)
        ("Well, that meeting certainly could not have been an email.", [2.2, 1.5], ExpectedKind::HeadOnly),
        ("How refreshingly original to blame the network again.", [2.0, 1.4], ExpectedKind::HeadOnly),
        ("Clearly the documentation wrote itself.", [2.1, 1.3], ExpectedKind::HeadOnly),
        ("Because adding another layer of indirection solves everything.", [2.3, 1.2], ExpectedKind::HeadOnly),
        // Literal positive, negative, neutral, and adversative controls (14)
        ("The new release is fast, reliable, and easy to deploy.", [-2.0, -1.5], ExpectedKind::NotSarcastic),
        ("We had a wonderful dinner and the staff was very kind.", [-2.1, -1.4], ExpectedKind::NotSarcastic),
        ("I love spending time with my family on the weekend 😊.", [-2.2, -1.6], ExpectedKind::NotSarcastic),
        ("The server crashed yesterday and caused an outage.", [-1.8, -1.2], ExpectedKind::NotSarcastic),
        ("My flight was delayed for two hours because of rain 😭.", [-1.9, -1.3], ExpectedKind::NotSarcastic),
        ("I have a terrible headache and need to rest.", [-2.0, -1.4], ExpectedKind::NotSarcastic),
        ("The battery life is great, but the screen is dim.", [-1.7, -1.2], ExpectedKind::NotSarcastic),
        ("The food was wonderful, however the service was slow.", [-1.8, -1.1], ExpectedKind::NotSarcastic),
        ("Although the hotel was nice, the street was noisy.", [-1.6, -1.3], ExpectedKind::NotSarcastic),
        ("The package arrived on Tuesday morning at nine.", [-2.2, -1.5], ExpectedKind::NotSarcastic),
        ("Please review the pull request before merging.", [-2.1, -1.4], ExpectedKind::NotSarcastic),
        ("We updated the configuration file and restarted the service.", [-2.0, -1.5], ExpectedKind::NotSarcastic),
        ("She passed the exam and celebrated with her friends 🎉.", [-2.3, -1.6], ExpectedKind::NotSarcastic),
        ("Traffic was heavy this morning on the bridge.", [-1.9, -1.4], ExpectedKind::NotSarcastic),
    ];

    let scorer = Arc::new(FixedMapVectorScorer { weights: vec![1.0, 1.0], bias: -1.2 });
    let annotator = PragmaticsAnnotator::new().with_vector_scorer(scorer);
    let threshold = annotator.positive_threshold();

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut tn = 0usize;

    for &(sentence_text, vec_row, expected) in cases {
        let mut doc = Document::new(sentence_text);
        annotate_sentences_and_tokens(&mut doc);
        let vecs = Vectors::<Sentences>::new(
            2,
            vec_row.to_vec(),
            Vec::new(),
            ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
        );
        doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .add(EmojiAnnotate::new())
            .step(LexiconSentiment::new().bound())
            .step(Bound::new(annotator.clone()).read(4, SENTENCE_SENTIMENT.name))
            .build()
            .unwrap();

        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];
        let tokens = doc.get::<Tokens>().unwrap();
        let prag = doc.get::<Pragmatics>().unwrap();
        assert_eq!(prag.len(), 1);
        let item = prag.get(Ix::new(0));
        let predicted_positive = item.score >= threshold;
        let gold_positive = !matches!(expected, ExpectedKind::NotSarcastic);

        match (gold_positive, predicted_positive) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
            (false, false) => tn += 1,
        }

        // Definition of done invariant: each positive carries its evidence
        // spans, or says it came from the head alone (`evidence == None`).
        match expected {
            ExpectedKind::ExplicitContrast => {
                assert!(
                    predicted_positive,
                    "expected positive on explicit contrast {sentence_text:?}, got score {}",
                    item.score
                );
                let (pos_sp, neg_sp) = prag
                    .evidence_spans(Ix::new(0), tokens)
                    .unwrap_or_else(|| panic!("missing evidence spans on {sentence_text:?}"));
                assert!(!pos_sp.is_empty());
                assert!(!neg_sp.is_empty());
                assert_ne!(pos_sp, neg_sp);
            }
            ExpectedKind::HeadOnly => {
                assert!(
                    predicted_positive,
                    "expected positive from head on {sentence_text:?}, got score {}",
                    item.score
                );
                assert!(item.evidence.is_none(), "head-only positive must report None evidence on {sentence_text:?}");
            }
            ExpectedKind::NotSarcastic => {
                assert!(
                    !predicted_positive,
                    "unexpected false positive on {sentence_text:?}: score {}, evidence {:?}",
                    item.score, item.evidence
                );
            }
        }
    }

    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_count) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall);
    assert_eq!(tp, 18);
    assert_eq!(fp, 0);
    assert_eq!(fn_count, 0);
    assert_eq!(tn, 14);
    assert!((f1 - 1.0).abs() < 1e-12, "expected F1 = 1.0 on fixture, got {f1}");
}

#[test]
fn config_hash_is_deterministic_and_pinned() {
    let base = PragmaticsAnnotator::new().config_hash();
    let high_thresh = PragmaticsAnnotator::new().with_positive_threshold(0.8).config_hash();
    let with_scorer = PragmaticsAnnotator::new()
        .with_vector_scorer(Arc::new(FixedMapVectorScorer { weights: vec![1.0], bias: 0.0 }))
        .config_hash();
    assert_eq!(base, 0xa06e_b5ad_6004_8da4, "got 0x{base:016x}");
    assert_ne!(base, high_thresh);
    assert_ne!(base, with_scorer);
    assert_ne!(high_thresh, with_scorer);
}
