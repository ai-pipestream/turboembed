use super::*;

#[test]
fn absa_benchmark_f1_for_aspect_terms_and_triplets() {
    // Ten project-written sentences covering laptop/hardware, restaurant/hospitality,
    // multi-aspect contrast, negation, and implicit aspects. The rules were written
    // beside these sentences, so this is a regression fixture, not a benchmark.
    let benchmark: &[(&str, &[AbsaTriplet])] = &[
        (
            "battery life is great but the screen scratches",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((0, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (7, 8),
                    polarity: Polarity::Negative,
                },
            ],
        ),
        (
            "The pizza was delicious and the staff was friendly.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Positive,
                },
            ],
        ),
        (
            "The keyboard is responsive, however the fan noise is loud.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((7, 9)),
                    sentence: 0,
                    opinion_tokens: (10, 11),
                    polarity: Polarity::Negative,
                },
            ],
        ),
        (
            "The customer service was not helpful.",
            &[AbsaTriplet {
                aspect_tokens: Some((1, 3)),
                sentence: 0,
                opinion_tokens: (5, 6),
                polarity: Polarity::Negative,
            }],
        ),
        (
            "The build quality is sturdy and the trackpad is smooth.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 3)),
                    sentence: 0,
                    opinion_tokens: (4, 5),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((7, 8)),
                    sentence: 0,
                    opinion_tokens: (9, 10),
                    polarity: Polarity::Positive,
                },
            ],
        ),
        (
            "The pasta was soggy and the waiter was rude.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Negative,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Negative,
                },
            ],
        ),
        (
            "The camera takes crisp photos, but the speakers are tinny.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((4, 5)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((8, 9)),
                    sentence: 0,
                    opinion_tokens: (10, 11),
                    polarity: Polarity::Negative,
                },
            ],
        ),
        (
            "The ambience was cozy and the wine list was generous.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 8)),
                    sentence: 0,
                    opinion_tokens: (9, 10),
                    polarity: Polarity::Positive,
                },
            ],
        ),
        (
            "The software is buggy and the interface is clunky.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Negative,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Negative,
                },
            ],
        ),
        (
            "Way too expensive and completely overpriced.",
            &[
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (2, 3), polarity: Polarity::Negative },
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (5, 6), polarity: Polarity::Negative },
            ],
        ),
    ];

    let mut gold_corpus = Vec::new();
    let mut pred_corpus = Vec::new();
    for &(text, gold_triplets) in benchmark {
        let doc = run_rule_absa(text);
        let ops = doc.get::<Opinions>().unwrap();
        let asps = doc.get::<Aspects>().unwrap();
        gold_corpus.push(gold_triplets.to_vec());
        pred_corpus.push(extract_triplets(ops, asps));
    }

    let metrics = evaluate_absa_corpus(&gold_corpus, &pred_corpus);
    assert!(
        metrics.aspect_f1 >= 0.95,
        "expected aspect F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
        metrics.aspect_f1,
        metrics.aspect_precision,
        metrics.aspect_recall
    );
    assert!(
        metrics.opinion_f1 >= 0.95,
        "expected opinion F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
        metrics.opinion_f1,
        metrics.opinion_precision,
        metrics.opinion_recall
    );
    assert!(
        metrics.triplet_f1 >= 0.95,
        "expected triplet F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
        metrics.triplet_f1,
        metrics.triplet_precision,
        metrics.triplet_recall
    );
}

#[test]
fn foreign_view_layers_are_rejected() {
    let doc = Document::new("The battery is great.");
    let view = doc.get::<View>().unwrap();
    let (clean_text, align) = turbo_analysis::ViewBuilder::new(view.text()).finish();
    let child_view = View::derived(view, clean_text, align);
    let foreign = child_view.id();

    let mut ok_sent_spans = Spans::new(ViewId::SOURCE);
    ok_sent_spans.push(Span::new(0, 21));
    let ok_sentences = Sentences { spans: ok_sent_spans };
    let bad_sentences = Sentences {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 21));
            s
        },
    };
    let mut ok_tok_spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 11), (12, 14), (15, 20), (20, 21)] {
        ok_tok_spans.push(Span::new(a, b));
    }
    let ok_tokens = Tokens { spans: ok_tok_spans, sentence: None };
    let bad_tokens = Tokens {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 3));
            s
        },
        sentence: None,
    };
    let bad_chunks = PhraseChunks {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 11));
            s
        },
        tokens: vec![IxRange::new(0, 2)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("chunk", &["NP"]),
    };

    let rule = RuleAspectSentiment::new();
    let err = rule.annotate((view, &bad_sentences, &ok_tokens, None, None, None)).unwrap_err();
    assert!(err.to_string().contains("sentences are in another view"), "{err}");

    let err = rule.annotate((view, &ok_sentences, &bad_tokens, None, None, None)).unwrap_err();
    assert!(err.to_string().contains("tokens are in another view"), "{err}");

    let err = rule.annotate((view, &ok_sentences, &ok_tokens, None, Some(&bad_chunks), None)).unwrap_err();
    assert!(err.to_string().contains("phrase chunks are in another view"), "{err}");
}

#[test]
fn config_hash_is_deterministic_and_pinned() {
    let base = RuleAspectSentiment::new().config_hash();
    let no_implicit = RuleAspectSentiment::new().with_implicit_aspects(false).config_hash();
    let custom_cats = RuleAspectSentiment::new().with_categories(TagSet::new("custom", &["a", "b"])).config_hash();
    assert_eq!(base, 0xac9c_00d5_eeec_addd, "got 0x{base:016x}");
    assert_ne!(base, no_implicit);
    assert_ne!(base, custom_cats);
    assert_ne!(no_implicit, custom_cats);
}
