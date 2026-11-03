//! Learned and rule aspect sentiment: triplets extracted, CPU and device
//! matching.

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{AspectTarget, Aspects, Opinions, Polarity, Sentences, Subwords, Tokens, Vectors};
use turbo_tokenize::SimpleTokenizer;

use heads_support::*;

#[test]
fn learned_and_rule_aspect_sentiment_extract_triplets_and_match_cpu_device() {
    use turbo_analysis_embed::{LabeledAbsaDoc, LearnedAspectSentiment, SubwordVectors};
    use turbo_doccat::{
        AbsaTriplet, DEFAULT_ASPECT_CATEGORIES, RuleAspectSentiment, evaluate_absa_corpus, extract_triplets,
    };

    let (_sub, vec_ann) = open_cpu(64);

    // 10-sentence benchmark covering multi-aspect contrast ("battery life is great
    // but the screen scratches"), laptop/hardware, restaurant/hospitality,
    // negation, and implicit aspects.
    struct BenchRow<'a> {
        text: &'a str,
        aspects: &'a [(IxRange<Tokens>, &'a str, &'a str)],
        opinions: &'a [(IxRange<Tokens>, &'a str)],
        gold: &'a [AbsaTriplet],
    }

    let benchmark: &[BenchRow<'_>] = &[
        BenchRow {
            text: "battery life is great but the screen scratches",
            aspects: &[(IxRange::new(0, 2), "battery", "positive"), (IxRange::new(6, 7), "display", "negative")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(7, 8), "negative")],
            gold: &[
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
        },
        BenchRow {
            text: "The pizza was delicious and the staff was friendly.",
            aspects: &[(IxRange::new(1, 2), "food", "positive"), (IxRange::new(6, 7), "service", "positive")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(8, 9), "positive")],
            gold: &[
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
        },
        BenchRow {
            text: "The keyboard is responsive, however the fan noise is loud.",
            aspects: &[(IxRange::new(1, 2), "build", "positive"), (IxRange::new(7, 9), "performance", "negative")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(10, 11), "negative")],
            gold: &[
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
        },
        BenchRow {
            text: "The customer service was not helpful.",
            aspects: &[(IxRange::new(1, 3), "service", "negative")],
            opinions: &[(IxRange::new(5, 6), "positive")], // negated in text -> effective negative
            gold: &[AbsaTriplet {
                aspect_tokens: Some((1, 3)),
                sentence: 0,
                opinion_tokens: (5, 6),
                polarity: Polarity::Negative,
            }],
        },
        BenchRow {
            text: "The build quality is sturdy and the trackpad is smooth.",
            aspects: &[(IxRange::new(1, 3), "build", "positive"), (IxRange::new(7, 8), "build", "positive")],
            opinions: &[(IxRange::new(4, 5), "positive"), (IxRange::new(9, 10), "positive")],
            gold: &[
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
        },
        BenchRow {
            text: "The pasta was soggy and the waiter was rude.",
            aspects: &[(IxRange::new(1, 2), "food", "negative"), (IxRange::new(6, 7), "service", "negative")],
            opinions: &[(IxRange::new(3, 4), "negative"), (IxRange::new(8, 9), "negative")],
            gold: &[
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
        },
        BenchRow {
            text: "The ambience was cozy and the wine list was generous.",
            aspects: &[(IxRange::new(1, 2), "ambience", "positive"), (IxRange::new(6, 8), "food", "positive")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(9, 10), "positive")],
            gold: &[
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
        },
        BenchRow {
            text: "The software is buggy and the interface is clunky.",
            aspects: &[(IxRange::new(1, 2), "software", "negative"), (IxRange::new(6, 7), "software", "negative")],
            opinions: &[(IxRange::new(3, 4), "negative"), (IxRange::new(8, 9), "negative")],
            gold: &[
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
        },
        BenchRow {
            text: "Way too expensive and completely overpriced.",
            aspects: &[],
            opinions: &[(IxRange::new(2, 3), "negative"), (IxRange::new(5, 6), "negative")],
            gold: &[
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (2, 3), polarity: Polarity::Negative },
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (5, 6), polarity: Polarity::Negative },
            ],
        },
    ];

    // Prepare tokenized + subword-segmented documents for fitting the learned heads.
    let prep_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .build()
        .unwrap();

    let mut prep_docs: Vec<Document> =
        benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
    for res in prep_pipe.run(&mut prep_docs) {
        res.unwrap();
    }

    let labeled_docs: Vec<LabeledAbsaDoc<'_>> = benchmark
        .iter()
        .zip(&prep_docs)
        .map(|(row, doc)| LabeledAbsaDoc {
            subwords: doc.get::<Subwords>().unwrap(),
            sentences: doc.get::<Sentences>().unwrap(),
            tokens: doc.get::<Tokens>().unwrap(),
            aspects: row.aspects,
            opinions: row.opinions,
        })
        .collect();

    let learned =
        LearnedAspectSentiment::fit(Arc::clone(&vec_ann), DEFAULT_ASPECT_CATEGORIES, &labeled_docs, 1e-5, 600, 5.0)
            .unwrap();

    let rule_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();

    let learned_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .add(SubwordVectors(Arc::clone(&vec_ann)))
        .add(learned.clone())
        .build()
        .unwrap();

    let mut rule_docs: Vec<Document> =
        benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
    for res in rule_pipe.run(&mut rule_docs) {
        res.unwrap();
    }

    let mut learned_docs: Vec<Document> =
        benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
    for res in learned_pipe.run(&mut learned_docs) {
        res.unwrap();
    }

    // Verify provenance distinguishes the rule path from the learned path while
    // writing the exact same `(Opinions, Aspects)` layer types.
    let rule_prov = rule_docs[0].provenance(Key::<Aspects>::DEFAULT.id()).unwrap();
    let learned_prov = learned_docs[0].provenance(Key::<Aspects>::DEFAULT.id()).unwrap();
    assert_eq!(rule_prov.annotator, "turbo-doccat:aspect-sentiment-rule");
    assert_eq!(rule_prov.model.as_ref().unwrap().id, "double-propagation:vader");
    assert_eq!(learned_prov.annotator, "turbo:aspect-sentiment-learned");
    assert_eq!(learned_prov.model.as_ref().unwrap().id, vec_ann.model().id);
    assert_eq!(learned_prov.execution.as_ref().unwrap().backend, "cpu");

    // Verify `"battery life is great but the screen scratches"` on both paths!
    for (label, doc) in [("rule", &rule_docs[0]), ("learned", &learned_docs[0])] {
        let asps = doc.get::<Aspects>().unwrap();
        let ops = doc.get::<Opinions>().unwrap();
        assert_eq!(asps.len(), 2, "{label}: expected 2 aspects");
        assert_eq!(ops.len(), 2, "{label}: expected 2 opinions");
        let a0 = asps.get(Ix::new(0));
        let a1 = asps.get(Ix::new(1));
        assert_eq!(
            a0.target,
            AspectTarget::Term { tokens: IxRange::new(0, 2), sentence: Some(Ix::new(0)) },
            "{label}: aspect 0 target"
        );
        assert_eq!(a0.polarity.label(), Polarity::Positive, "{label}: aspect 0 polarity");
        assert_eq!(a0.opinions, &[Ix::<Opinions>::new(0)], "{label}: aspect 0 opinions");

        assert_eq!(
            a1.target,
            AspectTarget::Term { tokens: IxRange::new(6, 7), sentence: Some(Ix::new(0)) },
            "{label}: aspect 1 target"
        );
        assert_eq!(a1.polarity.label(), Polarity::Negative, "{label}: aspect 1 polarity");
        assert_eq!(a1.opinions, &[Ix::<Opinions>::new(1)], "{label}: aspect 1 opinions");
    }

    // Verify implicit aspect sentence (`"Way too expensive and completely overpriced."`) on both paths.
    let last_idx = benchmark.len() - 1;
    for (label, doc) in [("rule", &rule_docs[last_idx]), ("learned", &learned_docs[last_idx])] {
        let asps = doc.get::<Aspects>().unwrap();
        assert_eq!(asps.len(), 1, "{label}: expected 1 implicit aspect");
        let a0 = asps.get(Ix::new(0));
        assert_eq!(a0.target, AspectTarget::Implicit { sentence: Ix::new(0) }, "{label}: implicit target");
        assert_eq!(a0.polarity.label(), Polarity::Negative, "{label}: implicit polarity");
    }

    let gold_corpus: Vec<Vec<AbsaTriplet>> = benchmark.iter().map(|r| r.gold.to_vec()).collect();
    let rule_pred: Vec<Vec<AbsaTriplet>> =
        rule_docs.iter().map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap())).collect();
    let learned_pred: Vec<Vec<AbsaTriplet>> = learned_docs
        .iter()
        .map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap()))
        .collect();

    let rule_metrics = evaluate_absa_corpus(&gold_corpus, &rule_pred);
    let learned_metrics = evaluate_absa_corpus(&gold_corpus, &learned_pred);
    println!(
        "ABSA training fixture (9 docs): Rule [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}], Learned [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}]",
        rule_metrics.aspect_f1,
        rule_metrics.opinion_f1,
        rule_metrics.triplet_f1,
        learned_metrics.aspect_f1,
        learned_metrics.opinion_f1,
        learned_metrics.triplet_f1,
    );
    assert!(rule_metrics.aspect_f1 >= 0.95 && rule_metrics.triplet_f1 >= 0.95);
    assert!(learned_metrics.aspect_f1 >= 0.90 && learned_metrics.triplet_f1 >= 0.90);

    // Evaluate both paths on a disjoint 4-sentence held-out set not seen during `LearnedAspectSentiment::fit`.
    // Note: `tiny-bert-bundle` is a 32-dim synthetic fixture with uncalibrated positional weights,
    // whereas `RuleAspectSentiment` uses the lexicon + propagation rules.
    let held_out: &[(&str, &[AbsaTriplet])] = &[
        (
            "The keyboard is sturdy and the trackpad is smooth.",
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
            "The pizza was soggy and the waiter was rude.",
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
            "The customer service was not friendly.",
            &[AbsaTriplet {
                aspect_tokens: Some((1, 3)),
                sentence: 0,
                opinion_tokens: (5, 6),
                polarity: Polarity::Negative,
            }],
        ),
        (
            "Way too expensive and completely overpriced!",
            &[
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (2, 3), polarity: Polarity::Negative },
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (5, 6), polarity: Polarity::Negative },
            ],
        ),
    ];
    let mut held_rule_docs: Vec<Document> =
        held_out.iter().map(|(text, _)| make_doc_with_sentences(text, &[(0, text.len() as u32)])).collect();
    for res in rule_pipe.run(&mut held_rule_docs) {
        res.unwrap();
    }
    let mut held_learned_docs: Vec<Document> =
        held_out.iter().map(|(text, _)| make_doc_with_sentences(text, &[(0, text.len() as u32)])).collect();
    for res in learned_pipe.run(&mut held_learned_docs) {
        res.unwrap();
    }
    let held_gold: Vec<Vec<AbsaTriplet>> = held_out.iter().map(|(_, g)| g.to_vec()).collect();
    let held_rule_pred: Vec<Vec<AbsaTriplet>> = held_rule_docs
        .iter()
        .map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap()))
        .collect();
    let held_learned_pred: Vec<Vec<AbsaTriplet>> = held_learned_docs
        .iter()
        .map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap()))
        .collect();
    let held_rule_m = evaluate_absa_corpus(&held_gold, &held_rule_pred);
    let held_learned_m = evaluate_absa_corpus(&held_gold, &held_learned_pred);
    println!(
        "ABSA held-out (4 docs): Rule [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}], Learned [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}]",
        held_rule_m.aspect_f1,
        held_rule_m.opinion_f1,
        held_rule_m.triplet_f1,
        held_learned_m.aspect_f1,
        held_learned_m.opinion_f1,
        held_learned_m.triplet_f1,
    );
    assert!(held_rule_m.aspect_f1 >= 0.95 && held_rule_m.triplet_f1 >= 0.95);
    assert!(held_learned_m.aspect_f1 >= 0.90 && held_learned_m.triplet_f1 >= 0.85);

    // LearnedAspectSentiment::annotate refuses layers from a different view even when
    // precomputed Vectors<Subwords> are supplied.
    {
        let doc0 = &learned_docs[0];
        let src_view = doc0.get::<View>().unwrap();
        let (clean_text, align) = ViewBuilder::new(src_view.text()).finish();
        let child_view = View::derived(src_view, clean_text, align);
        let err = learned
            .annotate((
                &child_view,
                doc0.get::<Sentences>().unwrap(),
                doc0.get::<Tokens>().unwrap(),
                doc0.get::<Subwords>().unwrap(),
                doc0.get::<Vectors<Subwords>>(),
                None,
                None,
            ))
            .unwrap_err();
        assert!(err.to_string().contains("not in the given view"), "{err}");
    }

    // When TURBO_TEST_DEVICE or TURBO_TEST_REQUIRE_CUDA=1 is set, verify CPU vs GPU parity for LearnedAspectSentiment.
    let want = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(w) => Some(w),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1") => Some("cuda".to_string()),
        Err(_) => None,
    };
    if let Some(want) = want {
        let device = match want.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(want.clone()),
        };
        let dev_opts = VectorOptions {
            encoder: EncoderOptions { device, max_seq: 64, ..EncoderOptions::default() },
            ..VectorOptions::default()
        };
        let dev_vec = Arc::new(
            VectorAnnotator::open(&bert(), &dev_opts).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")),
        );
        let mut dev_learned = learned.clone();
        dev_learned.vectors = Arc::clone(&dev_vec);

        let dev_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(SimpleTokenizer::default())
            .add(open_subwords())
            .add(SubwordVectors(Arc::clone(&dev_vec)))
            .add(dev_learned)
            .build()
            .unwrap();

        let mut dev_docs: Vec<Document> =
            benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
        for res in dev_pipe.run(&mut dev_docs) {
            res.unwrap();
        }
        let dev_prov = dev_docs[0].provenance(Key::<Aspects>::DEFAULT.id()).unwrap();
        assert_ne!(dev_prov.execution.as_ref().unwrap().backend, "cpu");

        for (cpu_doc, dev_doc) in learned_docs.iter().zip(&dev_docs) {
            let cpu_asps = cpu_doc.get::<Aspects>().unwrap();
            let dev_asps = dev_doc.get::<Aspects>().unwrap();
            assert_eq!(cpu_asps.len(), dev_asps.len());
            for i in 0..cpu_asps.len() {
                let ix = Ix::new(i as u32);
                let ca = cpu_asps.get(ix);
                let da = dev_asps.get(ix);
                assert_eq!(ca.target, da.target);
                assert_eq!(ca.category, da.category);
                assert_eq!(ca.opinions, da.opinions);
                assert!((ca.polarity.positive - da.polarity.positive).abs() <= 1e-5);
                assert!((ca.polarity.negative - da.polarity.negative).abs() <= 1e-5);
                assert!((ca.polarity.neutral - da.polarity.neutral).abs() <= 1e-5);
            }
        }
    }
}
