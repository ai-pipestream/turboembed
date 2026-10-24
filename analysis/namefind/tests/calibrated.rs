use std::sync::Arc;
use turbo_analysis::export::{export, import};
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Calibrated, Entities, Sentences, Tokens, export::codecs};
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest, metrics};
use turbo_maxent::{JavaString, ModelFile};
use turbo_namefind::calibrated::{CalibratedCrfNameAnnotator, CalibratedNameAnnotator, crf_identity, identity};
use turbo_sequence::{
    PatternMode,
    crf::CrfSequenceModel,
    featuregen::descriptor,
    namefind::{
        CODEC_PROPERTY, CrfNameFinder, DEFAULT_DESCRIPTOR, DefaultNameContext, MODEL_ENTRY_NAME, NameCodec, NameFinder,
        crf::{MODEL_ENTRY, MODEL_TYPE},
    },
};
use turbo_train::{
    Event, Key as TrainKey, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters,
    calibration::{Method, Samples, fit},
    crf::{CrfTrainer, CrfTrainingSequence},
};

fn file() -> Arc<ModelFile> {
    let events = [
        Event::from_strs("person-start", &["w=Alice"]),
        Event::from_strs("person-start", &["w=Bob"]),
        Event::from_strs("other", &["w=walked"]),
        Event::from_strs("other", &["w=home"]),
    ];
    let mut params = TrainingParameters::new();
    params.put(TrainKey::ITERATIONS, 50);
    params.put(TrainKey::CUTOFF, 0);
    let (trained, _) = turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut package = ModelPackage::new(Manifest::new("TokenNameFinder", "en", 1700000000000));
    package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&trained).unwrap());
    Arc::new(ModelFile::open(package.to_bytes()).unwrap())
}
fn artifact(file: &ModelFile) -> Calibration {
    Calibration::new(
        identity(file, "news").unwrap(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.1; 20],
    )
    .unwrap()
}
fn inputs() -> (Document, Sentences, Tokens) {
    let doc = Document::new("Alice walked");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 12));
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 5));
    t.push(Span::new(6, 12));
    (doc, Sentences { spans: s }, Tokens { spans: t, sentence: None })
}
#[test]
fn scored_search_preserves_names_probabilities_and_adaptive_state() {
    let f = file();
    let model = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let mut raw = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    let mut scored = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    let mut flat = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    for words in [vec!["Alice", "walked"], vec!["Bob", "Alice", "home"], vec![], vec!["Alice"]] {
        let a = raw.find(&words);
        let (b, rows) = scored.find_scored(&words);
        let (c, flat_rows) = flat.find_scored_flat(&words);
        assert_eq!(a, c);
        assert_eq!(flat_rows, rows.concat());
        assert_eq!(raw.probs(), flat.probs());
        assert_eq!(a, b);
        assert_eq!(raw.probs(), scored.probs());
        assert_eq!(rows.len(), words.len());
        for row in rows {
            turbo_calibration::check_distribution(&row).unwrap();
        }
    }
}
#[test]
fn name_annotator_preserves_raw_entities_and_validates_boundaries() {
    let f = file();
    let a = artifact(&f);
    let model = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let mut raw = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    let expected = raw.find(&["Alice", "walked"]);
    let ann = CalibratedNameAnnotator::new(
        f.clone(),
        a.clone(),
        "news",
        Policy { coverage: 0.9, max_set_size: 1, min_confidence: 1.0 },
    )
    .unwrap();
    let (doc, s, mut t) = inputs();
    let v = doc.get::<View>().unwrap();
    let (entities, side) = ann.annotate((v, &s, &t)).unwrap();
    side.check().unwrap();
    assert_eq!(side.len(), 2);
    assert_eq!(entities.len(), expected.len());
    for (i, n) in expected.iter().enumerate() {
        assert_eq!(entities.score[i], n.prob);
        assert_eq!(entities.tokens[i].start.get(), n.span.start);
    }
    assert!(side.rows().all(|r| r.abstention.is_some()));
    t.spans = Spans::new(ViewId(99));
    assert!(ann.annotate((v, &s, &t)).is_err());
    let bad = Document::new("é");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    let mut sent = Spans::new(ViewId::SOURCE);
    sent.push(Span::new(0, 2));
    assert!(
        ann.annotate((bad.get::<View>().unwrap(), &Sentences { spans: sent }, &Tokens { spans, sentence: None }))
            .is_err()
    );
    assert!(
        CalibratedNameAnnotator::new(
            f,
            a,
            "other-domain",
            Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.0 }
        )
        .is_err()
    );
}
#[test]
fn heldout_token_calibration_receipt() {
    let f = file();
    let model = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let generate = |mut seed: u64, n: usize| {
        let mut finder = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
        let mut rows = Vec::new();
        let mut gold = Vec::new();
        for i in 0..n {
            finder.clear_adaptive_data();
            let words = if i % 2 == 0 { ["Alice"] } else { ["walked"] };
            let (_, mut scored) = finder.find_scored(&words);
            let row = scored.pop().unwrap();
            let best = usize::from(row[1] > row[0]);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            gold.push(best ^ usize::from((seed >> 32) % 10 >= 7));
            rows.push(row);
        }
        (rows, gold)
    };
    let (fitting, fy) = generate(1, 300);
    let (conf, cy) = generate(123, 301);
    let (test, ty) = generate(998, 1000);
    let id = identity(&f, "synthetic-tokens").unwrap();
    let a = fit(
        id.clone(),
        Method::Temperature,
        Samples { split: "fit", probabilities: &fitting, gold: &fy },
        Samples { split: "conformal", probabilities: &conf, gold: &cy },
    )
    .unwrap();
    let p = turbo_calibration::Predictor::new(a, &id, Policy { coverage: 0.99, max_set_size: 2, min_confidence: 0.0 })
        .unwrap();
    let predictions: Vec<_> = test.iter().map(|r| p.predict(r).unwrap()).collect();
    let before = metrics::reliability(&test, &ty, 10).unwrap();
    let after =
        metrics::reliability(&predictions.iter().map(|r| r.calibrated.clone()).collect::<Vec<_>>(), &ty, 10).unwrap();
    let c = metrics::coverage(&predictions, &ty).unwrap();
    assert!(after.ece < before.ece);
    assert!((c.set_coverage - 0.99).abs() <= 0.03);
    println!(
        "NameFinderME: ECE {:.6} -> {:.6}; Brier {:.6} -> {:.6}; coverage {:.6}",
        before.ece, after.ece, before.brier, after.brier, c.set_coverage
    );
}

#[test]
fn crf_span_distributions_match_exhaustive_path_sums_bio_and_bilou() {
    for (codec, raw_labels) in [
        (NameCodec::Bio, vec!["other", "loc-start", "loc-cont", "per-start", "per-cont"]),
        (
            NameCodec::Bilou,
            vec![
                "other",
                "loc-start",
                "loc-cont",
                "loc-last",
                "loc-unit",
                "per-start",
                "per-cont",
                "per-last",
                "per-unit",
            ],
        ),
    ] {
        let labels: Vec<String> = raw_labels.into_iter().map(str::to_owned).collect();
        let n = labels.len();
        let features = vec![JavaString::new("f0"), JavaString::new("f1")];
        let count = CrfSequenceModel::parameter_count(n, features.len()).unwrap();
        let weights: Vec<f64> = (0..count).map(|i| ((i as f64 + 1.0) * 0.37).sin() * 0.6).collect();
        let model = CrfSequenceModel::new(labels.clone(), features.clone(), codec, weights).unwrap();
        assert_eq!(model.span_outcomes(), ["other", "loc", "per"]);
        let rows = vec![
            vec![(features[0].clone(), 1.0)],
            vec![(features[1].clone(), 0.8)],
            vec![(features[0].clone(), -0.5), (features[1].clone(), 1.2)],
            vec![(features[1].clone(), -0.4)],
        ];
        let lattice = model.lattice(&rows).unwrap();
        let paths = lattice.best_paths(4096).unwrap();
        let total_prob: f64 = paths.iter().map(|p| p.log_probability.exp()).sum();
        assert!((total_prob - 1.0).abs() < 1e-12);
        for start in 0..rows.len() {
            for end in (start + 1)..=rows.len() {
                let dist = model.span_distribution(&lattice, start, end).unwrap();
                turbo_calibration::check_distribution(&dist).unwrap();
                let mut exact = [0.0f64; 3];
                for p in &paths {
                    let outcomes: Vec<&str> = p.labels.iter().map(|&y| labels[y].as_str()).collect();
                    let decoded = codec.decode(&outcomes);
                    let prob = p.log_probability.exp();
                    if let Some(ent) = decoded.iter().find(|s| s.start == start && s.end == end) {
                        let k = if ent.kind.as_deref() == Some("loc") { 1 } else { 2 };
                        exact[k] += prob;
                    } else {
                        exact[0] += prob;
                    }
                }
                for (got, want) in dist.iter().zip(exact) {
                    assert!((got - want).abs() < 1e-11, "{codec:?} {start}..{end}: {dist:?} vs {exact:?}");
                }
            }
        }
    }
}

fn crf_file(codec: NameCodec) -> Arc<ModelFile> {
    let mut ctx = DefaultNameContext::new(
        vec![descriptor::build(DEFAULT_DESCRIPTOR.as_bytes(), None, PatternMode::Utf16Units).unwrap()],
        PatternMode::Utf16Units,
    );
    let s1 = ["Alice", "Smith", "visited", "Paris", "."];
    let o1 = codec.encode(
        &[
            turbo_sequence::codec::Span::new(0, 2, Some("person")),
            turbo_sequence::codec::Span::new(3, 4, Some("place")),
        ],
        5,
    );
    let s2 = ["Bob", "Jones", "walked", "home", "."];
    let o2 = codec.encode(&[turbo_sequence::codec::Span::new(0, 2, Some("person"))], 5);
    let seqs = vec![
        CrfTrainingSequence { features: ctx.sequence_features(&s1, &[]), outcomes: o1 },
        CrfTrainingSequence { features: ctx.sequence_features(&s2, &[]), outcomes: o2 },
    ];
    let model = CrfTrainer { codec, cutoff: 1, l2: 0.1, iterations: 60 }.train(&seqs).unwrap();
    let mut manifest = Manifest::new("TokenNameFinder", "en", 1700000000000);
    manifest.set("Model-Type", MODEL_TYPE);
    manifest.set("Algorithm", "CRF");
    manifest.set(CODEC_PROPERTY, codec.class_name());
    let mut pkg = ModelPackage::new(manifest);
    pkg.add_entry(MODEL_ENTRY, model.to_bytes());
    Arc::new(ModelFile::open(pkg.to_bytes()).unwrap())
}

#[test]
fn crf_span_annotator_preserves_alignment_and_round_trips_document_graph() {
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        let f = crf_file(codec);
        let id = crf_identity(&f, "news").unwrap();
        assert_eq!(id.labels, ["other", "person", "place"]);
        let cal = Calibration::new(
            id.clone(),
            Transform::Temperature(1.5),
            DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
            DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
            vec![0.05, 0.15, 0.25],
        )
        .unwrap();
        assert!(
            CalibratedNameAnnotator::new(
                file(),
                cal.clone(),
                "news",
                Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 },
            )
            .is_err()
        );
        let ann = CalibratedCrfNameAnnotator::new(
            f.clone(),
            cal,
            "news",
            Policy { coverage: 0.75, max_set_size: 1, min_confidence: 0.5 },
        )
        .unwrap();
        let text = "Alice Smith visited Paris .";
        let mut doc = Document::new(text);
        let mut sentences = Sentences { spans: Spans::new(ViewId::SOURCE) };
        sentences.spans.push(Span::new(0, text.len() as u32));
        let mut tokens = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: Some(vec![Ix::new(0); 5]) };
        for (a, b) in [(0, 5), (6, 11), (12, 19), (20, 25), (26, 27)] {
            tokens.spans.push(Span::new(a, b));
        }
        doc.add(&Key::DEFAULT, sentences, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(ann)
            .build()
            .unwrap();
        let mut docs = [doc];
        analyzer.run(&mut docs)[0].as_ref().unwrap();
        let entities = docs[0].get::<Entities>().unwrap();
        let side = docs[0].get::<Calibrated<Entities>>().unwrap();
        let toks = docs[0].get::<Tokens>().unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(side.len(), entities.len());
        for i in 0..entities.len() {
            let span = entities.spans.get(Ix::new(i as u32));
            let tr = entities.tokens[i];
            assert_eq!(span.start, toks.spans.get(tr.start).start);
            assert_eq!(span.end, toks.spans.get(Ix::new(tr.end.get() as u32 - 1)).end);
            let row = side.row(i);
            turbo_calibration::check_distribution(row.raw).unwrap();
            turbo_calibration::check_distribution(row.calibrated).unwrap();
            let kind_label = entities.kinds.label(entities.kind[i]);
            let kind_ix = id.labels.iter().position(|l| l == kind_label).unwrap();
            assert!((row.raw[kind_ix] - entities.score[i]).abs() < 1e-12);
        }
        let graph = export(&docs[0], &codecs()).unwrap();
        let restored = import(&graph, &codecs()).unwrap();
        assert_eq!(
            restored.get::<Calibrated<Entities>>().unwrap().rows().collect::<Vec<_>>(),
            side.rows().collect::<Vec<_>>(),
        );
    }
}

#[test]
fn germeval_real_corpus_token_and_crf_span_calibration_receipts() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/inputs/ner/germeval-excerpt.txt");
    let raw = std::fs::read_to_string(path).unwrap();
    assert_eq!(digest(raw.as_bytes()), "0bc6a217acfc21773d49b2ab3861e0cd318c2ce825374a4aba5b242571eaa53c");
    let mut sentences: Vec<(Vec<String>, Vec<turbo_sequence::codec::Span>)> = Vec::new();
    let mut cur_words: Vec<String> = Vec::new();
    let mut cur_tags: Vec<String> = Vec::new();
    for line in raw.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !cur_words.is_empty() {
                let refs: Vec<&str> = cur_tags.iter().map(String::as_str).collect();
                sentences.push((std::mem::take(&mut cur_words), NameCodec::Bio.decode(&refs)));
                cur_tags.clear();
            }
        } else if !line.starts_with('#') {
            let cols: Vec<&str> = line.split('\t').collect();
            cur_words.push(cols[1].to_owned());
            let tag = cols[2];
            let mapped = if let Some(rest) = tag.strip_prefix("B-") {
                format!("{}-start", rest[..3].to_ascii_lowercase())
            } else if let Some(rest) = tag.strip_prefix("I-") {
                format!("{}-cont", rest[..3].to_ascii_lowercase())
            } else {
                "other".to_owned()
            };
            cur_tags.push(mapped);
        }
    }
    assert_eq!(sentences.len(), 105);
    let mut train = Vec::new();
    let mut fit_set = Vec::new();
    let mut conf_set = Vec::new();
    let mut test_set = Vec::new();
    for (i, s) in sentences.into_iter().enumerate() {
        match i % 5 {
            0 | 1 => train.push(s),
            2 => fit_set.push(s),
            3 => conf_set.push(s),
            _ => test_set.push(s),
        }
    }
    let mut ctx = DefaultNameContext::new(
        vec![descriptor::build(DEFAULT_DESCRIPTOR.as_bytes(), None, PatternMode::Utf16Units).unwrap()],
        PatternMode::Utf16Units,
    );
    let crf_seqs: Vec<CrfTrainingSequence> = train
        .iter()
        .map(|(w, spans)| {
            let refs: Vec<&str> = w.iter().map(String::as_str).collect();
            CrfTrainingSequence {
                features: ctx.sequence_features(&refs, &[]),
                outcomes: NameCodec::Bio.encode(spans, refs.len()),
            }
        })
        .collect();
    let crf_model = CrfTrainer { codec: NameCodec::Bio, cutoff: 2, l2: 1.0, iterations: 60 }.train(&crf_seqs).unwrap();
    let mut manifest = Manifest::new("TokenNameFinder", "de", 1700000000000);
    manifest.set("Model-Type", MODEL_TYPE);
    manifest.set("Algorithm", "CRF");
    manifest.set(CODEC_PROPERTY, NameCodec::Bio.class_name());
    let mut pkg = ModelPackage::new(manifest);
    pkg.add_entry(MODEL_ENTRY, crf_model.to_bytes());
    let crf_file = ModelFile::open(pkg.to_bytes()).unwrap();
    let crf_id = crf_identity(&crf_file, "germeval-2014-excerpt").unwrap();
    let mut crf_finder = CrfNameFinder::from_file(&crf_model, &crf_file, PatternMode::Utf16Units).unwrap();
    let collect_spans = |finder: &mut CrfNameFinder<'_>, split: &[(Vec<String>, Vec<turbo_sequence::codec::Span>)]| {
        let mut probs = Vec::new();
        let mut gold = Vec::new();
        for (w, spans) in split {
            let refs: Vec<&str> = w.iter().map(String::as_str).collect();
            for (cand, dist) in finder.candidate_spans(&refs, 4).unwrap() {
                let g = spans
                    .iter()
                    .find(|s| s.start == cand.start && s.end == cand.end)
                    .and_then(|s| s.kind.as_deref())
                    .and_then(|k| crf_id.labels.iter().position(|l| l == k))
                    .unwrap_or(0);
                probs.push(dist);
                gold.push(g);
            }
        }
        (probs, gold)
    };
    let (f_rows, f_gold) = collect_spans(&mut crf_finder, &fit_set);
    let (c_rows, c_gold) = collect_spans(&mut crf_finder, &conf_set);
    let (t_rows, t_gold) = collect_spans(&mut crf_finder, &test_set);
    let raw_rel = metrics::reliability(&t_rows, &t_gold, 10).unwrap();
    println!(
        "GermEval CRF span (fit={}, conf={}, test={}): raw ECE {:.6}, Brier {:.6}",
        f_rows.len(),
        c_rows.len(),
        t_rows.len(),
        raw_rel.ece,
        raw_rel.brier
    );
    for method in [Method::Temperature, Method::Platt, Method::Isotonic] {
        let cal = fit(
            crf_id.clone(),
            method,
            Samples { split: "germeval-fit", probabilities: &f_rows, gold: &f_gold },
            Samples { split: "germeval-conformal", probabilities: &c_rows, gold: &c_gold },
        )
        .unwrap();
        let pred = turbo_calibration::Predictor::new(
            cal,
            &crf_id,
            Policy { coverage: 0.9, max_set_size: crf_id.labels.len(), min_confidence: 0.0 },
        )
        .unwrap();
        let preds: Vec<_> = t_rows.iter().map(|r| pred.predict(r).unwrap()).collect();
        let after =
            metrics::reliability(&preds.iter().map(|p| p.calibrated.clone()).collect::<Vec<_>>(), &t_gold, 10).unwrap();
        let cov = metrics::coverage(&preds, &t_gold).unwrap();
        assert!(after.ece.is_finite() && after.brier.is_finite());
        assert!(cov.set_coverage >= 0.80, "{method:?} coverage {:.6}", cov.set_coverage);
        println!(
            "GermEval CRF span {method:?}: ECE {:.6} -> {:.6}, Brier {:.6} -> {:.6}, set_cov {:.6}",
            raw_rel.ece, after.ece, raw_rel.brier, after.brier, cov.set_coverage
        );
    }
}
