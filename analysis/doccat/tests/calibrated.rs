use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest, metrics};
use turbo_doccat::{calibrated::*, *};
use turbo_train::calibration::{Method, Samples, fit};

fn svm() -> SvmCategorizer {
    let model = "svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 2\nrho 0.25\nlabel 0 1\nprobA -2.0\nprobB 0.0\nnr_sv 1 1\nSV\n1.0 1:1.0\n-1.0 2:1.0\n";
    let side = "svm-doccat 1\nlanguage\teng\nweighting\tBINARY\nselection\tNONE\t-1\nscaling\tnone\ngenerator\tbag-of-words\ncategory\t0\tpos\ncategory\t1\tneg\nfeature\t1\t1.0\t0.0\t1.0\tbow=good\nfeature\t2\t1.0\t0.0\t1.0\tbow=bad\n";
    SvmCategorizer::new(SvmDoccatModel::from_text(model, side).unwrap()).unwrap()
}

fn svm_ngram(max_gram: usize) -> SvmCategorizer {
    let model = "svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 2\nrho 0.10\nlabel 0 1\nprobA -1.5\nprobB 0.0\nnr_sv 1 1\nSV\n1.0 1:1.0\n-1.0 2:1.0\n";
    let side = "svm-doccat 1\nlanguage\teng\nweighting\tBINARY\nselection\tNONE\t-1\nscaling\tnone\ngenerator\tngram\t1\t2\ncategory\t0\tpos\ncategory\t1\tneg\nfeature\t1\t1.0\t0.0\t1.0\tng=:good\nfeature\t2\t1.0\t0.0\t1.0\tng=:good:bad\n";
    let side = side.replace("generator\tngram\t1\t2", &format!("generator\tngram\t1\t{max_gram}"));
    SvmCategorizer::new(SvmDoccatModel::from_text(model, &side).unwrap()).unwrap()
}

fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, text.len() as u32));
    Tokens { spans, sentence: None }
}

fn artifact<C: CalibratableCategorizer>(cat: &C) -> Calibration {
    Calibration::new(
        identity(cat, "reviews").unwrap(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.1; 20],
    )
    .unwrap()
}

fn parity<C: CalibratableCategorizer>(cat: C) {
    let doc = Document::new("good");
    let view = doc.get::<View>().unwrap();
    let t = tokens("good");
    let (raw,) = cat.annotate((view, &t)).unwrap();
    let a = artifact(&cat);
    let loaded = Calibration::from_bytes(&a.to_bytes(), &identity(&cat, "reviews").unwrap()).unwrap();
    let wrap = CalibratedCategorizer::new(
        cat,
        loaded,
        "reviews",
        Policy { coverage: 0.9, max_set_size: 1, min_confidence: 1.0 },
    )
    .unwrap();
    let (unchanged, side) = wrap.annotate((view, &t)).unwrap();
    side.check().unwrap();
    assert_eq!(raw.iter().next().unwrap().scores, unchanged.iter().next().unwrap().scores);
    assert_eq!(side.row(0).raw, raw.iter().next().unwrap().scores);
    assert_ne!(side.row(0).raw, side.row(0).calibrated);
    assert!(side.row(0).abstention.is_some());
    assert_eq!(unchanged.len(), 1);
    assert_eq!(wrap.config_hash(), wrap.predictor().config_hash());
}

#[test]
fn raw_svm_outputs_survive_abstention() {
    parity(svm());
}

#[test]
fn wrong_domain_or_producer_cannot_bind() {
    let cat = svm();
    let a = artifact(&cat);
    assert!(
        CalibratedCategorizer::new(
            cat,
            a.clone(),
            "different",
            Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 }
        )
        .err()
        .unwrap()
        .contains(&a.id())
    );
    assert!(
        CalibratedCategorizer::new(
            svm_ngram(2),
            a,
            "reviews",
            Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 }
        )
        .is_err()
    );
}

fn heldout<C: CalibratableCategorizer>(cat: C) {
    // Genuine producer probabilities, with independent seeded noisy labels.
    // This small synthetic domain is a wiring/calibration proof, not a corpus accuracy claim.
    let generate = |mut seed: u64, n: usize| {
        let mut rows = Vec::new();
        let mut gold = Vec::new();
        for i in 0..n {
            let word = if i % 2 == 0 { "good" } else { "bad" };
            let doc = Document::new(word);
            let t = tokens(word);
            let (raw,) = cat.annotate((doc.get::<View>().unwrap(), &t)).unwrap();
            let p = raw.iter().next().unwrap().scores.to_vec();
            let best = usize::from(p[1] > p[0]);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            gold.push(best ^ usize::from((seed >> 32) % 10 >= 7));
            rows.push(p);
        }
        (rows, gold)
    };
    let (f, fy) = generate(1, 300);
    let (c, cy) = generate(123, 301);
    let (t, ty) = generate(998, 1000);
    let id = identity(&cat, "synthetic-reviews").unwrap();
    for method in [Method::Temperature, Method::Platt, Method::Isotonic] {
        let a = fit(
            id.clone(),
            method,
            Samples { split: "fit", probabilities: &f, gold: &fy },
            Samples { split: "conformal", probabilities: &c, gold: &cy },
        )
        .unwrap();
        let p =
            turbo_calibration::Predictor::new(a, &id, Policy { coverage: 0.99, max_set_size: 2, min_confidence: 0.0 })
                .unwrap();
        let predictions: Vec<_> = t.iter().map(|r| p.predict(r).unwrap()).collect();
        let before = metrics::reliability(&t, &ty, 10).unwrap();
        let after =
            metrics::reliability(&predictions.iter().map(|r| r.calibrated.clone()).collect::<Vec<_>>(), &ty, 10)
                .unwrap();
        let coverage = metrics::coverage(&predictions, &ty).unwrap();
        assert!(after.ece < before.ece, "{before:?} {after:?}");
        assert!((coverage.set_coverage - 0.99).abs() <= 0.03);
        println!(
            "{}: ECE {:.6} -> {:.6}; Brier {:.6} -> {:.6}; coverage {:.6}",
            id.model, before.ece, after.ece, before.brier, after.brier, coverage.set_coverage
        );
    }
}

#[test]
fn heldout_receipts_for_categorizer() {
    heldout(svm());
}

#[test]
fn producer_feature_configuration_is_bound() {
    // Both producers use identical SVM bytes and feature vocabulary. Only
    // the maximum n-gram length changes, so binding must identify that setting.
    let cat = svm_ngram(1);
    let original = identity(&cat, "reviews").unwrap();
    let changed = svm_ngram(2);
    let expected = identity(&changed, "reviews").unwrap();
    assert_eq!(original.model, expected.model);
    assert_eq!(original.revision, expected.revision);
    assert_eq!(original.labels, expected.labels);
    assert_ne!(original.sha256, expected.sha256);
    let mut scratch = svm::Scratch::new();
    let unigram = cat.categorize(&["good", "bad"], &mut scratch).to_vec();
    let bigram = changed.categorize(&["good", "bad"], &mut scratch);
    assert_ne!(unigram, bigram, "the changed bound must affect real producer probabilities");
    assert!(artifact(&cat).bind(&expected).is_err());
    assert!(
        CalibratedCategorizer::new(
            changed,
            artifact(&cat),
            "reviews",
            Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 },
        )
        .is_err()
    );
}

#[test]
fn analyzer_provenance_and_document_graph_keep_the_calibrated_sidecar() {
    use turbo_analysis::export::{export, import};
    use turbo_analysis::{Analyzer, Key, Provenance};
    use turbo_analysis_layers::{Calibrated, Categories, export::codecs};
    let cat = svm();
    let a = artifact(&cat);
    let wrapped =
        CalibratedCategorizer::new(cat, a, "reviews", Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.99 })
            .unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(wrapped).build().unwrap();
    let mut doc = Document::new("good");
    doc.add(&Key::DEFAULT, tokens("good"), Provenance::by("test")).unwrap();
    let mut docs = [doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let side = docs[0].get::<Calibrated<Categories>>().unwrap();
    assert!(side.row(0).abstention.is_some());
    let graph = export(&docs[0], &codecs()).unwrap();
    let back = import(&graph, &codecs()).unwrap();
    assert_eq!(
        back.get::<Calibrated<Categories>>().unwrap().rows().collect::<Vec<_>>(),
        side.rows().collect::<Vec<_>>()
    );
}
