use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest, metrics};
use turbo_doccat::{calibrated::*, *};
use turbo_maxent::ModelFile;
use turbo_train::{
    Event, Key, TrainingConfiguration, TrainingParameters,
    calibration::{Method, Samples, fit},
    trainer,
};

fn maxent() -> DocumentCategorizerME {
    let events = vec![
        Event::from_strs("pos", &["bow=good"]),
        Event::from_strs("pos", &["bow=great"]),
        Event::from_strs("neg", &["bow=bad"]),
        Event::from_strs("neg", &["bow=awful"]),
    ];
    let mut params = TrainingParameters::new();
    params.put(Key::ALGORITHM, "MAXENT");
    params.put(Key::ITERATIONS, 50);
    params.put(Key::CUTOFF, 0);
    let (trained, _) = trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut package =
        turbo_train::ModelPackage::new(turbo_train::Manifest::new("DocumentCategorizerME", "en", 1700000000000));
    package.add_entry("doccat.model", trained.write().unwrap());
    DocumentCategorizerME::new(DoccatModel::open(ModelFile::open(package.to_bytes()).unwrap()).unwrap())
}
fn svm() -> SvmCategorizer {
    let model = "svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 2\nrho 0.25\nlabel 0 1\nprobA -2.0\nprobB 0.0\nnr_sv 1 1\nSV\n1.0 1:1.0\n-1.0 2:1.0\n";
    let side = "svm-doccat 1\nlanguage\teng\nweighting\tBINARY\nselection\tNONE\t-1\nscaling\tnone\ngenerator\tbag-of-words\ncategory\t0\tpos\ncategory\t1\tneg\nfeature\t1\t1.0\t0.0\t1.0\tbow=good\nfeature\t2\t1.0\t0.0\t1.0\tbow=bad\n";
    SvmCategorizer::new(SvmDoccatModel::from_text(model, side).unwrap()).unwrap()
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
    assert_eq!(side.rows[0].raw, raw.iter().next().unwrap().scores);
    assert_ne!(side.rows[0].raw, side.rows[0].calibrated);
    assert!(side.rows[0].abstention.is_some());
    assert_eq!(unchanged.len(), 1);
    assert_eq!(wrap.config_hash(), wrap.predictor().config_hash());
}
#[test]
fn raw_maxent_and_svm_outputs_survive_abstention() {
    parity(maxent());
    parity(svm());
}
#[test]
fn wrong_domain_or_producer_cannot_bind() {
    let cat = maxent();
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
        CalibratedCategorizer::new(svm(), a, "reviews", Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 })
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
    let a = fit(
        id.clone(),
        Method::Temperature,
        Samples { split: "fit", probabilities: &f, gold: &fy },
        Samples { split: "conformal", probabilities: &c, gold: &cy },
    )
    .unwrap();
    let p = turbo_calibration::Predictor::new(a, &id, Policy { coverage: 0.99, max_set_size: 2, min_confidence: 0.0 })
        .unwrap();
    let predictions: Vec<_> = t.iter().map(|r| p.predict(r).unwrap()).collect();
    let before = metrics::reliability(&t, &ty, 10).unwrap();
    let after =
        metrics::reliability(&predictions.iter().map(|r| r.calibrated.clone()).collect::<Vec<_>>(), &ty, 10).unwrap();
    let coverage = metrics::coverage(&predictions, &ty).unwrap();
    assert!(after.ece < before.ece, "{before:?} {after:?}");
    assert!((coverage.set_coverage - 0.99).abs() <= 0.03);
    println!(
        "{}: ECE {:.6} -> {:.6}; Brier {:.6} -> {:.6}; coverage {:.6}",
        id.model, before.ece, after.ece, before.brier, after.brier, coverage.set_coverage
    );
}
#[test]
fn heldout_receipts_for_each_categorizer() {
    heldout(maxent());
    heldout(svm());
}

#[test]
fn producer_feature_configuration_is_bound_and_custom_features_are_rejected() {
    let cat = maxent();
    let original = identity(&cat, "reviews").unwrap();
    let model = cat.model().clone();
    for generator in [
        BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::only_all_letter_tokens()),
        BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::new(1, 2).unwrap()),
    ] {
        let changed = DocumentCategorizerME::new(
            model.clone().with_context_generator(DoccatContextGenerator::from_builtins(vec![generator])),
        );
        let expected = identity(&changed, "reviews").unwrap();
        assert_ne!(original.inference_hash, expected.inference_hash);
        assert!(artifact(&cat).bind(&expected).is_err());
    }
    let custom = DocumentCategorizerME::new(model.with_context_generator(DoccatContextGenerator::from_custom(vec![
        std::sync::Arc::new(FeatureGenerator::BagOfWords),
    ])));
    assert!(identity(&custom, "reviews").unwrap_err().contains("custom"));
}

#[test]
fn analyzer_provenance_and_document_graph_keep_the_calibrated_sidecar() {
    use turbo_analysis::export::{export, import};
    use turbo_analysis::{Analyzer, Key, Provenance};
    use turbo_analysis_layers::{Calibrated, Categories, export::codecs};
    let cat = maxent();
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
    assert!(side.rows[0].abstention.is_some());
    let graph = export(&docs[0], &codecs()).unwrap();
    let back = import(&graph, &codecs()).unwrap();
    assert_eq!(back.get::<Calibrated<Categories>>().unwrap().rows, side.rows);
}
