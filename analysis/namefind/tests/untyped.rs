use std::sync::Arc;
use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest};
use turbo_maxent::ModelFile;
use turbo_namefind::CrfNameAnnotator;
use turbo_namefind::calibrated::{CalibratedNameAnnotator, identity};
use turbo_sequence::crf::CrfSequenceModel;
use turbo_sequence::namefind::{CODEC_PROPERTY, MODEL_ENTRY_NAME, NameCodec, crf};
use turbo_sequence::{PatternMode, namefind::NameFinder};
use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};

fn input() -> (Document, Sentences, Tokens) {
    let doc = Document::new("Alice");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    let sentences = Sentences { spans };
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    (doc, sentences, Tokens { spans, sentence: None })
}

#[test]
fn crf_annotator_handles_typed_and_untyped_bio_and_bilou() {
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        for prefix in ["", "person-"] {
            let suffix = if codec == NameCodec::Bio { "start" } else { "unit" };
            let labels = vec!["other".to_owned(), format!("{prefix}{suffix}")];
            let mut weights = vec![0.0; 8];
            weights[5] = 5.0; // Start score for the entity outcome.
            let model = CrfSequenceModel::new(labels, vec![], codec, weights).unwrap();
            let mut manifest = Manifest::new("TokenNameFinder", "en", 1700000000000);
            manifest.set("Model-Type", crf::MODEL_TYPE);
            manifest.set(CODEC_PROPERTY, codec.class_name());
            let mut package = ModelPackage::new(manifest);
            package.add_entry(crf::MODEL_ENTRY, model.to_bytes());
            let ann = CrfNameAnnotator::new(ModelFile::open(package.to_bytes()).unwrap(), 2).unwrap();
            let (doc, s, t) = input();
            let mut previous = None;
            for _ in 0..3 {
                let (entities, alternatives) = ann.annotate((doc.get::<View>().unwrap(), &s, &t)).unwrap();
                assert_eq!(entities.len(), 1);
                assert_eq!(
                    entities.kinds.label(entities.kind[0]),
                    if prefix.is_empty() { "default" } else { "person" }
                );
                assert!(entities.score[0] > 0.9);
                alternatives.check().unwrap();
                if let Some(ref kinds) = previous {
                    assert!(Arc::ptr_eq(kinds, &entities.kinds));
                }
                previous = Some(entities.kinds);
            }
        }
    }
}

#[test]
fn calibrated_annotator_handles_untyped_unit_and_clears_document_state() {
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        for prefix in ["", "person-"] {
            let suffix = if codec == NameCodec::Bio { "start" } else { "unit" };
            let outcome = format!("{prefix}{suffix}");
            let previous = format!("pd={outcome}");
            let events = [
                Event::from_strs(&outcome, &["w=alice", "pd=null"]),
                Event::from_strs("other", &["w=alice", &previous]),
                Event::from_strs("other", &["w=walked"]),
            ];
            let mut params = TrainingParameters::new();
            params.put(Key::ITERATIONS, 50);
            params.put(Key::CUTOFF, 0);
            let (model, _) =
                turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
            let mut manifest = Manifest::new("TokenNameFinder", "en", 1700000000000);
            manifest.set(CODEC_PROPERTY, codec.class_name());
            let mut package = ModelPackage::new(manifest);
            package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&model).unwrap());
            let file = Arc::new(ModelFile::open(package.to_bytes()).unwrap());
            let raw_model = file.maxent(MODEL_ENTRY_NAME).unwrap();
            let mut raw = NameFinder::new(&file, &raw_model, PatternMode::Utf16Units).unwrap();
            let fresh = raw.find_scored_flat(&["Alice"]).1;
            let continued = raw.find_scored_flat(&["Alice"]).1;
            assert_ne!(fresh, continued, "the model must respond to adaptive history");
            drop(raw);
            let artifact = Calibration::new(
                identity(&file, "test").unwrap(),
                Transform::Temperature(1.0),
                DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
                DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
                vec![0.1; 20],
            )
            .unwrap();
            let ann = CalibratedNameAnnotator::new(
                file,
                artifact,
                "test",
                Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.0 },
            )
            .unwrap();
            let (doc, s, t) = input();
            let mut score = None;
            for _ in 0..3 {
                let (entities, decisions) = ann.annotate((doc.get::<View>().unwrap(), &s, &t)).unwrap();
                assert_eq!(entities.len(), 1);
                assert_eq!(
                    entities.kinds.label(entities.kind[0]),
                    if prefix.is_empty() { "default" } else { "person" }
                );
                assert_eq!(decisions.len(), 1);
                decisions.check().unwrap();
                if let Some(score) = score {
                    assert_eq!(entities.score[0], score);
                }
                score = Some(entities.score[0]);
            }
        }
    }
}

#[test]
fn hash_compat_crf_model_and_calibration_identity() {
    let mut hashes = Vec::new();
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        let suffix = if codec == NameCodec::Bio { "start" } else { "unit" };
        let model = CrfSequenceModel::new(
            vec!["other".into(), format!("person-{suffix}")],
            vec![],
            codec,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        )
        .unwrap();
        let mut manifest = Manifest::new("TokenNameFinder", "en", 1700000000000);
        manifest.set("Model-Type", crf::MODEL_TYPE);
        manifest.set(CODEC_PROPERTY, codec.class_name());
        let mut package = ModelPackage::new(manifest);
        package.add_entry(crf::MODEL_ENTRY, model.to_bytes());
        let file = ModelFile::open(package.to_bytes()).unwrap();
        for k in [1, 2, 4096] {
            hashes.push(CrfNameAnnotator::new(file.clone(), k).unwrap().config_hash());
            hashes.push(
                turbo_namefind::calibrated::crf_identity_with_candidates(&file, "fixture", k).unwrap().inference_hash,
            );
        }
    }
    // Fixed values from the previous implementation, before migration.
    assert_eq!(
        hashes,
        [
            18391168008607507920,
            13510644368543982392,
            6641370155800724531,
            1760846515737199003,
            8312815378480799137,
            9606883738896318377,
            6099489347948913068,
            16322318960098287261,
            12796435568851681295,
            107890293356325054,
            2195728718301249053,
            7953927516515444428
        ]
    );
}
