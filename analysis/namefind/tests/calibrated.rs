//! Numerical entity decoding and calibration remain usable without archives.
//! Native document annotation is a separate integration tracked in #1540.
use turbo_analysis::{IxRange, Tag, TagSet};
use turbo_calibration::{Calibration, DataIdentity, Identity, Policy, Predictor, Task, Transform, digest};
use turbo_sequence::crf::{CrfScores, EntityDecoder, EntityEncoding, EntityLabel};

fn decoder(encoding: EntityEncoding) -> EntityDecoder {
    let mut labels = vec![EntityLabel::Outside];
    for kind in [Tag(0), Tag(1)] {
        labels.extend([EntityLabel::Begin(kind), EntityLabel::Inside(kind)]);
        if encoding == EntityEncoding::Bilou {
            labels.extend([EntityLabel::Last(kind), EntityLabel::Unit(kind)]);
        }
    }
    EntityDecoder::new(encoding, TagSet::new("entity-test", &["person", "place"]), labels).unwrap()
}

fn identity(decoder: &EntityDecoder) -> Identity {
    Identity {
        model: "numeric-entity-test".into(),
        revision: "1".into(),
        sha256: digest(b"two-token zero-potential arithmetic test; not a trained model"),
        task: Task::SpanTagging,
        domain: "numerical-test".into(),
        labels: vec!["other".into(), "person".into(), "place".into()],
        inference_hash: decoder.schema_hash(),
    }
}

fn calibration(id: Identity) -> Calibration {
    Calibration::new(
        id,
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit arithmetic") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conformal arithmetic") },
        vec![0.1; 20],
    )
    .unwrap()
}
fn policy() -> Policy {
    Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn native_bio_and_bilou_marginals_match_independent_path_counts() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let decoder = decoder(encoding);
        let n = decoder.labels().len();
        let emissions = vec![vec![0.0; n]; 2];
        let transitions = vec![0.0; n * n];
        let ends = vec![0.0; n];
        let scores = CrfScores { emissions: &emissions, transitions: &transitions, start: &ends, end: &ends };
        let lattice = decoder.lattice(scores).unwrap();
        // Each token may be outside or one of two unit entities: 3*3 paths.
        // Two additional paths contain one two-token entity, one per kind.
        let paths = lattice.best_paths(32).unwrap();
        assert_eq!(paths.len(), 11);
        close(lattice.numeric().log_partition().unwrap(), 11f64.ln());
        for path in &paths {
            close(path.path.log_probability.exp(), 1.0 / 11.0);
        }
        for (range, expected) in [
            (IxRange::new(0, 1), [5.0 / 11.0, 3.0 / 11.0, 3.0 / 11.0]),
            (IxRange::new(1, 2), [5.0 / 11.0, 3.0 / 11.0, 3.0 / 11.0]),
            (IxRange::new(0, 2), [9.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0]),
        ] {
            let actual = lattice.span_distribution(range).unwrap();
            turbo_calibration::check_distribution(&actual).unwrap();
            for (&got, want) in actual.iter().zip(expected) {
                close(got, want);
            }
        }
        assert!(lattice.span_distribution(IxRange::new(0, 3)).is_err());
    }
}

#[test]
fn native_span_probability_calibration_matches_temperature_equation() {
    let decoder = decoder(EntityEncoding::Bio);
    let n = decoder.labels().len();
    let emissions = vec![vec![0.0; n]; 2];
    let transitions = vec![0.0; n * n];
    let ends = vec![0.0; n];
    let lattice = decoder
        .lattice(CrfScores { emissions: &emissions, transitions: &transitions, start: &ends, end: &ends })
        .unwrap();
    let raw = lattice.span_distribution(IxRange::new(0, 1)).unwrap();
    let id = identity(&decoder);
    let predictor = Predictor::new(calibration(id.clone()), &id, policy()).unwrap();
    let prediction = predictor.predict(&raw).unwrap();
    let denominator: f64 = raw.iter().map(|p| p.sqrt()).sum();
    for (&got, want) in prediction.calibrated.iter().zip(raw.iter().map(|p| p.sqrt() / denominator)) {
        close(got, want);
    }
    assert_eq!(raw, lattice.span_distribution(IxRange::new(0, 1)).unwrap());
    assert!(predictor.predict(&[f64::NAN, 0.0, 1.0]).is_err());
    assert!(predictor.predict(&[0.5, 0.5]).is_err());
}

#[test]
fn calibration_refuses_schema_domain_and_parameter_identity_mismatches() {
    let bio = decoder(EntityEncoding::Bio);
    let id = identity(&bio);
    let artifact = calibration(id.clone());
    let mut wrong = id.clone();
    wrong.domain = "another-domain".into();
    assert!(Predictor::new(artifact.clone(), &wrong, policy()).is_err());
    wrong = id.clone();
    wrong.sha256 = digest(b"different numerical parameters");
    assert!(Predictor::new(artifact.clone(), &wrong, policy()).is_err());
    wrong = id.clone();
    wrong.labels.swap(1, 2);
    assert!(Predictor::new(artifact.clone(), &wrong, policy()).is_err());
    wrong = id;
    wrong.inference_hash = decoder(EntityEncoding::Bilou).schema_hash();
    assert!(Predictor::new(artifact, &wrong, policy()).is_err());
}

#[test]
fn typed_schema_requires_complete_states_and_valid_entity_tags() {
    let kinds = TagSet::new("test", &["person"]);
    assert!(EntityDecoder::new(EntityEncoding::Bio, kinds.clone(), vec![EntityLabel::Outside]).is_err());
    assert!(
        EntityDecoder::new(
            EntityEncoding::Bio,
            kinds,
            vec![EntityLabel::Outside, EntityLabel::Begin(Tag(1)), EntityLabel::Inside(Tag(1))],
        )
        .is_err()
    );
}
