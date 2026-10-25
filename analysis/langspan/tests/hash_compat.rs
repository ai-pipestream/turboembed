//! Fingerprints of actual small detector models and calibration profiles.
mod common;
use turbo_analysis::Annotator;
use turbo_calibration::{Calibration, DataIdentity, Transform, digest};
use turbo_langspan::{CalibratedSpanDetector, Settings, SpanDetector, calibration_identity};

#[test]
fn hash_compat_candidate_tables_and_profiles() {
    let mut hashes = Vec::new();
    for restricted in [false, true] {
        let model = common::small_detector(&["eng", "ell", "zzz"], &[("abc", &[(0, 2.0), (1, -1.0)])]);
        let detector = if restricted {
            SpanDetector::restricted(model, Settings::default(), |code| code != "ell")
        } else {
            SpanDetector::new(model, Settings::default())
        };
        let identity = calibration_identity(&detector, "hash-fixture");
        hashes.extend([detector.config_hash(), identity.inference_hash]);
        let profile = Calibration::new(
            identity,
            Transform::Temperature(1.25),
            DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
            DataIdentity { split: "reserve".into(), sha256: digest(b"reserve") },
            vec![0.1; 20],
        )
        .unwrap();
        hashes.push(CalibratedSpanDetector::new(detector, profile, "hash-fixture").unwrap().config_hash());
    }
    // Fixed values from the previous implementation, before migration.
    assert_eq!(
        hashes,
        [
            9271805774357292529,
            13746497565560355511,
            6098002148004997526,
            9271805774357292529,
            3148810852425046655,
            4123920057459090545
        ]
    );
}
