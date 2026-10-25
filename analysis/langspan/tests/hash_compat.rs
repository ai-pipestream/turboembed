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
    // Main #1103 adds model and retained-language inputs. These values were
    // captured from that merge before replacing the arithmetic with Fnv1a.
    assert_eq!(
        hashes,
        [
            10198147997765475433,
            693950649760826463,
            5437517471414260919,
            16938217562532895248,
            14980512779310312396,
            10487642117907239850
        ]
    );
}
