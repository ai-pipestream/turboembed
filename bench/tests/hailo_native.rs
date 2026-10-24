//! Opt-in tests of the real library and a receipt from the native HailoRT CLI.
//! Run on a board after `turbo-bench record`, with that bundle and receipt.
use std::path::PathBuf;

use serde_json::{Value, json};
use turbo_bench::hailo::from_evidence;
use turbo_bench::measure::{self, Plan, RowKind};

#[test]
#[ignore = "requires a physical Hailo board, TURBO_HAILO_BUNDLE and TURBO_HAILO_RECEIPT"]
fn native_receipt_matches_the_real_library_and_rejects_changed_inputs() {
    let bundle = PathBuf::from(std::env::var("TURBO_HAILO_BUNDLE").expect("set the deployment bundle"));
    let receipt = std::fs::read(std::env::var("TURBO_HAILO_RECEIPT").expect("set the native CLI receipt")).unwrap();
    let m = measure::measure(&Plan {
        bundle,
        device: "hailo".into(),
        precision: turbo::TURBO_PRECISION_MODEL,
        batch: Some(32),
        seq: Some(128),
        rows: RowKind::Mixed,
        warmup: 1,
        iterations: 3,
    })
    .unwrap();
    assert!(m.conformance.min_cosine >= 0.93);
    assert_eq!(m.timing.computed_tokens, Some(4096));
    let r = from_evidence(&receipt, &m).unwrap();
    assert_eq!(r.name, "hailort");
    assert_eq!(r.measured.as_ref().unwrap().iterations, 5);
    let source: Value = serde_json::from_slice(&receipt).unwrap();
    for (key, value) in [
        ("target", json!("hailo8l")),
        ("manifest_sha256", json!("0".repeat(64))),
        ("hef_sha256", json!("0".repeat(64))),
        ("rows_sha256", json!("0".repeat(64))),
        ("batch", json!(8)),
        ("seq", json!(64)),
        ("cases", json!([])),
        ("live_tokens", json!(1)),
        ("computed_tokens_per_pass", json!(724)),
        ("cli_sha256", json!("not-a-hash")),
        ("commands", json!([])),
        ("fps_samples", json!([0.0, 0.0, 0.0, 0.0, 0.0])),
        ("seconds_per_process", json!(0)),
    ] {
        let mut changed = source.clone();
        changed[key] = value;
        assert!(from_evidence(&serde_json::to_vec(&changed).unwrap(), &m).is_err(), "accepted changed {key}");
    }
}
