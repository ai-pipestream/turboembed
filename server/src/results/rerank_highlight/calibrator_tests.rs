//! Unit tests for bundle-cached rerank calibration loading and overrides.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo::{TURBO_TASK_EMBED, turbo_model_info};
use turbo_analysis::ModelRef;
use turbo_analysis_embed::{RERANK_CALIBRATION_LABELS, RerankCalibrator};
use turbo_calibration::{Calibration as ProbCalibration, DataIdentity, Identity, Task, Transform};
use turbo_scratch::Scratch;

use super::{BundleCalibration, Calibrator, resolve_with_bundle};
use crate::api::TURBO_E_INVALID_ARGUMENT;
use crate::proto::result::RerankCalibrationSpecProto;
use crate::proto::result::rerank_calibration_spec_proto::Kind as CalKind;

static DIR_SERIAL: AtomicUsize = AtomicUsize::new(0);

fn scratch_dir(tag: &str) -> Scratch {
    let id =
        format!("turbo-rerank-cal-unit-{tag}-{}-{}", std::process::id(), DIR_SERIAL.fetch_add(1, Ordering::Relaxed));
    let dir = Scratch::at(std::env::temp_dir().join(id));
    std::fs::create_dir(dir.path()).unwrap();
    dir
}

fn sample_mref() -> ModelRef {
    ModelRef { id: "cross-encoder/ms-marco-MiniLM-L6-v2".into(), revision: "main".into(), sha256: "a".repeat(64) }
}

fn sample_tcal(mref: &ModelRef, domain: &str, hash: u64, temp: f64) -> Vec<u8> {
    ProbCalibration::new(
        Identity {
            model: mref.id.clone(),
            revision: mref.revision.clone(),
            sha256: mref.sha256.clone(),
            task: Task::Categorization,
            domain: domain.into(),
            labels: RERANK_CALIBRATION_LABELS.iter().map(|&s| s.to_owned()).collect(),
            inference_hash: hash,
        },
        Transform::Temperature(temp),
        DataIdentity { split: "fit".into(), sha256: "1".repeat(64) },
        DataIdentity { split: "conf".into(), sha256: "2".repeat(64) },
        vec![0.10, 0.25, 0.50, 0.75],
    )
    .unwrap()
    .to_bytes()
}

#[test]
fn non_rerank_task_skips_sidecar_and_absent_uses_uncalibrated() {
    let dir = scratch_dir("non-rerank");
    std::fs::write(dir.join("calibration.tcal"), b"garbage").unwrap();
    let info: turbo_model_info = unsafe { std::mem::zeroed() };
    let bc = BundleCalibration::load_for_task(TURBO_TASK_EMBED, dir.to_str().unwrap(), &info);
    assert!(matches!(bc, BundleCalibration::Absent));

    let mref = sample_mref();
    let cal = resolve_with_bundle(&bc, &mref, None).unwrap();
    let expected = RerankCalibrator::uncalibrated().score(1.5).unwrap();
    assert_eq!(cal.score(1.5).unwrap(), expected);
}

#[test]
fn present_default_sidecar_is_shared_across_warm_calls_after_file_removal() {
    let dir = scratch_dir("present-default");
    let mref = sample_mref();
    let bytes = sample_tcal(&mref, "search", 0, 2.5);
    let sidecar = dir.join("calibration.tcal");
    std::fs::write(&sidecar, &bytes).unwrap();

    let bc = BundleCalibration::load_from_dir(dir.path(), &mref);
    // Remove sidecar from disk so warm resolution cannot perform filesystem reads.
    std::fs::remove_file(&sidecar).unwrap();

    let first = resolve_with_bundle(&bc, &mref, None).unwrap();
    let second = resolve_with_bundle(
        &bc,
        &mref,
        Some(&RerankCalibrationSpecProto { kind: None, domain: "  ".into(), inference_hash: 0 }),
    )
    .unwrap();

    let (Calibrator::Shared(a), Calibrator::Shared(b)) = (&first, &second) else {
        panic!("expected shared bundle calibrators");
    };
    assert!(Arc::ptr_eq(a, b), "warm default calls must share the pre-parsed Arc");

    let direct = RerankCalibrator::from_tcal_bytes(&bytes, &mref, "search", 0).unwrap().score(1.25).unwrap();
    assert_eq!(first.score(1.25).unwrap(), direct);
    assert!(first.score(1.25).unwrap().conformal_p_value.is_some());
}

#[test]
fn non_default_domain_and_hash_bind_against_cached_bytes_without_disk() {
    let dir = scratch_dir("custom-domain");
    let mref = sample_mref();
    let bytes = sample_tcal(&mref, "legal", 42, 1.8);
    let sidecar = dir.join("calibration.tcal");
    std::fs::write(&sidecar, &bytes).unwrap();

    let bc = BundleCalibration::load_from_dir(dir.path(), &mref);
    std::fs::remove_file(&sidecar).unwrap();

    // Default ("search", 0) fails identity check cleanly.
    let err = resolve_with_bundle(&bc, &mref, None).err().expect("default domain mismatch");
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);
    assert!(err.message.contains("calibration identity mismatch"), "{}", err.message);

    // Matching ("legal", 42) resolves from cached bytes in memory.
    let custom = resolve_with_bundle(
        &bc,
        &mref,
        Some(&RerankCalibrationSpecProto { kind: None, domain: "legal".into(), inference_hash: 42 }),
    )
    .unwrap();
    let expected = RerankCalibrator::from_tcal_bytes(&bytes, &mref, "legal", 42).unwrap().score(0.9).unwrap();
    assert_eq!(custom.score(0.9).unwrap(), expected);
}

#[test]
fn malformed_sidecar_reports_invalid_argument_unless_request_overrides() {
    let dir = scratch_dir("malformed");
    let mref = sample_mref();
    std::fs::write(dir.join("calibration.tcal"), b"not-tcal").unwrap();

    let bc = BundleCalibration::load_from_dir(dir.path(), &mref);
    let err = resolve_with_bundle(&bc, &mref, None).err().expect("malformed sidecar error");
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);
    assert!(err.message.contains("unknown calibration format"), "{}", err.message);

    let override_cal = resolve_with_bundle(
        &bc,
        &mref,
        Some(&RerankCalibrationSpecProto {
            kind: Some(CalKind::Temperature(0.75)),
            domain: String::new(),
            inference_hash: 0,
        }),
    )
    .unwrap();
    let expected = RerankCalibrator::from_transform(Transform::Temperature(0.75)).unwrap().score(0.8).unwrap();
    assert_eq!(override_cal.score(0.8).unwrap(), expected);
}
