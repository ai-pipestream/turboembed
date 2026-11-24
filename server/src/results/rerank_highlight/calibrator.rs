//! Bundle-level and request-override score calibration for `RerankAndHighlight`.

use std::path::Path;
use std::sync::Arc;

use turbo::{TURBO_TASK_RERANK, turbo_model_info};
use turbo_analysis::ModelRef;
use turbo_analysis_embed::{CalibratedRerankScore, RerankCalibrator};
use turbo_calibration::Transform;

use super::bad;
use crate::api::{Failure, Result, TURBO_E_INTERNAL, field};
use crate::proto::result::RerankCalibrationSpecProto;
use crate::proto::result::rerank_calibration_spec_proto::Kind as CalKind;
use crate::state::Loaded;

const CALIBRATION_SIDECAR_FILE: &str = "calibration.tcal";
const DEFAULT_CALIBRATION_DOMAIN: &str = "search";
const DEFAULT_INFERENCE_HASH: u64 = 0;

/// Bundle-level cross-encoder calibration sidecar (`calibration.tcal`) read
/// once when a `TASK_RERANK` model loads on the blocking pool so warm
/// `RerankAndHighlight` calls perform no filesystem I/O.
pub(crate) enum BundleCalibration {
    /// Non-rerank bundle or rerank bundle with no `calibration.tcal` file.
    Absent,
    /// Reading `calibration.tcal` at bundle load time failed with an I/O error.
    ReadFailed(Failure),
    /// Raw sidecar bytes plus the pre-parsed calibrator for default domain and hash.
    Present { bytes: Box<[u8]>, default: Result<Arc<RerankCalibrator>> },
}

impl BundleCalibration {
    /// Reads and pre-parses `<bundle_path>/calibration.tcal` when `task` is
    /// `TURBO_TASK_RERANK`; returns [`Self::Absent`] for all other tasks or
    /// when no sidecar file is present.
    pub(crate) fn load_for_task(task: u32, bundle_path: &str, info: &turbo_model_info) -> Self {
        if task != TURBO_TASK_RERANK {
            return Self::Absent;
        }
        let mref = ModelRef {
            id: field(&info.model_id),
            revision: field(&info.revision),
            sha256: field(&info.manifest_sha256),
        };
        Self::load_from_dir(Path::new(bundle_path), &mref)
    }

    /// Reads and pre-parses `<bundle_dir>/calibration.tcal` bound to `mref`.
    pub(super) fn load_from_dir(bundle_dir: &Path, mref: &ModelRef) -> Self {
        let sidecar = bundle_dir.join(CALIBRATION_SIDECAR_FILE);
        if !sidecar.is_file() {
            return Self::Absent;
        }
        let bytes = match std::fs::read(&sidecar) {
            Ok(b) => b,
            Err(e) => {
                return Self::ReadFailed(Failure::new(TURBO_E_INTERNAL, 0, format!("read {}: {e}", sidecar.display())));
            }
        };
        let default =
            RerankCalibrator::from_tcal_bytes(&bytes, mref, DEFAULT_CALIBRATION_DOMAIN, DEFAULT_INFERENCE_HASH)
                .map(Arc::new)
                .map_err(bad);
        Self::Present { bytes: bytes.into_boxed_slice(), default }
    }
}

/// Request-resolved cross-encoder score calibrator, either shared from the
/// loaded bundle via [`Arc`] or constructed from request-supplied parameters.
pub(super) enum Calibrator {
    Shared(Arc<RerankCalibrator>),
    Owned(RerankCalibrator),
}

impl Calibrator {
    /// Maps finite cross-encoder `raw_logit` into `[0.0, 1.0]` relevance
    /// probability and optional conformal p-value.
    pub(super) fn score(&self, raw_logit: f32) -> Result<CalibratedRerankScore> {
        let res = match self {
            Self::Shared(cal) => cal.score(raw_logit),
            Self::Owned(cal) => cal.score(raw_logit),
        };
        res.map_err(bad)
    }
}

/// Resolves the calibrator for one `RerankAndHighlight` call from inline
/// `spec` overrides or `loaded.rerank_calibration` without filesystem I/O.
pub(super) fn resolve_calibrator(
    loaded: &Loaded,
    mref: &ModelRef,
    spec: Option<&RerankCalibrationSpecProto>,
) -> Result<Calibrator> {
    resolve_with_bundle(&loaded.rerank_calibration, mref, spec)
}

pub(super) fn resolve_with_bundle(
    bundle: &BundleCalibration,
    mref: &ModelRef,
    spec: Option<&RerankCalibrationSpecProto>,
) -> Result<Calibrator> {
    let domain = spec.map(|s| s.domain.trim()).filter(|d| !d.is_empty()).unwrap_or(DEFAULT_CALIBRATION_DOMAIN);
    let hash = spec.map_or(DEFAULT_INFERENCE_HASH, |s| s.inference_hash);

    if let Some(kind) = spec.and_then(|s| s.kind.as_ref()) {
        let cal = match kind {
            CalKind::Temperature(t) => RerankCalibrator::from_transform(Transform::Temperature(*t)),
            CalKind::Platt(p) => {
                RerankCalibrator::from_transform(Transform::Platt { slope: p.slope, intercept: p.intercept })
            }
            CalKind::Isotonic(iso) => RerankCalibrator::from_transform(Transform::Isotonic {
                upper: iso.upper.clone(),
                value: iso.value.clone(),
            }),
            CalKind::TcalArtifact(bytes) => RerankCalibrator::from_tcal_bytes(bytes, mref, domain, hash),
        }
        .map_err(bad)?;
        return Ok(Calibrator::Owned(cal));
    }

    match bundle {
        BundleCalibration::Absent => Ok(Calibrator::Owned(RerankCalibrator::uncalibrated())),
        BundleCalibration::ReadFailed(err) => Err(err.clone()),
        BundleCalibration::Present { bytes, default } => {
            if domain == DEFAULT_CALIBRATION_DOMAIN && hash == DEFAULT_INFERENCE_HASH {
                default.as_ref().map(|cal| Calibrator::Shared(Arc::clone(cal))).map_err(Clone::clone)
            } else {
                RerankCalibrator::from_tcal_bytes(bytes, mref, domain, hash).map(Calibrator::Owned).map_err(bad)
            }
        }
    }
}

#[cfg(test)]
#[path = "calibrator_tests.rs"]
mod tests;
