//! The model-field validation every constructor and `open` share:
//! label names like the entry's reader enforces them, and the per-label
//! threshold vector.

use super::bytes::MAX_UTF_BYTES;
use super::error::MultiLabelError;

/// The label names every constructor validates: like the entry's reader,
/// none empty, over `writeUTF`'s 65535 bytes, or duplicated (`open`
/// refuses what `new` would otherwise smuggle in).
pub(super) fn check_label_names(labels: &[&str]) -> Result<(), MultiLabelError> {
    let mut seen: Vec<&str> = Vec::with_capacity(labels.len());
    for &name in labels {
        if name.is_empty() {
            return Err(MultiLabelError::Invalid("an empty label name".into()));
        }
        if name.len() > MAX_UTF_BYTES {
            return Err(MultiLabelError::Invalid(format!(
                "label {name:?} is {0} bytes, over writeUTF's {MAX_UTF_BYTES}",
                name.len()
            )));
        }
        if seen.contains(&name) {
            return Err(MultiLabelError::Invalid(format!("duplicate label {name:?}")));
        }
        seen.push(name);
    }
    Ok(())
}

/// The thresholds every constructor validates: `k` of them, each finite
/// and in the open interval `(0.0, 1.0)`.
pub(super) fn check_thresholds(thresholds: &[f64], k: usize) -> Result<(), MultiLabelError> {
    if thresholds.len() != k {
        return Err(MultiLabelError::Invalid(format!("{} thresholds for {k} labels", thresholds.len())));
    }
    for (i, &t) in thresholds.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 || t >= 1.0 {
            return Err(MultiLabelError::Invalid(format!("threshold {t} of label {i}, outside the open 0.0..1.0")));
        }
    }
    Ok(())
}
