use super::*;
/// Finite and above zero.
pub(super) fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

/// The fastest measured reference: its name and p50.
pub fn fastest(references: &[ReferenceRun]) -> Option<(&str, f64)> {
    references
        .iter()
        .filter_map(|r| r.measured.as_ref().map(|m| (r.name.as_str(), m.p50_ms)))
        .filter(|(_, p50)| positive(*p50))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// Rows in one measured pass. A single batch has one case per row.
/// A mixed pass whose frame batch is 1 has one case per frame of the
/// cycle, each frame `[1, seq]`.
pub(super) fn pass_rows(rows: &Rows) -> Result<u64, String> {
    let n = rows.cases.len();
    if rows.batch == 0 || rows.seq == 0 || n == 0 || !is_hex(&rows.sha256, 64) {
        return Err("rows: batch and seq must be above 0, cases one per row, sha256 64 lowercase hex".into());
    }
    if n != rows.batch as usize && rows.batch != 1 {
        return Err("rows: batch and seq must be above 0, cases one per row, sha256 64 lowercase hex".into());
    }
    Ok(n as u64)
}

/// speed_ratio and speed_reference as the references give them.
/// Invalid or unrepresentable ratios return no measurement.
pub fn speed(p50_ms: f64, references: &[ReferenceRun]) -> (Option<f64>, Option<String>) {
    if positive(p50_ms)
        && let Some((name, p50)) = fastest(references)
        && positive(p50_ms / p50)
    {
        (Some(p50_ms / p50), Some(name.to_owned()))
    } else {
        (None, None)
    }
}
