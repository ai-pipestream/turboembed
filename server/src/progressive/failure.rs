//! The message a failed progressive step reports.

use crate::proto::result::LayerErrorKind;
use crate::results::receipt::error::kind_name;

/// A code name, never the error's text: an annotator's detail can quote
/// the document (issue #1908). An input over a limit says which and by
/// how much, counts only (#2125, #2134).
pub(super) fn step_message(step: &str, err: &turbo_analysis::Error, kind: LayerErrorKind) -> String {
    match err {
        turbo_analysis::Error::OverLimit { limit, max, got, .. }
            if turbo_analysis::error::known_limit(limit).is_some() =>
        {
            format!("step '{step}' failed: over its {limit} {max}, got {got}")
        }
        _ => format!("step '{step}' failed: {}", kind_name(kind)),
    }
}
