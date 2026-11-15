//! Notice on `stderr` when a measured `levelzero` session falls back to
//! the backend's own linear kernels because `libturbo_onednn.so` or oneDNN
//! could not open on the device (`docs/levelzero.md`).

use std::io::{self, Write};

const LEVELZERO_BACKEND: &str = "levelzero";
const CHOICES_ENV: &str = "TURBO_LEVELZERO_CHOICES";
const DEVICE_PREFIX: &str = "levelzero device ";
const FALLBACK_MARKER: &str = ": oneDNN not used, the backend's own kernels run the linear layers: ";
const STDERR_PREFIX: &str = "turbo-bench: ";
const LINEAR_ONEDNN: &str = "linear=onednn";
const LINEAR_OWN: &str = "linear=own";
const FORCED_PREFIX: &str = "forced=";
const LINEAR_KEY: &str = "linear";

/// Value of `TURBO_LEVELZERO_CHOICES` in the process environment, or empty
/// when unset.
pub(crate) fn env_choices() -> String {
    std::env::var(CHOICES_ENV).unwrap_or_default()
}

/// Returns true when `choices` explicitly forced `linear=own`, either as
/// reported by `turbo_session_info.choices` (`linear=own;forced=linear`) or
/// as given directly in `TURBO_LEVELZERO_CHOICES` (`linear=own`).
fn choices_forces_own(choices: &str) -> bool {
    let mut own = false;
    let mut forced = None;
    for item in choices.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        if item == LINEAR_OWN {
            own = true;
        } else if let Some(rest) = item.strip_prefix(FORCED_PREFIX) {
            forced = Some(rest);
        }
    }
    own && forced.is_none_or(|list| list.split(',').any(|k| k.trim() == LINEAR_KEY))
}

/// Formats the single `stderr` line for a measured `levelzero` session that
/// ran without oneDNN because opening `libturbo_onednn.so` or oneDNN failed,
/// or `None` when oneDNN ran, `linear=own` was forced, or the backend is not
/// `levelzero`.
pub(crate) fn fallback_line(backend: &str, choices: &str, warnings: &[String]) -> Option<String> {
    if backend != LEVELZERO_BACKEND {
        return None;
    }
    if choices.split(';').any(|item| item.trim() == LINEAR_ONEDNN) || choices_forces_own(choices) {
        return None;
    }
    let warning = warnings.iter().find(|w| w.starts_with(DEVICE_PREFIX) && w.contains(FALLBACK_MARKER))?;
    let single_line = warning.lines().map(str::trim).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("; ");
    Some(format!("{STDERR_PREFIX}{single_line}"))
}

/// Writes the single-line fallback notice to `out`, if applicable.
pub(crate) fn write_fallback(out: &mut impl Write, backend: &str, choices: &str, warnings: &[String]) {
    if let Some(line) = fallback_line(backend, choices, warnings) {
        let _ = writeln!(out, "{line}");
    }
}

/// Prints one line on `stderr` when a measured `levelzero` session ran
/// without oneDNN because opening `libturbo_onednn.so` or oneDNN failed.
pub(crate) fn note_fallback(backend: &str, choices: &str, warnings: &[String]) {
    write_fallback(&mut io::stderr().lock(), backend, choices, warnings);
}

#[cfg(test)]
#[path = "onednn_tests.rs"]
mod tests;
