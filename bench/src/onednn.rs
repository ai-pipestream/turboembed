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
    #[cfg(test)]
    if TEST_SINK.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(sink) = guard.as_mut() else { return false };
        sink.calls.push((backend.to_owned(), choices.to_owned(), warnings.to_vec()));
        match &sink.override_inputs {
            Some((b, c, w)) => write_fallback(&mut sink.stderr, b, c, w),
            None => write_fallback(&mut sink.stderr, backend, choices, warnings),
        }
        true
    }) {
        return;
    }
    write_fallback(&mut io::stderr().lock(), backend, choices, warnings);
}

#[cfg(test)]
struct TestSink {
    override_inputs: Option<(String, String, Vec<String>)>,
    calls: Vec<(String, String, Vec<String>)>,
    stderr: Vec<u8>,
}

#[cfg(test)]
std::thread_local! {
    static TEST_SINK: std::cell::RefCell<Option<TestSink>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct CapturedFallback {
    pub calls: Vec<(String, String, Vec<String>)>,
    pub stderr: String,
}

#[cfg(test)]
pub(crate) fn capture_fallback<T>(
    override_inputs: Option<(&str, &str, Vec<String>)>,
    f: impl FnOnce() -> T,
) -> (T, CapturedFallback) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_SINK.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }
    }
    TEST_SINK.with(|cell| {
        *cell.borrow_mut() = Some(TestSink {
            override_inputs: override_inputs.map(|(b, c, w)| (b.to_owned(), c.to_owned(), w)),
            calls: Vec::new(),
            stderr: Vec::new(),
        });
    });
    let _reset = Reset;
    let out = f();
    let captured = TEST_SINK.with(|cell| {
        let sink = cell.borrow_mut().take().expect("test sink present");
        CapturedFallback { calls: sink.calls, stderr: String::from_utf8(sink.stderr).unwrap() }
    });
    (out, captured)
}

#[cfg(test)]
#[path = "onednn_tests.rs"]
mod tests;
