//! The runtime's log, kept (issue #2512). `turbo_runtime_create` takes
//! a log callback; with none, everything the library says is discarded:
//! a backend whose probe failed, a device it could not describe, a
//! kernel path or a tuning cache it gave up on. The server's runtime
//! logs here instead. Every message of info level and above is a line
//! on stderr as it is said, and the warnings and errors are kept, up to
//! [`MAX_RUNTIME_WARNINGS`] with the rest counted, for the capability
//! listing's typed `runtime_warnings`.

use std::ffi::c_void;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use turbo::{turbo_runtime_desc, turbo_text};

use crate::proto::result::{RuntimeLogEntry, RuntimeLogLevel};

/// The warnings and errors one runtime's log keeps; later ones are
/// printed and counted, not kept.
pub const MAX_RUNTIME_WARNINGS: usize = 64;

// turbo.h, turbo_log_fn: 0 error, 1 warning, 2 info, 3 debug.
const LOG_ERROR: u32 = 0;
const LOG_WARNING: u32 = 1;
const LOG_INFO: u32 = 2;

/// What one runtime logged.
#[derive(Default)]
pub struct RuntimeLog {
    kept: Mutex<Vec<RuntimeLogEntry>>,
    dropped: AtomicU64,
}

impl RuntimeLog {
    pub fn new() -> Arc<RuntimeLog> {
        Arc::new(RuntimeLog::default())
    }

    /// The descriptor that makes a runtime log here. The callback's user
    /// data is this allocation, so it must outlive the runtime and every
    /// handle made from it, which `api` keeps by holding an `Arc` of it
    /// in each of them. Crate-only, so no caller gets the raw pointer
    /// without that.
    pub(crate) fn desc(self: &Arc<Self>) -> turbo_runtime_desc {
        turbo_runtime_desc {
            struct_size: size_of::<turbo_runtime_desc>() as u32,
            reserved: 0,
            log: Some(said),
            log_user_data: Arc::as_ptr(self).cast_mut().cast(),
        }
    }

    /// The warnings and errors kept, oldest first, and how many more
    /// came after the cap.
    pub fn warnings(&self) -> (Vec<RuntimeLogEntry>, u64) {
        let kept = self.kept.lock().unwrap_or_else(|p| p.into_inner()).clone();
        (kept, self.dropped.load(Ordering::Relaxed))
    }

    fn take(&self, level: u32, message: &str) {
        let (name, kept) = match level {
            LOG_ERROR => ("error", Some(RuntimeLogLevel::Error)),
            LOG_WARNING => ("warning", Some(RuntimeLogLevel::Warning)),
            LOG_INFO => ("info", None),
            // Debug and anything newer: said for a developer's own sink.
            _ => return,
        };
        // Never eprintln!, which panics when stderr is closed: a panic
        // cannot unwind out of the extern "C" callback and would abort
        // the process. A line that cannot be written is still kept.
        let _ = writeln!(std::io::stderr(), "turbo-kserve: runtime: {name}: {message}");
        let Some(level) = kept else { return };
        let mut entries = self.kept.lock().unwrap_or_else(|p| p.into_inner());
        if entries.len() < MAX_RUNTIME_WARNINGS {
            entries.push(RuntimeLogEntry { level: level as i32, message: message.to_owned() });
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The `turbo_log_fn` of [`RuntimeLog::desc`].
unsafe extern "C" fn said(user_data: *mut c_void, level: u32, message: turbo_text) {
    if user_data.is_null() {
        return;
    }
    let bytes = if message.len == 0 || message.ptr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(message.ptr.cast::<u8>(), message.len as usize) }
    };
    let log = unsafe { &*(user_data as *const RuntimeLog) };
    log.take(level, &String::from_utf8_lossy(bytes));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn say(log: &Arc<RuntimeLog>, level: u32, s: &str) {
        let desc = log.desc();
        let t = turbo_text { ptr: s.as_ptr().cast(), len: s.len() as u64 };
        unsafe { (desc.log.expect("a log callback"))(desc.log_user_data, level, t) };
    }

    #[test]
    fn warnings_and_errors_are_kept_with_their_level_and_the_rest_is_not() {
        let log = RuntimeLog::new();
        say(&log, LOG_WARNING, "cuda backend: device_count failed: the driver is older than the runtime");
        say(&log, LOG_INFO, "the cpu backend lists 1 device");
        say(&log, 3, "a debug line");
        say(&log, LOG_ERROR, "levelzero: zeInitDrivers returned ZE_RESULT_ERROR_UNINITIALIZED");
        let (kept, dropped) = log.warnings();
        let got: Vec<(i32, &str)> = kept.iter().map(|e| (e.level, e.message.as_str())).collect();
        assert_eq!(
            got,
            [
                (
                    RuntimeLogLevel::Warning as i32,
                    "cuda backend: device_count failed: the driver is older than the runtime"
                ),
                (RuntimeLogLevel::Error as i32, "levelzero: zeInitDrivers returned ZE_RESULT_ERROR_UNINITIALIZED"),
            ]
        );
        assert_eq!(dropped, 0);
    }

    #[test]
    fn what_comes_after_the_cap_is_counted_not_lost_without_a_trace() {
        let log = RuntimeLog::new();
        for i in 0..MAX_RUNTIME_WARNINGS + 7 {
            say(&log, LOG_WARNING, &format!("warning {i}"));
        }
        let (kept, dropped) = log.warnings();
        assert_eq!(kept.len(), MAX_RUNTIME_WARNINGS);
        assert_eq!(kept[0].message, "warning 0");
        assert_eq!(dropped, 7);
    }
}
