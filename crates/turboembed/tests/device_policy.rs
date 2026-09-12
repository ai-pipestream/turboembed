//! Device-selection policy for TurboEmbed OpenVINO GenAI.
//!
//! Confirmed:
//! - explicit `device=CPU` compiles `"CPU"` (real path, not a fallback)
//! - `device=GPU` with no GPU plugin is a **loud fail** — never `"CPU"`
//!
//! These cases use `require_ov_device` with a synthetic plugin list so the
//! GPU-missing branch is asserted on every host. No mock embed. No Python.

#![cfg(feature = "genai")]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use turboembed::ffi::{turboembed_last_error, turboembed_status};

unsafe extern "C" {
    fn turboembed_test_require_ov_device(
        requested: *const c_char,
        available_csv: *const c_char,
        out: *mut c_char,
        out_len: usize,
    ) -> turboembed_status;
}

fn require(requested: &str, available_csv: &str) -> Result<String, String> {
    let req = CString::new(requested).expect("requested");
    let listed = CString::new(available_csv).expect("available_csv");
    let mut buf = vec![0u8; 32];
    let st = unsafe {
        turboembed_test_require_ov_device(
            req.as_ptr(),
            listed.as_ptr(),
            buf.as_mut_ptr().cast(),
            buf.len(),
        )
    };
    if st == turboembed_status::TURBOEMBED_OK {
        let s = unsafe { CStr::from_ptr(buf.as_ptr().cast()) }
            .to_string_lossy()
            .into_owned();
        assert_ne!(s, "", "OK must return a device string");
        Ok(s)
    } else {
        let msg = unsafe { CStr::from_ptr(turboembed_last_error(std::ptr::null())) }
            .to_string_lossy()
            .into_owned();
        Err(msg)
    }
}

#[test]
fn explicit_cpu_is_cpu_even_when_gpu_is_listed() {
    assert_eq!(require("CPU", "CPU").unwrap(), "CPU");
    assert_eq!(require("CPU", "CPU,GPU").unwrap(), "CPU");
    assert_eq!(require("CPU", "CPU,GPU.0").unwrap(), "CPU");
}

#[test]
fn gpu_selected_when_plugin_listed() {
    assert_eq!(require("GPU", "GPU").unwrap(), "GPU");
    assert_eq!(require("GPU", "CPU,GPU").unwrap(), "GPU");
    assert_eq!(require("GPU", "CPU,GPU.0").unwrap(), "GPU");
}

#[test]
fn gpu_missing_when_gpu_selected_is_loud_fail_never_cpu() {
    for listed in ["CPU", "CPU.0", "", "NPU"] {
        let err = require("GPU", listed).expect_err("GPU with no GPU plugin must fail");
        let lower = err.to_ascii_lowercase();
        assert!(
            lower.contains("gpu"),
            "error must name GPU, listed={listed:?}, got {err}"
        );
        assert!(
            lower.contains("fallback") || lower.contains("refus"),
            "error must say CPU fallback is refused, listed={listed:?}, got {err}"
        );
    }
}

#[test]
fn auto_is_rejected_at_the_pipeline_never_swapped() {
    let err = require("AUTO", "CPU,GPU").expect_err("AUTO is not a pipeline device");
    assert!(
        err.contains("AUTO"),
        "AUTO must be rejected (never rewritten to CPU or GPU), got {err}"
    );
}
