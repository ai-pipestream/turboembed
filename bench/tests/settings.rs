//! What a record's library.settings carries for a backend that reports
//! its kernel choices: what the bench copies from the session, and the
//! order the parser holds the names to, on the metal backend.

mod common;

use common::*;
use turbo::record::{self, Record};
use turbo::{TURBO_CHOICES_LEN, TURBO_TUNED_DEFAULT, TURBO_TUNED_MEASURED, turbo_session_info};
use turbo_bench::measure::library_settings;

fn reparse(r: &Record) -> Result<Record, String> {
    Record::parse(&record::file_name(r)?, &serde_json::to_vec(r).unwrap())
}

/// A session's report: `tuned` and the choices line `choices`.
fn session(tuned: u32, choices: &str) -> turbo_session_info {
    assert!(choices.len() < TURBO_CHOICES_LEN, "{choices}: longer than the field");
    let mut s: turbo_session_info = unsafe { std::mem::zeroed() };
    s.struct_size = size_of::<turbo_session_info>() as u32;
    s.tuned = tuned;
    for (d, b) in s.choices.iter_mut().zip(choices.bytes()) {
        *d = b as _;
    }
    s
}

/// A metal session's choices, and where they came from, are copied
/// whatever the environment says; a session with one path reports none,
/// and its record names none.
#[test]
fn a_metal_session_s_choices_are_copied() {
    let got = library_settings("metal", &session(TURBO_TUNED_MEASURED, "linear=h64;forced="));
    let set: Vec<String> = ["TURBO_AUTOTUNE", "TURBO_AUTOTUNE_BUDGET_MS"]
        .iter()
        .filter_map(|v| std::env::var(v).ok().map(|x| format!("{v}={x}")))
        .collect();
    let mut want = set.clone();
    want.push("TURBO_METAL_TUNED=measured".into());
    want.push("TURBO_METAL_CHOICES=linear=h64;forced=".into());
    assert_eq!(got, want);
    assert_eq!(library_settings("metal", &session(TURBO_TUNED_DEFAULT, "")), set);
}

/// A metal record names the kernels its session ran after the variables
/// that were set, in LIBRARY_VARS's order, and one out of that order, or
/// another backend's, is refused.
#[test]
fn a_metal_record_names_its_kernel_choices_last() {
    let mut g = cpu_record("metal-choices", vec![measured_reference(TEI)]);
    g.device.kind = "DEVICE_GPU".into();
    g.device.backend = "metal".into();
    g.machine.arch = "m2".into();
    let choices = "TURBO_METAL_CHOICES=linear=h64;forced=";
    g.library.settings = vec!["TURBO_AUTOTUNE=on".into(), "TURBO_METAL_TUNED=measured".into(), choices.into()];
    assert_eq!(reparse(&g).unwrap(), g);
    // A record from before the choices: none set, none reported.
    let mut old = g.clone();
    old.library.settings = vec![];
    assert_eq!(reparse(&old).unwrap(), old);
    let mut x = g.clone();
    x.library.settings = vec![choices.into(), "TURBO_METAL_TUNED=measured".into()];
    assert!(reparse(&x).unwrap_err().contains("in that order"));
    let mut x = g.clone();
    x.library.settings = vec!["TURBO_CUDA_CHOICES=linear=h64;forced=".into()];
    assert!(reparse(&x).unwrap_err().contains("library.settings"));
}
