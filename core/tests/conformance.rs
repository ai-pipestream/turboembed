//! The conformance check: a bundle's reference cases run through the C
//! interface on one device, and compared with the vectors the upstream
//! pipeline produced. The same test serves every backend; docs/conformance.md
//! says how to run it and what it checks.
//!
//! TURBO_TEST_BUNDLE names the bundle directory, absolute or relative to
//! the workspace root; without it, the small sealed bundle in
//! testdata/tiny-bert-bundle. TURBO_TEST_DEVICE names the device, as a
//! runtime device index or a backend name (the first device that backend
//! lists); without it, the CPU. TURBO_TEST_PRECISION names the session's
//! precision, model, fastest or exact; without it, model.

mod common;

use std::path::{Path, PathBuf};
use std::ptr;

use common::*;
use serde_json::Value;
use turbo::*;

/// TURBO_TEST_BUNDLE, a relative path read from the workspace root (cargo
/// runs tests in the package directory), or None.
fn named_bundle() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE")?);
    Some(if p.is_absolute() { p } else { Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(p) })
}

fn bundle() -> PathBuf {
    named_bundle().unwrap_or_else(tiny_bundle)
}

/// The device TURBO_TEST_DEVICE names, or the CPU.
fn device(rt: *mut turbo_runtime) -> u32 {
    let Ok(want) = std::env::var("TURBO_TEST_DEVICE") else {
        return cpu(rt);
    };
    if let Ok(i) = want.parse() {
        return i;
    }
    let mut n = 0;
    assert_eq!(unsafe { turbo_runtime_device_count(rt, &mut n, ptr::null_mut()) }, 0);
    (0..n)
        .find(|&i| {
            let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
            info.struct_size = size_of::<turbo_device_info>() as u32;
            assert_eq!(unsafe { turbo_runtime_device_info(rt, i, &mut info, ptr::null_mut()) }, 0);
            field(&info.backend) == want
        })
        .unwrap_or_else(|| panic!("TURBO_TEST_DEVICE={want}: no device of that backend is listed"))
}

/// The name of the backend whose device TURBO_TEST_DEVICE names.
fn device_backend() -> String {
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, ptr::null_mut()) }, 0);
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(rt, device(rt), &mut info, ptr::null_mut()) }, 0);
    unsafe { turbo_runtime_release(rt) };
    field(&info.backend)
}

/// The precision TURBO_TEST_PRECISION names, or MODEL.
fn precision() -> u32 {
    match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => TURBO_PRECISION_MODEL,
        Ok("fastest") => TURBO_PRECISION_FASTEST,
        Ok("exact") => TURBO_PRECISION_EXACT,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model, fastest or exact"),
    }
}

#[test]
fn every_reference_case_matches_upstream() {
    conformance::check(&bundle(), device, precision());
}

/// The small bundle as a fixed-shape artifact of 32 tokens would have it:
/// the cases longer than that are refused with TURBO_E_CAPACITY, and the
/// rest still match.
#[test]
fn a_fixed_shape_refuses_the_cases_it_cannot_hold() {
    // The bundle is the small one's raw weights, which a backend that loads
    // only compiled artifacts never takes; a real bundle covers it there.
    let backend = device_backend();
    let raw = backend::format_bit(backend::TURBO_FORMAT_SAFETENSORS);
    if let Some(b) = backend::linked().iter().find(|b| b.name() == backend)
        && b.formats() & raw == 0
    {
        eprintln!("skipped: the {backend} backend does not load FORMAT_SAFETENSORS");
        return;
    }
    let dir = std::env::temp_dir().join(format!("turbo-conformance-{}-fixed", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors", "weights/model.safetensors"] {
        std::fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        std::fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let mut m: Value = serde_json::from_slice(&std::fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["artifacts"][0]["fixed_seq"] = 32.into();
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let long = conformance::check(&dir, device, precision());
    assert!(long > 0, "some case is longer than 32 tokens");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn a_real_bundle_matches_its_reference() {
    conformance::check(&named_bundle().expect("TURBO_TEST_BUNDLE is not set"), device, precision());
}
