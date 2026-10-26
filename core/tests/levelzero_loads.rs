//! Models loaded at once on Level Zero (#1355): several threads, each with
//! its own runtime and context, load one bundle at the same moment, and
//! every load succeeds. The bundle is TURBO_TEST_BUNDLE, else
//! testdata/tiny-bert-bundle. On main a bundle of about a gigabyte failed
//! the second load on an Arc Pro B70 with an out of memory error, though
//! the card had room for both.
//!
//! Built with the `levelzero` feature only. Without a listed device the
//! test says it was skipped; with TURBO_TEST_REQUIRE_LEVELZERO=1 it fails
//! instead.

#![cfg(feature = "levelzero")]

mod common;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

use common::{Loaded, first_of};

fn bundle() -> PathBuf {
    std::env::var_os("TURBO_TEST_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle"))
}

fn gpu(rt: *mut turbo::turbo_runtime) -> u32 {
    first_of(rt, "levelzero").expect("a levelzero device")
}

#[test]
fn models_load_at_once_on_several_contexts() {
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo::turbo_runtime_create(std::ptr::null(), &mut rt, std::ptr::null_mut()) }, 0);
    let listed = first_of(rt, "levelzero").is_some();
    unsafe { turbo::turbo_runtime_release(rt) };
    if !listed {
        let required = std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| v == "1");
        assert!(!required, "TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("skipped: the levelzero backend lists no device");
        return;
    }
    for threads in [2, 4] {
        let gate = Arc::new(Barrier::new(threads));
        let loads: Vec<_> = (0..threads)
            .map(|_| {
                let gate = gate.clone();
                std::thread::spawn(move || {
                    gate.wait();
                    Loaded::load_on(&bundle(), gpu).map(drop).map_err(|f| format!("{f:?}"))
                })
            })
            .collect();
        let failed: Vec<String> = loads.into_iter().filter_map(|t| t.join().unwrap().err()).collect();
        assert!(failed.is_empty(), "{threads} loads at once on {}: {failed:?}", bundle().display());
    }
}
