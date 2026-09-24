//! Group `contract`, C ABI layer: a provider library whose ABI version is
//! not the core's is refused with `TURBO_E_ABI_MISMATCH`.
//!
//! This file holds one case and nothing else on purpose. It sets an
//! environment variable (`TURBO_PROVIDER_ABI_VERSION_OVERRIDE`, read only
//! where a provider library builds its vtable,
//! `crates/turbo-core/src/plugin_export.rs`) so that the mock provider's
//! cdylib reports a version the core does not implement. The environment is
//! process-wide and a provider library caches its vtable for the life of the
//! process, so a second case in the same binary would either race the
//! variable or see the cached vtable; one case per binary makes both
//! impossible.
//!
//! The matching half, a provider whose ABI does agree and therefore loads and
//! serves every task, is `providers/mock/tests/plugin_load.rs`.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c;

#[test]
fn abi_mismatch_a_provider_from_another_abi_is_refused() {
    let Some(path) = turbo_conformance::mock_plugin_path() else {
        // The artifact is a build product of another package in this
        // workspace, not something the suite can produce on its own. Every
        // workspace build (`cargo test --workspace`, what CI runs) makes it,
        // and this crate depends on the package so `cargo test -p
        // turbo-conformance` makes it too.
        panic!("the mock provider library is not built; run `cargo build -p turbo-provider-mock`");
    };
    let path = path.to_string_lossy().into_owned();
    // One version below the core's, so the message has two different numbers
    // to name and the refusal cannot be a comparison against itself.
    let reported = TURBO_PROVIDER_ABI_VERSION - 1;
    // SAFETY: this binary holds exactly one test, so no other thread is
    // reading or writing the environment while this runs.
    std::env::set_var(turbo::plugin_export::ENV_ABI_VERSION_OVERRIDE, reported.to_string());

    let mut e = c::err();
    let mut rt: *mut turbo_runtime = ptr::null_mut();
    // SAFETY: out pointers are valid; a NULL descriptor means defaults.
    unsafe {
        assert_eq!(turbo_runtime_create(ptr::null(), &mut rt, &mut e), TURBO_OK, "{}", c::message(&e));
        let rc = turbo_runtime_load_provider(rt, c::text(&path), &mut e);
        assert_eq!(
            rc,
            TURBO_E_ABI_MISMATCH,
            "a provider reporting ABI {reported} was not refused: got {} ({})",
            c::status_name(rc),
            c::message(&e)
        );
        let message = c::message(&e);
        assert!(message.contains(&reported.to_string()), "the message must name the provider's ABI: {message}");
        assert!(
            message.contains(&TURBO_PROVIDER_ABI_VERSION.to_string()),
            "the message must name the core's ABI: {message}"
        );
        assert_eq!(e.field, 0, "an ABI mismatch is not a field of a descriptor: {message}");
        // The runtime is intact: a refused provider changed nothing.
        let mut count = 0u32;
        assert_eq!(turbo_runtime_device_count(rt, &mut count, &mut e), TURBO_OK, "{}", c::message(&e));
        assert!(count > 0, "the built-in providers are still registered");
        turbo_runtime_release(rt);
    }
}
