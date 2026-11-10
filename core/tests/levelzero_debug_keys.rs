//! A caller that set NEOReadDebugKeys to anything but 1 keeps it, and
//! EnableRecoverablePageFaults is not set either: turning the keys on
//! would switch on every other debug key in its environment (#2023,
//! core/src/ze.rs). A binary of its own, since the environment is the
//! process's and the backend decides once.
//!
//! Built with the `levelzero` feature only; it needs no device.

#![cfg(feature = "levelzero")]

mod common;

use common::first_of;

#[test]
fn a_caller_s_debug_keys_are_kept() {
    // SAFETY: the binary's one test, before any other thread reads them.
    unsafe {
        std::env::set_var("NEOReadDebugKeys", "0");
        std::env::remove_var("EnableRecoverablePageFaults");
    }
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo::turbo_runtime_create(std::ptr::null(), &mut rt, std::ptr::null_mut()) }, 0);
    // Listing opens the loader where there is one.
    let _ = first_of(rt, "levelzero");
    unsafe { turbo::turbo_runtime_release(rt) };
    assert_eq!(std::env::var("NEOReadDebugKeys").as_deref(), Ok("0"));
    assert!(std::env::var_os("EnableRecoverablePageFaults").is_none(), "the caller's environment was changed");
    // On a box with a loader the backend has decided, and says why.
    if let Some(f) = turbo::ze::page_faults_peek() {
        assert_eq!(*f, turbo::ze::Faults::CallersKeys("0".into()));
    }
}
