//! The Intel runtime's recoverable page faults, which the levelzero
//! backend turns off before it opens the loader (#1765, core/src/ze.rs):
//! a caller that set EnableRecoverablePageFaults itself keeps its value,
//! and NEOReadDebugKeys is then not touched. The other half, the
//! variables set when the caller set nothing, is in tests/levelzero.rs,
//! since the environment is the process's and each test binary is one.
//!
//! Built with the `levelzero` feature only; it needs no device.

#![cfg(feature = "levelzero")]

mod common;

use common::first_of;

#[test]
fn a_caller_s_page_fault_setting_is_kept() {
    // SAFETY: the binary's one test, before any other thread reads them.
    unsafe {
        std::env::set_var("EnableRecoverablePageFaults", "1");
        std::env::remove_var("NEOReadDebugKeys");
    }
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo::turbo_runtime_create(std::ptr::null(), &mut rt, std::ptr::null_mut()) }, 0);
    // Listing opens the loader where there is one.
    let _ = first_of(rt, "levelzero");
    unsafe { turbo::turbo_runtime_release(rt) };
    assert_eq!(std::env::var("EnableRecoverablePageFaults").as_deref(), Ok("1"));
    assert!(std::env::var_os("NEOReadDebugKeys").is_none(), "the caller's environment was changed");
}
