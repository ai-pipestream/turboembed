//! The Hailo backend through the C interface: its table, the devices it
//! lists against what HailoRT scans, and a capability that runs nothing.
//!
//! Built with the `hailo` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run on
//! anything else in its place. With TURBO_TEST_REQUIRE_HAILO=1 it fails
//! instead, so a run on a Hailo machine cannot pass by finding no device.
//! docs/hailo.md says how to run them.

#![cfg(feature = "hailo")]

mod common;

use std::ffi::{c_char, c_int};
use std::ptr;

use common::*;
use turbo::status::{DEVICE_NOT_FOUND, UNSUPPORTED};
use turbo::*;

// HailoRT, which the library links, for what the tests check from the
// caller's side.
#[repr(C)]
#[derive(Clone, Copy)]
struct HailoDeviceId {
    id: [c_char; 32],
}

unsafe extern "C" {
    fn hailo_scan_devices(params: *mut std::ffi::c_void, ids: *mut HailoDeviceId, len: *mut usize) -> c_int;
}

struct Rt(*mut turbo_runtime);

impl Rt {
    fn new() -> Rt {
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, ptr::null_mut()) }, 0);
        Rt(rt)
    }

    fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, ptr::null_mut()) }, 0);
        n
    }

    fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }

    /// The runtime's indices of the devices the hailo backend listed.
    fn hailo(&self) -> Vec<u32> {
        (0..self.count()).filter(|&i| field(&self.info(i).backend) == "hailo").collect()
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_HAILO=1: a test that finds no device fails.
fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_HAILO").is_ok_and(|v| v == "1")
}

/// The runtime's first hailo device, or None after saying the test is
/// skipped.
fn hailo_device(rt: &Rt, test: &str) -> Option<u32> {
    let d = rt.hailo().first().copied();
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_HAILO=1 and the hailo backend lists no device");
        println!("{test}: skipped: the hailo feature is on and the hailo backend lists no device");
    }
    d
}

/// The device ids HailoRT scans, as it prints them: a PCIe device's BDF.
fn scanned() -> Vec<String> {
    let mut ids = [HailoDeviceId { id: [0; 32] }; 32];
    let mut n = ids.len();
    let rc = unsafe { hailo_scan_devices(ptr::null_mut(), ids.as_mut_ptr(), &mut n) };
    // HAILO_DRIVER_NOT_INSTALLED: no driver, so nothing is scanned.
    if rc == 64 {
        return Vec::new();
    }
    assert_eq!(rc, 0, "hailo_scan_devices");
    ids[..n].iter().map(|d| field(&d.id)).collect()
}

/// The architecture label the PCI device id of a scanned device gives,
/// read from sysfs: 1e60:45c4 is a Hailo-10H; 1e60:2864 is a Hailo-8 or a
/// Hailo-8L, which only the device tells apart.
fn labels_by_pci_id(bdf: &str) -> Option<&'static [&'static str]> {
    let dir = format!("/sys/bus/pci/devices/{bdf}");
    let read = |f: &str| std::fs::read_to_string(format!("{dir}/{f}")).ok().map(|s| s.trim().to_owned());
    match (read("vendor")?.as_str(), read("device")?.as_str()) {
        ("0x1e60", "0x45c4") => Some(&["hailo10h"]),
        ("0x1e60", "0x2864") => Some(&["hailo8", "hailo8l"]),
        _ => None,
    }
}

// ---- Without a device ------------------------------------------------------------

#[test]
fn the_table_stops_at_capability_and_names_itself() {
    let b = turbo::hailo::backend();
    assert_eq!(b.name(), "hailo");
    assert_eq!(b.struct_size as usize, std::mem::offset_of!(turbo::backend::turbo_backend, context_create));
    turbo::backend::check_table(b).unwrap();
    assert!(turbo::backend::linked().iter().any(|l| std::ptr::eq(*l, b)));
    let v = unsafe { std::ffi::CStr::from_ptr(turbo_version()) }.to_str().unwrap();
    assert!(v.split(' ').any(|w| w == "hailo"), "turbo_version() = {v}");
}

// ---- Devices ---------------------------------------------------------------------

/// Every device HailoRT scans is listed, in its order, as the hardware it
/// is; with none, the runtime is made and lists the rest.
#[test]
fn the_devices_listed_are_the_ones_hailort_scans() {
    let rt = Rt::new();
    let listed = rt.hailo();
    let scanned = scanned();
    if listed.is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_HAILO=1 and the hailo backend lists no device");
        assert!(scanned.is_empty(), "HailoRT scans {scanned:?} and the backend lists none: see the runtime's log");
        println!("the hailo backend lists no device: nothing to check but that the runtime was made");
        return;
    }
    assert_eq!(listed.len(), scanned.len(), "HailoRT scans {scanned:?}");
    for (o, (&i, bdf)) in listed.iter().zip(&scanned).enumerate() {
        let d = rt.info(i);
        assert_eq!(d.ordinal, o as u32);
        assert_eq!(d.kind, TURBO_DEVICE_NPU);
        assert_eq!(d.unified_memory, 0);
        assert_eq!((d.memory_total, d.memory_free), (0, 0), "HailoRT reports no memory size");
        assert_eq!(field(&d.vendor), "Hailo");
        let arch = field(&d.arch);
        if let Some(want) = labels_by_pci_id(bdf) {
            assert!(want.contains(&arch.as_str()), "{bdf}: arch {arch}, want one of {want:?}");
        }
        let name = field(&d.name);
        assert!(!name.is_empty());
        let runtime = field(&d.runtime_version);
        assert!(runtime.split('.').count() == 3 && runtime.split('.').all(|p| p.parse::<u32>().is_ok()), "{runtime}");
        let driver = field(&d.driver_version);
        assert!(driver.contains("firmware "), "{driver}");
        println!("hailo device {o} ({bdf}): {name}, arch {arch}, runtime {runtime}, driver {driver}");
    }
}

// ---- Capability ------------------------------------------------------------------

#[test]
fn embed_is_unsupported_at_every_precision_and_says_why() {
    let rt = Rt::new();
    let Some(d) = hailo_device(&rt, "embed_is_unsupported_at_every_precision_and_says_why") else { return };
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(rt.0, d, TURBO_TASK_EMBED, p, &mut cap, ptr::null_mut()) }, 0);
        assert_eq!(cap.status, backend::TURBO_CAP_UNSUPPORTED);
        assert_eq!((cap.dtype, cap.options_honored), (0, 0));
        assert!(field(&cap.benchmark).is_empty());
        assert!(field(&cap.reason).contains("runs no task"), "{}", field(&cap.reason));
    }
}

/// A device that runs no task is never the one select picks.
#[test]
fn select_does_not_pick_a_device_that_runs_nothing() {
    let rt = Rt::new();
    let Some(_) = hailo_device(&rt, "select_does_not_pick_a_device_that_runs_nothing") else { return };
    let mut out = u32::MAX;
    let mut err = new_error();
    let rc = unsafe { turbo_runtime_select(rt.0, TURBO_TASK_EMBED, &mut out, ptr::null_mut(), 0, &mut err) };
    if rc == 0 {
        assert_ne!(field(&rt.info(out).backend), "hailo");
    } else {
        assert_eq!(failure(rc, &err).code, DEVICE_NOT_FOUND);
    }
}

#[test]
fn a_context_is_refused_as_not_offered() {
    let rt = Rt::new();
    let Some(d) = hailo_device(&rt, "a_context_is_refused_as_not_offered") else { return };
    let mut ctx = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_context_create(rt.0, d, &mut ctx, &mut err) };
    let f = failure(rc, &err);
    assert_eq!(f.code, UNSUPPORTED, "{f:?}");
    assert!(f.message.contains("hailo") && f.message.contains("context_create"), "{}", f.message);
    assert!(ctx.is_null());
}
