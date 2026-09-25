//! The Metal backend's devices, through the C interface. Built only with
//! the metal feature, on a Mac whose GPU is real hardware.

#![cfg(feature = "metal")]

mod common;

use std::ffi::c_char;

use common::new_error;
use turbo::*;

struct Rt(*mut turbo_runtime);

impl Rt {
    fn new() -> Rt {
        let mut rt = std::ptr::null_mut();
        let mut err = new_error();
        assert_eq!(unsafe { turbo_runtime_create(std::ptr::null(), &mut rt, &mut err) }, 0);
        Rt(rt)
    }

    fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, std::ptr::null_mut()) }, 0);
        n
    }

    fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        assert_eq!(unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) }, 0, "{}", s(&err.message));
        out
    }

    fn capability(&self, i: u32, precision: u32) -> turbo_capability {
        let mut out: turbo_capability = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_capability>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_capability(self.0, i, TURBO_TASK_EMBED, precision, &mut out, &mut err) };
        assert_eq!(rc, 0, "{}", s(&err.message));
        out
    }

    /// The runtime's indexes of the devices the metal backend listed.
    fn metal(&self) -> Vec<u32> {
        (0..self.count()).filter(|&i| s(&self.info(i).backend) == "metal").collect()
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

fn s(b: &[c_char]) -> String {
    b.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect()
}

fn table() -> &'static backend::turbo_backend {
    backend::linked().iter().find(|b| b.name() == "metal").expect("the metal feature links the metal backend")
}

#[test]
fn the_gpu_is_listed_with_what_metal_says_about_it() {
    let rt = Rt::new();
    let metal = rt.metal();
    assert!(!metal.is_empty(), "every Mac this builds on has a Metal device");
    for (ordinal, &i) in metal.iter().enumerate() {
        let info = rt.info(i);
        let name = s(&info.name);
        assert_eq!(info.ordinal, ordinal as u32, "ordinals count within the backend");
        println!(
            "metal {ordinal}: {name} arch {} vendor {} kind {} memory {}/{} runtime {:?} driver {:?}",
            s(&info.arch),
            s(&info.vendor),
            info.kind,
            info.memory_free,
            info.memory_total,
            s(&info.runtime_version),
            s(&info.driver_version)
        );
        assert!(!name.is_empty());
        if let Some(chip) = name.strip_prefix("Apple ") {
            // Apple silicon: one memory, shared with the host.
            assert_eq!(info.kind, TURBO_DEVICE_IGPU);
            assert_eq!(info.unified_memory, 1);
            assert_eq!(s(&info.vendor), "Apple");
            assert_eq!(s(&info.arch), chip.to_lowercase().replace(' ', ""), "{name}");
        }
        assert!(!s(&info.arch).is_empty());
        assert!(info.memory_total > 0);
        assert!(info.memory_free <= info.memory_total, "{} {}", info.memory_free, info.memory_total);
        assert!(s(&info.runtime_version).starts_with("Metal, macOS SDK "), "{}", s(&info.runtime_version));
        assert!(s(&info.driver_version).starts_with("macOS "), "{}", s(&info.driver_version));
    }
}

#[test]
fn the_table_names_the_sdk_it_was_built_against() {
    let b = table();
    backend::check_table(b).unwrap();
    let v = unsafe { std::ffi::CStr::from_ptr(b.runtime_version) }.to_str().unwrap();
    let sdk = v.strip_prefix("Metal, macOS SDK ").unwrap_or_else(|| panic!("{v}"));
    assert!(sdk.split('.').all(|p| p.parse::<u32>().is_ok()), "{v}");
    let rt = Rt::new();
    for i in rt.metal() {
        assert_eq!(s(&rt.info(i).runtime_version), v);
    }
}

#[test]
fn ordinals_name_the_same_device_in_every_runtime() {
    let (a, b) = (Rt::new(), Rt::new());
    let names = |rt: &Rt| rt.metal().iter().map(|&i| s(&rt.info(i).name)).collect::<Vec<_>>();
    assert_eq!(names(&a), names(&b));
}

#[test]
fn embed_is_not_built_and_says_why() {
    let rt = Rt::new();
    for i in rt.metal() {
        for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
            let cap = rt.capability(i, p);
            assert_eq!(cap.status, backend::TURBO_CAP_UNSUPPORTED);
            assert_eq!(s(&cap.reason), "the metal backend has no embed kernels in this build");
            assert_eq!((cap.dtype, cap.options_honored), (0, 0));
        }
    }
}

#[test]
fn the_table_refuses_an_ordinal_it_did_not_list() {
    // The core checks first; the table still answers only for its own.
    let b = table();
    let mut n = 0;
    assert_eq!(unsafe { (b.device_count)(&mut n, std::ptr::null_mut()) }, 0);
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    let mut err = new_error();
    let rc = unsafe { (b.device_info)(n, &mut info, &mut err) };
    assert_eq!(rc, status::INVALID_ARGUMENT);
    assert_eq!(err.code, status::INVALID_ARGUMENT);
    assert_eq!(s(&err.message), format!("metal device {n}: {n} listed"));
    let rc = unsafe { (b.device_info)(n, &mut info, std::ptr::null_mut()) };
    assert_eq!(rc, status::INVALID_ARGUMENT, "a NULL error is allowed");
    let (mut st, mut dt, mut oh) = (9, 9, 9);
    let mut reason = [0 as c_char; 8];
    let rc = unsafe {
        (b.capability)(n, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 8, std::ptr::null_mut())
    };
    assert_eq!(rc, status::INVALID_ARGUMENT);
    assert_eq!((st, dt, oh), (9, 9, 9), "outputs are untouched on failure");
}

#[test]
fn the_reason_fits_the_buffer_it_is_given() {
    let b = table();
    let (mut st, mut dt, mut oh) = (0, 0, 0);
    let mut reason = [b'x' as c_char; 16];
    let rc = unsafe {
        (b.capability)(0, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 8, std::ptr::null_mut())
    };
    assert_eq!(rc, 0);
    assert_eq!(s(&reason), "the met");
    assert!(reason[8..].iter().all(|&c| c == b'x' as c_char), "nothing written past reason_len");
    let mut reason = [b'x' as c_char; 4];
    let rc = unsafe {
        (b.capability)(0, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 0, std::ptr::null_mut())
    };
    assert_eq!(rc, 0);
    assert!(reason.iter().all(|&c| c == b'x' as c_char), "reason_len 0 writes nothing");
}

#[test]
fn nothing_but_memory_free_differs_between_queries() {
    let rt = Rt::new();
    let i = rt.metal()[0];
    let (mut a, mut b) = (rt.info(i), rt.info(i));
    assert!(a.memory_free > 0);
    a.memory_free = 0;
    b.memory_free = 0;
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}
