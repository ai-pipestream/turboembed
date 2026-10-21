//! The Metal backend through the C interface: its devices, what it offers,
//! its buffers in every placement, import and export of Metal buffers and
//! host memory, and weights read where the core holds them.
//!
//! Built with the `metal` feature only, on macOS. A test that needs a
//! device that runs the kernels says it was skipped, and passes, when the
//! backend lists none; nothing is run on anything else in its place. With
//! TURBO_TEST_REQUIRE_METAL=1 it fails instead. docs/metal.md says how to
//! run them.

#![cfg(feature = "metal")]

mod common;
mod metal_support;

use std::ffi::c_void;
use std::ptr;

use common::*;
use metal_support::*;
use serde_json::json;
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;

// ---- Devices ---------------------------------------------------------------------

#[test]
fn the_backend_is_linked_before_the_cpu_and_names_itself() {
    let b = table();
    assert_eq!(b.name(), "metal");
    assert_eq!(b.struct_size as usize, size_of::<backend::turbo_backend>());
    backend::check_table(b).unwrap();
    assert!(b.buffer_read.is_some() && b.session_run.is_some());
    assert!(b.session_open.is_some() && b.embed_write_ranges.is_some(), "it pools ranges");
    let linked = backend::linked();
    let at = |name: &str| linked.iter().position(|t| t.name() == name);
    assert!(at("metal") < at("cpu"), "its devices come before the cpu's");
    let v = unsafe { CStr::from_ptr(b.runtime_version) }.to_str().unwrap();
    let sdk = v.strip_prefix("Metal, macOS SDK ").unwrap_or_else(|| panic!("{v}"));
    assert!(sdk.split('.').all(|p| p.parse::<u32>().is_ok()), "{v}");
}

#[test]
fn the_gpu_is_listed_with_what_metal_says_about_it() {
    let rt = Rt::new();
    let metal = rt.metal();
    if metal.is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        return println!("skipped: the metal backend lists no device");
    }
    for (ordinal, &i) in metal.iter().enumerate() {
        let info = rt.info(i);
        let name = field(&info.name);
        println!(
            "metal {ordinal}: {name} arch {} vendor {} kind {} memory {}/{} runtime {:?} driver {:?}",
            field(&info.arch),
            field(&info.vendor),
            info.kind,
            info.memory_free,
            info.memory_total,
            field(&info.runtime_version),
            field(&info.driver_version)
        );
        assert_eq!(info.ordinal, ordinal as u32, "ordinals count within the backend");
        assert!(!name.is_empty());
        if let Some(chip) = name.strip_prefix("Apple ") {
            // Apple silicon: one memory, shared with the host.
            assert_eq!(info.kind, TURBO_DEVICE_IGPU);
            assert_eq!(info.unified_memory, 1);
            assert_eq!(field(&info.vendor), "Apple");
            assert_eq!(field(&info.arch), chip.to_lowercase().replace(' ', ""), "{name}");
            assert!(info.memory_free > 0 && info.memory_free <= info.memory_total);
        }
        assert!(!field(&info.arch).is_empty());
        assert!(info.memory_total > 0);
        assert!(info.memory_free <= info.memory_total, "{} {}", info.memory_free, info.memory_total);
        assert_eq!(field(&info.runtime_version), unsafe { CStr::from_ptr(table().runtime_version) }.to_str().unwrap());
        assert!(field(&info.driver_version).starts_with("macOS "), "{}", field(&info.driver_version));
    }
}

#[test]
fn ordinals_name_the_same_device_in_every_runtime() {
    let (a, b) = (Rt::new(), Rt::new());
    let names = |rt: &Rt| rt.metal().iter().map(|&i| field(&rt.info(i).name)).collect::<Vec<_>>();
    assert_eq!(names(&a), names(&b));
}

#[test]
fn nothing_but_memory_free_differs_between_queries() {
    let rt = Rt::new();
    let Some(&i) = rt.metal().first() else {
        assert!(!required(), "TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        return println!("skipped: the metal backend lists no device");
    };
    let (mut a, mut b) = (rt.info(i), rt.info(i));
    a.memory_free = 0;
    b.memory_free = 0;
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn the_table_refuses_an_ordinal_it_did_not_list() {
    // The core checks first; the table still answers only for its own.
    let b = table();
    let mut n = 0;
    assert_eq!(unsafe { (b.device_count)(&mut n, null_err()) }, 0);
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    let mut err = new_error();
    let rc = unsafe { (b.device_info)(n, &mut info, &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT);
    assert_eq!(err.code, INVALID_ARGUMENT);
    assert_eq!(field(&err.message), format!("metal device {n}: {n} listed"));
    assert_eq!(unsafe { (b.device_info)(n, &mut info, null_err()) }, INVALID_ARGUMENT, "a NULL error is allowed");
    let (mut st, mut dt, mut oh) = (9, 9, 9);
    let mut reason = [0 as c_char; 8];
    let rc = unsafe {
        (b.capability)(n, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 8, null_err())
    };
    assert_eq!(rc, INVALID_ARGUMENT);
    assert_eq!((st, dt, oh), (9, 9, 9), "outputs are untouched on failure");
}

#[test]
fn a_reason_fits_the_buffer_it_is_given() {
    let b = table();
    if Rt::new().metal().is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        return println!("skipped: the metal backend lists no device");
    }
    let (mut st, mut dt, mut oh) = (0, 0, 0);
    let mut reason = [b'x' as c_char; 16];
    let rc = unsafe {
        (b.capability)(0, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 8, null_err())
    };
    assert_eq!(rc, 0);
    // Empty where the device runs embed, else cut to fit.
    let end = reason.iter().position(|&c| c == 0).expect("terminated");
    assert!(end < 8, "within 8 bytes");
    assert!(reason[8..].iter().all(|&c| c == b'x' as c_char), "nothing written past reason_len");
    let mut reason = [b'x' as c_char; 4];
    let rc = unsafe {
        (b.capability)(0, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 0, null_err())
    };
    assert_eq!(rc, 0);
    assert!(reason.iter().all(|&c| c == b'x' as c_char), "reason_len 0 writes nothing");
}

// ---- Capability ------------------------------------------------------------------

#[test]
fn embed_is_offered_in_the_dtype_sessions_run_it_in() {
    let _t = turn();
    let Some(_) = metal_device("embed_is_offered_in_the_dtype_sessions_run_it_in") else { return };
    let l = on_metal(&tiny_bundle());
    let rt = std::mem::ManuallyDrop::new(Rt(l.rt));
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let i = metal(l.rt);
        let cap = rt.capability(i, p);
        // A benchmark record compiled into the build for this machine's
        // cell makes it SUPPORTED, naming the record; else it runs
        // unmeasured.
        if cap.status == backend::TURBO_CAP_SUPPORTED {
            assert!(field(&cap.benchmark).starts_with(&format!("{}.metal.embed.", field(&rt.info(i).arch))));
            assert!(cap.cosine_floor > 0.999 && cap.speed_ratio > 0.0, "a record's numbers");
            assert_eq!(field(&cap.reason), "");
        } else {
            assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&cap.reason));
            assert_eq!(field(&cap.reason), "no benchmark record for this cell");
        }
        // FASTEST's linear layers compute in F16, the rest in F32.
        let dtype = if p == TURBO_PRECISION_FASTEST { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 };
        assert_eq!(cap.dtype, dtype, "precision {p}");
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(info.compute_dtype, cap.dtype, "precision {p}");
    }
}

// ---- Buffers ---------------------------------------------------------------------

#[test]
fn each_placement_is_the_memory_it_names() {
    let _t = turn();
    let Some(dev) = metal_device("each_placement_is_the_memory_it_names") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();

    // DEVICE: a private Metal buffer, with no host address.
    let d = ctx.alloc(TURBO_PLACE_DEVICE, 1000).unwrap();
    assert!(d.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = d.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!((h.kind, h.offset), (TURBO_HANDLE_MTL_BUFFER, 0));
    assert_eq!(send(h.handle, c"storageMode") as usize, STORAGE_PRIVATE);
    assert!(send(h.handle, c"length") >= 4000);
    assert_eq!(send(h.handle, c"device"), h.aux, "aux is the buffer's MTLDevice");
    assert!(d.export(TURBO_HANDLE_HOST_PTR).unwrap_err().is(UNSUPPORTED, "no host address"));
    let device = h.aux;

    // PINNED and SHARED: shared Metal buffers, one address for both.
    for pl in [TURBO_PLACE_PINNED, TURBO_PLACE_SHARED] {
        let b = ctx.alloc(pl, 1000).unwrap();
        let p = b.host().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p as *mut f32, 1000) };
        let h = b.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
        assert_eq!((h.aux, h.offset), (device, 0));
        assert_eq!(send(h.handle, c"storageMode") as usize, STORAGE_SHARED, "placement {pl}");
        assert_eq!(send(h.handle, c"contents"), p as u64, "placement {pl}: the host address is the buffer's");
        assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
    }

    // HOST: host memory, not a Metal buffer.
    let b = ctx.alloc(TURBO_PLACE_HOST, 1000).unwrap();
    let p = b.host().unwrap();
    assert_eq!(p as usize % 64, 0, "aligned");
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p as *mut f32, 1000) };
    assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
    assert!(b.export(TURBO_HANDLE_MTL_BUFFER).unwrap_err().is(UNSUPPORTED, "not a Metal buffer"));

    for kind in [TURBO_HANDLE_CUDA_PTR, TURBO_HANDLE_CL_MEM, TURBO_HANDLE_DMABUF_FD] {
        let e = d.export(kind).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
}

#[test]
fn import_wraps_metal_buffers_and_host_pages_without_a_copy() {
    let _t = turn();
    let Some(dev) = metal_device("import_wraps_metal_buffers_and_host_pages_without_a_copy") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let device = ctx.alloc(TURBO_PLACE_DEVICE, 1).unwrap().export(TURBO_HANDLE_MTL_BUFFER).unwrap().aux;

    // A shared buffer of the caller's, eight floats in: the host address is
    // the buffer's own plus 32 bytes.
    let shared = new_buffer(device, 256, 0);
    let contents = send(shared, c"contents");
    let w = ctx.import(TURBO_PLACE_SHARED, 56, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 32)).unwrap();
    assert_eq!(w.host().unwrap() as u64, contents + 32);
    let h = w.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!((h.handle, h.aux, h.offset), (shared, device, 32), "the same buffer handed back");
    let e = ctx.import(TURBO_PLACE_SHARED, 64, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 32)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "past the buffer"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "placement: DEVICE"), "a shared buffer is not DEVICE memory: {e:?}");
    let e = ctx.import(TURBO_PLACE_SHARED, 8, handle(TURBO_HANDLE_MTL_BUFFER, shared, device + 16, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "aux"), "another device is refused: {e:?}");
    let e = ctx.import(TURBO_PLACE_HOST, 8, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "placement: HOST"), "{e:?}");
    drop(w);

    // A private buffer as DEVICE, and never as SHARED.
    let private = new_buffer(device, 256, OPTIONS_PRIVATE);
    let w = ctx.import(TURBO_PLACE_DEVICE, 64, handle(TURBO_HANDLE_MTL_BUFFER, private, device, 0)).unwrap();
    assert!(w.host().is_err());
    let e = ctx.import(TURBO_PLACE_SHARED, 8, handle(TURBO_HANDLE_MTL_BUFFER, private, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "private"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_MTL_BUFFER, 0, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "NULL"), "{e:?}");
    drop(w);
    // The caller's buffers are still the caller's to release.
    send(shared, c"release");
    send(private, c"release");

    // Host memory as HOST anywhere, as PINNED or SHARED only on whole pages.
    let mut mine = vec![1.5f32; 16];
    let hp = mine.as_mut_ptr() as u64;
    let b = ctx.import(TURBO_PLACE_HOST, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, hp);
    let e = ctx.import(TURBO_PLACE_PINNED, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "whole pages"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "host memory"), "{e:?}");
    let pages = Pages::new(16384);
    let pp = pages.0 as u64;
    let b = ctx.import(TURBO_PLACE_SHARED, 4096, handle(TURBO_HANDLE_HOST_PTR, pp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, pp);
    let h = b.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!(send(h.handle, c"contents"), pp, "Metal reads the caller's pages where they are");
    drop(b);

    for kind in [TURBO_HANDLE_CUDA_PTR, TURBO_HANDLE_CL_MEM, TURBO_HANDLE_ZE_USM, TURBO_HANDLE_DMABUF_FD] {
        let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(kind, hp, 0, 0)).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("kind: TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
    drop(mine);
}

// ---- Models ----------------------------------------------------------------------

#[test]
fn the_device_reads_the_weights_where_the_core_holds_them() {
    let _t = turn();
    let Some(_) = metal_device("the_device_reads_the_weights_where_the_core_holds_them") else { return };
    let l = on_metal(&tiny_bundle());
    assert_eq!(unsafe { model_metal_in_place(l.m) }, Some(true), "no copy of the weights");
    assert!(unsafe { model_converted_weights(l.m) }.is_none());
}
