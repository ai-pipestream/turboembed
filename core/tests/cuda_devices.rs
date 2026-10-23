//! The CUDA backend's devices, its capability, and its buffers in every
//! placement, with import and export of CUDA pointers.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run
//! on anything else in its place. With TURBO_TEST_REQUIRE_CUDA=1 it fails
//! instead, so a run on a GPU machine cannot pass by finding no GPU.
//! docs/cuda.md says how to run them; tests/cuda_support/ holds what the
//! files share.

#![cfg(feature = "cuda")]

mod common;
#[path = "cuda_support/mod.rs"]
mod cuda_support;

use common::*;
use cuda_support::*;
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED};
use turbo::*;

// ---- Without a device ------------------------------------------------------------

#[test]
fn the_backend_is_linked_first_and_names_itself() {
    let b = turbo::cuda::backend();
    assert_eq!(b.name(), "cuda");
    assert_eq!(b.struct_size as usize, size_of::<turbo::backend::turbo_backend>());
    turbo::backend::check_table(b).unwrap();
    assert!(b.buffer_read.is_some() && b.session_run.is_some());
    assert!(b.session_open.is_some() && b.embed_write_ranges.is_some(), "it pools ranges");
    assert!(std::ptr::eq(turbo::backend::linked()[0], b), "its devices come before the cpu's");
    let v = unsafe { std::ffi::CStr::from_ptr(b.runtime_version) }.to_str().unwrap();
    assert!(v.split('.').count() == 2 && v.split('.').all(|p| p.parse::<u32>().is_ok()), "runtime version {v}");
}

#[test]
fn arch_labels_come_from_the_device_name() {
    for (name, want) in [
        ("NVIDIA GeForce RTX 4080", "rtx4080"),
        ("NVIDIA GeForce RTX 4080 SUPER", "rtx4080super"),
        ("NVIDIA GeForce RTX 4080 Laptop GPU", "rtx4080laptop"),
        ("NVIDIA GeForce RTX 4070 Ti SUPER", "rtx4070tisuper"),
        ("NVIDIA GeForce RTX 3090", "rtx3090"),
        ("NVIDIA A100-SXM4-80GB", "a100"),
        ("NVIDIA A100 80GB PCIe", "a100"),
        ("NVIDIA H100 80GB HBM3", "h100"),
        ("NVIDIA H100 NVL", "h100"),
        ("NVIDIA L40S", "l40s"),
        ("Tesla T4", "t4"),
        ("Tesla V100-SXM2-16GB", "v100"),
        ("Quadro RTX 8000", "rtx8000"),
        ("NVIDIA RTX A6000", "rtxa6000"),
        ("NVIDIA RTX 6000 Ada Generation", "rtx6000ada"),
        ("NVIDIA RTX 2000 Ada Generation", "rtx2000ada"),
        ("NVIDIA RTX 4000 SFF Ada Generation", "rtx4000sffada"),
        ("Orin", "orin"),
    ] {
        assert_eq!(turbo::cuda::arch_label(name), want, "{name}");
    }
    let long = "NVIDIA ".to_owned() + &"X".repeat(100);
    assert_eq!(turbo::cuda::arch_label(&long).len(), 31, "cut to fit arch[32]");
}

/// With no device, the runtime is made and lists the rest; with one,
/// each device the backend lists reads as the hardware it is.
#[test]
fn the_devices_listed_are_the_drivers() {
    let _t = turn();
    let rt = Rt::new();
    let mut n = 0;
    assert_eq!(unsafe { turbo_runtime_device_count(rt.0, &mut n, null_err()) }, 0);
    let listed: Vec<turbo_device_info> = (0..n).map(|i| rt.info(i)).filter(|d| field(&d.backend) == "cuda").collect();
    if listed.is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_CUDA=1 and the cuda backend lists no device");
        println!("the cuda backend lists no device: nothing to check but that the runtime was made");
        assert!(n >= 1, "the cpu is still listed");
        return;
    }
    for (o, d) in listed.iter().enumerate() {
        assert_eq!(d.ordinal, o as u32);
        assert!(d.kind == TURBO_DEVICE_GPU || d.kind == TURBO_DEVICE_IGPU);
        assert_eq!(d.unified_memory, u32::from(d.kind == TURBO_DEVICE_IGPU));
        assert_eq!(field(&d.vendor), "NVIDIA");
        let name = field(&d.name);
        assert!(!name.is_empty());
        assert_eq!(field(&d.arch), turbo::cuda::arch_label(&name));
        assert!(d.memory_total > 0 && d.memory_free <= d.memory_total, "{} {}", d.memory_free, d.memory_total);
        assert!(!field(&d.runtime_version).is_empty() && field(&d.driver_version).contains("CUDA "));
        println!(
            "cuda device {o}: {name}, arch {}, runtime {}, driver {}",
            field(&d.arch),
            field(&d.runtime_version),
            field(&d.driver_version)
        );
    }
}

// ---- Capability ------------------------------------------------------------------

/// MODEL and EXACT compute in F32, FASTEST in F16, and a session says
/// what its cell says; the cell is SUPPORTED where a record in the build
/// backs it on this machine, naming the record, and EXPERIMENTAL else.
#[test]
fn embed_is_offered_in_the_dtype_sessions_run_it() {
    let _t = turn();
    let Some(dev) = cuda_device("embed_is_offered_in_the_dtype_sessions_run_it") else { return };
    let l = on_cuda(&tiny_bundle());
    let arch = field(&Rt::new().info(dev).arch);
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(l.rt, cuda(l.rt), TURBO_TASK_EMBED, p, &mut cap, null_err()) }, 0);
        // A benchmark record compiled into the build for this machine's
        // cell makes it SUPPORTED, naming the record; else it runs
        // unmeasured.
        if cap.status == backend::TURBO_CAP_SUPPORTED {
            let name = field(&cap.benchmark);
            assert!(name.starts_with(&format!("{arch}.cuda.embed.")), "{name}");
            assert!(cap.cosine_floor > 0.999 && cap.speed_ratio > 0.0, "a record's numbers");
            println!("precision {p}: SUPPORTED, backed by {name}");
        } else {
            assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&cap.reason));
        }
        let want = if p == TURBO_PRECISION_FASTEST { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 };
        assert_eq!(cap.dtype, want, "precision {p}");
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(info.compute_dtype, cap.dtype, "precision {p}");
    }
}

// ---- Buffers ---------------------------------------------------------------------

#[test]
fn each_placement_is_the_memory_it_names() {
    let _t = turn();
    let Some(dev) = cuda_device("each_placement_is_the_memory_it_names") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let ordinal = rt.info(dev).ordinal as u64;

    // DEVICE: no host address; its CUDA pointer is where the bytes are.
    let d = ctx.alloc(TURBO_PLACE_DEVICE, 1000).unwrap();
    assert!(d.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = d.export(TURBO_HANDLE_CUDA_PTR).unwrap();
    assert_eq!((h.kind, h.aux, h.offset), (TURBO_HANDLE_CUDA_PTR, ordinal, 0));
    to_device(h.handle, &values);
    assert_eq!(from_device(h.handle, 1000), values);
    assert!(d.export(TURBO_HANDLE_HOST_PTR).unwrap_err().is(UNSUPPORTED, "no host address"));

    // HOST and PINNED: host memory, exported as such.
    for pl in [TURBO_PLACE_HOST, TURBO_PLACE_PINNED] {
        let b = ctx.alloc(pl, 1000).unwrap();
        let p = b.host().unwrap();
        assert_eq!(p as usize % 64, 0, "placement {pl}: aligned");
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p as *mut f32, 1000) };
        assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
        assert!(b.export(TURBO_HANDLE_CUDA_PTR).unwrap_err().is(UNSUPPORTED, "host memory"), "placement {pl}");
    }

    // SHARED: one address, for the host and the device both.
    let s = ctx.alloc(TURBO_PLACE_SHARED, 1000).unwrap();
    let p = s.host().unwrap();
    let h = s.export(TURBO_HANDLE_CUDA_PTR).unwrap();
    assert_eq!((h.handle, h.aux), (p as u64, ordinal));
    to_device(h.handle, &values);
    assert_eq!(unsafe { std::slice::from_raw_parts(p as *const f32, 1000) }, values.as_slice());
    assert_eq!(s.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);

    for kind in [TURBO_HANDLE_CL_MEM, TURBO_HANDLE_DMABUF_FD] {
        let e = d.export(kind).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
}

#[test]
fn import_wraps_cuda_and_host_pointers_without_a_copy() {
    let _t = turn();
    let Some(dev) = cuda_device("import_wraps_cuda_and_host_pointers_without_a_copy") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let ordinal = rt.info(dev).ordinal as u64;
    let owner = ctx.alloc(TURBO_PLACE_DEVICE, 64).unwrap();
    let at = owner.export(TURBO_HANDLE_CUDA_PTR).unwrap().handle;
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    to_device(at, &values);

    // Eight floats in, no copy: the wrapped pointer is the owner's plus 32 bytes.
    let w = ctx.import(TURBO_PLACE_DEVICE, 56, handle(TURBO_HANDLE_CUDA_PTR, at, ordinal, 32)).unwrap();
    let h = w.export(TURBO_HANDLE_CUDA_PTR).unwrap();
    assert_eq!((h.handle, h.offset), (at + 32, 0));
    assert_eq!(from_device(h.handle, 56), values[8..]);
    assert!(w.host().is_err());

    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_CUDA_PTR, at, ordinal + 1, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "aux: device"), "another ordinal is refused: {e:?}");
    let e = ctx.import(TURBO_PLACE_HOST, 8, handle(TURBO_HANDLE_CUDA_PTR, at, ordinal, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "placement"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_SHARED, 8, handle(TURBO_HANDLE_CUDA_PTR, at, ordinal, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "not SHARED"), "device memory is not managed: {e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_CUDA_PTR, 0, ordinal, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "NULL"), "{e:?}");

    // Host memory as HOST, never as PINNED unless it is page-locked.
    let mut mine = vec![1.5f32; 16];
    let hp = mine.as_mut_ptr() as u64;
    let b = ctx.import(TURBO_PLACE_HOST, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, hp);
    let e = ctx.import(TURBO_PLACE_PINNED, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "not page-locked"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "host memory"), "{e:?}");
    let pinned = ctx.alloc(TURBO_PLACE_PINNED, 16).unwrap();
    let pp = pinned.host().unwrap() as u64;
    let b = ctx.import(TURBO_PLACE_PINNED, 16, handle(TURBO_HANDLE_HOST_PTR, pp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, pp);

    for kind in [TURBO_HANDLE_CL_MEM, TURBO_HANDLE_ZE_USM, TURBO_HANDLE_MTL_BUFFER, TURBO_HANDLE_DMABUF_FD] {
        let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(kind, at, 0, 0)).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("kind: TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
    drop(mine);
}
