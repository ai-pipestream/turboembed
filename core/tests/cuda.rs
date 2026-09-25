//! The CUDA backend through the C interface: its devices, its buffers in
//! every placement, import and export of CUDA pointers, a run's vectors
//! left on the device, the bytes a run moves, and a warm run's allocation
//! count. The encoder's vectors are held to the f64 arithmetic of
//! tests/common and to the CPU backend's; tests/conformance.rs with
//! TURBO_TEST_DEVICE=cuda holds them to the upstream reference.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run
//! on anything else in its place. With TURBO_TEST_REQUIRE_CUDA=1 it fails
//! instead, so a run on a GPU machine cannot pass by finding no GPU.
//! docs/cuda.md says how to run them.

#![cfg(feature = "cuda")]

mod common;

use std::ffi::c_void;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::*;
use serde_json::json;
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;

// The CUDA runtime the library links, for what the tests check from the
// caller's side of a pointer.
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
}
const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

/// The tests take turns: the allocation counts and the device's free
/// memory are the process's, and another test's work would move them.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}

const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];

struct Rt(*mut turbo_runtime);

impl Rt {
    fn new() -> Rt {
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, null_err()) }, 0);
        Rt(rt)
    }

    fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_CUDA=1: a test that finds no device fails.
fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| v == "1")
}

/// The first CUDA device, or None after saying the test is skipped.
fn cuda_device(test: &str) -> Option<u32> {
    let rt = Rt::new();
    let d = first_of(rt.0, "cuda");
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_CUDA=1 and the cuda backend lists no device");
        println!("{test}: skipped: the cuda feature is on and the cuda backend lists no device");
    }
    d
}

fn cuda(rt: *mut turbo_runtime) -> u32 {
    first_of(rt, "cuda").expect("a cuda device")
}

fn on_cuda(dir: &std::path::Path) -> Loaded {
    Loaded::load_on(dir, cuda).unwrap_or_else(|e| panic!("{e:?}"))
}

fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

// ---- Without a device ------------------------------------------------------------

#[test]
fn the_backend_is_linked_first_and_names_itself() {
    let b = turbo::cuda::backend();
    assert_eq!(b.name(), "cuda");
    assert_eq!(b.struct_size as usize, size_of::<turbo::backend::turbo_backend>());
    turbo::backend::check_table(b).unwrap();
    assert!(b.buffer_read.is_some() && b.session_run.is_some());
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

#[test]
fn embed_is_offered_in_f32_as_sessions_run_it() {
    let _t = turn();
    let Some(_) = cuda_device("embed_is_offered_in_f32_as_sessions_run_it") else { return };
    let l = on_cuda(&tiny_bundle());
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(l.rt, cuda(l.rt), TURBO_TASK_EMBED, p, &mut cap, null_err()) }, 0);
        assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&cap.reason));
        assert_eq!(cap.dtype, TURBO_DTYPE_F32);
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(info.compute_dtype, cap.dtype, "precision {p}");
    }
}

// ---- Buffers ---------------------------------------------------------------------

/// A context, released on drop when `owned`.
struct Ctx(*mut turbo_context, bool);

impl Ctx {
    fn alloc(&self, placement: u32, n: u64) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &d, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }

    fn import(&self, placement: u32, n: u64, h: turbo_native_handle) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_import(self.0, &d, &h, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        if self.1 {
            unsafe { turbo_context_release(self.0) };
        }
    }
}

#[derive(Debug)]
struct Buf(*mut turbo_buffer);

impl Buf {
    fn host(&self) -> Result<*mut c_void, Failure> {
        let mut p = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_host_ptr(self.0, &mut p, &mut err) } {
            0 => Ok(p),
            rc => Err(failure(rc, &err)),
        }
    }

    fn export(&self, kind: u32) -> Result<turbo_native_handle, Failure> {
        let mut h: turbo_native_handle = unsafe { std::mem::zeroed() };
        h.struct_size = size_of::<turbo_native_handle>() as u32;
        let mut err = new_error();
        match unsafe { turbo_buffer_export(self.0, kind, &mut h, &mut err) } {
            0 => Ok(h),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { turbo_buffer_release(self.0) };
    }
}

/// n F32 values in one row.
fn desc(placement: u32, n: u64) -> turbo_buffer_desc {
    turbo_buffer_desc {
        struct_size: size_of::<turbo_buffer_desc>() as u32,
        placement,
        dtype: TURBO_DTYPE_F32,
        ndim: 1,
        shape: [n, 0],
        bytes: 0,
    }
}

fn handle(kind: u32, at: u64, aux: u64, offset: u64) -> turbo_native_handle {
    turbo_native_handle { struct_size: size_of::<turbo_native_handle>() as u32, kind, handle: at, aux, offset }
}

fn context(rt: &Rt, device: u32) -> Ctx {
    let mut c = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_context_create(rt.0, device, &mut c, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    Ctx(c, true)
}

fn to_device(dst: u64, src: &[f32]) {
    assert_eq!(
        unsafe { cudaMemcpy(dst as *mut c_void, src.as_ptr() as *const c_void, src.len() * 4, HOST_TO_DEVICE) },
        0
    );
}

fn from_device(src: u64, n: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; n];
    assert_eq!(unsafe { cudaMemcpy(v.as_mut_ptr() as *mut c_void, src as *const c_void, n * 4, DEVICE_TO_HOST) }, 0);
    v
}

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

// ---- Runs ------------------------------------------------------------------------

/// Rows with padding on either side, a masked token inside a row and token
/// types, as tests/sessions.rs writes them for the CPU.
fn awkward_rows(dir: &std::path::Path) -> Tokens {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    let mut t = Tokens::new(&[vec![0; seq], vec![0; seq], vec![0; seq]], 0);
    t.mask.fill(0);
    let mut types = vec![0; 3 * seq];
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            if r == 2 && p > 0 {
                types[r * seq + start + p] = 1;
            }
        }
    }
    t.mask[2 * seq + 1] = 0;
    t.types = Some(types);
    t
}

/// Every pooling, with and without normalization, against the f64
/// encoder of tests/common and against the CPU backend.
#[test]
fn every_option_matches_the_arithmetic_and_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("every_option_matches_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_cuda(&dir), Loaded::load(&dir).unwrap());
    let (gs, cs) = (Session::create(g.m, None).unwrap(), Session::create(c.m, None).unwrap());
    let t = awkward_rows(&dir);
    let seq = t.seq as usize;
    let types = t.types.clone().unwrap();
    let (mut worst, mut lowest) = (0f64, 1f64);
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            gs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = gs.run().unwrap().rows();
            cs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let cpu = cs.run().unwrap().rows();
            for (r, row) in got.iter().enumerate() {
                let at = r * seq..(r + 1) * seq;
                let want = plain.embed(
                    &t.ids[at.clone()],
                    &t.mask[at.clone()],
                    &types[at],
                    pooling,
                    32,
                    normalize == TURBO_NORMALIZE_L2,
                );
                for (a, w) in row.iter().zip(&want) {
                    let d = (*a as f64 - w).abs();
                    worst = worst.max(d);
                    assert!(d < 1e-5 * (1.0 + w.abs()), "pooling {pooling} normalize {normalize} row {r}: {a} vs {w}");
                }
                let c = cosine(row, &cpu[r]);
                lowest = lowest.min(c);
                assert!(c >= 0.9999, "pooling {pooling} normalize {normalize} row {r}: cosine {c} with the cpu");
            }
        }
    }
    println!(
        "largest difference from the f64 encoder {worst:.3e}; 1 - lowest cosine with the cpu {:.3e}",
        1.0 - lowest
    );

    // Text, and the same rows as tokens at a wider stride, give the same vectors.
    let from_text = gs.embed(&TEXTS, None).unwrap();
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let packed = Tokens::new(&rows, 0);
    let seq = packed.seq as usize;
    let mut wide = Tokens::new(&rows, 0);
    wide.stride = seq as u32 + 5;
    wide.ids = vec![-9; rows.len() * wide.stride as usize];
    wide.mask = vec![-9; rows.len() * wide.stride as usize];
    for r in 0..rows.len() {
        let at = r * wide.stride as usize;
        wide.ids[at..at + seq].copy_from_slice(&packed.ids[r * seq..(r + 1) * seq]);
        wide.mask[at..at + seq].copy_from_slice(&packed.mask[r * seq..(r + 1) * seq]);
    }
    gs.write_tokens(&wide.batch(), None).unwrap();
    assert_eq!(gs.run().unwrap().rows(), from_text, "the stride's gap is never read");
}

/// The vectors stay on the device: the result's buffer is device memory
/// whose CUDA pointer holds what turbo_result_read copies back, and the
/// counts are the bytes that crossed.
#[test]
fn a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed() {
    let _t = turn();
    let Some(_) = cuda_device("a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed") else { return };
    let l = on_cuda(&tiny_bundle());
    let mi = l.info();
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let seq = TEXTS.iter().map(|t| tok.row(t, None).unwrap().len()).max().unwrap() as u64;

    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let i = r.info();
    assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32));
    assert_eq!((i.dtype, i.compute_dtype, i.placement), (TURBO_DTYPE_F32, TURBO_DTYPE_F32, TURBO_PLACE_DEVICE));
    assert_eq!(i.device, cuda(l.rt));
    assert_eq!(i.bytes, 3 * 32 * 4);
    assert_eq!(i.h2d_bytes, 2 * 3 * seq * 4, "ids and mask, [3, {seq}] int32 each");
    assert_eq!(i.d2h_bytes, 0, "nothing came back yet");
    assert_eq!((i.host_allocs, i.device_allocs), (0, 0));
    let (h, d, f, u) = (TURBO_STAGE_HOST, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED);
    assert_eq!(i.stage[..7], [h, d, d, d, d, f, u], "tokenize, upload, lookup, encode, pool, normalize, download");
    assert!(i.stage[7..].iter().all(|&s| s == u));
    assert_eq!(field(&i.backend), "cuda");
    assert_eq!(field(&i.manifest_sha256), field(&mi.manifest_sha256));

    let read = r.rows().concat();
    assert_eq!(r.info().d2h_bytes, i.bytes, "a read is counted");
    r.rows();
    assert_eq!(r.info().d2h_bytes, 2 * i.bytes, "each time");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let buf = Buf(buf);
    let mut bd: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    bd.struct_size = size_of::<turbo_buffer_desc>() as u32;
    assert_eq!(unsafe { turbo_buffer_get_desc(buf.0, &mut bd, null_err()) }, 0);
    assert_eq!((bd.placement, bd.shape, bd.bytes), (TURBO_PLACE_DEVICE, [3, 32], 384));
    assert!(buf.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = buf.export(TURBO_HANDLE_CUDA_PTR).unwrap();
    assert_eq!(h.aux, rt_ordinal(&l), "the device's own ordinal");
    assert_eq!(from_device(h.handle, read.len()), read, "the device pointer holds the vectors, no copy made");
    drop(buf);
    drop(r);

    // Tokens the caller wrote, with types: three arrays cross; nothing
    // is tokenized, and an unnormalized vector is not normalized.
    let mut b = Tokens::new(&[vec![101, 7592, 102]], 0);
    b.types = Some(vec![0, 1, 1]);
    s.write_tokens(&b.batch(), Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    let i = s.run().unwrap().info();
    assert_eq!(i.h2d_bytes, 3 * 3 * 4);
    assert_eq!(i.stage[..7], [u, d, d, d, d, u, u]);
    assert_eq!((i.batch, i.d2h_bytes), (1, 0), "a run's count starts again");
}

fn rt_ordinal(l: &Loaded) -> u64 {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(l.rt, cuda(l.rt), &mut info, null_err()) }, 0);
    info.ordinal as u64
}

/// Rows in page-locked memory the caller got from a PINNED buffer go to
/// the device from where they are, and give the same vectors.
#[test]
fn rows_in_pinned_memory_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = cuda_device("rows_in_pinned_memory_give_the_same_vectors") else { return };
    let l = on_cuda(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let t = awkward_rows(&tiny_bundle());
    s.write_tokens(&t.batch(), None).unwrap();
    let want = s.run().unwrap().rows();

    let ctx = Ctx(l.ctx, false);
    let n = t.ids.len() as u64;
    let bufs: Vec<Buf> = (0..3).map(|_| ctx.alloc(TURBO_PLACE_PINNED, n).unwrap()).collect();
    let at: Vec<*mut i32> = bufs.iter().map(|b| b.host().unwrap() as *mut i32).collect();
    let types = t.types.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(t.ids.as_ptr(), at[0], t.ids.len());
        std::ptr::copy_nonoverlapping(t.mask.as_ptr(), at[1], t.mask.len());
        std::ptr::copy_nonoverlapping(types.as_ptr(), at[2], types.len());
    }
    let mut b = t.batch();
    b.ids = at[0];
    b.mask = at[1];
    b.types = at[2];
    s.write_tokens(&b, None).unwrap();
    let r = s.run().unwrap();
    assert_eq!(r.rows(), want);
    assert_eq!(r.info().h2d_bytes, 3 * n * 4);
}

/// A warm run allocates nothing on the host or the device: the backend's
/// own count does not move, the result says 0, and the device's free
/// memory is what it was.
#[test]
fn a_warm_run_allocates_nothing() {
    let _t = turn();
    let Some(_) = cuda_device("a_warm_run_allocates_nothing") else { return };
    let l = on_cuda(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let small = Tokens::new(&rows, 0);
    let long = Tokens::new(&vec![(0..64).map(|i| 1000 + i).collect(); 64], 0);
    let mut dst = vec![0f32; 64 * 32];
    let mut run = |t: &Tokens| {
        s.write_tokens(&t.batch(), None).unwrap();
        let r = s.run().unwrap();
        let i = r.info();
        let rc = unsafe {
            turbo_result_read(r.0, dst.as_mut_ptr() as *mut _, (dst.len() * 4) as u64, ptr::null_mut(), null_err())
        };
        assert_eq!(rc, 0);
        (i.host_allocs, i.device_allocs)
    };
    // Cold: the first runs of each shape.
    run(&small);
    run(&long);
    let free = || {
        assert_eq!(unsafe { cudaSetDevice(rt_ordinal(&l) as i32) }, 0);
        let (mut f, mut t) = (0usize, 0usize);
        assert_eq!(unsafe { cudaMemGetInfo(&mut f, &mut t) }, 0);
        f
    };
    let (before, free_before) = (turbo::cuda::allocations(), free());
    for (i, t) in [&small, &long, &small, &long].into_iter().enumerate() {
        assert_eq!(run(t), (0, 0), "warm run {i}: the result's count");
    }
    assert_eq!(turbo::cuda::allocations(), before, "the backend allocated nothing in warm runs");
    // The count above is exact for the backend's own allocations. This
    // catches what it cannot count, cuBLAS or the driver growing a pool:
    // the driver reserves device memory in pages of 2 MiB, so free memory
    // moves in those steps when it moves, and another process's use of
    // the device moves it by less.
    let (after, page) = (free(), 2usize << 20);
    println!("free device memory before the warm runs {free_before}, after {after}");
    assert!(after + page > free_before, "the device's free memory fell by {} bytes", free_before - after);
}

/// A model stored in F16 or BF16: MODEL is refused, EXACT and FASTEST
/// share one F32 copy on the device, and give the vectors of the same
/// values stored as F32.
#[test]
fn half_weights_compute_in_f32_from_one_shared_copy() {
    let _t = turn();
    let Some(_) = cuda_device("half_weights_compute_in_f32_from_one_shared_copy") else { return };
    for (dtype, narrow) in [("F16", to_f16 as fn(f32) -> u16), ("BF16", to_bf16)] {
        let (n, w): (Vec<Tensor>, Vec<Tensor>) = tiny_weights(0)
            .into_iter()
            .map(|t| {
                let halves: Vec<u16> =
                    t.data.chunks(4).map(|c| narrow(f32::from_le_bytes(c.try_into().unwrap()))).collect();
                let wide = |h: u16| if dtype == "F16" { from_f16(h) } else { f32::from_bits((h as u32) << 16) };
                let nt = Tensor {
                    name: t.name.clone(),
                    dtype,
                    shape: t.shape.clone(),
                    data: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
                };
                let wt = Tensor { data: halves.iter().flat_map(|&h| wide(h).to_le_bytes()).collect(), ..t };
                (nt, wt)
            })
            .unzip();
        let mut f = Fixture::model(&format!("cuda-half-{dtype}"));
        f.weights("weights/model.safetensors", &n);
        let l = f.load_on(cuda).unwrap();
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");
        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copy, "one copy, shared");
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F32);

        let mut g = Fixture::model(&format!("cuda-half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &w);
        let lw = g.load_on(cuda).unwrap();
        let want = Session::create(lw.m, None).unwrap().embed(&TEXTS, None).unwrap();
        assert_eq!(a.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert_eq!(b.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert!(unsafe { model_converted_weights(lw.m) }.is_none(), "F32 weights are used as loaded");

        // The same F16 or BF16 bundle on the CPU, which widens it too.
        let lc = f.load().unwrap();
        let cpu = Session::create(lc.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        for (r, (g, c)) in want.iter().zip(cpu.embed(&TEXTS, None).unwrap()).enumerate() {
            let cos = cosine(g, &c);
            assert!(cos >= 0.9999, "{dtype} row {r}: cosine {cos} with the cpu");
        }
    }
}

fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if x == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "{x} is not a normal half");
    let man = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn from_f16(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    if exp == 0 && man == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Sessions on one context, run from several threads at once, each get
/// their own vectors: the context's stream and cuBLAS handle are shared
/// under its lock.
#[test]
fn sessions_on_one_context_run_from_many_threads() {
    let _t = turn();
    let Some(_) = cuda_device("sessions_on_one_context_run_from_many_threads") else { return };
    let l = on_cuda(&tiny_bundle());
    let want: Vec<Vec<Vec<f32>>> = {
        let s = Session::create(l.m, None).unwrap();
        TEXTS.iter().map(|t| s.embed(&[t], None).unwrap()).collect()
    };
    let m = l.m as usize;
    std::thread::scope(|scope| {
        for (i, text) in TEXTS.iter().enumerate() {
            let want = &want[i];
            scope.spawn(move || {
                let s = Session::create(m as *mut turbo_model, None).unwrap();
                for _ in 0..20 {
                    assert_eq!(&s.embed(&[text], None).unwrap(), want, "text {i}");
                }
            });
        }
    });
}

/// An output_dim the bundle lists is cut, then normalized, on the device.
#[test]
fn an_output_dim_is_cut_then_normalized_on_the_device() {
    let _t = turn();
    let Some(_) = cuda_device("an_output_dim_is_cut_then_normalized_on_the_device") else { return };
    let mut f = Fixture::new("cuda-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(cuda).unwrap();
    let s = Session::create(l.m, None).unwrap();
    let got = s.embed(&TEXTS, Some(&opts(|o| o.output_dim = 4))).unwrap();
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for (t, g) in TEXTS.iter().zip(&got) {
        assert_eq!(g.len(), 4);
        let ids = tok.row(t, None).unwrap();
        let want = plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, 4, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}

// ---- Limits and the largest shape ----------------------------------------------------

/// A session over 65535 rows is refused by field 1: a grid dimension of
/// the kernels that run one block per row.
#[test]
fn more_rows_than_a_launch_takes_are_refused_by_field() {
    let _t = turn();
    let Some(_) = cuda_device("more_rows_than_a_launch_takes_are_refused_by_field") else { return };
    let mut m = model_manifest();
    m["embed"]["max_batch"] = json!(70000);
    let mut f = Fixture::new("cuda-rows", m);
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(cuda).unwrap();
    let e = Session::create(l.m, Some(&session_desc(65536, 8, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 1), "{e:?}");
    assert!(e.message.contains("at most 65535 rows"), "{e:?}");
    Session::create(l.m, Some(&session_desc(65535, 8, 0))).unwrap();
}

/// A session longer than attention's shared memory holds on this device
/// is refused by field 2. 65536 positions need 256 KiB of scores per
/// block, more than any device gives one.
#[test]
fn more_tokens_than_attention_holds_are_refused_by_field() {
    let _t = turn();
    let Some(_) = cuda_device("more_tokens_than_attention_holds_are_refused_by_field") else { return };
    let positions = 65536u64;
    let mut m = model_manifest();
    m["architecture"]["max_positions"] = json!(positions);
    m["embed"]["max_seq"] = json!(positions);
    // A case longer than max_seq, which a manifest needs: 600 paragraphs of
    // about 120 tokens.
    m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    let mut f = Fixture::new("cuda-positions", m);
    let weights: Vec<Tensor> = tiny_weights(0)
        .into_iter()
        .map(|t| {
            if t.name != "embeddings.position_embeddings.weight" {
                return t;
            }
            let n = positions as usize * 8;
            let data = (0..n).flat_map(|i| (((i * 37 % 101) as f32 - 50.0) / 500.0).to_le_bytes()).collect();
            Tensor { shape: vec![positions, 8], data, ..t }
        })
        .collect();
    f.weights("weights/model.safetensors", &weights);
    let l = f.load_on(cuda).unwrap();
    let e = Session::create(l.m, Some(&session_desc(1, positions as u32, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    assert!(e.message.contains("shared memory"), "{e:?}");
    Session::create(l.m, Some(&session_desc(1, 512, 0))).unwrap();
}

/// TURBO_TEST_BUNDLE, a relative path read from the workspace root, as
/// tests/conformance.rs reads it.
fn named_bundle() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE")?);
    Some(if p.is_absolute() { p } else { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(p) })
}

/// One run at the session's largest shape, max_batch rows of max_seq
/// tokens, on the device and on the CPU: every row's cosine reaches
/// 0.9999. Rows are full or end early, so the padding is exercised too.
fn largest_shape_matches_the_cpu(dir: &std::path::Path) {
    let (g, c) = (on_cuda(dir), Loaded::load(dir).unwrap());
    let (gs, cs) = (Session::create(g.m, None).unwrap(), Session::create(c.m, None).unwrap());
    let si = gs.info();
    let (batch, seq) = (si.max_batch as usize, si.max_seq as usize);
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let rows: Vec<Vec<i32>> = (0..batch)
        .map(|r| {
            let len = seq - (r * 7) % (seq / 2);
            (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect()
        })
        .collect();
    let t = Tokens::new(&rows, 0);
    assert_eq!((t.batch as usize, t.seq as usize), (batch, seq));
    gs.write_tokens(&t.batch(), None).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let cos = cosine(a, b);
        lowest = lowest.min(cos);
        assert!(cos >= 0.9999, "row {r}: cosine {cos} with the cpu");
    }
    println!("{}: {batch} rows of {seq} tokens, 1 - lowest cosine with the cpu {:.3e}", dir.display(), 1.0 - lowest);
}

#[test]
fn the_largest_shape_matches_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("the_largest_shape_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_largest_shape_of_a_real_bundle_matches_the_cpu() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = cuda_device("the_largest_shape_of_a_real_bundle_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&dir);
}
