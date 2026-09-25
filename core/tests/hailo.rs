//! The Hailo backend through the C interface: its table, the devices it
//! lists against what HailoRT scans, its capability, contexts and host
//! buffers, and, on a bundle with a HEF for the device, what a session
//! computes in and what a run reports. tests/conformance.rs holds the
//! vectors to the bundle's reference.
//!
//! Built with the `hailo` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run on
//! anything else in its place. With TURBO_TEST_REQUIRE_HAILO=1 it fails
//! instead, so a run on a Hailo machine cannot pass by finding no device.
//! The tests that need a bundle with a HEF are ignored unless asked for,
//! and read it from TURBO_TEST_BUNDLE. docs/hailo.md says how to run them.

#![cfg(feature = "hailo")]

mod common;

use std::ffi::{c_char, c_int};
use std::ptr;

use common::*;
use turbo::status::{BUNDLE_NO_ARTIFACT, UNSUPPORTED, UNSUPPORTED_OPTION};
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
fn the_table_is_whole_and_loads_hef_alone() {
    let b = turbo::hailo::backend();
    assert_eq!(b.name(), "hailo");
    assert_eq!(b.struct_size as usize, size_of::<turbo::backend::turbo_backend>());
    turbo::backend::check_table(b).unwrap();
    assert_eq!(b.formats, 1 << (backend::TURBO_FORMAT_HEF - 1), "FORMAT_HEF and nothing else");
    assert!(b.model_load.is_some() && b.session_run.is_some() && b.buffer_alloc.is_some());
    assert!(b.buffer_read.is_none(), "every buffer it gives has a host address");
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
        assert_eq!(name, name.trim(), "no padding around the name");
        let runtime = field(&d.runtime_version);
        assert!(runtime.split('.').count() == 3 && runtime.split('.').all(|p| p.parse::<u32>().is_ok()), "{runtime}");
        let driver = field(&d.driver_version);
        assert!(driver.contains("firmware "), "{driver}");
        println!("hailo device {o} ({bdf}): {name}, arch {arch}, runtime {runtime}, driver {driver}");
    }
}

// ---- Capability ------------------------------------------------------------------

#[test]
fn embed_is_offered_in_i8_and_exact_is_refused() {
    let rt = Rt::new();
    let Some(d) = hailo_device(&rt, "embed_is_offered_in_i8_and_exact_is_refused") else { return };
    let cap = |p| {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(rt.0, d, TURBO_TASK_EMBED, p, &mut cap, ptr::null_mut()) }, 0);
        cap
    };
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let c = cap(p);
        assert_eq!(c.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&c.reason));
        assert_eq!(c.dtype, TURBO_DTYPE_I8);
        assert_eq!(c.options_honored, 0b111111, "every field of turbo_embed_options");
    }
    let c = cap(TURBO_PRECISION_EXACT);
    assert_eq!((c.status, c.dtype, c.options_honored), (backend::TURBO_CAP_UNSUPPORTED, 0, 0));
    assert!(field(&c.reason).contains("EXACT"), "{}", field(&c.reason));
}

// ---- Contexts and buffers --------------------------------------------------------

struct Ctx(*mut turbo_context);

impl Ctx {
    fn create(rt: &Rt, d: u32) -> Ctx {
        let mut c = ptr::null_mut();
        let mut err = new_error();
        let rc = unsafe { turbo_context_create(rt.0, d, &mut c, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        Ctx(c)
    }

    fn alloc(&self, placement: u32, bytes: u64) -> Result<*mut turbo_buffer, Failure> {
        let desc = turbo_buffer_desc {
            struct_size: size_of::<turbo_buffer_desc>() as u32,
            placement,
            dtype: TURBO_DTYPE_F32,
            ndim: 1,
            shape: [bytes / 4, 0],
            bytes: 0,
        };
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &desc, &mut b, &mut err) } {
            0 => Ok(b),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { turbo_context_release(self.0) };
    }
}

/// Two contexts on one device share its vdevice; a buffer is host memory
/// and exports as its host address; any other placement is refused.
#[test]
fn contexts_share_the_device_and_buffers_are_host_memory() {
    let rt = Rt::new();
    let Some(d) = hailo_device(&rt, "contexts_share_the_device_and_buffers_are_host_memory") else { return };
    let (a, b) = (Ctx::create(&rt, d), Ctx::create(&rt, d));
    let buf = a.alloc(TURBO_PLACE_HOST, 4096).unwrap();
    let mut host = ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf, &mut host, ptr::null_mut()) }, 0);
    assert_eq!(host as usize % 64, 0, "aligned to a cache line");
    unsafe { std::ptr::write_bytes(host as *mut u8, 0xab, 4096) };
    let mut h: turbo_native_handle = unsafe { std::mem::zeroed() };
    h.struct_size = size_of::<turbo_native_handle>() as u32;
    assert_eq!(unsafe { turbo_buffer_export(buf, TURBO_HANDLE_HOST_PTR, &mut h, ptr::null_mut()) }, 0);
    assert_eq!((h.kind, h.handle as usize), (TURBO_HANDLE_HOST_PTR, host as usize));
    let mut err = new_error();
    let rc = unsafe { turbo_buffer_export(buf, TURBO_HANDLE_DMABUF_FD, &mut h, &mut err) };
    assert_eq!(failure(rc, &err).code, UNSUPPORTED);
    unsafe { turbo_buffer_release(buf) };
    for p in [TURBO_PLACE_PINNED, TURBO_PLACE_DEVICE, TURBO_PLACE_SHARED] {
        let f = b.alloc(p, 4096).unwrap_err();
        assert!(f.is(UNSUPPORTED, "TURBO_PLACE_HOST only"), "placement {p}: {f:?}");
    }
}

/// A bundle of raw weights has nothing the hailo backend loads.
#[test]
fn raw_weights_are_not_an_artifact_for_hailo() {
    let rt = Rt::new();
    let Some(_) = hailo_device(&rt, "raw_weights_are_not_an_artifact_for_hailo") else { return };
    drop(rt);
    let f = Loaded::load_on(&tiny_bundle(), |rt| first_of(rt, "hailo").unwrap()).err().expect("no artifact");
    assert!(f.is(BUNDLE_NO_ARTIFACT, "hailo"), "{f:?}");
}

// ---- A bundle with a HEF ---------------------------------------------------------

/// TURBO_TEST_BUNDLE: a bundle with a HEF for the first hailo device.
fn hef_bundle() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"))
}

fn on_hailo() -> Loaded {
    Loaded::load_on(&hef_bundle(), |rt| first_of(rt, "hailo").expect("a hailo device"))
        .unwrap_or_else(|e| panic!("{e:?}"))
}

#[test]
#[ignore = "needs a Hailo device and TURBO_TEST_BUNDLE with a HEF for it"]
fn a_session_computes_in_i8_and_exact_is_refused() {
    let l = on_hailo();
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let s = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap();
        assert_eq!(s.info().compute_dtype, TURBO_DTYPE_I8, "precision {p}");
    }
    let f = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).err().expect("refused");
    assert!(f.is(UNSUPPORTED_OPTION, "EXACT") && f.field == 3, "{f:?}");
}

/// Each row is one frame: its word rows and bias go to the device and its
/// hidden states come back. The lookup, pooling and normalize run on the
/// host, the encoder on the device, and the vectors stay on the host.
#[test]
#[ignore = "needs a Hailo device and TURBO_TEST_BUNDLE with a HEF for it"]
fn a_run_reports_its_frames_and_where_each_stage_ran() {
    let l = on_hailo();
    let s = Session::create(l.m, Some(&session_desc(4, 0, TURBO_PRECISION_MODEL))).unwrap();
    assert_eq!(s.info().max_batch, 4, "the HEF's frame of one row does not cap the session");
    let batch = 4;
    let seq = s.info().max_seq as u64;
    let hidden = l.info().dim as u64;
    let texts = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
    let texts = &texts[..batch.min(texts.len())];
    let n = texts.len() as u64;
    s.write_text(texts, None).unwrap();
    let r = s.run().unwrap();
    let info = r.info();
    assert_eq!((info.batch as u64, info.dim, info.compute_dtype), (n, hidden as u32, TURBO_DTYPE_I8));
    assert_eq!(info.placement, TURBO_PLACE_HOST);
    assert_eq!(field(&info.backend), "hailo");
    // Per row: the rows [seq, hidden] and the bias [seq, heads * seq], each
    // at least a byte an element, and the hidden states [seq, hidden] back.
    assert!(info.h2d_bytes >= n * seq * hidden && info.h2d_bytes.is_multiple_of(n), "{}", info.h2d_bytes);
    assert!(info.d2h_bytes >= n * seq * hidden, "{}", info.d2h_bytes);
    assert_eq!((info.host_allocs, info.device_allocs), (0, 0));
    let st = &info.stage;
    assert_eq!(st[TURBO_EMBED_STAGE_TOKENIZE], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_UPLOAD], TURBO_STAGE_DEVICE);
    assert_eq!(st[TURBO_EMBED_STAGE_LOOKUP], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_ENCODE], TURBO_STAGE_DEVICE);
    assert_eq!(st[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_NORMALIZE], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_DOWNLOAD], TURBO_STAGE_DEVICE);
    for v in r.rows() {
        let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "unit length: {n}");
    }
}

/// The HEF computes token type 0 only: rows with another type are refused,
/// naming the row, and nothing runs.
#[test]
#[ignore = "needs a Hailo device and TURBO_TEST_BUNDLE with a HEF for it"]
fn a_token_type_other_than_0_is_refused() {
    let l = on_hailo();
    let s = Session::create(l.m, Some(&session_desc(2, 0, TURBO_PRECISION_MODEL))).unwrap();
    let mut t = Tokens::new(&[vec![101, 7592, 102], vec![101, 2088, 102]], 0);
    let mut types = vec![0; t.ids.len()];
    types[4] = 1;
    t.types = Some(types);
    let f = s.write_tokens(&t.batch(), None).unwrap_err();
    assert!(f.is(UNSUPPORTED_OPTION, "token type 1 in row 1"), "{f:?}");
    t.types = Some(vec![0; t.ids.len()]);
    s.write_tokens(&t.batch(), None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 2);
}

/// Pooling and normalize are the backend's, on the host: CLS differs from
/// mean, and NONE leaves the vector at its own length.
#[test]
#[ignore = "needs a Hailo device and TURBO_TEST_BUNDLE with a HEF for it"]
fn pooling_and_normalize_follow_the_options() {
    let l = on_hailo();
    let s = Session::create(l.m, Some(&session_desc(1, 0, TURBO_PRECISION_MODEL))).unwrap();
    let text = ["Embedding models turn text into vectors."];
    let mut o = embed_options();
    o.pooling = TURBO_POOLING_MEAN;
    let mean = s.embed(&text, Some(&o)).unwrap().remove(0);
    o.pooling = TURBO_POOLING_CLS;
    let cls = s.embed(&text, Some(&o)).unwrap().remove(0);
    let cos: f64 = mean.iter().zip(&cls).map(|(a, b)| *a as f64 * *b as f64).sum();
    assert!(cos < 0.999, "CLS and mean pooling give different vectors: cosine {cos}");
    o.pooling = TURBO_POOLING_MEAN;
    o.normalize = TURBO_NORMALIZE_NONE;
    let raw = s.embed(&text, Some(&o)).unwrap().remove(0);
    let n = raw.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    assert!((n - 1.0).abs() > 1e-3, "NONE leaves the length as pooled: {n}");
    let back: f64 = raw.iter().zip(&mean).map(|(a, b)| *a as f64 / n * *b as f64).sum();
    assert!(back > 0.99999, "the same direction as the normalized vector: {back}");
}
