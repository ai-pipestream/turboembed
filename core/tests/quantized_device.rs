//! int8 and ubinary results on the device TURBO_TEST_DEVICE names (cuda,
//! levelzero or metal): pooled rows and ranges, at the model's dim and a
//! declared narrower width, at EXACT and FASTEST. The codes are those of
//! the same device's f32 result for the same write; the result names the
//! device's backend, sits in host memory, and counts what the core added
//! to fetch a device-placed result: its f32 bytes in d2h_bytes, and the
//! staging memory in the first such run's host_allocs. cuda and levelzero
//! return f32 vectors in device memory, which the test asserts, so their
//! runs take the staging path; metal's are in shared memory and do not.
//!
//! Without TURBO_TEST_DEVICE, or when the backend lists no device, a test
//! says it was skipped and passes. With TURBO_TEST_REQUIRE_<BACKEND>=1
//! (CUDA, LEVELZERO or METAL) it fails instead, so a device run cannot
//! pass by finding no device.

mod common;

use std::ptr;

use common::*;
use serde_json::json;
use turbo::quantize::{self, Quantization};
use turbo::*;

const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a b c d"];
const BACKENDS: [(&str, &str); 3] = [("cuda", "CUDA"), ("levelzero", "LEVELZERO"), ("metal", "METAL")];

fn required(var: &str) -> bool {
    std::env::var(format!("TURBO_TEST_REQUIRE_{var}")).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// The backend TURBO_TEST_DEVICE names, if it lists a device; None after
/// saying the test is skipped.
fn backend(test: &str) -> Option<&'static str> {
    let Ok(want) = std::env::var("TURBO_TEST_DEVICE") else {
        for (_, var) in BACKENDS {
            assert!(!required(var), "{test}: TURBO_TEST_REQUIRE_{var}=1 and TURBO_TEST_DEVICE is not set");
        }
        println!("{test}: skipped: TURBO_TEST_DEVICE is not set");
        return None;
    };
    let Some(&(name, var)) = BACKENDS.iter().find(|(n, _)| *n == want) else {
        println!("{test}: skipped: TURBO_TEST_DEVICE={want} is not cuda, levelzero or metal");
        return None;
    };
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, name);
    unsafe { turbo_runtime_release(rt) };
    if d.is_none() {
        assert!(!required(var), "{test}: TURBO_TEST_REQUIRE_{var}=1 and the {name} backend lists no device");
        println!("{test}: skipped: the {name} backend lists no device");
        return None;
    }
    Some(name)
}

fn with(output_dim: u32, quantization: u32, ranges: Option<&[f32]>) -> turbo_embed_options {
    opts(|o| {
        o.normalize = TURBO_NORMALIZE_L2;
        o.pooling = TURBO_POOLING_MEAN;
        o.output_dim = output_dim;
        o.quantization = quantization;
        o.int8_ranges = ranges.map_or(ptr::null(), <[f32]>::as_ptr);
    })
}

/// A result's info, taken before any read, and its bytes.
fn take(r: Outcome) -> (turbo_result_info, Vec<u8>) {
    let info = r.info();
    let mut out = vec![0u8; info.bytes as usize];
    let mut err = new_error();
    let rc = unsafe { turbo_result_read(r.0, out.as_mut_ptr().cast(), info.bytes, ptr::null_mut(), &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    (info, out)
}

/// One write's three results on the device, held to each other.
fn check(backend: &str, what: &str, mut write: impl FnMut(&turbo_embed_options) -> Outcome, w: u32, staged: &mut bool) {
    let (fi, fbytes) = take(write(&with(w, TURBO_QUANTIZE_NONE, None)));
    assert_eq!(field(&fi.backend), backend, "{what}: the run's backend");
    // cuda and levelzero leave the vectors in device memory, so their
    // quantized runs take the staging path; metal's are shared memory.
    let placed = if backend == "metal" { TURBO_PLACE_SHARED } else { TURBO_PLACE_DEVICE };
    assert_eq!(fi.placement, placed, "{what}: the f32 result's placement");
    let flat: Vec<f32> = fbytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let (batch, dim) = (fi.batch as usize, w as usize);
    assert_eq!(flat.len(), batch * dim, "{what}");
    let rows: Vec<&[f32]> = flat.chunks(dim).collect();
    let min = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::INFINITY, f32::min));
    let max = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::NEG_INFINITY, f32::max));
    let ranges: Vec<f32> = min.chain(max).collect();
    for (q, code, dtype) in [
        (Quantization::Int8, TURBO_QUANTIZE_INT8, TURBO_DTYPE_I8),
        (Quantization::UBinary, TURBO_QUANTIZE_UBINARY, TURBO_DTYPE_U8),
    ] {
        let r = (q == Quantization::Int8).then_some(&ranges[..]);
        let (qi, got) = take(write(&with(w, code, r)));
        let mut want = vec![0u8; batch * q.row_bytes(dim)];
        quantize::rows(q, &flat, dim, &ranges, &mut want);
        let what = format!("{what} {q:?}");
        assert_eq!(field(&qi.backend), backend, "{what}: the run's backend");
        assert_eq!((qi.dtype, qi.dim, qi.batch, qi.placement), (dtype, w, fi.batch, TURBO_PLACE_HOST), "{what}");
        assert_eq!(got, want, "{what}: the codes of the same device's f32 vectors");
        // What the core added to the backend's own counts.
        let device = fi.placement == TURBO_PLACE_DEVICE;
        let fetched = if device { batch as u64 * dim as u64 * 4 } else { 0 };
        assert_eq!(qi.d2h_bytes, fi.d2h_bytes + fetched, "{what}: d2h_bytes, f32 placement {}", fi.placement);
        let made = u64::from(device && !*staged);
        assert_eq!(qi.host_allocs, fi.host_allocs + made, "{what}: host_allocs");
        *staged |= device;
        println!("{what}: f32 placement {} d2h {} -> {}", fi.placement, fi.d2h_bytes, qi.d2h_bytes);
    }
}

#[test]
fn device_codes_are_the_codes_of_the_same_devices_f32_vectors() {
    let test = "device_codes_are_the_codes_of_the_same_devices_f32_vectors";
    let Some(backend) = backend(test) else { return };
    let mut f = Fixture::model("quantized-device");
    f.manifest["embed"]["output_dims"] = json!([4]);
    let l = f.load_on(|rt| first_of(rt, backend).expect("a device")).unwrap_or_else(|e| panic!("{e:?}"));
    let tok = Tok::create(&f.dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let t = Tokens::new(&rows, 0);
    let ranges: Vec<_> = rows.iter().enumerate().map(|(r, v)| range(r as u32, 1, v.len() as u32 - 1)).collect();
    for precision in [TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        let mut d = session_desc(0, 0, precision);
        d.max_ranges = 4;
        let s = Session::create(l.m, Some(&d)).unwrap_or_else(|e| panic!("{backend} precision {precision}: {e:?}"));
        let mut staged = false;
        for w in [8, 4] {
            let what = format!("{backend} precision {precision} width {w}");
            let pooled = |o: &turbo_embed_options| {
                s.write_tokens(&t.batch(), Some(o)).unwrap();
                s.run().unwrap()
            };
            check(backend, &format!("{what} pooled"), pooled, w, &mut staged);
            let ranged = |o: &turbo_embed_options| {
                s.write_ranges(&t.batch(), &ranges, Some(o)).unwrap();
                s.run().unwrap()
            };
            check(backend, &format!("{what} ranges"), ranged, w, &mut staged);
        }
    }
}
