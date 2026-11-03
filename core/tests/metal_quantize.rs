//! Metal int8 and ubinary kernels against `core/src/quantize.rs`.
//!
//! Built with the `metal` feature only. Needs TURBO_TEST_DEVICE=metal;
//! with TURBO_TEST_REQUIRE_METAL=1 it fails when none is listed. Device
//! receipts:
//!   TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 \
//!     cargo test -p turbo --features metal metal_quantize -- --nocapture

#![cfg(feature = "metal")]

mod common;

use common::*;
use turbo::quantize::{self, Quantization};
use turbo::*;

fn cases() -> Vec<(usize, Vec<f32>, Vec<f32>)> {
    vec![
        (1, vec![0.0], vec![0.0, 0.0]),
        (1, vec![f32::NAN], vec![0.0, 1.0]),
        (1, vec![f32::INFINITY], vec![0.0, 1.0]),
        (1, vec![f32::NEG_INFINITY], vec![0.0, 1.0]),
        (4, vec![1.0, -1.0, 0.0, -0.0], vec![-2.0, -1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0]),
        (
            9,
            {
                let mut v = vec![0.0f32; 9];
                v[0] = 1.0;
                v[7] = -3.0;
                v[8] = 2.5;
                v
            },
            {
                let mut r = vec![0.0f32; 18];
                for d in 0..9 {
                    r[d] = -4.0;
                    r[9 + d] = 4.0;
                }
                r
            },
        ),
        (16, (0..16).map(|i| (i as f32) - 8.0).collect(), {
            let mut r = vec![0.0f32; 32];
            for d in 0..16 {
                r[d] = -8.0;
                r[16 + d] = 8.0;
            }
            r
        }),
    ]
}

fn host_codes(q: Quantization, x: &[f32], dim: usize, ranges: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; q.row_bytes(dim)];
    quantize::rows(q, x, dim, ranges, &mut out);
    out
}

fn require_or_skip(test: &str, backend: &str, var: &str) -> bool {
    let selected = std::env::var("TURBO_TEST_DEVICE").ok();
    let required = std::env::var(format!("TURBO_TEST_REQUIRE_{var}")).is_ok_and(|v| !v.is_empty() && v != "0");
    assert!(
        !required || selected.as_deref() == Some(backend),
        "{test}: TURBO_TEST_REQUIRE_{var}=1 requires TURBO_TEST_DEVICE={backend}, got {selected:?}"
    );
    if selected.as_deref() != Some(backend) {
        println!("{test}: skipped: TURBO_TEST_DEVICE is not {backend}");
        return false;
    }
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(std::ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, backend);
    unsafe { turbo_runtime_release(rt) };
    if d.is_none() {
        assert!(!required, "{test}: TURBO_TEST_REQUIRE_{var}=1 and the {backend} backend lists no device");
        println!("{test}: skipped: the {backend} backend lists no device");
        return false;
    }
    true
}

#[test]
fn metal_codes_match_host_byte_for_byte() {
    let test = "metal_codes_match_host_byte_for_byte";
    if !require_or_skip(test, "metal", "METAL") {
        return;
    }
    for (dim, x, ranges) in cases() {
        for (q, code) in [(Quantization::Int8, TURBO_QUANTIZE_INT8), (Quantization::UBinary, TURBO_QUANTIZE_UBINARY)] {
            let want = host_codes(q, &x, dim, &ranges);
            let mut got = vec![0u8; want.len()];
            turbo::metal::quantize(0, code, &x, &ranges, 1, dim as i32, &mut got)
                .unwrap_or_else(|e| panic!("metal quantize {q:?} dim {dim}: status {e}"));
            assert_eq!(got, want, "metal {q:?} dim {dim} x={x:?}");
        }
    }
}

#[test]
fn metal_codes_of_several_rows_stay_inside_their_slice() {
    let test = "metal_codes_of_several_rows_stay_inside_their_slice";
    if !require_or_skip(test, "metal", "METAL") {
        return;
    }
    // Each row is padded to whole bytes on its own, so 3 rows of 9 values
    // take 6 ubinary bytes; a byte past them must keep its canary.
    let rows = 3;
    for dim in [8usize, 9, 16, 17] {
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
        let ranges: Vec<f32> = (0..dim).map(|_| -5.0).chain((0..dim).map(|_| 5.0)).collect();
        for (q, code) in [(Quantization::Int8, TURBO_QUANTIZE_INT8), (Quantization::UBinary, TURBO_QUANTIZE_UBINARY)] {
            let want_len = rows * q.row_bytes(dim);
            let mut want = vec![0u8; want_len];
            quantize::rows(q, &x, dim, &ranges, &mut want);
            let mut got = vec![0xa5u8; want_len + 1];
            turbo::metal::quantize(0, code, &x, &ranges, rows as i32, dim as i32, &mut got[..want_len])
                .unwrap_or_else(|e| panic!("metal quantize {q:?} dim {dim}: status {e}"));
            assert_eq!(&got[..want_len], &want[..], "metal {q:?} {rows} rows of dim {dim}");
            assert_eq!(got[want_len], 0xa5, "metal {q:?} dim {dim} wrote past its codes");
        }
    }
}

#[test]
fn metal_quantize_refuses_slices_under_the_shape() {
    let test = "metal_quantize_refuses_slices_under_the_shape";
    if !require_or_skip(test, "metal", "METAL") {
        return;
    }
    let x = vec![1.0f32; 8];
    let ranges = vec![0.0f32; 8];
    let mut out = vec![0xa5u8; 8];
    let short_out = turbo::metal::quantize(0, TURBO_QUANTIZE_INT8, &x, &ranges, 2, 4, &mut out[..7]);
    let short_x = turbo::metal::quantize(0, TURBO_QUANTIZE_INT8, &x[..7], &ranges, 2, 4, &mut out);
    let short_ranges = turbo::metal::quantize(0, TURBO_QUANTIZE_INT8, &x, &ranges[..7], 2, 4, &mut out);
    let ubinary_rows = turbo::metal::quantize(0, TURBO_QUANTIZE_UBINARY, &x, &[], 2, 4, &mut out[..1]);
    for (what, rc) in [("out", short_out), ("x", short_x), ("ranges", short_ranges), ("ubinary out", ubinary_rows)] {
        assert_eq!(rc, Err(turbo::status::INVALID_ARGUMENT), "metal: short {what}");
    }
    assert_eq!(out, vec![0xa5u8; 8], "metal: a refused call writes nothing");
}
