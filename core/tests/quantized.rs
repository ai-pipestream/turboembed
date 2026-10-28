//! int8 and ubinary results (turbo_embed_options fields 7 and 8) through
//! the C ABI: for pooled rows written as text and as tokens, and for
//! ranges, at the full width and at a declared Matryoshka width, a
//! quantized run returns exactly the codes of the f32 run's vectors, in the
//! layout src/quantize.rs pins to sentence-transformers; and every option
//! that cannot be honored is refused by its field.

mod common;

use common::*;
use serde_json::json;
use turbo::quantize::{self, Quantization};
use turbo::status::*;
use turbo::*;

const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
const CALIBRATION: [&str; 4] = ["a calibration sentence", "another one, longer than the first", "x", "the end"];

/// The small BERT (dim 8) declaring width 4.
fn model(name: &str) -> (Fixture, Loaded) {
    let mut f = Fixture::model(name);
    f.manifest["embed"]["output_dims"] = json!([4]);
    let l = f.load().unwrap_or_else(|e| panic!("{e:?}"));
    (f, l)
}

fn with(output_dim: u32, quantization: u32, ranges: Option<&[f32]>) -> turbo_embed_options {
    opts(|o| {
        o.normalize = TURBO_NORMALIZE_L2;
        o.output_dim = output_dim;
        o.quantization = quantization;
        o.int8_ranges = ranges.map_or(std::ptr::null(), <[f32]>::as_ptr);
    })
}

/// The result's bytes through turbo_result_read, and through
/// turbo_result_buffer's host memory, which must agree.
fn bytes_of(r: &Outcome) -> Vec<u8> {
    let info = r.info();
    let mut out = vec![0xAAu8; info.bytes as usize];
    let (mut written, mut err) = (0u64, new_error());
    let rc = unsafe { turbo_result_read(r.0, out.as_mut_ptr().cast(), info.bytes, &mut written, &mut err) };
    assert_eq!((rc, written), (0, info.bytes), "{:?}", failure(rc, &err));
    let mut buf = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, &mut err) }, 0, "{:?}", failure(1, &err));
    let mut desc: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    desc.struct_size = size_of::<turbo_buffer_desc>() as u32;
    assert_eq!(unsafe { turbo_buffer_get_desc(buf, &mut desc, &mut err) }, 0);
    let mut host = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf, &mut host, &mut err) }, 0);
    let mapped = unsafe { std::slice::from_raw_parts(host as *const u8, desc.bytes as usize) }.to_vec();
    unsafe { turbo_buffer_release(buf) };
    let width = if info.dtype == TURBO_DTYPE_U8 { info.dim.div_ceil(8) } else { info.dim };
    assert_eq!(
        (desc.placement, desc.dtype, desc.shape, desc.bytes),
        (TURBO_PLACE_HOST, info.dtype, [info.batch as u64, width as u64], info.bytes),
        "the buffer is the result's codes, a row's bytes wide"
    );
    assert_eq!(mapped, out, "read and mapped bytes agree");
    out
}

/// Per-dimension minimums then maximums of `rows`, as sentence-transformers'
/// `_get_quantization_ranges` takes them from calibration embeddings.
fn ranges_of(rows: &[Vec<f32>]) -> Vec<f32> {
    let dim = rows[0].len();
    let min = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::INFINITY, f32::min));
    let max = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::NEG_INFINITY, f32::max));
    min.chain(max).collect()
}

/// Every quantization of one write, held to the codes of its f32 result.
fn check(write: impl Fn(&turbo_embed_options) -> Outcome, s: &Session, output_dim: u32) {
    let f32_rows = write(&with(output_dim, TURBO_QUANTIZE_NONE, None)).rows();
    let dim = f32_rows[0].len();
    assert_eq!(dim, output_dim as usize);
    let calib = s.embed(&CALIBRATION, Some(&with(output_dim, 0, None))).unwrap();
    let ranges = ranges_of(&calib);
    let flat: Vec<f32> = f32_rows.concat();
    for (q, code, dtype, r) in [
        (Quantization::Int8, TURBO_QUANTIZE_INT8, TURBO_DTYPE_I8, Some(&ranges[..])),
        (Quantization::UBinary, TURBO_QUANTIZE_UBINARY, TURBO_DTYPE_U8, None),
    ] {
        let mut want = vec![0u8; f32_rows.len() * q.row_bytes(dim)];
        quantize::rows(q, &flat, dim, r.unwrap_or(&[]), &mut want);
        let res = write(&with(output_dim, code, r));
        let info = res.info();
        assert_eq!(
            (info.dtype, info.dim, info.batch, info.bytes, info.placement),
            (dtype, output_dim, f32_rows.len() as u32, want.len() as u64, TURBO_PLACE_HOST),
            "{q:?} at width {output_dim}"
        );
        assert_eq!(bytes_of(&res), want, "{q:?} at width {output_dim}");
    }
    // The bits are the f32 signs, top bit first.
    let res = write(&with(output_dim, TURBO_QUANTIZE_UBINARY, None));
    let bits = bytes_of(&res);
    let row_bytes = dim.div_ceil(8);
    for (r, row) in f32_rows.iter().enumerate() {
        for (d, &v) in row.iter().enumerate() {
            let bit = bits[r * row_bytes + d / 8] >> (7 - d % 8) & 1;
            assert_eq!(bit == 1, v > 0.0, "row {r} dimension {d}: {v}");
        }
    }
    drop(res);
    // A later f32 write is f32 again.
    let info = write(&with(output_dim, 0, None)).info();
    assert_eq!((info.dtype, info.bytes), (TURBO_DTYPE_F32, f32_rows.len() as u64 * dim as u64 * 4));
}

#[test]
fn pooled_text_and_tokens_quantize_as_their_f32_vectors_at_full_and_cut_width() {
    let (f, l) = model("quantized-pooled");
    let s = Session::create(l.m, None).unwrap();
    let (t, _) = tokens(&f.dir, &TEXTS);
    for w in [8, 4] {
        check(
            |o| {
                s.write_text(&TEXTS, Some(o)).unwrap();
                s.run().unwrap()
            },
            &s,
            w,
        );
        check(
            |o| {
                s.write_tokens(&t.batch(), Some(o)).unwrap();
                s.run().unwrap()
            },
            &s,
            w,
        );
    }
}

#[test]
fn ranges_quantize_as_their_f32_vectors_at_full_and_cut_width() {
    let (f, l) = model("quantized-ranges");
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = 8;
    let s = Session::create(l.m, Some(&d)).unwrap();
    let (t, lens) = tokens(&f.dir, &TEXTS);
    let l0 = lens[0] as u32;
    let ranges = [range(0, 0, l0 / 2), range(0, l0 / 2, l0), range(1, 1, 4), range(2, 0, lens[2] as u32)];
    for w in [8, 4] {
        check(
            |o| {
                s.write_ranges(&t.batch(), &ranges, Some(o)).unwrap();
                s.run().unwrap()
            },
            &s,
            w,
        );
    }
}

fn tokens(dir: &std::path::Path, texts: &[&str]) -> (Tokens, Vec<usize>) {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = texts.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let lens = rows.iter().map(Vec::len).collect();
    (Tokens::new(&rows, 0), lens)
}

#[test]
fn options_that_cannot_be_honored_are_refused_by_their_field() {
    let (_, l) = model("quantized-refused");
    let s = Session::create(l.m, None).unwrap();
    let refused = |o: &turbo_embed_options| s.write_text(&TEXTS, Some(o)).expect_err("refused");
    let good = [-1.0f32; 4].into_iter().chain([1.0; 4]).collect::<Vec<_>>();
    let e = refused(&with(0, 4, None));
    assert_eq!((e.code, e.field), (INVALID_ENUM, 0), "{e:?}");
    assert!(e.message.contains("quantization: 4 is not a TURBO_QUANTIZE_* value"), "{e:?}");
    let e = refused(&with(4, TURBO_QUANTIZE_INT8, None));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 8), "{e:?}");
    for q in [0, TURBO_QUANTIZE_NONE, TURBO_QUANTIZE_UBINARY] {
        let e = refused(&with(4, q, Some(&good)));
        assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 8), "quantization {q}: {e:?}");
    }
    let mut nan = good.clone();
    nan[5] = f32::NAN;
    let e = refused(&with(4, TURBO_QUANTIZE_INT8, Some(&nan)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 8), "{e:?}");
    let mut upside = good.clone();
    upside[0] = 2.0;
    let e = refused(&with(4, TURBO_QUANTIZE_INT8, Some(&upside)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 8), "{e:?}");
    assert!(e.message.contains("dimension 0's minimum 2 is over its maximum 1"), "{e:?}");
    // Widths are checked before codes: undeclared, then over dim.
    let e = refused(&with(3, TURBO_QUANTIZE_UBINARY, None));
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    let e = refused(&with(9, TURBO_QUANTIZE_UBINARY, None));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 6), "{e:?}");
    // A refused write leaves nothing written.
    assert_eq!(s.run().err().expect("nothing written").code, INVALID_STATE);
    // The good table at width 4 is taken.
    s.write_text(&TEXTS, Some(&with(4, TURBO_QUANTIZE_INT8, Some(&good)))).unwrap();
    assert_eq!(s.run().unwrap().info().dtype, TURBO_DTYPE_I8);
}

#[test]
fn the_struct_before_quantization_is_still_taken_and_returns_f32() {
    let (_, l) = model("quantized-v1");
    let s = Session::create(l.m, None).unwrap();
    assert_eq!(TURBO_EMBED_OPTIONS_SIZE_V1, 28);
    // Only the first 28 bytes are the caller's; the rest would be refused.
    let mut o = with(4, 99, Some(&[f32::NAN]));
    o.struct_size = TURBO_EMBED_OPTIONS_SIZE_V1 as u32;
    s.write_text(&TEXTS, Some(&o)).unwrap();
    let info = s.run().unwrap().info();
    assert_eq!((info.dtype, info.dim), (TURBO_DTYPE_F32, 4));
    o.struct_size = 32;
    let e = s.write_text(&TEXTS, Some(&o)).err().unwrap();
    assert_eq!(e.code, INVALID_STRUCT_SIZE, "{e:?}");
}
