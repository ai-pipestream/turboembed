//! int8 and ubinary vectors over ModelInfer (docs/kserve.md, Quantization):
//! the `quantization` parameter and the `int8_ranges` input, against the
//! codes of the same request's f32 vectors, typed and raw, and the
//! refusals by field.

#[path = "kserve/mod.rs"]
mod kserve;

use kserve::*;
use turbo::quantize::{self, Quantization};

const TEXTS: [&str; 3] = ["The quick brown fox", "reset a password", "a"];

/// The per-dimension minimums, then maximums, of `rows`.
fn ranges_of(rows: &[Vec<f32>]) -> Vec<f32> {
    let dim = rows[0].len();
    let min = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::INFINITY, f32::min));
    let max = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::NEG_INFINITY, f32::max));
    min.chain(max).collect()
}

fn fp32(v: &[f32]) -> InferTensorContents {
    InferTensorContents { fp32_contents: v.to_vec(), ..Default::default() }
}

fn with_ranges(mut r: ModelInferRequest, ranges: &[f32], raw: bool) -> ModelInferRequest {
    let width = (ranges.len() / 2) as i64;
    if raw {
        r.inputs.push(input("int8_ranges", "FP32", &[2, width], None));
        r.raw_input_contents.push(ranges.iter().flat_map(|v| v.to_le_bytes()).collect());
    } else {
        r.inputs.push(input("int8_ranges", "FP32", &[2, width], Some(fp32(ranges))));
    }
    r
}

/// The one output's datatype, shape and bytes.
fn codes(r: &ModelInferResponse) -> (String, Vec<i64>, Vec<u8>) {
    let o = &r.outputs[0];
    (o.datatype.clone(), o.shape.clone(), r.raw_output_contents[0].to_vec())
}

#[tokio::test]
async fn int8_and_ubinary_are_the_codes_of_the_f32_vectors() {
    let (s, mut c) = serve(1).await;
    let rows = embed(&mut c, texts(&TEXTS)).await;
    let dim = rows[0].len();
    let ranges = ranges_of(&rows);
    let flat = rows.concat();
    for (q, name, datatype) in
        [(Quantization::Int8, "QUANTIZE_INT8", "INT8"), (Quantization::UBinary, "QUANTIZE_UBINARY", "UINT8")]
    {
        let mut want = vec![0u8; rows.len() * q.row_bytes(dim)];
        quantize::rows(q, &flat, dim, &ranges, &mut want);
        for raw in [false, true] {
            let base = if raw { raw_texts(&TEXTS.map(str::as_bytes)) } else { texts(&TEXTS) };
            let mut r = with(base, "quantization", st(name));
            if q == Quantization::Int8 {
                r = with_ranges(r, &ranges, raw);
            }
            let res = c.model_infer(r).await.unwrap().into_inner();
            let width = q.row_bytes(dim) as i64;
            assert_eq!(codes(&res), (datatype.to_string(), vec![3, width], want.clone()), "{name} raw {raw}");
            let p = |k: &str| res.parameters[k].parameter_choice.clone();
            assert_eq!(p("dim"), Some(int(dim as i64)), "dim counts dimensions");
            assert_eq!(p("bytes"), Some(int(want.len() as i64)));
        }
    }
    // QUANTIZE_NONE is the f32 vectors.
    assert_eq!(embed(&mut c, with(texts(&TEXTS), "quantization", st("QUANTIZE_NONE"))).await, rows);
    s.stop().await;
}

#[tokio::test]
async fn quantization_refusals_name_their_field() {
    let (s, mut c) = serve(1).await;
    let rows = embed(&mut c, texts(&TEXTS)).await;
    let ranges = ranges_of(&rows);
    let int8 = || with(texts(&TEXTS), "quantization", st("QUANTIZE_INT8"));
    // INT8 without ranges, and ranges with another quantization: the library's.
    let e = refused(&mut c, int8()).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 8), "{}", e.message());
    assert!(e.message().contains("field 8 (int8_ranges)"), "{}", e.message());
    let e =
        refused(&mut c, with_ranges(with(texts(&TEXTS), "quantization", st("QUANTIZE_UBINARY")), &ranges, false)).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 8), "{}", e.message());
    // A table narrower than the vectors: the server's, before the library reads it.
    let e = refused(&mut c, with_ranges(int8(), &ranges[..16], false)).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 8), "{}", e.message());
    assert!(e.message().contains("is [2, 8] and the vectors are 32 wide"), "{}", e.message());
    // A table of another datatype or shape.
    let mut r = with_ranges(int8(), &ranges, false);
    r.inputs[1].datatype = "FP64".into();
    let e = refused(&mut c, r).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 0), "{}", e.message());
    let mut r = with_ranges(int8(), &ranges, false);
    r.inputs[1].shape = vec![4, 16];
    let e = refused(&mut c, r).await;
    assert_eq!(refusal(&e).1, 260, "{}", e.message());
    // An unknown constant, and the parameter as a number.
    let e = refused(&mut c, with(texts(&TEXTS), "quantization", st("INT8"))).await;
    assert_eq!(refusal(&e).1, 262, "{}", e.message());
    let e = refused(&mut c, with(texts(&TEXTS), "quantization", int(2))).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 7), "{}", e.message());
    s.stop().await;
}
