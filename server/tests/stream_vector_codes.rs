//! Vector codes over AnalyzeStream (#1601, docs/grpc-analysis.md):
//! ANALYZE_OUTPUT_VECTOR_CODES answers each document with the codes of
//! its single-document f32 row under the asked quantization, as a
//! VectorCodesResult outcome of its own; codes refuse f32 (quantization
//! 0) per document, and a codes document never batches with a vectors
//! document of the same quantization.

mod common;

use turbo::quantize::{self, Quantization};
use turbo::{TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY};
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeDocument, AnalyzeOptions, AnalyzeOutput, AnalyzeStreamRequest};

use common::*;

const TEXTS: [&str; 3] = ["The quick brown fox", "reset a password", "jumps over the lazy dog"];
const DIM: usize = 32;

async fn send_codes(stream: &mut Stream, id: &str, text: &str, options: Option<AnalyzeOptions>) {
    stream
        .tx
        .send(AnalyzeStreamRequest {
            kind: Some(Kind::Document(AnalyzeDocument {
                id: id.into(),
                model: NAME.into(),
                text: text.into(),
                options,
                output: AnalyzeOutput::VectorCodes as i32,
                offsets: 0,
            })),
        })
        .await
        .unwrap();
    stream.sent.push(id.into());
}

async fn f32_rows(k: &mut Kserve) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    for t in TEXTS {
        let out = k.model_infer(texts_req(NAME, &[t.as_bytes()])).await.unwrap().into_inner();
        rows.push(out.raw_output_contents[0].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect());
    }
    rows
}

fn ranges_of(rows: &[Vec<f32>]) -> Vec<f32> {
    let min = (0..DIM).map(|d| rows.iter().map(|r| r[d]).fold(f32::INFINITY, f32::min));
    let max = (0..DIM).map(|d| rows.iter().map(|r| r[d]).fold(f32::NEG_INFINITY, f32::max));
    min.chain(max).collect()
}

fn codes_of(q: Quantization, row: &[f32], ranges: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; q.row_bytes(DIM)];
    quantize::rows(q, row, DIM, ranges, &mut out);
    out
}

#[tokio::test]
async fn each_document_gets_its_codes_as_a_codes_outcome() {
    let (s, mut c) = serve().await;
    let mut k = kserve(&s).await;
    let rows = f32_rows(&mut k).await;
    let ranges = ranges_of(&rows);
    let int8 = AnalyzeOptions { quantization: TURBO_QUANTIZE_INT8, int8_ranges: ranges.clone(), ..Default::default() };
    let ubinary = AnalyzeOptions { quantization: TURBO_QUANTIZE_UBINARY, ..Default::default() };
    let before = s.stream_stats().runs;
    let mut stream = open_stream(&mut c, 1, 0, 1000).await;
    for (i, t) in TEXTS.iter().enumerate() {
        send_codes(&mut stream, &format!("ci8-{i}"), t, Some(int8.clone())).await;
        send_codes(&mut stream, &format!("cub-{i}"), t, Some(ubinary.clone())).await;
        // A vectors document under the same int8 table: another batch key.
        stream.send(&format!("vi8-{i}"), NAME, t, Some(int8.clone())).await;
    }
    let got = by_id(stream.half_close()).await;
    for (i, row) in rows.iter().enumerate() {
        for (id, q, code, r) in [
            (format!("ci8-{i}"), Quantization::Int8, TURBO_QUANTIZE_INT8, &ranges),
            (format!("cub-{i}"), Quantization::UBinary, TURBO_QUANTIZE_UBINARY, &ranges),
        ] {
            match got.get(&id).unwrap_or_else(|| panic!("{id} is answered")) {
                Outcome::VectorCodes(v) => {
                    assert_eq!((v.dim, v.quantization), (DIM as u32, code), "{id}");
                    assert_eq!(v.data.to_vec(), codes_of(q, row, r), "{id}: the codes of its f32 row");
                }
                other => panic!("{id}: {other:?}"),
            }
        }
        match got.get(&format!("vi8-{i}")).unwrap() {
            Outcome::Vectors(v) => assert_eq!(v.data.to_vec(), codes_of(Quantization::Int8, row, &ranges), "vi8-{i}"),
            other => panic!("vi8-{i}: {other:?}"),
        }
    }
    assert_eq!(s.stream_stats().runs - before, 3, "codes int8, codes ubinary and vectors int8 batch apart");
}

#[tokio::test]
async fn codes_without_a_quantization_are_the_documents_refusal() {
    let (s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    send_codes(&mut stream, "none", "a", None).await;
    send_codes(&mut stream, "explicit-none", "a", Some(AnalyzeOptions::default())).await;
    let missing = AnalyzeOptions { quantization: TURBO_QUANTIZE_INT8, ..Default::default() };
    send_codes(&mut stream, "missing", "a", Some(missing)).await;
    let narrow = AnalyzeOptions { quantization: TURBO_QUANTIZE_INT8, int8_ranges: vec![0.0; 8], ..Default::default() };
    send_codes(&mut stream, "narrow", "a", Some(narrow)).await;
    let ubinary = AnalyzeOptions { quantization: TURBO_QUANTIZE_UBINARY, ..Default::default() };
    send_codes(&mut stream, "good", "a", Some(ubinary)).await;
    let got = by_id(stream.half_close()).await;
    for (id, code, field) in [("none", 513, 7), ("explicit-none", 513, 7), ("missing", 256, 8), ("narrow", 256, 8)] {
        match &got[id] {
            Outcome::Failure(f) => assert_eq!((f.turbo_code, f.field), (code, field), "{id}: {}", f.message),
            other => panic!("{id}: {other:?}"),
        }
    }
    assert!(matches!(got["good"], Outcome::VectorCodes(_)), "the stream continues past refusals: {:?}", got["good"]);
    let _ = s;
}
