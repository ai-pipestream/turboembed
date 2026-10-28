//! int8 and ubinary vectors over AnalyzeStream (docs/grpc-analysis.md):
//! `AnalyzeOptions.quantization` and `int8_ranges` give each document the
//! codes of its single-document f32 row, `AnalyzeVectors.quantization`
//! says which, documents batch together only under one table, and a table
//! of the wrong width or a quantized non-vector output is the document's
//! own refusal by field.

mod common;

use turbo::quantize::{self, Quantization};
use turbo::{TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY};
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeOptions, AnalyzeStreamRequest};

use common::*;

const TEXTS: [&str; 4] = ["The quick brown fox", "reset a password", "a", "jumps over the lazy dog"];
const DIM: usize = 32;

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

fn codes(q: Quantization, row: &[f32], ranges: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; q.row_bytes(DIM)];
    quantize::rows(q, row, DIM, ranges, &mut out);
    out
}

#[tokio::test]
async fn each_document_gets_the_codes_of_its_f32_row() {
    let (s, mut c) = serve().await;
    let mut k = kserve(&s).await;
    let rows = f32_rows(&mut k).await;
    let ranges = ranges_of(&rows);
    // A second table: the documents under it must not batch with the first.
    let other: Vec<f32> = ranges.iter().map(|v| v * 2.0).collect();
    let int8 =
        |r: &[f32]| AnalyzeOptions { quantization: TURBO_QUANTIZE_INT8, int8_ranges: r.to_vec(), ..Default::default() };
    let ubinary = AnalyzeOptions { quantization: TURBO_QUANTIZE_UBINARY, ..Default::default() };
    let before = s.stream_stats().runs;
    let mut stream = open_stream(&mut c, 1, 0, 1000).await;
    for (i, t) in TEXTS.iter().enumerate() {
        stream.send(&format!("i8-{i}"), NAME, t, Some(int8(&ranges))).await;
        stream.send(&format!("i8b-{i}"), NAME, t, Some(int8(&other))).await;
        stream.send(&format!("ub-{i}"), NAME, t, Some(ubinary.clone())).await;
    }
    let got = by_id(stream.half_close()).await;
    for (i, row) in rows.iter().enumerate() {
        for (id, q, code, r) in [
            (format!("i8-{i}"), Quantization::Int8, TURBO_QUANTIZE_INT8, &ranges),
            (format!("i8b-{i}"), Quantization::Int8, TURBO_QUANTIZE_INT8, &other),
            (format!("ub-{i}"), Quantization::UBinary, TURBO_QUANTIZE_UBINARY, &ranges),
        ] {
            match got.get(&id).unwrap_or_else(|| panic!("{id} is answered")) {
                Outcome::Vectors(v) => {
                    assert_eq!((v.dim, v.quantization), (DIM as u32, code), "{id}");
                    assert_eq!(v.data.to_vec(), codes(q, row, r), "{id}: the codes of its f32 row");
                }
                other => panic!("{id}: {other:?}"),
            }
        }
    }
    // Three keys, each of four documents: three model executions.
    assert_eq!(s.stream_stats().runs - before, 3, "the two tables and ubinary batch apart");
}

#[tokio::test]
async fn a_wrong_table_or_a_quantized_non_vector_output_is_the_documents_refusal() {
    let (_s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    let narrow = AnalyzeOptions { quantization: TURBO_QUANTIZE_INT8, int8_ranges: vec![0.0; 8], ..Default::default() };
    stream.send("narrow", NAME, "a", Some(narrow)).await;
    let missing = AnalyzeOptions { quantization: TURBO_QUANTIZE_INT8, ..Default::default() };
    stream.send("missing", NAME, "a", Some(missing)).await;
    let tokens = AnalyzeStreamRequest {
        kind: Some(Kind::Document(turbo_kserve::proto::result::AnalyzeDocument {
            id: "tokens".into(),
            model: NAME.into(),
            text: "a".into(),
            options: Some(AnalyzeOptions { quantization: TURBO_QUANTIZE_UBINARY, ..Default::default() }),
            output: 2,
            offsets: 0,
        })),
    };
    stream.tx.send(tokens).await.unwrap();
    stream.sent.push("tokens".into());
    let bad = AnalyzeOptions { quantization: 9, ..Default::default() };
    stream.send("bad", NAME, "a", Some(bad)).await;
    let got = by_id(stream.half_close()).await;
    for (id, code, field) in [("narrow", 256, 8), ("missing", 256, 8), ("tokens", 513, 7), ("bad", 262, 0)] {
        match &got[id] {
            Outcome::Failure(f) => assert_eq!((f.turbo_code, f.field), (code, field), "{id}: {}", f.message),
            other => panic!("{id}: {other:?}"),
        }
    }
}
