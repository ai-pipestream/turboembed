//! The speed bar of docs/receipts/server-723/README.md, release only
//! (`cargo test --release -p turbo-kserve --test speed -- --ignored`):
//! encoding and decoding each new result message, per byte, against the
//! current vector message (the vendored protocol's InferTensorContents
//! fp32 contents) and span pairs as that protocol's int contents carries
//! them. Rounds alternate which side runs first, so a warm cache or a
//! lazy allocator never decides the answer.

use std::time::{Duration, Instant};

use prost::Message;
use turbo_kserve::proto::InferTensorContents;
use turbo_kserve::proto::result::{
    ExtractionResult, ModelRef, RerankScoresResult, SparseTermsResult, TokenVectorsResult,
};

const ROUNDS: usize = 40;

fn model() -> ModelRef {
    ModelRef { id: "org/m".into(), revision: "r1".into(), sha256: "abc".into() }
}

/// One alternating measurement: `new` against `base`, timed over the same
/// number of rounds. Returns (new, base) bytes-per-second, each at its own
/// encoded size.
fn alternating(
    new: (&str, impl Fn() -> Vec<u8>, impl Fn(&[u8])),
    base: (&str, impl Fn() -> Vec<u8>, impl Fn(&[u8])),
) -> (f64, f64) {
    let (new_name, new_enc, new_dec) = new;
    let (base_name, base_enc, base_dec) = base;
    // Decode from a fixed buffer, what the wire hands a server.
    let new_buf = new_enc();
    let base_buf = base_enc();
    // Two warm-up rounds take allocator and page faults out of the answer.
    for _ in 0..2 {
        let _ = new_enc();
        new_dec(&new_buf);
        let _ = base_enc();
        base_dec(&base_buf);
    }
    let (mut tn, mut tb) = (Duration::ZERO, Duration::ZERO);
    for r in 0..ROUNDS {
        // The side running first alternates; the accumulators key on the
        // implementation, never on the order of the tuple.
        let (t_new, t_base) = if r % 2 == 0 {
            let a = time(&new_enc, &new_dec, &new_buf);
            let b = time(&base_enc, &base_dec, &base_buf);
            (a, b)
        } else {
            let b = time(&base_enc, &base_dec, &base_buf);
            let a = time(&new_enc, &new_dec, &new_buf);
            (a, b)
        };
        tn += t_new;
        tb += t_base;
    }
    let per = |bytes: usize, t: Duration| (ROUNDS * bytes) as f64 / t.as_secs_f64();
    println!("{new_name}: {} encoded bytes; {base_name}: {} encoded bytes", new_buf.len(), base_buf.len());
    (per(new_buf.len(), tn), per(base_buf.len(), tb))
}

/// One encode and one decode, timed together: what a serving pass costs.
/// The encode is checked against the fixed buffer before its stop.
fn time(enc: &impl Fn() -> Vec<u8>, dec: &impl Fn(&[u8]), buf: &[u8]) -> Duration {
    let t = Instant::now();
    let b = enc();
    let e = t.elapsed();
    assert_eq!(b, buf, "the encoding is stable");
    let t = Instant::now();
    dec(buf);
    e + t.elapsed()
}

fn gb(bytes_per_s: f64) -> f64 {
    bytes_per_s / 1e9
}

/// The current vector message: packed fp32 contents.
fn vector_msg(floats: usize) -> (InferTensorContents, impl Fn(&[u8])) {
    let v: Vec<f32> = (0..floats).map(|i| i as f32 * 0.5).collect();
    (InferTensorContents { fp32_contents: v.clone(), ..Default::default() }, move |b: &[u8]| {
        let m = InferTensorContents::decode(b).unwrap();
        assert_eq!(m.fp32_contents.len(), floats);
    })
}

/// The span message as the current protocol carries one: packed int pairs,
/// offsets up to 2^21 so a varint costs the three bytes a text offset
/// costs (the same encoding shape as the new messages' u32 columns).
fn span_msg(spans: usize) -> (InferTensorContents, impl Fn(&[u8])) {
    let n = 2 * spans;
    let v: Vec<i32> = (0..n as u32).map(|i| (i % 2_000_000) as i32).collect();
    (InferTensorContents { int_contents: v.clone(), ..Default::default() }, move |b: &[u8]| {
        let m = InferTensorContents::decode(b).unwrap();
        assert_eq!(m.int_contents.len(), n);
    })
}

fn report(what: &str, bytes: usize, new: f64, base_name: &str, base: f64) {
    println!("{what}: {bytes} bytes; {new:.0} B/s vs {base_name} {base:.0} B/s ({:.2}x)", new / base);
    assert!(new >= base * 0.97, "{what}: {new:.0} B/s under {base_name}'s {base:.0} B/s");
}

#[test]
#[ignore = "release speed bar: run for the receipt (docs/receipts/server-723)"]
fn result_kinds_encode_and_decode_as_fast_per_byte_as_vector_and_span_messages() {
    // Sparse terms: 100k entries with spans and 150k occurrences, against
    // the span baseline of the same byte count.
    let entries = 100_000usize;
    let occs = 150_000usize;
    let sparse = SparseTermsResult {
        model: Some(model()),
        row: (0..=entries as u32).step_by(100).collect(),
        term: (0..entries as u32).map(|i| i * 7 % 50_000).collect(),
        weight: (0..entries).map(|i| 1.0 + i as f32 * 1e-6).collect(),
        has_span: vec![true; entries],
        span_start: (0..entries as u32).map(|i| i * 3).collect(),
        span_end: (0..entries as u32).map(|i| i * 3 + 2).collect(),
        occ_start: (0..occs as u32).map(|i| i * 2).collect(),
        occ_end: (0..occs as u32).map(|i| i * 2 + 1).collect(),
        occ_at: (0..entries as u32).map(|i| (i as u64 * occs as u64 / entries as u64) as u32).collect(),
        ..Default::default()
    };
    let (sv, sdec) = span_msg(sparse.encoded_len() / 6);
    let (sn, bn) = alternating(
        (
            "sparse",
            || sparse.encode_to_vec(),
            |b| {
                assert_eq!(SparseTermsResult::decode(b).unwrap().term.len(), entries);
            },
        ),
        ("span", || sv.encode_to_vec(), sdec),
    );
    report("sparse terms", sparse.encoded_len(), sn, "span pairs", bn);

    // Token vectors: an 8 MiB block with windows, against the vector
    // baseline of the same byte count (the block itself is bytes, what
    // raw_output_contents already carries vectors as).
    let dim = 128u32;
    let floats = 2 << 20;
    let data = vec![0xabu8; 4 * floats];
    let windows = floats as u32 / dim / 16;
    let vectors = TokenVectorsResult {
        model: Some(model()),
        dim,
        data: data.clone().into(),
        empty: vec![3, 77],
        view: 0,
        window_start: (0..windows).map(|i| i * 16 * dim).collect(),
        window_end: (0..windows).map(|i| i * 16 * dim + 16 * dim).collect(),
        ..Default::default()
    };
    let (vv, vdec) = vector_msg(floats);
    let (vn, bv) = alternating(
        (
            "token vectors",
            || vectors.encode_to_vec(),
            |b| {
                assert_eq!(TokenVectorsResult::decode(b).unwrap().data.len(), 4 * floats);
            },
        ),
        ("vector", || vv.encode_to_vec(), vdec),
    );
    report("token vectors", vectors.encoded_len(), vn, "fp32 contents", bv);

    // Rerank: 100k scored items, against the span baseline of the same
    // byte count (an item and a score are four-byte columns each).
    let items = 100_000usize;
    let rerank = RerankScoresResult {
        model: Some(model()),
        source: "turbo:chunks".into(),
        source_instance: "candidates".into(),
        item: (0..items as u32).rev().collect(),
        score: (0..items).map(|i| items as f32 - i as f32).collect(),
    };
    let (rv, rdec) = span_msg(rerank.encoded_len() / 6);
    let (rn, br) = alternating(
        (
            "rerank",
            || rerank.encode_to_vec(),
            |b| {
                assert_eq!(RerankScoresResult::decode(b).unwrap().item.len(), items);
            },
        ),
        ("span", || rv.encode_to_vec(), rdec),
    );
    report("rerank scores", rerank.encoded_len(), rn, "span pairs", br);

    // Extraction: 10k answers with scores, against the span baseline of
    // the same byte count.
    let answers = 10_000usize;
    let extraction = ExtractionResult {
        model: Some(model()),
        view: 0,
        span_start: (0..answers as u32).map(|i| i * 9).collect(),
        span_end: (0..answers as u32).map(|i| i * 9 + 8).collect(),
        score: (0..answers).map(|i| (answers - i) as f64 * 0.125).collect(),
        probability: vec![1.0 / answers as f64; answers],
        null_score: -1.0,
        null_probability: 0.0,
        ..Default::default()
    };
    let (ev, edec) = span_msg(extraction.encoded_len() / 6);
    let (en, be) = alternating(
        (
            "extraction",
            || extraction.encode_to_vec(),
            |b| {
                assert_eq!(ExtractionResult::decode(b).unwrap().span_start.len(), answers);
            },
        ),
        ("span", || ev.encode_to_vec(), edec),
    );
    report("extraction answers", extraction.encoded_len(), en, "span pairs", be);

    println!(
        "all in GiB/s: sparse {:.2}, token vectors {:.2}, rerank {:.2}, extraction {:.2}",
        gb(sn),
        gb(vn),
        gb(rn),
        gb(en)
    );
}
