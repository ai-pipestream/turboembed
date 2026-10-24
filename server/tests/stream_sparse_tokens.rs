//! Integration tests for `AnalysisService::AnalyzeStream` computing
//! `SparseTermsResult` (`ANALYZE_OUTPUT_SPARSE_TERMS`) and
//! `TokenVectorsResult` (`ANALYZE_OUTPUT_TOKEN_VECTORS`) (`#917`).

mod common;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo_analysis::{Annotator, Document, Ix, Layer, View, ViewId};
use turbo_analysis_embed::{SparseAnnotator, SparseOptions, SubwordAnnotator, VectorAnnotator, VectorOptions};
use turbo_analysis_layers::{SparseTerms, TokenVectorWindows, Vectors};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzeOptions, AnalyzeOutput, AnalyzeStreamRequest, OffsetMode as ProtoOffsetMode,
};
use turbo_kserve::results::Wire;
use turbo_kserve::results::sparse::SparseTermsWire;
use turbo_kserve::results::tokens::TokenVectors;
use turbo_kserve::{Serve, Server};

struct TempBundle(PathBuf);

impl Drop for TempBundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn next_temp_dir(tag: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-kserve-917-{}-{tag}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn extend_bundle(
    tag: &str,
    tensors: &[(&str, Vec<usize>, Vec<u8>)],
    patch_manifest: impl FnOnce(&mut Value),
) -> TempBundle {
    let dir = next_temp_dir(tag);
    let src_dir = tiny();
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(src_dir.join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(src_dir.join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();
    for (name, shape, bytes) in tensors {
        let at = data.len();
        data.extend_from_slice(bytes);
        header.insert((*name).into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut weights = (h.len() as u64).to_le_bytes().to_vec();
    weights.extend(h);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(src_dir.join("manifest.json")).unwrap()).unwrap();
    patch_manifest(&mut m);
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempBundle(dir)
}

/// Builds a SPLADE bundle from `testdata/tiny-bert-bundle` with an MLM head
/// appended to `weights/model.safetensors` and `embed.sparse` in `manifest.json`.
fn splade_bundle(tag: &str) -> TempBundle {
    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = i as f32 / (count - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    let tensors = [
        ("mlm.transform.weight", vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
    ];
    extend_bundle(tag, &tensors, |m| {
        m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
        tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
        tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
        tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
        tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    })
}

/// Builds a ColBERT-style linear projection token-vector bundle (`32 -> 16`)
/// from `testdata/tiny-bert-bundle`.
fn colbert_bundle(tag: &str) -> TempBundle {
    let proj_dim = 16usize;
    let weight_bytes: Vec<u8> =
        (0..proj_dim * 32).flat_map(|i| (((i as f32 * 0.037).sin()) * 0.25).to_le_bytes()).collect();
    let bias_bytes: Vec<u8> = (0..proj_dim).flat_map(|i| (i as f32 * 0.01 - 0.05).to_le_bytes()).collect();
    let tensors = [
        ("colbert.linear.weight", vec![proj_dim, 32], weight_bytes),
        ("colbert.linear.bias", vec![proj_dim], bias_bytes),
    ];
    extend_bundle(tag, &tensors, |m| {
        m["projection"] = json!({ "dim": proj_dim });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("projection_weight".into(), json!("colbert.linear.weight"));
        tn.insert("projection_bias".into(), json!("colbert.linear.bias"));
    })
}

async fn serve_models(models: Vec<(&str, &Path)>) -> (Server, Analysis) {
    let cpu_idx = cpu();
    let configs = models
        .into_iter()
        .map(|(name, path)| ModelConfig {
            bundle: path.to_str().unwrap().into(),
            name: Some(name.into()),
            device: Device::Index(cpu_idx),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 2,
        })
        .collect();
    let server = Server::start("127.0.0.1:0".parse().unwrap(), configs, MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    (server, client)
}

#[allow(clippy::too_many_arguments)]
async fn send_typed(
    stream: &mut Stream,
    id: &str,
    model: &str,
    text: &[u8],
    output: AnalyzeOutput,
    offsets: ProtoOffsetMode,
    options: Option<AnalyzeOptions>,
) {
    stream
        .tx
        .send(AnalyzeStreamRequest {
            kind: Some(Kind::Document(AnalyzeDocument {
                id: id.into(),
                model: model.into(),
                text: text.to_vec(),
                options,
                output: output as i32,
                offsets: offsets as i32,
            })),
        })
        .await
        .unwrap();
    stream.sent.push(id.into());
}

fn assert_sparse_eq(got: &SparseTerms<View>, want: &SparseTerms<View>) {
    assert_eq!(got.model, want.model);
    assert_eq!(got.row, want.row);
    assert_eq!(got.term, want.term);
    assert_eq!(got.span, want.span);
    assert_eq!(got.occ, want.occ);
    assert_eq!(got.occ_at, want.occ_at);
    assert_eq!(got.weight.len(), want.weight.len());
    for (a, b) in got.weight.iter().zip(&want.weight) {
        assert!((a - b).abs() <= 1e-6, "sparse weight {a} vs {b}");
    }
}

fn assert_token_vectors_eq(
    got_v: &Vectors<turbo_analysis_layers::Subwords>,
    got_w: &TokenVectorWindows,
    want_v: &Vectors<turbo_analysis_layers::Subwords>,
    want_w: &TokenVectorWindows,
) {
    assert_eq!(got_v.model, want_v.model);
    assert_eq!(got_v.dim, want_v.dim);
    assert_eq!(got_v.empty, want_v.empty);
    assert_eq!(got_v.data.len(), want_v.data.len());
    for (a, b) in got_v.data.iter().zip(&want_v.data) {
        assert!((a - b).abs() <= 1e-6, "token vector value {a} vs {b}");
    }
    assert_eq!(got_w.spans.view(), want_w.spans.view());
    assert_eq!(got_w.spans.columns(), want_w.spans.columns());
}

#[tokio::test]
async fn stream_sparse_terms_match_in_process_splade_across_batches_orders_and_offsets() {
    let splade = splade_bundle("splade");
    let (s, mut c) = serve_models(vec![("splade", &splade.0)]).await;

    let annotator = SparseAnnotator::open(&splade.0, &SparseOptions::default()).unwrap();
    let cut_opts = SparseOptions { top_k: 12, floor: 0.05, ..SparseOptions::default() };
    let cut_annotator = SparseAnnotator::open(&splade.0, &cut_opts).unwrap();
    let sub_annotator = SubwordAnnotator::open(&splade.0).unwrap();

    let long_text = "Embedding models turn a passage of text into a vector. ".repeat(12);
    let texts = [
        "",
        "The quick brown fox jumps over the lazy dog.",
        "Café naïve RÉSUMÉ",
        "东京是日本的首都。",
        "emoji 🙂 and tabs\tand\nnewlines",
        long_text.as_str(),
    ];

    let want_default: Vec<SparseTerms<View>> = texts
        .iter()
        .map(|&t| {
            let doc = Document::new(t);
            let view = doc.get::<View>().unwrap();
            let sub = sub_annotator.pieces(view).unwrap();
            annotator.annotate((view, &sub)).unwrap().0
        })
        .collect();

    for order in [0, 1] {
        let before = s.stream_stats().runs;
        let mut stream = open_stream(&mut c, order, 0, 1000).await;
        for (i, &t) in texts.iter().enumerate() {
            send_typed(
                &mut stream,
                &format!("sp-{i}"),
                "splade",
                t.as_bytes(),
                AnalyzeOutput::SparseTerms,
                ProtoOffsetMode::Utf8,
                None,
            )
            .await;
        }
        let answers = stream.half_close().collect().await;
        if order == 0 {
            let ids: Vec<String> = answers.iter().map(|(id, _)| id.clone()).collect();
            let expected_ids: Vec<String> = (0..texts.len()).map(|i| format!("sp-{i}")).collect();
            assert_eq!(ids, expected_ids);
        }
        let got: HashMap<_, _> = answers.into_iter().collect();
        for (i, want) in want_default.iter().enumerate() {
            match got.get(&format!("sp-{i}")).unwrap() {
                Outcome::SparseTerms(msg) => {
                    let wire = SparseTermsWire::from_msg(msg).unwrap();
                    assert_eq!(wire.text, texts[i]);
                    assert_sparse_eq(&wire.terms, want);
                }
                other => panic!("sp-{i} unexpected outcome: {other:?}"),
            }
        }
        // All 6 documents shared one micro-batch execution.
        assert_eq!(s.stream_stats().runs - before, 1, "batched sparse terms ran in 1 session execution");
    }

    // Verify top_k + floor cut and UTF-16 / CodePoint wire offsets on multilingual text.
    let multi = "Café naïve 🙂 东京是日本的首都。";
    let doc = Document::new(multi);
    let view = doc.get::<View>().unwrap();
    let sub = sub_annotator.pieces(view).unwrap();
    let want_multi = annotator.annotate((view, &sub)).unwrap().0;
    let want_cut = cut_annotator.annotate((view, &sub)).unwrap().0;

    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    send_typed(
        &mut stream,
        "cut",
        "splade",
        multi.as_bytes(),
        AnalyzeOutput::SparseTerms,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { sparse_top_k: 12, sparse_floor: 0.05, ..Default::default() }),
    )
    .await;
    for (mode, tag) in
        [(ProtoOffsetMode::Utf8, "utf8"), (ProtoOffsetMode::Utf16, "utf16"), (ProtoOffsetMode::CodePoint, "cp")]
    {
        send_typed(&mut stream, tag, "splade", multi.as_bytes(), AnalyzeOutput::SparseTerms, mode, None).await;
    }
    let got = by_id(stream.half_close()).await;
    let sparse_at = |k: &str| match got.get(k).unwrap() {
        Outcome::SparseTerms(m) => m.clone(),
        other => panic!("{k}: {other:?}"),
    };
    let decoded_cut = SparseTerms::from_msg(&sparse_at("cut")).unwrap();
    assert_eq!(decoded_cut.term.len(), 12);
    assert_sparse_eq(&decoded_cut, &want_cut);
    let (utf8_msg, utf16_msg, cp_msg) = (sparse_at("utf8"), sparse_at("utf16"), sparse_at("cp"));
    assert!(!utf8_msg.occ_end.is_empty(), "input terms carry occurrence spans");
    assert_ne!(utf8_msg.occ_end, utf16_msg.occ_end, "UTF-16 wire offsets differ on non-ASCII text");
    assert_ne!(utf16_msg.occ_end, cp_msg.occ_end, "CodePoint wire offsets differ from UTF-16 on supplementary emoji");
    for msg in [&utf8_msg, &utf16_msg, &cp_msg] {
        let decoded = SparseTerms::from_msg(msg).unwrap();
        assert_sparse_eq(&decoded, &want_multi);
    }
}

#[tokio::test]
async fn stream_token_vectors_match_in_process_unprojected_and_colbert_projected_bundles() {
    let colbert = colbert_bundle("colbert");
    let tiny_path = tiny();
    let (s, mut c) = serve_models(vec![("base", &tiny_path), ("colbert", &colbert.0)]).await;

    let base_ann = VectorAnnotator::open(&tiny_path, &VectorOptions::default()).unwrap();
    let colbert_ann =
        VectorAnnotator::open(&colbert.0, &VectorOptions { stride: 16, ..VectorOptions::default() }).unwrap();

    let long_text = "Café naïve 🙂 东京是日本的首都。 The quick brown fox jumps over the lazy dog. ".repeat(8);
    let texts =
        ["", "The quick brown fox jumps over the lazy dog.", "Café naïve 🙂 东京是日本的首都。", long_text.as_str()];

    // 1. Base (unprojected 32-dim) bundle batching and parity.
    let before = s.stream_stats().runs;
    let mut stream = open_stream(&mut c, 0, 0, 1000).await;
    for (i, &t) in texts.iter().enumerate() {
        send_typed(
            &mut stream,
            &format!("tv-{i}"),
            "base",
            t.as_bytes(),
            AnalyzeOutput::TokenVectors,
            ProtoOffsetMode::Utf8,
            None,
        )
        .await;
    }
    let got = by_id(stream.half_close()).await;
    assert_eq!(s.stream_stats().runs - before, 1, "batched token vectors ran in 1 session execution");
    for (i, &t) in texts.iter().enumerate() {
        let want_ctx = base_ann.embed_text_tokens(t).unwrap();
        let want_v = want_ctx.to_subword_vectors();
        let want_w = want_ctx.to_token_vector_windows(ViewId::SOURCE);
        match got.get(&format!("tv-{i}")).unwrap() {
            Outcome::TokenVectors(msg) => {
                let decoded = TokenVectors::from_msg(msg).unwrap();
                let got_v = decoded.vectors().unwrap();
                let got_w = decoded.windows_layer(got_v.data.len() / got_v.dim as usize).unwrap();
                assert_token_vectors_eq(&got_v, &got_w, &want_v, &want_w);
            }
            other => panic!("tv-{i} unexpected outcome: {other:?}"),
        }
    }

    // 2. ColBERT projected (16-dim) bundle with custom token_stride and UTF-8 / UTF-16 / CodePoint offsets.
    let want_colbert = colbert_ann.embed_text_tokens_with_stride(&long_text, 16).unwrap();
    let want_v = want_colbert.to_subword_vectors();
    let want_w = want_colbert.to_token_vector_windows(ViewId::SOURCE);
    assert_eq!(want_v.dim, 16);
    assert!(want_w.spans.len() >= 2, "long text spans multiple encoder windows");

    let mut stream = open_stream(&mut c, 1, 0, 0).await;
    for (mode, tag) in [
        (ProtoOffsetMode::Utf8, "cb-utf8"),
        (ProtoOffsetMode::Utf16, "cb-utf16"),
        (ProtoOffsetMode::CodePoint, "cb-cp"),
    ] {
        send_typed(
            &mut stream,
            tag,
            "colbert",
            long_text.as_bytes(),
            AnalyzeOutput::TokenVectors,
            mode,
            Some(AnalyzeOptions { token_stride: 16, ..Default::default() }),
        )
        .await;
    }
    let got = by_id(stream.half_close()).await;
    let tv_at = |k: &str| match got.get(k).unwrap() {
        Outcome::TokenVectors(m) => m.clone(),
        other => panic!("{k}: {other:?}"),
    };
    let (u8_msg, u16_msg, cp_msg) = (tv_at("cb-utf8"), tv_at("cb-utf16"), tv_at("cb-cp"));
    assert_ne!(u8_msg.window_end, u16_msg.window_end);
    assert_ne!(u16_msg.window_end, cp_msg.window_end);
    for msg in [&u8_msg, &u16_msg, &cp_msg] {
        let decoded = TokenVectors::from_msg(msg).unwrap();
        let got_v = decoded.vectors().unwrap();
        let got_w = decoded.windows_layer(got_v.data.len() / got_v.dim as usize).unwrap();
        assert_token_vectors_eq(&got_v, &got_w, &want_v, &want_w);
        assert_eq!(got_w.spans.get(Ix::new(0)).start, 0);
    }
}

#[tokio::test]
async fn stream_sparse_and_token_vector_refusals_report_per_document_failures_and_continue() {
    let tiny_path = tiny();
    let (_s, mut c) = serve_models(vec![("base", &tiny_path)]).await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;

    // 1. Asking for sparse terms on a bundle with no MLM head.
    send_typed(&mut stream, "no-mlm", "base", b"hello", AnalyzeOutput::SparseTerms, ProtoOffsetMode::Utf8, None).await;
    // 2. Negative sparse_floor.
    send_typed(
        &mut stream,
        "bad-floor",
        "base",
        b"hello",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { sparse_floor: -0.5, ..Default::default() }),
    )
    .await;
    // 3. Invalid UTF-8 document bytes.
    send_typed(
        &mut stream,
        "bad-utf8",
        "base",
        &[0xff, 0xfe],
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        None,
    )
    .await;
    // 4. token_stride exceeding window width.
    send_typed(
        &mut stream,
        "bad-stride",
        "base",
        b"hello",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { token_stride: 999, ..Default::default() }),
    )
    .await;
    // 5. Sound token-vectors request after failures.
    send_typed(
        &mut stream,
        "ok-tokens",
        "base",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        None,
    )
    .await;

    let got = by_id(stream.half_close()).await;
    for (k, want_code) in [("no-mlm", 513), ("bad-floor", 256), ("bad-utf8", 258), ("bad-stride", 256)] {
        match got.get(k).unwrap() {
            Outcome::Failure(f) => assert_eq!(f.turbo_code, want_code, "{k}: {f:?}"),
            other => panic!("{k}: {other:?}"),
        }
    }
    match got.get("ok-tokens").unwrap() {
        Outcome::TokenVectors(m) => {
            let decoded = TokenVectors::from_msg(m).unwrap();
            assert_eq!(decoded.vectors().unwrap().len(), 2);
        }
        other => panic!("ok-tokens: {other:?}"),
    }
}
