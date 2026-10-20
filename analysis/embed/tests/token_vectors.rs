//! Per-token contextual vector output with source spans (`#709`):
//!
//! 1. Reference parity: unpooled token hidden states and ColBERT-style
//!    linearly projected token vectors (`manifest.projection` /
//!    `ProjectionHead`) match standalone range runs and the bundle's
//!    upstream PyTorch reference embeddings within embedding tolerance.
//! 2. Span reconstruction: special tokens and padding are excluded, and
//!    concatenating `&text[span.range()]` across the returned spans
//!    reconstructs every non-whitespace token of the input byte-for-byte
//!    through normalization, multi-piece UTF-8 characters, and multi-window
//!    stitching; pieces sharing a multi-byte character span keep distinct
//!    per-position token vectors.
//! 3. Window-seam selection: tokens in multi-window inputs are taken from
//!    the window where they are most interior (greatest distance to window
//!    edges, ties to the earlier window), and a token's chosen window and
//!    vector remain invariant when the stride changes.
//! 4. `SubwordVectors` (`Vectors<Subwords>`) document layer and provider
//!    integration.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::head::ProjectionHead;
use turbo::manifest::PromptRole;
use turbo::safetensors::{Dtype, File};
use turbo::turbo_token_range;
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::window::{
    Window, interior_distance, most_interior_window, place_tokens_in_windows, plan_sliding_windows,
};
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Sentences, Subwords, Vectors};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Copies `testdata/tiny-bert-bundle` into an isolated temp directory and
/// attaches a ColBERT-style linear projection head (`[proj_dim, 32]`) to both
/// `weights/model.safetensors` and `manifest.json`.
fn bert_with_projection(name: &str, proj_dim: usize, weight: &[f32], bias: Option<&[f32]>) -> TempDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-embed-proj-{}-{name}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(bert().join(f), dir.join(f)).unwrap();
    }

    let src = fs::read(bert().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let raw_header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let body = &src[8 + n..];

    let mut entries: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    for (k, v) in raw_header {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        entries.push((k, shape, body[a..b].to_vec()));
    }
    let w_bytes: Vec<u8> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();
    entries.push(("colbert.linear.weight".into(), vec![proj_dim as u64, 32], w_bytes));
    if let Some(b) = bias {
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        entries.push(("colbert.linear.bias".into(), vec![proj_dim as u64], b_bytes));
    }

    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (k, shape, bytes) in entries {
        let a = data.len();
        data.extend_from_slice(&bytes);
        let b = data.len();
        header.insert(k, json!({ "dtype": "F32", "shape": shape, "data_offsets": [a, b] }));
    }
    let mut h_bytes = serde_json::to_vec(&header).unwrap();
    while !h_bytes.len().is_multiple_of(8) {
        h_bytes.push(b' ');
    }
    let mut st_bytes = (h_bytes.len() as u64).to_le_bytes().to_vec();
    st_bytes.extend_from_slice(&h_bytes);
    st_bytes.extend_from_slice(&data);

    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &st_bytes).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(bert().join("manifest.json")).unwrap()).unwrap();
    m["projection"] = json!({ "dim": proj_dim });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("projection_weight".into(), json!("colbert.linear.weight"));
    if bias.is_some() {
        tn.insert("projection_bias".into(), json!("colbert.linear.bias"));
    }
    let files = m["files"].as_array_mut().unwrap();
    let w_entry = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w_entry["size"] = json!(st_bytes.len());
    w_entry["sha256"] = json!(sha256_hex(&st_bytes));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();

    TempDir(dir)
}

#[test]
fn unpooled_and_projected_token_vectors_match_reference_model() {
    let opts = VectorOptions { prompt: PromptRole::None, ..VectorOptions::default() };
    let va = VectorAnnotator::open(&bert(), &opts).unwrap();
    assert_eq!(va.dim(), 32);
    assert_eq!(va.token_dim(), 32);
    assert!(va.projection().is_none());

    // Reference cases from `testdata/tiny-bert-bundle/reference/reference.safetensors`:
    // case 1: "The quick brown fox jumps over the lazy dog."
    // case 2: "Café naïve RÉSUMÉ"
    // case 3: "东京是日本的首都。"
    let ref_bytes = fs::read(bert().join("reference/reference.safetensors")).unwrap();
    let ref_st = File::parse("reference.safetensors", &ref_bytes).unwrap();
    let ref_ids = ref_st.get("ids", Dtype::I32, 2).unwrap().i32s();
    let ref_lens = ref_st.get("lengths", Dtype::I32, 1).unwrap().i32s();
    let ref_embs = ref_st.get("embeddings", Dtype::F32, 2).unwrap().f32s();
    let row_width = ref_st.get("ids", Dtype::I32, 2).unwrap().shape[1] as usize;

    let mut enc = Encoder::open(&bert(), &EncoderOptions::default()).unwrap();
    let (head, tail) = va.row_ends();
    assert_eq!(head.len(), 1); // [CLS]
    assert_eq!(tail.len(), 1); // [SEP]

    for (case_idx, text) in [
        (1usize, "The quick brown fox jumps over the lazy dog."),
        (2usize, "Café naïve RÉSUMÉ"),
        (3usize, "东京是日本的首都。"),
    ] {
        let tokens = va.embed_text_tokens(text).unwrap();
        let full_row = &ref_ids[case_idx * row_width..case_idx * row_width + ref_lens[case_idx] as usize];
        // Special tokens ([CLS] at 0 and [SEP] at end) are excluded from `tokens`.
        assert_eq!(tokens.len() + 2, full_row.len(), "case {case_idx}");
        assert_eq!(&tokens.ids[..], &full_row[1..full_row.len() - 1], "case {case_idx}");
        assert_eq!(tokens.windows, [Window { doc: 0, start: 0, end: tokens.len() as u32 }]);
        assert!(tokens.window_indices.iter().all(|&w| w == 0));

        // Verify each token's contextual vector matches a standalone 1-token range
        // run in the encoder, and that pooling all row tokens matches PyTorch reference.
        let mut rows = Rows::new(va.info().max_seq, 0);
        let row = rows.push(&[head, &tokens.ids, tail]);
        let mut all_ranges = Vec::with_capacity(full_row.len());
        for pos in 0..full_row.len() as u32 {
            all_ranges.push(turbo_token_range { row, start: pos, end: pos + 1 });
        }
        let mut unnorm = Vec::new();
        enc.run_ranges(&rows, &all_ranges, &mut unnorm).unwrap();

        // Check every non-special token vector against `tokens.vector(i)`.
        for i in 0..tokens.len() {
            let mut want = unnorm[(i + 1) * 32..(i + 2) * 32].to_vec();
            normalize(&mut want);
            let got = tokens.vector(i);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-6, "case {case_idx} token {i}: {g} vs {w}");
            }
        }

        // Mean of all live tokens in `unnorm` (L2-normalized) matches PyTorch `reference.safetensors`.
        let mut mean = vec![0.0f32; 32];
        for pos in 0..full_row.len() {
            for (m, &v) in mean.iter_mut().zip(&unnorm[pos * 32..(pos + 1) * 32]) {
                *m += v / full_row.len() as f32;
            }
        }
        normalize(&mut mean);
        let want_ref = &ref_embs[case_idx * 32..(case_idx + 1) * 32];
        for (m, &w) in mean.iter().zip(want_ref) {
            assert!((m - w).abs() < 1e-5, "case {case_idx} reference parity: {m} vs {w}");
        }
    }

    // Now test a bundle carrying a ColBERT-style projection head (`[12, 32]`).
    let proj_dim = 12usize;
    let weight: Vec<f32> = (0..proj_dim * 32).map(|i| -0.25 + 0.5 * (i as f32 / (proj_dim * 32 - 1) as f32)).collect();
    let proj_bundle = bert_with_projection("colbert", proj_dim, &weight, None);
    let va_proj = VectorAnnotator::open(&proj_bundle.0, &opts).unwrap();
    assert_eq!(va_proj.dim(), 32);
    assert_eq!(va_proj.token_dim(), proj_dim as u32);
    assert!(va_proj.projection().is_some());

    let text = "The quick brown fox jumps over the lazy dog.";
    let projected = va_proj.embed_text_tokens(text).unwrap();
    assert_eq!(projected.dim, proj_dim as u32);
    assert_eq!(projected.data.len(), projected.len() * proj_dim);

    // Compare against projecting the raw unnormalized hidden states by hand.
    let mut rows = Rows::new(va_proj.info().max_seq, 0);
    let row = rows.push(&[head, &projected.ids, tail]);
    let piece_ranges: Vec<turbo_token_range> =
        (0..projected.len() as u32).map(|i| turbo_token_range { row, start: 1 + i, end: 2 + i }).collect();
    let mut raw_states = Vec::new();
    enc.run_ranges(&rows, &piece_ranges, &mut raw_states).unwrap();
    let manual_head = ProjectionHead::new(32, proj_dim, weight, None).unwrap();
    let want_proj = manual_head.project(&raw_states, true).unwrap();
    for (i, (&g, &w)) in projected.data.iter().zip(&want_proj).enumerate() {
        assert!((g - w).abs() <= 1e-6, "projected[{i}]: got {g}, want {w}");
    }
}

#[test]
fn spans_reconstruct_every_non_whitespace_token_across_normalization_utf8_and_windows() {
    let o = VectorOptions {
        encoder: EncoderOptions { max_seq: 14, ..EncoderOptions::default() },
        prompt: PromptRole::None,
        stride: 5,
    };
    let va = VectorAnnotator::open(&bert(), &o).unwrap();
    assert_eq!(va.width(), 12);
    assert_eq!(va.stride(), 5);

    let cases = [
        "The quick brown fox jumps over the lazy dog.",
        "Café naïve RÉSUMÉ",
        "东京是日本的首都。",
        "unaffable TOKENIZERS, café naïve 東京! Multiple   spaces\tand\nnewlines.",
        "Embedding models turn a passage of text into contextual token vectors. \
         Long inputs that exceed max_seq are split across overlapping windows, \
         and each token keeps its exact byte span in the source text.",
    ];

    for text in cases {
        let out = va.embed_text_tokens(text).unwrap();
        assert!(!out.is_empty());
        for &span in &out.spans {
            assert!(!span.is_empty(), "every token span must be non-empty: {span:?}");
            assert!(
                text.is_char_boundary(span.start as usize) && text.is_char_boundary(span.end as usize),
                "span {span:?} must lie on UTF-8 char boundaries in {text:?}"
            );
        }
        let reconstructed: String = out.spans.iter().map(|s| &text[s.range()]).collect();
        let want_non_ws: String = text.split_whitespace().collect();
        assert_eq!(reconstructed, want_non_ws, "byte-for-byte non-whitespace reconstruction failed for {text:?}");
    }
    // Confirm the long case actually stitched multiple windows.
    let long_out = va.embed_text_tokens(cases[4]).unwrap();
    assert!(long_out.windows.len() >= 4, "expected >= 4 windows, got {:?}", long_out.windows);
}

#[test]
fn multi_piece_utf8_character_sharing_a_span_keeps_distinct_token_vectors() {
    // Two subword pieces sharing the exact same 4-byte UTF-8 character span
    // (as byte-level BPE and SentencePiece byte fallback produce) must receive
    // their own distinct per-position contextual vectors rather than being
    // mean-pooled together by span overlap.
    let va = VectorAnnotator::open(&bert(), &VectorOptions::default()).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(0, 4));
    let subwords = Subwords { spans, ids: vec![1996, 2003], model: va.model().clone(), token: None };
    let tokens = va.embed_subword_tokens("𓀀", &subwords, None, 0).unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens.spans, [Span::new(0, 4), Span::new(0, 4)]);
    assert_ne!(tokens.vector(0), tokens.vector(1), "distinct token positions sharing a span must not be averaged");
}

#[test]
fn window_seam_tokens_select_most_interior_window_and_stay_stable_across_stride_changes() {
    // 1. Pure window planner verification on a 25-piece document with width = 10:
    // For stride = 5: windows are 0..10, 5..15, 10..20, 15..25.
    let windows_s5 = plan_sliding_windows(0, 25, 10, 5);
    assert_eq!(
        windows_s5,
        [
            Window { doc: 0, start: 0, end: 10 },
            Window { doc: 0, start: 5, end: 15 },
            Window { doc: 0, start: 10, end: 20 },
            Window { doc: 0, start: 15, end: 25 },
        ]
    );
    let placed_s5 = place_tokens_in_windows(0, 25, &windows_s5);
    // Piece 9 sits on the right seam of window 0 (dist 0 in 0..10), but in
    // window 1 (5..15) its distance is min(9 - 5, 14 - 9) = 4.
    assert_eq!(interior_distance(9, windows_s5[0]), Some(0));
    assert_eq!(interior_distance(9, windows_s5[1]), Some(4));
    assert_eq!(placed_s5[9].window, 1);
    assert_eq!(placed_s5[9].interior_distance, 4);

    // Piece 7 is a tie between window 0 (min(7, 2) = 2) and window 1 (min(2, 7) = 2);
    // ties break deterministically to the earlier window (window 0).
    assert_eq!(most_interior_window(0, 7, &windows_s5), Some((0, 2)));
    assert_eq!(placed_s5[7].window, 0);

    // 2. End-to-end encoder verification:
    // With max_seq = 12 and prompt = None (overhead = 2), window width = 10.
    let base_opts = VectorOptions {
        encoder: EncoderOptions { max_seq: 12, ..EncoderOptions::default() },
        prompt: PromptRole::None,
        stride: 5,
    };
    let va_s5 = VectorAnnotator::open(&bert(), &base_opts).unwrap();
    let va_s4 = va_s5.clone_with_stride(4);

    let text = "Embedding models turn a passage of text into a vector so that passages with similar meaning land near each other.";
    let out_s5 = va_s5.embed_text_tokens(text).unwrap();
    let out_s4 = va_s4.embed_text_tokens(text).unwrap();
    assert!(out_s5.len() > 20);
    assert_eq!(out_s5.ids, out_s4.ids);
    assert_eq!(out_s5.spans, out_s4.spans);

    // Every token in `0..=6` is most interior in Window 0 (`0..10`) under both
    // stride = 5 (where Window 1 starts at 5) and stride = 4 (where Window 1
    // starts at 4):
    // - At t = 6: in Window 0 (0..10), dist = min(6, 3) = 3.
    //   In Window 1 of stride 5 (5..15), dist = min(1, 8) = 1 < 3 -> Window 0.
    //   In Window 1 of stride 4 (4..14), dist = min(2, 7) = 2 < 3 -> Window 0.
    // Therefore both the chosen window (0) and the resulting contextual vector
    // for tokens 0..=6 are bit-for-bit identical across the stride change!
    for t in 0..=6 {
        assert_eq!(out_s5.window_indices[t], 0, "token {t} with stride 5");
        assert_eq!(out_s4.window_indices[t], 0, "token {t} with stride 4");
        assert_eq!(out_s5.vector(t), out_s4.vector(t), "token {t} vector invariant across stride change");
    }

    // Token 9 (right edge of Window 0, `0..10`) is at distance 0 in Window 0,
    // so under both strides it avoids Window 0's seam and selects Window 1!
    assert_eq!(out_s5.window_indices[9], 1);
    assert_eq!(out_s4.window_indices[9], 1);
    assert!(out_s5.interior_distances[9] >= 3);
    assert!(out_s4.interior_distances[9] >= 3);

    // Verify that every token's vector in `out_s5` is bit-for-bit equal to
    // encoding its chosen window alone in a fresh session.
    let mut enc = Encoder::open(
        &bert(),
        &EncoderOptions { max_seq: 12, max_batch: 1, max_ranges: 1, ..EncoderOptions::default() },
    )
    .unwrap();
    let (head, tail) = va_s5.row_ends();
    let mut rows = Rows::new(12, 0);
    let mut flat = Vec::new();
    for t in 0..out_s5.len() {
        let w_idx = out_s5.window_indices[t] as usize;
        let win = out_s5.windows[w_idx];
        assert!(win.start <= t as u32 && (t as u32) < win.end);
        rows.clear();
        let row = rows.push(&[head, &out_s5.ids[win.start as usize..win.end as usize], tail]);
        let col = head.len() as u32 + (t as u32 - win.start);
        enc.run_ranges(&rows, &[turbo_token_range { row, start: col, end: col + 1 }], &mut flat).unwrap();
        normalize(&mut flat);
        assert_eq!(out_s5.vector(t), &flat[..], "token {t} in window {w_idx} ({win:?})");
    }
}

trait CloneWithStride {
    fn clone_with_stride(&self, stride: u32) -> VectorAnnotator;
}

impl CloneWithStride for VectorAnnotator {
    fn clone_with_stride(&self, stride: u32) -> VectorAnnotator {
        let o = VectorOptions {
            encoder: EncoderOptions { max_seq: self.info().max_seq, ..EncoderOptions::default() },
            prompt: PromptRole::None,
            stride,
        };
        VectorAnnotator::open(&bert(), &o).unwrap()
    }
}

#[test]
fn subword_vectors_annotator_and_provider_populate_document_layer() {
    let mut r: Registry<dyn turbo_analysis::annotate::Step> = Registry::default();
    r.register(SubwordsProvider);
    r.register(VectorsProvider);

    let sub_spec = Spec { location: Some(bert()), options: Vec::new() };
    let vec_spec = Spec { location: Some(bert()), options: Vec::new() }.with("layers", "subwords").with("stride", "6");
    let pipe = r.pipeline(&[("subwords", sub_spec), ("vectors", vec_spec)], |b| b).unwrap();
    assert_eq!(pipe.plan(), vec![vec!["turbo:subwords"], vec!["turbo:subword-vectors"]]);

    let mut docs = vec![
        Document::new("The quick brown fox jumps over the lazy dog."),
        Document::new("Café naïve RÉSUMÉ and 东京是日本的首都。"),
    ];
    for res in pipe.run(&mut docs) {
        res.unwrap();
    }
    for doc in &docs {
        let sub = doc.get::<Subwords>().unwrap();
        let vecs = doc.get::<Vectors<Subwords>>().unwrap();
        assert_eq!(vecs.len(), sub.len());
        assert_eq!(vecs.dim, 32);
        let prov = doc.provenance(Key::<Vectors<Subwords>>::DEFAULT.id()).unwrap();
        assert_eq!(prov.annotator, "turbo:subword-vectors");
        assert_ne!(prov.config, 0);
    }

    // Also verify sentence-guided SubwordVectors when Sentences are present.
    let mut doc = Document::new("First sentence here. Second sentence follows.");
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, 20));
    s_spans.push(Span::new(21, 45));
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("test")).unwrap();

    let va = Arc::new(VectorAnnotator::open(&bert(), &VectorOptions::default()).unwrap());
    let pipe_with_sents = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(&bert()).unwrap())
        .add(SubwordVectors(va))
        .build()
        .unwrap();
    let mut s_docs = vec![doc];
    pipe_with_sents.run(&mut s_docs).remove(0).unwrap();
    let sub = s_docs[0].get::<Subwords>().unwrap();
    let vecs = s_docs[0].get::<Vectors<Subwords>>().unwrap();
    assert_eq!(vecs.len(), sub.len());
}
