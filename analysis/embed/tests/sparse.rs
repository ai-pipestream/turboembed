//! Learned sparse terms on the small BERT of testdata/tiny-bert-bundle,
//! given an MLM head the test appends to its weights: the term selection
//! against the same pieces run by hand in a session of their own, the span
//! rule (input terms keep their pieces' span, byte for byte; expanded
//! terms have none), the cut options, empty texts, windowing past a row,
//! batching, and the refusals.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::manifest::PromptRole;
use turbo::mlm::MlmHead;
use turbo::sparse::SparseOptions as Cut;
use turbo::tokenizer::Encode;
use turbo::tokenizer::Tokenizer;
use turbo::turbo_token_range;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::{
    Encoder, EncoderOptions, Rows, SparseAnnotator, SparseOptions, SparseProvider, SubwordAnnotator,
};
use turbo_analysis_layers::{SparseTerms, Subwords};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

/// `tiny-bert-bundle` copied to scratch, with an MLM head appended to its
/// weights and declared in its manifest: the sparse bundle every test
/// opens. The decoder is tied to the word embeddings.
fn sparse_bundle(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-embed-sparse-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(bert().join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(bert().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();
    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = i as f32 / (count - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    for (name, shape, bytes) in [
        ("mlm.transform.weight", vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
    ] {
        let at = data.len();
        data.extend(bytes);
        header.insert(name.into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
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

    let mut m: Value = serde_json::from_slice(&fs::read(bert().join("manifest.json")).unwrap()).unwrap();
    m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
    tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
    tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
    tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(turbo::bundle::sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn open(dir: &std::path::Path, top_k: u32, floor: f32) -> SparseAnnotator {
    let o = SparseOptions { top_k, floor, ..SparseOptions::default() };
    SparseAnnotator::open(dir, &o).unwrap_or_else(|e| panic!("{e}"))
}

/// The pieces of `text` with the bundle's tokenizer.
fn pieces(dir: &std::path::Path, text: &str) -> (turbo_analysis::Document, Subwords) {
    let doc = turbo_analysis::Document::new(text);
    let view = doc.get::<View>().unwrap_or_else(|| panic!("no view"));
    let sub = SubwordAnnotator::open(dir).unwrap().pieces(view).unwrap_or_else(|e| panic!("{e}"));
    (doc, sub)
}

/// The annotator's terms for one text.
fn terms(dir: &std::path::Path, a: &SparseAnnotator, text: &str) -> (turbo_analysis::Document, SparseTerms<View>) {
    let (doc, sub) = pieces(dir, text);
    let view = doc.get::<View>().unwrap();
    let (t,) = a.annotate((view, &sub)).unwrap_or_else(|e| panic!("{e}"));
    (doc, t)
}

const TEXT: &str = "Embedding models turn a passage of text into a vector. Passages with similar meaning land near \
each other. A search system embeds every document once, and stores the vectors in an index for later.";

#[test]
fn the_terms_match_the_same_pieces_run_by_hand() {
    let dir = sparse_bundle("by-hand");
    let a = open(&dir, 0, 0.0);
    let (doc, t) = terms(&dir, &a, TEXT);
    let (_, sub) = pieces(&dir, TEXT);

    // The same rows and one-token ranges in a session of their own, then
    // the same head and pooling: the annotator must give exactly this.
    let bundle = Bundle::open(&dir).unwrap();
    let tokenizer = Tokenizer::load(&bundle).unwrap();
    let head = MlmHead::load(&bundle).unwrap();
    let mut enc = Encoder::open(&dir, &EncoderOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    let (mut row_head, row_tail) = tokenizer.template_ends();
    let bare = Encode {
        add_special_tokens: false,
        truncation: turbo::manifest::Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::Document,
    };
    tokenizer.encode_into("", bare, &mut row_head).unwrap();
    let mut rows = Rows::new(enc.info().max_seq, tokenizer.fill_id());
    let mut ranges = Vec::new();
    let width = enc.info().max_seq as usize - row_head.len() - row_tail.len();
    for w in 0..sub.ids.len().div_ceil(width) {
        let lo = w * width;
        let hi = (lo + width).min(sub.ids.len());
        let row = rows.push(&[&row_head, &sub.ids[lo..hi], &row_tail]);
        for k in lo..hi {
            let at = row_head.len() as u32 + (k - lo) as u32;
            ranges.push(turbo_token_range { row, start: at, end: at + 1 });
        }
    }
    let mut states = Vec::new();
    enc.run_ranges(&rows, &ranges, &mut states).unwrap_or_else(|e| panic!("{e}"));
    let want = head.sparse_terms(&states, &Cut::all()).unwrap_or_else(|e| panic!("{e}"));

    let got: Vec<(u32, f32)> = {
        let (ids, weights) = t.entries(Ix::new(0));
        ids.iter().copied().zip(weights.iter().copied()).collect()
    };
    let want: Vec<(u32, f32)> = want.iter().map(|t| (t.term, t.weight)).collect();
    assert_eq!(got.len(), want.len(), "{} terms, want {}", got.len(), want.len());
    for ((g, w), (gt, wt)) in got.iter().zip(&want) {
        assert_eq!(g, gt);
        assert!((wt - w).abs() <= 1e-6 * wt.abs().max(1.0), "term {g}: {w} vs {wt}");
    }
    let _ = doc;
}

#[test]
fn input_terms_keep_their_spans_and_expanded_terms_have_none() {
    let dir = sparse_bundle("spans");
    let a = open(&dir, 0, 0.0);
    let (_, t) = terms(&dir, &a, TEXT);
    let (_, sub) = pieces(&dir, TEXT);
    let (ids, weights) = t.entries(Ix::new(0));
    let spans = t.spans(Ix::new(0));
    assert_eq!(ids.len(), spans.len());
    // Sorted by id, positive finite weights.
    assert!(ids.windows(2).all(|w| w[0] < w[1]));
    assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
    assert!(t.check().is_ok());

    // Where each id occurs among the pieces.
    let mut input: Vec<(u32, Vec<usize>)> = Vec::new();
    for (k, &id) in sub.ids.iter().enumerate() {
        match input.iter_mut().find(|(t, _)| *t == id as u32) {
            Some((_, occ)) => occ.push(k),
            None => input.push((id as u32, vec![k])),
        }
    }
    let expanded: Vec<u32> = ids.iter().zip(spans).filter(|(_, s)| s.is_none()).map(|(t, _)| *t).collect();
    assert!(!expanded.is_empty(), "the model adds terms the input does not carry");
    let covered: Vec<u32> = ids.iter().zip(spans).filter(|(_, s)| s.is_some()).map(|(t, _)| *t).collect();
    assert!(!covered.is_empty(), "the input's terms keep their spans");

    // The span rule: an input term's span runs from its first occurrence's
    // start to its last occurrence's end, and its occurrences' spans lie
    // inside it.
    for (t, s) in covered.iter().zip(spans.iter().flatten()) {
        let occ = &input.iter().find(|(id, _)| id == t).unwrap().1;
        let first = sub.spans.get(Ix::new(occ[0] as u32));
        let last = sub.spans.get(Ix::new(occ[occ.len() - 1] as u32));
        assert_eq!(*s, Span::new(first.start, last.end), "term {t}: the span of its pieces");
        for &k in occ {
            let p = sub.spans.get(Ix::new(k as u32));
            assert!(s.start <= p.start && p.end <= s.end, "term {t}: occurrence inside its span");
        }
    }
}

/// A text whose pieces occur once each: the input terms' spans do not
/// overlap and, with the dropped pieces' spans, cover the pieces byte for
/// byte.
const ONCE: &str = "embedding models turn passages into vectors for search retrieval";

#[test]
fn input_spans_tile_a_text_of_single_occurrences_byte_for_byte() {
    let dir = sparse_bundle("once");
    let a = open(&dir, 0, 0.0);
    let (_, t) = terms(&dir, &a, ONCE);
    let (_, sub) = pieces(&dir, ONCE);
    let (ids, _) = t.entries(Ix::new(0));
    let spans = t.spans(Ix::new(0));
    // No id repeats, so no merged span: every kept id occurs exactly once.
    let mut seen = ids.to_vec();
    seen.sort();
    assert!(seen.windows(2).all(|w| w[0] != w[1]));
    assert_eq!(sub.ids.iter().collect::<std::collections::HashSet<_>>().len(), sub.ids.len());

    let mut tiling: Vec<(u32, u32)> = spans.iter().flatten().map(|s| (s.start, s.end)).collect();
    for k in 0..sub.ids.len() {
        if !ids.contains(&(sub.ids[k] as u32)) {
            let s = sub.spans.get(Ix::new(k as u32));
            tiling.push((s.start, s.end));
        }
    }
    tiling.sort();
    let (starts, ends) = sub.spans.columns();
    assert_eq!(tiling.len(), starts.len(), "every piece is covered exactly once");
    for (i, ((a, b), (&s, &e))) in tiling.iter().zip(starts.iter().zip(ends)).enumerate() {
        assert_eq!((*a, *b), (s, e), "piece {i}: the tiling covers it byte for byte");
    }
    // And the terms the input carries are the ids of the pieces, minus
    // what the floor dropped: here the piece bytes are covered span for
    // span, so at least the text's first piece is carried.
    assert!(!tiling.is_empty());
}

#[test]
fn the_cut_options_and_boundary_texts() {
    let dir = sparse_bundle("cuts");
    let a = open(&dir, 3, 0.0);
    let (_, t) = terms(&dir, &a, TEXT);
    let (ids, _) = t.entries(Ix::new(0));
    assert!(ids.len() <= 3, "{} terms, at most 3", ids.len());
    // The cut keeps the heaviest of the uncut selection.
    let (_, full) = terms(&dir, &open(&dir, 0, 0.0), TEXT);
    let (all_ids, all_w) = full.entries(Ix::new(0));
    let mut by_weight: Vec<(u32, f32)> = all_ids.iter().copied().zip(all_w.iter().copied()).collect();
    by_weight.sort_by(|x, y| y.1.total_cmp(&x.1));
    let want: Vec<u32> = {
        let mut w: Vec<u32> = by_weight.iter().take(3).map(|t| t.0).collect();
        w.sort();
        w
    };
    assert_eq!(ids, want.as_slice());

    let a = open(&dir, 0, 1.0);
    let (_, t) = terms(&dir, &a, TEXT);
    let (_, weights) = t.entries(Ix::new(0));
    assert!(weights.iter().all(|w| *w > 1.0), "the floor drops what is under it");

    // top-k past the count keeps every positive weight; an empty text has
    // no pieces, no logits and no terms.
    let a = open(&dir, u32::MAX, 0.0);
    let (_, t) = terms(&dir, &a, TEXT);
    let full = terms(&dir, &open(&dir, 0, 0.0), TEXT).1;
    assert_eq!(t.entries(Ix::new(0)).0, full.entries(Ix::new(0)).0);
    let (_, t) = terms(&dir, &a, "");
    assert_eq!(t.len(), 1);
    assert!(t.entries(Ix::new(0)).0.is_empty());
    assert_eq!(t.check(), Ok(()));
}

#[test]
fn a_long_text_is_read_in_windows_and_pooled_as_one() {
    // Rows of 24 tokens: the text is several windows long.
    let dir = sparse_bundle("windows");
    let o = SparseOptions {
        encoder: EncoderOptions { max_seq: 24, ..EncoderOptions::default() },
        ..SparseOptions::default()
    };
    let a = SparseAnnotator::open(&dir, &o).unwrap_or_else(|e| panic!("{e}"));
    let text = TEXT.repeat(6);
    let (_, t) = terms(&dir, &a, &text);
    let (_, sub) = pieces(&dir, &text);
    // Every input term's span still runs from its first occurrence's
    // start to its last occurrence's end, however many windows the text
    // crossed; the layer holds together.
    let (ids, _) = t.entries(Ix::new(0));
    let spans = t.spans(Ix::new(0));
    for &id in ids {
        // Expanded terms, the model's additions, carry no span.
        let (Some(first), Some(last)) =
            (sub.ids.iter().position(|&x| x == id as i32), sub.ids.iter().rposition(|&x| x == id as i32))
        else {
            continue;
        };
        let p = ids.iter().position(|&t| t == id).unwrap();
        let s = spans[p].unwrap_or_else(|| panic!("term {id} has a span"));
        assert_eq!(s.start, sub.spans.get(Ix::new(first as u32)).start, "term {id}: first piece");
        assert_eq!(s.end, sub.spans.get(Ix::new(last as u32)).end, "term {id}: last piece");
    }
    assert_eq!(t.check(), Ok(()));
}

#[test]
fn refusals_and_the_provider() {
    // A bundle with no sparse declaration refuses the step at open.
    let err = match SparseAnnotator::open(&bert(), &SparseOptions::default()) {
        Ok(_) => panic!("a bundle with no sparse declaration opened the step"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("embed.sparse"), "{err}");

    let dir = sparse_bundle("refusals");
    let a = open(&dir, 0, 0.0);
    // Subwords of another bundle (its manifest's hash differs) fail alone.
    let doc = turbo_analysis::Document::new(TEXT);
    let view = doc.get::<View>().unwrap();
    let other = SubwordAnnotator::open(&bert()).unwrap().pieces(view).unwrap();
    let err = match a.annotate((view, &other)) {
        Ok(_) => panic!("another bundle's subwords passed, and an error is wanted"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("revision"), "{err}");

    // The provider builds the step from a spec whose location is the bundle.
    let spec = Spec { location: Some(sparse_bundle("provider")), options: Vec::new() };
    let step = SparseProvider.create(&spec).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(step.id(), "turbo:sparse");
}
