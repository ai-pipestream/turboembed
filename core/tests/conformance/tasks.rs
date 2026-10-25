//! The conformance checks for the outputs besides pooled vectors: rerank
//! scores, extractive QA logits, token vectors and sparse term weights, on
//! the device TURBO_TEST_DEVICE names, at TURBO_TEST_PRECISION
//! (docs/conformance.md, Other outputs). A device whose backend does not
//! offer the task, or does not offer the range write that token vectors
//! and sparse terms are read through, is skipped with the reason.

use std::path::Path;
use std::ptr;

use turbo::backend::TURBO_CAP_UNSUPPORTED;
use turbo::bundle::Bundle;
use turbo::head::Linear;
use turbo::mlm::MlmHead;
use turbo::qa::{ExtractiveQa, QaOptions, WindowLogits};
use turbo::safetensors::{Dtype, File};
use turbo::sparse::MaxLogPool;
use turbo::*;

use super::scores::{decision, same_answer, same_order, scale, score_bound, scores_within};
use super::{device, device_backend, precision};
use crate::common::conformance::tolerance;
use crate::common::*;
use crate::{qa_common, rerank_common};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

/// Whether the device offers `task` at the session's precision; when it
/// does not, says so with the backend's reason.
fn offered(task: u32, check: &str) -> bool {
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, ptr::null_mut()) }, 0);
    let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
    cap.struct_size = size_of::<turbo_capability>() as u32;
    let mut err = new_error();
    let rc = unsafe { turbo_runtime_capability(rt, device(rt), task, precision(), &mut cap, &mut err) };
    unsafe { turbo_runtime_release(rt) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    if cap.status == TURBO_CAP_UNSUPPORTED {
        println!("{check}: skipped: the {} backend does not offer it: {}", device_backend(), field(&cap.reason));
        return false;
    }
    true
}

/// `got` against `want` under the tolerance for `compute_dtype`, the
/// values taken as one vector: its cosine and, where the table bounds it,
/// the largest absolute difference. Returns both.
fn within(what: &str, got: &[f32], want: &[f32], compute_dtype: u32) -> (f64, f64) {
    let floor = tolerance(compute_dtype);
    assert_eq!(got.len(), want.len(), "{what}: values");
    let (c, d) = (cosine(got, want), max_abs_diff(got, want));
    assert!(c >= floor.min_cosine, "{what}: cosine {c} is under {}", floor.min_cosine);
    if let Some(most) = floor.max_abs_diff {
        assert!(d <= most, "{what}: max abs diff {d:e} is over {most:e}");
    }
    (c, d)
}

/// One line of `--nocapture` output: `count` of what was compared, and
/// the worst cosine and absolute difference among them.
fn report(what: &str, count: String, compute_dtype: u32, (c, d): (f64, f64)) {
    println!("  {what}: {count}, compute dtype {compute_dtype}, 1 - min cosine {:.3e}, max abs diff {d:.3e}", 1.0 - c);
}

/// One line of `--nocapture` output for a run of `count` scores: the
/// largest difference from the reference, as a fraction of the run's
/// scale, and the bound.
fn report_scores(what: &str, count: usize, compute_dtype: u32, d: f64) {
    println!(
        "  {what}: {count} scores, compute dtype {compute_dtype}, max diff {d:.3e} of the scale, bound {:.0e}",
        score_bound(compute_dtype)
    );
}

/// Rerank scores through turbo_rerank_write_pairs, one pair a write and
/// as many as the session takes, against the transformers scores of
/// testdata/tiny-rerank-reference.
#[test]
fn rerank_scores_match_upstream() {
    if !offered(TURBO_TASK_RERANK, "rerank_scores_match_upstream") {
        return;
    }
    let f = rerank_common::Fixture::new(root());
    let l = Loaded::load_on(&f.dir, device).unwrap_or_else(|e| panic!("{e:?}"));
    let s = Session::create(l.m, Some(&session_desc(0, 0, precision()))).unwrap_or_else(|e| panic!("{e:?}"));
    let si = s.info();
    let (cases, want) = (f.cases(), f.reference());
    let opts = turbo_rerank_options {
        struct_size: size_of::<turbo_rerank_options>() as u32,
        query_tokens: 0,
        max_tokens: 0,
        stride: 0,
    };
    for per in [1, si.max_batch as usize] {
        let mut got = Vec::new();
        for group in cases.chunks(per) {
            let pairs: Vec<turbo_text_pair> =
                group.iter().map(|(q, p)| turbo_text_pair { query: text(q), passage: text(p) }).collect();
            let mut err = new_error();
            let rc = unsafe { turbo_rerank_write_pairs(s.0, pairs.as_ptr(), pairs.len() as u32, &opts, &mut err) };
            assert_eq!(rc, 0, "{:?}", failure(rc, &err));
            got.extend(s.run().unwrap_or_else(|e| panic!("{e:?}")).rows().into_iter().flatten());
        }
        let what = format!("rerank, {per} pairs a write");
        let d = scores_within(&what, &got, &want, si.compute_dtype);
        let margin = 2.0 * score_bound(si.compute_dtype) * scale(&want);
        let (compared, close) = same_order(&what, &got, &want, margin);
        // At F32 the bound is narrow enough that the small bundle's scores
        // are told apart; a check that compared nothing would say nothing.
        if si.compute_dtype == TURBO_DTYPE_F32 {
            assert!(compared > 0, "{what}: no two scores are far enough apart to compare");
        }
        report_scores(&what, got.len(), si.compute_dtype, d);
        println!("    order: {compared} pairs as the reference ranks them, {close} too close to say");
    }
}

/// Extractive QA start and end logits for every window of every case,
/// one window a run and as many as the session takes, against the
/// transformers logits of testdata/tiny-qa-reference. The head runs where
/// the backend puts it (docs/extractive-qa.md).
#[test]
fn extractive_qa_logits_match_upstream() {
    if !offered(TURBO_TASK_EXTRACTIVE_QA, "extractive_qa_logits_match_upstream") {
        return;
    }
    let f = qa_common::Fixture::new(root());
    let bundle = Bundle::open(&f.dir).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let want = File::parse("qa", &bytes).unwrap().get("logits", Dtype::F32, 2).unwrap().f32s();
    let cases: Vec<(&str, &str)> =
        bundle.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.context.as_deref().unwrap())).collect();
    // The compute dtype a session at this precision reports on the device.
    let l = Loaded::load_on(&f.dir, device).unwrap_or_else(|e| panic!("{e:?}"));
    let dtype = Session::create(l.m, Some(&session_desc(0, 0, precision()))).unwrap().info().compute_dtype;
    for max_batch in [1, 0] {
        let o = QaOptions { backend: Some(device_backend()), precision: precision(), max_batch, ..Default::default() };
        let mut qa = ExtractiveQa::open_with(&f.dir, &o).unwrap_or_else(|e| panic!("{e:?}"));
        let got = qa.logits_many(&cases).unwrap();
        let flat: Vec<f32> =
            got.iter().flatten().flat_map(|w| w.start.iter().zip(&w.end).flat_map(|(&s, &e)| [s, e])).collect();
        let what = format!("extractive QA, max_batch {max_batch}");
        let d = scores_within(&what, &flat, &want, dtype);
        // The reference's logits in the device's windows, which hold the
        // same tokens in the same order.
        let (pairs, _) = want.as_chunks::<2>();
        let mut rest = pairs.iter();
        let margin = 4.0 * score_bound(dtype) * scale(&want);
        let mut decided = 0;
        for (i, (windows, (_, context))) in got.iter().zip(&cases).enumerate() {
            let reference: Vec<WindowLogits> = windows
                .iter()
                .map(|w| {
                    let (start, end) = rest.by_ref().take(w.start.len()).map(|&[s, e]| (s, e)).unzip();
                    WindowLogits { start, end, ..w.clone() }
                })
                .collect();
            let (g, r) = (decision(context, windows), decision(context, &reference));
            decided += usize::from(same_answer(&format!("{what}, case {i}"), &g, &r, margin));
        }
        if dtype == TURBO_DTYPE_F32 {
            assert!(decided > 0, "{what}: no case's answer is clear enough to compare");
        }
        report_scores(&what, flat.len(), dtype, d);
        println!(
            "    answers: {decided} of {} cases as the reference answers them, the rest too close to say",
            cases.len()
        );
    }
}

/// The small bundle's reference cases that fit the session, with the
/// session and its tokenizer's pad id, for the checks that read every
/// token's final hidden state through turbo_embed_write_ranges.
struct TokenRun {
    _loaded: Loaded,
    s: Session,
    compute_dtype: u32,
    cases: Vec<(Vec<i32>, Vec<f32>)>,
    pad: i32,
}

impl TokenRun {
    fn new() -> TokenRun {
        let dir = tiny_bundle();
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let r = Reference::read(&dir, &m);
        let loaded = Loaded::load_on(&dir, device).unwrap_or_else(|e| panic!("{e:?}"));
        let mi = loaded.info();
        let mut desc = session_desc(0, 0, precision());
        desc.max_ranges = mi.max_batch * mi.max_seq;
        let s = Session::create(loaded.m, Some(&desc)).unwrap_or_else(|e| panic!("{e:?}"));
        let si = s.info();
        let cases = r.ids.into_iter().zip(r.embeddings).filter(|(ids, _)| ids.len() <= si.max_seq as usize).collect();
        let pad = Tok::create(&dir).unwrap().info().pad_id.max(0);
        TokenRun { _loaded: loaded, s, compute_dtype: si.compute_dtype, cases, pad }
    }

    /// Every token's state of the `rows`, unnormalized and at full width,
    /// per row; or the failure of the write.
    fn states(&self, rows: &[Vec<i32>]) -> Result<Vec<Vec<Vec<f32>>>, Failure> {
        let t = Tokens::new(rows, self.pad);
        let ranges: Vec<turbo_token_range> = rows
            .iter()
            .enumerate()
            .flat_map(|(r, ids)| (0..ids.len() as u32).map(move |p| range(r as u32, p, p + 1)))
            .collect();
        let o = turbo_embed_options { normalize: TURBO_NORMALIZE_NONE, ..embed_options() };
        self.s.write_ranges(&t.batch(), &ranges, Some(&o))?;
        let mut flat = self.s.run()?.rows().into_iter();
        Ok(rows.iter().map(|ids| flat.by_ref().take(ids.len()).collect()).collect())
    }

    /// The states of every case, one row a write and as many as the
    /// session takes; None, having said why, when the device does not
    /// offer the range write.
    fn all_states(&self, check: &str) -> Option<[Vec<Vec<Vec<f32>>>; 2]> {
        let rows: Vec<Vec<i32>> = self.cases.iter().map(|(ids, _)| ids.clone()).collect();
        let mut one = Vec::new();
        for r in &rows {
            match self.states(std::slice::from_ref(r)) {
                Ok(mut s) => one.push(s.remove(0)),
                Err(e) if e.code == status::UNSUPPORTED => {
                    println!("{check}: skipped: the {} backend: {}", device_backend(), e.message);
                    return None;
                }
                Err(e) => panic!("{e:?}"),
            }
        }
        let max_batch = self.s.info().max_batch as usize;
        let full = rows.chunks(max_batch).flat_map(|g| self.states(g).unwrap_or_else(|e| panic!("{e:?}"))).collect();
        Some([one, full])
    }
}

/// The f64 encoder's last hidden states of `ids`, as f32.
fn plain_states(plain: &PlainBert, ids: &[i32]) -> Vec<Vec<f32>> {
    let (mask, types) = (vec![1; ids.len()], vec![0; ids.len()]);
    plain.hidden_states(ids, &mask, &types).iter().map(|r| r.iter().map(|&v| v as f32).collect()).collect()
}

/// Token vectors: every token's final hidden state through
/// turbo_embed_write_ranges, against the f64 encoder over the bundle's
/// weights (common::PlainBert); and their mean, normalized, against the
/// vector sentence-transformers pooled from the same states.
#[test]
fn token_vectors_match_the_reference_encoder_and_pool_to_upstream() {
    let run = TokenRun::new();
    let Some(ways) = run.all_states("token_vectors_match_the_reference_encoder_and_pool_to_upstream") else { return };
    let plain = PlainBert::new(&tiny_bundle());
    let dt = run.compute_dtype;
    for (way, states) in ["one row a write", "full batches"].iter().zip(&ways) {
        let (mut tokens, mut pooled) = ((1.0f64, 0.0f64), (1.0f64, 0.0f64));
        for (i, ((ids, upstream), got)) in run.cases.iter().zip(states).enumerate() {
            for (p, (g, w)) in got.iter().zip(plain_states(&plain, ids)).enumerate() {
                let (c, d) = within(&format!("{way}, case {i}, token {p}"), g, &w, dt);
                tokens = (tokens.0.min(c), tokens.1.max(d));
            }
            let mut mean = vec![0f32; got[0].len()];
            for g in got {
                mean.iter_mut().zip(g).for_each(|(m, v)| *m += v / got.len() as f32);
            }
            let n = mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            mean.iter_mut().for_each(|v| *v /= n);
            let (c, d) = within(&format!("{way}, case {i}, pooled"), &mean, upstream, dt);
            pooled = (pooled.0.min(c), pooled.1.max(d));
        }
        report(
            &format!("token vectors, {way}"),
            format!("{} tokens", states.iter().map(Vec::len).sum::<usize>()),
            dt,
            tokens,
        );
        report(&format!("token vectors pooled, {way}"), format!("{} rows", states.len()), dt, pooled);
    }
}

/// An MLM head on the small bundle's hidden width, its decoder tied to
/// the word embeddings as BERT ties it, with seeded transform values: the
/// small bundle has no trained head, and the head's arithmetic is held to
/// numpy in core/tests/mlm.rs.
fn tiny_mlm_head() -> MlmHead {
    let st = St::read(&tiny_bundle().join("weights/model.safetensors"));
    let shape = st.shape("embeddings.word_embeddings.weight");
    let (vocab, hidden) = (shape[0], shape[1]);
    let line = |lo: f32, hi: f32, n: usize| -> Vec<f32> {
        (0..n).map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32).collect()
    };
    MlmHead::new(
        hidden,
        vocab,
        Linear::new(hidden, hidden, line(-0.2, 0.3, hidden * hidden), line(-0.1, 0.1, hidden)).unwrap(),
        vec![1.0; hidden],
        vec![0.0; hidden],
        1e-12,
        Linear::new(hidden, vocab, st.f32s("embeddings.word_embeddings.weight"), line(-0.05, 0.05, vocab)).unwrap(),
    )
}

/// The SPLADE max-log pool of `states` through `head`.
fn sparse_weights(head: &MlmHead, states: &[Vec<f32>]) -> Vec<f32> {
    let mut pool = MaxLogPool::new(head.vocab);
    for s in states {
        pool.observe(&head.logits(s).unwrap()).unwrap();
    }
    pool.weights()
}

/// Sparse term weights: the device's token states through the MLM head
/// and the max-log pool on the host (as turbo-embed's sparse encoder
/// reads them), against the same head and pool over the f64 encoder's
/// states.
#[test]
fn sparse_weights_match_the_reference_encoder() {
    let run = TokenRun::new();
    let Some(ways) = run.all_states("sparse_weights_match_the_reference_encoder") else { return };
    let (plain, head) = (PlainBert::new(&tiny_bundle()), tiny_mlm_head());
    let want: Vec<Vec<f32>> =
        run.cases.iter().map(|(ids, _)| sparse_weights(&head, &plain_states(&plain, ids))).collect();
    for (way, states) in ["one row a write", "full batches"].iter().zip(&ways) {
        let mut worst = (1.0f64, 0.0f64);
        for (i, (got, w)) in states.iter().zip(&want).enumerate() {
            let got = sparse_weights(&head, got);
            assert!(w.iter().any(|&v| v > 0.0), "case {i}: the reference has no term");
            let (c, d) = within(&format!("sparse, {way}, case {i}"), &got, w, run.compute_dtype);
            worst = (worst.0.min(c), worst.1.max(d));
        }
        report(
            &format!("sparse weights, {way}"),
            format!("{} rows of {} terms", want.len(), head.vocab),
            run.compute_dtype,
            worst,
        );
    }
}
