//! Group `tasks`, Rust layer: what each task must produce.
//! Rerank scores in input order, classifier rows that sum to one, token
//! spans that slice the original bytes, and the generic RUN contract.

use turbo::abi::*;
use turbo::buffer::BufferDesc;
use turbo::provider::{ClassifyOptions, EmbedOptions, RerankOptions, RunOptions, SessionDesc, TokenBatch};
use turbo::types::{DType, ModelKind, Placement};
use turbo_conformance::{assert_err, needs, read_f32, read_i32, read_rows, BundleKind, Target};

const DOCS: [&str; 5] = ["alpha beta", "beta gamma", "alpha beta", "zzz", "alpha beta gamma delta"];

#[test]
fn tasks_rerank_scores_are_in_input_order() {
    let t = Target::from_env();
    needs!(t, Reranker);
    let (model, session) = t.session(BundleKind::Reranker);
    // The bundle's contract says whether the head's logits are activated: a
    // sigmoid reranker scores in [0, 1], an identity one (what the ms-marco
    // cross-encoders declare) hands back logits.
    let sigmoid = model.bundle().contract().activation.as_deref() == Some("sigmoid");
    session.write_pairs("alpha beta", &DOCS, &RerankOptions::default()).expect("write pairs");
    let result = session.run(&RunOptions::default()).expect("run");
    let scores = read_f32(&result, 0);
    assert_eq!(scores.len(), DOCS.len(), "one score per document, in input order");
    assert_eq!(result.output(0).expect("output").shape, vec![DOCS.len() as u64]);
    for (i, s) in scores.iter().enumerate() {
        assert!(s.is_finite(), "score {i} is not finite: {s}");
        if sigmoid {
            assert!((0.0..=1.0).contains(s), "a sigmoid head's score {i} is outside 0..=1: {s}");
        }
    }
    // Identical documents must score identically, wherever they sit.
    assert_eq!(scores[0], scores[2], "identical documents must score identically");
    // The query's own terms must not lower the score below an unrelated document.
    assert!(scores[0] > scores[3], "a matching document must outrank an unrelated one: {scores:?}");
    drop(result);

    // Re-running with a different document order permutes the scores the same way.
    let reversed: Vec<&str> = DOCS.iter().rev().copied().collect();
    session.write_pairs("alpha beta", &reversed, &RerankOptions::default()).expect("write pairs");
    let result = session.run(&RunOptions::default()).expect("run");
    let other = read_f32(&result, 0);
    for (i, s) in other.iter().enumerate() {
        assert_eq!(*s, scores[DOCS.len() - 1 - i], "scores must follow the input order");
    }
}

#[test]
fn tasks_rerank_sorted_output_is_descending_and_stable_on_ties() {
    let t = Target::from_env();
    needs!(t, Reranker);
    if !t.has(TURBO_CAP_OPT_TOP_N) {
        println!(
            "not applicable: {} device {} does not advertise TURBO_CAP_OPT_TOP_N; \
             capability_top_n_is_honored_or_rejected asserts the refusal",
            t.provider_id(),
            t.ordinal()
        );
        return;
    }
    let (_m, session) = t.session(BundleKind::Reranker);
    let opts = RerankOptions { return_sorted: true, ..Default::default() };
    session.write_pairs("alpha beta", &DOCS, &opts).expect("write pairs");
    let result = session.run(&RunOptions::default()).expect("run");
    let scores = read_f32(&result, 0);
    let sorted = read_i32(&result, 1);
    assert_eq!(sorted.len(), DOCS.len(), "with no top_n the sorted output covers every document");

    let mut seen = vec![false; DOCS.len()];
    for &i in &sorted {
        let i = usize::try_from(i).expect("a sorted index is never negative");
        assert!(i < DOCS.len(), "sorted index {i} is out of range");
        assert!(!seen[i], "document {i} appears twice in the sorted output");
        seen[i] = true;
    }
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0] as usize, pair[1] as usize);
        assert!(scores[a] >= scores[b], "sorted output is not descending: {scores:?} / {sorted:?}");
        if scores[a] == scores[b] {
            assert!(a < b, "ties must keep input order: {a} came after {b} for equal scores");
        }
    }
}

#[test]
fn tasks_rerank_raw_scores_are_unactivated() {
    let t = Target::from_env();
    needs!(t, Reranker);
    let (model, session) = t.session(BundleKind::Reranker);
    let sigmoid = model.bundle().contract().activation.as_deref() == Some("sigmoid");
    session.write_pairs("alpha beta", &DOCS, &RerankOptions::default()).expect("write pairs");
    let activated = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
    let raw_opts = RerankOptions { raw_scores: true, ..Default::default() };
    if !t.has(TURBO_CAP_OPT_RAW_SCORES) {
        // The bit is clear, so the option is refused naming raw_scores;
        // capability_raw_scores_is_honored_or_rejected asserts that too.
        println!(
            "not applicable: {} device {} does not advertise TURBO_CAP_OPT_RAW_SCORES",
            t.provider_id(),
            t.ordinal()
        );
        assert_err!(session.write_pairs("alpha beta", &DOCS, &raw_opts), TURBO_E_UNSUPPORTED_OPTION, field = 6);
        return;
    }
    session.write_pairs("alpha beta", &DOCS, &raw_opts).expect("write pairs");
    let raw = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
    assert_eq!(raw.len(), activated.len());
    assert!(raw.iter().all(|r| r.is_finite()), "a raw score is a finite logit: {raw:?}");
    if sigmoid {
        // The activation is the sigmoid, so the logits are not the scores
        // and each score is computable from its logit.
        assert!(
            raw.iter().zip(&activated).any(|(r, a)| (r - a).abs() > 1e-6),
            "raw_scores must not return the activated scores: {raw:?}"
        );
        for (r, a) in raw.iter().zip(&activated) {
            let s = 1.0 / (1.0 + (-r).exp());
            assert!((s - a).abs() < 1e-4, "the score is the sigmoid of the logit: {r} -> {a}, not {s}");
        }
    } else {
        // An identity head has nothing to undo: its raw scores are its
        // scores, bit for bit, not a second computation of them.
        assert_eq!(raw, activated, "an identity head's raw scores are its scores");
    }
    for i in 0..raw.len() {
        for j in 0..raw.len() {
            assert_eq!(
                raw[i] > raw[j],
                activated[i] > activated[j],
                "the activation must be monotonic: docs {i} and {j}"
            );
        }
    }
}

#[test]
fn tasks_classify_rows_sum_to_one_unless_raw() {
    let t = Target::from_env();
    needs!(t, Classifier);
    let (model, session) = t.session(BundleKind::Classifier);
    let n_labels = model.info().labels.len();
    assert!(n_labels > 0, "a classifier bundle must declare labels");
    let texts = ["hello world", "another line", ""];
    session.write_text_classify(&texts, &ClassifyOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    assert_eq!(result.output(0).expect("output").shape, vec![texts.len() as u64, n_labels as u64]);
    let rows = read_rows(&result, 0, n_labels);
    assert_eq!(rows.len(), texts.len());
    for (i, row) in rows.iter().enumerate() {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}, not 1: {row:?}");
        assert!(row.iter().all(|p| (0.0..=1.0).contains(p)), "row {i} has a probability outside 0..=1: {row:?}");
    }
    drop(result);

    let raw = ClassifyOptions { raw_scores: true, ..Default::default() };
    if !t.has(TURBO_CAP_OPT_RAW_SCORES) {
        println!(
            "not applicable: {} device {} does not advertise TURBO_CAP_OPT_RAW_SCORES",
            t.provider_id(),
            t.ordinal()
        );
        assert_err!(session.write_text_classify(&texts, &raw), TURBO_E_UNSUPPORTED_OPTION, field = 5);
        return;
    }
    session.write_text_classify(&texts, &raw).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let raw_rows = read_rows(&result, 0, n_labels);
    // Every row must be logits, not probabilities: a raw row that still sums
    // to one would be the activated row handed back under another name.
    for (i, r) in raw_rows.iter().enumerate() {
        assert!(
            (r.iter().sum::<f32>() - 1.0).abs() > 1e-4,
            "raw row {i} sums to one, so raw_scores returned the activated scores: {r:?}"
        );
    }
    // The bundle names the activation, so the activated rows are computable
    // from the logits: softmax(raw) must reproduce them.
    let activation = model.bundle().contract().activation.clone();
    assert_eq!(
        activation.as_deref(),
        Some("softmax"),
        "a classifier whose rows sum to one declares softmax in its contract"
    );
    for (i, (r, a)) in raw_rows.iter().zip(&rows).enumerate() {
        let max = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = r.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for (k, (e, p)) in exp.iter().zip(a).enumerate() {
            let want = e / sum;
            assert!(
                (want - p).abs() < 1e-4,
                "row {i} label {k}: softmax of the logits is {want}, the activated score is {p} ({r:?} vs {a:?})"
            );
        }
    }
    // The argmax is unchanged by the activation.
    for (r, a) in raw_rows.iter().zip(&rows) {
        let ri = argmax(r);
        let ai = argmax(a);
        assert_eq!(ri, ai, "the activation changed the predicted label: {r:?} vs {a:?}");
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
}

#[test]
fn tasks_classify_labels_come_from_the_bundle() {
    let t = Target::from_env();
    needs!(t, Classifier);
    let model = t.model(BundleKind::Classifier);
    let labels = model.info().labels.clone();
    assert!(!labels.is_empty());
    for (i, expected) in labels.iter().enumerate() {
        assert_eq!(model.label(i as u32).expect("label"), expected);
    }
    assert_err!(model.label(labels.len() as u32), TURBO_E_INVALID_ARGUMENT);
}

#[test]
fn tasks_token_classify_spans_slice_whole_words() {
    let t = Target::from_env();
    needs!(t, TokenClassifier);
    let (model, session) = t.session(BundleKind::TokenClassifier);
    let texts = ["Alice went to Paris", "Bob"];
    session.write_text_classify(&texts, &ClassifyOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let spans = result.spans().to_vec();
    assert!(!spans.is_empty(), "token classification must report spans");

    for span in &spans {
        let row = span.row as usize;
        assert!(row < texts.len(), "span row {row} is out of range");
        let text = texts[row];
        let (s, e) = (span.byte_start as usize, span.byte_end as usize);
        assert!(s < e, "empty or inverted span {span:?}");
        assert!(e <= text.len(), "span {span:?} reaches past row {row} ({} bytes)", text.len());
        assert!(text.is_char_boundary(s) && text.is_char_boundary(e), "span {span:?} splits a character");
        let slice = &text[s..e];
        assert!(!slice.starts_with(char::is_whitespace), "span {span:?} starts on whitespace: {slice:?}");
        assert!(!slice.ends_with(char::is_whitespace), "span {span:?} ends on whitespace: {slice:?}");
        assert!(s == 0 || text[..s].ends_with(char::is_whitespace), "span {span:?} starts mid-word: {slice:?}");
        assert!(
            e == text.len() || text[e..].starts_with(char::is_whitespace),
            "span {span:?} ends mid-word: {slice:?}"
        );
        assert!((span.label as usize) < model.info().labels.len(), "span {span:?} names no label");
        assert!(
            span.score > 0.0 && span.score <= 1.0 + 1e-5,
            "an aggregated span score is a probability, so it lies in (0, 1]: {span:?}"
        );
        assert!(
            span.score >= 1.0 / model.info().labels.len() as f32,
            "a span carries the winning label, so its score is at least the uniform share 1/{}: {span:?}",
            model.info().labels.len()
        );
    }
    // Spans are grouped by row and ordered inside a row.
    let mut last = (0u32, 0u64);
    for span in &spans {
        if span.row == last.0 {
            assert!(span.byte_start >= last.1, "spans within a row must not go backwards: {spans:?}");
        } else {
            assert!(span.row > last.0, "spans must be grouped by row: {spans:?}");
        }
        last = (span.row, span.byte_end);
    }
    assert!(spans.iter().any(|s| s.row == 1), "the second row must produce spans too");
}

#[test]
fn tasks_token_classify_scores_have_batch_seq_label_shape() {
    let t = Target::from_env();
    needs!(t, TokenClassifier);
    let (model, session) = t.session(BundleKind::TokenClassifier);
    let texts = ["Alice went to Paris", "Bob"];
    session.write_text_classify(&texts, &ClassifyOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let n_labels = model.info().labels.len() as u64;
    let shape = result.output(0).expect("output").shape.clone();
    assert_eq!(shape.len(), 3, "the score tensor is [batch, seq, n_labels], got {shape:?}");
    assert_eq!(shape[0], texts.len() as u64, "batch");
    assert!(shape[1] > 0 && shape[1] <= model.info().max_seq as u64, "seq {} is out of range", shape[1]);
    assert_eq!(shape[2], n_labels, "n_labels");
    let values = read_f32(&result, 0);
    assert_eq!(values.len() as u64, shape.iter().product::<u64>(), "the tensor is packed");
    assert!(values.iter().all(|v| v.is_finite()), "the score tensor has a non-finite value");
}

#[test]
fn tasks_generic_run_computes_y_equals_two_x() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    assert_eq!(model.info().kind, ModelKind::Generic);
    assert_eq!(model.info().inputs.len(), 1, "the generic model declares one input");
    assert_eq!(model.info().inputs[0].name, "x");
    assert_eq!(model.info().outputs[0].name, "y");
    let session = model.create_session(&SessionDesc::default()).expect("session");

    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let x = ctx.alloc(&desc).expect("alloc");
    let values: Vec<f32> = (0..6).map(|i| i as f32 - 2.5).collect();
    write_f32(&x, &values);
    session.bind("x", &x).expect("bind x");
    let result = session.run(&RunOptions::default()).expect("run");
    assert_eq!(result.output(0).expect("output").shape, vec![2, 3], "RUN keeps the input shape");
    let y = read_f32(&result, 0);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(y[i], v * 2.0, "y[{i}] must be 2x");
    }
}

#[test]
fn tasks_generic_run_writes_into_a_caller_bound_output() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 2]).expect("descriptor");
    let x = ctx.alloc(&desc).expect("alloc x");
    let y = ctx.alloc(&desc).expect("alloc y");
    let values = [1.0f32, 2.0, 3.0, 4.0];
    write_f32(&x, &values);
    session.bind("x", &x).expect("bind x");
    session.bind("y", &y).expect("bind y");
    let result = session.run(&RunOptions::default()).expect("run");
    // The caller's own buffer holds the answer, with no copy.
    let got = read_host_f32(&y, 4);
    assert_eq!(got, vec![2.0, 4.0, 6.0, 8.0], "the bound output must be written in place");
    assert_eq!(read_f32(&result, 0), got, "the result must view the bound output");
}

#[test]
fn tasks_generic_run_with_a_small_output_is_capacity() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    let big = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let small = BufferDesc::packed(Placement::Host, DType::F32, &[2]).expect("descriptor");
    let x = ctx.alloc(&big).expect("alloc x");
    let y = ctx.alloc(&small).expect("alloc y");
    session.bind("x", &x).expect("bind x");
    session.bind("y", &y).expect("bind y");
    assert_err!(session.run(&RunOptions::default()), TURBO_E_CAPACITY);
    // The session is still usable with a correctly sized output.
    let good = ctx.alloc(&big).expect("alloc y");
    session.bind("y", &good).expect("bind y");
    session.run(&RunOptions::default()).expect("run");
}

#[test]
fn tasks_generic_run_with_aliased_x_and_y_is_invalid_argument() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let shared = ctx.alloc(&desc).expect("alloc");
    session.bind("x", &shared).expect("bind x");
    session.bind("y", &shared).expect("bind y");
    assert_err!(session.run(&RunOptions::default()), TURBO_E_INVALID_ARGUMENT);
}

#[test]
fn tasks_generic_run_without_a_binding_is_invalid_state() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    // Nothing bound at all: no inputs are ready.
    assert_err!(session.run(&RunOptions::default()), TURBO_E_INVALID_STATE);
    // Only the output bound: the input is still missing.
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let y = ctx.alloc(&desc).expect("alloc");
    session.bind("y", &y).expect("bind y");
    assert_err!(session.run(&RunOptions::default()), TURBO_E_INVALID_STATE);
}

#[test]
fn tasks_result_read_into_a_small_buffer_is_capacity() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let needed = result.output(0).expect("output").logical_bytes().expect("bytes") as usize;
    assert_eq!(needed, model.info().dim as usize * 4);
    let mut small = vec![0u8; needed - 1];
    assert_err!(result.read(0, &mut small), TURBO_E_CAPACITY);
    let mut empty: [u8; 0] = [];
    assert_err!(result.read(0, &mut empty), TURBO_E_CAPACITY);
    // Exactly the right size works, and a larger buffer is fine too.
    let mut exact = vec![0u8; needed];
    assert_eq!(result.read(0, &mut exact).expect("read"), needed);
    let mut larger = vec![0u8; needed * 2];
    assert_eq!(result.read(0, &mut larger).expect("read"), needed, "read reports the logical size");
    assert_eq!(&larger[..needed], &exact[..]);
    // The first index past the end is an argument error, not a wrap-around.
    let past = result.outputs().len() as u32;
    assert_err!(result.read(past, &mut exact), TURBO_E_INVALID_ARGUMENT);
    assert_err!(result.output(past), TURBO_E_INVALID_ARGUMENT);
    assert_err!(result.read(u32::MAX, &mut exact), TURBO_E_INVALID_ARGUMENT);
}

#[test]
fn tasks_result_info_reports_the_output_shape() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    let texts = ["a", "b", "c"];
    session.write_text(&texts, &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let out = result.output(0).expect("output");
    assert_eq!(out.shape, vec![texts.len() as u64, model.info().dim as u64]);
    assert_eq!(out.dtype(), DType::F32);
    let expected = if t.has(TURBO_CAP_DEVICE_RESULT) { Placement::Device } else { Placement::Host };
    assert_eq!(out.placement(), expected, "placement follows the device's TURBO_CAP_DEVICE_RESULT bit");
    assert_eq!(out.logical_bytes().expect("bytes"), (texts.len() * model.info().dim as usize * 4) as u64);
    assert_eq!(&*out.name, "embeddings", "the primary embedding output is named");
    assert_eq!(result.outputs().len(), 1);
}

#[test]
fn tasks_a_task_the_model_does_not_offer_is_unsupported_task() {
    let t = Target::from_env();
    // Each bundle kind carries its own refusals: a device that does not
    // offer one kind must still be held to the refusals of the kinds it
    // does offer, so the gates are per block rather than one over the case.
    needs!(t, Embedding);
    let ctx = t.context();
    let (_m, embed) = t.session(BundleKind::Embedding);
    assert_err!(embed.write_pairs("q", &["d"], &RerankOptions::default()), TURBO_E_UNSUPPORTED_TASK);
    assert_err!(embed.write_text_classify(&["x"], &ClassifyOptions::default()), TURBO_E_UNSUPPORTED_TASK);
    // Binding a named tensor on a non-RUN model is equally refused.
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[1]).expect("descriptor");
    let buffer = ctx.alloc(&desc).expect("alloc");
    assert_err!(embed.bind("x", &buffer), TURBO_E_UNSUPPORTED_TASK);

    match t.offered(BundleKind::Reranker) {
        Ok(()) => {
            let (_m2, rerank) = t.session(BundleKind::Reranker);
            assert_err!(rerank.write_text(&["x"], &EmbedOptions::default()), TURBO_E_UNSUPPORTED_TASK);
        }
        Err(why) => println!("not applicable: the reranker half of this case: {why}"),
    }

    match t.offered(BundleKind::Generic) {
        Ok(()) => {
            let generic = t.model_on(&ctx, BundleKind::Generic);
            let generic_session = generic.create_session(&SessionDesc::default()).expect("session");
            assert_err!(generic_session.write_text(&["x"], &EmbedOptions::default()), TURBO_E_UNSUPPORTED_TASK);
            let ids = [1, 2];
            let mask = [1, 1];
            let batch = TokenBatch { batch: 1, seq: 2, row_stride: 2, ids: &ids, mask: &mask, types: None };
            assert_err!(generic_session.write_tokens(&batch), TURBO_E_UNSUPPORTED_TASK);
        }
        Err(why) => println!("not applicable: the generic half of this case: {why}"),
    }

    match t.offered(BundleKind::Generative) {
        // A generative model runs through the generation API, not a session:
        // the session itself is refused.
        Ok(()) => {
            let generative = t.model(BundleKind::Generative);
            assert_err!(generative.create_session(&SessionDesc::default()), TURBO_E_UNSUPPORTED_TASK);
        }
        Err(why) => println!("not applicable: the generative half of this case: {why}"),
    }
}

fn write_f32(buffer: &turbo::handles::Buffer, values: &[f32]) {
    let ptr = buffer.host_ptr().expect("a host-visible buffer").as_ptr().cast::<f32>();
    // SAFETY: the descriptor reserved at least `values.len()` f32 elements and
    // the test holds the only reference to this buffer.
    unsafe {
        for (i, v) in values.iter().enumerate() {
            *ptr.add(i) = *v;
        }
    }
}

fn read_host_f32(buffer: &turbo::handles::Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.host_ptr().expect("a host-visible buffer").as_ptr().cast::<f32>();
    // SAFETY: as above; `n` elements are inside the allocation.
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}
