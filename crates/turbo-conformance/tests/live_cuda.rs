//! CUDA-specific live checks against real ONNX bundles on an NVIDIA device.
//!
//! Selection is described in `turbo_conformance::live`: these tests run only
//! when `TURBO_LIVE_PROVIDER` is `cuda` and print a reason and return
//! otherwise, so the file is safe inside `cargo test --workspace`.
//!
//! `live_embed.rs` and `live_tasks.rs` cover what every provider must do.
//! This file covers what the CUDA provider's own data path has to get right
//! and what the shared files do not reach: the four bundle kinds on one
//! context with the placement each stage really runs at, the error paths
//! (an over-long pair, a token id outside the vocabulary, batch and sequence
//! limits, a role the bundle has no prefix for, a task written to the wrong
//! model kind), the H2D/D2H byte accounting across repeated runs, two
//! models running concurrently on one device, and the refusal to fall back
//! when a second device ordinal does not exist.
//!
//! Bundles: `TURBO_LIVE_BUNDLE` (MiniLM), `TURBO_LIVE_RERANK_BUNDLE`,
//! `TURBO_LIVE_CLASSIFY_BUNDLE`, `TURBO_LIVE_NER_BUNDLE`.

use std::sync::Arc;

use turbo::abi;
use turbo::provider::TokenBatch;
use turbo::{
    CapStatus, ClassifyOptions, Context, ContextDesc, DeviceSelector, EmbedOptions, Modality, ModelDesc, ModelKind,
    Placement, PromptRole, RerankOptions, RuntimeDesc, SelectPolicy, SessionDesc, Stage, StagePlacement, Task,
    Truncate,
};
use turbo_conformance::live::{cosine, live, Live};
use turbo_conformance::read_f32;

/// The live device, but only when it is a CUDA one. Every other provider
/// prints a reason and skips, so this file adds nothing to their runs.
fn cuda() -> Option<Live> {
    let live = live()?;
    if live.provider != "cuda" {
        println!("not applicable: TURBO_LIVE_PROVIDER is `{}`, not `cuda`", live.provider);
        return None;
    }
    Some(live)
}
/// The bundle directory `var` names, for a task the device under test
/// offers. A device whose capability cell does not offer the task prints
/// `not applicable` and the case returns; a device that does offer it with
/// no bundle configured is a configuration error and panics naming the
/// variable, so a live run never skips a case it could have run (the rule
/// `Target::offered` applies in `crates/turbo-conformance/src/lib.rs`).
fn bundle_for(live: &Live, task: Task, var: &str) -> Option<std::path::PathBuf> {
    let cell = live
        .ctx
        .runtime()
        .capability(live.ctx.device_index(), task, Modality::Text)
        .unwrap_or_else(|e| panic!("{task:?} x TEXT capability of `{}`: {e}", live.device.name));
    if matches!(cell.status, CapStatus::Unsupported | CapStatus::Planned) {
        println!(
            "not applicable: {} device {} (`{}`) does not offer {task:?} for Text (capability {:?})",
            live.provider, live.device.ordinal, live.device.name, cell.status
        );
        return None;
    }
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Some(std::path::PathBuf::from(v)),
        _ => panic!(
            "{} device {} (`{}`) offers {task:?} but {var} is not set; point it at a bundle of that kind",
            live.provider, live.device.ordinal, live.device.name
        ),
    }
}

/// The provider library path, for the tests that build their own runtime.
fn lib() -> String {
    std::env::var("TURBO_LIVE_LIB").expect("TURBO_LIVE_LIB (cuda() already checked it)")
}

#[test]
fn live_cuda_runs_all_four_bundle_kinds_on_one_context_with_honest_stage_placement() {
    let Some(live) = cuda() else { return };
    let Some(embed_dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let Some(rerank_dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
    let Some(classify_dir) = bundle_for(&live, Task::Classify, "TURBO_LIVE_CLASSIFY_BUNDLE") else { return };
    let Some(ner_dir) = bundle_for(&live, Task::TokenClassify, "TURBO_LIVE_NER_BUNDLE") else { return };
    let text = "Angela Merkel visited Berlin in 2015.";

    // Embedding: pooling and normalization run on the device, tokenization
    // on the host, and there is no post-processing stage.
    let model = live.ctx.load_model(&embed_dir, &ModelDesc::default()).expect("load MiniLM");
    assert_eq!(model.info().kind, ModelKind::Embedding);
    let stages = model.info().stages;
    assert_eq!(stages.0[Stage::Tokenize as usize], StagePlacement::Host, "WordPiece runs on the host");
    assert_eq!(stages.0[Stage::Encode as usize], StagePlacement::Device);
    assert_eq!(stages.0[Stage::Pool as usize], StagePlacement::Device, "pooling runs where the hidden state is");
    assert_eq!(stages.0[Stage::Normalize as usize], StagePlacement::Device);
    assert_eq!(stages.0[Stage::Postprocess as usize], StagePlacement::Unused);
    assert!(!stages.fully_accelerated(), "tokenization is host work, so the model is not fully accelerated");
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    session.write_text(&[text], &EmbedOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(0).unwrap().shape, vec![1, 384]);
    drop(r);

    // Reranker: one score per document, post-processing (the sigmoid) on
    // the device, no pooling stage.
    let model = live.ctx.load_model(&rerank_dir, &ModelDesc::default()).expect("load reranker");
    assert_eq!(model.info().kind, ModelKind::Reranker);
    let stages = model.info().stages;
    assert_eq!(stages.0[Stage::Pool as usize], StagePlacement::Unused);
    assert_eq!(stages.0[Stage::Postprocess as usize], StagePlacement::Device);
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    session.write_pairs("who visited Berlin?", &[text], &RerankOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(0).unwrap().shape, vec![1]);
    drop(r);

    // Classifier: one row of label scores, softmax on the device.
    let model = live.ctx.load_model(&classify_dir, &ModelDesc::default()).expect("load classifier");
    assert_eq!(model.info().kind, ModelKind::Classifier);
    let labels = model.info().labels.len() as u64;
    assert!(labels >= 2, "a sequence classifier has at least two labels");
    assert_eq!(model.info().stages.0[Stage::Postprocess as usize], StagePlacement::Device);
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    session.write_text_classify(&[text], &ClassifyOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(0).unwrap().shape, vec![1, labels]);
    drop(r);

    // Token classifier: per-token label scores, and span aggregation is
    // host work the provider names as such rather than claiming the device.
    let model = live.ctx.load_model(&ner_dir, &ModelDesc::default()).expect("load token classifier");
    assert_eq!(model.info().kind, ModelKind::TokenClassifier);
    assert_eq!(
        model.info().stages.0[Stage::Postprocess as usize],
        StagePlacement::Host,
        "span aggregation runs on the host and the provider says so"
    );
    let ner_labels = model.info().labels.len() as u64;
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    session.write_text_classify(&[text], &ClassifyOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    let shape = &r.output(0).unwrap().shape;
    assert_eq!(shape.len(), 3, "token classification is [rows, seq, labels]: {shape:?}");
    assert_eq!(shape[0], 1);
    assert_eq!(shape[2], ner_labels);
    assert!(!r.spans().is_empty(), "`{text}` has at least one entity span");
}

#[test]
fn live_cuda_rerank_refuses_an_over_long_pair_instead_of_reporting_an_internal_error() {
    let Some(live) = cuda() else { return };
    let Some(dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load reranker");
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 32, ..Default::default() }).unwrap();
    let query = "how many people live in berlin";
    let long = "berlin has a population of three and a half million registered inhabitants ".repeat(8);
    // Truncation NONE over the budget is a capacity error naming the budget,
    // not an internal one: the packer's "too long" status is its own code.
    let e = session
        .write_pairs(query, &[long.as_str()], &RerankOptions { truncate: Truncate::None, ..Default::default() })
        .expect_err("an over-long pair with truncation NONE must be refused");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "an over-long pair is a capacity error, not {e}");
    // The same pair fits under the model's own truncation.
    session.write_pairs(query, &[long.as_str()], &RerankOptions::default()).expect("model truncation packs the pair");
    session.run(&Default::default()).expect("run the truncated pair");
    // LEFT truncation of a pair is not offered, and the refusal names the
    // field rather than silently truncating from the other end.
    let e = session
        .write_pairs(query, &[long.as_str()], &RerankOptions { truncate: Truncate::Left, ..Default::default() })
        .expect_err("pair truncation from the left is not offered");
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{e}");
    assert_eq!(e.field(), RerankOptions::FIELD_TRUNCATE, "the refusal names truncate: {e}");
}

#[test]
fn live_cuda_prepared_tokens_reject_an_id_outside_the_vocabulary() {
    let Some(live) = cuda() else { return };
    let Some(dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let vocab = model.info().vocab_size;
    assert!(vocab > 0, "the MiniLM bundle states its vocabulary size");
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 16, ..Default::default() }).unwrap();
    let seq = 4u32;
    let good = [101i32, 7592, 2088, 102];
    let mask = [1i32, 1, 1, 1];
    fn row<'a>(seq: u32, ids: &'a [i32], mask: &'a [i32]) -> TokenBatch<'a> {
        TokenBatch { batch: 1, seq, row_stride: seq, ids, mask, types: None }
    }
    session.write_tokens(&row(seq, &good, &mask)).expect("a valid token row is accepted");
    let r = session.run(&Default::default()).expect("run prepared tokens");
    let baseline = read_f32(&r, 0);
    assert_eq!(baseline.len(), 384);
    drop(r);
    // One id past the end of the vocabulary would index the embedding table
    // out of bounds on the device; it is refused before anything is uploaded.
    let over = [101i32, vocab as i32, 2088, 102];
    let e = session.write_tokens(&row(seq, &over, &mask)).expect_err("a token id at vocab_size must be refused");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "{e}");
    let negative = [101i32, -1, 2088, 102];
    let e = session.write_tokens(&row(seq, &negative, &mask)).expect_err("a negative token id must be refused");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "{e}");
    // A mask value that is neither 0 nor 1 is refused as well.
    let odd_mask = [1i32, 2, 1, 1];
    let e = session
        .write_tokens(&TokenBatch { batch: 1, seq, row_stride: seq, ids: &good, mask: &odd_mask, types: None })
        .expect_err("a mask value of 2 must be refused");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "{e}");
    // After the refusals the session still runs the last good input.
    session.write_tokens(&row(seq, &good, &mask)).expect("the session survives the refusals");
    let r = session.run(&Default::default()).expect("run again");
    assert!(cosine(&read_f32(&r, 0), &baseline) > 0.99999, "the same tokens give the same vector");
}

#[test]
fn live_cuda_batch_and_sequence_limits_are_capacity_errors() {
    let Some(live) = cuda() else { return };
    let Some(dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let model_max_seq = model.info().max_seq;
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 16, ..Default::default() }).unwrap();
    // More rows than the session declared.
    let e = session.write_text(&["a", "b", "c"], &EmbedOptions::default()).expect_err("3 rows into a 2-row session");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "{e}");
    // No rows at all.
    let e = session.write_text(&[], &EmbedOptions::default()).expect_err("an empty batch");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "{e}");
    // max_tokens above the model's own maximum, naming the field.
    let over = EmbedOptions { max_tokens: model_max_seq + 1, ..Default::default() };
    let e = session.write_text(&["a"], &over).expect_err("max_tokens above the model maximum");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "{e}");
    assert_eq!(e.field(), EmbedOptions::FIELD_MAX_TOKENS, "{e}");
    // max_tokens between the session width and the model maximum: the
    // session, not the model, is the binding limit and the error says so.
    let over = EmbedOptions { max_tokens: 17, ..Default::default() };
    let e = session.write_text(&["a"], &over).expect_err("max_tokens above the session width");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "{e}");
    // A prepared-token row wider than the session.
    let ids = vec![101i32; 32];
    let mask = vec![1i32; 32];
    let e = session
        .write_tokens(&TokenBatch { batch: 1, seq: 32, row_stride: 32, ids: &ids, mask: &mask, types: None })
        .expect_err("a 32-column row into a 16-column session");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "{e}");
    // A session wider than the model is refused at creation, not clamped.
    let e = model
        .create_session(&SessionDesc { max_batch: 1, max_seq: model_max_seq + 1, ..Default::default() })
        .expect_err("a session wider than the model");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "{e}");
}

#[test]
fn live_cuda_a_task_written_to_the_wrong_model_kind_is_unsupported_task() {
    let Some(live) = cuda() else { return };
    let Some(embed_dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let Some(rerank_dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
    let embed = live.ctx.load_model(&embed_dir, &ModelDesc::default()).expect("load MiniLM");
    let embed_session = embed.create_session(&SessionDesc { max_batch: 2, max_seq: 32, ..Default::default() }).unwrap();
    let e = embed_session.write_pairs("q", &["d"], &RerankOptions::default()).expect_err("an embedder cannot rerank");
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_TASK, "{e}");
    let e = embed_session
        .write_text_classify(&["d"], &ClassifyOptions::default())
        .expect_err("an embedder cannot classify");
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_TASK, "{e}");
    let rerank = live.ctx.load_model(&rerank_dir, &ModelDesc::default()).expect("load reranker");
    let rerank_session =
        rerank.create_session(&SessionDesc { max_batch: 2, max_seq: 32, ..Default::default() }).unwrap();
    let e = rerank_session.write_text(&["t"], &EmbedOptions::default()).expect_err("a reranker cannot embed");
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_TASK, "{e}");
    // Running with nothing written is a state error, not an empty result.
    let e = rerank_session.run(&Default::default()).expect_err("a run with no input");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_STATE, "{e}");
}

#[test]
fn live_cuda_a_prompt_role_the_bundle_has_no_prefix_for_is_refused() {
    let Some(live) = cuda() else { return };
    let Some(dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 32, ..Default::default() }).unwrap();
    let query_prefix = model.info().prefix_query.clone();
    let opts = EmbedOptions { prompt_role: PromptRole::Query, ..Default::default() };
    match session.write_text(&["what is a gpu"], &opts) {
        Ok(()) => {
            assert!(!query_prefix.is_empty(), "a role was honored, so the bundle must declare a query prefix");
            // The prefix has to change the vector; accepting the role and
            // embedding the bare text would be a silent substitution.
            let r = session.run(&Default::default()).unwrap();
            let with_role = read_f32(&r, 0);
            drop(r);
            session.write_text(&["what is a gpu"], &EmbedOptions::default()).unwrap();
            let r = session.run(&Default::default()).unwrap();
            assert!(cosine(&with_role, &read_f32(&r, 0)) < 0.99999, "the query prefix must change the vector");
        }
        Err(e) => {
            assert!(query_prefix.is_empty(), "the role was refused, so the bundle declares no query prefix: {e}");
            assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "{e}");
            assert_eq!(e.field(), EmbedOptions::FIELD_PROMPT_ROLE, "the refusal names prompt_role: {e}");
        }
    }
}

#[test]
fn live_cuda_byte_counters_add_up_over_repeated_runs() {
    let Some(live) = cuda() else { return };
    let Some(embed_dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let Some(rerank_dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
    assert!(live.has_cap(abi::TURBO_CAP_DEVICE_RESULT), "the CUDA device keeps results on the device");

    let model = live.ctx.load_model(&embed_dir, &ModelDesc::default()).expect("load MiniLM");
    let session = model.create_session(&SessionDesc { max_batch: 4, max_seq: 64, ..Default::default() }).unwrap();
    let texts = ["the cat sat on the mat", "an entirely different sentence about gpus"];
    session.write_text(&texts, &EmbedOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(0).unwrap().placement(), Placement::Device, "embeddings stay on the device");
    let mut bytes = vec![0u8; r.output(0).unwrap().logical_bytes().unwrap() as usize];
    r.read(0, &mut bytes).unwrap();
    drop(r);
    let one = session.stats().unwrap();
    assert_eq!(one.runs, 1, "{one:?}");
    assert!(one.h2d_bytes > 0, "the token rows were uploaded: {one:?}");
    assert_eq!(one.d2h_bytes, 0, "an embedding run moves nothing back; the read is a separate copy: {one:?}");
    // The identical second run uploads exactly the same bytes again: the
    // counters accumulate per run and nothing is cached behind the caller's
    // back.
    session.write_text(&texts, &EmbedOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    drop(r);
    let two = session.stats().unwrap();
    assert_eq!(two.runs, 2, "{two:?}");
    assert_eq!(two.h2d_bytes, 2 * one.h2d_bytes, "the same input uploads the same bytes: {one:?} then {two:?}");
    assert_eq!(two.d2h_bytes, 0, "{two:?}");

    // Sorting a rerank result is the one path that reads the scores back
    // inside the run, and it accounts for exactly one f32 per row.
    let model = live.ctx.load_model(&rerank_dir, &ModelDesc::default()).expect("load reranker");
    let session = model.create_session(&SessionDesc { max_batch: 4, max_seq: 64, ..Default::default() }).unwrap();
    let docs = ["berlin is a large city", "the eiffel tower is in paris", "new york is populous"];
    session.write_pairs("how large is berlin", &docs, &RerankOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    drop(r);
    let unsorted = session.stats().unwrap();
    assert_eq!(unsorted.d2h_bytes, 0, "an unsorted rerank leaves the scores on the device: {unsorted:?}");
    session
        .write_pairs("how large is berlin", &docs, &RerankOptions { return_sorted: true, ..Default::default() })
        .unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(read_f32(&r, 0).len(), docs.len());
    drop(r);
    let sorted = session.stats().unwrap();
    assert_eq!(
        sorted.d2h_bytes,
        (docs.len() * 4) as u64,
        "sorting reads back one f32 per row and counts it: {unsorted:?} then {sorted:?}"
    );
}

#[test]
fn live_cuda_two_models_run_concurrently_on_one_device() {
    let Some(live) = cuda() else { return };
    let Some(embed_dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let Some(classify_dir) = bundle_for(&live, Task::Classify, "TURBO_LIVE_CLASSIFY_BUNDLE") else { return };
    let text = "I absolutely loved this movie, it was wonderful.";

    let embed_model = live.ctx.load_model(&embed_dir, &ModelDesc::default()).expect("load MiniLM");
    let classify_model = live.ctx.load_model(&classify_dir, &ModelDesc::default()).expect("load classifier");
    let embed_session =
        embed_model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    let classify_session =
        classify_model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();

    // Alone first, so the concurrent runs have something exact to match.
    embed_session.write_text(&[text], &EmbedOptions::default()).unwrap();
    let r = embed_session.run(&Default::default()).unwrap();
    let embed_alone = read_f32(&r, 0);
    drop(r);
    classify_session.write_text_classify(&[text], &ClassifyOptions::default()).unwrap();
    let r = classify_session.run(&Default::default()).unwrap();
    let classify_alone = read_f32(&r, 0);
    drop(r);

    let embed_session = Arc::new(embed_session);
    let classify_session = Arc::new(classify_session);
    let a = {
        let s = embed_session.clone();
        std::thread::spawn(move || {
            let mut last = Vec::new();
            for _ in 0..8 {
                s.write_text(&[text], &EmbedOptions::default()).expect("write_text");
                let r = s.run(&Default::default()).expect("embed run");
                last = read_f32(&r, 0);
            }
            last
        })
    };
    let b = {
        let s = classify_session.clone();
        std::thread::spawn(move || {
            let mut last = Vec::new();
            for _ in 0..8 {
                s.write_text_classify(&[text], &ClassifyOptions::default()).expect("write_text_classify");
                let r = s.run(&Default::default()).expect("classify run");
                last = read_f32(&r, 0);
            }
            last
        })
    };
    let embed_together = a.join().expect("the embedding thread");
    let classify_together = b.join().expect("the classification thread");
    assert!(
        cosine(&embed_alone, &embed_together) > 0.99999,
        "a concurrent classifier must not disturb the embedding: {embed_alone:?} vs {embed_together:?}"
    );
    for (alone, together) in classify_alone.iter().zip(&classify_together) {
        assert!(
            (alone - together).abs() < 1e-4,
            "a concurrent embedder must not disturb the classifier: {classify_alone:?} vs {classify_together:?}"
        );
    }
}

#[test]
fn live_cuda_a_device_ordinal_that_does_not_exist_is_never_a_fallback() {
    let Some(live) = cuda() else { return };
    let Some(dir) = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE") else { return };
    let runtime = turbo::create_runtime(RuntimeDesc { provider_paths: vec![lib()], ..Default::default() })
        .expect("load the cuda provider");
    assert!(runtime.failures().is_empty(), "provider failures: {:?}", runtime.failures());
    let ordinals: Vec<u32> =
        runtime.devices().iter().filter(|d| d.info.provider_id == "cuda").map(|d| d.info.ordinal).collect();
    assert!(!ordinals.is_empty(), "the cuda provider enumerated no device");
    eprintln!("cuda ordinals on this machine: {ordinals:?}");
    let select = |ordinal: u32| {
        runtime.select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: "cuda".to_string(),
            ordinal,
            ..Default::default()
        })
    };
    let embed = |index: u32| {
        let ctx = Context::create(runtime.clone(), index, &ContextDesc::default()).expect("context");
        let model = ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
        let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).unwrap();
        session.write_text(&["the cat sat on the mat"], &EmbedOptions::default()).unwrap();
        let r = session.run(&Default::default()).unwrap();
        read_f32(&r, 0)
    };
    let first = embed(select(ordinals[0]).expect("select the first cuda device"));

    match ordinals.get(1) {
        // A second GPU must give the same vectors as the first: same
        // bundle, same FP32 kernels, a different device.
        Some(&second) => {
            let other = embed(select(second).expect("select the second cuda device"));
            let c = cosine(&first, &other);
            eprintln!("cosine between ordinal {} and ordinal {second}: {c:.6}", ordinals[0]);
            assert!(c > 0.9995, "the second device must agree with the first: {c}");
        }
        // One device: asking for a second is an error naming it, never a
        // quiet return of the first.
        None => {
            let missing = ordinals[0] + 1;
            let e = select(missing).expect_err("a cuda ordinal that does not exist must not resolve");
            assert_eq!(e.code(), abi::TURBO_E_DEVICE_NOT_FOUND, "{e}");
        }
    }
}
