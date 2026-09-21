//! Group `allocation`, Rust layer: `turbo_session_run` never allocates after
//! warmup (PLAN.md section 4.5). Sessions preallocate for their declared max
//! shape, so a steady stream of same-shape runs must not move any counter but
//! `runs`.

use turbo::buffer::BufferDesc;
use turbo::handles::Session;
use turbo::provider::{ClassifyOptions, EmbedOptions, RerankOptions, RunOptions, SessionDesc, SessionStats};
use turbo::types::{DType, Placement};
use turbo_conformance::{BundleKind, Target};

const ITERATIONS: u64 = 100;

fn check_steady_state(name: &str, session: &Session, mut write_and_run: impl FnMut()) {
    // One warmup run, then the counters must stand still.
    write_and_run();
    let after_warmup = session.stats().expect("stats");
    assert_eq!(after_warmup.runs, 1, "{name}: the warmup run must be counted");

    for i in 0..ITERATIONS {
        write_and_run();
        let now = session.stats().expect("stats");
        assert_eq!(now.runs, i + 2, "{name}: run counter drifted at iteration {i}");
    }

    let end = session.stats().expect("stats");
    assert_eq!(end.runs, ITERATIONS + 1, "{name}: runs must count every completed run");
    assert_eq!(end.input_bytes, after_warmup.input_bytes, "{name}: bound input storage must not grow after warmup");
    assert_eq!(end.output_bytes, after_warmup.output_bytes, "{name}: bound output storage must not grow after warmup");
    assert_eq!(
        end.host_allocs,
        after_warmup.host_allocs,
        "{name}: the adapter allocated on the run path ({} allocations)",
        end.host_allocs.saturating_sub(after_warmup.host_allocs)
    );
    match (after_warmup.provider_allocs, end.provider_allocs) {
        (Some(start), Some(finish)) => {
            assert_eq!(
                finish,
                start,
                "{name}: the provider allocated {} time(s) across {ITERATIONS} steady-state runs",
                finish - start
            );
            assert_eq!(finish, 0, "{name}: the provider reports {finish} allocation(s) on the run path");
        }
        _ => println!("{name}: the provider does not report its own allocations"),
    }
}

fn report(name: &str, stats: &SessionStats) {
    println!(
        "{name}: runs={} host_allocs={} provider_allocs={:?} h2d={} d2h={} in={} out={}",
        stats.runs,
        stats.host_allocs,
        stats.provider_allocs,
        stats.h2d_bytes,
        stats.d2h_bytes,
        stats.input_bytes,
        stats.output_bytes
    );
}

#[test]
fn allocation_embed_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Embedding);
    check_steady_state("embed", &session, || {
        session.write_text(&["hello world", "a second document"], &EmbedOptions::default()).expect("write");
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
    report("embed", &session.stats().expect("stats"));
}

#[test]
fn allocation_embed_from_tokens_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let seq = (model.info().max_seq as usize).min(8);
    let ids: Vec<i32> = (0..seq).map(|i| (i % 50) as i32 + 3).collect();
    let mask: Vec<i32> = vec![1; seq];
    check_steady_state("embed(tokens)", &session, || {
        let batch = turbo::provider::TokenBatch {
            batch: 1,
            seq: seq as u32,
            row_stride: seq as u32,
            ids: &ids,
            mask: &mask,
            types: None,
        };
        session.write_tokens(&batch).expect("write tokens");
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
}

#[test]
fn allocation_rerank_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Reranker);
    let docs = ["alpha beta", "gamma delta", "epsilon"];
    check_steady_state("rerank", &session, || {
        session.write_pairs("alpha gamma", &docs, &RerankOptions::default()).expect("write pairs");
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
    report("rerank", &session.stats().expect("stats"));
}

#[test]
fn allocation_classify_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Classifier);
    check_steady_state("classify", &session, || {
        session.write_text_classify(&["hello world", "another line"], &ClassifyOptions::default()).expect("write");
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
    report("classify", &session.stats().expect("stats"));
}

#[test]
fn allocation_token_classify_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::TokenClassifier);
    check_steady_state("token_classify", &session, || {
        session
            .write_text_classify(&["Alice went to Paris", "Bob stayed home"], &ClassifyOptions::default())
            .expect("write");
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
    report("token_classify", &session.stats().expect("stats"));
}

#[test]
fn allocation_generic_run_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let x = ctx.alloc(&desc).expect("alloc x");
    let y = ctx.alloc(&desc).expect("alloc y");
    session.bind("x", &x).expect("bind x");
    session.bind("y", &y).expect("bind y");
    check_steady_state("run", &session, || {
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
    report("run", &session.stats().expect("stats"));
}

#[test]
fn allocation_embed_with_a_prompt_role_is_allocation_free_after_warmup() {
    // Every option the device advertises has to be usable on the hot path:
    // PLAN.md section 4.5 requires an allocation-free steady state, and
    // section 4.4 requires an advertised option to be honored without a
    // penalty the caller cannot see. With TURBO_CAP_OPT_PROMPT_ROLE set, the
    // mock builds a fresh `String` for the prefix on every write and counts
    // it in `provider_allocs`, so the counter grows by one per run forever.
    let t = Target::from_env();
    if !t.has(turbo::abi::TURBO_CAP_OPT_PROMPT_ROLE) {
        println!("allocation_embed_with_a_prompt_role: device does not advertise TURBO_CAP_OPT_PROMPT_ROLE");
        return;
    }
    let (_m, session) = t.session(BundleKind::Embedding);
    let opts = EmbedOptions { prompt_role: turbo::types::PromptRole::Query, ..Default::default() };
    check_steady_state("embed(prompt_role)", &session, || {
        session.write_text(&["hello world"], &opts).expect("write");
        let r = session.run(&RunOptions::default()).expect("run");
        drop(r);
    });
}

#[test]
fn allocation_counters_are_reported_and_plausible() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Embedding);
    let before = session.stats().expect("stats");
    assert_eq!(before.runs, 0, "a fresh session has not run");
    assert_eq!(before.host_allocs, 0, "host_allocs starts at zero");
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let after = session.stats().expect("stats");
    assert_eq!(after.runs, 1);
    assert!(after.input_bytes > 0, "a session preallocates its inputs");
    assert!(after.output_bytes > 0, "a session preallocates its outputs");
    assert!(after.d2h_bytes >= before.d2h_bytes, "byte counters never go backwards");
    assert!(after.h2d_bytes >= before.h2d_bytes, "byte counters never go backwards");
    // Stats are readable while a result is leased.
    drop(result);
    // A session whose max shape is smaller reserves less.
    let model = t.model(BundleKind::Embedding);
    let small = model
        .create_session(&SessionDesc { max_batch: 1, max_seq: model.info().max_seq, ..Default::default() })
        .expect("session");
    small.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    small.run(&RunOptions::default()).expect("run");
    let small_stats = small.stats().expect("stats");
    assert!(
        small_stats.input_bytes <= after.input_bytes,
        "a narrower session must not reserve more input storage than a wider one"
    );
}

#[test]
fn allocation_stats_survive_the_release_of_parents() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    session.run(&RunOptions::default()).expect("run");
    drop(model);
    let stats = session.stats().expect("stats after the model is released");
    assert_eq!(stats.runs, 1);
    if let Some(n) = stats.provider_allocs {
        assert_eq!(n, 0, "the first run is the warmup and must not be charged");
    }
}
