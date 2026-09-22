//! Group `generation`, Rust layer: the pull iterator (PLAN.md section 5).
//! Prompt, step until done, finish reasons, cancellation, seeds, logprobs,
//! and the state machine around all of it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use turbo::abi::*;
use turbo::provider::{GenerateDesc, Message, SessionDesc};
use turbo::types::FinishReason;
use turbo_conformance::{assert_err, BundleKind, Target};

const PROMPT: [Message<'static>; 1] = [Message { role: "user", content: "say something" }];

/// Drain a generation, returning (tokens, text, finish reason, chunk count).
fn drain(generation: &turbo::handles::Generation, limit: usize) -> (Vec<i32>, String, FinishReason, usize) {
    let mut tokens = Vec::new();
    let mut text = String::new();
    let mut chunks = 0;
    loop {
        let chunk = generation.step().expect("step");
        tokens.extend_from_slice(&chunk.tokens);
        text.push_str(&chunk.text);
        chunks += 1;
        let done = chunk.done;
        let reason = chunk.finish_reason;
        let generated = chunk.generated_tokens;
        drop(chunk);
        if done {
            assert_ne!(reason, FinishReason::None, "a finished chunk must say why");
            assert!(generated as usize >= tokens.len() || generated == 0, "generated_tokens went backwards");
            return (tokens, text, reason, chunks);
        }
        assert_eq!(reason, FinishReason::None, "an unfinished chunk must not name a finish reason");
        assert!(chunks < limit, "the generation did not finish within {limit} steps");
    }
}

#[test]
fn generation_prompt_then_step_until_done() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 6, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    let (tokens, text, reason, chunks) = drain(&generation, 64);
    assert!(!tokens.is_empty(), "the stream produced no tokens");
    assert!(tokens.len() <= 6, "max_new_tokens was exceeded: {} tokens", tokens.len());
    assert!(!text.is_empty(), "the stream produced no text");
    assert_eq!(reason, FinishReason::Length);
    assert!(chunks >= 1);
    for id in &tokens {
        assert!(*id >= 0, "a token id is never negative: {id}");
        assert!(
            (*id as u32) < model.info().vocab_size || model.info().vocab_size == 0,
            "token {id} is outside the vocabulary"
        );
    }
}

#[test]
fn generation_first_chunk_reports_the_prompt_size() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 2, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    let chunk = generation.step().expect("step");
    assert!(chunk.prompt_tokens > 0, "the first chunk must report the prompt token count");
    assert!(
        chunk.prompt_tokens <= model.info().max_seq,
        "prompt_tokens {} exceeds max_seq {}",
        chunk.prompt_tokens,
        model.info().max_seq
    );
    assert_eq!(chunk.generated_tokens, chunk.tokens.len() as u32, "generated_tokens counts the stream so far");
    assert_eq!(chunk.sequence, 0, "a single-sequence generation uses sequence 0");
}

#[test]
fn generation_finish_reason_is_length_at_max_new_tokens() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    for max in [1u32, 2, 5] {
        let desc = GenerateDesc { max_new_tokens: max, ..Default::default() };
        let generation = model.create_generation(&desc).expect("generation");
        generation.prompt(&PROMPT).expect("prompt");
        let (tokens, _, reason, _) = drain(&generation, 64);
        assert!(tokens.len() <= max as usize, "max_new_tokens {max} produced {} tokens", tokens.len());
        assert!(
            reason == FinishReason::Length || reason == FinishReason::Eos,
            "expected LENGTH (or an early EOS), got {reason:?} for max_new_tokens {max}"
        );
        if reason == FinishReason::Length {
            assert_eq!(tokens.len(), max as usize, "LENGTH must mean the budget was used up");
        }
    }
}

#[test]
fn generation_min_new_tokens_above_max_is_invalid_argument() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 2, min_new_tokens: 5, ..Default::default() };
    assert_err!(model.create_generation(&desc), TURBO_E_INVALID_ARGUMENT, field = 3);
}

#[test]
fn generation_sampling_parameters_outside_their_range_are_invalid_argument() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let cases: [(u32, GenerateDesc); 4] = [
        (5, GenerateDesc { temperature: 5.0, ..Default::default() }),
        (5, GenerateDesc { temperature: f32::NAN, ..Default::default() }),
        (7, GenerateDesc { top_p: 1.5, ..Default::default() }),
        (8, GenerateDesc { min_p: -0.5, ..Default::default() }),
    ];
    for (field, desc) in cases {
        assert_err!(model.create_generation(&desc), TURBO_E_INVALID_ARGUMENT, field = field);
    }
}

#[test]
fn generation_stop_string_finishes_with_stop() {
    let t = Target::from_env();
    if !t.has(TURBO_CAP_OPT_GEN_STOP_STRINGS) {
        println!("generation_stop_string: device does not advertise TURBO_CAP_OPT_GEN_STOP_STRINGS");
        return;
    }
    let model = t.model(BundleKind::Generative);
    // Learn what this model emits first, then stop on exactly that text.
    let desc = GenerateDesc { max_new_tokens: 4, ..Default::default() };
    let probe = model.create_generation(&desc).expect("generation");
    probe.prompt(&PROMPT).expect("prompt");
    let first_text = {
        let chunk = probe.step().expect("step");
        chunk.text.clone()
    };
    assert!(!first_text.is_empty(), "the model emits no text to stop on");
    drop(probe);

    let stopping = GenerateDesc { max_new_tokens: 4, stop: vec![first_text.clone()], ..Default::default() };
    let generation = model.create_generation(&stopping).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    let (tokens, text, reason, chunks) = drain(&generation, 64);
    assert_eq!(reason, FinishReason::Stop, "the stop string {first_text:?} must end the stream");
    assert_eq!(chunks, 1, "the stop must be detected on the chunk that completes it");
    assert!(text.ends_with(&first_text), "the stream must stop right after the stop string: {text:?}");
    assert_eq!(tokens.len(), 1);

    // A stop string that never matches does not shorten the stream.
    let never = GenerateDesc { max_new_tokens: 4, stop: vec!["\u{1F980}never".into()], ..Default::default() };
    let generation = model.create_generation(&never).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    let (tokens, _, reason, _) = drain(&generation, 64);
    assert_ne!(reason, FinishReason::Stop, "an unmatched stop string must not stop the stream");
    assert_eq!(tokens.len(), 4);
}

#[test]
fn generation_the_same_seed_reproduces_the_same_tokens() {
    let t = Target::from_env();
    if !t.has(TURBO_CAP_OPT_GEN_SEED) {
        println!("generation_seed: device does not advertise TURBO_CAP_OPT_GEN_SEED");
        return;
    }
    let model = t.model(BundleKind::Generative);
    let run = |seed: u64| {
        let desc = GenerateDesc { max_new_tokens: 8, seed: Some(seed), temperature: 1.0, ..Default::default() };
        let generation = model.create_generation(&desc).expect("generation");
        generation.prompt(&PROMPT).expect("prompt");
        drain(&generation, 64).0
    };
    let a = run(12345);
    let b = run(12345);
    assert_eq!(a, b, "the same seed must reproduce the same token sequence");
    let c = run(67890);
    assert_ne!(a, c, "a different seed must change the token sequence (mock: seed feeds the hash chain)");
}

#[test]
fn generation_cancel_yields_one_cancelled_chunk_then_invalid_state() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 64, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    // Take one chunk, then cancel mid-stream.
    {
        let chunk = generation.step().expect("step");
        assert!(!chunk.done, "the stream ended before it could be cancelled");
    }
    generation.cancel().expect("cancel");
    {
        let chunk = generation.step().expect("the chunk that reports the cancellation");
        assert!(chunk.done, "cancel must be reported on the next chunk");
        assert_eq!(chunk.finish_reason, FinishReason::Cancelled);
    }
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
    // Cancelling again is harmless.
    generation.cancel().expect("cancel is idempotent");
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
}

#[test]
fn generation_cancel_before_any_step_is_reported_on_the_first_chunk() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let generation = model.create_generation(&GenerateDesc::default()).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    generation.cancel().expect("cancel");
    let chunk = generation.step().expect("step");
    assert!(chunk.done);
    assert_eq!(chunk.finish_reason, FinishReason::Cancelled);
    assert!(chunk.tokens.is_empty(), "a cancelled stream must not emit new tokens");
}

#[test]
fn generation_step_before_prompt_is_invalid_state() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let generation = model.create_generation(&GenerateDesc::default()).expect("generation");
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
    // Cancelling an unprompted generation is allowed; stepping is still refused.
    generation.cancel().expect("cancel");
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
}

#[test]
fn generation_prompting_twice_is_invalid_state() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let generation = model.create_generation(&GenerateDesc::default()).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    assert_err!(generation.prompt(&PROMPT), TURBO_E_INVALID_STATE);
    assert_err!(generation.prompt_tokens(&[1, 2, 3]), TURBO_E_INVALID_STATE);

    let other = model.create_generation(&GenerateDesc::default()).expect("generation");
    other.prompt_tokens(&[1, 2, 3]).expect("prompt tokens");
    assert_err!(other.prompt_tokens(&[1, 2, 3]), TURBO_E_INVALID_STATE);
    assert_err!(other.prompt(&PROMPT), TURBO_E_INVALID_STATE);
}

#[test]
fn generation_empty_prompts_are_invalid_argument() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let generation = model.create_generation(&GenerateDesc::default()).expect("generation");
    assert_err!(generation.prompt(&[]), TURBO_E_INVALID_ARGUMENT);
    assert_err!(generation.prompt_tokens(&[]), TURBO_E_INVALID_ARGUMENT);
    let no_role = [Message { role: "", content: "hello" }];
    assert_err!(generation.prompt(&no_role), TURBO_E_INVALID_ARGUMENT);
    // The generation is still unprompted and usable.
    generation.prompt(&PROMPT).expect("prompt");
}

#[test]
fn generation_prompt_tokens_outside_the_vocabulary_are_invalid_argument() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let vocab = model.info().vocab_size;
    let generation = model.create_generation(&GenerateDesc::default()).expect("generation");
    assert_err!(generation.prompt_tokens(&[1, -5]), TURBO_E_INVALID_ARGUMENT);
    if vocab != 0 {
        assert_err!(generation.prompt_tokens(&[1, vocab as i32]), TURBO_E_INVALID_ARGUMENT);
        assert_err!(generation.prompt_tokens(&[i32::MAX]), TURBO_E_INVALID_ARGUMENT);
    }
    // A valid prompt still works afterwards: the rejection left no state behind.
    generation.prompt_tokens(&[1, 2, 3]).expect("prompt tokens");
    let (tokens, _, _, _) = drain(&generation, 64);
    assert!(!tokens.is_empty());
}

#[test]
fn generation_a_prompt_longer_than_max_seq_is_capacity() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let generation = model.create_generation(&GenerateDesc::default()).expect("generation");
    let too_many: Vec<i32> = vec![3; model.info().max_seq as usize + 1];
    assert_err!(generation.prompt_tokens(&too_many), TURBO_E_CAPACITY);
}

#[test]
fn generation_logprobs_count_matches_the_token_count() {
    let t = Target::from_env();
    if !t.has(TURBO_CAP_OPT_GEN_LOGPROBS) {
        println!("generation_logprobs: device does not advertise TURBO_CAP_OPT_GEN_LOGPROBS");
        return;
    }
    let model = t.model(BundleKind::Generative);
    for k in [1u32, 3] {
        let desc = GenerateDesc { max_new_tokens: 4, logprobs: k, ..Default::default() };
        let generation = model.create_generation(&desc).expect("generation");
        generation.prompt(&PROMPT).expect("prompt");
        loop {
            let chunk = generation.step().expect("step");
            assert_eq!(
                chunk.logprobs.len(),
                chunk.tokens.len() * k as usize,
                "logprobs must be n_tokens * logprobs ({} tokens, {k} per token)",
                chunk.tokens.len()
            );
            for p in &chunk.logprobs {
                assert!(p.is_finite() && *p <= 0.0, "a logprob must be a finite non-positive number: {p}");
            }
            if chunk.done {
                break;
            }
        }
    }
    // Without the option there are no logprobs at all.
    let desc = GenerateDesc { max_new_tokens: 2, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    let chunk = generation.step().expect("step");
    assert!(chunk.logprobs.is_empty(), "logprobs = 0 must return none");
}

#[test]
fn generation_on_a_non_generative_model_is_unsupported_task() {
    let t = Target::from_env();
    for kind in [BundleKind::Embedding, BundleKind::Reranker, BundleKind::Classifier, BundleKind::Generic] {
        let model = t.model(kind);
        assert_err!(model.create_generation(&GenerateDesc::default()), TURBO_E_UNSUPPORTED_TASK);
    }
    // And a generative model has no session-level task.
    let generative = t.model(BundleKind::Generative);
    let session = generative.create_session(&SessionDesc::default()).expect("session");
    assert_err!(session.run(&turbo::provider::RunOptions::default()), TURBO_E_INVALID_STATE);
}

#[test]
fn generation_echo_is_accepted_and_still_terminates() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 3, echo: true, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");
    let (tokens, text, reason, _) = drain(&generation, 64);
    assert_eq!(tokens.len(), 3);
    assert_eq!(reason, FinishReason::Length);
    let plain = {
        let desc = GenerateDesc { max_new_tokens: 3, ..Default::default() };
        let generation = model.create_generation(&desc).expect("generation");
        generation.prompt(&PROMPT).expect("prompt");
        drain(&generation, 64).1
    };
    assert!(text.len() >= plain.len(), "echo must not shorten the output text");
}

#[test]
fn generation_unknown_provider_options_are_rejected() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc {
        options: turbo::handles::options_from_pairs([("definitely_not_an_option", "1")]),
        ..Default::default()
    };
    assert_err!(model.create_generation(&desc), TURBO_E_INVALID_ARGUMENT, field = 1);
}

#[test]
fn generation_cancel_from_another_thread_is_never_busy() {
    // `Generation` is `Send + Sync` (crates/turbo-core/src/handles.rs): the
    // state sits behind a mutex and the cancel request is an atomic, so the
    // handle itself crosses threads and only the operations are
    // single-owner. `create_generation` hands back an `Arc`, which is what a
    // binding shares between a decode loop and a cancel button. Cancel is
    // the one operation that must never report TURBO_E_BUSY.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<turbo::handles::Generation>();

    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 64, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");

    // Thread A steps once and keeps the chunk borrowed, so thread B's cancel
    // runs while the chunk storage is still leased out.
    let chunk = generation.step().expect("step");
    assert!(!chunk.done, "the stream ended before it could be cancelled");

    let shared = Arc::clone(&generation);
    let result = thread::spawn(move || shared.cancel()).join().expect("cancel thread");
    if let Err(e) = result {
        panic!(
            "cancel from another thread must be Ok, never {} ({}): {}",
            turbo::error::status_name(e.code()),
            e.code(),
            e.message()
        );
    }
    drop(chunk);

    let chunk = generation.step().expect("the chunk that reports the cancellation");
    assert!(chunk.done, "a cancel from another thread must end the stream on the very next step");
    assert_eq!(chunk.finish_reason, FinishReason::Cancelled, "the stream must say it was cancelled, not why else");
    drop(chunk);
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
}

#[test]
fn generation_cancel_while_another_thread_steps_never_errors() {
    // A cancel never touches the state lock: it is recorded with an atomic
    // and honored at the start of the next step (crates/turbo-core/src/
    // handles.rs). The contract this pins: the cancel returns Ok, a step
    // racing it never fails, and the stream ends with `Cancelled` within
    // two further steps - the one already in flight, then the one that
    // reports the cancellation.
    const GRACE: usize = 2;
    let t = Target::from_env();
    if !t.has(TURBO_CAP_OPT_GEN_MIN_TOKENS) {
        println!("generation_concurrent_cancel: device does not advertise TURBO_CAP_OPT_GEN_MIN_TOKENS");
        return;
    }
    let model = t.model(BundleKind::Generative);
    // A budget the loop cannot exhaust in the time a cancel takes, with the
    // EOS floor raised to the same number so an early end-of-sequence cannot
    // be mistaken for the cancel landing: only a cancel can stop this stream.
    const BUDGET: u32 = 100_000;
    let desc = GenerateDesc { max_new_tokens: BUDGET, min_new_tokens: BUDGET, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&PROMPT).expect("prompt");

    let steps = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));
    let (reason, total, at_cancel) = thread::scope(|scope| {
        let canceller = {
            let generation = Arc::clone(&generation);
            let steps = Arc::clone(&steps);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                // Wait until the stepping thread is under way, so the cancel
                // is delivered into a running loop rather than before it.
                while steps.load(Ordering::SeqCst) == 0 {
                    std::hint::spin_loop();
                }
                let result = generation.cancel();
                let seen = steps.load(Ordering::SeqCst);
                (result, seen)
            })
        };
        barrier.wait();
        let mut reason = FinishReason::None;
        for _ in 0..BUDGET as usize + GRACE {
            // A cancel from another thread never makes a step fail: the
            // cancel takes no lock, so there is no instant at which the
            // step could find the state held.
            let chunk = generation.step().unwrap_or_else(|e| panic!("a step racing a cancel must not fail: {e}"));
            let done = chunk.done;
            let finish = chunk.finish_reason;
            drop(chunk);
            steps.fetch_add(1, Ordering::SeqCst);
            if done {
                reason = finish;
                break;
            }
            std::thread::yield_now();
        }
        let (result, seen) = canceller.join().expect("cancel thread");
        if let Err(e) = result {
            panic!(
                "a cancel racing a step must be Ok, never {} ({}): {}",
                turbo::error::status_name(e.code()),
                e.code(),
                e.message()
            );
        }
        (reason, steps.load(Ordering::SeqCst), seen)
    });

    assert_eq!(reason, FinishReason::Cancelled, "the stream must end with CANCELLED, not {reason:?}");
    assert!(
        total <= at_cancel + GRACE,
        "the cancel returned after step {at_cancel} but the stream ran to step {total}; \
         it must end within {GRACE} further steps"
    );
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
}
