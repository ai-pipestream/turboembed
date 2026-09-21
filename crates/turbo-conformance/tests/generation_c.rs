//! Group `generation`, C ABI layer: the pull iterator through
//! `turbo_generation_*`, and the push form `turbo_generate`, which is
//! declared from P0 and must say `TURBO_E_NOT_IMPLEMENTED` until P6.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, BundleKind, Target};

struct Gen {
    _ct: c::CTarget,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
}

impl Gen {
    fn new(t: &Target) -> Self {
        let ct = c::CTarget::new(t);
        let ctx = ct.context();
        let model = ct.model(ctx, BundleKind::Generative);
        Self { _ct: ct, ctx, model }
    }

    fn create(&self, desc: &turbo_generate_desc) -> *mut turbo_generation {
        let mut e = c::err();
        let mut g: *mut turbo_generation = ptr::null_mut();
        // SAFETY: valid model handle; the descriptor's pointers outlive the call.
        let rc = unsafe { turbo_generation_create(self.model, desc, &mut g, &mut e) };
        assert_eq!(rc, TURBO_OK, "turbo_generation_create: {}", c::message(&e));
        g
    }

    fn prompted(&self, desc: &turbo_generate_desc) -> *mut turbo_generation {
        let g = self.create(desc);
        let mut e = c::err();
        let msg = turbo_message { role: c::text("user"), content: c::text("say something") };
        // SAFETY: one readable message that outlives the call.
        assert_rc!(unsafe { turbo_generation_prompt(g, &msg, 1, &mut e) }, TURBO_OK, e);
        g
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        // SAFETY: each handle is released exactly once.
        unsafe {
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
        }
    }
}

/// Drain a generation; returns (tokens, text, finish reason, chunk count).
fn drain(g: *mut turbo_generation, limit: usize) -> (Vec<i32>, String, u32, usize) {
    let mut e = c::err();
    let mut chunk = c::chunk();
    let mut tokens = Vec::new();
    let mut text = String::new();
    let mut chunks = 0;
    loop {
        // SAFETY: valid generation handle and chunk with a correct struct_size.
        assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
        assert_eq!(chunk.struct_size, ssz::<turbo_generation_chunk>(), "the library must not change struct_size");
        tokens.extend(c::chunk_tokens(&chunk));
        text.push_str(&c::chunk_text(&chunk));
        chunks += 1;
        if chunk.done != 0 {
            assert_ne!(chunk.finish_reason, TURBO_FINISH_NONE, "a finished chunk must say why");
            return (tokens, text, chunk.finish_reason, chunks);
        }
        assert_eq!(chunk.finish_reason, TURBO_FINISH_NONE, "an unfinished chunk must not name a reason");
        assert!(chunks < limit, "the generation did not finish within {limit} steps");
    }
}

#[test]
fn generation_prompt_then_step_until_done() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 6;
    let g = f.prompted(&gd);
    let (tokens, text, reason, chunks) = drain(g, 64);
    assert!(!tokens.is_empty() && tokens.len() <= 6);
    assert!(!text.is_empty());
    assert_eq!(reason, TURBO_FINISH_LENGTH);
    assert!(chunks >= 1);
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };
}

#[test]
fn generation_chunk_pointers_are_valid_until_the_next_step() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 4;
    let g = f.prompted(&gd);
    let mut e = c::err();
    let mut chunk = c::chunk();
    // SAFETY: valid generation handle and chunk.
    assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
    assert!(chunk.n_tokens > 0, "the first chunk carries a token");
    assert!(!chunk.tokens.is_null(), "n_tokens > 0 needs a token array");
    assert!(!chunk.text.ptr.is_null() || chunk.text.len == 0, "a non-empty text view needs a pointer");
    let first = c::chunk_tokens(&chunk);
    let text = c::chunk_text(&chunk);
    assert_eq!(first.len(), chunk.n_tokens as usize);
    assert!(std::str::from_utf8(text.as_bytes()).is_ok(), "chunk text must be UTF-8");
    assert!(chunk.prompt_tokens > 0, "the prompt size is reported");
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };
}

#[test]
fn generation_finish_reason_is_length_at_max_new_tokens() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    for max in [1u32, 3] {
        let mut gd = c::generate_desc();
        gd.max_new_tokens = max;
        let g = f.prompted(&gd);
        let (tokens, _, reason, _) = drain(g, 64);
        assert!(tokens.len() <= max as usize);
        assert!(reason == TURBO_FINISH_LENGTH || reason == TURBO_FINISH_EOS, "expected LENGTH or EOS, got {reason}");
        // SAFETY: released once.
        unsafe { turbo_generation_release(g) };
    }
}

#[test]
fn generation_cancel_yields_one_cancelled_chunk_then_invalid_state() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 64;
    let g = f.prompted(&gd);
    let mut e = c::err();
    let mut chunk = c::chunk();
    // SAFETY: valid generation handle and chunk throughout.
    unsafe {
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_OK, e);
        assert_eq!(chunk.done, 0, "the stream ended before it could be cancelled");
        assert_rc!(turbo_generation_cancel(g, &mut e), TURBO_OK, e);
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_OK, e);
        assert_eq!(chunk.done, 1, "cancel is reported on the next chunk");
        assert_eq!(chunk.finish_reason, TURBO_FINISH_CANCELLED);
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_E_INVALID_STATE, e);
        // Cancel after the end is still a no-op, not an error.
        assert_rc!(turbo_generation_cancel(g, &mut e), TURBO_OK, e);
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_E_INVALID_STATE, e);
        turbo_generation_release(g);
    }
}

#[test]
fn generation_step_before_prompt_is_invalid_state() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let g = f.create(&c::generate_desc());
    let mut e = c::err();
    let mut chunk = c::chunk();
    // SAFETY: valid generation handle and chunk.
    unsafe {
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_E_INVALID_STATE, e);
        // A NULL out-chunk and a bad struct_size are refused too.
        assert_rc!(turbo_generation_step(g, ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        let mut big = c::chunk();
        big.struct_size = ssz::<turbo_generation_chunk>() + 8;
        assert_rc!(turbo_generation_step(g, &mut big, &mut e), TURBO_E_INVALID_STRUCT_SIZE, e);
        turbo_generation_release(g);
    }
}

#[test]
fn generation_prompting_twice_is_invalid_state() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let g = f.create(&c::generate_desc());
    let mut e = c::err();
    let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
    let ids = [1i32, 2, 3];
    // SAFETY: valid generation handle; the arrays outlive their calls.
    unsafe {
        assert_rc!(turbo_generation_prompt(g, &msg, 1, &mut e), TURBO_OK, e);
        assert_rc!(turbo_generation_prompt(g, &msg, 1, &mut e), TURBO_E_INVALID_STATE, e);
        assert_rc!(turbo_generation_prompt_tokens(g, ids.as_ptr(), 3, &mut e), TURBO_E_INVALID_STATE, e);
        turbo_generation_release(g);
    }
}

#[test]
fn generation_empty_and_null_prompts_are_invalid_argument() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let g = f.create(&c::generate_desc());
    let mut e = c::err();
    let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
    let ids = [1i32];
    // SAFETY: valid generation handle; the deliberate NULLs must be reported.
    unsafe {
        assert_rc!(turbo_generation_prompt(g, &msg, 0, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_generation_prompt(g, ptr::null(), 1, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_generation_prompt_tokens(g, ids.as_ptr(), 0, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_generation_prompt_tokens(g, ptr::null(), 1, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // An empty role is not a message.
        let no_role = turbo_message { role: c::text(""), content: c::text("hello") };
        assert_rc!(turbo_generation_prompt(g, &no_role, 1, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // Invalid UTF-8 in a message is caught before anything else.
        let bad = [0xffu8, 0xfe];
        let broken = turbo_message { role: c::text("user"), content: turbo_text { ptr: bad.as_ptr().cast(), len: 2 } };
        assert_rc!(turbo_generation_prompt(g, &broken, 1, &mut e), TURBO_E_INVALID_UTF8, e);
        // After every rejection the generation is still unprompted.
        assert_rc!(turbo_generation_prompt(g, &msg, 1, &mut e), TURBO_OK, e);
        turbo_generation_release(g);
    }
}

#[test]
fn generation_prompt_tokens_outside_the_vocabulary_are_invalid_argument() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid model handle and out pointer.
    assert_rc!(unsafe { turbo_model_get_info(f.model, &mut info, &mut e) }, TURBO_OK, e);
    let g = f.create(&c::generate_desc());
    let negative = [1i32, -1];
    let above = [1i32, info.vocab_size as i32];
    let good = [1i32, 2, 3];
    // SAFETY: valid generation handle; each array outlives its call.
    unsafe {
        assert_rc!(turbo_generation_prompt_tokens(g, negative.as_ptr(), 2, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        if info.vocab_size != 0 {
            assert_rc!(turbo_generation_prompt_tokens(g, above.as_ptr(), 2, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        }
        assert_rc!(turbo_generation_prompt_tokens(g, good.as_ptr(), 3, &mut e), TURBO_OK, e);
        turbo_generation_release(g);
    }
}

#[test]
fn generation_logprobs_count_matches_the_token_count() {
    let t = Target::from_env();
    if t.caps() & TURBO_CAP_OPT_GEN_LOGPROBS == 0 {
        println!("generation_logprobs: device does not advertise TURBO_CAP_OPT_GEN_LOGPROBS");
        return;
    }
    let f = Gen::new(&t);
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 3;
    gd.logprobs = 2;
    let g = f.prompted(&gd);
    let mut e = c::err();
    let mut chunk = c::chunk();
    loop {
        // SAFETY: valid generation handle and chunk.
        assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
        assert_eq!(chunk.n_logprobs, chunk.n_tokens * 2, "n_logprobs must be n_tokens * logprobs");
        if chunk.n_logprobs > 0 {
            assert!(!chunk.logprobs.is_null(), "n_logprobs > 0 needs an array");
            // SAFETY: the library promises `n_logprobs` readable floats.
            let values = unsafe { std::slice::from_raw_parts(chunk.logprobs, chunk.n_logprobs as usize) };
            assert!(values.iter().all(|p| p.is_finite() && *p <= 0.0), "logprobs: {values:?}");
        }
        if chunk.done != 0 {
            break;
        }
    }
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };

    // Without the option the array is NULL.
    let mut plain = c::generate_desc();
    plain.max_new_tokens = 1;
    let g = f.prompted(&plain);
    // SAFETY: valid generation handle and chunk.
    assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
    assert_eq!(chunk.n_logprobs, 0);
    assert!(chunk.logprobs.is_null(), "logprobs = 0 must return a NULL array");
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };
}

#[test]
fn generation_the_same_seed_reproduces_the_same_tokens() {
    let t = Target::from_env();
    if t.caps() & TURBO_CAP_OPT_GEN_SEED == 0 {
        println!("generation_seed: device does not advertise TURBO_CAP_OPT_GEN_SEED");
        return;
    }
    let f = Gen::new(&t);
    let run = |seed: u64| {
        let mut gd = c::generate_desc();
        gd.max_new_tokens = 6;
        gd.has_seed = 1;
        gd.seed = seed;
        gd.temperature = 1.0;
        let g = f.prompted(&gd);
        let out = drain(g, 64).0;
        // SAFETY: released once.
        unsafe { turbo_generation_release(g) };
        out
    };
    let a = run(4242);
    assert_eq!(a, run(4242), "the same seed must reproduce the same tokens");
    assert_ne!(a, run(99), "a different seed must change the tokens");

    // `has_seed` is a boolean, not a count.
    let mut bad = c::generate_desc();
    bad.has_seed = 2;
    let mut e = c::err();
    let mut g: *mut turbo_generation = ptr::null_mut();
    // SAFETY: valid model handle and descriptor.
    assert_rc!(
        unsafe { turbo_generation_create(f.model, &bad, &mut g, &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e,
        field = 12
    );
}

#[test]
fn generation_stop_string_finishes_with_stop() {
    let t = Target::from_env();
    if t.caps() & TURBO_CAP_OPT_GEN_STOP_STRINGS == 0 {
        println!("generation_stop_string: device does not advertise TURBO_CAP_OPT_GEN_STOP_STRINGS");
        return;
    }
    let f = Gen::new(&t);
    // Learn the first piece of text, then stop on it.
    let mut probe_desc = c::generate_desc();
    probe_desc.max_new_tokens = 4;
    let probe = f.prompted(&probe_desc);
    let mut e = c::err();
    let mut chunk = c::chunk();
    // SAFETY: valid generation handle and chunk.
    assert_rc!(unsafe { turbo_generation_step(probe, &mut chunk, &mut e) }, TURBO_OK, e);
    let first = c::chunk_text(&chunk);
    // SAFETY: released once, after the text was copied out.
    unsafe { turbo_generation_release(probe) };
    assert!(!first.is_empty());

    let stop = [c::text(&first)];
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 4;
    gd.n_stop = 1;
    gd.stop = stop.as_ptr();
    let g = f.prompted(&gd);
    let (tokens, text, reason, chunks) = drain(g, 64);
    assert_eq!(reason, TURBO_FINISH_STOP, "the stop string must end the stream");
    assert_eq!(chunks, 1);
    assert_eq!(tokens.len(), 1);
    assert!(text.ends_with(&first));
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };

    // A count without an array is an argument error naming its field.
    let mut missing = c::generate_desc();
    missing.n_stop = 1;
    let mut out: *mut turbo_generation = ptr::null_mut();
    // SAFETY: valid model handle; the NULL array is deliberate.
    assert_rc!(unsafe { turbo_generation_create(f.model, &missing, &mut out, &mut e) }, TURBO_E_INVALID_ARGUMENT, e);
}

#[test]
fn generation_descriptor_struct_size_is_validated() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let mut e = c::err();
    let mut g: *mut turbo_generation = ptr::null_mut();
    let mut big = c::generate_desc();
    big.struct_size = ssz::<turbo_generate_desc>() + 8;
    // SAFETY: valid model handle and descriptor.
    unsafe {
        assert_rc!(turbo_generation_create(f.model, &big, &mut g, &mut e), TURBO_E_INVALID_STRUCT_SIZE, e);
        // A NULL descriptor means "model defaults".
        assert_rc!(turbo_generation_create(f.model, ptr::null(), &mut g, &mut e), TURBO_OK, e);
        turbo_generation_release(g);
        // An unknown structured-output constant is an enum error naming its field.
        let mut bad = c::generate_desc();
        bad.structured_kind = 77;
        assert_rc!(turbo_generation_create(f.model, &bad, &mut g, &mut e), TURBO_E_INVALID_ENUM, e, field = 21);
        // `echo` is a boolean.
        let mut echo = c::generate_desc();
        echo.echo = 3;
        assert_rc!(turbo_generation_create(f.model, &echo, &mut g, &mut e), TURBO_E_INVALID_ARGUMENT, e, field = 22);
    }
}

#[test]
fn generation_push_form_is_not_implemented_until_p6() {
    let t = Target::from_env();
    let f = Gen::new(&t);
    let mut e = c::err();
    let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 2;
    // A valid model, a valid descriptor, a valid message: the only reason to
    // fail is that the push form is not implemented yet (PLAN.md P6).
    // SAFETY: valid model handle; the message outlives the call.
    let rc = unsafe { turbo_generate(f.model, &gd, &msg, 1, None, ptr::null_mut(), &mut e) };
    assert_eq!(
        rc,
        TURBO_E_NOT_IMPLEMENTED,
        "turbo_generate must report NOT_IMPLEMENTED until P6 lands it, got {}",
        c::status_name(rc)
    );
    assert!(c::message(&e).contains("turbo_generate"), "the message must name the entry point: {}", c::message(&e));
    // It is still a handle check first.
    // SAFETY: the NULL handle is deliberate.
    let rc = unsafe { turbo_generate(ptr::null_mut(), &gd, &msg, 1, None, ptr::null_mut(), &mut e) };
    assert_eq!(rc, TURBO_E_INVALID_HANDLE);
}

#[test]
fn generation_on_a_non_generative_model_is_unsupported_task() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let mut e = c::err();
    for kind in [BundleKind::Embedding, BundleKind::Reranker, BundleKind::Classifier, BundleKind::Generic] {
        let model = ct.model(ctx, kind);
        let mut g: *mut turbo_generation = ptr::null_mut();
        // SAFETY: valid model handle and out pointer.
        unsafe {
            assert_rc!(turbo_generation_create(model, ptr::null(), &mut g, &mut e), TURBO_E_UNSUPPORTED_TASK, e);
            assert!(g.is_null());
            turbo_model_release(model);
        }
    }
    // SAFETY: released once.
    unsafe { turbo_context_release(ctx) };
    drop(ct);
}
