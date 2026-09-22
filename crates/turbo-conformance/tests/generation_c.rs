//! Group `generation`, C ABI layer: the pull iterator through
//! `turbo_generation_*`, and the push form `turbo_generate`, which is
//! declared from P0 and must say `TURBO_E_NOT_IMPLEMENTED` until P6.

use std::ffi::c_void;
use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, needs, BundleKind, Target};

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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
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
    needs!(t, Generative);
    if t.caps() & TURBO_CAP_OPT_GEN_SEED == 0 {
        println!("generation_seed: device does not advertise TURBO_CAP_OPT_GEN_SEED");
        return;
    }
    let f = Gen::new(&t);
    // Temperature 2 over 16 tokens: two seeds drawing the same sequence
    // from a real model has a probability far below any flake rate worth
    // naming (6 tokens at temperature 1 coincided on Qwen2.5-0.5B).
    let run = |seed: u64| {
        let mut gd = c::generate_desc();
        gd.max_new_tokens = 16;
        gd.has_seed = 1;
        gd.seed = seed;
        gd.temperature = 2.0;
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
    needs!(t, Generative);
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
    // The matched string is never delivered; it was the whole first piece.
    assert!(!text.contains(&first), "the stop string must not be delivered: text {text:?}, stop {first:?}");
    assert_eq!(text, "", "nothing precedes a stop that is the first piece: {text:?}");
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
    needs!(t, Generative);
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

/// Collects what the push form delivers.
struct Pushed {
    tokens: Vec<i32>,
    chunks: u32,
    done: bool,
    finish: u32,
    stop_after: u32,
}

unsafe extern "C" fn collect(user_data: *mut c_void, chunk: *const turbo_generation_chunk) -> u32 {
    // SAFETY: the library passes the Pushed pointer we gave it and a chunk
    // valid for the duration of this call.
    let p = unsafe { &mut *user_data.cast::<Pushed>() };
    let c = unsafe { &*chunk };
    assert_eq!(c.struct_size as usize, std::mem::size_of::<turbo_generation_chunk>());
    if c.n_tokens > 0 {
        p.tokens.extend_from_slice(unsafe { std::slice::from_raw_parts(c.tokens, c.n_tokens as usize) });
    }
    p.chunks += 1;
    p.done = c.done != 0;
    p.finish = c.finish_reason;
    if p.stop_after != 0 && p.chunks >= p.stop_after {
        TURBO_STREAM_STOP
    } else {
        TURBO_STREAM_CONTINUE
    }
}

#[test]
fn generation_push_form_delivers_the_pull_forms_tokens() {
    let t = Target::from_env();
    needs!(t, Generative);
    let f = Gen::new(&t);
    let mut e = c::err();
    let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 6;
    // Pull first, for the reference sequence.
    let mut g = ptr::null_mut();
    assert_rc!(unsafe { turbo_generation_create(f.model, &gd, &mut g, &mut e) }, TURBO_OK, e);
    assert_rc!(unsafe { turbo_generation_prompt(g, &msg, 1, &mut e) }, TURBO_OK, e);
    let mut pulled = Vec::new();
    loop {
        let mut chunk =
            turbo_generation_chunk { struct_size: ssz::<turbo_generation_chunk>(), ..unsafe { std::mem::zeroed() } };
        assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
        pulled.extend_from_slice(unsafe { std::slice::from_raw_parts(chunk.tokens, chunk.n_tokens as usize) });
        if chunk.done != 0 {
            break;
        }
    }
    unsafe { turbo_generation_release(g) };
    // Push: the same tokens, in order, through the callback.
    let mut pushed = Pushed { tokens: Vec::new(), chunks: 0, done: false, finish: 0, stop_after: 0 };
    let rc =
        unsafe { turbo_generate(f.model, &gd, &msg, 1, Some(collect), (&mut pushed as *mut Pushed).cast(), &mut e) };
    assert_rc!(rc, TURBO_OK, e);
    assert_eq!(pushed.tokens, pulled, "push and pull must produce one token sequence");
    assert!(pushed.done, "the last chunk is the finished one");
    assert_ne!(pushed.finish, TURBO_FINISH_NONE);
    // TURBO_STREAM_STOP ends the stream after the chunk that returned it.
    let mut early = Pushed { tokens: Vec::new(), chunks: 0, done: false, finish: 0, stop_after: 2 };
    let rc =
        unsafe { turbo_generate(f.model, &gd, &msg, 1, Some(collect), (&mut early as *mut Pushed).cast(), &mut e) };
    assert_rc!(rc, TURBO_OK, e);
    assert_eq!(early.chunks, 2);
    assert_eq!(early.tokens, pulled[..early.tokens.len()]);
    assert!(!early.done, "a stopped stream never reports a finished chunk");
    // Argument checks: NULL callback, NULL model.
    let rc = unsafe { turbo_generate(f.model, &gd, &msg, 1, None, ptr::null_mut(), &mut e) };
    assert_eq!(rc, TURBO_E_INVALID_ARGUMENT, "{}", c::message(&e));
    let rc = unsafe { turbo_generate(ptr::null_mut(), &gd, &msg, 1, Some(collect), ptr::null_mut(), &mut e) };
    assert_eq!(rc, TURBO_E_INVALID_HANDLE);
}

#[test]
fn generation_on_a_non_generative_model_is_unsupported_task() {
    let t = Target::from_env();
    needs!(t, Embedding, Reranker, Classifier, Generic);
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

/// Raw handles are pointers, which are not `Send` by default. A generation
/// handle is usable from any thread (PLAN.md section 4.6; the operations on
/// it are single-owner, the handle is not), so the suite wraps it exactly as
/// a binding would to hand it to a cancel thread.
struct Shared<T>(*mut T);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
// A raw pointer is `Copy` whatever it points at; the derive would add a
// needless `T: Copy` bound that opaque handle types cannot satisfy.
impl<T> Copy for Shared<T> {}
// SAFETY: `turbo_generation_cancel` is documented as callable from any
// thread at any time; it records the request and the next step honors it.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

#[test]
fn generation_cancel_from_another_thread_is_never_busy() {
    let t = Target::from_env();
    needs!(t, Generative);
    let f = Gen::new(&t);
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 64;
    let g = f.prompted(&gd);
    let mut e = c::err();
    let mut chunk = c::chunk();
    // Thread A takes one chunk, then hands the handle to thread B.
    // SAFETY: valid generation handle and chunk.
    assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
    assert_eq!(chunk.done, 0, "the stream ended before it could be cancelled");

    let shared = Shared(g);
    let rc = std::thread::spawn(move || {
        let shared = shared;
        let mut e = c::err();
        // SAFETY: the generation outlives the thread; cancel is thread-safe.
        let rc = unsafe { turbo_generation_cancel(shared.0, &mut e) };
        (rc, c::message(&e))
    })
    .join()
    .expect("cancel thread");
    assert_eq!(
        rc.0,
        TURBO_OK,
        "a cancel from another thread must be TURBO_OK, never {}: {}",
        c::status_name(rc.0),
        rc.1
    );

    // SAFETY: valid generation handle and chunk.
    unsafe {
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_OK, e);
        assert_eq!(chunk.done, 1, "a cancel from another thread must end the stream on the very next step");
        assert_eq!(chunk.finish_reason, TURBO_FINISH_CANCELLED, "got {}", chunk.finish_reason);
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_E_INVALID_STATE, e);
        turbo_generation_release(g);
    }
}

#[test]
fn generation_a_chunk_with_no_tokens_nulls_its_arrays() {
    // `turbo_generation_chunk` documents `tokens` as the new token ids and
    // `logprobs` as "Logprobs, or NULL": a chunk that carries neither must
    // hand back NULL for both rather than a stale or dangling pointer, so a
    // binding can branch on the pointer as well as on the count.
    let t = Target::from_env();
    needs!(t, Generative);
    let f = Gen::new(&t);
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 8;
    if t.caps() & TURBO_CAP_OPT_GEN_LOGPROBS != 0 {
        // Ask for logprobs so the empty chunk is empty because it has no
        // tokens, not because the whole generation was never asked for any.
        gd.logprobs = 2;
    }
    let g = f.prompted(&gd);
    let mut e = c::err();

    // Poison both pointers: the library must overwrite them with NULL, not
    // leave whatever the caller had there.
    let mut chunk = c::chunk();
    chunk.tokens = 8usize as *const i32;
    chunk.logprobs = 8usize as *const f32;

    // Take one chunk, cancel, and read the chunk that reports it: the mock
    // emits no new tokens on a cancelled stream.
    // SAFETY: valid generation handle and chunk throughout.
    unsafe {
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_OK, e);
        assert!(chunk.n_tokens > 0, "the first chunk carries a token");
        assert!(!chunk.tokens.is_null(), "n_tokens > 0 needs a token array");
        chunk.tokens = 8usize as *const i32;
        chunk.logprobs = 8usize as *const f32;
        assert_rc!(turbo_generation_cancel(g, &mut e), TURBO_OK, e);
        assert_rc!(turbo_generation_step(g, &mut chunk, &mut e), TURBO_OK, e);
    }
    assert_eq!(chunk.done, 1, "cancel is reported on the next chunk");
    assert_eq!(chunk.finish_reason, TURBO_FINISH_CANCELLED);
    assert_eq!(chunk.n_tokens, 0, "a cancelled stream emits no new tokens");
    assert!(chunk.tokens.is_null(), "n_tokens == 0 must carry a NULL token array, got {:p}", chunk.tokens);
    assert_eq!(chunk.n_logprobs, 0, "no tokens means no logprobs");
    assert!(chunk.logprobs.is_null(), "n_logprobs == 0 must carry a NULL logprob array, got {:p}", chunk.logprobs);
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };

    // The same rule on every chunk of an ordinary stream, in case this
    // provider ever yields an empty one mid-flight.
    let g = f.prompted(&gd);
    let mut chunk = c::chunk();
    let mut empty = 0;
    loop {
        chunk.tokens = 8usize as *const i32;
        chunk.logprobs = 8usize as *const f32;
        // SAFETY: valid generation handle and chunk.
        assert_rc!(unsafe { turbo_generation_step(g, &mut chunk, &mut e) }, TURBO_OK, e);
        if chunk.n_tokens == 0 {
            empty += 1;
            assert!(chunk.tokens.is_null(), "an empty chunk must carry a NULL token array");
        } else {
            assert!(!chunk.tokens.is_null(), "n_tokens {} needs a token array", chunk.n_tokens);
        }
        if chunk.n_logprobs == 0 {
            assert!(chunk.logprobs.is_null(), "an empty logprob list must be NULL");
        } else {
            assert!(!chunk.logprobs.is_null(), "n_logprobs {} needs an array", chunk.n_logprobs);
        }
        if chunk.done != 0 {
            break;
        }
    }
    println!("generation_empty_chunk_arrays: {empty} chunk(s) carried no tokens");
    // SAFETY: released once.
    unsafe { turbo_generation_release(g) };
}
