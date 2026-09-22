//! Group `lifetime`, Rust layer: children retain parents, results lease the
//! session's storage, and releasing a parent in any order never invalidates a
//! child (PLAN.md principle 5).

use std::any::Any;
use std::sync::Arc;

use turbo::abi::*;
use turbo::buffer::BufferDesc;
use turbo::handles::Context;
use turbo::provider::{ContextDesc, EmbedOptions, GenerateDesc, Message, ModelDesc, RunOptions, SessionDesc};
use turbo::types::{DType, Placement};
use turbo_conformance::{assert_err, needs, permutations, read_f32, BundleKind, Target};

const TEXT: &str = "hello world";

fn reference_vector(t: &Target) -> Vec<f32> {
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&[TEXT], &EmbedOptions::default()).expect("write");
    read_f32(&session.run(&RunOptions::default()).expect("run"), 0)
}

#[test]
fn lifetime_releasing_parents_in_every_order_keeps_the_result_readable() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let expected = reference_vector(&t);
    for perm in permutations(4) {
        let (runtime, index) = t.detached();
        let ctx = Context::create(runtime.clone(), index, &ContextDesc::default()).expect("context");
        let model = ctx.load_model(&t.bundle(BundleKind::Embedding), &ModelDesc::default()).expect("model");
        let session = model.create_session(&SessionDesc::default()).expect("session");
        session.write_text(&[TEXT], &EmbedOptions::default()).expect("write");
        let result = session.run(&RunOptions::default()).expect("run");

        let mut slots: Vec<Option<Box<dyn Any>>> =
            vec![Some(Box::new(runtime)), Some(Box::new(ctx)), Some(Box::new(model)), Some(Box::new(session))];
        for &i in &perm {
            slots[i].take();
        }
        assert_eq!(read_f32(&result, 0), expected, "release order {perm:?} changed the result");
        // The result's own metadata is still intact after every parent is gone.
        assert_eq!(result.output(0).expect("output 0").shape[0], 1);
    }
}

#[test]
fn lifetime_releasing_parents_in_every_order_keeps_the_session_usable() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let expected = reference_vector(&t);
    for perm in permutations(3) {
        let (runtime, index) = t.detached();
        let ctx = Context::create(runtime.clone(), index, &ContextDesc::default()).expect("context");
        let model = ctx.load_model(&t.bundle(BundleKind::Embedding), &ModelDesc::default()).expect("model");
        let session = model.create_session(&SessionDesc::default()).expect("session");

        let mut slots: Vec<Option<Box<dyn Any>>> =
            vec![Some(Box::new(runtime)), Some(Box::new(ctx)), Some(Box::new(model))];
        for &i in &perm {
            slots[i].take();
        }
        session.write_text(&[TEXT], &EmbedOptions::default()).expect("write after the parents are released");
        let result = session.run(&RunOptions::default()).expect("run after the parents are released");
        assert_eq!(read_f32(&result, 0), expected, "release order {perm:?} changed the result");
    }
}

#[test]
fn lifetime_generation_outlives_its_model_and_context() {
    let t = Target::from_env();
    needs!(t, Generative);
    let (runtime, index) = t.detached();
    let ctx = Context::create(runtime.clone(), index, &ContextDesc::default()).expect("context");
    let model = ctx.load_model(&t.bundle(BundleKind::Generative), &ModelDesc::default()).expect("model");
    let desc = GenerateDesc { max_new_tokens: 3, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&[Message { role: "user", content: "hello" }]).expect("prompt");
    drop(model);
    drop(ctx);
    drop(runtime);
    let mut steps = 0;
    loop {
        let chunk = generation.step().expect("step after the model was released");
        steps += 1;
        if chunk.done {
            assert_eq!(chunk.generated_tokens, 3);
            break;
        }
        assert!(steps < 10, "generation did not finish");
    }
}

#[test]
fn lifetime_buffer_outlives_its_context() {
    let t = Target::from_env();
    let (runtime, index) = t.detached();
    let ctx = Context::create(runtime.clone(), index, &ContextDesc::default()).expect("context");
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[4]).expect("descriptor");
    let buffer = ctx.alloc(&desc).expect("alloc");
    drop(ctx);
    drop(runtime);
    assert_eq!(buffer.desc().bytes, 16);
    assert!(buffer.context().device_index() == index, "the buffer still knows its context");
}

#[test]
fn lifetime_result_view_keeps_the_lease_after_the_result_is_released() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&[TEXT], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let view = result.buffer(0).expect("result view");
    assert!(view.is_result_view());
    drop(result);
    // The view still holds the lease: the session is busy.
    assert_err!(session.run(&RunOptions::default()), TURBO_E_BUSY);
    assert_err!(session.write_text(&[TEXT], &EmbedOptions::default()), TURBO_E_BUSY);
    // The view's memory is still readable.
    let mut bytes = vec![0u8; view.desc().bytes as usize];
    view.read_to_host(&mut bytes).expect("read the leased buffer");
    drop(view);
    session.run(&RunOptions::default()).expect("the lease came back with the last view");
}

#[test]
fn lifetime_releasing_the_last_view_returns_the_lease() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&[TEXT], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    let first = result.buffer(0).expect("view");
    let second = result.buffer(0).expect("second view");
    drop(result);
    assert_err!(session.run(&RunOptions::default()), TURBO_E_BUSY);
    drop(first);
    assert_err!(session.run(&RunOptions::default()), TURBO_E_BUSY);
    drop(second);
    session.run(&RunOptions::default()).expect("the last view returned the lease");
}

#[test]
fn lifetime_result_outlives_the_session_and_stays_readable() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let expected = reference_vector(&t);
    let (model, session) = t.session(BundleKind::Embedding);
    session.write_text(&[TEXT], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    drop(session);
    drop(model);
    assert_eq!(read_f32(&result, 0), expected);
    let view = result.buffer(0).expect("a view after the session is gone");
    assert_eq!(view.desc().dtype, DType::F32);
}

#[test]
fn lifetime_generation_step_after_finish_is_invalid_state() {
    let t = Target::from_env();
    needs!(t, Generative);
    let model = t.model(BundleKind::Generative);
    let desc = GenerateDesc { max_new_tokens: 1, ..Default::default() };
    let generation = model.create_generation(&desc).expect("generation");
    generation.prompt(&[Message { role: "user", content: "hello" }]).expect("prompt");
    loop {
        let done = generation.step().expect("step").done;
        if done {
            break;
        }
    }
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
    // And a second time: the state is stable, not a one-shot.
    assert_err!(generation.step(), TURBO_E_INVALID_STATE);
}

#[test]
fn lifetime_binding_a_buffer_from_another_context_is_invalid_argument() {
    let t = Target::from_env();
    needs!(t, Generic);
    let first = t.context();
    let second = t.context();
    let model = t.model_on(&first, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let foreign = second.alloc(&desc).expect("alloc on the other context");
    assert_err!(session.bind("x", &foreign), TURBO_E_INVALID_ARGUMENT);
    // The same shape from the model's own context binds.
    let own = first.alloc(&desc).expect("alloc");
    session.bind("x", &own).expect("bind from the owning context");
}

#[test]
fn lifetime_a_result_view_cannot_be_bound_as_an_input() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ctx = t.context();
    let model = t.model_on(&ctx, BundleKind::Generic);
    let session = model.create_session(&SessionDesc::default()).expect("session");
    let desc = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).expect("descriptor");
    let x = ctx.alloc(&desc).expect("alloc");
    session.bind("x", &x).expect("bind");
    let result = session.run(&RunOptions::default()).expect("run");
    let view = result.buffer(0).expect("view");
    assert_err!(session.bind("x", &view), TURBO_E_INVALID_ARGUMENT);
}

#[test]
fn lifetime_two_contexts_on_one_device_are_independent() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let first = t.context();
    let second = t.context();
    assert!(!Arc::ptr_eq(&first, &second));
    let a = t.model_on(&first, BundleKind::Embedding);
    let b = t.model_on(&second, BundleKind::Embedding);
    let sa = a.create_session(&SessionDesc::default()).expect("session a");
    let sb = b.create_session(&SessionDesc::default()).expect("session b");
    sa.write_text(&[TEXT], &EmbedOptions::default()).expect("write a");
    sb.write_text(&[TEXT], &EmbedOptions::default()).expect("write b");
    let ra = sa.run(&RunOptions::default()).expect("run a");
    let rb = sb.run(&RunOptions::default()).expect("run b");
    assert_eq!(read_f32(&ra, 0), read_f32(&rb, 0), "two contexts must agree on the same input");
    // A lease on one session says nothing about the other.
    assert_err!(sa.run(&RunOptions::default()), TURBO_E_BUSY);
    drop(rb);
    sb.write_text(&[TEXT], &EmbedOptions::default()).expect("the other session is free");
}

#[test]
fn lifetime_many_sessions_on_one_model_are_independent() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let model = t.model(BundleKind::Embedding);
    let sessions: Vec<_> = (0..4).map(|_| model.create_session(&SessionDesc::default()).expect("session")).collect();
    for s in &sessions {
        s.write_text(&[TEXT], &EmbedOptions::default()).expect("write");
    }
    let results: Vec<_> = sessions.iter().map(|s| s.run(&RunOptions::default()).expect("run")).collect();
    let first = read_f32(&results[0], 0);
    for r in &results[1..] {
        assert_eq!(read_f32(r, 0), first, "sessions on one model must agree");
    }
    drop(results);
    drop(model);
    for s in &sessions {
        s.write_text(&[TEXT], &EmbedOptions::default()).expect("write after the model is released");
        s.run(&RunOptions::default()).expect("run after the model is released");
    }
}
