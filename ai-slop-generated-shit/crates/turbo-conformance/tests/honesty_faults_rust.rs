//! Group `contract`, Rust layer: the status codes a caller can only reach
//! when something fails.
//!
//! `TURBO_E_DEVICE_UNAVAILABLE`, `TURBO_E_UNSUPPORTED_DTYPE`,
//! `TURBO_E_OVERLOADED`, `TURBO_E_RUNTIME`, `TURBO_E_INTERNAL` and a provider
//! panic are documented in `docs/c-api.md` and were reachable by no case: a
//! conformance run cannot make a device vanish. The mock provider's `fault`
//! option (`FAULT_OPTION` in `crates/turbo-core/src/mock.rs`) injects each of
//! them at a named stage, so the contract around them is asserted instead of
//! assumed.
//!
//! On any other provider `fault` is simply an unknown option, and these cases
//! assert that it is refused as one rather than ignored: there is no skip.

use turbo::abi::*;
use turbo::handles::{options_from_pairs, Context};
use turbo::provider::{ContextDesc, EmbedOptions, ModelDesc, RunOptions, SessionDesc};
use turbo::types::{CapStatus, Modality, Task};
use turbo_conformance::{assert_err, needs, BundleKind, Target};

/// The fault option a descriptor carries, as provider options.
fn fault(value: &str) -> turbo::provider::Options {
    options_from_pairs([("fault", value)])
}

/// Assert that a provider without the mock's fault hook refuses the option
/// as the unknown key it is, naming its 1-based index (PLAN.md principle 2:
/// an option is honored or refused, never ignored).
fn refused_as_unknown(t: &Target, what: &str, r: turbo::Result<impl Sized>) {
    let e = assert_err!(r, TURBO_E_INVALID_ARGUMENT);
    assert_eq!(e.field(), 1, "{}: {what} must name the option's index: {}", t.provider_id(), e.message());
}

#[test]
fn faults_context_creation_reports_device_unavailable() {
    let t = Target::from_env();
    let desc = ContextDesc { options: fault("context=device_unavailable") };
    let r = Context::create(t.runtime.clone(), t.device_index, &desc);
    if !t.is_mock() {
        refused_as_unknown(&t, "an unknown context option", r);
        return;
    }
    let e = assert_err!(r, TURBO_E_DEVICE_UNAVAILABLE);
    assert!(e.message().contains("device_unavailable"), "the message must name the fault: {}", e.message());
    // The device is not broken: the next context on it is fine.
    t.context();
}

#[test]
fn faults_model_load_reports_unsupported_dtype() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ctx = t.context();
    let desc = ModelDesc { options: fault("load=unsupported_dtype") };
    let r = ctx.load_model(&t.bundle(BundleKind::Embedding), &desc);
    if !t.is_mock() {
        refused_as_unknown(&t, "an unknown model option", r);
        return;
    }
    let e = assert_err!(r, TURBO_E_UNSUPPORTED_DTYPE);
    assert!(e.message().contains("unsupported_dtype"), "the message must name the fault: {}", e.message());
    // The bundle itself is fine.
    ctx.load_model(&t.bundle(BundleKind::Embedding), &ModelDesc::default()).expect("the same bundle loads");
}

#[test]
fn faults_run_reports_overloaded_runtime_and_internal() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let model = t.model(BundleKind::Embedding);
    for (value, code) in
        [("run=overloaded", TURBO_E_OVERLOADED), ("run=runtime", TURBO_E_RUNTIME), ("run=internal", TURBO_E_INTERNAL)]
    {
        let desc = SessionDesc { options: fault(value), ..Default::default() };
        let created = model.create_session(&desc);
        if !t.is_mock() {
            refused_as_unknown(&t, "an unknown session option", created);
            continue;
        }
        let session = created.unwrap_or_else(|e| panic!("`{value}` is a session fault, not a refusal: {e}"));
        session.write_text(&["hello world"], &EmbedOptions::default()).expect("the write itself succeeds");
        let e = assert_err!(session.run(&RunOptions::default()), code);
        assert!(e.message().contains(value.trim_start_matches("run=")), "{}", e.message());
        assert_eq!(e.field(), 0, "a failure of the run is not a field of the descriptor: {}", e.message());
        // A failed run is not a lease: the session takes work again and
        // fails the same way, which is what a retry loop depends on.
        session.write_text(&["hello world"], &EmbedOptions::default()).expect("the session is still usable");
        assert_err!(session.run(&RunOptions::default()), code);
    }
    if !t.is_mock() {
        return;
    }
    // A session without the fault, on the same model, still runs.
    let clean = model.create_session(&SessionDesc::default()).expect("session");
    clean.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    clean.run(&RunOptions::default()).expect("an unfaulted session on the same model runs");
}

#[test]
fn faults_a_panicking_provider_leaves_the_session_inconsistent() {
    // `Session::lock` (crates/turbo-core/src/handles.rs) documents the rule:
    // a provider panic under the lock leaves the state half updated, the
    // mutex poisoned, and every later call on that session
    // `TURBO_E_INVALID_STATE` telling the caller to create a new one. The C
    // layer turns the panic itself into `TURBO_E_PANIC`
    // (`faults_a_panicking_provider_is_turbo_e_panic` in
    // `honesty_faults_c.rs`); through the safe Rust API it is a Rust panic.
    let t = Target::from_env();
    needs!(t, Embedding);
    let model = t.model(BundleKind::Embedding);
    let desc = SessionDesc { options: fault("run=panic"), ..Default::default() };
    let created = model.create_session(&desc);
    if !t.is_mock() {
        refused_as_unknown(&t, "an unknown session option", created);
        return;
    }
    let session = created.expect("`run=panic` is a session fault, not a refusal");
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.run(&RunOptions::default())));
    std::panic::set_hook(previous);
    let payload = caught.expect_err("the injected panic must unwind, not be swallowed");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(message.contains("panic"), "the panic payload must name the fault: {message:?}");

    // Every later call on that session says so, with the documented text.
    let e = assert_err!(session.run(&RunOptions::default()), TURBO_E_INVALID_STATE);
    assert!(e.message().contains("inconsistent"), "the refusal must say the session is spent: {}", e.message());
    let e = assert_err!(session.write_text(&["x"], &EmbedOptions::default()), TURBO_E_INVALID_STATE);
    assert!(e.message().contains("inconsistent"), "{}", e.message());
    assert_err!(session.stats(), TURBO_E_INVALID_STATE);
    // The model is not spent: a new session on it works.
    let fresh = model.create_session(&SessionDesc::default()).expect("a new session on the same model");
    fresh.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    fresh.run(&RunOptions::default()).expect("run");
}

#[test]
fn faults_an_unknown_fault_value_is_refused_by_field() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let model = t.model(BundleKind::Embedding);
    // A value that is not `<stage>=<status>`, a status the stage does not
    // have, and a stage that belongs to another descriptor: all three are
    // caller errors naming the option, never a fault that quietly does
    // nothing.
    for value in ["", "panic", "run=teleport", "load=runtime", "context=runtime", "run=unsupported_dtype"] {
        let desc = SessionDesc { options: fault(value), ..Default::default() };
        let e = assert_err!(model.create_session(&desc), TURBO_E_INVALID_ARGUMENT);
        assert_eq!(e.field(), 1, "`fault={value}` must name the option's index: {}", e.message());
        if t.is_mock() {
            assert!(e.message().contains("fault"), "the refusal must name the option: {}", e.message());
        }
    }
}

#[test]
fn faults_a_planned_cell_is_reported_but_not_offered() {
    // PLAN.md principle 7 and `Capability::is_offered`: PLANNED is an honest
    // "not here yet", so it is reported to a caller that asks and refused to
    // a caller that runs. The mock's CHUNK x TEXT cell is the tree's one
    // planned cell; every other provider reports CHUNK as unsupported, and
    // the refusal is then the same code by the same path.
    let t = Target::from_env();
    let cell = t.runtime.capability(t.device_index, Task::Chunk, Modality::Text).expect("cell");
    if t.is_mock() {
        assert_eq!(cell.status, CapStatus::Planned, "the mock's CHUNK x TEXT cell is the planned one");
        assert!(!cell.notes.is_empty(), "a planned cell says what is planned");
        assert!(cell.dtype.is_none(), "a planned cell runs nothing, so it names no compute dtype");
    } else {
        assert!(
            matches!(cell.status, CapStatus::Unsupported | CapStatus::Planned),
            "{} offers CHUNK x TEXT on a device ({:?}); the case below assumes it does not",
            t.provider_id(),
            cell.status
        );
    }
    assert!(!cell.is_offered(), "a {:?} cell is not runnable", cell.status);
    // A call that lands on it is refused, with the same code an unsupported
    // cell gets: the device offers CHUNK for no modality at all.
    let e = assert_err!(
        t.runtime.can_run(t.device_index, &t.bundle(BundleKind::Embedding), Task::Chunk, Modality::Text),
        TURBO_E_UNSUPPORTED_TASK
    );
    assert!(e.message().contains("Chunk"), "the refusal must name the task: {}", e.message());
    assert!(e.message().contains("any modality"), "no modality offers it, so the refusal says so: {}", e.message());
}
