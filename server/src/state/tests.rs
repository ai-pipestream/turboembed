use super::*;
use crate::config::Device;

fn config(name: &str) -> ModelConfig {
    ModelConfig {
        bundle: format!("b/{name}"),
        name: Some(name.into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

/// The error of a refused registration.
fn refused(r: api::Result<Arc<Served>>) -> Failure {
    match r {
        Err(e) => e,
        Ok(_) => panic!("a capped registration was accepted"),
    }
}

#[test]
fn registration_is_capped_with_the_limit_named() {
    let state = State::new(vec!["base".into()], vec![config("base")]);
    for k in 0..MAX_LOADED_MODELS - 1 {
        state.add(format!("m{k}"), config(&format!("m{k}"))).unwrap();
    }
    let e = refused(state.add("one-more".into(), config("one-more")));
    assert_eq!(e.code, TURBO_E_CAPACITY);
    assert!(e.message.contains("max_loaded_models"), "{e:?}");
    // The base model still serves.
    assert!(state.peek("base").is_ok());
}

#[test]
fn rollback_removes_only_the_exact_registration() {
    let state = State::new(Vec::new(), Vec::new());
    let a = state.add("x".into(), config("x")).unwrap();
    // A same-named registration is a different identity: rolling
    // back against it must not remove the live one.
    state.remove("x").unwrap();
    let b = state.add("x".into(), config("x")).unwrap();
    assert!(!Arc::ptr_eq(&a, &b));
    state.remove_if(&a).unwrap();
    assert!(state.peek("x").is_ok(), "the replacement registration stands");
    assert!(Arc::ptr_eq(&state.peek("x").unwrap().0, &b));
    // Rolling back the live one removes it.
    state.remove_if(&b).unwrap();
    assert!(state.peek("x").is_err());
    // Rolling back what is already gone is not an error.
    state.remove_if(&b).unwrap();
}

#[test]
fn peek_returns_the_registration_without_cloning_the_name() {
    let state = State::new(vec!["x".into()], vec![config("x")]);
    let (m, loaded) = state.peek("x").unwrap();
    assert_eq!(m.name, "x");
    assert!(loaded.is_none(), "nothing is loaded yet; ready() reports that on the same path");
    assert!(state.ready("x", "").is_err());
}

#[test]
fn load_error_preserves_status() {
    let err = LoadError {
        model: "m".into(),
        call: "turbo_session_create",
        failure: Failure::new(TURBO_E_UNSUPPORTED_OPTION, 2, "over max_seq"),
        fields: &SESSION_DESC,
    };
    let status = err.to_status();
    assert_eq!(status.code(), tonic::Code::Unimplemented);
    assert_eq!(status.metadata().get("turbo-code").unwrap(), "513");
    assert_eq!(status.metadata().get("turbo-field").unwrap(), "2");
    assert!(status.message().contains("field 2 (max_seq)"), "{}", status.message());
}
