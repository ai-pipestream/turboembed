//! Engine lifecycle and concurrency through the safe wrapper.
//!
//! Covers the multi-engine and multi-thread ground that `ownership.rs`
//! (single-engine lifetime rules, subprocess idiom) and `abi_smoke.rs`
//! (one create/list/load/embed pass) leave open: alternating mock engines
//! with result isolation, retained results across later calls and failures,
//! per-engine error storage, `Send`/`!Sync` thread contract, and the
//! `status_name` / `device_name` / `abi_version` surface over an engine
//! churn. Everything here is always-on and uses only `Device::Mock`.

use std::collections::HashSet;

use turboembed::ffi::turboembed_status;
use turboembed::{
    abi_version, register_provider_stub, Device, EmbedOptions, Embeddings, Engine, Error, ModelInfo,
};

fn mock_engine() -> Engine {
    Engine::create(Device::Mock).expect("create mock engine")
}

fn only_model(engine: &Engine) -> ModelInfo {
    let models = engine.list_models().expect("list models");
    assert_eq!(models.len(), 1, "mock engine lists exactly one model");
    models.get(0).expect("one model row")
}

// --- Compile-time thread contract -----------------------------------------

mod sync_probe {
    use std::marker::PhantomData;

    pub struct Probe<T: ?Sized>(PhantomData<T>);

    pub trait AmbiguousIfSync<A> {
        fn marker() {}
    }

    impl<T: ?Sized> AmbiguousIfSync<()> for Probe<T> {}
    impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for Probe<T> {}
}

/// Compiles only while `T` is not `Sync`: when `T: Sync`, both
/// `AmbiguousIfSync` impls apply and `Probe::<T>::marker` is ambiguous.
fn assert_not_sync<T: ?Sized>() {
    use sync_probe::AmbiguousIfSync as _;
    let _ = sync_probe::Probe::<T>::marker;
}

fn assert_send<T: Send>() {}

#[test]
fn engine_is_send_but_not_sync() {
    assert_send::<Engine>();
    assert_send::<Embeddings>();
    // Documented contract (lib.rs): movable to another thread, never
    // shareable as `&Engine` — `Engine` carries same-engine reentry state.
    assert_not_sync::<Engine>();
    assert_not_sync::<Embeddings>();
}

#[test]
fn engine_moves_across_threads_sequentially() {
    let engine = mock_engine();
    engine.load_model("mock-embed").unwrap();
    let opts = EmbedOptions::default();
    let anchor = engine
        .embed_one("mock-embed", "handoff", &opts)
        .unwrap()
        .values()
        .to_vec();

    // Hand the engine down a chain of threads; it must stay fully usable
    // (and deterministic) after every move.
    let mut engine = engine;
    for leg in 0..3 {
        let anchor = anchor.clone();
        engine = std::thread::spawn(move || {
            let got = engine
                .embed_one("mock-embed", "handoff", &opts)
                .unwrap_or_else(|e| panic!("leg {leg}: embed after handoff failed: {e}"));
            assert_eq!(got.values(), anchor.as_slice(), "leg {leg}");
            engine
        })
        .join()
        .unwrap_or_else(|e| panic!("leg {leg} panicked: {e:?}"));
    }
    let home = engine
        .embed_one("mock-embed", "handoff", &opts)
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(home, anchor);
}

#[test]
fn independent_engines_embed_concurrently_on_scoped_threads() {
    const WORKERS: usize = 8;
    const CALLS: usize = 32;

    // Engines are created on the main thread first: the native create path
    // keeps process-wide state (g_create_error), which embed calls never touch.
    let engines: Vec<Engine> = (0..WORKERS).map(|_| mock_engine()).collect();
    let last_rows = std::thread::scope(|scope| {
        let handles: Vec<_> = engines
            .into_iter()
            .enumerate()
            .map(|(worker, engine)| {
                scope.spawn(move || {
                    let opts = EmbedOptions::default();
                    let mut last = Vec::new();
                    for call in 0..CALLS {
                        let text = format!("worker-{worker}-call-{call}");
                        let result = engine
                            .embed_one("mock-embed", &text, &opts)
                            .unwrap_or_else(|e| panic!("worker {worker} call {call}: {e}"));
                        assert_eq!(result.dim(), 8);
                        assert_eq!(result.count(), 1);
                        last = result.values().to_vec();
                    }
                    last
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    // Mock rows are a pure function of the input bytes, so a main-thread
    // engine must reproduce every worker's final row exactly; any
    // cross-engine arena bleed would show up here as a mismatch.
    let checker = mock_engine();
    let opts = EmbedOptions::default();
    for (worker, row) in last_rows.iter().enumerate() {
        let text = format!("worker-{worker}-call-{}", CALLS - 1);
        let expected = checker.embed_one("mock-embed", &text, &opts).unwrap();
        assert_eq!(
            row.as_slice(),
            expected.values(),
            "worker {worker} produced a different row for identical input"
        );
    }
}

// --- Multi-engine isolation -------------------------------------------------

#[test]
fn alternating_engines_keep_results_isolated() {
    let a = mock_engine();
    let b = mock_engine();
    let opts = EmbedOptions::default();

    let mut held: Vec<Embeddings> = Vec::new();
    let mut snapshots: Vec<Vec<f32>> = Vec::new();
    for i in 0..32 {
        let text = format!("alternating-{i}");
        let (owner, peer) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
        let result = owner.embed_one("mock-embed", &text, &opts).unwrap();
        snapshots.push(result.values().to_vec());
        held.push(result);
        // The same text through the other engine must produce the same row:
        // identical deterministic mock output proves the two engines'
        // buffers are not bleeding into each other.
        let peer_row = peer.embed_one("mock-embed", &text, &opts).unwrap();
        assert_eq!(
            peer_row.values(),
            snapshots[i],
            "engines disagree on {text}"
        );
    }

    // Every retained snapshot survived the interleaving.
    for (result, snapshot) in held.iter().zip(&snapshots) {
        assert_eq!(result.values(), snapshot.as_slice());
    }

    // Dropping the peer engine must not disturb the survivor's results.
    drop(b);
    for (result, snapshot) in held.iter().zip(&snapshots) {
        assert_eq!(result.values(), snapshot.as_slice());
    }
    assert_eq!(
        a.embed_one("mock-embed", "after peer drop", &opts)
            .unwrap()
            .count(),
        1
    );
}

#[test]
fn retained_result_survives_later_calls_and_failures() {
    let engine = mock_engine();
    let opts = EmbedOptions::default();
    let held = engine.embed_one("mock-embed", "anchor", &opts).unwrap();
    let snapshot = held.values().to_vec();

    // Later successful calls on the same engine must not recycle or
    // overwrite the retained buffer.
    for i in 0..64 {
        let text = format!("later-{i}");
        let fresh = engine.embed_one("mock-embed", &text, &opts).unwrap();
        assert_eq!(fresh.dim(), 8);
        assert_eq!(
            held.values(),
            snapshot.as_slice(),
            "retained result mutated by call {i}"
        );
    }

    // Neither must a failing call (mock refuses catalog aliases).
    let err = engine.load_model("minilm").unwrap_err();
    assert!(
        matches!(err, Error::NotImplemented(_)),
        "mock must refuse catalog aliases, got {err:?}"
    );
    assert_eq!(
        held.values(),
        snapshot.as_slice(),
        "retained result mutated by failing call"
    );

    // Nor a streamed batch.
    let streamed = engine
        .embed_stream("mock-embed", &["x", "y"], &opts, |_, _, _| {})
        .unwrap();
    assert_eq!(streamed.count(), 2);
    assert_eq!(
        held.values(),
        snapshot.as_slice(),
        "retained result mutated by streamed call"
    );

    // The retained copy is still bit-identical to a fresh recompute.
    let recomputed = engine.embed_one("mock-embed", "anchor", &opts).unwrap();
    assert_eq!(recomputed.values(), held.values());
}

#[test]
fn last_error_is_per_engine_and_cleared_on_success() {
    let a = mock_engine();
    let b = mock_engine();
    assert!(a.last_error().is_empty(), "fresh engine has no error");
    assert!(b.last_error().is_empty(), "fresh engine has no error");

    let err = a.load_model("minilm").unwrap_err();
    let Error::NotImplemented(msg_a) = &err else {
        panic!("mock must refuse catalog aliases, got {err:?}");
    };
    assert!(
        !a.last_error().is_empty(),
        "failing call must record the error on its own engine"
    );
    assert_eq!(
        a.last_error(),
        *msg_a,
        "last_error must return the message the caller already saw"
    );
    assert!(
        b.last_error().is_empty(),
        "engine B must not observe engine A's error"
    );

    // A failure on B with its own message must not clobber A's stored error.
    let err_b = b.load_model("bge-small-en").unwrap_err();
    assert!(matches!(err_b, Error::NotImplemented(_)));
    assert!(!b.last_error().is_empty());
    assert_eq!(a.last_error(), *msg_a, "B's failure overwrote A's error");

    // Success clears the stored error on the engine that succeeded.
    a.load_model("mock-embed").unwrap();
    assert!(
        a.last_error().is_empty(),
        "successful call must clear the stored error"
    );
    assert!(
        !b.last_error().is_empty(),
        "B's error must survive A's success"
    );

    // An engine created after both failures starts clean.
    let c = mock_engine();
    assert!(
        c.last_error().is_empty(),
        "create-time state must not leak earlier failures into a new engine"
    );
}

#[test]
fn list_models_is_consistent_across_calls_and_engines() {
    let a = mock_engine();
    let b = mock_engine();

    let first = only_model(&a);
    // Repeated listings on the same engine agree, as do listings on an
    // independent engine: the mock catalog row is per-process constant.
    assert_eq!(only_model(&a), first, "second listing on engine A changed");
    assert_eq!(only_model(&b), first, "engine B lists a different row");

    // Field-level contract for the mock row.
    assert_eq!(first.alias, "mock-embed");
    assert_eq!(first.dim, 8);
    assert_eq!(first.device, Device::Mock);
    assert!(first.ready, "create(Device::Mock) auto-loads the mock");
    assert!(!first.alias.is_empty());

    // An explicit reload is idempotent and must not change the listing.
    a.load_model("mock-embed").unwrap();
    assert_eq!(only_model(&a), first, "listing changed after reload");
}

// --- Frozen name/version surface --------------------------------------------

/// The wrapper's `Error::from_status` mapping, mirrored here so the test
/// fails to compile if a variant is added without status-name coverage.
fn status_for(err: &Error) -> Option<turboembed_status> {
    use turboembed::ffi::turboembed_status as st;
    match err {
        Error::InvalidArgument(_) => Some(st::TURBOEMBED_ERR_INVALID_ARGUMENT),
        Error::NotFound(_) => Some(st::TURBOEMBED_ERR_NOT_FOUND),
        Error::NotImplemented(_) => Some(st::TURBOEMBED_ERR_NOT_IMPLEMENTED),
        Error::Unavailable(_) => Some(st::TURBOEMBED_ERR_UNAVAILABLE),
        Error::Internal(_) => Some(st::TURBOEMBED_ERR_INTERNAL),
        Error::OutOfMemory(_) => Some(st::TURBOEMBED_ERR_OUT_OF_MEMORY),
        Error::UnsupportedDevice(_) => Some(st::TURBOEMBED_ERR_UNSUPPORTED_DEVICE),
        // Fallback for status codes outside the frozen enum: no name to check.
        Error::Other { .. } => None,
    }
}

#[test]
fn status_name_matches_every_error_variant() {
    use turboembed::ffi::turboembed_status as st;

    let named = [
        (
            Error::InvalidArgument("bad".into()),
            st::TURBOEMBED_ERR_INVALID_ARGUMENT,
            "INVALID_ARGUMENT",
        ),
        (
            Error::NotFound("missing".into()),
            st::TURBOEMBED_ERR_NOT_FOUND,
            "NOT_FOUND",
        ),
        (
            Error::NotImplemented("reserved".into()),
            st::TURBOEMBED_ERR_NOT_IMPLEMENTED,
            "NOT_IMPLEMENTED",
        ),
        (
            Error::Unavailable("down".into()),
            st::TURBOEMBED_ERR_UNAVAILABLE,
            "UNAVAILABLE",
        ),
        (
            Error::Internal("bug".into()),
            st::TURBOEMBED_ERR_INTERNAL,
            "INTERNAL",
        ),
        (
            Error::OutOfMemory("heap".into()),
            st::TURBOEMBED_ERR_OUT_OF_MEMORY,
            "OUT_OF_MEMORY",
        ),
        (
            Error::UnsupportedDevice("odd gpu".into()),
            st::TURBOEMBED_ERR_UNSUPPORTED_DEVICE,
            "UNSUPPORTED_DEVICE",
        ),
    ];
    let mut seen = HashSet::new();
    for (variant, status, want) in &named {
        assert_eq!(status_for(variant), Some(*status));
        let got = turboembed::status_name(*status);
        assert!(!got.is_empty(), "{variant:?} has an empty status name");
        assert_eq!(got, *want, "wrong status name for {variant:?}");
        assert!(seen.insert(got), "status name reused: {got}");
    }
    // `Other` is the only variant without a frozen status code.
    assert_eq!(
        status_for(&Error::Other {
            code: 99,
            message: "future status".into(),
        }),
        None
    );
    assert_eq!(seen.len(), named.len(), "a variant shares its status name");
    assert_eq!(turboembed::status_name(st::TURBOEMBED_OK), "OK");
}

#[test]
fn device_names_are_distinct_nonempty_tokens() {
    // Exact strings are part of the frozen ABI (identical in the C++ stub
    // and the Swift ABI) and must stay kebab-case tokens.
    let cases = [
        (Device::Auto, "auto"),
        (Device::Cpu, "cpu"),
        (Device::Cuda, "cuda"),
        (Device::TensorRt, "tensorrt"),
        (Device::OpenVinoCpu, "openvino-cpu"),
        (Device::OpenVinoGpu, "openvino-gpu"),
        (Device::OpenVinoNpu, "openvino-npu"),
        (Device::Metal, "metal"),
        (Device::Mock, "mock"),
        (Device::Hailo, "hailo"),
    ];
    let mut seen = HashSet::new();
    for (device, want) in cases {
        let got = device.as_str();
        assert!(!got.is_empty(), "{device:?} has an empty device name");
        assert_eq!(got, want, "wrong device name for {device:?}");
        assert!(
            got.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "{device:?} name {got} is not a kebab-case token"
        );
        assert!(seen.insert(got), "device name reused: {got}");
    }
    assert_eq!(seen.len(), cases.len(), "a Device variant shares its name");
}

#[test]
fn register_provider_stub_stays_reserved_across_engine_lifecycle() {
    let assert_reserved = |where_: &str| {
        let err = register_provider_stub().unwrap_err();
        let Error::NotImplemented(msg) = err else {
            panic!("register_provider_stub must stay reserved {where_}, got {err:?}");
        };
        assert!(
            msg.contains("reserved"),
            "registration refusal must say it is reserved {where_}: {msg}"
        );
    };

    assert_reserved("before any engine");
    let engine = mock_engine();
    engine.load_model("mock-embed").unwrap();
    let _ = engine
        .embed_one("mock-embed", "mid-lifecycle", &EmbedOptions::default())
        .unwrap();
    assert_reserved("between embed calls");
    drop(engine);
    assert_reserved("after engine drop");
}

#[test]
fn engine_create_drop_churn_keeps_abi_and_results_stable() {
    assert_eq!(abi_version(), 1);
    assert_eq!(abi_version(), turboembed::ffi::TURBOEMBED_ABI_VERSION);

    let anchor_engine = mock_engine();
    let opts = EmbedOptions::default();
    let anchor = anchor_engine
        .embed_one("mock-embed", "churn-anchor", &opts)
        .unwrap();
    let snapshot = anchor.values().to_vec();

    for i in 0..64 {
        let engine = mock_engine();
        let result = engine
            .embed_one("mock-embed", &format!("churn-{i}"), &opts)
            .unwrap();
        assert_eq!(result.dim(), 8);
        drop(engine);
        assert_eq!(abi_version(), 1, "ABI version moved during churn");
        assert_eq!(
            anchor.values(),
            snapshot.as_slice(),
            "retained result mutated during churn iteration {i}"
        );
    }

    let home = anchor_engine
        .embed_one("mock-embed", "churn-anchor", &opts)
        .unwrap();
    assert_eq!(home.values(), snapshot.as_slice());
    assert_eq!(abi_version(), 1);
}
