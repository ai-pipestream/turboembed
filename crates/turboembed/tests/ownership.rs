//! Safe-wrapper regressions. Potential process-aborting failures run in a child.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::Command;
use turboembed::{Device, EmbedOptions, Engine, Error};

fn in_child(name: &str, test: impl FnOnce()) {
    if std::env::var("TURBOEMBED_OWNERSHIP_CASE").as_deref() == Ok(name) {
        test();
        return;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("TURBOEMBED_OWNERSHIP_CASE", name)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child {name} failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn results_survive_engine_and_other_results() {
    in_child("results_survive_engine_and_other_results", || {
        let engine = Engine::create(Device::Mock).unwrap();
        let opts = EmbedOptions::default();
        let texts = vec!["retained result"; 8192];
        let first = engine.embed("mock-embed", &texts, &opts).unwrap();
        assert!(first.row(usize::MAX).is_none());
        let expected = first.values().to_vec();
        let other = engine.embed_one("mock-embed", "other", &opts).unwrap();
        drop(engine);
        std::thread::spawn(move || {
            drop(other);
            assert_eq!(first.values(), expected);
            assert_eq!(first.packed().len(), expected.len() * 4);
            drop(first);
        })
        .join()
        .unwrap();
    });
}

#[test]
fn empty_aliases_are_rejected_without_reading_backing_storage() {
    let engine = Engine::create(Device::Mock).unwrap();
    let backing = String::from("mock-embed\0");
    // The legacy ABI treats len=0 as strlen. A safe &str must not do that.
    let empty = &backing[..0];
    let opts = EmbedOptions::default();
    assert!(matches!(
        engine.load_model(empty),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.embed_one(empty, "", &opts),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.embed(empty, &[""], &opts),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.embed_stream(empty, &[""], &opts, |_, _, _| panic!(
            "invalid request called back"
        )),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn text_inputs_honor_exact_utf8_spans() {
    let engine = Engine::create(Device::Mock).unwrap();
    let opts = EmbedOptions::default();
    let embed = |text: &str| {
        let result = engine.embed_one("mock-embed", text, &opts).unwrap();
        assert_eq!(result.count(), 1);
        result.values().to_vec()
    };

    // Embedded NUL must not truncate the advertised pointer+length view.
    let with_nul = embed("a\0bc");
    assert_ne!(with_nul, embed("a"), "input was truncated at the NUL");
    assert_ne!(with_nul, embed("abc"), "the NUL byte was dropped");
    assert_eq!(with_nul, embed("a\0bc"), "same bytes must be deterministic");

    // Empty text is a valid zero-length view, not an error or a strlen probe.
    let empty = embed("");
    assert!(!empty.is_empty());
    assert_ne!(
        empty,
        embed("\0"),
        "zero length must not read past the view"
    );

    // Batch rows see the same per-row spans as single-text calls.
    let batch = engine
        .embed("mock-embed", &["a\0bc", "", "héllo"], &opts)
        .unwrap();
    assert_eq!(batch.count(), 3);
    assert_eq!(batch.row(0).unwrap(), with_nul.as_slice());
    assert_eq!(batch.row(1).unwrap(), empty.as_slice());
    assert_eq!(batch.row(2).unwrap(), embed("héllo").as_slice());
}

#[test]
fn callback_panic_unwinds_after_native_return() {
    in_child("callback_panic_unwinds_after_native_return", || {
        let engine = Engine::create(Device::Mock).unwrap();
        let opts = EmbedOptions::default();
        let mut calls = 0;
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = engine.embed_stream("mock-embed", &["a", "b"], &opts, |_, _, _| {
                calls += 1;
                panic!("callback failure");
            });
        }))
        .expect_err("callback panic should reach Rust caller");
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"callback failure"));
        assert_eq!(calls, 1);
        assert_eq!(
            engine
                .embed_one("mock-embed", "still usable", &opts)
                .unwrap()
                .count(),
            1
        );
    });
}

#[test]
fn callback_reentry_is_rejected_and_result_release_is_safe() {
    let engine = Engine::create(Device::Mock).unwrap();
    let other = Engine::create(Device::Mock).unwrap();
    let opts = EmbedOptions::default();
    let mut previous = Some(engine.embed_one("mock-embed", "previous", &opts).unwrap());
    let result = engine
        .embed_stream("mock-embed", &["a", "b"], &opts, |_, _, _| {
            drop(previous.take());
            assert!(matches!(
                engine.load_model("mock-embed"),
                Err(Error::InvalidArgument(_))
            ));
            assert!(matches!(
                engine.list_models(),
                Err(Error::InvalidArgument(_))
            ));
            assert!(matches!(
                engine.embed_one("mock-embed", "nested", &opts),
                Err(Error::InvalidArgument(_))
            ));
            assert_eq!(
                other
                    .embed_one("mock-embed", "independent", &opts)
                    .unwrap()
                    .count(),
                1
            );
        })
        .unwrap();
    assert_eq!(result.count(), 2);
    engine.load_model("mock-embed").unwrap();
}

#[test]
fn callback_can_join_result_release_on_another_thread() {
    let engine = Engine::create(Device::Mock).unwrap();
    let opts = EmbedOptions::default();
    let mut previous = Some(engine.embed_one("mock-embed", "previous", &opts).unwrap());
    engine
        .embed_stream("mock-embed", &["a"], &opts, |_, _, _| {
            let result = previous.take().unwrap();
            std::thread::spawn(move || drop(result)).join().unwrap();
        })
        .unwrap();
    engine.load_model("mock-embed").unwrap();
}

#[test]
fn results_can_be_released_while_owner_runs_on_another_thread() {
    let engine = Engine::create(Device::Mock).unwrap();
    let opts = EmbedOptions::default();
    let results: Vec<_> = (0..128)
        .map(|_| engine.embed_one("mock-embed", "old", &opts).unwrap())
        .collect();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let worker_barrier = barrier.clone();
    let release = std::thread::spawn(move || {
        worker_barrier.wait();
        for result in results {
            drop(result);
        }
    });
    barrier.wait();
    for _ in 0..128 {
        let result = engine.embed_one("mock-embed", "new", &opts).unwrap();
        assert_eq!(result.count(), 1);
    }
    release.join().unwrap();
}
