//! Requires the Intel GPU and the pinned MiniLM bundle; writes no receipts or goldens.
#![cfg(feature = "genai")]

use std::sync::{mpsc, Arc, Barrier};
use turboembed::{Device, EmbedOptions, Engine, Error};

fn close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), 384);
    assert_eq!(right.len(), left.len());
    for (&a, &b) in left.iter().zip(right) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 1e-5, "embedding changed: {a} vs {b}");
    }
}

#[test]
fn independent_gpu_engines_retain_results_and_survive_peer_destruction() {
    let first = Engine::create(Device::OpenVinoGpu).expect("Intel GPU required");
    let second = Engine::create(Device::OpenVinoGpu).expect("second Intel GPU engine");
    first.load_model("minilm").unwrap();
    second.load_model("minilm").unwrap();
    let start = Arc::new(Barrier::new(2));
    let (released, wait_release) = mpsc::channel();
    let first_start = Arc::clone(&start);
    let first_thread = std::thread::spawn(move || {
        let opts = EmbedOptions::default();
        let held = first.embed_one("minilm", "hello world", &opts).unwrap();
        let expected = held.values().to_vec();
        first_start.wait();
        for _ in 0..8 {
            let another = first
                .embed_one("minilm", "Straße [MASK] café\0world", &opts)
                .unwrap();
            assert_eq!(another.dim(), 384);
            assert!(another.values().iter().all(|x| x.is_finite()));
            close(held.values(), &expected);
        }
        drop(first);
        close(held.values(), &expected);
        drop(held); // Last owner: native destruction must not affect the peer.
        released.send(()).unwrap();
    });
    let second_thread = std::thread::spawn(move || {
        let opts = EmbedOptions::default();
        let held = second.embed_one("minilm", "hello world", &opts).unwrap();
        let expected = held.values().to_vec();
        start.wait();
        for _ in 0..8 {
            let result = second.embed_one("minilm", "hello world", &opts).unwrap();
            close(result.values(), &expected);
            close(held.values(), &expected);
        }
        wait_release.recv().unwrap();
        let after = second.embed_one("minilm", "hello world", &opts).unwrap();
        close(after.values(), &expected);
        drop(after);
        drop(second);
        close(held.values(), &expected);
    });
    first_thread.join().unwrap();
    second_thread.join().unwrap();
}

#[test]
fn gpu_rejects_unsupported_options_then_remains_usable() {
    let engine = Engine::create(Device::OpenVinoGpu).expect("Intel GPU required");
    engine.load_model("minilm").unwrap();
    for opts in [
        EmbedOptions {
            truncate_to: Some(16),
            ..Default::default()
        },
        EmbedOptions {
            normalize: Some(false),
            ..Default::default()
        },
    ] {
        let result = engine.embed_one("minilm", "hello world", &opts);
        assert!(matches!(result, Err(Error::NotImplemented(_))));
    }
    let result = engine
        .embed_one("minilm", "hello world", &EmbedOptions::default())
        .unwrap();
    assert_eq!(result.dim(), 384);
    assert!(result.values().iter().all(|x| x.is_finite()));
}
