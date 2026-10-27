//! The npu backend's NATIVE path on a device: a graph loaded from the
//! compiled blob embeds as the compiled IR does, and a second load of the
//! same IR takes the blob from the process's cache instead of compiling.
//! The path is opt-in (`npu-native`); this test turns it on for its own
//! thread, so it runs on a default `npu` build.
//!
//! Built with the `npu` feature only, and ignored unless asked for: it
//! reads a bundle with an OpenVINO IR for the npu backend from
//! TURBO_TEST_BUNDLE. docs/npu.md says how to run it. Run it with
//! --nocapture to see each load's time, step by step.

#![cfg(feature = "npu")]

mod common;

use std::time::{Duration, Instant};

use common::*;
use turbo::*;

fn bundle() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// A load on the first NPU, native or not, with where its graph came
/// from ("lite", "compiled" or "cache"). Prints the load's steps; what
/// they leave of the whole is the core's part (the bundle read, the
/// files' SHA-256 check against the manifest) and the device listing.
fn timed_load(name: &str, native: bool) -> (Loaded, &'static str) {
    turbo::npu::set_native(Some(native));
    let t = Instant::now();
    let l = Loaded::load_on(&bundle(), |rt| first_of(rt, "npu").expect("an npu device"))
        .unwrap_or_else(|e| panic!("{e:?}"));
    let total = t.elapsed();
    let from = turbo::npu::last_load().expect("a load on this thread");
    let phases = turbo::npu::last_phases();
    let backend: Duration = phases.iter().map(|(_, d)| *d).sum();
    let steps: Vec<String> = phases.iter().map(|(n, d)| format!("{n} {:.1}", ms(*d))).collect();
    println!(
        "{name} ({from}): {:.1} ms; backend {:.1} ms [{}]; core and listing {:.1} ms",
        ms(total),
        ms(backend),
        steps.join(", "),
        ms(total.saturating_sub(backend))
    );
    (l, from)
}

fn rows(l: &Loaded, texts: &[&str]) -> Vec<Vec<f32>> {
    let s = Session::create(l.m, Some(&session_desc(texts.len() as u32, 0, TURBO_PRECISION_MODEL))).unwrap();
    s.embed(texts, Some(&embed_options())).unwrap()
}

#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn a_native_graph_matches_the_compiled_ir_and_a_second_load_hits_the_cache() {
    let texts = ["Embedding models turn text into vectors.", "The NPU runs the compiled graph.", "short"];

    let (lite, from) = timed_load("NGRAPH_LITE", false);
    assert_eq!(from, "lite");
    let want = rows(&lite, &texts);

    let (first, from) = timed_load("NATIVE first", true);
    assert!(
        from == "compiled" || from == "cache",
        "the driver exports the blob and takes it back as NATIVE: this load was {from}"
    );
    let (second, from) = timed_load("NATIVE second", true);
    assert_eq!(from, "cache", "the second load of the same IR takes the kept blob");
    let steps: Vec<&str> = turbo::npu::last_phases().iter().map(|(n, _)| *n).collect();
    assert!(!steps.contains(&"container") && !steps.contains(&"compile"), "a cache hit compiles nothing: {steps:?}");
    turbo::npu::set_native(None);

    for (name, l) in [("compiled", &first), ("cache", &second)] {
        let got = rows(l, &texts);
        assert_eq!(got.len(), want.len());
        let worst =
            got.iter().zip(&want).flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs())).fold(0f32, f32::max);
        println!("{name}: max |NATIVE - NGRAPH_LITE| = {worst:e}");
        assert!(worst <= 1e-5, "{name}: the NATIVE graph embeds as the compiled IR does: {worst:e}");
    }
}
