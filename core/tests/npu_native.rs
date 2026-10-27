//! The npu backend's NATIVE path on a device: a graph loaded from the
//! compiled blob embeds as the compiled IR does, and a second load of the
//! same IR takes the blob from the process's cache instead of compiling.
//!
//! Built with the `npu` feature only, and ignored unless asked for: it
//! reads a bundle with an OpenVINO IR for the npu backend from
//! TURBO_TEST_BUNDLE. docs/npu.md says how to run it. Run it with
//! --nocapture to see how long each load took.

#![cfg(feature = "npu")]

mod common;

use std::time::{Duration, Instant};

use common::*;
use turbo::*;

fn bundle() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"))
}

/// A load on the first NPU, how long it took, and where its graph came
/// from: "lite", "compiled" or "cache".
fn timed_load() -> (Loaded, Duration, &'static str) {
    let t = Instant::now();
    let l = Loaded::load_on(&bundle(), |rt| first_of(rt, "npu").expect("an npu device"))
        .unwrap_or_else(|e| panic!("{e:?}"));
    let took = t.elapsed();
    (l, took, turbo::npu::last_load().expect("a load on this thread"))
}

fn rows(l: &Loaded, texts: &[&str]) -> Vec<Vec<f32>> {
    let s = Session::create(l.m, Some(&session_desc(texts.len() as u32, 0, TURBO_PRECISION_MODEL))).unwrap();
    s.embed(texts, Some(&embed_options())).unwrap()
}

#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn a_native_graph_matches_the_compiled_ir_and_a_second_load_hits_the_cache() {
    let texts = ["Embedding models turn text into vectors.", "The NPU runs the compiled graph.", "short"];

    turbo::npu::prefer_lite(true);
    let (lite, lite_took, from) = timed_load();
    assert_eq!(from, "lite");
    let want = rows(&lite, &texts);
    turbo::npu::prefer_lite(false);

    let (first, first_took, from) = timed_load();
    assert!(
        from == "compiled" || from == "cache",
        "the driver exports the blob and takes it back as NATIVE: this load was {from}"
    );
    let (second, second_took, from) = timed_load();
    assert_eq!(from, "cache", "the second load of the same IR takes the kept blob");

    for (name, l) in [("compiled", &first), ("cache", &second)] {
        let got = rows(l, &texts);
        assert_eq!(got.len(), want.len());
        let worst =
            got.iter().zip(&want).flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs())).fold(0f32, f32::max);
        println!("{name}: max |NATIVE - NGRAPH_LITE| = {worst:e}");
        assert!(worst <= 1e-5, "{name}: the NATIVE graph embeds as the compiled IR does: {worst:e}");
    }
    println!(
        "load: NGRAPH_LITE {} ms, NATIVE compiled {} ms, NATIVE from cache {} ms",
        lite_took.as_millis(),
        first_took.as_millis(),
        second_took.as_millis()
    );
}
