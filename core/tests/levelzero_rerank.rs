//! Cross-encoder rerank on Level Zero against the reference scores and the
//! CPU backend, as tests/cuda_rerank.rs holds CUDA: pairs batched,
//! passages windowed, and the max over windows.
//!
//! Built with the `levelzero` feature only. A test that needs a device says
//! it was skipped, and passes, when the backend lists none. With
//! TURBO_TEST_REQUIRE_LEVELZERO=1 it fails instead.

#![cfg(feature = "levelzero")]

mod common;
#[path = "rerank/held.rs"]
mod held;
#[path = "rerank/common.rs"]
mod rerank_common;

use std::path::Path;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::{Loaded, first_of, new_error};
use turbo::bundle::Bundle;
use turbo::rerank::{RerankOptions, RerankSessionOptions, Reranker};
use turbo::safetensors::{Dtype, File};

/// The tests take turns on the one device.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

/// The first levelzero device, or None after saying the test is skipped.
fn gpu_device(test: &str) -> Option<u32> {
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo::turbo_runtime_create(ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, "levelzero");
    unsafe { turbo::turbo_runtime_release(rt) };
    if d.is_none() {
        let required = std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| v == "1");
        assert!(!required, "{test}: TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("{test}: skipped: the levelzero feature is on and the levelzero backend lists no device");
    }
    d
}

fn on_gpu(dir: &Path) -> Loaded {
    Loaded::load_on(dir, |rt| first_of(rt, "levelzero").expect("a levelzero device"))
        .unwrap_or_else(|e| panic!("{e:?}"))
}

fn fixture() -> rerank_common::Fixture {
    rerank_common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

fn on(dir: &Path, backend: &str, precision: u32) -> Reranker {
    let o = RerankSessionOptions { backend: Some(backend.into()), precision, cpu_threads: 0 };
    Reranker::open_with(dir, &o).unwrap_or_else(|e| panic!("{backend}: {e:?}"))
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

/// The tiny model's pairs on Level Zero within 1e-4 of the reference and
/// 1e-5 of the CPU; more pairs than a run holds, and overlapping windows,
/// give the CPU's scores too.
#[test]
fn levelzero_scores_match_the_reference_and_the_cpu_path() {
    let _t = turn();
    let Some(_) = gpu_device("levelzero_scores_match_the_reference_and_the_cpu_path") else { return };
    let f = fixture();
    let cases = f.cases();
    let pairs: Vec<_> = cases.iter().map(|(q, p)| (q.as_str(), p.as_str())).collect();
    let mut cpu = on(&f.dir, "cpu", turbo::TURBO_PRECISION_EXACT);
    let mut gpu = on(&f.dir, "levelzero", turbo::TURBO_PRECISION_EXACT);
    let got = gpu.score(&pairs, RerankOptions::default()).unwrap();
    let to_reference = worst(&got.scores, &f.reference());
    println!("tiny: max abs diff {to_reference:e} to the reference");
    assert!(to_reference < 1e-4);
    assert_eq!(got.model, cpu.identity().clone());
    let many: Vec<_> = pairs.iter().cycle().take(pairs.len() * 3).copied().collect();
    let windowed = RerankOptions { max_tokens: 16, stride: 4, ..Default::default() };
    for o in [RerankOptions::default(), windowed] {
        let (a, b) = (gpu.score(&many, o).unwrap().scores, cpu.score(&many, o).unwrap().scores);
        println!("tiny {o:?}: max abs diff {:e} to the cpu path", worst(&a, &b));
        assert!(worst(&a, &b) < 1e-5, "{o:?}: {a:?} vs {b:?}");
    }
    assert!(gpu.score(&[], RerankOptions::default()).unwrap().scores.is_empty());
}

/// On Level Zero the scores are the session's host memory, read back from
/// the device: the encoder's stage is the device's, and a buffer of them
/// exports its host pointer, not a USM pointer.
#[test]
fn held_rerank_scores_on_levelzero_export_their_host_pointer() {
    let _t = turn();
    let Some(_) = gpu_device("held_rerank_scores_on_levelzero_export_their_host_pointer") else { return };
    let f = fixture();
    held::held_scores(&on_gpu(&f.dir), turbo::TURBO_STAGE_DEVICE, turbo::TURBO_HANDLE_ZE_USM);
}

/// On Level Zero a run of more windows than max_batch counts every encoder
/// run's uploads and read-backs.
#[test]
fn rerank_transfers_on_levelzero_add_up_over_window_batches() {
    let _t = turn();
    let Some(_) = gpu_device("rerank_transfers_on_levelzero_add_up_over_window_batches") else { return };
    let f = fixture();
    held::transfers_add_up_over_window_batches(&on_gpu(&f.dir), &f.dir, true);
}

/// With `TURBO_TEST_RERANK_BUNDLES` naming real rerank bundles, comma
/// separated (docs/reranking.md), each one's reference pairs on Level Zero
/// at EXACT and MODEL within 1e-4 of the reference scores and the CPU path;
/// skipped without it.
#[test]
fn real_rerank_bundles_on_levelzero_match_their_reference_and_the_cpu_path() {
    let _t = turn();
    let test = "real_rerank_bundles_on_levelzero_match_their_reference_and_the_cpu_path";
    let Ok(dirs) = std::env::var("TURBO_TEST_RERANK_BUNDLES") else {
        println!("{test}: skipped: TURBO_TEST_RERANK_BUNDLES is not set");
        return;
    };
    let Some(_) = gpu_device(test) else { return };
    for dir in dirs.split(',').filter(|d| !d.is_empty()) {
        let dir = Path::new(dir);
        let bundle = Bundle::open(dir).unwrap();
        let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
        let reference = File::parse("rerank", &bytes).unwrap().get("scores", Dtype::F32, 2).unwrap().f32s();
        let cases: Vec<(&str, &str)> =
            bundle.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.passage.as_deref().unwrap())).collect();
        let cpu = on(dir, "cpu", turbo::TURBO_PRECISION_EXACT).score(&cases, RerankOptions::default()).unwrap().scores;
        for precision in [turbo::TURBO_PRECISION_EXACT, turbo::TURBO_PRECISION_MODEL] {
            let got = on(dir, "levelzero", precision).score(&cases, RerankOptions::default()).unwrap().scores;
            let (to_reference, to_cpu) = (worst(&got, &reference), worst(&got, &cpu));
            println!(
                "{}: precision {precision}: max abs diff {to_reference:e} to the reference, {to_cpu:e} to the cpu path",
                dir.display()
            );
            assert!(to_reference < 1e-4 && to_cpu < 1e-4, "{}: precision {precision}", dir.display());
        }
    }
}
