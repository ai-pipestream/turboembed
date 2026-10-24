//! Cross-encoder rerank on CUDA against the reference scores and the CPU
//! backend: pairs batched, passages windowed, and the max over windows.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none. With
//! TURBO_TEST_REQUIRE_CUDA=1 it fails instead.

#![cfg(feature = "cuda")]

mod common;
#[path = "cuda_support/mod.rs"]
mod cuda_support;
#[path = "rerank/held.rs"]
mod held;
#[path = "rerank/common.rs"]
mod rerank_common;

use std::path::Path;

use cuda_support::*;
use turbo::bundle::Bundle;
use turbo::rerank::{RerankOptions, RerankSessionOptions, Reranker};
use turbo::safetensors::{Dtype, File};

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

/// The tiny model's pairs on CUDA within 1e-4 of the reference and 1e-5 of
/// the CPU; more pairs than a run holds, and overlapping windows, give the
/// CPU's scores too.
#[test]
fn cuda_scores_match_the_reference_and_the_cpu_path() {
    let _t = turn();
    let Some(_) = cuda_device("cuda_scores_match_the_reference_and_the_cpu_path") else { return };
    let f = fixture();
    let cases = f.cases();
    let pairs: Vec<_> = cases.iter().map(|(q, p)| (q.as_str(), p.as_str())).collect();
    let mut cpu = on(&f.dir, "cpu", turbo::TURBO_PRECISION_EXACT);
    let mut cuda = on(&f.dir, "cuda", turbo::TURBO_PRECISION_EXACT);
    let got = cuda.score(&pairs, RerankOptions::default()).unwrap();
    assert!(worst(&got.scores, &f.reference()) < 1e-4);
    assert_eq!(got.model, cpu.identity().clone());
    let many: Vec<_> = pairs.iter().cycle().take(pairs.len() * 3).copied().collect();
    let windowed = RerankOptions { max_tokens: 16, stride: 4, ..Default::default() };
    for o in [RerankOptions::default(), windowed] {
        let (a, b) = (cuda.score(&many, o).unwrap().scores, cpu.score(&many, o).unwrap().scores);
        assert!(worst(&a, &b) < 1e-5, "{o:?}: {a:?} vs {b:?}");
    }
    assert!(cuda.score(&[], RerankOptions::default()).unwrap().scores.is_empty());
}

/// On CUDA the scores are the session's host memory, read back from the
/// device: the encoder's stage is the device's, and a buffer of them
/// exports its host pointer, not a CUDA pointer.
#[test]
fn held_rerank_scores_on_cuda_export_their_host_pointer() {
    let _t = turn();
    let Some(_) = cuda_device("held_rerank_scores_on_cuda_export_their_host_pointer") else { return };
    let f = fixture();
    held::held_scores(&on_cuda(&f.dir), turbo::TURBO_STAGE_DEVICE, turbo::TURBO_HANDLE_CUDA_PTR);
}

/// On CUDA a run of more windows than max_batch counts every encoder run's
/// uploads and read-backs.
#[test]
fn rerank_transfers_on_cuda_add_up_over_window_batches() {
    let _t = turn();
    let Some(_) = cuda_device("rerank_transfers_on_cuda_add_up_over_window_batches") else { return };
    let f = fixture();
    held::transfers_add_up_over_window_batches(&on_cuda(&f.dir), &f.dir);
}

/// `Reranker::open` stays on the CPU in a build with CUDA, and the CUDA
/// cell reports rerank as experimental.
#[test]
fn open_stays_on_the_cpu_and_cuda_reports_experimental() {
    let _t = turn();
    let Some(device) = cuda_device("open_stays_on_the_cpu_and_cuda_reports_experimental") else { return };
    let f = fixture();
    let r = Reranker::open(&f.dir, 1).unwrap();
    assert_eq!(r.backend(), "cpu");
    let rt = Rt::new();
    let mut cap: turbo::turbo_capability = unsafe { std::mem::zeroed() };
    cap.struct_size = size_of::<turbo::turbo_capability>() as u32;
    let mut e = common::new_error();
    let rc = unsafe {
        turbo::turbo_runtime_capability(
            rt.0,
            device,
            turbo::TURBO_TASK_RERANK,
            turbo::TURBO_PRECISION_EXACT,
            &mut cap,
            &mut e,
        )
    };
    assert_eq!(rc, 0);
    // TURBO_CAP_EXPERIMENTAL, turbo.h.
    assert_eq!(cap.status, 1);
}

/// With `TURBO_TEST_RERANK_BUNDLES` naming real rerank bundles, comma
/// separated (docs/reranking.md), each one's reference pairs on CUDA at
/// EXACT and MODEL within 1e-4 of the reference scores and the CPU path;
/// skipped without it.
#[test]
fn real_rerank_bundles_on_cuda_match_their_reference_and_the_cpu_path() {
    let _t = turn();
    let test = "real_rerank_bundles_on_cuda_match_their_reference_and_the_cpu_path";
    let Ok(dirs) = std::env::var("TURBO_TEST_RERANK_BUNDLES") else {
        println!("{test}: skipped: TURBO_TEST_RERANK_BUNDLES is not set");
        return;
    };
    let Some(_) = cuda_device(test) else { return };
    for dir in dirs.split(',').filter(|d| !d.is_empty()) {
        let dir = Path::new(dir);
        let bundle = Bundle::open(dir).unwrap();
        let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
        let reference = File::parse("rerank", &bytes).unwrap().get("scores", Dtype::F32, 2).unwrap().f32s();
        let cases: Vec<(&str, &str)> =
            bundle.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.passage.as_deref().unwrap())).collect();
        let cpu = on(dir, "cpu", turbo::TURBO_PRECISION_EXACT).score(&cases, RerankOptions::default()).unwrap().scores;
        for precision in [turbo::TURBO_PRECISION_EXACT, turbo::TURBO_PRECISION_MODEL] {
            let got = on(dir, "cuda", precision).score(&cases, RerankOptions::default()).unwrap().scores;
            let (to_reference, to_cpu) = (worst(&got, &reference), worst(&got, &cpu));
            println!(
                "{}: precision {precision}: max abs diff {to_reference:e} to the reference, {to_cpu:e} to the cpu path",
                dir.display()
            );
            assert!(to_reference < 1e-4 && to_cpu < 1e-4, "{}: precision {precision}", dir.display());
        }
    }
}
