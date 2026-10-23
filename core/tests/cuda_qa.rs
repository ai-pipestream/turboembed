//! Extractive QA on CUDA against the transformers reference logits and the
//! CPU backend, one window per run and windows batched.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none. With
//! TURBO_TEST_REQUIRE_CUDA=1 it fails instead.

#![cfg(feature = "cuda")]

mod common;
#[path = "cuda_support/mod.rs"]
mod cuda_support;
#[path = "qa/common.rs"]
mod qa_common;

use std::path::Path;

use cuda_support::*;
use turbo::bundle::Bundle;
use turbo::qa::{Answer, DecodeOptions, ExtractiveQa, QaOptions, WindowLogits};
use turbo::safetensors::{Dtype, File};

fn fixture() -> qa_common::Fixture {
    qa_common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

fn flat(windows: &[Vec<WindowLogits>]) -> Vec<f32> {
    windows.iter().flatten().flat_map(|w| w.start.iter().zip(&w.end).flat_map(|(&s, &e)| [s, e])).collect()
}

/// Each case's windows on CUDA, one per run and all at once, match the
/// reference logits within the bundle check's 1e-4, and the CPU path's
/// within 1e-5; the decoded answers are the CPU path's.
#[test]
fn cuda_logits_match_the_reference_and_the_cpu_path() {
    let _t = turn();
    let Some(_) = cuda_device("cuda_logits_match_the_reference_and_the_cpu_path") else { return };
    let f = fixture();
    let bundle = Bundle::open(&f.dir).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let reference = File::parse("qa", &bytes).unwrap().get("logits", Dtype::F32, 2).unwrap().f32s();
    let cases: Vec<(&str, &str)> =
        bundle.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.context.as_deref().unwrap())).collect();
    let on = |backend: &str, max_batch: u32| {
        let o = QaOptions { backend: Some(backend.into()), max_batch, ..Default::default() };
        ExtractiveQa::open_with(&f.dir, &o).unwrap_or_else(|e| panic!("{backend}: {e:?}"))
    };
    let mut cpu = on("cpu", 1);
    let cpu_logits = cpu.logits_many(&cases).unwrap();
    for max_batch in [1, 0] {
        let mut cuda = on("cuda", max_batch);
        let got = cuda.logits_many(&cases).unwrap();
        let (got, want, cpu_flat) = (flat(&got), &reference, flat(&cpu_logits));
        assert_eq!(got.len(), want.len());
        for (i, ((&a, &r), &c)) in got.iter().zip(want).zip(&cpu_flat).enumerate() {
            assert!((a - r).abs() < 1e-4, "max_batch {max_batch}, logit {i}: {a} vs reference {r}");
            assert!((a - c).abs() < 1e-5, "max_batch {max_batch}, logit {i}: {a} vs cpu {c}");
        }
        for &(q, c) in &cases {
            let o = DecodeOptions { null_threshold: -1e6, ..Default::default() };
            let (a, b) = (cuda.answer(q, c, o).unwrap(), cpu.answer(q, c, o).unwrap());
            let spans = |x: &Answer| match x {
                Answer::Spans { answers, null_score, .. } => {
                    (answers.iter().map(|s| (s.bytes.clone(), s.score)).collect::<Vec<_>>(), *null_score)
                }
                Answer::None { null_score } => (Vec::new(), *null_score),
            };
            let ((sa, na), (sb, nb)) = (spans(&a), spans(&b));
            assert!((na - nb).abs() < 1e-5, "{q:?}: null score {na} vs cpu {nb}");
            assert_eq!(sa.len(), sb.len(), "{q:?}");
            for ((ra, xa), (rb, xb)) in sa.iter().zip(&sb) {
                assert_eq!(ra, rb, "{q:?}: the same spans in the same order");
                assert!((xa - xb).abs() < 1e-5, "{q:?}: span score {xa} vs cpu {xb}");
            }
        }
    }
}

/// The capability cell says CUDA runs the task, as experimental.
#[test]
fn cuda_offers_extractive_qa_as_experimental() {
    let _t = turn();
    let Some(device) = cuda_device("cuda_offers_extractive_qa_as_experimental") else { return };
    let rt = Rt::new();
    let mut cap: turbo::turbo_capability = unsafe { std::mem::zeroed() };
    cap.struct_size = size_of::<turbo::turbo_capability>() as u32;
    let mut e = common::new_error();
    let rc = unsafe {
        turbo::turbo_runtime_capability(
            rt.0,
            device,
            turbo::TURBO_TASK_EXTRACTIVE_QA,
            turbo::TURBO_PRECISION_EXACT,
            &mut cap,
            &mut e,
        )
    };
    assert_eq!(rc, 0);
    // TURBO_CAP_EXPERIMENTAL, turbo.h.
    assert_eq!(cap.status, 1);
}
