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

/// With `TURBO_TEST_QA_BUNDLE` naming a real extractive QA bundle
/// (docs/extractive-qa.md), its reference cases on CUDA at EXACT and
/// MODEL against the bundle's transformers logits and the CPU path, with
/// windows batched; skipped without it.
#[test]
fn a_real_qa_bundle_on_cuda_matches_its_reference_and_the_cpu_path() {
    let _t = turn();
    let Ok(dir) = std::env::var("TURBO_TEST_QA_BUNDLE") else {
        println!(
            "a_real_qa_bundle_on_cuda_matches_its_reference_and_the_cpu_path: skipped: TURBO_TEST_QA_BUNDLE is not set"
        );
        return;
    };
    let Some(_) = cuda_device("a_real_qa_bundle_on_cuda_matches_its_reference_and_the_cpu_path") else { return };
    let dir = Path::new(&dir);
    let bundle = Bundle::open(dir).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let reference = File::parse("qa", &bytes).unwrap().get("logits", Dtype::F32, 2).unwrap().f32s();
    let cases: Vec<(&str, &str)> =
        bundle.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.context.as_deref().unwrap())).collect();
    let open = |backend: &str, precision: u32| {
        let o = QaOptions { backend: Some(backend.into()), precision, max_batch: 0, ..Default::default() };
        ExtractiveQa::open_with(dir, &o).unwrap_or_else(|e| panic!("{backend}: {e:?}"))
    };
    let cpu = flat(&open("cpu", turbo::TURBO_PRECISION_EXACT).logits_many(&cases).unwrap());
    let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    for (precision, tolerance) in [(turbo::TURBO_PRECISION_EXACT, 1e-4), (turbo::TURBO_PRECISION_MODEL, 1e-4)] {
        let got = flat(&open("cuda", precision).logits_many(&cases).unwrap());
        assert_eq!(got.len(), reference.len());
        let (to_reference, to_cpu) = (worst(&got, &reference), worst(&got, &cpu));
        println!("precision {precision}: max abs diff {to_reference:e} to the reference, {to_cpu:e} to the cpu path");
        assert!(to_reference < tolerance && to_cpu < tolerance, "precision {precision}");
    }
}

/// `ExtractiveQa::open`, the bundle check's and the annotator's path,
/// stays on the CPU in a build with CUDA: the QA cells tie as
/// experimental, so device order would otherwise pick the GPU.
#[test]
fn open_stays_on_the_cpu_in_a_cuda_build() {
    let _t = turn();
    let Some(_) = cuda_device("open_stays_on_the_cpu_in_a_cuda_build") else { return };
    let f = fixture();
    let qa = ExtractiveQa::open(&f.dir, 1).unwrap();
    let b = qa.device_info().unwrap().backend;
    let name: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    assert_eq!(name, b"cpu");
}

/// The span head runs on the device: a range write of a window's tokens
/// to the QA model on CUDA returns a [start, end] row per token, which
/// match the head applied in F64 to the CPU backend's hidden states, and
/// reading them copies 8 bytes a token, not the token's hidden state. The
/// CPU path still returns the hidden states. A normalized or narrowed
/// range write is refused by its field, as the head reads a whole vector.
#[test]
fn cuda_applies_the_span_head_and_copies_back_only_the_logits() {
    let _t = turn();
    let Some(_) = cuda_device("cuda_applies_the_span_head_and_copies_back_only_the_logits") else { return };
    let f = fixture();
    let bundle = Bundle::open(&f.dir).unwrap();
    let head = turbo::head::Head::load(&bundle).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let file = File::parse("qa", &bytes).unwrap();
    let ids = file.get("ids", Dtype::I32, 2).unwrap();
    let width = ids.shape[1] as usize;
    let len = file.get("lengths", Dtype::I32, 1).unwrap().i32s()[0] as usize;
    let types = file.get("types", Dtype::I32, 2).unwrap().i32s()[..len].to_vec();
    let mut t = common::Tokens::new(&[ids.i32s()[..len].to_vec()], 0);
    t.types = Some(types);
    assert!(len <= width);
    let ranges: Vec<_> = (0..len as u32).map(|i| common::range(0, i, i + 1)).collect();
    let session = |l: &common::Loaded| {
        let mut d = common::session_desc(1, 0, turbo::TURBO_PRECISION_EXACT);
        d.max_ranges = len as u32;
        common::Session::create(l.m, Some(&d)).unwrap()
    };
    let none = opts(|o| o.normalize = turbo::TURBO_NORMALIZE_NONE);

    let cpu = common::Loaded::load(&f.dir).unwrap();
    let s = session(&cpu);
    s.write_ranges(&t.batch(), &ranges, Some(&none)).unwrap();
    let states = s.run().unwrap().rows();
    assert_eq!(states[0].len(), head.hidden, "the cpu path returns the hidden states");
    let h = &head.classifier;
    let want: Vec<[f32; 2]> = states
        .iter()
        .map(|x| {
            std::array::from_fn(|o| {
                let w = &h.weight[o * h.in_dim..(o + 1) * h.in_dim];
                w.iter().zip(x).fold(f64::from(h.bias[o]), |a, (&w, &v)| a + f64::from(w) * f64::from(v)) as f32
            })
        })
        .collect();

    let l = on_cuda(&f.dir);
    let s = session(&l);
    s.write_ranges(&t.batch(), &ranges, Some(&none)).unwrap();
    let out = s.run().unwrap();
    let info = out.info();
    assert_eq!((info.batch, info.dim, info.bytes), (len as u32, 2, len as u64 * 8));
    let got = out.rows();
    assert_eq!(out.info().d2h_bytes, len as u64 * 8, "the read copies the logits alone");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        for o in 0..2 {
            assert!((g[o] - w[o]).abs() < 1e-5, "token {i} logit {o}: {} vs the cpu path's {}", g[o], w[o]);
        }
    }
    drop(out);

    let l2 = opts(|o| o.normalize = turbo::TURBO_NORMALIZE_L2);
    let e = s.write_ranges(&t.batch(), &ranges, Some(&l2)).unwrap_err();
    assert_eq!((e.code, e.field), (turbo::status::UNSUPPORTED_OPTION, 4), "{e:?}");
    let narrow = opts(|o| {
        o.normalize = turbo::TURBO_NORMALIZE_NONE;
        o.output_dim = 8;
    });
    let e = s.write_ranges(&t.batch(), &ranges, Some(&narrow)).unwrap_err();
    assert_eq!((e.code, e.field), (turbo::status::UNSUPPORTED_OPTION, 6), "{e:?}");
}
