//! Extractive QA on Level Zero against the transformers reference logits
//! and the CPU backend, one window per run and windows batched, with the
//! span head on the device, as tests/cuda_qa.rs holds CUDA.
//!
//! Built with the `levelzero` feature only. A test that needs a device says
//! it was skipped, and passes, when the backend lists none. With
//! TURBO_TEST_REQUIRE_LEVELZERO=1 it fails instead.

#![cfg(feature = "levelzero")]

mod common;
#[path = "qa/common.rs"]
mod qa_common;

use std::path::Path;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::{Loaded, first_of, new_error, opts};
use turbo::bundle::Bundle;
use turbo::qa::{Answer, DecodeOptions, ExtractiveQa, QaOptions, WindowLogits};
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

fn fixture() -> qa_common::Fixture {
    qa_common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

fn flat(windows: &[Vec<WindowLogits>]) -> Vec<f32> {
    windows.iter().flatten().flat_map(|w| w.start.iter().zip(&w.end).flat_map(|(&s, &e)| [s, e])).collect()
}

fn open(dir: &Path, backend: &str, precision: u32, max_batch: u32) -> ExtractiveQa {
    let o = QaOptions { backend: Some(backend.into()), precision, max_batch, ..Default::default() };
    ExtractiveQa::open_with(dir, &o).unwrap_or_else(|e| panic!("{backend}: {e:?}"))
}

/// The reference cases of the bundle at `dir`, and their logits.
fn reference(dir: &Path) -> (Vec<(String, String)>, Vec<f32>) {
    let bundle = Bundle::open(dir).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let logits = File::parse("qa", &bytes).unwrap().get("logits", Dtype::F32, 2).unwrap().f32s();
    let cases = bundle.manifest.reference.cases.iter().map(|c| (c.text.clone(), c.context.clone().unwrap())).collect();
    (cases, logits)
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

/// Each case's windows on Level Zero, one per run and all at once, match
/// the reference logits within the bundle check's 1e-4, and the CPU path's
/// within 1e-5; the decoded answers are the CPU path's.
#[test]
fn levelzero_logits_match_the_reference_and_the_cpu_path() {
    let _t = turn();
    let Some(_) = gpu_device("levelzero_logits_match_the_reference_and_the_cpu_path") else { return };
    let f = fixture();
    let (cases, want) = reference(&f.dir);
    let cases: Vec<(&str, &str)> = cases.iter().map(|(q, c)| (q.as_str(), c.as_str())).collect();
    let mut cpu = open(&f.dir, "cpu", turbo::TURBO_PRECISION_EXACT, 1);
    let cpu_flat = flat(&cpu.logits_many(&cases).unwrap());
    for max_batch in [1, 0] {
        let mut gpu = open(&f.dir, "levelzero", turbo::TURBO_PRECISION_EXACT, max_batch);
        let got = flat(&gpu.logits_many(&cases).unwrap());
        let (to_reference, to_cpu) = (worst(&got, &want), worst(&got, &cpu_flat));
        println!("max_batch {max_batch}: max abs diff {to_reference:e} to the reference, {to_cpu:e} to the cpu path");
        assert!(to_reference < 1e-4 && to_cpu < 1e-5, "max_batch {max_batch}");
        for &(q, c) in &cases {
            let o = DecodeOptions { null_threshold: -1e6, ..Default::default() };
            let spans = |x: Answer| match x {
                Answer::Spans { answers, null_score, .. } => {
                    (answers.into_iter().map(|s| (s.bytes, s.score)).collect::<Vec<_>>(), null_score)
                }
                Answer::None { null_score } => (Vec::new(), null_score),
            };
            let ((sa, na), (sb, nb)) = (spans(gpu.answer(q, c, o).unwrap()), spans(cpu.answer(q, c, o).unwrap()));
            assert!((na - nb).abs() < 1e-5, "{q:?}: null score {na} vs cpu {nb}");
            assert_eq!(sa.len(), sb.len(), "{q:?}");
            for ((ra, xa), (rb, xb)) in sa.iter().zip(&sb) {
                assert_eq!(ra, rb, "{q:?}: the same spans in the same order");
                assert!((xa - xb).abs() < 1e-5, "{q:?}: span score {xa} vs cpu {xb}");
            }
        }
    }
}

/// With `TURBO_TEST_QA_BUNDLE` naming a real extractive QA bundle
/// (docs/extractive-qa.md), its reference cases on Level Zero at EXACT and
/// MODEL within 1e-4 of the bundle's transformers logits and the CPU path,
/// with windows batched; skipped without it.
#[test]
fn a_real_qa_bundle_on_levelzero_matches_its_reference_and_the_cpu_path() {
    let _t = turn();
    let test = "a_real_qa_bundle_on_levelzero_matches_its_reference_and_the_cpu_path";
    let Ok(dir) = std::env::var("TURBO_TEST_QA_BUNDLE") else {
        println!("{test}: skipped: TURBO_TEST_QA_BUNDLE is not set");
        return;
    };
    let Some(_) = gpu_device(test) else { return };
    let dir = turbo_pins::require(&dir);
    let (cases, want) = reference(&dir);
    let cases: Vec<(&str, &str)> = cases.iter().map(|(q, c)| (q.as_str(), c.as_str())).collect();
    let cpu = flat(&open(&dir, "cpu", turbo::TURBO_PRECISION_EXACT, 0).logits_many(&cases).unwrap());
    for precision in [turbo::TURBO_PRECISION_EXACT, turbo::TURBO_PRECISION_MODEL] {
        let got = flat(&open(&dir, "levelzero", precision, 0).logits_many(&cases).unwrap());
        let (to_reference, to_cpu) = (worst(&got, &want), worst(&got, &cpu));
        println!("precision {precision}: max abs diff {to_reference:e} to the reference, {to_cpu:e} to the cpu path");
        assert!(to_reference < 1e-4 && to_cpu < 1e-4, "precision {precision}");
    }
}

/// The span head runs on the device: a range write of a window's tokens
/// to the QA model on Level Zero returns a [start, end] row per token,
/// which match the head applied in F64 to the CPU backend's hidden states,
/// and reading them copies 8 bytes a token, not the token's hidden state.
/// A normalized or narrowed range write is refused by its field, as the
/// head reads a whole vector.
#[test]
fn levelzero_applies_the_span_head_and_copies_back_only_the_logits() {
    let _t = turn();
    let Some(_) = gpu_device("levelzero_applies_the_span_head_and_copies_back_only_the_logits") else { return };
    let f = fixture();
    let bundle = Bundle::open(&f.dir).unwrap();
    let head = turbo::head::Head::load(&bundle).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let file = File::parse("qa", &bytes).unwrap();
    let ids = file.get("ids", Dtype::I32, 2).unwrap();
    let len = file.get("lengths", Dtype::I32, 1).unwrap().i32s()[0] as usize;
    let mut t = common::Tokens::new(&[ids.i32s()[..len].to_vec()], 0);
    t.types = Some(file.get("types", Dtype::I32, 2).unwrap().i32s()[..len].to_vec());
    let ranges: Vec<_> = (0..len as u32).map(|i| common::range(0, i, i + 1)).collect();
    let session = |l: &Loaded| {
        let mut d = common::session_desc(1, 0, turbo::TURBO_PRECISION_EXACT);
        d.max_ranges = len as u32;
        common::Session::create(l.m, Some(&d)).unwrap()
    };
    let none = opts(|o| o.normalize = turbo::TURBO_NORMALIZE_NONE);

    let cpu = Loaded::load(&f.dir).unwrap();
    let s = session(&cpu);
    s.write_ranges(&t.batch(), &ranges, Some(&none)).unwrap();
    let states = s.run().unwrap().rows();
    assert_eq!(states[0].len(), head.hidden, "the cpu path returns the hidden states");
    let h = &head.classifier;
    let want: Vec<[f32; 2]> = states
        .iter()
        .map(|x| {
            std::array::from_fn(|o| {
                let w = &h.weight()[o * h.in_dim..(o + 1) * h.in_dim];
                w.iter().zip(x).fold(f64::from(h.bias()[o]), |a, (&w, &v)| a + f64::from(w) * f64::from(v)) as f32
            })
        })
        .collect();

    let l = Loaded::load_on(&f.dir, |rt| first_of(rt, "levelzero").unwrap()).unwrap();
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
