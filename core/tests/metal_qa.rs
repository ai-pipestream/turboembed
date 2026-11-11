//! Extractive QA on Metal with the span head on the device: the logits
//! against the CPU path and the transformers reference, and the read
//! holding the logits alone.
//!
//! Built with the `metal` feature only, on macOS. A test that needs a
//! device that runs the kernels says it was skipped, and passes, when the
//! backend lists none. With TURBO_TEST_REQUIRE_METAL=1 it fails instead.

#![cfg(feature = "metal")]

mod common;
mod metal_support;
#[path = "qa/common.rs"]
mod qa_common;

use std::path::Path;

use common::opts;
use metal_support::*;
use turbo::bundle::Bundle;
use turbo::qa::{Answer, DecodeOptions, ExtractiveQa, QaOptions, WindowLogits};
use turbo::safetensors::{Dtype, File};

fn fixture() -> qa_common::Fixture {
    qa_common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

fn flat(windows: &[Vec<WindowLogits>]) -> Vec<f32> {
    windows.iter().flatten().flat_map(|w| w.start.iter().zip(&w.end).flat_map(|(&s, &e)| [s, e])).collect()
}

/// The spans and null score of an answer, for comparing two.
fn spans(x: &Answer) -> (Vec<(std::ops::Range<usize>, f64)>, f64) {
    match x {
        Answer::Spans { answers, null_score, .. } => {
            (answers.iter().map(|s| (s.bytes.clone(), s.score)).collect(), *null_score)
        }
        Answer::None { null_score } => (Vec::new(), *null_score),
    }
}

/// Each case's windows on Metal at MODEL, one per run and all at once,
/// match the reference logits within the F32 score bound's 1e-4 and the
/// CPU path's within 1e-5; the decoded answers are the CPU path's.
#[test]
fn metal_logits_match_the_reference_and_the_cpu_path() {
    let _t = turn();
    let Some(_) = metal_device("metal_logits_match_the_reference_and_the_cpu_path") else { return };
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
    let cpu_logits = flat(&cpu.logits_many(&cases).unwrap());
    for max_batch in [1, 0] {
        let mut metal = on("metal", max_batch);
        let got = flat(&metal.logits_many(&cases).unwrap());
        assert_eq!(got.len(), reference.len());
        for (i, ((&a, &r), &c)) in got.iter().zip(&reference).zip(&cpu_logits).enumerate() {
            assert!((a - r).abs() < 1e-4, "max_batch {max_batch}, logit {i}: {a} vs reference {r}");
            assert!((a - c).abs() < 1e-5, "max_batch {max_batch}, logit {i}: {a} vs cpu {c}");
        }
        for &(q, c) in &cases {
            let o = DecodeOptions { null_threshold: -1e6, ..Default::default() };
            let ((sa, na), (sb, nb)) = (spans(&metal.answer(q, c, o).unwrap()), spans(&cpu.answer(q, c, o).unwrap()));
            assert!((na - nb).abs() < 1e-5, "{q:?}: null score {na} vs cpu {nb}");
            assert_eq!(sa.len(), sb.len(), "{q:?}");
            for ((ra, xa), (rb, xb)) in sa.iter().zip(&sb) {
                assert_eq!(ra, rb, "{q:?}: the same spans in the same order");
                assert!((xa - xb).abs() < 1e-5, "{q:?}: span score {xa} vs cpu {xb}");
            }
        }
    }
}

/// The span head runs on the device: a range write of a window's tokens
/// to the QA model on Metal returns a [start, end] row per token, 8 bytes
/// a token rather than its hidden state, which match the head applied in
/// F64 to the CPU backend's hidden states, at EXACT and MODEL. The CPU
/// path still returns the hidden states. A normalized or narrowed range
/// write is refused by its field, as the head reads a whole vector.
#[test]
fn metal_applies_the_span_head_and_returns_only_the_logits() {
    let _t = turn();
    let Some(_) = metal_device("metal_applies_the_span_head_and_returns_only_the_logits") else { return };
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
    let session = |l: &common::Loaded, precision: u32| {
        let mut d = common::session_desc(1, 0, precision);
        d.max_ranges = len as u32;
        common::Session::create(l.m, Some(&d)).unwrap()
    };
    let none = opts(|o| o.normalize = turbo::TURBO_NORMALIZE_NONE);

    let cpu = common::Loaded::load(&f.dir).unwrap();
    let s = session(&cpu, turbo::TURBO_PRECISION_EXACT);
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

    let l = on_metal(&f.dir);
    for precision in [turbo::TURBO_PRECISION_EXACT, turbo::TURBO_PRECISION_MODEL] {
        let s = session(&l, precision);
        s.write_ranges(&t.batch(), &ranges, Some(&none)).unwrap();
        let out = s.run().unwrap();
        let info = out.info();
        assert_eq!((info.batch, info.dim, info.bytes), (len as u32, 2, len as u64 * 8), "the logits alone");
        for (i, (g, w)) in out.rows().iter().zip(&want).enumerate() {
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
}
