//! turbo_embed_write_layer_ranges: each range from each of the last
//! layers in one pass, its last block the plain range write's bit for bit
//! on every encoder family the cpu backend reads the layers of (the sealed
//! BERT, ModernBERT, DeBERTa-v2 and nomic-bert bundles, and the small BERT
//! as RoBERTa), the run's shape, and the refusals turbo.h lists, with a
//! count over the model's layers refused alike on a device that is not
//! the CPU. Each block against a model cut after its layer is the cpu
//! encoder's own test (cpu/encoder/taps_tests.rs).

use std::path::Path;

use crate::common::*;
use crate::support::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

/// The ranges' vectors from the last `layers` layers, and from the plain
/// range write, on the bundle at `dir`, of `deep` layers.
fn agree(dir: &Path, deep: u32) {
    let bundle = dir.display();
    let l = Loaded::load(dir).unwrap_or_else(|e| panic!("{bundle}: {e:?}"));
    let (t, lens) = tokens_of(dir, &RANGES_TEXTS);
    let ranges = [range(0, 0, lens[0] as u32), range(1, 1, 4), range(2, 0, 2)];
    let s = session_of(&l, 3 * deep);
    let plain = run_ranges(&s, &t, &ranges, &embed_options());
    for layers in 1..=deep {
        s.write_layer_ranges(&t.batch(), &ranges, layers, None).unwrap_or_else(|e| panic!("{e:?}"));
        let r = s.run().unwrap();
        assert_eq!(r.info().batch, 3 * layers, "{bundle}: {layers} layers");
        let rows = r.rows();
        drop(r);
        let last = &rows[3 * (layers as usize - 1)..];
        assert_eq!(last.iter().map(|v| bits(v)).collect::<Vec<_>>(), plain.iter().map(|v| bits(v)).collect::<Vec<_>>());
        for (k, block) in rows.chunks(3).enumerate().take(layers as usize - 1) {
            assert!(block.iter().flatten().all(|v| v.is_finite()), "{bundle}: tap {k}");
            assert_ne!(block, last, "{bundle}: tap {k} of {layers} is an earlier layer's");
        }
        let via_opts = run_ranges(&s, &t, &ranges, &turbo_embed_options { layers, ..embed_options() });
        assert_eq!(
            via_opts.iter().map(|v| bits(v)).collect::<Vec<_>>(),
            rows.iter().map(|v| bits(v)).collect::<Vec<_>>(),
            "{bundle}: turbo_embed_options.layers = {layers}"
        );
    }
}

#[test]
fn the_last_layer_read_is_the_plain_range_write() {
    agree(&testdata().join("tiny-bert-bundle"), 2);
    agree(&testdata().join("tiny-modernbert-bundle"), 3);
    agree(&testdata().join("tiny-deberta-bundle"), 2);
    agree(&testdata().join("tiny-nomic-bundle"), 2);
    let f = roberta("layers-roberta");
    agree(f.write(), 2);
}

/// The small BERT as RoBERTa, its positions read from row 2 of the table.
fn roberta(name: &str) -> Fixture {
    let mut m = model_manifest();
    m["architecture"]["family"] = json!("FAMILY_ROBERTA");
    m["architecture"]["position_offset"] = json!(2);
    let mut f = Fixture::new(name, m);
    f.weights("weights/model.safetensors", &tiny_weights(0));
    f
}

#[test]
fn a_layer_range_write_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = session_of(&l, 4);
    let (t, _) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    let good = [range(0, 0, 2), range(1, 0, 3)];
    let refused = |ranges: &[turbo_token_range], layers: u32| -> Failure {
        s.write_ranges(&t.batch(), &good, None).unwrap();
        let e = s.write_layer_ranges(&t.batch(), ranges, layers, None).expect_err("the write is refused");
        assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "{e:?}");
        e
    };
    let e = refused(&good, 0);
    assert_eq!(e.code, INVALID_ARGUMENT, "{e:?}");
    let e = refused(&good, 3);
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 9), "past the model's 2 layers: {e:?}");
    assert!(e.message.contains("1 to 2"), "{e:?}");
    let e = s
        .write_ranges(&t.batch(), &good, Some(&turbo_embed_options { layers: 3, ..embed_options() }))
        .expect_err("past the model's 2 layers via options");
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 9), "{e:?}");
    let e = s
        .write_text(&["hello"], Some(&turbo_embed_options { layers: 2, ..embed_options() }))
        .expect_err("layers on text write");
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 9), "{e:?}");
    let e = s
        .write_tokens(&t.batch(), Some(&turbo_embed_options { layers: 2, ..embed_options() }))
        .expect_err("layers on token write");
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 9), "{e:?}");
    let e = refused(&[good[0], good[1], good[0]], 2);
    assert_eq!(e.code, CAPACITY, "six vectors over max_ranges 4: {e:?}");
    assert!(e.message.contains("is 6 vectors"), "{e:?}");
    s.write_layer_ranges(&t.batch(), &good, 2, None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 4);

    // A 40-byte V2 turbo_embed_options (before layers was added) defaults layers to 0.
    assert_eq!(TURBO_EMBED_OPTIONS_SIZE_V2, 40);
    let mut v2 = embed_options();
    v2.struct_size = TURBO_EMBED_OPTIONS_SIZE_V2 as u32;
    v2.layers = 99;
    s.write_ranges(&t.batch(), &good, Some(&v2)).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 2);

    // A static embedding model has no encoder layers to tap.
    let static_dir = testdata().join("tiny-static-bundle");
    let st = Loaded::load(&static_dir).unwrap_or_else(|e| panic!("{e:?}"));
    let s = session_of(&st, 4);
    let (t, _) = tokens_of(&static_dir, &RANGES_TEXTS);
    let e = s.write_ranges(&t.batch(), &good, Some(&turbo_embed_options { layers: 2, ..embed_options() })).unwrap_err();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 9), "{e:?}");
    s.write_ranges(&t.batch(), &good, Some(&turbo_embed_options { layers: 1, ..embed_options() })).unwrap();
}

/// The backends whose embed cell sets field 9's bit.
const READS_LAYERS: [&str; 3] = ["cpu", "metal", "levelzero"];

/// The device backends this build lists, failing where
/// TURBO_TEST_REQUIRE_<BACKEND>=1 names one it does not.
fn devices() -> Vec<&'static str> {
    ["cuda", "metal", "levelzero"]
        .into_iter()
        .filter(|b| {
            let required = std::env::var(format!("TURBO_TEST_REQUIRE_{}", b.to_uppercase())).is_ok_and(|v| v == "1");
            assert!(
                !required || listed(b),
                "TURBO_TEST_REQUIRE_{}=1 and the backend lists no device",
                b.to_uppercase()
            );
            listed(b)
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

/// Each range from every layer of the bundle at `dir`, of `deep` layers,
/// on `backend` at `precision` against the CPU at EXACT, normalized and
/// not: the last block is the device's own plain range write bit for bit,
/// and every vector within docs/conformance.md's bound for the session's
/// compute dtype (a cosine of 0.9999 and a largest difference of 1e-4 of
/// the vector's largest value in F32, a cosine of 0.999 in F16).
fn agree_on(backend: &str, precision: u32, dir: &Path, deep: u32) {
    let what = format!("{backend}, precision {precision}, {}", dir.display());
    let (t, lens) = tokens_of(dir, &RANGES_TEXTS);
    let ranges = [range(0, 0, lens[0] as u32), range(1, 1, 4), range(2, 0, 2)];
    let cpu = Loaded::load(dir).unwrap_or_else(|e| panic!("{what}: {e:?}"));
    let cs = session_of(&cpu, 3 * deep);
    let l = Loaded::load_on(dir, |rt| first_of(rt, backend).unwrap()).unwrap_or_else(|e| panic!("{what}: {e:?}"));
    let mut d = session_desc(0, 0, precision);
    d.max_ranges = 3 * deep;
    // A backend that runs a family in F32 alone refuses FASTEST for it,
    // naming the precision; any other failure fails the test.
    let s = match Session::create(l.m, Some(&d)) {
        Ok(s) => s,
        Err(e) if precision == TURBO_PRECISION_FASTEST && (e.code, e.field) == (UNSUPPORTED_OPTION, 3) => {
            return println!("{what}: skipped: {e:?}");
        }
        Err(e) => panic!("{what}: {e:?}"),
    };
    let dtype = s.info().compute_dtype;
    // Every count of layers above 1: the first layers tapped or not, and
    // ModernBERT's final_norm on each block.
    for (normalize, taps) in [0, TURBO_NORMALIZE_NONE].into_iter().flat_map(|n| (2..=deep).map(move |k| (n, k))) {
        let what = format!("{what}, {taps} layers");
        let o = turbo_embed_options { normalize, ..embed_options() };
        let plain = run_ranges(&s, &t, &ranges, &o);
        let o = turbo_embed_options { layers: taps, ..o };
        let (got, want) = (run_ranges(&s, &t, &ranges, &o), run_ranges(&cs, &t, &ranges, &o));
        assert_eq!(got.len(), 3 * taps as usize, "{what}");
        let last = got[3 * (taps as usize - 1)..].iter().map(|v| bits(v)).collect::<Vec<_>>();
        assert_eq!(last, plain.iter().map(|v| bits(v)).collect::<Vec<_>>(), "{what}: the last block");
        let mut worst = (1.0f64, 0.0f64);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = g.iter().zip(w).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            worst = (worst.0.min(cosine(g, w)), worst.1.max(f64::from(diff)));
            let (floor, most) = if dtype == TURBO_DTYPE_F32 { (0.9999, 1e-4) } else { (0.999, f64::INFINITY) };
            assert!(worst.0 >= floor && worst.1 <= most, "{what}, normalize {normalize}, vector {i}: {worst:?}");
        }
        println!(
            "{what}, normalize {normalize}: 1 - min cosine {:.3e}, max diff {:.3e} of the scale",
            1.0 - worst.0,
            worst.1
        );
    }
}

/// Every device whose cell sets field 9's bit, on every family the CPU
/// reads the layers of, at MODEL and FASTEST, against the CPU; a device
/// that is not listed is skipped, unless TURBO_TEST_REQUIRE_<BACKEND>=1.
#[test]
fn every_layer_on_a_device_matches_the_cpu() {
    for backend in devices().into_iter().filter(|b| READS_LAYERS.contains(b)) {
        for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
            agree_on(backend, precision, &testdata().join("tiny-bert-bundle"), 2);
            agree_on(backend, precision, &testdata().join("tiny-modernbert-bundle"), 3);
            agree_on(backend, precision, &testdata().join("tiny-deberta-bundle"), 2);
            agree_on(backend, precision, &testdata().join("tiny-nomic-bundle"), 2);
            let f = roberta("layers-roberta-devices");
            agree_on(backend, precision, f.write(), 2);
        }
    }
}

/// Whether this build lists a device of `backend`.
fn listed(backend: &str) -> bool {
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(std::ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, backend);
    unsafe { turbo_runtime_release(rt) };
    d.is_some()
}

/// The small BERT's layer counts on the CPU and on each device this build
/// lists that is not one: 3, over its 2 layers, is INVALID_ARGUMENT
/// naming field 9 on every device, and 2 runs exactly where the device's
/// cell sets field 9's bit, and is UNSUPPORTED_OPTION naming it elsewhere.
/// Without such a device the order is the session's own unit test
/// (session/tests.rs).
#[test]
fn a_count_over_the_model_is_refused_alike_on_every_device() {
    let dir = tiny_bundle();
    let (t, _) = tokens_of(&dir, &RANGES_TEXTS);
    let good = [range(0, 0, 2), range(1, 0, 3)];
    let mut devices = devices();
    if devices.is_empty() {
        println!("no device but the CPU: the order is pinned by session/tests.rs alone");
    }
    devices.push("cpu");
    for backend in devices {
        let l = Loaded::load_on(&dir, |rt| first_of(rt, backend).expect("listed"))
            .unwrap_or_else(|e| panic!("{backend}: {e:?}"));
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        let at = first_of(l.rt, backend).unwrap();
        let rc = unsafe {
            turbo_runtime_capability(l.rt, at, TURBO_TASK_EMBED, TURBO_PRECISION_MODEL, &mut cap, &mut new_error())
        };
        assert_eq!(rc, 0, "{backend}");
        let reads = cap.options_honored & 1 << 8 != 0;
        assert_eq!(reads, READS_LAYERS.contains(&backend), "{backend}: which backends read the earlier layers");
        let s = session_of(&l, 0);
        let over = s.write_layer_ranges(&t.batch(), &good, 3, None).unwrap_err();
        assert_eq!((over.code, over.field), (INVALID_ARGUMENT, 9), "{backend}: {over:?}");
        let o = turbo_embed_options { layers: 3, ..embed_options() };
        let over = s.write_ranges(&t.batch(), &good, Some(&o)).unwrap_err();
        assert_eq!((over.code, over.field), (INVALID_ARGUMENT, 9), "{backend}: {over:?}");
        match s.write_layer_ranges(&t.batch(), &good, 2, None) {
            Ok(()) => assert!(reads, "{backend}: two layers run without field 9's bit"),
            Err(e) => {
                assert!(!reads, "{backend}: {e:?} with field 9's bit set");
                assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 9), "{backend}: {e:?}");
            }
        }
    }
}
