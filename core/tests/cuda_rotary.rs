//! ModernBERT and nomic-bert on CUDA (#1461) against the CPU backend on
//! the same bundle, as tests/cuda_deberta.rs holds DeBERTa-v2: the tiny
//! sealed bundles (testdata/tiny-modernbert-bundle, window 8 and a global
//! layer every 3, and testdata/tiny-nomic-bundle) every value within 1e-4
//! of the CPU's at MODEL on each GEMM kernel and attention kernel, and
//! within the cosine floor of the sealed vectors at FASTEST; and a
//! generated ModernBERT of head width 64 at a window of 128, whose rows of
//! up to 256 tokens take the tensor cores' attention at FASTEST.
//!
//! Built with the `cuda` feature only. TURBO_TEST_DEVICE=cuda runs the
//! tests on the first cuda device; without it they say they were skipped,
//! as the other device tests do. With TURBO_TEST_REQUIRE_CUDA=1 a test
//! that finds no cuda device, or is not told to use one, fails instead.

#![cfg(feature = "cuda")]

mod common;
#[path = "common/rotary.rs"]
mod rotary;

use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::*;
use serde_json::json;
use turbo::cuda::Tile;
use turbo::*;

/// The largest difference allowed from the CPU at MODEL: the F32 device
/// bound.
const BOUND: f32 = 1e-4;
/// The cosine floor at FASTEST.
const FASTEST_COSINE: f64 = 0.999;

/// The tests take turns on the one device.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Whether the test runs: TURBO_TEST_DEVICE=cuda and a cuda device listed;
/// else it says it is skipped, or fails under TURBO_TEST_REQUIRE_CUDA.
fn device(test: &str) -> bool {
    match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Ok("cuda") => {}
        Ok(d) => panic!("{test}: TURBO_TEST_DEVICE={d}: this test runs on cuda alone"),
        Err(_) => {
            assert!(!required(), "{test}: TURBO_TEST_REQUIRE_CUDA=1 and TURBO_TEST_DEVICE is not cuda");
            println!("{test}: skipped: TURBO_TEST_DEVICE is not set");
            return false;
        }
    }
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, "cuda");
    unsafe { turbo_runtime_release(rt) };
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_CUDA=1 and the cuda backend lists no device");
        println!("{test}: skipped: the cuda backend lists no device");
    }
    d.is_some()
}

fn cuda(rt: *mut turbo_runtime) -> u32 {
    first_of(rt, "cuda").expect("a cuda device")
}

/// The bundle at `dir` on the cuda device, checked to be one: no session
/// of it runs anywhere else.
fn on_gpu(dir: &Path) -> Loaded {
    let l = Loaded::load_on(dir, cuda).unwrap_or_else(|e| panic!("{e:?}"));
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(l.rt, cuda(l.rt), &mut info, ptr::null_mut()) }, 0);
    assert_eq!(field(&info.backend), "cuda", "the model is on the cuda backend");
    l
}

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata").join(name)
}

/// Rows of the given lengths, right padded to the longest; `left` rows
/// are padded on the left instead, `holes` are (row, column) masked
/// inside a row, and with `types` every other row's tokens are type 1.
fn rows(lens: &[usize], left: &[usize], holes: &[(usize, usize)], types: bool) -> Tokens {
    let seq = *lens.iter().max().unwrap();
    let mut t = Tokens::new(&vec![vec![0; seq]; lens.len()], 0);
    t.mask.fill(0);
    let mut ty = vec![0; lens.len() * seq];
    for (r, &len) in lens.iter().enumerate() {
        let start = if left.contains(&r) { seq - len } else { 0 };
        for p in 0..len {
            t.ids[r * seq + start + p] = 10 + ((r * 7919 + p * 104_729) % 1100) as i32;
            t.mask[r * seq + start + p] = 1;
            ty[r * seq + start + p] = (r % 2) as i32;
        }
    }
    for &(r, c) in holes {
        t.mask[r * seq + c] = 0;
    }
    if types {
        t.types = Some(ty);
    }
    t
}

/// The largest difference between two runs' vectors; infinite where either
/// is NaN.
fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            x.iter().zip(y).map(|(p, q)| if p.is_nan() || q.is_nan() { f32::INFINITY } else { (p - q).abs() })
        })
        .fold(0f32, f32::max)
}

/// The least cosine between two runs' vectors, row by row.
fn least_cosine(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
            let n = |v: &[f32]| v.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(x) * n(y))
        })
        .fold(1.0, f64::min)
}

/// The cases a tiny bundle runs, rows of up to `seq` tokens: lengths about
/// the window, left padding (whose first pads see no live key in a local
/// layer's band), holes, one token, a whole row.
fn cases(seq: usize, types: bool) -> Vec<(&'static str, Tokens)> {
    vec![
        ("mixed lengths", rows(&[17, 5, seq, 9, 1, 33, 48, 2], &[], &[], types)),
        ("left padding and holes", rows(&[17, 5, 40, 9, 1], &[3], &[(2, 11), (0, 3)], types)),
        ("one token", rows(&[1], &[], &[], types)),
        ("a row of max_seq tokens", rows(&[seq, 3], &[], &[], types)),
    ]
}

/// Every case of `cases` and token ranges on the device and the CPU, the
/// largest difference printed and held to BOUND (MODEL) or the cosine to
/// FASTEST_COSINE; every value finite.
fn holds(test: &str, what: &str, g: &Session, c: &Session, seq: usize, types: bool, fastest: bool) {
    let check = |name: &str, got: &[Vec<f32>], want: &[Vec<f32>]| {
        assert!(got.iter().flatten().all(|x| x.is_finite()), "{what}: {name}: a value is not finite");
        let (d, cos) = (worst(got, want), least_cosine(got, want));
        println!("{test}: {what}: {name}: largest difference {d:e}, 1 - least cosine {:e}", 1.0 - cos);
        if fastest {
            assert!(cos >= FASTEST_COSINE, "{what}: {name}: cosine {cos}");
        } else {
            assert!(d < BOUND, "{what}: {name}: {d:e}");
        }
    };
    for (name, t) in cases(seq, types) {
        g.write_tokens(&t.batch(), None).unwrap();
        let got = g.run().unwrap().rows();
        c.write_tokens(&t.batch(), None).unwrap();
        check(name, &got, &c.run().unwrap().rows());
    }
    let t = rows(&[40, 12, seq], &[], &[(0, 20)], types);
    let ranges = [range(0, 0, 40), range(0, 1, 21), range(1, 3, 9), range(2, 10, seq as u32)];
    g.write_ranges(&t.batch(), &ranges, None).unwrap();
    let got = g.run().unwrap().rows();
    c.write_ranges(&t.batch(), &ranges, None).unwrap();
    check("token ranges", &got, &c.run().unwrap().rows());
}

/// A kernel choice by name, set (true) or undone (false) around a
/// session's making.
type Knob = (&'static str, Box<dyn Fn(bool)>);

/// A device session at `precision` with `set` applied while it is made,
/// then undone: the knobs act on the sessions made while they are set.
fn session_with(m: *mut turbo_model, batch: u32, seq: u32, precision: u32, set: impl Fn(bool)) -> Session {
    set(true);
    let s = Session::create(m, Some(&session_desc(batch, seq, precision)));
    set(false);
    s.unwrap_or_else(|e| panic!("{e:?}"))
}

/// The knobs a MODEL session is held to the CPU at: each GEMM kernel
/// family the rotary epilogues have (the transposed FMA kernel, the FMA
/// kernel's tiles, TF32 on the tensor cores) and the FMA attentions.
fn model_kernels() -> Vec<Knob> {
    vec![
        ("default", Box::new(|_| {})),
        ("tile 64x64", Box::new(|on| turbo::cuda::use_tile(on.then_some(Tile::T64x64)))),
        ("tile 128x128 16x8", Box::new(|on| turbo::cuda::use_tile(on.then_some(Tile::T128x128Thread16x8)))),
        ("CUTLASS SIMT tile", Box::new(|on| turbo::cuda::use_tile(on.then_some(Tile::Cs)))),
        ("TF32", Box::new(|on| turbo::cuda::use_tf32(on.then_some(true)))),
        ("split attention", Box::new(|on| turbo::cuda::use_split_attention(on.then_some(true)))),
        ("attention of 32 queries", Box::new(|on| turbo::cuda::use_narrow_tiles(on.then_some(true)))),
    ]
}

/// The knobs a FASTEST session is held at: the eight-warp tiles, 64 x 64,
/// F16 accumulators, a swizzled tile (which the rotary epilogues run on
/// the eight-warp kernel), and the tensor cores' attentions.
fn fastest_kernels() -> Vec<Knob> {
    vec![
        ("default", Box::new(|_| {})),
        ("tile 64x64", Box::new(|on| turbo::cuda::use_tile(on.then_some(Tile::T64x64)))),
        ("F16 accumulators", Box::new(|on| turbo::cuda::use_tile(on.then_some(Tile::EightWarpsF16Accumulate)))),
        ("swizzled tile", Box::new(|on| turbo::cuda::use_tile(on.then_some(Tile::Swizzled)))),
        ("attention of 64 queries", Box::new(|on| turbo::cuda::use_wide_attention(on.then_some(false)))),
        ("exact attention", Box::new(|on| turbo::cuda::use_exact_attention(on.then_some(true)))),
        ("acc32 attention", Box::new(|on| turbo::cuda::use_acc32_attention(on.then_some(true)))),
    ]
}

/// The bundle at `dir` at MODEL on every kernel of model_kernels, held to
/// the CPU.
fn matches_the_cpu(test: &str, dir: &Path, seq: usize, types: bool) {
    let _t = turn();
    if !device(test) {
        return;
    }
    let (g, c) = (on_gpu(dir), Loaded::load(dir).unwrap());
    let cs = Session::create(c.m, Some(&session_desc(8, seq as u32, TURBO_PRECISION_MODEL))).unwrap();
    for (what, set) in model_kernels() {
        let gs = session_with(g.m, 8, seq as u32, TURBO_PRECISION_MODEL, set);
        assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F32);
        // TF32 rounds the operands: held to the cosine floor, as FASTEST.
        holds(test, what, &gs, &cs, seq, types, what == "TF32");
    }
}

#[test]
fn the_tiny_modernbert_matches_the_cpu() {
    matches_the_cpu("the_tiny_modernbert_matches_the_cpu", &testdata("tiny-modernbert-bundle"), 64, false);
}

#[test]
fn the_tiny_nomic_matches_the_cpu() {
    matches_the_cpu("the_tiny_nomic_matches_the_cpu", &testdata("tiny-nomic-bundle"), 64, true);
}

/// Each tiny bundle's sealed cases at FASTEST, within the cosine floor of
/// the upstream vectors (common/conformance.rs), on the cuda device.
#[test]
fn the_tiny_bundles_match_their_reference_at_fastest() {
    let _t = turn();
    if !device("the_tiny_bundles_match_their_reference_at_fastest") {
        return;
    }
    for name in ["tiny-modernbert-bundle", "tiny-nomic-bundle"] {
        on_gpu(&testdata(name));
        conformance::check(&testdata(name), cuda, TURBO_PRECISION_FASTEST);
        conformance::check(&testdata(name), cuda, TURBO_PRECISION_MODEL);
    }
}

/// The batch of tests/rotary_load.rs's left-padded row on the tiny
/// ModernBERT: rows of 17, 5, 40, 9 and 1 tokens padded to 40, row 3 at the
/// right end of its columns, so its first pads see no live key in a local
/// layer's band. Every value is finite and within BOUND of the CPU, whose
/// such queries read nothing.
#[test]
fn a_left_padded_modernbert_row_stays_finite() {
    let _t = turn();
    if !device("a_left_padded_modernbert_row_stays_finite") {
        return;
    }
    let dir = testdata("tiny-modernbert-bundle");
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let desc = session_desc(8, 64, TURBO_PRECISION_MODEL);
    let (gs, cs) = (Session::create(g.m, Some(&desc)).unwrap(), Session::create(c.m, Some(&desc)).unwrap());
    let t = rows(&[17, 5, 40, 9, 1], &[3], &[], false);
    gs.write_tokens(&t.batch(), None).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    for (r, v) in got.iter().enumerate() {
        assert!(v.iter().all(|x| x.is_finite()), "row {r}: {v:?}");
    }
    let d = worst(&got, &want);
    println!("a_left_padded_modernbert_row_stays_finite: largest difference {d:e}");
    assert!(d < BOUND, "{d:e}");
}

/// A ModernBERT of hidden 128 over heads of 64, intermediate 64, three
/// layers (one global, two local), window 128 and 512 positions, its
/// weights from each element's place, scaled small enough for F16. With
/// `long`, that many positions and a max_seq of as many, one row a run.
fn wide_modernbert(name: &str, long: Option<u64>) -> Fixture {
    let mut m = rotary::modernbert(&format!("{name}-base")).manifest.clone();
    let (h, i, layers) = (128u64, 64u64, 3usize);
    let a = &mut m["architecture"];
    a["hidden"] = h.into();
    a["heads"] = 2.into();
    a["intermediate"] = i.into();
    a["layers"] = layers.into();
    a["max_positions"] = long.unwrap_or(512).into();
    a["attention"] = json!({ "window": 128, "global_every": 3 });
    m["embed"]["dim"] = h.into();
    if let Some(n) = long {
        m["embed"]["max_seq"] = n.into();
        m["embed"]["max_batch"] = 1.into();
        // A case longer than max_seq, which the loader asks for.
        m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    }
    m["files"] = json!([]);
    let mut f = Fixture::new(name, m);
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("model.embeddings.tok_embeddings.weight".into(), vec![rotary::VOCAB, h]),
        ("model.embeddings.norm.weight".into(), vec![h]),
        ("model.final_norm.weight".into(), vec![h]),
    ];
    for l in 0..layers {
        if l > 0 {
            shapes.push((format!("model.layers.{l}.attn_norm.weight"), vec![h]));
        }
        for (stem, shape) in [
            ("attn.Wqkv.weight", vec![3 * h, h]),
            ("attn.Wo.weight", vec![h, h]),
            ("mlp_norm.weight", vec![h]),
            ("mlp.Wi.weight", vec![2 * i, h]),
            ("mlp.Wo.weight", vec![h, i]),
        ] {
            shapes.push((format!("model.layers.{l}.{stem}"), shape));
        }
    }
    let mut k = 0u64;
    let tensors: Vec<Tensor> = shapes
        .into_iter()
        .map(|(name, shape)| {
            // Norm weights about 1, matrices about 0.
            let norm = shape.len() == 1;
            let n: u64 = shape.iter().product();
            let mut data = Vec::with_capacity(n as usize * 4);
            for _ in 0..n {
                k += 1;
                let u = ((k * 7919) % 2001) as f32 / 1000.0 - 1.0;
                data.extend((if norm { 1.0 + 0.25 * u } else { 0.15 * u }).to_le_bytes());
            }
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect();
    f.weights("weights/model.safetensors", &tensors);
    f
}

/// The generated ModernBERT at head width 64 against the CPU: at MODEL
/// within BOUND, and at FASTEST, on each tensor-core attention, at the
/// cosine floor; rows of up to 256 tokens, past several windows.
#[test]
fn a_modernbert_of_head_width_64_matches_the_cpu() {
    let test = "a_modernbert_of_head_width_64_matches_the_cpu";
    let _t = turn();
    if !device(test) {
        return;
    }
    let f = wide_modernbert("cu-rotary-wide", None);
    f.write();
    let (g, c) = (on_gpu(&f.dir), Loaded::load(&f.dir).unwrap());
    let cs = Session::create(c.m, Some(&session_desc(8, 256, TURBO_PRECISION_MODEL))).unwrap();
    for (what, set) in model_kernels() {
        let tf32 = what == "TF32";
        let gs = session_with(g.m, 8, 256, TURBO_PRECISION_MODEL, set);
        holds(test, &format!("MODEL {what}"), &gs, &cs, 256, false, tf32);
    }
    for (what, set) in fastest_kernels() {
        let gs = session_with(g.m, 8, 256, TURBO_PRECISION_FASTEST, set);
        assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F16);
        holds(test, &format!("FASTEST {what}"), &gs, &cs, 256, false, true);
    }
}

/// Attention past 4096 keys: the generated ModernBERT at 4097 positions
/// takes the tensor cores' attention with F32 cross-chunk sums
/// (mma128-acc32) in a FASTEST session of max_seq 4097, tuned or not, and
/// the F16 sums (mma128) at 4096; and on a row of 4097 tokens with a
/// masked key and a second row past the window, MODEL (fma-tiled, 64
/// queries to a block) holds to the CPU within BOUND and FASTEST
/// (mma128-acc32) at the cosine floor.
#[test]
fn fastest_takes_f32_attention_sums_past_4096_keys() {
    let test = "fastest_takes_f32_attention_sums_past_4096_keys";
    let _t = turn();
    if !device(test) {
        return;
    }
    let f = wide_modernbert("cu-rotary-long", Some(4097));
    f.write();
    let (g, c) = (on_gpu(&f.dir), Loaded::load(&f.dir).unwrap());
    turbo::tuning::use_cache_dir(Some(None));
    for (seq, tuning, attn) in [
        (4096, TURBO_AUTOTUNE_OFF, "attn=mma128,"),
        (4097, TURBO_AUTOTUNE_OFF, "attn=mma128-acc32,"),
        (4097, TURBO_AUTOTUNE_ON, "attn=mma128-acc32,"),
    ] {
        let mut d = session_desc(1, seq, TURBO_PRECISION_FASTEST);
        d.tuning = tuning;
        let s = Session::create(g.m, Some(&d)).unwrap_or_else(|e| panic!("{e:?}"));
        let choices = field(&s.info().choices);
        println!("{test}: max_seq {seq}, tuning {tuning}: {choices}");
        assert!(choices.contains(attn), "max_seq {seq}, tuning {tuning}: {choices}");
        let other = if attn == "attn=mma128," { "attn=mma128-acc32," } else { "attn=mma128," };
        assert!(!choices.contains(other), "max_seq {seq}, tuning {tuning}: {choices}");
    }
    turbo::tuning::use_cache_dir(None);
    let t = rows(&[4097], &[], &[(0, 2000)], false);
    let cs = Session::create(c.m, Some(&session_desc(1, 4097, TURBO_PRECISION_MODEL))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    for (prec, attn) in [(TURBO_PRECISION_MODEL, "attn=fma-tiled,"), (TURBO_PRECISION_FASTEST, "attn=mma128-acc32,")] {
        let gs = Session::create(g.m, Some(&session_desc(1, 4097, prec))).unwrap();
        assert!(field(&gs.info().choices).contains(attn));
        gs.write_tokens(&t.batch(), None).unwrap();
        let got = gs.run().unwrap().rows();
        let (d, cos) = (worst(&got, &want), least_cosine(&got, &want));
        println!("{test}: precision {prec} ({attn}): largest difference {d:e}, 1 - least cosine {:e}", 1.0 - cos);
        if prec == TURBO_PRECISION_FASTEST {
            assert!(cos >= FASTEST_COSINE, "cosine {cos}");
        } else {
            assert!(d < BOUND, "diff {d:e}");
        }
    }
}

/// FASTEST on the tiny bundles held to the CPU at the cosine floor on each
/// kernel of fastest_kernels (heads of 16 take the FMA attention in F16).
#[test]
fn the_tiny_bundles_at_fastest_hold_on_every_kernel() {
    let test = "the_tiny_bundles_at_fastest_hold_on_every_kernel";
    let _t = turn();
    if !device(test) {
        return;
    }
    for (name, types) in [("tiny-modernbert-bundle", false), ("tiny-nomic-bundle", true)] {
        let dir = testdata(name);
        let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
        let cs = Session::create(c.m, Some(&session_desc(8, 64, TURBO_PRECISION_MODEL))).unwrap();
        for (what, set) in fastest_kernels() {
            let gs = session_with(g.m, 8, 64, TURBO_PRECISION_FASTEST, set);
            holds(test, &format!("{name}: {what}"), &gs, &cs, 64, types, true);
        }
    }
}

/// #1617's overflow fixtures, whose layer 0 gated MLP middle product
/// passes F16's 65504 on every token, at FASTEST on each kernel of
/// fastest_kernels and over rows short, full and ragged: the F16 middle
/// product saturates, so every output is finite (#1632). Without the
/// clamp every value was infinite or NaN.
#[test]
fn a_gated_middle_past_f16_s_range_stays_finite_at_fastest() {
    let test = "a_gated_middle_past_f16_s_range_stays_finite_at_fastest";
    let _t = turn();
    if !device(test) {
        return;
    }
    for (name, f, types) in [
        ("modernbert", rotary::overflow_modernbert("cu-rotary-overflow-mb"), false),
        ("nomic", rotary::overflow_nomic("cu-rotary-overflow-nb"), true),
    ] {
        f.write();
        let g = on_gpu(&f.dir);
        for lens in [&[9usize, 1, 32][..], &[32usize, 32][..], &[64usize, 33, 17][..]] {
            let t = rows(lens, &[], &[], types);
            for (what, set) in fastest_kernels() {
                let gs = session_with(g.m, 4, 64, TURBO_PRECISION_FASTEST, set);
                assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F16, "{name}: {what}");
                gs.write_tokens(&t.batch(), None).unwrap();
                let got = gs.run().unwrap().rows();
                let bad = got.iter().flatten().filter(|v| !v.is_finite()).count();
                assert_eq!(bad, 0, "{name}: {what}: rows {lens:?}: {bad} values not finite");
            }
        }
        println!("{test}: {name}: finite on every kernel and row shape");
    }
}
