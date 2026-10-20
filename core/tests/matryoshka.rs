//! Declared Matryoshka widths on the dense write path: a session cut to
//! each width the bundle declares gives the f64 encoder's truncation and
//! renormalization of the full vector, the narrowed result moves no more
//! bytes than the width needs, a truncated zero vector renormalizes to
//! zero and never NaN, and an undeclared width is refused by name.
//! tests/sessions.rs holds the single-width case and the option checks;
//! tests/conformance.rs holds the vectors to the upstream reference.

mod common;

use std::time::Instant;

use common::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];

fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

/// The small BERT (dim 8) declaring `widths`.
fn declaring(name: &str, widths: &[u32]) -> (Fixture, Loaded) {
    let mut f = Fixture::model(name);
    f.manifest["embed"]["output_dims"] = json!(widths);
    let l = f.load().unwrap_or_else(|e| panic!("{e:?}"));
    (f, l)
}

#[test]
fn each_declared_width_matches_the_truncation_and_renormalization() {
    let (f, l) = declaring("widths-each", &[1, 2, 4]);
    let s = Session::create(l.m, None).unwrap();
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for w in [1u32, 2, 4, 8] {
        let o = opts(|o| o.output_dim = w);
        s.write_text(&TEXTS, Some(&o)).unwrap();
        let r = s.run().unwrap();
        // The result is the batch at the asked width: fewer bytes than the
        // full-width write, never more, so no full-width vector is copied
        // before the cut.
        let info = r.info();
        assert_eq!((info.dim, info.bytes), (w, 3 * w as u64 * 4), "width {w}");
        let got = r.rows();
        for (t, g) in TEXTS.iter().zip(&got) {
            let ids = tok.row(t, None).unwrap();
            let want =
                plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, w as usize, true);
            for (a, b) in g.iter().zip(&want) {
                assert!((*a as f64 - b).abs() < 1e-5, "width {w}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn undeclared_widths_are_refused_by_name() {
    let (_, l) = declaring("widths-refused", &[4]);
    let s = Session::create(l.m, None).unwrap();
    let e = s.write_text(&TEXTS, Some(&opts(|o| o.output_dim = 3))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    assert!(e.message.contains("output_dim 3: the model was trained to be cut to [4]"), "{e:?}");
    let e = s.write_text(&TEXTS, Some(&opts(|o| o.output_dim = 9))).err().unwrap();
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 6), "{e:?}");
    assert!(e.message.contains("output_dim 9 is over the model's dim 8"), "{e:?}");
    // 0 is the bundle's own width; the model's dim is always served.
    assert_eq!(s.embed(&TEXTS, Some(&opts(|o| o.output_dim = 0))).unwrap()[0].len(), 8);
    assert_eq!(s.embed(&TEXTS, Some(&opts(|o| o.output_dim = 8))).unwrap()[0].len(), 8);
}

/// The small BERT with every LayerNorm's affine zeroed on its first two
/// coordinates: every hidden state is zero there, so a vector cut to
/// width 2 is the zero vector, and NORMALIZE_L2 divides by the 1e-12
/// floor: the result is zero, never NaN.
#[test]
fn a_truncated_zero_vector_renormalizes_without_nan() {
    let zeroed: Vec<Tensor> = tiny_weights(0)
        .into_iter()
        .map(|mut t| {
            if t.name.ends_with("LayerNorm.weight") || t.name.ends_with("LayerNorm.bias") {
                for b in t.data[0..8].iter_mut() {
                    *b = 0;
                }
            }
            t
        })
        .collect();
    let mut f = Fixture::new("widths-zero", model_manifest());
    f.manifest["embed"]["output_dims"] = json!([2]);
    f.weights("weights/model.safetensors", &zeroed);
    let l = f.load().unwrap();
    let s = Session::create(l.m, None).unwrap();
    let got = s.embed(&TEXTS, Some(&opts(|o| o.output_dim = 2))).unwrap();
    for v in &got {
        assert_eq!(v, &vec![0.0; 2], "the cut is zero, and the floor keeps it zero");
        assert!(v.iter().all(|x| x.is_finite()));
    }
    // The f64 encoder, written plainly, agrees to the bit.
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for (t, g) in TEXTS.iter().zip(&got) {
        let ids = tok.row(t, None).unwrap();
        let want = plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, 2, true);
        assert_eq!(g, &want.iter().map(|&x| x as f32).collect::<Vec<_>>());
    }
}

/// A MiniLM-shaped model at widths the bundle declares, timed the way
/// docs/benchmarks.md times a batch: the narrower write moves fewer bytes
/// and must not be slower. Ignored, since it measures rather than checks;
/// run it with
///
/// ```
/// cargo test --release -p turbo --test matryoshka -- --ignored --nocapture
/// ```
///
/// TURBO_SPEED_RUNS sets the timed runs (default 20).
#[test]
#[ignore]
fn narrowed_writes_move_fewer_bytes_and_take_no_longer() {
    let (batch, seq) = (32u32, 256u32);
    let mut m = model_manifest();
    m["architecture"] = json!({
        "family": "FAMILY_BERT",
        "layers": 6,
        "hidden": 384,
        "heads": 12,
        "intermediate": 1536,
        "activation": "ACTIVATION_GELU_ERF",
        "layer_norm_eps": 1e-12,
        "position_embedding": "POSITION_ABSOLUTE",
        "max_positions": 512,
        "token_types": 2,
        "vocab_size": 30522
    });
    m["embed"]["dim"] = json!(384);
    m["embed"]["output_dims"] = json!([192, 96, 48]);
    let mut f = Fixture::new("widths-speed", m);
    f.weights("weights/model.safetensors", &bert_weights(6, 384, 1536, 0, 1.0));
    let l = f.load().unwrap();
    let s = Session::create(l.m, Some(&session_desc(batch, seq, TURBO_PRECISION_MODEL))).unwrap();

    // One batch, tokenized by construction: every run writes the same
    // rows again.
    let lens: Vec<usize> = (0..batch as usize).map(|r| if r == 0 { seq as usize } else { 6 + (r * 29) % 53 }).collect();
    let rows: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(r, &n)| {
            (0..n)
                .map(|p| match p {
                    0 => 101,
                    p if p == n - 1 => 102,
                    p => 1000 + ((r * 7919 + p * 104729) % 28000) as i32,
                })
                .collect()
        })
        .collect();
    let t = Tokens::new(&rows, 0);

    let runs: usize = std::env::var("TURBO_SPEED_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    for w in [384u32, 192, 96, 48] {
        let o = opts(|o| o.output_dim = w);
        let mut best = f64::INFINITY;
        for _ in 0..runs {
            let start = Instant::now();
            s.write_tokens(&t.batch(), Some(&o)).unwrap();
            let r = s.run().unwrap();
            let mut flat = Vec::new();
            r.read_into(&mut flat);
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
            assert_eq!(r.info().bytes, batch as u64 * w as u64 * 4);
        }
        println!("width {w:3}: {best:8.3} ms a run, {} bytes a vector", w * 4);
    }
}
