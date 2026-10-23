//! Rows, weights, comparisons and kernel choices the CUDA test files share.

use super::*;
use crate::common::*;
use serde_json::json;
use turbo::*;

/// Rows with padding on either side, a masked token inside a row and token
/// types, as tests/sessions.rs writes them for the CPU.
pub fn awkward_rows(dir: &std::path::Path) -> Tokens {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    let mut t = Tokens::new(&[vec![0; seq], vec![0; seq], vec![0; seq]], 0);
    t.mask.fill(0);
    let mut types = vec![0; 3 * seq];
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            if r == 2 && p > 0 {
                types[r * seq + start + p] = 1;
            }
        }
    }
    t.mask[2 * seq + 1] = 0;
    t.types = Some(types);
    t
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `batch` rows of `seq` columns whose lengths through their last live
/// token differ as much as they can: full rows, rows of the CLS token alone
/// with every other column padding (whose ids are not the pad id, so a
/// kernel that read them would show it), rows of two, and lengths in
/// between; every other row longer than three has a masked token inside
/// it, and the second half of each row has token type 1 in odd rows.
pub fn ragged_rows(dir: &std::path::Path, batch: usize, seq: usize) -> Tokens {
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let (mut ids, mut mask, mut types) = (vec![0; batch * seq], vec![0; batch * seq], vec![0; batch * seq]);
    for r in 0..batch {
        let len = match r % 5 {
            0 => seq,
            1 => 1,
            2 => 2.min(seq),
            3 => 1 + (r * 13) % seq,
            _ => seq / 2 + 1,
        };
        let row = r * seq..(r + 1) * seq;
        for (p, id) in ids[row.clone()].iter_mut().enumerate() {
            *id = (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32;
        }
        ids[r * seq] = 101;
        mask[r * seq..r * seq + len].fill(1);
        if len > 3 && r % 2 == 0 {
            mask[r * seq + len / 2] = 0;
        }
        types[r * seq + len / 2..r * seq + len].fill((r % 2) as i32);
    }
    Tokens { ids, mask, types: Some(types), batch: batch as u32, seq: seq as u32, stride: 0 }
}

/// Row r of `t` alone, at the same width.
pub fn one_row(t: &Tokens, r: usize) -> Tokens {
    let row = r * t.seq as usize..(r + 1) * t.seq as usize;
    Tokens {
        ids: t.ids[row.clone()].to_vec(),
        mask: t.mask[row.clone()].to_vec(),
        types: t.types.as_ref().map(|ty| ty[row].to_vec()),
        batch: 1,
        seq: t.seq,
        stride: 0,
    }
}

/// Each row of `got` is within `tol` of the same row of `want`; returns
/// one minus the lowest cosine and the largest absolute difference.
pub fn within(what: &str, got: &[Vec<f32>], want: &[Vec<f32>], tol: record::Tolerance) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    let (mut lowest, mut worst) = (1f64, 0f64);
    for (r, (a, b)) in got.iter().zip(want).enumerate() {
        let c = cosine(a, b);
        let d = max_abs_diff(a, b);
        assert!(c >= tol.min_cosine, "{what} row {r}: cosine {c} is under {}", tol.min_cosine);
        if let Some(most) = tol.max_abs_diff {
            assert!(d <= most, "{what} row {r}: max abs diff {d:e} is over {most:e}");
        }
        lowest = lowest.min(c);
        worst = worst.max(d);
    }
    (1.0 - lowest, worst)
}

/// Rows of `lens` live tokens each, `seq` wide, the rest padding whose ids
/// are not the pad id; a row with `hole` gets a masked token at its third
/// position when it is longer than three.
pub fn rows_of(dir: &std::path::Path, lens: &[usize], seq: usize, hole: bool) -> Tokens {
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let batch = lens.len();
    let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
    for (r, &len) in lens.iter().enumerate() {
        for p in 0..seq {
            ids[r * seq + p] = (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32;
        }
        ids[r * seq] = 101;
        mask[r * seq..r * seq + len].fill(1);
        if hole && len > 3 {
            mask[r * seq + 2] = 0;
        }
    }
    Tokens { ids, mask, types: None, batch: batch as u32, seq: seq as u32, stride: 0 }
}

/// Every tensor of a BERT `h` wide with an intermediate `i` wide and two
/// layers, under the upstream names, as tiny_weights makes them, with the
/// linear layers' weights scaled to a tenth so the scores stay moderate.
pub fn bert_weights(h: u64, i: u64) -> Vec<Tensor> {
    let (v, p, t) = (30522u64, 512u64, 2u64);
    let mut shapes: Vec<(String, Vec<u64>, f32)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![v, h], 1.0),
        ("embeddings.position_embeddings.weight".into(), vec![p, h], 1.0),
        ("embeddings.token_type_embeddings.weight".into(), vec![t, h], 1.0),
        ("embeddings.LayerNorm.weight".into(), vec![h], 1.0),
        ("embeddings.LayerNorm.bias".into(), vec![h], 1.0),
    ];
    for l in 0..2 {
        let at = |s: &str| format!("encoder.layer.{l}.{s}");
        for (s, shape, scale) in [
            ("attention.self.query.weight", vec![h, h], 0.1),
            ("attention.self.query.bias", vec![h], 1.0),
            ("attention.self.key.weight", vec![h, h], 0.1),
            ("attention.self.key.bias", vec![h], 1.0),
            ("attention.self.value.weight", vec![h, h], 0.1),
            ("attention.self.value.bias", vec![h], 1.0),
            ("attention.output.dense.weight", vec![h, h], 0.1),
            ("attention.output.dense.bias", vec![h], 1.0),
            ("attention.output.LayerNorm.weight", vec![h], 1.0),
            ("attention.output.LayerNorm.bias", vec![h], 1.0),
            ("intermediate.dense.weight", vec![i, h], 0.1),
            ("intermediate.dense.bias", vec![i], 1.0),
            ("output.dense.weight", vec![h, i], 0.1),
            ("output.dense.bias", vec![h], 1.0),
            ("output.LayerNorm.weight", vec![h], 1.0),
            ("output.LayerNorm.bias", vec![h], 1.0),
        ] {
            shapes.push((at(s), shape, scale));
        }
    }
    let mut k = 0u64;
    shapes
        .into_iter()
        .map(|(name, shape, scale)| {
            let n: u64 = shape.iter().product();
            let data = (0..n)
                .flat_map(|_| {
                    k += 1;
                    (scale * (((k * 7919) % 2001) as f32 / 1000.0 - 1.0)).to_le_bytes()
                })
                .collect();
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect()
}

/// FASTEST with each tensor-core attention kernel, 128 queries to a block
/// (the default) and 64 (TURBO_CUDA_ATTENTION=64): the CPU's vectors
/// `want` within FASTEST's bound, the same bits again, and each of the
/// first rows alone within the bound of its vector in the batch.
pub fn attention_matches(g: &Loaded, t: &Tokens, batch: u32, seq: u32, want: &[Vec<f32>], what: &str) {
    for wide in [true, false] {
        attention_kernel_matches(g, t, batch, seq, want, what, wide);
    }
}

pub fn attention_kernel_matches(
    g: &Loaded,
    t: &Tokens,
    batch: u32,
    seq: u32,
    want: &[Vec<f32>],
    what: &str,
    wide: bool,
) {
    turbo::cuda::use_wide_attention(Some(wide));
    let gs = Session::create(g.m, Some(&session_desc(batch, seq, TURBO_PRECISION_FASTEST)));
    turbo::cuda::use_wide_attention(None);
    let gs = gs.unwrap();
    let tol = record::tolerance(gs.info().compute_dtype).unwrap();
    gs.write_tokens(&t.batch(), None).unwrap();
    let got = gs.run().unwrap().rows();
    let what = format!("{what}, attention of {} queries", if wide { 128 } else { 64 });
    let (cos, abs) = within(&what, &got, want, tol);
    println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
    gs.write_tokens(&t.batch(), None).unwrap();
    assert_eq!(gs.run().unwrap().rows(), got, "{what}: the same bits again");
    for r in 0..3.min(t.batch as usize) {
        gs.write_tokens(&one_row(t, r).batch(), None).unwrap();
        let alone = gs.run().unwrap().rows();
        within(&format!("{what}: row {r} alone"), &alone, &got[r..r + 1], tol);
    }
}

/// A session's kernel choices without its forced= item, which says only
/// what fixed them.
pub fn kernels_of(choices: &str) -> String {
    choices.split(';').filter(|i| !i.starts_with("forced=")).collect::<Vec<_>>().join(";")
}

/// f with TURBO_CUDA_CHOICES in its place for the sessions f makes.
pub fn forcing<T>(choices: &str, f: impl FnOnce() -> T) -> T {
    turbo::cuda::use_choices(Some(choices));
    let r = f();
    turbo::cuda::use_choices(None);
    r
}

/// The small model of the tile tests, heads of 32, whose sessions of 40
/// rows of 160 tokens have the bins up to 256, 1024, 4096 and 16384.
pub fn small_model(name: &str) -> (Fixture, Loaded) {
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(160);
    m["embed"]["max_batch"] = json!(40);
    let mut f = Fixture::new(name, m);
    f.weights("weights/model.safetensors", &bert_weights(64, 256));
    let g = f.load_on(cuda).unwrap();
    (f, g)
}

/// TURBO_TEST_BUNDLE, a relative path read from the workspace root, as
/// tests/conformance.rs reads it.
pub fn named_bundle() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE")?);
    Some(if p.is_absolute() { p } else { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(p) })
}
