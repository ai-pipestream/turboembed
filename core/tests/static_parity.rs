//! A static bundle made from a Model2Vec model against that model's own
//! golden ids and vectors (bundle/reference/static_golden.py), through
//! the C interface on the CPU backend: every text's ids exactly, and
//! every vector to the bit, at the model's max_length and at none.
//!
//! The bundle and the goldens are not in the repository: the test runs
//! when TURBO_PARITY_BUNDLE names the bundle and TURBO_PARITY_GOLDEN the
//! directory static_golden.py wrote, and passes with a note otherwise.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::Value;
use turbo::safetensors::{Dtype, File};
use turbo::*;

/// Texts embedded per call: the session's batch.
const BATCH: u32 = 256;

#[test]
fn every_golden_text_gives_model2vecs_ids_and_vector_to_the_bit() {
    let (Some(bundle), Some(golden)) =
        (std::env::var_os("TURBO_PARITY_BUNDLE"), std::env::var_os("TURBO_PARITY_GOLDEN"))
    else {
        println!("TURBO_PARITY_BUNDLE and TURBO_PARITY_GOLDEN are not both set: nothing to compare");
        return;
    };
    let (bundle, golden) = (PathBuf::from(bundle), PathBuf::from(golden));
    let texts: Vec<String> = serde_json::from_slice(&fs::read(golden.join("texts.json")).unwrap()).unwrap();
    let versions: Value = serde_json::from_slice(&fs::read(golden.join("versions.json")).unwrap()).unwrap();
    println!("{}: goldens from {}", bundle.display(), versions["versions"]);

    let l = Loaded::load(&bundle).unwrap_or_else(|e| panic!("{e:?}"));
    let mi = l.info();
    let max_seq = mi.max_seq;
    let s = Session::create(l.m, Some(&session_desc(BATCH, max_seq, TURBO_PRECISION_MODEL)))
        .unwrap_or_else(|e| panic!("{e:?}"));
    let tok = Tok::create(&bundle).unwrap();
    let dim = mi.dim as usize;

    let mut failures = Vec::new();
    for (name, max_length) in versions["files"].as_object().unwrap() {
        let bytes = fs::read(golden.join(name)).unwrap();
        let f = File::parse(name, &bytes).unwrap();
        let ids = f.get("ids", Dtype::I32, 2).unwrap();
        let width = ids.shape[1] as usize;
        let ids = ids.i32s();
        let lengths = f.get("lengths", Dtype::I32, 1).unwrap().i32s();
        let want = f.get("embeddings", Dtype::F32, 2).unwrap();
        assert_eq!(want.shape, [texts.len() as u64, dim as u64], "{name}");
        let want = want.f32s();

        // The model's max_length is the bundle's: TURBO_TRUNCATE_MODEL. No
        // max_length is TURBO_TRUNCATE_NONE, up to the session's max_seq.
        let (truncate, max_tokens) = match max_length.as_u64() {
            Some(n) => {
                assert_eq!(n as u32, tok_max_length(&bundle), "{name}: the bundle's max_length");
                (TURBO_TRUNCATE_MODEL, 0)
            }
            None => (TURBO_TRUNCATE_NONE, max_seq),
        };

        let mut id_mismatch = 0;
        for (i, t) in texts.iter().enumerate() {
            let want = &ids[i * width..i * width + lengths[i] as usize];
            // The longest a text can give: a byte a token at least.
            let o = options(0, truncate, if max_tokens == 0 { 0 } else { (t.len() + 16) as u32 }, 0);
            let got = tok.row(t, Some(&o)).unwrap_or_else(|e| panic!("{name} text {i}: {e:?}"));
            if got != want {
                id_mismatch += 1;
                if id_mismatch <= 5 {
                    let p = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                    failures.push(format!(
                        "{name} text {i} ({:?}): ids differ at {p}: library {:?}, model2vec {:?}",
                        preview(t),
                        &got[p..got.len().min(p + 6)],
                        &want[p..want.len().min(p + 6)]
                    ));
                }
            }
        }

        // Vectors, in batches, for every text that fits the session.
        let fits: Vec<usize> = (0..texts.len()).filter(|&i| lengths[i] as u32 <= max_seq).collect();
        let mut o = embed_options();
        o.truncate = truncate;
        o.max_tokens = max_tokens;
        let (mut bits, mut worst_ulps) = (0, 0u32);
        for chunk in fits.chunks(BATCH as usize) {
            let batch: Vec<&str> = chunk.iter().map(|&i| texts[i].as_str()).collect();
            let got = s.embed(&batch, Some(&o)).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (&i, g) in chunk.iter().zip(&got) {
                let w = &want[i * dim..(i + 1) * dim];
                let ulps = g.iter().zip(w).map(|(a, b)| ulps(*a, *b)).max().unwrap_or(0);
                if ulps != 0 {
                    bits += 1;
                    worst_ulps = worst_ulps.max(ulps);
                    if bits <= 5 {
                        failures.push(format!("{name} text {i} ({:?}): vector off by {ulps} ulps", preview(&texts[i])));
                    }
                }
            }
        }
        println!(
            "  {name}: {} texts, {id_mismatch} with other ids; {} vectors compared, {bits} not to the bit \
             (worst {worst_ulps} ulps); {} longer than max_seq {max_seq}",
            texts.len(),
            fits.len(),
            texts.len() - fits.len()
        );
        if id_mismatch > 5 || bits > 5 {
            failures.push(format!("{name}: {id_mismatch} texts with other ids, {bits} vectors not to the bit"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The bundle's static_embedding.max_length.
fn tok_max_length(bundle: &std::path::Path) -> u32 {
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    m["static_embedding"]["max_length"].as_u64().unwrap() as u32
}

/// How many representable F32 values lie between a and b.
fn ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let b = v.to_bits() as i32;
        if b < 0 { i32::MIN - b } else { b }
    };
    key(a).abs_diff(key(b))
}

fn preview(t: &str) -> String {
    let p: String = t.chars().take(40).collect();
    if p.len() < t.len() { format!("{p}...") } else { p }
}
