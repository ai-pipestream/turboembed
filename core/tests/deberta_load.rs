//! A DeBERTa-v2 bundle loaded through the C interface on the CPU: its
//! vectors are turbo::deberta's encoder's (held to transformers in
//! tests/deberta.rs), mean pooled and normalized as the manifest says.

#![cfg(feature = "cpu")]

mod common;
#[path = "common/deberta.rs"]
mod deberta_common;

use common::*;
use deberta_common::{TINY, VOCAB, fixture};
use turbo::deberta::{Config, Encoder};

const H: u64 = TINY.hidden;

/// The encoder's mean of the live tokens' last states, L2 normalized.
fn expected(encoder: &Encoder, ids: &[i32]) -> Vec<f32> {
    let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let states = encoder.encode(&ids, &vec![true; ids.len()]).unwrap();
    let mut v = vec![0f64; H as usize];
    for t in states.data().chunks(H as usize) {
        for (d, &x) in v.iter_mut().zip(t) {
            *d += x as f64 / ids.len() as f64;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| (x / norm) as f32).collect()
}

#[test]
fn a_deberta_bundle_embeds_as_its_encoder_on_the_cpu() {
    let f = fixture("deberta-load", &TINY);
    let loaded = f.load().unwrap();
    assert_eq!(loaded.info().dim, H as u32);
    let config = Config {
        hidden: H as usize,
        intermediate: TINY.intermediate as usize,
        heads: TINY.heads as usize,
        layers: TINY.layers,
        vocabulary: VOCAB as usize,
        max_sequence: 512,
        position_buckets: TINY.buckets as usize,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let bytes = std::fs::read(f.dir.join("weights/model.safetensors")).unwrap();
    let encoder = Encoder::from_safetensors(config, &bytes).unwrap();
    let texts = ["The quick brown fox jumps over the lazy dog.", "a", "Café naïve RÉSUMÉ, twice: Café naïve RÉSUMÉ"];
    let tok = f.open().unwrap();
    let session = Session::create(loaded.m, None).unwrap();
    let got = session.embed(&texts, None).unwrap();
    for (text, v) in texts.iter().zip(&got) {
        let want = expected(&encoder, &tok.row(text, None).unwrap());
        let error = v.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(error < 1e-5, "{text}: largest difference {error}");
    }
}

#[test]
fn a_deberta_bundle_without_its_relative_table_is_refused() {
    let mut f = fixture("deberta-no-relative", &TINY);
    f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap().remove("relative_embeddings");
    let e = f.load().err().unwrap();
    assert!(e.is(turbo::status::BUNDLE_INVALID, "no relative_embeddings, which a DeBERTa-v2 encoder needs"), "{e:?}");
}
