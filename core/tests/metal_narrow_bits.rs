//! The F32 narrow attention's vectors as bits, for comparing two builds
//! byte for byte (#1406): two-layer BERTs whose heads run the narrow kernel
//! at EXACT (heads of 136, 26 and 128) and, past 128, at FASTEST, at 8 x 128
//! and 32 x 256, each run's every vector written as little-endian F32 bits
//! to the file METAL_BITS_OUT names, in order. Build it at two commits and
//! compare the files (docs/receipts/metal-narrow-1406). Ignored: it writes
//! a file and checks nothing itself.
//!
//! Built with the `metal` feature only, on macOS.

#![cfg(feature = "metal")]

mod common;
mod metal_support;

use common::*;
use metal_support::*;
use serde_json::json;
use turbo::*;

#[test]
#[ignore = "writes METAL_BITS_OUT for a comparison between builds"]
fn narrow_attention_vectors_as_bits() {
    let _t = turn();
    let out = std::env::var("METAL_BITS_OUT").expect("METAL_BITS_OUT is not set");
    let Some(_) = metal_device("narrow_attention_vectors_as_bits") else { return };
    let mut bytes = Vec::new();
    for (hidden, heads) in [(272u64, 2u64), (104, 4), (768, 6)] {
        let mut m = model_manifest();
        m["architecture"]["layers"] = json!(2);
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(heads);
        m["architecture"]["intermediate"] = json!(2 * hidden);
        m["embed"]["dim"] = json!(hidden);
        let mut f = Fixture::new(&format!("metal-narrow-bits-{hidden}"), m);
        f.weights("weights/model.safetensors", &bert_weights(2, hidden, 2 * hidden, 1, 0.5));
        f.write();
        let g = on_metal(&f.dir);
        for (batch, seq) in [(8usize, 128usize), (32, 256)] {
            let rows: Vec<Vec<i32>> = (0..batch)
                .map(|r| (0..seq - (r * 7) % 64).map(|p| (1000 + (r * 131 + p * 17) % 29000) as i32).collect())
                .collect();
            let t = Tokens::new(&rows, 0);
            for p in [TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
                let s = Session::create(g.m, Some(&session_desc(batch as u32, seq as u32, p))).unwrap();
                s.write_tokens(&t.batch(), None).unwrap();
                for row in s.run().unwrap().rows() {
                    bytes.extend(row.iter().flat_map(|v| v.to_bits().to_le_bytes()));
                }
            }
        }
    }
    std::fs::write(out, bytes).unwrap();
}
