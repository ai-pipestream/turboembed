//! The SPLADE pooling's acceptance golden (core/src/sparse.rs), against an
//! independent numpy reference (testdata/splade-golden/make_golden.py).
//!
//! `naver/splade-v3` and the other SPLADE checkpoints are licence-gated
//! (HuggingFace answers 401 without an accepted-licence token), so no
//! reference checkpoint could be fetched in this environment. The
//! substitution, as the issue allows: a hand-seeded logits tensor stands
//! in for an MLM head's output, and the numpy script pools it exactly as
//! SPLADE pools (arXiv 2403.06789). This test holds the Rust pooling and
//! the term selection to that output within float tolerance.

use std::path::Path;

use serde_json::Value;
use turbo::sparse::{SparseOptions, pool_logits_maxlog, select_terms};

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/splade-golden/golden.json");
    serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

#[test]
fn rust_pooling_and_selection_match_the_numpy_reference() {
    let g = golden();
    let vocab = g["vocab"].as_u64().unwrap() as usize;
    let flat: Vec<f32> = g["logits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
        .collect();
    assert_eq!(flat.len(), g["tokens"].as_u64().unwrap() as usize * vocab);

    let mut pool = vec![0.0f32; vocab];
    pool_logits_maxlog(&flat, vocab, &mut pool).unwrap();

    for case in g["cases"].as_array().unwrap() {
        let o = &case["options"];
        let opts =
            SparseOptions { top_k: o["top_k"].as_u64().unwrap() as u32, floor: o["floor"].as_f64().unwrap() as f32 };
        let got = select_terms(&pool, &opts).unwrap();
        let want = case["terms"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{opts:?}: {} terms, want {}", got.len(), want.len());
        for (t, w) in got.iter().zip(want) {
            let (term, weight) = (w[0].as_u64().unwrap() as u32, w[1].as_f64().unwrap());
            assert_eq!(t.term, term, "{opts:?}: term id");
            let d = (t.weight as f64 - weight).abs();
            assert!(d <= 1e-6 * weight.max(1.0), "{opts:?}: term {term} weight {} vs {weight}", t.weight);
        }
        // The pairs come out sorted by id.
        assert!(got.windows(2).all(|w| w[0].term < w[1].term), "{opts:?}: sorted by id");
    }
}
