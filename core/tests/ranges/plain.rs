//! Ranges of every kind against the encoder written plainly in f64.

use std::path::Path;

use crate::common::*;
use crate::support::*;
use serde_json::json;
use turbo::*;

/// Ranges of every kind in three rows of `RANGES_TEXTS`, run on the model in `dir`
/// and held to PlainBert; the largest difference seen.
fn check_against_plain(dir: &Path, l: &Loaded, dim: usize, output_dim: u32) -> f64 {
    let plain = PlainBert::new(dir);
    let s = session_of(l, 0);
    let (mut t, lens) = tokens_of(dir, &RANGES_TEXTS);
    let (seq, n) = (t.seq, t.seq as usize);
    // Row 2 has a masked token inside its text, and types of 1 after its
    // first token.
    t.mask[2 * n + 3] = 0;
    let mut types = vec![0; 3 * n];
    for p in 1..lens[2] {
        types[2 * n + p] = 1;
    }
    t.types = Some(types.clone());
    let (l0, l1, l2) = (lens[0] as u32, lens[1] as u32, lens[2] as u32);
    assert!(l0 > l1 && l1 > 4 && l2 > 4, "{lens:?}");
    let ranges = [
        // Row 0 as three sentence-like pieces, and one token alone.
        range(0, 0, l0 / 3),
        range(0, l0 / 3, 2 * l0 / 3),
        range(0, 2 * l0 / 3, l0),
        range(0, 2, 3),
        // Row 1: the last two live tokens and the padding after them, and
        // the whole row.
        range(1, l1 - 2, seq),
        range(1, 0, l1),
        // Row 2: ranges over the masked token, and one that ends in padding.
        range(2, 0, 5),
        range(2, 2, 4),
        range(2, 3, seq),
    ];
    let mut worst = 0f64;
    for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
        let o = opts(|o| {
            o.normalize = normalize;
            o.output_dim = output_dim;
        });
        let got = run_ranges(&s, &t, &ranges, &o);
        assert_eq!(got.len(), ranges.len());
        for (i, (g, rg)) in got.iter().zip(&ranges).enumerate() {
            let (ids, mask, ty) = row_of(&t, rg.row as usize);
            let want = plain.embed_range(
                ids,
                mask,
                &ty,
                rg.start as usize,
                rg.end as usize,
                dim,
                normalize == TURBO_NORMALIZE_L2,
            );
            assert_eq!(g.len(), dim, "range {i}");
            for (a, b) in g.iter().zip(&want) {
                let d = (*a as f64 - b).abs();
                worst = worst.max(d);
                assert!(d < 1e-5 * (1.0 + b.abs()), "normalize {normalize} range {i}: {a} vs {b}");
            }
        }
    }
    worst
}

#[test]
fn ranges_match_the_arithmetic_written_plainly() {
    let l = tiny();
    let mut worst = check_against_plain(&tiny_bundle(), &l, 32, 0);
    worst = worst.max(check_against_plain(&tiny_bundle(), &l, 32, 32));

    // A bundle that lists an output width: the vector is cut, then
    // normalized.
    let mut f = Fixture::new("range-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let cut = f.load().unwrap();
    worst = worst.max(check_against_plain(&f.dir, &cut, 4, 4));
    worst = worst.max(check_against_plain(&f.dir, &cut, 8, 0));
    println!("largest difference from the f64 encoder: {worst:.3e}");
}
