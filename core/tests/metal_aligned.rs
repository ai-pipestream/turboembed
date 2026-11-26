//! The 32 x 32 and 64 x 32 F16 tiles' aligned forms, which the backend
//! runs when m is a whole multiple of the tile's rows and n and k of 32,
//! against the checked forms: the same sums in the same order, so the
//! same bits.
//!
//! Built with the `metal` feature only, on macOS. A test that needs a
//! device that runs the kernels fails when the backend lists none.
//! TURBO_TEST_REQUIRE_METAL=0 lets it say it was skipped, and pass.

#![cfg(feature = "metal")]

mod common;
mod metal_support;

use common::*;
use metal_support::*;
use serde_json::json;
use turbo::*;

/// Rows of the given lengths, with ids spread over the vocabulary.
fn rows_of(lens: &[usize]) -> Tokens {
    let rows: Vec<Vec<i32>> =
        lens.iter().enumerate().map(|(r, &len)| (0..len).map(|p| (1000 + r * 131 + p * 17) as i32).collect()).collect();
    Tokens::new(&rows, 0)
}

/// A model of width 64 (feed-forward 128, so every n and k a multiple of
/// 32, the packed tokens rounded up to 32): its rows with the 32 x 32 or
/// the 64 x 32 tile forced against the 32 x 64 tile's, which checks every
/// range and sums in the same order, are the same bit for bit, and within
/// FASTEST's floor of the CPU. At 160 packed tokens the 32 x 32 runs
/// aligned and the 64 x 32 checked; at 128 both run aligned.
#[test]
fn the_aligned_tile_gives_the_checked_tile_s_bits() {
    let _t = turn();
    let Some(_) = metal_device("the_aligned_tile_gives_the_checked_tile_s_bits") else { return };
    let mut m = model_manifest();
    m["architecture"]["layers"] = json!(2);
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(128);
    m["embed"]["dim"] = json!(64);
    let mut f = Fixture::new("metal-aligned", m);
    f.weights("weights/model.safetensors", &bert_weights(2, 64, 128, 7, 0.05));
    f.write();
    let (g, c) = (on_metal(&f.dir), Loaded::load(&f.dir).unwrap());
    for lens in [&[64, 64, 3][..], &[64, 61]] {
        let t = rows_of(lens);
        let rows = |line: &str| {
            turbo::metal::use_choices(Some(line));
            let s = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST)));
            turbo::metal::use_choices(None);
            let s = s.unwrap();
            assert_eq!(field(&s.info().choices), format!("{line};attn=auto;forced=linear"));
            s.write_tokens(&t.batch(), None).unwrap();
            s.run().unwrap().rows()
        };
        let checked = rows("linear=h32");
        for line in ["linear=h32n32", "linear=h64n32"] {
            assert_eq!(rows(line), checked, "{line} and the checked 32 x 64 differ at {lens:?}");
        }
        let cpu = Session::create(c.m, None).unwrap();
        cpu.write_tokens(&t.batch(), None).unwrap();
        for (r, (a, b)) in checked.iter().zip(cpu.run().unwrap().rows()).enumerate() {
            let cos = cosine(a, &b);
            assert!(cos >= 0.999, "row {r} at {lens:?}: cosine {cos} with the cpu");
        }
    }
}
