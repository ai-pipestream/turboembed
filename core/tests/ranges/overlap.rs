//! Overlap, repeats, order and skipped rows.

use crate::common::*;
use crate::support::*;
use turbo::*;

#[test]
fn ranges_may_overlap_repeat_and_skip_rows() {
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let l = tiny();
    let s = session_of(&l, 5);
    let (t, lens) = tokens_of(&dir, &RANGES_TEXTS);
    let (l0, l2) = (lens[0] as u32, lens[2] as u32);
    // Row 1 has no range; row 2 comes before row 0; the first and third
    // ranges are the same; the fourth overlaps the second.
    let ranges = [range(2, 1, l2), range(0, 0, l0), range(2, 1, l2), range(0, 3, l0 - 1), range(2, 0, 2)];
    let o = normalized(TURBO_NORMALIZE_L2);
    let got = run_ranges(&s, &t, &ranges, &o);
    assert_eq!(got.len(), 5);
    assert_eq!(bits(&got[0]), bits(&got[2]), "the same range twice");
    for (i, (g, rg)) in got.iter().zip(&ranges).enumerate() {
        let (ids, mask, ty) = row_of(&t, rg.row as usize);
        let want = plain.embed_range(ids, mask, &ty, rg.start as usize, rg.end as usize, 32, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5 * (1.0 + b.abs()), "range {i}: {a} vs {b}");
        }
    }

    // The order of the ranges is the order of the vectors, and nothing else
    // about a vector depends on it.
    let reversed: Vec<_> = ranges.iter().rev().copied().collect();
    let back = run_ranges(&s, &t, &reversed, &o);
    for (i, g) in got.iter().enumerate() {
        assert_eq!(bits(g), bits(&back[4 - i]), "range {i}");
    }
}
