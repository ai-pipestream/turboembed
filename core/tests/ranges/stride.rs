//! A range reads its row through the row stride.

use crate::common::*;
use crate::support::*;
use turbo::*;

/// `t` laid out with `gap` more columns per row than seq, the gap filled
/// with live tokens the batch must not reach.
fn strided(t: &Tokens, gap: u32) -> Tokens {
    let (seq, stride) = (t.seq as usize, (t.seq + gap) as usize);
    let mut s = Tokens {
        ids: vec![1000; t.batch as usize * stride],
        mask: vec![1; t.batch as usize * stride],
        types: None,
        batch: t.batch,
        seq: t.seq,
        stride: stride as u32,
    };
    for r in 0..t.batch as usize {
        s.ids[r * stride..r * stride + seq].copy_from_slice(&t.ids[r * seq..(r + 1) * seq]);
        s.mask[r * stride..r * stride + seq].copy_from_slice(&t.mask[r * seq..(r + 1) * seq]);
    }
    s
}

#[test]
fn a_range_reads_its_row_through_the_row_stride() {
    let dir = tiny_bundle();
    let l = tiny();
    let s = session_of(&l, 5);
    let (t, lens) = tokens_of(&dir, &RANGES_TEXTS);
    let (l0, l1, l2) = (lens[0] as u32, lens[1] as u32, lens[2] as u32);
    let ranges = [range(0, 0, l0), range(1, 1, l1), range(2, 0, l2), range(0, 2, 5), range(1, 0, t.seq)];
    let o = normalized(TURBO_NORMALIZE_L2);
    let packed = run_ranges(&s, &t, &ranges, &o);
    let wide = strided(&t, 3);
    let got = run_ranges(&s, &wide, &ranges, &o);
    for (i, (a, b)) in packed.iter().zip(&got).enumerate() {
        assert_eq!(bits(a), bits(b), "range {i}: the gap past seq is not read");
    }
    // A range holding no live token pools those mask-0 columns through
    // the stride exactly as packed: the same bits either way.
    let masked = [range(2, l2, t.seq)];
    let packed = run_ranges(&s, &t, &masked, &o);
    let got = run_ranges(&s, &wide, &masked, &o);
    assert_eq!(bits(&packed[0]), bits(&got[0]), "a mask-0 range reads its row through the stride");
    assert!(packed[0].iter().all(|v| v.is_finite()));
}
