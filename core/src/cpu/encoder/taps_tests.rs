//! A range write with taps (turbo_backend.h, `taps`): block k of the
//! output is what a model of the first L - T + k + 1 layers writes for the
//! same ranges, bit for bit, on any number of threads, for a BERT, a
//! ModernBERT (final_norm applied to every tap) and a nomic-bert; the last
//! block is the plain range write's.

use super::super::kernels::{Isa, Packed};
use super::super::pool::Pool;
use super::rotary::tests::Model as Rotary;
use super::tests::Model as Bert;
use super::*;
use crate::{TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

/// The rows every model reads: three rows of 40 columns, some masked.
const BATCH: usize = 3;
const SEQ: usize = 40;
const LENS: [usize; BATCH] = [40, 23, 9];

fn ranges() -> Vec<turbo_token_range> {
    let r = |row, start, end| turbo_token_range { row, start, end };
    vec![r(0, 0, 40), r(0, 3, 4), r(1, 5, 17), r(2, 0, 9), r(1, 0, 1), r(0, 10, 30), r(2, 4, 6)]
}

/// The ranges' vectors from the last `taps` layers, [taps, ranges, od].
#[allow(clippy::too_many_arguments)]
fn run(
    desc: &turbo_backend_model,
    slices: Vec<&[f32]>,
    packed: &Packed,
    threads: usize,
    taps: u32,
    normalize: u32,
    od: usize,
    types: bool,
) -> Vec<f32> {
    let ids: Vec<i32> = (0..BATCH * SEQ).map(|k| ((k % SEQ * 37 + LENS[k / SEQ] * 11) % 300) as i32).collect();
    let mask: Vec<i32> = (0..BATCH * SEQ).map(|k| i32::from(k % SEQ < LENS[k / SEQ])).collect();
    let tys: Vec<i32> = (0..BATCH * SEQ).map(|k| (k % SEQ).is_multiple_of(5) as i32).collect();
    let ranges = ranges();
    let taps_n = (taps as usize).max(1);
    let size = Size { threads, max_batch: BATCH, max_seq: SEQ, max_ranges: ranges.len() * taps_n };
    let mut e = Encoder::new(desc, slices, packed, size).unwrap();
    let rows = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch: BATCH as u32,
        seq: SEQ as u32,
        row_stride: SEQ as u32,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: if types { tys.as_ptr() } else { std::ptr::null() },
        pooling: TURBO_POOLING_MEAN,
        normalize,
        output_dim: od as u32,
        head: 0,
        taps,
        reserved: 0,
    };
    let mut pool = Pool::new(threads);
    let mut out = vec![f32::NAN; taps_n * ranges.len() * od];
    e.write_ranges(&rows, &ids, &mask, types.then_some(&tys[..]), &ranges);
    e.run(&mut pool, &mut out);
    assert!(out.iter().all(|v| v.is_finite()));
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Each block of `tapped`, from the last `taps` of `layers` layers,
/// against `alone(j)`, the plain range write of a model of j layers.
fn agree(tapped: &[f32], taps: usize, layers: usize, alone: &dyn Fn(usize) -> Vec<f32>, what: &str) {
    let block = tapped.len() / taps;
    for k in 0..taps {
        let j = layers - taps + k + 1;
        assert_eq!(bits(&tapped[k * block..][..block]), bits(&alone(j)), "{what}: tap {k}, layer {j}");
    }
}

#[test]
fn a_bert_tap_is_the_bert_cut_after_its_layer() {
    let isa = Isa::detect();
    let model = |l: usize| Bert::shaped(l, 96, 3, 192, 128);
    let deep = model(3);
    let packed = deep.packed(isa);
    for (normalize, od) in [(TURBO_NORMALIZE_NONE, 96), (TURBO_NORMALIZE_L2, 64)] {
        let alone = |j: usize| {
            let m = model(j);
            run(&m.desc, m.slices(), &m.packed(isa), 2, 0, normalize, od, false)
        };
        for taps in [1u32, 2, 3] {
            let one = run(&deep.desc, deep.slices(), &packed, 1, taps, normalize, od, false);
            let many = run(&deep.desc, deep.slices(), &packed, 4, taps, normalize, od, false);
            assert_eq!(bits(&one), bits(&many), "taps {taps}: threads");
            agree(&one, taps as usize, 3, &alone, &format!("bert normalize {normalize} od {od}"));
        }
    }
}

#[test]
fn a_rotary_tap_is_the_rotary_model_cut_after_its_layer() {
    let isa = Isa::detect();
    for modern in [true, false] {
        let deep = Rotary::layered(modern, 4);
        let packed = Packed::rotary(&deep.desc, &deep.slices(), isa).unwrap();
        let alone = |j: usize| {
            let m = Rotary::layered(modern, j);
            let p = Packed::rotary(&m.desc, &m.slices(), isa).unwrap();
            run(&m.desc, m.slices(), &p, 2, 0, TURBO_NORMALIZE_NONE, 64, !modern)
        };
        let one = run(&deep.desc, deep.slices(), &packed, 1, 4, TURBO_NORMALIZE_NONE, 64, !modern);
        let many = run(&deep.desc, deep.slices(), &packed, 3, 4, TURBO_NORMALIZE_NONE, 64, !modern);
        assert_eq!(bits(&one), bits(&many), "modern {modern}: threads");
        agree(&one, 4, 4, &alone, &format!("modern {modern}"));
    }
}
