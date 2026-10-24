use std::sync::Arc;

use super::super::kernels::Isa;
use super::super::pool::Pool;
use super::round::f32_to_f16;
use super::*;
use crate::backend::turbo_backend_embed_rows;
use crate::{TURBO_DTYPE_F32, TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, turbo_token_range};

const VOCAB: usize = 300;
const DIM: usize = 77;

/// A table's stored values, kept alive while the table reads them.
enum Values {
    F32(Vec<f32>),
    F16(Vec<u16>),
    F64(Vec<f64>),
}

/// A synthetic table of `dtype` (F32, F16 or F64), with token weights
/// when asked, its values kept alongside.
fn table(dtype: u32, weighted: bool) -> (Arc<Table>, Values) {
    let value = |i: usize| ((i * 7919 % 2003) as f32 - 1001.0) / 997.0;
    let values = match dtype {
        TURBO_DTYPE_F16 => Values::F16((0..VOCAB * DIM).map(|i| f32_to_f16(value(i))).collect()),
        TURBO_DTYPE_F64 => Values::F64((0..VOCAB * DIM).map(|i| value(i) as f64 / 3.0).collect()),
        _ => Values::F32((0..VOCAB * DIM).map(value).collect()),
    };
    let rows = match &values {
        Values::F32(v) => v.as_ptr() as *const u8,
        Values::F16(v) => v.as_ptr() as *const u8,
        Values::F64(v) => v.as_ptr() as *const u8,
    };
    let weights = weighted.then(|| (0..VOCAB).map(|i| 0.5 + (i % 13) as f64 / 8.0).collect::<Box<[f64]>>());
    let t = Table {
        rows,
        dtype,
        dim: DIM,
        height: VOCAB,
        weights,
        mapping: None,
        wide: dtype == TURBO_DTYPE_F64,
        _copy: None,
    };
    (Arc::new(t), values)
}

/// Every kind of table the sum has a path for: F32 with and without
/// weights, F16 (rounded to F16 as numpy stores it), F64 (summed in
/// F64), and the I8 copy.
fn sessions(max_batch: usize, max_seq: usize, max_ranges: usize) -> Vec<(String, Static, Values)> {
    let mut all = Vec::new();
    for (dtype, weighted, quantized) in [
        (TURBO_DTYPE_F32, false, false),
        (TURBO_DTYPE_F32, true, false),
        (TURBO_DTYPE_F16, false, false),
        (TURBO_DTYPE_F16, true, false),
        (TURBO_DTYPE_F64, false, false),
        (TURBO_DTYPE_F32, true, true),
    ] {
        let (t, values) = table(dtype, weighted);
        let q = quantized.then(|| Arc::new(Quantized::new(&t, false).unwrap()));
        let s = Static::new(t, q, Isa::detect(), max_batch, max_seq, max_ranges).unwrap();
        all.push((format!("dtype {dtype}, weights {weighted}, i8 {quantized}"), s, values));
    }
    all
}

fn rows(ids: &[i32], mask: &[i32], batch: usize, seq: usize, normalize: u32, od: usize) -> turbo_backend_embed_rows {
    turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch: batch as u32,
        seq: seq as u32,
        row_stride: seq as u32,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: std::ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize,
        output_dim: od as u32,
        head: 0,
    }
}

/// Rows of assorted lengths with padding, one with masked tokens
/// between live ones.
fn batch() -> (Vec<i32>, Vec<i32>, usize, usize) {
    let (batch, seq) = (5, 40);
    let lens = [40, 1, 23, 9, 31];
    let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
    for (r, &n) in lens.iter().enumerate() {
        for p in 0..n {
            ids[r * seq + p] = ((r * 131 + p * 17) % VOCAB) as i32;
            mask[r * seq + p] = 1;
        }
    }
    for p in [0, 4, 5, 22] {
        mask[2 * seq + p] = 0;
    }
    (ids, mask, batch, seq)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn range(row: usize, start: usize, end: usize) -> turbo_token_range {
    turbo_token_range { row: row as u32, start: start as u32, end: end as u32 }
}

/// Each range's vector is the vector of a row holding the range's
/// columns alone, bit for bit: the mean of the same live tokens in
/// the same order, rounded and normalized the same way. Ranges over
/// a whole row, single tokens, and ranges into padding or over masked
/// tokens, cut to the full width and to less.
#[test]
fn a_range_is_the_mean_of_its_live_tokens() {
    let (ids, mask, b, s) = batch();
    let ranges = [
        range(0, 0, 40),
        range(4, 0, 40),
        range(1, 0, 40),
        range(0, 39, 40),
        range(0, 7, 8),
        range(2, 0, 2),
        range(2, 3, 7),
        range(2, 20, 40),
        range(3, 8, 9),
        range(4, 10, 30),
        range(0, 7, 8),
    ];
    // The rows each range alone would make, as a batch.
    let n = ranges.len();
    let (mut sub_ids, mut sub_mask) = (vec![0; n * s], vec![0; n * s]);
    for (k, r) in ranges.iter().enumerate() {
        for (j, p) in (r.start as usize..r.end as usize).enumerate() {
            sub_ids[k * s + j] = ids[r.row as usize * s + p];
            sub_mask[k * s + j] = mask[r.row as usize * s + p];
        }
    }
    for (normalize, od) in [(TURBO_NORMALIZE_L2, DIM), (TURBO_NORMALIZE_NONE, DIM), (TURBO_NORMALIZE_L2, 64)] {
        for (name, mut st, _values) in sessions(n, s, n) {
            let mut pool = Pool::new(2);
            let mut want = vec![f32::NAN; n * od];
            st.write(&rows(&sub_ids, &sub_mask, n, s, normalize, od), &sub_ids, &sub_mask);
            st.run(&mut pool, &mut want);
            let mut got = vec![f32::NAN; n * od];
            st.write_ranges(&rows(&ids, &mask, b, s, normalize, od), &ids, &mask, &ranges);
            st.run(&mut pool, &mut got);
            assert!(want.iter().all(|v| v.is_finite()), "{name}");
            for k in 0..n {
                let (g, w) = (&got[k * od..][..od], &want[k * od..][..od]);
                assert!(bits(g) == bits(w), "{name}, {od} values: range {k} {:?}", ranges[k]);
            }
        }
    }
}

/// A range over a whole row, padding included, is that row's mean.
#[test]
fn a_range_over_a_whole_row_is_the_row_s_mean() {
    let (ids, mask, b, s) = batch();
    let ranges: Vec<_> = (0..b).rev().map(|r| range(r, 0, s)).collect();
    for (name, mut st, _values) in sessions(b, s, b) {
        let mut pool = Pool::new(1);
        let mut want = vec![f32::NAN; b * DIM];
        st.write(&rows(&ids, &mask, b, s, TURBO_NORMALIZE_L2, DIM), &ids, &mask);
        st.run(&mut pool, &mut want);
        let mut got = vec![f32::NAN; b * DIM];
        st.write_ranges(&rows(&ids, &mask, b, s, TURBO_NORMALIZE_L2, DIM), &ids, &mask, &ranges);
        st.run(&mut pool, &mut got);
        for (k, r) in ranges.iter().enumerate() {
            let row = r.row as usize;
            assert!(bits(&got[k * DIM..][..DIM]) == bits(&want[row * DIM..][..DIM]), "{name}: row {row}");
        }
    }
}

/// Enough ranges, and tokens in them, that the run splits them over
/// threads: the same bits on 1, 3 and every processor's thread.
#[test]
fn ranges_are_the_same_bits_on_any_number_of_threads() {
    let (b, s) = (8, 300);
    let mut ids = vec![0; b * s];
    let mut mask = vec![1; b * s];
    for (i, id) in ids.iter_mut().enumerate() {
        *id = ((i * 37) % VOCAB) as i32;
    }
    // Masked columns are even, and every range starts on an odd one,
    // so each holds a live token.
    for p in (2..s).step_by(10) {
        mask[3 * s + p] = 0;
    }
    let mut ranges = Vec::new();
    for r in 0..b {
        for start in (1..s).step_by(2) {
            ranges.push(range(r, start, (start + 1 + start % 23).min(s)));
        }
    }
    let tokens: usize = ranges.iter().map(|r| (r.end - r.start) as usize).sum();
    assert!(tokens > 4 * TOKENS_PER_TASK);
    let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    for (name, mut st, _values) in sessions(b, s, ranges.len()) {
        let mut want = Vec::new();
        for threads in [1, 3, most] {
            let mut pool = Pool::with_spin(threads, super::super::pool::SPIN_BATCH);
            let mut got = vec![f32::NAN; ranges.len() * DIM];
            st.write_ranges(&rows(&ids, &mask, b, s, TURBO_NORMALIZE_L2, DIM), &ids, &mask, &ranges);
            st.run(&mut pool, &mut got);
            if want.is_empty() {
                want = got;
            } else {
                assert!(bits(&got) == bits(&want), "{name}: {threads} threads differ from 1");
            }
        }
    }
}
