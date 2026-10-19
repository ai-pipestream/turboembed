use crate::backend::turbo_backend_embed_rows;
use crate::npu::ze;
use crate::status::{CAPACITY, INVALID_ARGUMENT, UNSUPPORTED_OPTION};
use crate::{TURBO_NORMALIZE_L2, TURBO_POOLING_CLS, TURBO_POOLING_MEAN, turbo_token_range};

use super::buffers::*;
use super::host::*;
use super::run::*;
use super::session::*;
use super::tests::ok_msg;
use super::write::*;

/// Rows of `seq` columns at stride `stride` with the given masks, set
/// up for a write of rows or of ranges, with hidden width `hidden`.
fn pooling_rows(masks: &[&[i32]], stride: usize, hidden: usize, normalize: u32, ranges: &[turbo_token_range]) -> Rows {
    let mut mask = vec![0; masks.len() * stride];
    for (r, m) in masks.iter().enumerate() {
        mask[r * stride..r * stride + m.len()].copy_from_slice(m);
    }
    let mut kept = Vec::with_capacity(8);
    kept.extend_from_slice(ranges);
    Rows {
        ids: vec![0; masks.len() * stride],
        mask,
        types: vec![0; masks.len() * stride],
        acc: vec![0.0; hidden],
        token: vec![0.0; hidden],
        gathered_rows: Vec::new(),
        gathered_bias: Vec::new(),
        ranges: kept,
        state: State {
            batch: masks.len() as u32,
            seq: masks[0].len() as u32,
            pooling: TURBO_POOLING_MEAN,
            normalize,
            output_dim: hidden as u32,
            vectors: if ranges.is_empty() { Vectors::Rows } else { Vectors::Ranges },
            written: true,
        },
    }
}

/// Hidden states for a frame of `frame_batch` rows: distinct values
/// that are not exact sums of one another, so a wrong token pooled
/// changes the bits.
fn hidden_frame(frame: usize, frame_batch: usize, model_seq: usize, hidden: usize) -> Vec<f32> {
    (0..frame_batch * model_seq * hidden).map(|i| ((frame * 1000 + i) as f32 * 0.37).sin() * 3.1).collect()
}

/// Every frame of a batch through pool_frame, as run does, into a
/// result of `vectors` rows.
fn pool_all(rows: &mut Rows, frames: &[Vec<f32>], frame_batch: usize, model_seq: usize, vectors: usize) -> Vec<f32> {
    let hidden = rows.acc.len();
    let stride = rows.mask.len() / rows.state.batch as usize;
    let batch = rows.state.batch as usize;
    let mut result = vec![f32::NAN; vectors * hidden];
    for (f, values) in frames.iter().enumerate() {
        let first = f * frame_batch;
        let live = (batch - first).min(frame_batch);
        let states = FrameStates { values: Hidden::F32(values), model_seq, hidden };
        pool_frame(&states, first, live, stride, rows, &mut result);
    }
    result
}

/// The mean in f64 of row `r` of a frame over the columns `cols`,
/// cut to nothing and not normalized: the oracle for a range.
fn host_mean(values: &[f32], r: usize, cols: &[usize], model_seq: usize, hidden: usize) -> Vec<f32> {
    let mut acc = vec![0f64; hidden];
    for &t in cols {
        for (h, a) in acc.iter_mut().enumerate() {
            *a += values[(r * model_seq + t) * hidden + h] as f64;
        }
    }
    acc.iter().map(|a| (a / cols.len() as f64) as f32).collect()
}

#[test]
fn a_whole_row_range_pools_the_bits_the_row_does() {
    // Three rows over two frames of two, written at a stride longer
    // than seq, with padding at the ends of rows 1 and 2.
    let (frame_batch, model_seq, hidden, stride) = (2, 5, 3, 6);
    let masks: [&[i32]; 3] = [&[1, 1, 1, 1], &[1, 1, 1, 0], &[1, 1, 0, 0]];
    let frames: Vec<Vec<f32>> = (0..2).map(|f| hidden_frame(f, frame_batch, model_seq, hidden)).collect();
    for normalize in [TURBO_NORMALIZE_L2, crate::TURBO_NORMALIZE_NONE] {
        let mut by_rows = pooling_rows(&masks, stride, hidden, normalize, &[]);
        let rows = pool_all(&mut by_rows, &frames, frame_batch, model_seq, 3);
        // Out of row order, the last frame's row first, a row twice.
        let ranges = [
            turbo_token_range { row: 2, start: 0, end: 4 },
            turbo_token_range { row: 0, start: 0, end: 4 },
            turbo_token_range { row: 1, start: 0, end: 4 },
            turbo_token_range { row: 2, start: 0, end: 4 },
        ];
        let mut by_ranges = pooling_rows(&masks, stride, hidden, normalize, &ranges);
        let got = pool_all(&mut by_ranges, &frames, frame_batch, model_seq, 4);
        for (i, g) in ranges.iter().enumerate() {
            let row = g.row as usize;
            let want: Vec<u32> = rows[row * hidden..(row + 1) * hidden].iter().map(|v| v.to_bits()).collect();
            let have: Vec<u32> = got[i * hidden..(i + 1) * hidden].iter().map(|v| v.to_bits()).collect();
            assert_eq!(have, want, "range {i} over the whole of row {row}, normalize {normalize}");
        }
    }
}

#[test]
fn a_range_is_the_mean_of_its_live_tokens_alone() {
    // Row 0 has a hole at column 1; row 1 has padding at the end.
    // Ranges over part of a row, over the hole, and over padding pool
    // the live tokens they cover and nothing else.
    let (frame_batch, model_seq, hidden, stride) = (2, 4, 3, 4);
    let masks: [&[i32]; 2] = [&[1, 0, 1, 1], &[1, 1, 0, 0]];
    let frames = vec![hidden_frame(0, frame_batch, model_seq, hidden)];
    let ranges = [
        turbo_token_range { row: 0, start: 1, end: 3 },
        turbo_token_range { row: 0, start: 0, end: 3 },
        turbo_token_range { row: 1, start: 1, end: 4 },
        turbo_token_range { row: 0, start: 2, end: 4 },
        turbo_token_range { row: 1, start: 0, end: 1 },
    ];
    let live: [(usize, &[usize]); 5] = [(0, &[2]), (0, &[0, 2]), (1, &[1]), (0, &[2, 3]), (1, &[0])];
    let mut rows = pooling_rows(&masks, stride, hidden, crate::TURBO_NORMALIZE_NONE, &ranges);
    let got = pool_all(&mut rows, &frames, frame_batch, model_seq, ranges.len());
    for (i, (row, cols)) in live.iter().enumerate() {
        let want = host_mean(&frames[0], *row, cols, model_seq, hidden);
        assert_eq!(&got[i * hidden..(i + 1) * hidden], &want[..], "range {i}");
    }
}

#[test]
fn a_range_vector_is_cut_before_it_is_normalized() {
    let (frame_batch, model_seq, hidden) = (1, 3, 4);
    let masks: [&[i32]; 1] = [&[1, 1, 1]];
    let frames = vec![hidden_frame(0, frame_batch, model_seq, hidden)];
    let ranges = [turbo_token_range { row: 0, start: 0, end: 2 }];
    let mut rows = pooling_rows(&masks, 3, hidden, TURBO_NORMALIZE_L2, &ranges);
    rows.state.output_dim = 2;
    let got = pool_all(&mut rows, &frames, frame_batch, model_seq, 2);
    let mean = host_mean(&frames[0], 0, &[0, 1], model_seq, hidden);
    let norm = mean[..2].iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    for h in 0..2 {
        assert!((got[h] as f64 - mean[h] as f64 / norm).abs() < 1e-6, "{h}: {} {}", got[h], mean[h]);
    }
    assert!(got[2..].iter().all(|v| v.is_nan()), "nothing is written past output_dim");
}

#[test]
fn an_fp16_frame_pools_the_values_its_halves_name() {
    // Halves of values exact in both precisions: the FP16 frame,
    // widened on the way, pools to the F32 frame's bits.
    let (model_seq, hidden) = (3, 2);
    let wide = [1.0f32, -2.0, 0.5, 4.0, -0.25, 3.0];
    let mut halves: Vec<u16> = wide.iter().map(|&v| f32_to_half(v)).collect();
    let buf = Buffer { free: None, ptr: halves.as_mut_ptr().cast(), bytes: (halves.len() * 2) as u64 };
    let fp16 = unsafe { FrameStates::read(&buf, ze::GRAPH_ARGUMENT_PRECISION_FP16, model_seq, hidden) };
    assert!(matches!(fp16.values, Hidden::F16(h) if h.len() == 6));
    let fp32 = FrameStates { values: Hidden::F32(&wide), model_seq, hidden };
    let ranges = [turbo_token_range { row: 0, start: 0, end: 3 }];
    let mut a = pooling_rows(&[&[1, 1, 0]], 3, hidden, TURBO_NORMALIZE_L2, &ranges);
    let mut b = pooling_rows(&[&[1, 1, 0]], 3, hidden, TURBO_NORMALIZE_L2, &ranges);
    let (mut ra, mut rb) = (vec![0f32; hidden], vec![0f32; hidden]);
    pool_frame(&fp16, 0, 1, 3, &mut a, &mut ra);
    pool_frame(&fp32, 0, 1, 3, &mut b, &mut rb);
    assert_eq!(ra, rb);
    // The mean of tokens 0 and 1 alone: (1 + 0.5) / 2 and (-2 + 4) / 2.
    let norm = (0.75f64 * 0.75 + 1.0).sqrt();
    assert_eq!(ra, [(0.75 / norm) as f32, (1.0 / norm) as f32]);
}

fn embed_rows(batch: u32, seq: u32, pooling: u32) -> turbo_backend_embed_rows {
    turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq,
        row_stride: seq,
        ids: std::ptr::null(),
        mask: std::ptr::null(),
        types: std::ptr::null(),
        pooling,
        normalize: 0,
        output_dim: 2,
        reserved: 0,
    }
}

#[test]
fn ranges_are_kept_in_order_in_the_memory_made_for_them() {
    let mut rows = pooling_rows(&[&[1, 1, 0], &[0, 1, 1]], 4, 2, 0, &[]);
    let capacity = rows.ranges.capacity();
    let ranges = [
        turbo_token_range { row: 1, start: 1, end: 3 },
        turbo_token_range { row: 0, start: 0, end: 3 },
        turbo_token_range { row: 1, start: 1, end: 3 },
    ];
    ok_msg(keep_ranges(3, 4, &embed_rows(2, 3, TURBO_POOLING_MEAN), &ranges, &mut rows));
    assert_eq!(rows.ranges.len(), 3);
    assert_eq!(rows.ranges.capacity(), capacity, "nothing was allocated");
    for (k, g) in rows.ranges.iter().zip(&ranges) {
        assert_eq!((k.row, k.start, k.end), (g.row, g.start, g.end));
    }
}

#[test]
fn a_range_the_rows_cannot_pool_is_refused_by_its_index() {
    let rows_of = || pooling_rows(&[&[1, 1, 0], &[0, 1, 1]], 4, 2, 0, &[]);
    let r = embed_rows(2, 3, TURBO_POOLING_MEAN);
    let ok = turbo_token_range { row: 0, start: 0, end: 2 };
    let refused = |ranges: &[turbo_token_range], rows_desc: &turbo_backend_embed_rows| {
        let mut rows = rows_of();
        match keep_ranges(3, 4, rows_desc, ranges, &mut rows) {
            Ok(()) => panic!("expected a refusal"),
            Err(e) => {
                assert!(rows.ranges.is_empty(), "a refused write keeps no ranges");
                e
            }
        }
    };
    let e = refused(&[ok; 4], &r);
    assert_eq!(e.code, CAPACITY, "{}", e.message);
    assert!(e.message.contains("max_ranges 3"), "{}", e.message);
    let e = refused(&[ok], &embed_rows(2, 3, TURBO_POOLING_CLS));
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 5), "{}", e.message);
    for (bad, why) in [
        (turbo_token_range { row: 2, start: 0, end: 1 }, "row outside the batch"),
        (turbo_token_range { row: 0, start: 1, end: 1 }, "start not under end"),
        (turbo_token_range { row: 0, start: 2, end: 4 }, "end over seq"),
    ] {
        let e = refused(&[ok, bad], &r);
        assert_eq!(e.code, INVALID_ARGUMENT, "{why}: {}", e.message);
        assert!(e.message.contains("range 1 "), "{why}: {}", e.message);
    }
    let e = refused(&[ok, ok, turbo_token_range { row: 1, start: 0, end: 1 }], &r);
    assert_eq!(e.code, INVALID_ARGUMENT);
    assert!(e.message.contains("range 2 has no token with mask 1"), "{}", e.message);
}
