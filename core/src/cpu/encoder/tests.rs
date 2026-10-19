use super::super::kernels::{Isa, isas, values};
use super::*;
use crate::{TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

/// A small BERT with synthetic weights: 2 layers, hidden 96 in 3
/// heads of 32, intermediate 192.
struct Model {
    desc: turbo_backend_model,
    tensors: Vec<Vec<f32>>,
}

impl Model {
    fn new() -> Model {
        Model::shaped(2, 96, 3, 192, 128)
    }

    fn shaped(layers: usize, h: usize, heads: u32, i: usize, positions: usize) -> Model {
        let vocab = 500usize;
        let mut shapes = vec![vocab * h, positions * h, 2 * h, h, h];
        for _ in 0..layers {
            shapes.extend([h * h, h, h * h, h, h * h, h, h * h, h, h, h, i * h, i, h * i, h, h, h]);
        }
        let tensors =
            shapes.iter().enumerate().map(|(n, &len)| values(len, n, if len > h { 0.1 } else { 0.5 })).collect();
        let desc = turbo_backend_model {
            struct_size: size_of::<turbo_backend_model>() as u32,
            family: crate::backend::TURBO_FAMILY_BERT,
            dtype: TURBO_DTYPE_F32,
            layers: layers as u32,
            hidden: h as u32,
            heads,
            intermediate: i as u32,
            vocab_size: vocab as u32,
            max_positions: positions as u32,
            token_types: 2,
            layer_norm_eps: 1e-12,
            tensor_count: shapes.len() as u32,
            position_offset: 0,
            tensors: std::ptr::null(),
            format: crate::backend::TURBO_FORMAT_SAFETENSORS,
            graph_input: crate::backend::TURBO_INPUT_TOKEN_IDS,
            graph_output: crate::backend::TURBO_OUTPUT_HIDDEN_STATES,
            compute_dtype: 0,
            fixed_seq: 0,
            fixed_batch: 0,
            artifact: std::ptr::null(),
            artifact_bytes: 0,
            artifact2: std::ptr::null(),
            artifact2_bytes: 0,
        };
        Model { desc, tensors }
    }

    fn slices(&self) -> Vec<&[f32]> {
        self.tensors.iter().map(Vec::as_slice).collect()
    }

    fn packed(&self, isa: Isa) -> Packed {
        let t = self.slices();
        let layer = |l: usize, r: usize| t[5 + l * 16 + r];
        Packed::new(&self.desc, layer, isa).unwrap()
    }

    /// The batch's vectors, run on `threads` threads with `isa`.
    fn run(&self, isa: Isa, threads: usize, ids: &[i32], mask: &[i32], batch: usize, seq: usize) -> Vec<f32> {
        self.timed(isa, threads, ids, mask, batch, seq, 2).0
    }

    /// As run, `runs` times; also the fastest run's time.
    #[allow(clippy::too_many_arguments)]
    fn timed(
        &self,
        isa: Isa,
        threads: usize,
        ids: &[i32],
        mask: &[i32],
        batch: usize,
        seq: usize,
        runs: usize,
    ) -> (Vec<f32>, std::time::Duration) {
        let packed = self.packed(isa);
        let mut pool = Pool::new(threads);
        assert_eq!(pool.threads(), threads);
        let size = Size { threads, max_batch: batch, max_seq: seq, max_ranges: batch };
        let mut e = Encoder::new(&self.desc, self.slices(), &packed, size).unwrap();
        let rows = turbo_backend_embed_rows {
            struct_size: size_of::<turbo_backend_embed_rows>() as u32,
            batch: batch as u32,
            seq: seq as u32,
            row_stride: seq as u32,
            ids: ids.as_ptr(),
            mask: mask.as_ptr(),
            types: std::ptr::null(),
            pooling: TURBO_POOLING_MEAN,
            normalize: TURBO_NORMALIZE_L2,
            output_dim: self.desc.hidden,
            reserved: 0,
        };
        let mut out = vec![f32::NAN; batch * self.desc.hidden as usize];
        // More than once, to see later runs on the same session agree.
        let mut first = Vec::new();
        let mut best = std::time::Duration::MAX;
        for _ in 0..runs {
            let t = std::time::Instant::now();
            e.write(&rows, ids, mask, None);
            e.run(&mut pool, &mut out);
            best = best.min(t.elapsed());
            if first.is_empty() {
                first = out.clone();
            }
        }
        assert!(bits(&first) == bits(&out), "{isa:?} on {threads} threads: a later run differs");
        (out, best)
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Rows of assorted lengths, some longer than an attention block, one
/// with masked tokens between live ones, and padding.
fn batch() -> (Vec<i32>, Vec<i32>, usize, usize) {
    let (batch, seq) = (9, 100);
    let lens = [100, 1, 37, 64, 65, 2, 33, 90, 17];
    let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
    for (r, &n) in lens.iter().enumerate() {
        for p in 0..n {
            ids[r * seq + p] = ((r * 131 + p * 17) % 500) as i32;
            mask[r * seq + p] = 1;
        }
    }
    for p in [3, 10, 11, 40] {
        mask[2 * seq + p] = 0;
    }
    (ids, mask, batch, seq)
}

/// The same batch gives the same bits on any number of threads, and
/// with any kernel that fuses multiply and add.
#[test]
fn vectors_are_the_same_bits_on_any_number_of_threads() {
    let m = Model::new();
    let (ids, mask, b, s) = batch();
    let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    for isa in isas() {
        let want = m.run(isa, 1, &ids, &mask, b, s);
        assert!(want.iter().all(|v| v.is_finite()));
        for threads in [2, 3, 4, 7, 16, most] {
            let got = m.run(isa, threads, &ids, &mask, b, s);
            assert!(bits(&got) == bits(&want), "{isa:?}: {threads} threads differ from 1");
        }
    }
    let fused: Vec<Vec<u32>> = isas()
        .into_iter()
        .filter(|&i| i != Isa::Portable || cfg!(target_arch = "aarch64"))
        .map(|i| bits(&m.run(i, 3, &ids, &mask, b, s)))
        .collect();
    assert!(fused.windows(2).all(|w| w[0] == w[1]), "the fused kernels differ");
}

/// A run of `ranges` over the batch, on `threads` threads with the
/// fastest kernel: the vectors, and the hidden states each was pooled
/// from with each row's place among them.
struct Ranged {
    out: Vec<f32>,
    x: Vec<f32>,
    rows: Vec<(usize, usize)>,
}

#[allow(clippy::too_many_arguments)]
fn ranged(
    m: &Model,
    threads: usize,
    (ids, mask, batch, seq): (&[i32], &[i32], usize, usize),
    ranges: &[turbo_token_range],
    normalize: u32,
    output_dim: usize,
) -> Ranged {
    let isa = Isa::detect();
    let packed = m.packed(isa);
    let mut pool = Pool::new(threads);
    let size = Size { threads, max_batch: batch, max_seq: seq, max_ranges: ranges.len() };
    let mut e = Encoder::new(&m.desc, m.slices(), &packed, size).unwrap();
    let rows = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch: batch as u32,
        seq: seq as u32,
        row_stride: seq as u32,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: std::ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize,
        output_dim: output_dim as u32,
        reserved: 0,
    };
    let mut out = vec![f32::NAN; ranges.len() * output_dim];
    e.write_ranges(&rows, ids, mask, None, ranges);
    e.run(&mut pool, &mut out);
    let (x, packed_rows) = e.hidden_states();
    Ranged { out, x: x.to_vec(), rows: packed_rows.to_vec() }
}

fn range(row: usize, start: usize, end: usize) -> turbo_token_range {
    turbo_token_range { row: row as u32, start: start as u32, end: end as u32 }
}

/// A range over a whole row, padding and all, is the row's mean
/// vector bit for bit, normalized the same way, in any order and
/// when given twice.
#[test]
fn a_range_over_a_whole_row_is_the_row_s_mean() {
    let m = Model::new();
    let (ids, mask, b, s) = batch();
    let want = m.run(Isa::detect(), 3, &ids, &mask, b, s);
    let h = m.desc.hidden as usize;
    let mut ranges: Vec<_> = (0..b).rev().map(|r| range(r, 0, s)).collect();
    ranges.push(range(4, 0, s));
    let got = ranged(&m, 3, (&ids, &mask, b, s), &ranges, TURBO_NORMALIZE_L2, h).out;
    for (i, r) in ranges.iter().enumerate() {
        let row = r.row as usize;
        assert!(bits(&got[i * h..][..h]) == bits(&want[row * h..][..h]), "range {i} over row {row}");
    }
}

/// Each range's vector is the mean of the hidden states of its live
/// tokens, summed in F32 in column order and scaled by one over the
/// count, cut to output_dim: single tokens, ranges that start or end
/// on masked tokens, and ranges that run into padding.
#[test]
fn a_range_s_vector_is_the_mean_of_its_live_tokens() {
    let m = Model::new();
    let (ids, mask, b, s) = batch();
    let h = m.desc.hidden as usize;
    let ranges = [
        range(0, 0, 1),
        range(0, 99, 100),
        range(0, 17, 18),
        range(1, 0, 100),
        range(2, 2, 12),
        range(2, 3, 5),
        range(2, 11, 13),
        range(2, 30, 100),
        range(3, 63, 64),
        range(4, 31, 35),
        range(8, 16, 60),
    ];
    for od in [h, 64, 1] {
        let run = ranged(&m, 2, (&ids, &mask, b, s), &ranges, TURBO_NORMALIZE_NONE, od);
        for (i, r) in ranges.iter().enumerate() {
            let (row, start, end) = (r.row as usize, r.start as usize, r.end as usize);
            let (first, len) = run.rows[row];
            let mut want = vec![0f32; od];
            let mut n = 0u32;
            for p in start..end.min(len) {
                if mask[row * s + p] != 0 {
                    for (w, &x) in want.iter_mut().zip(&run.x[(first + p) * h..][..od]) {
                        *w += x;
                    }
                    n += 1;
                }
            }
            let inv = 1.0 / n as f32;
            want.iter_mut().for_each(|w| *w *= inv);
            assert!(bits(&run.out[i * od..][..od]) == bits(&want), "range {i}, {od} values");
        }
        // Masked tokens and padding add nothing: row 2's columns 3,
        // 10 and 11 are masked, and its live tokens end at 37.
        let same = [
            (range(2, 3, 5), range(2, 4, 5)),
            (range(2, 10, 13), range(2, 12, 13)),
            (range(2, 30, 100), range(2, 30, 37)),
            (range(1, 0, 100), range(1, 0, 1)),
        ];
        let pairs: Vec<turbo_token_range> = same.iter().flat_map(|&(a, b)| [a, b]).collect();
        let run = ranged(&m, 2, (&ids, &mask, b, s), &pairs, TURBO_NORMALIZE_NONE, od);
        for (k, (x, y)) in same.iter().enumerate() {
            let (a, b) = (&run.out[2 * k * od..][..od], &run.out[(2 * k + 1) * od..][..od]);
            assert!(bits(a) == bits(b), "{x:?} and {y:?}");
        }
    }
}

/// Ranges give the same bits on any number of threads, with more
/// ranges than threads so that each task takes several.
#[test]
fn ranges_are_the_same_bits_on_any_number_of_threads() {
    let m = Model::new();
    let (ids, mask, b, s) = batch();
    let lens = [100, 1, 37, 64, 65, 2, 33, 90, 17];
    let mut ranges = Vec::new();
    for (r, &len) in lens.iter().enumerate() {
        for start in (0..len).step_by(3) {
            ranges.push(range(r, start, (start + 1 + start % 7).min(s)));
        }
    }
    let h = m.desc.hidden as usize;
    let want = ranged(&m, 1, (&ids, &mask, b, s), &ranges, TURBO_NORMALIZE_L2, h).out;
    assert!(want.iter().all(|v| v.is_finite()));
    let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    for threads in [3, most] {
        let got = ranged(&m, threads, (&ids, &mask, b, s), &ranges, TURBO_NORMALIZE_L2, h).out;
        assert!(bits(&got) == bits(&want), "{threads} threads differ from 1");
    }
}

/// A write of rows after a write of ranges pools rows again.
#[test]
fn a_write_of_rows_after_ranges_pools_rows() {
    let m = Model::new();
    let (ids, mask, b, s) = batch();
    let h = m.desc.hidden as usize;
    let isa = Isa::detect();
    let want = m.run(isa, 2, &ids, &mask, b, s);
    let packed = m.packed(isa);
    let mut pool = Pool::new(2);
    let size = Size { threads: 2, max_batch: b, max_seq: s, max_ranges: 3 };
    let mut e = Encoder::new(&m.desc, m.slices(), &packed, size).unwrap();
    let rows = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch: b as u32,
        seq: s as u32,
        row_stride: s as u32,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: std::ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize: TURBO_NORMALIZE_L2,
        output_dim: h as u32,
        reserved: 0,
    };
    let mut out = vec![f32::NAN; b * h];
    e.write_ranges(&rows, &ids, &mask, None, &[range(0, 0, 5), range(1, 0, 1), range(2, 0, 3)]);
    e.run(&mut pool, &mut out);
    assert!(out[3 * h..].iter().all(|v| v.is_nan()), "three ranges write three vectors");
    e.write(&rows, &ids, &mask, None);
    e.run(&mut pool, &mut out);
    assert!(bits(&out) == bits(&want));
}

/// How the run's time goes with threads and kernels, on a batch of
/// 32 rows of up to 256 tokens, 1353 live, at all-MiniLM-L6-v2's shape
/// and the small bundle's.
#[test]
#[ignore = "measures; run with --ignored --nocapture"]
fn speed_by_threads_and_kernel() {
    let (batch, seq) = (32, 256);
    let mut lens: Vec<usize> = (0..batch).map(|r| if r == 0 { seq } else { 6 + (r * 29) % 53 }).collect();
    lens[batch - 1] += 1353 - lens.iter().sum::<usize>();
    let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
    for (r, &n) in lens.iter().enumerate() {
        for p in 0..n {
            ids[r * seq + p] = ((r * 131 + p * 17) % 500) as i32;
            mask[r * seq + p] = 1;
        }
    }
    let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    for (name, m) in
        [("MiniLM shape", Model::shaped(6, 384, 12, 1536, 512)), ("small", Model::shaped(2, 32, 4, 64, 512))]
    {
        for isa in isas() {
            for threads in [1, 2, most] {
                let (_, t) = m.timed(isa, threads, &ids, &mask, batch, seq, 8);
                println!("{name}: {isa:?}, {threads} threads: {:.2} ms", t.as_secs_f64() * 1e3);
            }
        }
    }
}
