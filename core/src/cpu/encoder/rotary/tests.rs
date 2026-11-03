//! A rotary encoder's vectors are the same bits on any number of threads,
//! its local layers' windows included, and agree across instruction sets.
//! Their agreement with upstream is the conformance test's, on the sealed
//! tiny bundles (docs/conformance.md).

use super::super::{Encoder, Size};
use crate::backend::{
    TURBO_ACTIVATION_GELU_ERF, TURBO_MLP_GATED, TURBO_NORM_LAYER, TURBO_NORM_LAYER_NO_BIAS, turbo_backend_embed_rows,
};
use crate::cpu::kernels::{Isa, Packed, isas, values};
use crate::cpu::pool::Pool;
use crate::{TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

use super::*;

const VOCAB: usize = 300;
const POSITIONS: usize = 128;
const LAYERS: usize = 4;
const H: usize = 64;
const HEADS: usize = 2;
const I: usize = 48;

pub(in crate::cpu::encoder) struct Model {
    pub(in crate::cpu::encoder) desc: turbo_backend_model,
    tensors: Vec<Vec<f32>>,
    _table: Vec<f32>,
}

impl Model {
    /// A ModernBERT (global layers 0 and 3, a window of 16) or a nomic-bert.
    fn new(modern: bool) -> Model {
        Model::layered(modern, LAYERS)
    }

    /// As `new` with `layers` layers: the first layers of a deeper one,
    /// the same tensors in the same order.
    pub(in crate::cpu::encoder) fn layered(modern: bool, layers: usize) -> Model {
        let (h, i) = (H, I);
        let mut n = 0;
        let mut t = |len: usize, norm: bool| {
            n += 1;
            let v = values(len, n, if norm { 0.2 } else { 0.15 });
            if norm { v.iter().map(|x| 1.0 + x).collect() } else { v }
        };
        let mut tensors = Vec::new();
        if modern {
            tensors.extend([t(VOCAB * h, false), t(h, true), t(h, true)]);
            for l in 0..layers {
                tensors.push(if l == 0 { Vec::new() } else { t(h, true) });
                tensors.extend([
                    t(3 * h * h, false),
                    t(h * h, false),
                    t(h, true),
                    t(2 * i * h, false),
                    t(h * i, false),
                ]);
            }
        } else {
            tensors.extend([t(VOCAB * h, false), t(2 * h, false), t(h, true), t(h, false)]);
            for _ in 0..layers {
                tensors.extend([t(3 * h * h, false), t(h * h, false), t(h, true), t(h, false)]);
                tensors.extend([t(i * h, false), t(i * h, false), t(h * i, false), t(h, true), t(h, false)]);
            }
        }
        let pairs = h / HEADS / 2;
        let table = crate::rope::table(160000.0, 10000.0, POSITIONS, pairs).unwrap();
        // SAFETY: every field of the description is a number or a pointer,
        // for which zero is a value.
        let mut desc: turbo_backend_model = unsafe { std::mem::zeroed() };
        desc.struct_size = size_of::<turbo_backend_model>() as u32;
        desc.family = if modern { TURBO_FAMILY_MODERNBERT } else { TURBO_FAMILY_NOMIC_BERT };
        desc.dtype = TURBO_DTYPE_F32;
        desc.layers = layers as u32;
        desc.hidden = h as u32;
        desc.heads = HEADS as u32;
        desc.intermediate = i as u32;
        desc.vocab_size = VOCAB as u32;
        desc.max_positions = POSITIONS as u32;
        desc.token_types = if modern { 0 } else { 2 };
        desc.layer_norm_eps = 1e-5;
        desc.tensor_count = tensors.len() as u32;
        desc.rope_table = table.as_ptr();
        desc.rope_pairs = pairs as u32;
        desc.mlp = TURBO_MLP_GATED;
        (desc.activation, desc.norm) = if modern {
            (TURBO_ACTIVATION_GELU_ERF, TURBO_NORM_LAYER_NO_BIAS)
        } else {
            (TURBO_ACTIVATION_SILU, TURBO_NORM_LAYER)
        };
        if modern {
            (desc.attention_window, desc.global_every) = (16, 3);
        }
        assert!(crate::cpu::rotary::check(&turbo_backend_model { tensors: [].as_ptr(), ..desc }).is_ok());
        Model { desc, tensors, _table: table }
    }

    pub(in crate::cpu::encoder) fn slices(&self) -> Vec<&[f32]> {
        self.tensors.iter().map(Vec::as_slice).collect()
    }

    /// The vectors of rows of `lens` live tokens (the rest of `seq`
    /// masked), on `threads` threads with `isa`; run twice, which must
    /// agree.
    fn run(&self, isa: Isa, threads: usize, lens: &[usize], seq: usize) -> Vec<f32> {
        let packed = Packed::rotary(&self.desc, &self.slices(), isa).unwrap();
        let mut pool = Pool::new(threads);
        let batch = lens.len();
        let size = Size { threads, max_batch: batch, max_seq: seq, max_ranges: batch };
        let mut e = Encoder::new(&self.desc, self.slices(), &packed, size).unwrap();
        // A row's ids and types from its length and columns alone.
        let ids: Vec<i32> = (0..batch * seq).map(|k| ((k % seq * 37 + lens[k / seq] * 11) % VOCAB) as i32).collect();
        let mask: Vec<i32> = (0..batch * seq).map(|k| i32::from(k % seq < lens[k / seq])).collect();
        let types: Vec<i32> = (0..batch * seq).map(|k| (k % seq).is_multiple_of(5) as i32).collect();
        let rows = turbo_backend_embed_rows {
            struct_size: size_of::<turbo_backend_embed_rows>() as u32,
            batch: batch as u32,
            seq: seq as u32,
            row_stride: seq as u32,
            ids: ids.as_ptr(),
            mask: mask.as_ptr(),
            types: types.as_ptr(),
            pooling: TURBO_POOLING_MEAN,
            normalize: TURBO_NORMALIZE_NONE,
            output_dim: H as u32,
            head: 0,
            taps: 0,
            reserved: 0,
        };
        let mut first = Vec::new();
        for _ in 0..2 {
            let mut out = vec![f32::NAN; batch * H];
            e.write(&rows, &ids, &mask, Some(&types));
            e.run(&mut pool, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "{isa:?} {threads} threads");
            if first.is_empty() {
                first = out;
            } else {
                assert_eq!(bits(&first), bits(&out), "a second run on the session");
            }
        }
        first
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn rotary_vectors_are_the_same_bits_on_any_number_of_threads() {
    let lens = [100, 61, 17, 1];
    for modern in [true, false] {
        let m = Model::new(modern);
        let mut by_isa: Vec<Vec<f32>> = Vec::new();
        for isa in isas() {
            let one = m.run(isa, 1, &lens, 100);
            for threads in [2, 4, 8] {
                assert_eq!(bits(&m.run(isa, threads, &lens, 100)), bits(&one), "{isa:?} {threads} threads");
            }
            by_isa.push(one);
        }
        for v in &by_isa[1..] {
            let worst = v.iter().zip(&by_isa[0]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "modern {modern}: the instruction sets differ by {worst}");
        }
    }
}

/// A row's vector does not depend on the rows beside it, nor on the
/// padding past its last live token.
#[test]
fn a_rotary_row_is_its_own() {
    for modern in [true, false] {
        let m = Model::new(modern);
        let isa = Isa::detect();
        let batch = m.run(isa, 3, &[100, 61, 17], 100);
        for (r, len) in [(1, 61), (2, 17)] {
            let alone = m.run(isa, 3, &[len], len);
            assert_eq!(bits(&alone), bits(&batch[r * H..][..H]), "modern {modern} row {r}");
        }
    }
}
