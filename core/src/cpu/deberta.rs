//! A DeBERTa-v2 model's sessions (TURBO_FAMILY_DEBERTA_V2): the rows run
//! one at a time through turbo::deberta's encoder, the reference every
//! device's kernels are held to, on the session's threads, then pooled as
//! a BERT session pools. Each row's hidden states are computed from its
//! tokens through its last live one, which gives the bits a longer pad
//! would (deberta/attention.rs), so a row's vectors do not depend on what
//! else is in the batch. As on the BERT path, a run allocates nothing: the
//! session holds the workspace and the states for its largest batch and
//! sequence (tests/allocations.rs).

use std::sync::Arc;

use crate::backend::{TURBO_DEBERTA_EMBEDDING_TENSORS, turbo_backend_embed_rows, turbo_backend_model};
use crate::deberta::{Config, Encoder, Workspace};
use crate::session::turbo_token_range;
use crate::{TURBO_NORMALIZE_L2, TURBO_POOLING_CLS, TURBO_POOLING_LAST};

use super::pool::Pool;

/// The encoder's configuration from the backend's description: the
/// architecture, with max_positions as the longest sequence.
pub(super) fn config(d: &turbo_backend_model) -> Config {
    Config {
        hidden: d.hidden as usize,
        intermediate: d.intermediate as usize,
        heads: d.heads as usize,
        layers: d.layers as usize,
        vocabulary: d.vocab_size as usize,
        max_sequence: d.max_positions as usize,
        position_buckets: d.position_buckets as usize,
        max_relative_positions: d.max_relative_positions as usize,
        layer_norm_eps: d.layer_norm_eps,
    }
}

/// The tensors a DeBERTa-v2 model of `layers` layers hands model_load.
pub(super) fn tensor_count(layers: u32) -> u64 {
    TURBO_DEBERTA_EMBEDDING_TENSORS as u64 + layers as u64 * crate::backend::TURBO_BERT_LAYER_TENSORS as u64
}

pub(super) struct Rows {
    encoder: Arc<Encoder>,
    hidden: usize,
    ids: Vec<u32>,
    mask: Vec<bool>,
    /// Each row's tokens through its last live one.
    lens: Vec<usize>,
    seq: usize,
    ranges: Vec<turbo_token_range>,
    pooling: u32,
    normalize: u32,
    output_dim: usize,
    /// Each row's last hidden states from the last run, row r's
    /// [lens[r], hidden] at r * seq * hidden: room for the session's
    /// largest batch and sequence.
    states: Vec<f32>,
    /// The encoder's workspace, for the session's largest sequence.
    run: Workspace,
}

impl Rows {
    /// The rows of a session of `max_batch` rows of `max_seq` tokens on
    /// `pool`, everything a run computes in allocated here.
    pub(super) fn new(
        encoder: Arc<Encoder>,
        hidden: usize,
        max_batch: usize,
        max_seq: usize,
        max_ranges: usize,
        pool: &Pool,
    ) -> Result<Rows, String> {
        let run = encoder.workspace(pool, max_seq).map_err(|e| e.message)?;
        let tokens = max_batch * max_seq;
        let mut states = Vec::new();
        states
            .try_reserve_exact(tokens * hidden)
            .map_err(|_| format!("{} values for the session's states", tokens * hidden))?;
        states.resize(tokens * hidden, 0.0);
        let (mut ids, mut mask, mut lens, mut ranges) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let short = |_| "the session's rows".to_string();
        ids.try_reserve_exact(tokens).map_err(short)?;
        mask.try_reserve_exact(tokens).map_err(short)?;
        lens.try_reserve_exact(max_batch).map_err(short)?;
        ranges.try_reserve_exact(max_ranges).map_err(short)?;
        Ok(Rows {
            encoder,
            hidden,
            ids,
            mask,
            lens,
            seq: 0,
            ranges,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
            states,
            run,
        })
    }

    /// Row r's last hidden states from the last run.
    fn state(&self, r: usize) -> &[f32] {
        &self.states[r * self.seq * self.hidden..]
    }

    pub(super) fn normalize(&self) -> u32 {
        self.normalize
    }

    /// Copy in the rows the core checked; a row's token types are 0, the
    /// only type the family has.
    pub(super) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32]) {
        let (b, s, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
        self.ids.clear();
        self.mask.clear();
        self.lens.clear();
        for row in 0..b {
            let at = row * stride;
            self.ids.extend(ids[at..at + s].iter().map(|&i| i as u32));
            self.mask.extend(mask[at..at + s].iter().map(|&m| m != 0));
            let live = mask[at..at + s].iter().rposition(|&m| m != 0).map_or(0, |p| p + 1);
            self.lens.push(live);
        }
        self.seq = s;
        self.ranges.clear();
        self.pooling = r.pooling;
        self.normalize = r.normalize;
        self.output_dim = r.output_dim as usize;
    }

    /// As `write`, and the ranges the run pools instead of the rows.
    pub(super) fn write_ranges(
        &mut self,
        r: &turbo_backend_embed_rows,
        ids: &[i32],
        mask: &[i32],
        ranges: &[turbo_token_range],
    ) {
        self.write(r, ids, mask);
        self.ranges.extend_from_slice(ranges);
    }

    /// Run the written rows into `out`: a vector per row, or per range
    /// after `write_ranges`, output_dim values each. False when the
    /// encoder refused a row, which the core's checks leave no input for.
    pub(super) fn run(&mut self, pool: &mut Pool, out: &mut [f32]) -> bool {
        let (seq, h) = (self.seq, self.hidden);
        for (r, &len) in self.lens.iter().enumerate() {
            let at = r * seq;
            let out = &mut self.states[at * h..][..len * h];
            let (ids, mask) = (&self.ids[at..at + len], &self.mask[at..at + len]);
            if self.encoder.encode_into(pool, &mut self.run, ids, mask, out).is_err() {
                return false;
            }
        }
        let od = self.output_dim;
        if self.ranges.is_empty() {
            for r in 0..self.lens.len() {
                let dst = &mut out[r * od..][..od];
                let len = self.lens[r];
                match self.pooling {
                    TURBO_POOLING_CLS => dst.copy_from_slice(&self.state(r)[..od]),
                    TURBO_POOLING_LAST => dst.copy_from_slice(&self.state(r)[(len - 1) * self.hidden..][..od]),
                    _ => self.mean(r, 0..len, dst),
                }
                self.finish(dst);
            }
        } else {
            for (i, range) in self.ranges.iter().enumerate() {
                let row = range.row as usize;
                let cols = range.start as usize..(range.end as usize).min(self.lens[row]);
                let dst = &mut out[i * od..][..od];
                self.mean(row, cols, dst);
                self.finish(dst);
            }
        }
        true
    }

    /// The mean of row `r`'s live tokens in `cols`, as the BERT path takes
    /// it: summed in F32 in column order, then scaled by one over the count.
    fn mean(&self, r: usize, cols: std::ops::Range<usize>, dst: &mut [f32]) {
        let mask = &self.mask[r * self.seq..];
        dst.fill(0.0);
        let mut n = 0u32;
        for p in cols.filter(|&p| mask[p]) {
            for (d, &x) in dst.iter_mut().zip(&self.state(r)[p * self.hidden..]) {
                *d += x;
            }
            n += 1;
        }
        let inv = 1.0 / n.max(1) as f32;
        for d in dst.iter_mut() {
            *d *= inv;
        }
    }

    /// Normalize a pooled vector as the write asked, as the BERT path does.
    fn finish(&self, dst: &mut [f32]) {
        if self.normalize == TURBO_NORMALIZE_L2 {
            let norm = dst.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt().max(1e-12);
            let inv = (1.0 / norm) as f32;
            for d in dst.iter_mut() {
                *d *= inv;
            }
        }
    }
}
