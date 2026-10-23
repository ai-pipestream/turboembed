//! One encoder layer over a run's visible positions. Masked positions
//! never reach a visible row: visible queries attend to visible keys only,
//! and relative buckets depend on position differences alone, so the
//! visible rows are computed from the visible rows and their original
//! positions, bit for bit the same whatever padding surrounds them. Every
//! masked position has the same hidden state (zero after the embedding,
//! then a uniform attention row over all positions, as upstream's finite
//! mask minimum gives), so one shared row stands for all of them.

use super::{
    Config,
    math::bucket,
    weights::{Layer, zeros},
};
use crate::{
    cpu::{Matrix, Mode, softmax},
    status::{Result, invalid},
};

/// Queries whose scores are held at once.
const QUERY_BLOCK: usize = 64;

/// One run's workspace, allocated once and reused by every layer and head.
pub(super) struct Run {
    /// The visible positions, ascending.
    pub at: Vec<usize>,
    /// The sequence length, masked positions included.
    pub n: usize,
    /// Whether any position is masked: their shared row is row at.len().
    pub pad: bool,
    /// Largest distance between visible positions.
    span: usize,
    /// The relative rows reached: r0..r0 + rr.
    r0: usize,
    rr: usize,
    /// rt[i - j + span] is the bucket of distance i - j, less r0.
    rt: Vec<u32>,
    /// [rows, hidden] hidden states, attention residual, and context;
    /// [rows, 3 * hidden] projections; [rows, intermediate].
    pub x: Vec<f32>,
    a: Vec<f32>,
    ctx: Vec<f32>,
    qkv: Vec<f32>,
    mid: Vec<f32>,
    /// [rr, 3 * hidden] relative projections when not cached at load.
    pos: Vec<f32>,
    /// Scores of a block of queries, [QUERY_BLOCK, visible]; c2p of the
    /// block, [QUERY_BLOCK, rr]; p2c of every key, [visible, rr].
    s: Vec<f32>,
    c2p: Vec<f32>,
    p2c: Vec<f32>,
    /// One head's keys, values (transposed), relative keys and queries.
    mk: Matrix,
    mv: Matrix,
    mpk: Matrix,
    mpq: Matrix,
}

impl Run {
    /// `mask` has at least one visible position.
    pub fn new(c: &Config, mask: &[bool], cached: bool) -> Result<Run> {
        let at: Vec<usize> = mask.iter().enumerate().filter_map(|(i, &m)| m.then_some(i)).collect();
        let nv = at.len();
        let pad = nv < mask.len();
        let rows = nv + usize::from(pad);
        let span = at[nv - 1] - at[0];
        let b = |d: isize| bucket(d, c.position_buckets, c.max_relative_positions);
        let r0 = b(-(span as isize));
        let rr = b(span as isize) + 1 - r0;
        let rt = (0..=2 * span).map(|i| (b(i as isize - span as isize) - r0) as u32).collect();
        let (h, qb) = (c.hidden, nv.min(QUERY_BLOCK));
        Ok(Run {
            at,
            n: mask.len(),
            pad,
            span,
            r0,
            rr,
            rt,
            x: zeros(rows * h)?,
            a: zeros(rows * h)?,
            ctx: zeros(rows * h)?,
            qkv: zeros(rows * 3 * h)?,
            mid: zeros(rows * c.intermediate)?,
            pos: if cached { Vec::new() } else { zeros(rr * 3 * h)? },
            s: zeros(qb * nv)?,
            c2p: zeros(qb * rr)?,
            p2c: zeros(nv * rr)?,
            mk: Matrix::empty(),
            mv: Matrix::empty(),
            mpk: Matrix::empty(),
            mpq: Matrix::empty(),
        })
    }

    pub fn rows(&self) -> usize {
        self.at.len() + usize::from(self.pad)
    }

    /// Attention, residual and LayerNorm, feed-forward, residual and
    /// LayerNorm, over the visible rows and the shared masked row.
    pub fn layer(&mut self, c: &Config, l: &Layer, relative: &[f32]) -> Result<()> {
        let (h, rows) = (c.hidden, self.rows());
        self.attention(c, l, relative)?;
        self.a.copy_from_slice(&self.x);
        l.attention.w.apply(&self.ctx, h, rows, Some(&l.attention.b), &mut self.a, h, Mode::Residual);
        l.attention_norm.apply(&mut self.a, c.layer_norm_eps)?;
        let f = c.intermediate;
        l.intermediate.w.apply(&self.a, h, rows, Some(&l.intermediate.b), &mut self.mid, f, Mode::Gelu);
        self.x.copy_from_slice(&self.a);
        l.output.w.apply(&self.mid, f, rows, Some(&l.output.b), &mut self.x, h, Mode::Residual);
        l.output_norm.apply(&mut self.x, c.layer_norm_eps)
    }

    /// Disentangled attention (c2c + c2p + p2c) / sqrt(3 d) of each head,
    /// into ctx. c2p and p2c are products against only the relative rows
    /// the run's distances reach, gathered by bucket.
    fn attention(&mut self, c: &Config, l: &Layer, relative: &[f32]) -> Result<()> {
        let (h, d, w3) = (c.hidden, c.hidden / c.heads, 3 * c.hidden);
        let (nv, rows, rr, span) = (self.at.len(), self.rows(), self.rr, self.span);
        l.qkv.w.apply(&self.x, h, rows, Some(&l.qkv.b), &mut self.qkv, w3, Mode::Plain);
        let (pos, ld): (&[f32], usize) = match &l.pos {
            Some(p) => (&p[self.r0 * 2 * h..], 2 * h),
            None => {
                l.qkv.w.apply(&relative[self.r0 * h..], h, rr, Some(&l.qkv.b), &mut self.pos, w3, Mode::Plain);
                (&self.pos, w3)
            }
        };
        let oom = |_| invalid("deberta attention workspace allocation failed");
        let inv = (1.0 / (3.0 * d as f64).sqrt()) as f32;
        for head in 0..c.heads {
            let o = head * d;
            self.mk.pack_rows(&self.qkv[h + o..], w3, nv, d).map_err(oom)?;
            self.mv.pack_cols(&self.qkv[2 * h + o..], w3, d, nv).map_err(oom)?;
            self.mpk.pack_rows(&pos[h + o..], ld, rr, d).map_err(oom)?;
            self.mpq.pack_rows(&pos[o..], ld, rr, d).map_err(oom)?;
            // p2c[j][r] = k_j . pos_query_r
            self.mpq.apply(&self.qkv[h + o..], w3, nv, None, &mut self.p2c, rr, Mode::Plain);
            let mut q0 = 0;
            while q0 < nv {
                let b = (nv - q0).min(QUERY_BLOCK);
                let q = &self.qkv[q0 * w3 + o..];
                self.mk.apply(q, w3, b, None, &mut self.s, nv, Mode::Plain);
                self.mpk.apply(q, w3, b, None, &mut self.c2p, rr, Mode::Plain);
                for i in 0..b {
                    let base = self.at[q0 + i] + span;
                    let row = &mut self.s[i * nv..][..nv];
                    let c2p = &self.c2p[i * rr..][..rr];
                    for ((sc, &pj), p2c) in row.iter_mut().zip(&self.at).zip(self.p2c.chunks_exact(rr)) {
                        let k = self.rt[base - pj] as usize;
                        *sc = (*sc + c2p[k] + p2c[k]) * inv;
                    }
                    softmax(row);
                }
                self.mv.apply(&self.s, nv, b, None, &mut self.ctx[q0 * h + o..], h, Mode::Plain);
                q0 += b;
            }
        }
        if self.pad {
            // The masked row's uniform attention over all n positions:
            // the visible values, and n - visible copies of its own.
            let (masked, n) = ((self.n - nv) as f64, self.n as f64);
            for col in 0..h {
                let v = |j: usize| f64::from(self.qkv[j * w3 + 2 * h + col]);
                let sum: f64 = (0..nv).map(v).sum::<f64>() + masked * v(nv);
                self.ctx[nv * h + col] = (sum / n) as f32;
            }
        }
        Ok(())
    }
}
