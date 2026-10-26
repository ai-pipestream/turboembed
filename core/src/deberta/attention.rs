//! One encoder layer over a run's visible positions. Masked positions
//! never reach a visible row: visible queries attend to visible keys only,
//! and relative buckets depend on position differences alone, so the
//! visible rows are computed from the visible rows and their original
//! positions, bit for bit the same whatever padding surrounds them. Every
//! masked position has the same hidden state (zero after the embedding,
//! then a uniform attention row over all positions, as upstream's finite
//! mask minimum gives), so one shared row stands for all of them.

use std::sync::atomic::{AtomicBool, Ordering};

use super::{
    Config,
    math::bucket,
    weights::{Layer, zeros},
};
use crate::{
    cpu::{Matrix, Mode, pool::Pool, softmax},
    status::{Result, invalid},
};

/// Queries whose scores are held at once.
const QUERY_BLOCK: usize = 64;

/// One run's workspace, allocated once and reused by every layer and head,
/// and by later runs of no more positions (`reuse`).
pub(crate) struct Run {
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
    /// One head's workspace for each thread that computes heads.
    heads: Vec<Head>,
}

/// What one head's attention computes in.
struct Head {
    /// Scores of a block of queries, [QUERY_BLOCK, visible]; c2p of the
    /// block, [QUERY_BLOCK, rr]; p2c of every key, [visible, rr].
    s: Vec<f32>,
    c2p: Vec<f32>,
    p2c: Vec<f32>,
    /// The head's keys, values (transposed), relative keys and queries.
    mk: Matrix,
    mv: Matrix,
    mpk: Matrix,
    mpq: Matrix,
}

/// What every head reads, and where each writes its columns of ctx.
struct Heads<'a> {
    run: &'a Run,
    pos: &'a [f32],
    ld: usize,
    d: usize,
    ctx: *mut f32,
    work: *mut Head,
    failed: AtomicBool,
}

// SAFETY: ctx is written only in the columns of the head a task computes,
// and work only at the index of the thread that runs the task, each of
// which the pool gives to one thread at a time (Pool::run).
unsafe impl Sync for Heads<'_> {}

impl Run {
    /// `mask` has at least one visible position. Heads run on up to
    /// `threads` threads.
    pub(super) fn new(c: &Config, mask: &[bool], cached: bool, threads: usize) -> Result<Run> {
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
        let mut heads = Vec::with_capacity(threads.max(1));
        for _ in 0..threads.max(1) {
            heads.push(Head {
                s: zeros(qb * nv)?,
                c2p: zeros(qb * rr)?,
                p2c: zeros(nv * rr)?,
                mk: Matrix::empty(),
                mv: Matrix::empty(),
                mpk: Matrix::empty(),
                mpq: Matrix::empty(),
            });
        }
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
            heads,
        })
    }

    /// The workspace for a run of `mask`, in this one's memory: no
    /// allocation when the run has no more positions than the mask this
    /// was made for, and its pool no more threads. Every buffer starts at
    /// zero, as `new` gives it.
    pub(super) fn reuse(&mut self, c: &Config, mask: &[bool]) {
        self.at.clear();
        self.at.extend(mask.iter().enumerate().filter_map(|(i, &m)| m.then_some(i)));
        let nv = self.at.len();
        self.n = mask.len();
        self.pad = nv < mask.len();
        let rows = nv + usize::from(self.pad);
        self.span = self.at[nv - 1] - self.at[0];
        let span = self.span;
        let b = |d: isize| bucket(d, c.position_buckets, c.max_relative_positions);
        self.r0 = b(-(span as isize));
        self.rr = b(span as isize) + 1 - self.r0;
        let r0 = self.r0;
        self.rt.clear();
        self.rt.extend((0..=2 * span).map(|i| (b(i as isize - span as isize) - r0) as u32));
        let (h, qb, rr) = (c.hidden, nv.min(QUERY_BLOCK), self.rr);
        let zero = |v: &mut Vec<f32>, n: usize| {
            v.clear();
            v.resize(n, 0.0);
        };
        zero(&mut self.x, rows * h);
        zero(&mut self.a, rows * h);
        zero(&mut self.ctx, rows * h);
        zero(&mut self.qkv, rows * 3 * h);
        zero(&mut self.mid, rows * c.intermediate);
        // Empty when the layers hold their relative projections.
        if self.pos.capacity() > 0 {
            zero(&mut self.pos, rr * 3 * h);
        }
        for w in &mut self.heads {
            zero(&mut w.s, qb * nv);
            zero(&mut w.c2p, qb * rr);
            zero(&mut w.p2c, nv * rr);
        }
    }

    /// Grow each head's packed matrices to this run's sizes now, so that
    /// a later run of no more positions packs them without allocating.
    pub(super) fn presize(&mut self, c: &Config) -> Result<()> {
        let (h, d, nv, rr) = (c.hidden, c.hidden / c.heads, self.at.len(), self.rr);
        let w3 = 3 * h;
        let rel = zeros(rr * w3)?;
        let full = |_| invalid("deberta attention workspace allocation failed");
        for w in &mut self.heads {
            w.mk.pack_rows(&self.qkv, w3, nv, d).map_err(full)?;
            w.mv.pack_cols(&self.qkv, w3, d, nv).map_err(full)?;
            w.mpk.pack_rows(&rel, w3, rr, d).map_err(full)?;
            w.mpq.pack_rows(&rel, w3, rr, d).map_err(full)?;
        }
        Ok(())
    }

    pub(super) fn rows(&self) -> usize {
        self.at.len() + usize::from(self.pad)
    }

    /// Attention, residual and LayerNorm, feed-forward, residual and
    /// LayerNorm, over the visible rows and the shared masked row; the
    /// products and the heads across `pool`'s threads when there is one.
    pub(super) fn layer(&mut self, c: &Config, l: &Layer, relative: &[f32], mut pool: Option<&mut Pool>) -> Result<()> {
        let (h, rows) = (c.hidden, self.rows());
        self.attention(c, l, relative, pool.as_deref_mut())?;
        self.a.copy_from_slice(&self.x);
        let ab = Some(l.attention.b.as_slice());
        l.attention.w.apply_on(pool.as_deref_mut(), &self.ctx, h, rows, ab, &mut self.a, h, Mode::Residual);
        l.attention_norm.apply(&mut self.a, c.layer_norm_eps)?;
        let f = c.intermediate;
        let ib = Some(l.intermediate.b.as_slice());
        l.intermediate.w.apply_on(pool.as_deref_mut(), &self.a, h, rows, ib, &mut self.mid, f, Mode::Gelu);
        self.x.copy_from_slice(&self.a);
        let ob = Some(l.output.b.as_slice());
        l.output.w.apply_on(pool, &self.mid, f, rows, ob, &mut self.x, h, Mode::Residual);
        l.output_norm.apply(&mut self.x, c.layer_norm_eps)
    }

    /// Disentangled attention (c2c + c2p + p2c) / sqrt(3 d) of each head,
    /// into ctx. c2p and p2c are products against only the relative rows
    /// the run's distances reach, gathered by bucket. Each head is one
    /// task, so its columns of ctx are the same on any thread.
    fn attention(&mut self, c: &Config, l: &Layer, relative: &[f32], mut pool: Option<&mut Pool>) -> Result<()> {
        let (h, d, w3) = (c.hidden, c.hidden / c.heads, 3 * c.hidden);
        let (nv, rows, rr) = (self.at.len(), self.rows(), self.rr);
        let qb = Some(l.qkv.b.as_slice());
        l.qkv.w.apply_on(pool.as_deref_mut(), &self.x, h, rows, qb, &mut self.qkv, w3, Mode::Plain);
        let cached = l.pos.is_some();
        if !cached {
            let rel = &relative[self.r0 * h..];
            l.qkv.w.apply_on(pool.as_deref_mut(), rel, h, rr, qb, &mut self.pos, w3, Mode::Plain);
        }
        let mut ctx = std::mem::take(&mut self.ctx);
        let mut work = std::mem::take(&mut self.heads);
        let threads = work.len();
        let (pos, ld): (&[f32], usize) = match &l.pos {
            Some(p) => (&p[self.r0 * 2 * h..], 2 * h),
            None => (&self.pos, w3),
        };
        let shared = Heads {
            run: self,
            pos,
            ld,
            d,
            ctx: ctx.as_mut_ptr(),
            work: work.as_mut_ptr(),
            failed: AtomicBool::new(false),
        };
        match pool.filter(|p| p.threads() > 1 && p.threads() <= threads) {
            // SAFETY: as Heads; a thread index is under the pool's
            // threads, which work holds as many of.
            Some(p) => p.run(c.heads, &|t, th| unsafe { shared.head(c, t, th) }),
            // SAFETY: one thread, workspace 0.
            None => (0..c.heads).for_each(|t| unsafe { shared.head(c, t, 0) }),
        }
        let failed = shared.failed.load(Ordering::Relaxed);
        self.ctx = ctx;
        self.heads = work;
        if failed {
            return Err(invalid("deberta attention workspace allocation failed"));
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

impl Heads<'_> {
    /// Head `head` into its columns of ctx, in the workspace of thread
    /// `thread`.
    ///
    /// # Safety
    /// No other thread uses workspace `thread`, or computes head `head`,
    /// while this runs; `thread` is under the workspaces' count.
    unsafe fn head(&self, c: &Config, head: usize, thread: usize) {
        let r = self.run;
        let (h, w3, d) = (c.hidden, 3 * c.hidden, self.d);
        let (nv, rr, span) = (r.at.len(), r.rr, r.span);
        // SAFETY: the caller's.
        let w = unsafe { &mut *self.work.add(thread) };
        let o = head * d;
        let packed = w.mk.pack_rows(&r.qkv[h + o..], w3, nv, d).is_ok()
            && w.mv.pack_cols(&r.qkv[2 * h + o..], w3, d, nv).is_ok()
            && w.mpk.pack_rows(&self.pos[h + o..], self.ld, rr, d).is_ok()
            && w.mpq.pack_rows(&self.pos[o..], self.ld, rr, d).is_ok();
        if !packed {
            self.failed.store(true, Ordering::Relaxed);
            return;
        }
        let inv = (1.0 / (3.0 * d as f64).sqrt()) as f32;
        // p2c[j][r] = k_j . pos_query_r
        w.mpq.apply(&r.qkv[h + o..], w3, nv, None, &mut w.p2c, rr, Mode::Plain);
        let mut q0 = 0;
        while q0 < nv {
            let b = (nv - q0).min(QUERY_BLOCK);
            let q = &r.qkv[q0 * w3 + o..];
            w.mk.apply(q, w3, b, None, &mut w.s, nv, Mode::Plain);
            w.mpk.apply(q, w3, b, None, &mut w.c2p, rr, Mode::Plain);
            for i in 0..b {
                let base = r.at[q0 + i] + span;
                let row = &mut w.s[i * nv..][..nv];
                let c2p = &w.c2p[i * rr..][..rr];
                for ((sc, &pj), p2c) in row.iter_mut().zip(&r.at).zip(w.p2c.chunks_exact(rr)) {
                    let k = r.rt[base - pj] as usize;
                    *sc = (*sc + c2p[k] + p2c[k]) * inv;
                }
                softmax(row);
            }
            // SAFETY: rows q0..q0 + b of ctx, this head's d columns of
            // each, which no other task reads or writes; they end inside
            // ctx.
            unsafe {
                let out = self.ctx.add(q0 * h + o);
                w.mv.apply_to(None, &w.s, nv, b, None, out, (b - 1) * h + d, h, Mode::Plain);
            }
            q0 += b;
        }
    }
}
