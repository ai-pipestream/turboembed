//! Slot range-pooling execution and post-processing helpers for
//! `ANALYZE_OUTPUT_SPARSE_TERMS` and `ANALYZE_OUTPUT_TOKEN_VECTORS` (`#917`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use turbo::head::ProjectionHead;
use turbo::mlm::MlmHead;
use turbo::sparse::{MaxLogPool, SparseTerm};
use turbo::{
    TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, TURBO_TASK_EMBED, turbo_embed_options, turbo_token_range,
};
use turbo_analysis::{Ix, Span, View};
use turbo_analysis_embed::window::Placed;
use turbo_analysis_embed::{Rows, normalize};
use turbo_analysis_layers::{SparseTerms, Subwords};

use crate::api::{
    Failure, Result, TURBO_E_INTERNAL, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED, TURBO_E_UNSUPPORTED_OPTION,
};
use crate::state::{Held, Loaded, Slot};
use crate::stream::Shared;

impl Loaded {
    /// Lazily loads the companion tokenizer and optional MLM / projection heads
    /// when a structured `TASK_EMBED` output (`SparseTerms` or `TokenVectors`)
    /// is requested, so plain `AnalyzeVectors` never pays for or fails on
    /// companion head files.
    pub(crate) fn embed(&self) -> Result<&super::EmbedModel> {
        if self.task != TURBO_TASK_EMBED {
            return Err(Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, "the model has no embed companion"));
        }
        self.embed
            .get_or_init(|| super::EmbedModel::load(&self.bundle, &self.model_info))
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Held {
    /// Ensures this slot's session reserves `max_batch * max_seq` output rows
    /// for per-token structured outputs (`SparseTerms` or `TokenVectors`),
    /// upgrading the session on first structured use just as
    /// `Encoder::reserve_token_ranges` does.
    pub(crate) fn reserve_token_ranges(&self) -> Result<()> {
        let slot = self.slot();
        let needed = slot.info.max_batch.saturating_mul(slot.info.max_seq);
        if slot.max_ranges() >= needed {
            return Ok(());
        }
        let mut desc = slot.desc;
        desc.max_ranges = needed;
        match self.loaded.model.session(&desc) {
            Ok(s) => {
                slot.session.replace(s);
                slot.max_ranges.store(needed, Ordering::Release);
                Ok(())
            }
            Err(f) if f.code == TURBO_E_UNSUPPORTED_OPTION && f.field == 7 => Ok(()),
            Err(f) => Err(f),
        }
    }
}

pub(super) fn check_alive(dead: &AtomicBool) -> Result<()> {
    if dead.load(Ordering::Acquire) {
        return Err(Failure::new(TURBO_E_INVALID_STATE, 0, "the client cancelled the stream"));
    }
    Ok(())
}

pub(super) fn fill_sparse_terms(out: &mut SparseTerms<View>, sub: &Subwords, selected: &[SparseTerm]) {
    let mut spans: HashMap<i32, (Span, Vec<Span>)> = HashMap::new();
    for k in 0..sub.ids.len() {
        let s = sub.spans.get(Ix::new(k as u32));
        spans
            .entry(sub.ids[k])
            .and_modify(|(hull, occ)| {
                hull.start = hull.start.min(s.start);
                hull.end = hull.end.max(s.end);
                occ.push(s);
            })
            .or_insert((s, vec![s]));
    }
    for t in selected {
        match spans.get(&(t.term as i32)) {
            Some((hull, occ)) => {
                out.push(t.term, t.weight, Some(*hull));
                out.push_occurrences(occ);
            }
            None => {
                out.push(t.term, t.weight, None);
                out.push_occurrences(&[]);
            }
        }
    }
    out.unit();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn flush_sparse(
    shared: &Shared,
    dead: &AtomicBool,
    slot: &Slot,
    mlm: &MlmHead,
    rows: &mut Rows,
    ranges: &mut Vec<turbo_token_range>,
    owners: &mut Vec<(u32, u32)>,
    states: &mut Vec<f32>,
    pools: &mut [Option<MaxLogPool>],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    run_ranges_on_slot(shared, dead, slot, rows, ranges, mlm.hidden as u32, states)?;
    let mut at = 0usize;
    while at < owners.len() {
        let d = owners[at].0;
        let end = owners[at..].iter().position(|&(x, _)| x != d).map_or(owners.len(), |k| at + k);
        let pool = pools[d as usize].as_mut().expect("a document with rows has a pool");
        mlm.relu_peak_into(&states[at * mlm.hidden..end * mlm.hidden], pool)
            .map_err(|e| Failure::new(e.code, e.field, e.message))?;
        at = end;
    }
    rows.clear();
    ranges.clear();
    owners.clear();
    Ok(())
}

pub(super) struct TokenRunBufs {
    pub(super) rows: Rows,
    pub(super) ranges: Vec<turbo_token_range>,
    pub(super) targets: Vec<u32>,
    flat: Vec<f32>,
}

impl TokenRunBufs {
    pub(super) fn new(max_seq: u32, fill: i32) -> Self {
        Self { rows: Rows::new(max_seq, fill), ranges: Vec::new(), targets: Vec::new(), flat: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn flush(
        &mut self,
        shared: &Shared,
        dead: &AtomicBool,
        slot: &Slot,
        hidden_dim: usize,
        out_dim: usize,
        projection: Option<&ProjectionHead>,
        l2: bool,
        placed: &[Placed],
        out: &mut [Vec<f32>],
    ) -> Result<()> {
        run_ranges_on_slot(shared, dead, slot, &self.rows, &self.ranges, hidden_dim as u32, &mut self.flat)?;
        for (v, &t) in self.flat.chunks_exact(hidden_dim).zip(self.targets.iter()) {
            let x = placed[t as usize];
            let idx = x.target.item as usize;
            let dst = &mut out[x.target.doc as usize][idx * out_dim..(idx + 1) * out_dim];
            if let Some(proj) = projection {
                proj.project_into(v, l2, dst).map_err(|e| Failure::new(e.code, e.field, e.message))?;
            } else {
                dst.copy_from_slice(&v[..out_dim]);
                if l2 {
                    normalize(dst);
                }
            }
        }
        self.rows.clear();
        self.ranges.clear();
        self.targets.clear();
        Ok(())
    }
}

fn run_ranges_on_slot(
    shared: &Shared,
    dead: &AtomicBool,
    slot: &Slot,
    rows: &Rows,
    ranges: &[turbo_token_range],
    dim: u32,
    out: &mut Vec<f32>,
) -> Result<()> {
    check_alive(dead)?;
    let opts = turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: 0,
        normalize: TURBO_NORMALIZE_NONE,
        pooling: TURBO_POOLING_MEAN,
        output_dim: 0,
    };
    slot.session.write_ranges(&rows.batch(), ranges, &opts).map_err(|f| {
        if f.code == TURBO_E_UNSUPPORTED && f.message.contains("does not offer embed_write_ranges") {
            let msg = format!("the backend does not offer range pooling (embed_write_ranges): {}", f.message);
            Failure::new(TURBO_E_UNSUPPORTED, f.field, msg)
        } else {
            f
        }
    })?;
    check_alive(dead)?;
    shared.metrics.record_batch_size(rows.len() as usize);
    shared.runs.fetch_add(1, Ordering::Relaxed);
    let result = slot.session.run()?;
    check_alive(dead)?;
    let info = result.info()?;
    if info.dtype != TURBO_DTYPE_F32 {
        return Err(Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, "range result is not f32"));
    }
    if info.batch != ranges.len() as u32 || info.dim != dim {
        let msg = format!("range result has [{}, {}], expected [{}, {dim}]", info.batch, info.dim, ranges.len());
        return Err(Failure::new(TURBO_E_INTERNAL, 0, msg));
    }
    let bytes = result.read(info.bytes)?;
    out.clear();
    out.reserve(ranges.len() * dim as usize);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(())
}
