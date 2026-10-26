//! Native CPU DeBERTa-v3 shared-key encoder foundation for extraction heads.
//! This Rust API consumes token IDs and masks, not text. It does not register
//! a C session task or implement GLiNER span/relation/schema decoding.
//!
//! An encoder computes on its own pool of threads: one sequence's products
//! and heads across them, or a batch's sequences one to a thread. Every
//! output is computed the same on any thread, so the bits do not depend on
//! the thread count or on what else is in a batch.
mod attention;
mod config;
mod math;
pub mod package;
mod projection_load;
mod projections;
#[cfg(test)]
mod tests;
mod weights;
use std::sync::Mutex;

use crate::{
    cpu::pool::Pool,
    safetensors::File,
    status::{Error, Result, invalid},
};
use attention::Run;
pub use config::Config;
pub use projections::TokenPromptProjections;
use weights::Weights;

/// Immutable, checked weights. Calls may run concurrently on the same
/// model: one at a time takes the pool, and the others run on their own
/// thread, with the same bits.
pub struct Encoder {
    config: Config,
    weights: Weights,
    pool: Mutex<Pool>,
}

/// Hidden states retain the input mask so heads cannot select padding rows.
pub struct HiddenStates {
    data: Vec<f32>,
    mask: Vec<bool>,
    width: usize,
}
impl HiddenStates {
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
}

impl Encoder {
    /// Load canonical F32 safetensors; all required weights must be finite
    /// and exactly match the validated configuration before they are copied.
    pub fn from_safetensors(config: Config, bytes: &[u8]) -> Result<Self> {
        config.check()?;
        if bytes.len() as u64 > package::MAX_WEIGHT_BYTES {
            return Err(invalid("deberta weight byte limit exceeded"));
        }
        let file = File::parse("deberta weights", bytes)?;
        let weights = Weights::load(&config, &file)?;
        let threads = crate::cpu::threads(0, std::env::var("TURBO_CPU_THREADS").ok().as_deref())
            .map_err(|r| Error::new(r.code, r.message))?;
        Ok(Self { config, weights, pool: Mutex::new(Pool::new(threads)) })
    }

    /// The encoder from a backend table's F32 tensors, in TURBO_DEBERTA_*
    /// then TURBO_BERT_* layer order (turbo_backend.h). It computes on the
    /// pool its caller hands `encode_on`.
    pub(crate) fn from_table(config: Config, tensors: &[&[f32]]) -> Result<Self> {
        config.check()?;
        let weights = Weights::from_table(&config, tensors)?;
        Ok(Self { config, weights, pool: Mutex::new(Pool::new(1)) })
    }

    /// One sequence's last hidden states, [ids.len(), hidden], on `pool`'s
    /// threads: `encode`'s bits whatever the thread count.
    pub(crate) fn encode_on(&self, pool: &mut Pool, ids: &[u32], mask: &[bool]) -> Result<Vec<f32>> {
        self.run(ids, mask, false, Some(pool)).map(|(_, last)| last)
    }

    /// The same encoder on `threads` threads, the caller's among them:
    /// 1 to 1024. Without it, TURBO_CPU_THREADS when set, else one per
    /// processor this process may run on, as for a CPU session
    /// (docs/cpu.md).
    pub fn with_threads(self, threads: usize) -> Result<Self> {
        if !(1..=crate::cpu::MAX_THREADS).contains(&threads) {
            return Err(invalid(format!("deberta threads {threads}: the count must be 1 to 1024")));
        }
        Ok(Self { pool: Mutex::new(Pool::new(threads)), ..self })
    }

    /// The threads the encoder's pool runs on.
    pub fn threads(&self) -> usize {
        self.pool.lock().map_or(1, |p| p.threads())
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run one sequence. Visible token IDs, masked token IDs and sequence
    /// length are all validated before embedding lookup. An all-masked or
    /// empty sequence is an error. Padding row values follow HF semantics.
    pub fn encode(&self, ids: &[u32], mask: &[bool]) -> Result<HiddenStates> {
        let mut pool = self.pool.try_lock().ok();
        let (_, last) = self.run(ids, mask, false, pool.as_deref_mut())?;
        Ok(HiddenStates { data: last, mask: mask.to_vec(), width: self.config.hidden })
    }

    /// `encode` of each row, in order: the rows one to a thread of the
    /// pool, each giving the bits it gives alone. A row that fails fails
    /// alone.
    pub fn encode_batch(&self, rows: &[(&[u32], &[bool])]) -> Vec<Result<HiddenStates>> {
        let out: Vec<Mutex<Option<Result<HiddenStates>>>> = rows.iter().map(|_| Mutex::new(None)).collect();
        let one = |i: usize| {
            let (ids, mask) = rows[i];
            let r = self.run(ids, mask, false, None);
            let r = r.map(|(_, last)| HiddenStates { data: last, mask: mask.to_vec(), width: self.config.hidden });
            if let Ok(mut slot) = out[i].lock() {
                *slot = Some(r);
            }
        };
        match self.pool.try_lock() {
            Ok(mut pool) => pool.run(rows.len(), &|i, _| one(i)),
            Err(_) => (0..rows.len()).for_each(one),
        }
        out.into_iter()
            .map(|m| {
                m.into_inner().ok().flatten().unwrap_or_else(|| Err(invalid("deberta batch row was not computed")))
            })
            .collect()
    }

    /// Embedding output then each encoder layer, for numerical reference
    /// comparison. Normal inference uses `encode` and retains only the last.
    pub fn encode_layers(&self, ids: &[u32], mask: &[bool]) -> Result<Vec<Vec<f32>>> {
        let mut pool = self.pool.try_lock().ok();
        self.run(ids, mask, true, pool.as_deref_mut()).map(|(trace, _)| trace)
    }

    fn run(
        &self,
        ids: &[u32],
        mask: &[bool],
        trace: bool,
        mut pool: Option<&mut Pool>,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let c = &self.config;
        if ids.is_empty() || ids.len() > c.max_sequence || ids.len() != mask.len() || !mask.iter().any(|x| *x) {
            return Err(invalid("deberta sequence/mask shape or max_sequence limit"));
        }
        if ids.iter().any(|&id| id as usize >= c.vocabulary) {
            return Err(invalid("deberta token ID outside vocabulary"));
        }
        let h = c.hidden;
        let cached = self.weights.layers.iter().all(|l| l.pos.is_some());
        let mut run = Run::new(c, mask, cached, pool.as_ref().map_or(1, |p| p.threads()))?;
        // Visible rows are their embeddings, LayerNorm'd; the shared
        // masked row stays zero, as upstream multiplies it by its mask.
        for (row, &p) in run.x.chunks_exact_mut(h).zip(&run.at) {
            let id = ids[p] as usize;
            row.copy_from_slice(&self.weights.embedding[id * h..(id + 1) * h]);
        }
        let visible = run.at.len() * h;
        self.weights.embedding_norm.apply(&mut run.x[..visible], c.layer_norm_eps)?;
        let mut layers = Vec::new();
        if trace {
            layers.push(scatter(&run, h)?);
        }
        for layer in &self.weights.layers {
            run.layer(c, layer, &self.weights.relative, pool.as_deref_mut())?;
            if trace {
                layers.push(scatter(&run, h)?);
            }
        }
        let last = scatter(&run, h)?;
        Ok((layers, last))
    }
}

/// The run's rows at their positions: each visible row where it was, the
/// shared masked row at every masked position.
fn scatter(run: &Run, h: usize) -> Result<Vec<f32>> {
    let mut out = weights::zeros(run.n * h)?;
    let mut next = run.at.iter().peekable();
    let mut seen = 0;
    for (p, dst) in out.chunks_exact_mut(h).enumerate() {
        let row = if next.peek() == Some(&&p) {
            next.next();
            seen += 1;
            seen - 1
        } else {
            run.at.len()
        };
        dst.copy_from_slice(&run.x[row * h..(row + 1) * h]);
    }
    Ok(out)
}
