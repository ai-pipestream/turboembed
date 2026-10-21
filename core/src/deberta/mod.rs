//! Native CPU DeBERTa-v3 shared-key encoder foundation for extraction heads.
//! This Rust API consumes token IDs and masks, not text. It does not register
//! a C session task or implement GLiNER span/relation/schema decoding.
mod attention;
mod config;
mod math;
pub mod package;
mod projection_load;
mod projections;
#[cfg(test)]
mod tests;
mod weights;
use crate::{
    safetensors::File,
    status::{Result, invalid},
};
use attention::Run;
pub use config::Config;
pub use projections::TokenPromptProjections;
use weights::Weights;

/// Immutable, checked weights. Calls may run concurrently on the same model.
pub struct Encoder {
    config: Config,
    weights: Weights,
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
        Ok(Self { config, weights })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run one sequence. Visible token IDs, masked token IDs and sequence
    /// length are all validated before embedding lookup. An all-masked or
    /// empty sequence is an error. Padding row values follow HF semantics.
    pub fn encode(&self, ids: &[u32], mask: &[bool]) -> Result<HiddenStates> {
        let (_, last) = self.run(ids, mask, false)?;
        Ok(HiddenStates { data: last, mask: mask.to_vec(), width: self.config.hidden })
    }

    /// Embedding output then each encoder layer, for numerical reference
    /// comparison. Normal inference uses `encode` and retains only the last.
    pub fn encode_layers(&self, ids: &[u32], mask: &[bool]) -> Result<Vec<Vec<f32>>> {
        self.run(ids, mask, true).map(|(trace, _)| trace)
    }

    fn run(&self, ids: &[u32], mask: &[bool], trace: bool) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let c = &self.config;
        if ids.is_empty() || ids.len() > c.max_sequence || ids.len() != mask.len() || !mask.iter().any(|x| *x) {
            return Err(invalid("deberta sequence/mask shape or max_sequence limit"));
        }
        if ids.iter().any(|&id| id as usize >= c.vocabulary) {
            return Err(invalid("deberta token ID outside vocabulary"));
        }
        let h = c.hidden;
        let cached = self.weights.layers.iter().all(|l| l.pos.is_some());
        let mut run = Run::new(c, mask, cached)?;
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
            run.layer(c, layer, &self.weights.relative)?;
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
