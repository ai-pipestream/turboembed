//! Native CPU DeBERTa-v3 shared-key encoder foundation for extraction heads.
//! This Rust API consumes token IDs and masks, not text. It does not register
//! a C session task or implement GLiNER span/relation/schema decoding.
mod attention;
mod config;
mod math;
pub mod package;
mod projection_load;
mod projections;
mod weights;
use crate::{
    safetensors::File,
    status::{Result, invalid},
};
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
        let mut x = Vec::with_capacity(ids.len() * c.hidden);
        for &id in ids {
            x.extend_from_slice(&self.weights.embedding[id as usize * c.hidden..(id as usize + 1) * c.hidden]);
        }
        self.weights.embedding_norm.apply(&mut x, c.layer_norm_eps)?;
        for (row, &visible) in x.chunks_exact_mut(c.hidden).zip(mask) {
            if !visible {
                row.fill(0.0);
            }
        }
        let mut layers = Vec::new();
        if trace {
            layers.push(x.clone());
        }
        for layer in &self.weights.layers {
            let mut a = attention::attention(c, layer, &x, &self.weights.relative, mask)?;
            for (v, old) in a.iter_mut().zip(&x) {
                *v += old;
            }
            layer.attention_norm.apply(&mut a, c.layer_norm_eps)?;
            let mut middle = math::linear(&a, &layer.intermediate)?;
            for v in &mut middle {
                *v = crate::cpu::gelu(*v);
            }
            x = math::linear(&middle, &layer.output)?;
            for (v, residual) in x.iter_mut().zip(&a) {
                *v += residual;
            }
            layer.output_norm.apply(&mut x, c.layer_norm_eps)?;
            if trace {
                layers.push(x.clone());
            }
        }
        Ok((layers, x))
    }
}
