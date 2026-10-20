use super::{Config, math::Norm};
use crate::{
    head::Linear,
    safetensors::{Dtype, File},
    status::{Result, invalid},
};

pub(super) struct Layer {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub attention: Linear,
    pub intermediate: Linear,
    pub output: Linear,
    pub attention_norm: Norm,
    pub output_norm: Norm,
}

pub(super) struct Weights {
    pub embedding: Vec<f32>,
    pub embedding_norm: Norm,
    pub relative: Vec<f32>,
    pub layers: Vec<Layer>,
}

impl Weights {
    pub fn load(config: &Config, file: &File<'_>) -> Result<Self> {
        // Check all shapes before copying any tensor. Header dimensions never
        // determine allocation sizes until matched to validated configuration.
        for (name, shape) in config.tensor_shapes()? {
            let t = file.get(&name, Dtype::F32, shape.len())?;
            if t.shape != shape {
                return Err(invalid(format!("deberta {name}: shape mismatch")));
            }
            if t.data.as_chunks::<4>().0.iter().any(|b| !f32::from_le_bytes(*b).is_finite()) {
                return Err(invalid(format!("deberta {name}: non-finite weight")));
            }
        }
        let get = |name: &str| file.tensor(name).unwrap().f32s();
        let norm = |stem: &str| Norm { weight: get(&format!("{stem}.weight")), bias: get(&format!("{stem}.bias")) };
        let linear = |stem: &str, input, output| {
            Linear::new(input, output, get(&format!("{stem}.weight")), get(&format!("{stem}.bias")))
        };
        let h = config.hidden;
        let mut layers = Vec::new();
        for i in 0..config.layers {
            let prefix = format!("encoder.layer.{i}");
            layers.push(Layer {
                q: linear(&format!("{prefix}.attention.self.query_proj"), h, h)?,
                k: linear(&format!("{prefix}.attention.self.key_proj"), h, h)?,
                v: linear(&format!("{prefix}.attention.self.value_proj"), h, h)?,
                attention: linear(&format!("{prefix}.attention.output.dense"), h, h)?,
                intermediate: linear(&format!("{prefix}.intermediate.dense"), h, config.intermediate)?,
                output: linear(&format!("{prefix}.output.dense"), config.intermediate, h)?,
                attention_norm: norm(&format!("{prefix}.attention.output.LayerNorm")),
                output_norm: norm(&format!("{prefix}.output.LayerNorm")),
            });
        }
        let mut relative = get("encoder.rel_embeddings.weight");
        norm("encoder.LayerNorm").apply(&mut relative, config.layer_norm_eps)?;
        Ok(Self {
            embedding: get("embeddings.word_embeddings.weight"),
            embedding_norm: norm("embeddings.LayerNorm"),
            relative,
            layers,
        })
    }
}
