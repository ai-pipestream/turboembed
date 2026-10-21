use super::{Config, math::Norm};
use crate::{
    cpu::{Matrix, Mode},
    safetensors::{Dtype, File},
    status::{Result, invalid},
};

/// One linear layer packed for the CPU kernels, and its bias.
pub(super) struct Dense {
    pub w: Matrix,
    pub b: Vec<f32>,
}

pub(super) struct Layer {
    /// Q, K and V as one product of 3 * hidden columns.
    pub qkv: Dense,
    pub attention: Dense,
    pub intermediate: Dense,
    pub output: Dense,
    pub attention_norm: Norm,
    pub output_norm: Norm,
    /// The shared-key projections of every normalized relative row,
    /// [2 * position_buckets, 2 * hidden]: query_proj then key_proj. They
    /// depend on weights alone, so they are made once at load. None when
    /// they would pass POSITION_CACHE_BYTES; a run then projects only the
    /// rows its distances reach, with the same product.
    pub pos: Option<Vec<f32>>,
}

pub(super) struct Weights {
    pub embedding: Vec<f32>,
    pub embedding_norm: Norm,
    pub relative: Vec<f32>,
    pub layers: Vec<Layer>,
}

/// The most bytes of relative projections kept across all layers.
pub(super) const POSITION_CACHE_BYTES: usize = 1 << 30;

pub(super) fn zeros(n: usize) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| invalid("deberta workspace allocation failed"))?;
    v.resize(n, 0.0);
    Ok(v)
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
        let oom = |_| invalid("deberta weight packing allocation failed");
        let h = config.hidden;
        let dense = |stems: &[String], input: usize| -> Result<Dense> {
            let ws: Vec<Vec<f32>> = stems.iter().map(|s| get(&format!("{s}.weight"))).collect();
            let parts: Vec<&[f32]> = ws.iter().map(Vec::as_slice).collect();
            let b = stems.iter().flat_map(|s| get(&format!("{s}.bias"))).collect();
            Ok(Dense { w: Matrix::from_rows(&parts, input).map_err(oom)?, b })
        };
        let mut relative = get("encoder.rel_embeddings.weight");
        norm("encoder.LayerNorm").apply(&mut relative, config.layer_norm_eps)?;
        let rows = 2 * config.position_buckets;
        let cache = config.layers.saturating_mul(rows).saturating_mul(2 * h).saturating_mul(4) <= POSITION_CACHE_BYTES;
        let mut layers = Vec::new();
        for i in 0..config.layers {
            let p = format!("encoder.layer.{i}");
            let stem = |s: &str| format!("{p}.{s}");
            let qkv = dense(
                &[
                    stem("attention.self.query_proj"),
                    stem("attention.self.key_proj"),
                    stem("attention.self.value_proj"),
                ],
                h,
            )?;
            let pos = if cache {
                let mut all = zeros(rows * 3 * h)?;
                qkv.w.apply(&relative, h, rows, Some(&qkv.b), &mut all, 3 * h, Mode::Plain);
                let mut pos = zeros(rows * 2 * h)?;
                for (dst, src) in pos.chunks_exact_mut(2 * h).zip(all.chunks_exact(3 * h)) {
                    dst.copy_from_slice(&src[..2 * h]);
                }
                Some(pos)
            } else {
                None
            };
            layers.push(Layer {
                qkv,
                attention: dense(&[stem("attention.output.dense")], h)?,
                intermediate: dense(&[stem("intermediate.dense")], h)?,
                output: dense(&[stem("output.dense")], config.intermediate)?,
                attention_norm: norm(&stem("attention.output.LayerNorm")),
                output_norm: norm(&stem("output.LayerNorm")),
                pos,
            });
        }
        Ok(Self {
            embedding: get("embeddings.word_embeddings.weight"),
            embedding_norm: norm("embeddings.LayerNorm"),
            relative,
            layers,
        })
    }
}
