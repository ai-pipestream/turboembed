use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::status::{Result, invalid};

/// The shared-key, c2p+p2c DeBERTa-v3 inference profile. Dropout is disabled.
/// Absolute/type embeddings, convolution and separate positional keys are not
/// part of this profile; conversion must reject checkpoints requiring them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub layers: usize,
    pub vocabulary: usize,
    pub max_sequence: usize,
    pub position_buckets: usize,
    pub max_relative_positions: usize,
    pub layer_norm_eps: f64,
}

impl Config {
    /// Named limits bound parameter metadata and runtime workspace sizes.
    pub fn check(&self) -> Result<()> {
        for (name, value, limit) in [
            ("hidden", self.hidden, 4096),
            ("intermediate", self.intermediate, 16384),
            ("heads", self.heads, 128),
            ("layers", self.layers, 128),
            ("vocabulary", self.vocabulary, 1_000_000),
            ("max_sequence", self.max_sequence, 4096),
            ("position_buckets", self.position_buckets, 4096),
            ("max_relative_positions", self.max_relative_positions, 65536),
        ] {
            if value == 0 || value > limit {
                return Err(invalid(format!("deberta {name} limit: 1..={limit}")));
            }
        }
        if !self.hidden.is_multiple_of(self.heads) {
            return Err(invalid("deberta hidden must divide by heads"));
        }
        if self.position_buckets < 4
            || !self.position_buckets.is_multiple_of(2)
            || self.max_relative_positions <= self.position_buckets / 2 + 1
        {
            return Err(invalid("deberta invalid relative position bucket geometry"));
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 || self.layer_norm_eps > 1.0 {
            return Err(invalid("deberta layer_norm_eps must be finite in (0,1]"));
        }
        Ok(())
    }

    /// Canonical F32 tensor names/shapes, also used by checkpoint conversion.
    pub fn tensor_shapes(&self) -> Result<BTreeMap<String, Vec<u64>>> {
        self.check()?;
        let (h, f) = (self.hidden as u64, self.intermediate as u64);
        let mut s = BTreeMap::new();
        s.insert("embeddings.word_embeddings.weight".into(), vec![self.vocabulary as u64, h]);
        s.insert("encoder.rel_embeddings.weight".into(), vec![2 * self.position_buckets as u64, h]);
        for stem in ["embeddings.LayerNorm", "encoder.LayerNorm"] {
            for suffix in ["weight", "bias"] {
                s.insert(format!("{stem}.{suffix}"), vec![h]);
            }
        }
        for layer in 0..self.layers {
            let prefix = format!("encoder.layer.{layer}");
            for (stem, input, output) in [
                ("attention.self.query_proj", h, h),
                ("attention.self.key_proj", h, h),
                ("attention.self.value_proj", h, h),
                ("attention.output.dense", h, h),
                ("intermediate.dense", h, f),
                ("output.dense", f, h),
            ] {
                s.insert(format!("{prefix}.{stem}.weight"), vec![output, input]);
                s.insert(format!("{prefix}.{stem}.bias"), vec![output]);
            }
            for stem in ["attention.output.LayerNorm", "output.LayerNorm"] {
                for suffix in ["weight", "bias"] {
                    s.insert(format!("{prefix}.{stem}.{suffix}"), vec![h]);
                }
            }
        }
        Ok(s)
    }
}
