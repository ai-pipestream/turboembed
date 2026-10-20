use super::TokenPromptProjections;
use crate::{
    head::Linear,
    safetensors::{Dtype, File},
    status::{Result, invalid},
};

impl TokenPromptProjections {
    /// Load explicitly named linear heads from a F32 checkpoint. A stem
    /// names `.weight` and optional `.bias`; missing bias means zeros.
    /// GLiNER variants choose their own heads; no tensor name is guessed.
    pub fn from_safetensors(hidden: usize, bytes: &[u8], token_stem: &str, prompt_stem: &str) -> Result<Self> {
        if hidden == 0 || hidden > 4096 {
            return Err(invalid("deberta projection hidden limit: 1..=4096"));
        }
        let f = File::parse("deberta projections", bytes)?;
        let load = |stem: &str| -> Result<Linear> {
            let w = f.get(&format!("{stem}.weight"), Dtype::F32, 2)?;
            if w.shape[1] != hidden as u64 || w.shape[0] == 0 || w.shape[0] > 4096 {
                return Err(invalid("deberta projection checkpoint dimensions invalid"));
            }
            let out = w.shape[0] as usize;
            let bias = if f.tensor(&format!("{stem}.bias")).is_some() {
                let b = f.get(&format!("{stem}.bias"), Dtype::F32, 1)?;
                if b.shape != [out as u64] {
                    return Err(invalid("deberta projection checkpoint bias dimension invalid"));
                }
                b.f32s()
            } else {
                vec![0.0; out]
            };
            Linear::new(hidden, out, w.f32s(), bias)
        };
        Self::new(load(token_stem)?, load(prompt_stem)?)
    }
}
