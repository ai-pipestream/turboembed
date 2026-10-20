use super::{HiddenStates, math::finite};
use crate::{
    head::Linear,
    status::{Result, invalid},
};

/// Checked, immutable linear heads for token and prompt positions from one
/// encoder pass. Head weights are model-specific; no identity head is assumed.
/// Nonlinear GLiNER task heads are outside this encoder foundation.
pub struct TokenPromptProjections {
    token: Linear,
    prompt: Linear,
}
impl TokenPromptProjections {
    pub fn new(token: Linear, prompt: Linear) -> Result<Self> {
        token.validate("deberta token projection")?;
        prompt.validate("deberta prompt projection")?;
        if token.in_dim != prompt.in_dim {
            return Err(invalid("deberta projection input widths differ"));
        }
        Ok(Self { token, prompt })
    }
    pub fn tokens(&self, states: &HiddenStates, positions: &[usize]) -> Result<Vec<f32>> {
        Self::project(&self.token, states, positions)
    }
    pub fn prompts(&self, states: &HiddenStates, positions: &[usize]) -> Result<Vec<f32>> {
        Self::project(&self.prompt, states, positions)
    }
    fn project(head: &Linear, states: &HiddenStates, positions: &[usize]) -> Result<Vec<f32>> {
        if states.width != head.in_dim
            || positions.len() > states.mask.len()
            || positions.iter().any(|&p| states.mask.get(p) != Some(&true))
        {
            return Err(invalid("deberta projection width, position count, bounds or padding mismatch"));
        }
        let count =
            positions.len().checked_mul(head.out_dim).ok_or_else(|| invalid("deberta projection size overflow"))?;
        let mut out = vec![0.0; count];
        for (&p, row) in positions.iter().zip(out.chunks_exact_mut(head.out_dim)) {
            head.apply(&states.data[p * head.in_dim..(p + 1) * head.in_dim], row);
        }
        finite(&out)?;
        Ok(out)
    }
}
