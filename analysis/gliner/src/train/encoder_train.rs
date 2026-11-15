use super::*;

impl OutputTrainer {
    /// Adds one sentence's output and enabled head gradients.
    pub fn gradients(
        &self,
        input: &TrainingInput,
        candidates: &[Candidate],
        gold: &[(usize, usize, usize)],
        grads: &mut [Vec<f32>],
    ) -> Result<f64, String> {
        self.gradients_impl(input, candidates, gold, grads).map(|(loss, _)| loss)
    }

    /// Also returns the gradient of the selected prompt and word encoder rows.
    pub fn gradients_with_encoder(
        &self,
        input: &TrainingInput,
        candidates: &[Candidate],
        gold: &[(usize, usize, usize)],
        grads: &mut [Vec<f32>],
    ) -> Result<(f64, Vec<f32>), String> {
        if self.projection_offset().is_none() {
            return Err("gliner encoder gradients require token projection training".into());
        }
        self.gradients_impl(input, candidates, gold, grads)
    }
}
