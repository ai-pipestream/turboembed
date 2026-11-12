use super::*;

impl OutputTrainer {
    /// Weight decay applies to matrices, never biases.
    fn decay(&self) -> Vec<bool> {
        let dense = self.recurrent_offset().unwrap_or(self.params.len());
        let projection = self.projection_offset().unwrap_or(self.params.len());
        (0..self.params.len())
            .map(|i| if i >= dense && i < projection { (i - dense) % 4 < 2 } else { i % 2 == 0 })
            .collect()
    }

    pub fn step(&mut self, optimizer: &mut AdamW<f32>, grads: &mut [Vec<f32>]) -> Result<f64, String> {
        let decay = self.decay();
        optimizer.step(&mut self.params, grads, &decay).map_err(|e| e.message)
    }
}

pub fn output_optimizer(settings: Settings, trainer: &OutputTrainer) -> Result<AdamW<f32>, String> {
    AdamW::new(settings, &trainer.params.iter().map(Vec::len).collect::<Vec<_>>()).map_err(|e| e.message)
}
