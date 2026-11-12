use turbo::safetensors::File;
use turbo_trainer::adamw::AdamW;
use turbo_trainer::encoder::deberta_layer::{DebertaLayer, Work};
use turbo_trainer::rng::Rng;

use crate::train::TrainingInput;
use crate::{Backend, Candidate, ExtractionSchema, Model, PackedInput};

pub(crate) struct State {
    layer: DebertaLayer<f32>,
    relative: Vec<f32>,
    index: usize,
}

pub struct LoraInput {
    pub training: TrainingInput,
    pub candidates: Vec<Candidate>,
    work: Work<f32>,
    positions: Vec<usize>,
}

impl Model {
    pub fn enable_lora(&mut self, encoder: &[u8], rank: usize, alpha: f64, seed: u64) -> Result<(), String> {
        if self.lora.is_some() {
            return Err("gliner LoRA is already enabled".into());
        }
        if !matches!(self.encoder, Backend::Cpu(_)) {
            return Err("gliner LoRA training requires the CPU encoder".into());
        }
        let config = self.encoder_config().clone();
        let index = config.layers.checked_sub(1).ok_or("gliner LoRA needs an encoder layer")?;
        let file = File::parse("gliner encoder", encoder).map_err(|e| e.message)?;
        let layer = DebertaLayer::load(config.clone(), &file, index, rank, alpha, &mut Rng::new(seed))
            .map_err(|e| e.message)?;
        let relative = DebertaLayer::relative(&config, &file).map_err(|e| e.message)?;
        self.lora = Some(State { layer, relative, index });
        Ok(())
    }

    pub fn lora_enabled(&self) -> bool {
        self.lora.is_some()
    }

    pub(crate) fn lora_forward(&self, ids: &[u32]) -> Result<(Vec<f32>, Work<f32>), String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        let Backend::Cpu(encoder) = &self.encoder else {
            return Err("gliner LoRA training requires the CPU encoder".into());
        };
        let trace = encoder.encode_layers(ids, &vec![true; ids.len()]).map_err(|e| e.message)?;
        let x = trace.get(state.index).ok_or("gliner LoRA encoder layer is out of range")?.clone();
        let mut work = Work::default();
        state.layer.forward(&x, &state.relative, &mut work);
        Ok((x, work))
    }

    pub(crate) fn lora_selected(&self, p: &PackedInput) -> Result<Option<Vec<f32>>, String> {
        if self.lora.is_none() {
            return Ok(None);
        }
        let (_, work) = self.lora_forward(&p.ids)?;
        let h = self.encoder_config().hidden;
        let mut out = Vec::new();
        for &at in p.prompts.iter().chain(&p.words) {
            out.extend_from_slice(&work.out[at * h..(at + 1) * h]);
        }
        Ok(Some(out))
    }

    pub fn training_input_lora(&self, words: &[&str], schema: &ExtractionSchema) -> Result<LoraInput, String> {
        let packed = self.pack(words, schema)?;
        let (_, work) = self.lora_forward(&packed.ids)?;
        let h = self.encoder_config().hidden;
        let mut states = Vec::new();
        for &at in packed.prompts.iter().chain(&packed.words) {
            states.extend_from_slice(&work.out[at * h..(at + 1) * h]);
        }
        let labels = packed.prompts.len();
        let prompt_rows: Vec<usize> = (0..labels).collect();
        let word_rows: Vec<usize> = (labels..labels + packed.words.len()).collect();
        let (training, candidates) =
            self.heads.training(&states, h, &prompt_rows, &word_rows, self.config.max_width)?;
        let positions = packed.prompts.iter().chain(&packed.words).copied().collect();
        Ok(LoraInput { training, candidates, work, positions })
    }

    pub fn lora_zero_grads(&self) -> Result<Vec<Vec<f32>>, String> {
        Ok(self.lora.as_ref().ok_or("gliner LoRA is not enabled")?.layer.zero_grads())
    }

    pub fn lora_backward(&self, input: &LoraInput, selected: &[f32], grads: &mut [Vec<f32>]) -> Result<(), String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        let h = self.encoder_config().hidden;
        let expected = state.layer.zero_grads();
        if selected.len() != input.positions.len() * h
            || grads.len() != expected.len()
            || grads.iter().zip(&expected).any(|(a, b)| a.len() != b.len())
        {
            return Err("gliner LoRA gradient shape mismatch".into());
        }
        let mut dstates = vec![0.0; input.work.out.len()];
        for (row, &at) in input.positions.iter().enumerate() {
            dstates[at * h..(at + 1) * h].copy_from_slice(&selected[row * h..(row + 1) * h]);
        }
        state.layer.backward(&input.work, &state.relative, &dstates, grads);
        Ok(())
    }

    pub fn lora_step(&mut self, optimizer: &mut AdamW<f32>, grads: &mut [Vec<f32>]) -> Result<f64, String> {
        self.lora.as_mut().ok_or("gliner LoRA is not enabled")?.layer.step(optimizer, grads).map_err(|e| e.message)
    }

    pub fn write_lora(&self, original: &[u8]) -> Result<Vec<u8>, String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        state.layer.write_merged(original, state.index).map_err(|e| e.message)
    }
}
