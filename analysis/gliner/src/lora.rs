use turbo::safetensors::File;
use turbo_trainer::adamw::AdamW;
use turbo_trainer::encoder::deberta_layer::{DebertaLayer, Work};
use turbo_trainer::rng::Rng;

use crate::train::TrainingInput;
use crate::{Backend, Candidate, ExtractionSchema, Model, PackedInput};

pub(crate) struct State {
    layers: Vec<DebertaLayer<f32>>,
    relative: Vec<f32>,
    start: usize,
}

pub struct LoraInput {
    pub training: TrainingInput,
    pub candidates: Vec<Candidate>,
    pub work: Vec<Work<f32>>,
    positions: Vec<usize>,
}

impl Model {
    pub fn enable_lora(&mut self, encoder: &[u8], rank: usize, alpha: f64, seed: u64) -> Result<(), String> {
        self.enable_lora_layers(encoder, rank, alpha, seed, 1)
    }

    pub fn enable_lora_layers(
        &mut self,
        encoder: &[u8],
        rank: usize,
        alpha: f64,
        seed: u64,
        layers: usize,
    ) -> Result<(), String> {
        if self.lora.is_some() {
            return Err("gliner LoRA is already enabled".into());
        }
        if !matches!(self.encoder, Backend::Cpu(_)) {
            return Err("gliner LoRA training requires the CPU encoder".into());
        }
        let config = self.encoder_config().clone();
        if layers == 0 || layers > config.layers {
            return Err(format!("gliner LoRA layers must be between 1 and {}", config.layers));
        }
        let start = config.layers - layers;
        let file = File::parse("gliner encoder", encoder).map_err(|e| e.message)?;
        let relative = DebertaLayer::relative(&config, &file).map_err(|e| e.message)?;
        let mut rng = Rng::new(seed);
        let mut trained = Vec::with_capacity(layers);
        for index in start..config.layers {
            trained
                .push(DebertaLayer::load(config.clone(), &file, index, rank, alpha, &mut rng).map_err(|e| e.message)?);
        }
        self.lora = Some(State { layers: trained, relative, start });
        Ok(())
    }

    pub fn lora_enabled(&self) -> bool {
        self.lora.is_some()
    }

    pub(crate) fn lora_forward(&self, ids: &[u32]) -> Result<Vec<Work<f32>>, String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        let Backend::Cpu(encoder) = &self.encoder else {
            return Err("gliner LoRA training requires the CPU encoder".into());
        };
        let trace = encoder.encode_layers(ids, &vec![true; ids.len()]).map_err(|e| e.message)?;
        let mut works = Vec::with_capacity(state.layers.len());
        for (offset, layer) in state.layers.iter().enumerate() {
            let index = state.start + offset;
            let x = trace.get(index).ok_or("gliner LoRA encoder layer is out of range")?;
            let mut work = Work::default();
            layer.forward(x, &state.relative, &mut work);
            works.push(work);
        }
        Ok(works)
    }

    pub(crate) fn lora_selected(&self, p: &PackedInput) -> Result<Option<Vec<f32>>, String> {
        if self.lora.is_none() {
            return Ok(None);
        }
        let works = self.lora_forward(&p.ids)?;
        let work = works.last().ok_or("gliner LoRA needs an encoder layer")?;
        let h = self.encoder_config().hidden;
        let mut out = Vec::new();
        for &at in p.prompts.iter().chain(&p.words) {
            out.extend_from_slice(&work.out[at * h..(at + 1) * h]);
        }
        Ok(Some(out))
    }

    pub fn training_input_lora(&self, words: &[&str], schema: &ExtractionSchema) -> Result<LoraInput, String> {
        let packed = self.pack(words, schema)?;
        let works = self.lora_forward(&packed.ids)?;
        let work = works.last().ok_or("gliner LoRA needs an encoder layer")?;
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
        Ok(LoraInput { training, candidates, work: works, positions })
    }

    pub fn lora_zero_grads(&self) -> Result<Vec<Vec<f32>>, String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        Ok(state.layers.iter().flat_map(|layer| layer.zero_grads()).collect())
    }

    pub fn lora_backward(&self, input: &LoraInput, selected: &[f32], grads: &mut [Vec<f32>]) -> Result<(), String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        let h = self.encoder_config().hidden;
        let shapes_match = input.work.len() == state.layers.len()
            && grads.len() == state.layers.iter().map(|layer| layer.params().len()).sum::<usize>()
            && grads.iter().zip(state.layers.iter().flat_map(|layer| layer.params())).all(|(a, b)| a.len() == b.len());
        if selected.len() != input.positions.len() * h || !shapes_match {
            return Err("gliner LoRA gradient shape mismatch".into());
        }
        let mut dstates = vec![0.0; input.work.last().ok_or("gliner LoRA needs an encoder layer")?.out.len()];
        for (row, &at) in input.positions.iter().enumerate() {
            dstates[at * h..(at + 1) * h].copy_from_slice(&selected[row * h..(row + 1) * h]);
        }
        let mut offset = grads.len();
        for (layer, work) in state.layers.iter().zip(&input.work).rev() {
            offset -= layer.params().len();
            dstates =
                layer.backward(work, &state.relative, &dstates, &mut grads[offset..offset + layer.params().len()]);
        }
        Ok(())
    }

    pub fn lora_step(&mut self, optimizer: &mut AdamW<f32>, grads: &mut [Vec<f32>]) -> Result<f64, String> {
        let state = self.lora.as_mut().ok_or("gliner LoRA is not enabled")?;
        DebertaLayer::step_many(&mut state.layers, optimizer, grads).map_err(|e| e.message)
    }

    pub fn write_lora(&self, original: &[u8]) -> Result<Vec<u8>, String> {
        let state = self.lora.as_ref().ok_or("gliner LoRA is not enabled")?;
        let mut merged = original.to_vec();
        for (offset, layer) in state.layers.iter().enumerate() {
            merged = layer.write_merged(&merged, state.start + offset).map_err(|e| e.message)?;
        }
        Ok(merged)
    }
}
