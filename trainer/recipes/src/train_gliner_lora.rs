use std::collections::BTreeMap;

use turbo_gliner::Model;
use turbo_gliner::train::OutputTrainer;
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::{Error, Result, gliner_eval};

pub(super) type Gradients = (f64, Vec<Vec<f32>>, Option<Vec<Vec<f32>>>);

pub(super) fn sentence(
    model: &Model,
    trainer: &OutputTrainer,
    sentence: &Sentence,
    labels: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
) -> Result<Gradients> {
    let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
    let mut grads = trainer.zero_grads();
    let mut lora = if model.lora_enabled() { Some(model.lora_zero_grads().map_err(Error)?) } else { None };
    let mut loss = 0.0;
    for batch in gliner_eval::batches(model, &words, labels, threshold, guidelines)? {
        let schema = gliner_eval::schema(&batch, threshold, guidelines)?;
        let gold = super::train_gliner::gold_spans(sentence, &batch)?;
        if let Some(encoder) = lora.as_mut() {
            let input = model.training_input_lora(&words, &schema).map_err(Error)?;
            let (part, dencoder) =
                trainer.gradients_with_encoder(&input.training, &input.candidates, &gold, &mut grads).map_err(Error)?;
            model.lora_backward(&input, &dencoder, encoder).map_err(Error)?;
            loss += part;
        } else {
            let (input, candidates) = model.training_input(&words, &schema).map_err(Error)?;
            loss += trainer.gradients(&input, &candidates, &gold, &mut grads).map_err(Error)?;
        }
    }
    Ok((loss, grads, lora))
}
