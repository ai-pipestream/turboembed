//! Head training through the public API on the tiny CPU encoder, for every
//! combination of output form, boundary heads, prompt head and BiLSTM, and with a
//! prompt width that differs from the output width: the trainer loads, its
//! own gradient buffer trains, the tuned heads install, and the written
//! heads reload to the same scores.

use std::path::{Path, PathBuf};

use turbo::deberta::{Config as EncoderConfig, Encoder};
use turbo_analysis::ModelRef;
use turbo_gliner::train::{OutputTrainer, output_optimizer};
use turbo_gliner::{Candidate, Config, Decode, ExtractionSchema, Model};
use turbo_trainer::adamw::Settings;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn encoder_config() -> EncoderConfig {
    EncoderConfig {
        hidden: 32,
        intermediate: 64,
        heads: 4,
        layers: 2,
        vocabulary: 30522,
        max_sequence: 128,
        position_buckets: 8,
        max_relative_positions: 512,
        layer_norm_eps: 1e-7,
    }
}

fn config(out_project_mlp: bool, prompt_factor: usize) -> Config {
    Config {
        hidden: 16,
        projection_factor: 4,
        prompt_factor,
        out_project_mlp,
        token_level: false,
        prompt_projection: true,
        max_width: 3,
        entity_token: 30000,
        separator_token: 30001,
        cls_token: 101,
        eos_token: 102,
    }
}

/// A heads file of the configuration's own tensors, filled with small
/// values of both signs from a fixed sequence.
fn heads(config: &Config) -> Vec<u8> {
    heads_with_projection_width(config, encoder_config().hidden)
}

fn heads_with_projection_width(config: &Config, projection_width: usize) -> Vec<u8> {
    let mut state = 0x2545_f491u32;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, mut shape) in config.tensor_shapes(&encoder_config()).unwrap() {
        if name == "token_rep_layer.projection.weight" {
            shape[1] = projection_width as u64;
        }
        let start = data.len();
        for _ in 0..shape.iter().product::<u64>() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.4;
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(name, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    out
}

fn model(config: &Config, heads: &[u8]) -> Model {
    let bytes = std::fs::read(root().join("testdata/tiny-deberta-bundle/weights/model.safetensors")).unwrap();
    let encoder = Encoder::from_safetensors(encoder_config(), &bytes).unwrap().with_threads(1).unwrap();
    let spm = std::fs::read(root().join("testdata/sentencepiece/unigram-identity.model")).unwrap();
    let identity = ModelRef { id: "turboembed/tiny-deberta".into(), revision: "1".into(), sha256: String::new() };
    Model::new(encoder, config.clone(), heads, &spm, identity).unwrap()
}

const WORDS: [&str; 6] = ["Alice", "visited", "Paris", "in", "May", "."];
const GOLD: [(usize, usize, usize); 2] = [(0, 1, 0), (2, 3, 1)];

fn schema() -> ExtractionSchema {
    ExtractionSchema::new(vec!["person".into(), "place".into()], 0.5, Decode::Flat).unwrap()
}

fn scores(model: &Model) -> Vec<Candidate> {
    model.training_input(&WORDS, &schema()).unwrap().1
}

/// One optimizer step of the chosen heads, installed and reloaded.
fn train_install_reload(config: &Config, boundaries: bool, prompt: bool, recurrent: bool, projection: bool) {
    let label = format!(
        "mlp {} prompt factor {} boundaries {boundaries} prompt {prompt} recurrent {recurrent} projection {projection}",
        config.out_project_mlp, config.prompt_factor
    );
    let original = heads(config);
    let mut tuned = model(config, &original);
    let before = scores(&tuned);
    let mut trainer =
        OutputTrainer::load_with_heads(config, &original, boundaries, prompt, recurrent, projection).expect(&label);
    let output = if config.out_project_mlp { 4 } else { 2 };
    let dense = output + if boundaries { 8 } else { 0 } + if prompt { 4 } else { 0 };
    let expected = dense + if recurrent { 8 } else { 0 } + if projection { 2 } else { 0 };
    assert_eq!(trainer.params().len(), expected, "{label}");
    if prompt {
        let first = &trainer.params()[dense - 4];
        assert_eq!(first.len(), config.prompt_factor * config.hidden * config.hidden, "{label}");
    }

    let (input, candidates) = tuned.training_input(&WORDS, &schema()).unwrap();
    let mut grads = trainer.zero_grads();
    let loss = trainer.gradients(&input, &candidates, &GOLD, &mut grads).expect(&label);
    assert!(loss.is_finite() && loss > 0.0, "{label}: {loss}");
    for (at, tensor) in grads.iter().enumerate() {
        assert!(tensor.iter().any(|&g| g != 0.0), "{label}: tensor {at} has no gradient");
    }
    let settings = Settings { lr: 1e-2, total: 1, ..Settings::default() };
    let mut optimizer = output_optimizer(settings, &trainer).unwrap();
    trainer.step(&mut optimizer, &mut grads).unwrap();

    tuned.apply_output(&trainer).expect(&label);
    let after = scores(&tuned);
    assert_ne!(after, before, "{label}: the step changed no score");
    let written = trainer.write_heads(&original).expect(&label);
    assert_ne!(written, original, "{label}");
    assert_eq!(scores(&model(config, &written)), after, "{label}: written heads reload to other scores");
}

#[test]
fn every_head_combination_trains_installs_and_reloads() {
    for out_project_mlp in [true, false] {
        for boundaries in [false, true] {
            for prompt in [false, true] {
                train_install_reload(&config(out_project_mlp, 4), boundaries, prompt, false, false);
                if boundaries {
                    train_install_reload(&config(out_project_mlp, 4), boundaries, prompt, true, false);
                }
            }
        }
        train_install_reload(&config(out_project_mlp, 4), true, true, true, true);
    }
}

#[test]
fn a_prompt_width_other_than_the_output_width_trains() {
    // Output and boundary factor four, prompt factor two.
    for out_project_mlp in [true, false] {
        for boundaries in [false, true] {
            train_install_reload(&config(out_project_mlp, 2), boundaries, true, boundaries, false);
        }
        train_install_reload(&config(out_project_mlp, 2), true, true, true, true);
    }
}

#[test]
fn a_trainer_of_another_form_is_not_installed() {
    let (mlp, linear) = (config(true, 4), config(false, 4));
    let trainer = OutputTrainer::load_with_heads(&linear, &heads(&linear), true, true, true, false).unwrap();
    let mut other = model(&mlp, &heads(&mlp));
    assert!(other.apply_output(&trainer).is_err());
    // A prompt trainer of another prompt width is refused too.
    let narrow = config(true, 2);
    let trainer = OutputTrainer::load_with_heads(&narrow, &heads(&narrow), false, true, false, false).unwrap();
    assert!(other.apply_output(&trainer).unwrap_err().contains("does not match"));
}

#[test]
fn projection_widths_are_checked_before_installation() {
    let config = config(true, 4);
    let zero = heads_with_projection_width(&config, 0);
    let error = OutputTrainer::load_with_heads(&config, &zero, true, true, true, true).map(|_| ()).unwrap_err();
    assert!(error.contains("width must be positive"), "{error}");

    let narrow = heads_with_projection_width(&config, config.hidden);
    let trainer = OutputTrainer::load_with_heads(&config, &narrow, true, true, true, true).unwrap();
    let mut tuned = model(&config, &heads(&config));
    let before = scores(&tuned);
    assert!(tuned.apply_output(&trainer).unwrap_err().contains("shape mismatch"));
    assert_eq!(scores(&tuned), before);
}

#[test]
fn recurrent_training_needs_boundaries_and_an_even_hidden_size() {
    let even = config(true, 4);
    let bytes = heads(&even);
    let refused = |config: &Config, boundaries| {
        OutputTrainer::load_with_heads(config, &bytes, boundaries, false, true, false).map(|_| ()).unwrap_err()
    };
    assert!(refused(&even, false).contains("requires boundary training"));
    let odd = Config { hidden: 15, ..even.clone() };
    assert!(refused(&odd, true).contains("even hidden size"));
    let projection = OutputTrainer::load_with_heads(&even, &bytes, false, false, false, true).map(|_| ()).unwrap_err();
    assert!(projection.contains("requires recurrent and prompt training"));
}
