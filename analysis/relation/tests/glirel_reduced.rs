//! The GLiREL heads against the reference's own layer classes, on the
//! reduced fixture `reference/glirel/build_fixture.py` writes: a seeded
//! random encoder (width 32) and heads (width 16) under a GLiREL
//! checkpoint's tensor names, fixed token sequences, and the reference's
//! projected rows, LSTM rows, entity representations, label
//! representations and pair logits. The tolerance follows the encoder's
//! own fixture.

use serde_json::Value;
use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo_relation::glirel::heads::Heads;
use turbo_relation::glirel::prompt::Prompt;
use turbo_relation::glirel::run;
use turbo_relation::glirel::words::Word;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glirel");
const PREFIX: &str = "token_rep_layer.bert_layer.model.";

fn config() -> Config {
    Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 800,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    }
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn ints(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

fn pairs(v: &Value) -> Vec<(usize, usize)> {
    v.as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)).collect()
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 2e-5 * w.abs().max(1.0), "{what}[{i}]: {g} against {w}");
    }
}

fn load() -> (Encoder, Heads) {
    let bytes = std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap();
    let canonical = turbo_bundle::deberta::convert(&config(), &bytes, PREFIX).unwrap();
    let encoder = Encoder::from_safetensors(config(), &canonical).unwrap();
    let heads = Heads::load(&File::parse("reduced", &bytes).unwrap(), 32).unwrap();
    (encoder, heads)
}

#[test]
fn the_heads_match_the_reference_layers() {
    let (encoder, heads) = load();
    let reference: Value = serde_json::from_slice(&std::fs::read(format!("{DIR}/reduced.json")).unwrap()).unwrap();
    for (c, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let ids: Vec<u32> = ints(&case["ids"]).into_iter().map(|i| i as u32).collect();
        let words = ints(&case["words"]);
        let labels: Vec<[usize; 2]> = pairs(&case["labels"]).into_iter().map(|(a, b)| [a, b]).collect();
        let spans = pairs(&case["spans"]);
        let hidden = encoder.encode(&ids, &vec![true; ids.len()]).unwrap();
        let projected = heads.project(hidden.data(), ids.len());
        close(&projected, &floats(&case["projected"]), &format!("case {c} projected"));
        let row = |p: usize| &projected[p * 16..(p + 1) * 16];
        let word_rows: Vec<f32> = words.iter().flat_map(|&p| row(p).iter().copied()).collect();
        let lstm = heads.lstm(&word_rows, words.len());
        close(&lstm, &floats(&case["lstm"]), &format!("case {c} lstm"));
        let entities = heads.entities(&lstm, &spans);
        close(&entities, &floats(&case["entities"]), &format!("case {c} entities"));
        let means: Vec<f32> =
            labels.iter().flat_map(|&[r, l]| row(r).iter().zip(row(l)).map(|(a, b)| (a + b) / 2.0)).collect();
        let types = heads.types(&means, labels.len());
        close(&types, &floats(&case["types"]), &format!("case {c} types"));
        let pairs = pairs(&case["pairs"]);
        let logits = heads.logits(&entities, &pairs, &types);
        close(&logits, &floats(&case["logits"]), &format!("case {c} logits"));
        // The same through the pass the scorer runs.
        let prompt = Prompt {
            ids,
            words: words.iter().map(|&p| Word { start: p, end: p + 1 }).collect(),
            word_first: words.iter().map(|&p| Some(p)).collect(),
            labels,
        };
        let pass = run(&encoder, &heads, prompt).unwrap();
        close(pass.lstm(), &floats(&case["lstm"]), &format!("case {c} pass lstm"));
        close(pass.types(), &floats(&case["types"]), &format!("case {c} pass types"));
    }
}

#[test]
fn heads_of_another_width_are_refused() {
    let bytes = std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap();
    let file = File::parse("reduced", &bytes).unwrap();
    assert!(Heads::load(&file, 48).is_err());
}
