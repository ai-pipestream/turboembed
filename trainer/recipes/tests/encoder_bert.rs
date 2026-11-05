//! BERT-family bundles through the trainer and their export back.

use std::collections::BTreeMap;
use std::path::Path;

use turbo::bundle::Bundle;
use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Format, GraphInput, WordPooling};
use turbo::safetensors::{Dtype, File};
use turbo_tagger::WordEncoder;
use turbo_trainer::encoder::{Encoder, Work, pool};

fn check(dir: &Path, layers: usize, hidden: usize) {
    let bundle = Bundle::open(dir).unwrap();
    let encoder = Encoder::<f32>::load(dir).unwrap();
    let mut words = WordEncoder::open(dir, &Default::default()).unwrap();
    let forms = ["they", "walked", "dogs", "."];
    let refs: Vec<&str> = forms.to_vec();
    let expected = words.features(&[refs], WordPooling::First).unwrap().remove(0);

    let tokenizer = turbo::tokenizer::Tokenizer::load(&bundle).unwrap();
    let (cls, sep) = tokenizer.template_ends();
    let mut ids: Vec<u32> = cls.iter().map(|&id| id as u32).collect();
    let mut map = vec![NO_WORD; ids.len()];
    for (i, form) in forms.iter().enumerate() {
        let pieces = words.pieces(form).unwrap();
        map.extend(std::iter::repeat_n(i as u32, pieces.len()));
        ids.extend(pieces.into_iter().map(|id| id as u32));
    }
    map.extend(std::iter::repeat_n(NO_WORD, sep.len()));
    ids.extend(sep.iter().map(|&id| id as u32));
    let mut work = Work::default();
    encoder.forward(&ids, &mut work).unwrap();
    let (mut actual, mut count) = (Vec::new(), Vec::new());
    pool::forward(&work.out, &map, hidden, WordPooling::First, &mut actual, &mut count).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() < 0.000_02, "feature {i}: {a} vs {b}");
    }

    let artifact = bundle
        .manifest
        .artifacts
        .iter()
        .find(|a| a.format == Format::Safetensors && a.graph_input == GraphInput::TokenIds)
        .unwrap();
    let path = artifact.files.first().unwrap();
    let bytes = bundle.read_verified(path).unwrap();
    let file = File::parse(path, &bytes).unwrap();
    let exported: BTreeMap<String, (Vec<u64>, Vec<f32>)> = encoder
        .named(&artifact.tensor_names)
        .unwrap()
        .into_iter()
        .map(|(name, shape, values)| (name, (shape, values)))
        .collect();
    assert_eq!(exported.len(), 5 + layers * 16);
    for (name, (shape, values)) in exported {
        let tensor = file.get(&name, Dtype::F32, shape.len()).unwrap();
        assert_eq!(tensor.shape, shape, "{name}");
        assert_eq!(tensor.f32s(), values, "{name}");
    }
}

#[test]
fn bert_states_match_the_core_and_export_reproduces_every_tensor() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    check(&dir, 2, 32);
    if let Some(dir) = std::env::var_os("TURBO_TEST_XLM_BUNDLE").map(turbo_pins::require) {
        check(Path::new(&dir), 12, 768);
    }
}
