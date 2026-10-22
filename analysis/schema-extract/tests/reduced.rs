//! The heads schema extraction adds to the relation scorer's, against
//! GLiNER2's own layer classes on the reduced fixtures
//! `reference/build_fixture.py` writes: the count queries across three and
//! four fields (the relation scorer only ever has two), for one and three
//! instances, and the classifier's logits. The encoder and span heads are
//! the relation scorer's reduced fixture; the classifier is sliced to the
//! same width 32.

use serde_json::Value;
use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo_relation::gliner2::heads::Heads;
use turbo_schema_extract::model::Classifier;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const RELATION: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../relation/tests/fixtures/gliner2/reduced.safetensors");

fn config() -> Config {
    Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 16,
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

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 2e-5 * w.abs().max(1.0), "{what}[{i}]: {g} against {w}");
    }
}

#[test]
fn queries_over_many_fields_and_the_classifier_match_the_reference() {
    let bytes = std::fs::read(RELATION).unwrap();
    let canonical = turbo_bundle::deberta::convert(&config(), &bytes, "encoder.").unwrap();
    let encoder = Encoder::from_safetensors(config(), &canonical).unwrap();
    let heads = Heads::load(&File::parse("reduced", &bytes).unwrap(), 32, 8).unwrap();
    let classifier_bytes = std::fs::read(format!("{DIR}/extra.safetensors")).unwrap();
    let classifier = Classifier::load(&File::parse("classifier", &classifier_bytes).unwrap(), 32).unwrap();
    let reference: Value = serde_json::from_slice(&std::fs::read(format!("{DIR}/reduced.json")).unwrap()).unwrap();
    for (c, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let ids: Vec<u32> = ints(&case["ids"]).into_iter().map(|i| i as u32).collect();
        let hidden = encoder.encode(&ids, &vec![true; ids.len()]).unwrap();
        let row = |p: usize| &hidden.data()[p * 32..(p + 1) * 32];
        let fields: Vec<&[f32]> = ints(&case["fields"]).into_iter().map(row).collect();
        for k in [1, 3] {
            let got = heads.queries_of(&fields, k);
            close(
                &got,
                &floats(&case["queries"][k.to_string()]),
                &format!("case {c}, {} fields, {k} instances", fields.len()),
            );
        }
        let logits: Vec<f32> = ints(&case["rows"]).into_iter().map(|r| classifier.logit(row(r))).collect();
        close(&logits, &floats(&case["logits"]), &format!("case {c} classifier"));
    }
}

#[test]
fn a_classifier_of_another_width_is_refused() {
    let bytes = std::fs::read(format!("{DIR}/extra.safetensors")).unwrap();
    assert!(Classifier::load(&File::parse("classifier", &bytes).unwrap(), 48).is_err());
}
