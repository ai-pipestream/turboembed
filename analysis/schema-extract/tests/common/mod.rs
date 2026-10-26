//! The fixtures' schema description as an `ExtractionSchema`, and the
//! reduced model the offline tests share.

use serde_json::Value;
use turbo_schema_extract::{
    Cardinality, ClassificationTask, EntityType, ExtractionSchema, FieldSpec, Label, StructureSpec,
};

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

fn cardinality(v: &Value) -> Cardinality {
    if v.get("single").and_then(Value::as_bool) == Some(true) { Cardinality::Single } else { Cardinality::Repeated }
}

fn threshold(v: &Value) -> Option<f32> {
    v.get("threshold").and_then(Value::as_f64).map(|t| t as f32)
}

fn list<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v.get(k).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// The schema `build_fixture.py` describes with `spec`.
#[allow(dead_code)]
pub fn schema(spec: &Value) -> ExtractionSchema {
    ExtractionSchema {
        entities: list(spec, "entities")
            .iter()
            .map(|e| EntityType {
                name: s(e, "name").unwrap(),
                description: s(e, "description"),
                threshold: threshold(e),
                cardinality: cardinality(e),
                checks: Vec::new(),
            })
            .collect(),
        classifications: list(spec, "classifications")
            .iter()
            .map(|c| ClassificationTask {
                name: s(c, "task").unwrap(),
                labels: list(c, "labels")
                    .iter()
                    .map(|l| Label { name: s(l, "name").unwrap(), description: s(l, "description") })
                    .collect(),
                multi_label: c.get("multi").and_then(Value::as_bool).unwrap_or(false),
                threshold: threshold(c).unwrap_or(0.5),
                top_k: c.get("top_k").and_then(Value::as_u64).map(|k| k as usize),
            })
            .collect(),
        structures: list(spec, "structures")
            .iter()
            .map(|st| StructureSpec {
                name: s(st, "name").unwrap(),
                fields: list(st, "fields")
                    .iter()
                    .map(|f| FieldSpec {
                        name: s(f, "name").unwrap(),
                        description: s(f, "description"),
                        threshold: threshold(f),
                        cardinality: cardinality(f),
                        choices: f
                            .get("choices")
                            .and_then(Value::as_array)
                            .map(|c| c.iter().map(|x| x.as_str().unwrap().to_owned()).collect()),
                        checks: Vec::new(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::ModelRef;
use turbo_relation::gliner2::heads::Heads;
use turbo_schema_extract::model::Classifier;
use turbo_schema_extract::prompt::Specials;
use turbo_schema_extract::{Gliner2Extractor, Limits};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const RELATION: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../relation/tests/fixtures/gliner2/reduced.safetensors");
const SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/sentencepiece/unigram-identity.model");

/// `base` with every tensor of `over` in place of its namesake, as one
/// safetensors file.
fn overlay(base: &[u8], over: &[u8]) -> Vec<u8> {
    let (b, o) = (File::parse("base", base).unwrap(), File::parse("over", over).unwrap());
    let names_of = |bytes: &[u8]| -> Vec<String> {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        header.keys().filter(|k| *k != "__metadata__").cloned().collect()
    };
    let mut names: Vec<String> = names_of(base).into_iter().chain(names_of(over)).collect();
    names.sort();
    names.dedup();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for n in names {
        let t = o.tensor(&n).or_else(|| b.tensor(&n)).unwrap();
        let start = data.len();
        data.extend_from_slice(t.data);
        header.insert(n, serde_json::json!({"dtype": "F32", "shape": t.shape, "data_offsets": [start, data.len()]}));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}

/// The reduced GLiNER2 model (width 32) with 800 word embeddings, reading
/// real text through `testdata/sentencepiece/unigram-identity.model`. Its
/// outputs mean nothing; it runs every path the checkpoint runs.
#[allow(dead_code)]
pub fn offline(limits: Limits) -> Gliner2Extractor {
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 800,
        max_sequence: 512,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let canonical = turbo_bundle::deberta::convert(&config, &reduced(), "encoder.").unwrap();
    on(Encoder::from_safetensors(config, &canonical).unwrap(), limits)
}

/// The reduced checkpoint with the classifier laid over it.
fn reduced() -> Vec<u8> {
    let extra = std::fs::read(format!("{FIXTURES}/extra.safetensors")).unwrap();
    overlay(&std::fs::read(RELATION).unwrap(), &extra)
}

/// The reduced model's heads, classifier and tokenizer over another
/// encoder of width 32 (tests/device.rs), its special tokens 790 to 796.
#[allow(dead_code)]
pub fn on(encoder: impl Into<turbo_relation::encoder::DebertaEncoder>, limits: Limits) -> Gliner2Extractor {
    let bytes = reduced();
    let file = File::parse("reduced", &bytes).unwrap();
    let heads = Heads::load(&file, 32, 8).unwrap();
    let classifier = Classifier::load(&file, 32).unwrap();
    let tokenizer = SentencePiece::parse(&std::fs::read(SPM).unwrap()).unwrap();
    let specials = Specials { sep_struct: 790, sep_text: 791, p: 792, c: 793, e: 794, l: 795, description: 796 };
    let model = ModelRef { id: "reduced".into(), revision: "sliced".into(), sha256: "none".into() };
    Gliner2Extractor::new(encoder, heads, classifier, tokenizer, specials, model, limits).unwrap()
}
