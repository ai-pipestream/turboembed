//! GLiNER2 schema extraction with the encoder through the backend table
//! against the CPU encoder (#920): the reduced heads and classifier
//! (width 32) over the tiny DeBERTa-v2 bundle, every instance count equal
//! and every span and label logit within BOUND, on the device the
//! environment names (the relation crate's device_support).

mod common;
#[path = "../../relation/tests/device_support/mod.rs"]
mod device_support;

use device_support::*;
use turbo_schema_extract::model::Scores;
use turbo_schema_extract::schema::Limits;
use turbo_schema_extract::{
    Cardinality, ClassificationTask, EntityType, ExtractionSchema, FieldSpec, Label, StructureSpec,
};

fn schema() -> ExtractionSchema {
    let entity = |name: &str| EntityType {
        name: name.into(),
        description: None,
        threshold: None,
        cardinality: Cardinality::Repeated,
        checks: Vec::new(),
    };
    let field = |name: &str| FieldSpec {
        name: name.into(),
        description: None,
        threshold: None,
        cardinality: Cardinality::Repeated,
        choices: None,
        checks: Vec::new(),
    };
    ExtractionSchema {
        entities: vec![entity("person"), entity("place")],
        classifications: vec![ClassificationTask {
            name: "tone".into(),
            labels: vec![
                Label { name: "calm".into(), description: None },
                Label { name: "urgent".into(), description: None },
            ],
            multi_label: true,
            threshold: 0.0,
            top_k: None,
        }],
        structures: vec![StructureSpec { name: "venture".into(), fields: vec![field("partner"), field("city")] }],
    }
}

#[test]
fn schema_extraction_scores_as_on_the_cpu_encoder() {
    let limits = Limits { max_words: 64, ..Limits::default() };
    let (cpu, on) = (common::on(cpu(), limits), common::on(device(), limits));
    let schema = schema();
    let mut largest = 0f32;
    for text in TEXTS {
        let (want, got) = (cpu.pass(text, &schema).unwrap(), on.pass(text, &schema).unwrap());
        assert_eq!(want.prompt().ids, got.prompt().ids, "{text}: ids");
        assert_eq!(want.scores().len(), got.scores().len());
        for (t, (a, b)) in want.scores().iter().zip(got.scores()).enumerate() {
            let d = match (a, b) {
                (Scores::Labels(a), Scores::Labels(b)) => worst(a, b, text),
                (
                    Scores::Spans { count: c, instances: i, logits: a },
                    Scores::Spans { count: d, instances: j, logits: b },
                ) => {
                    assert_eq!((c, i), (d, j), "{text}: task {t}: counts");
                    worst(a, b, text)
                }
                _ => panic!("{text}: task {t}: scores of different kinds"),
            };
            assert!(d <= BOUND, "{text}: task {t}: logits {d}");
            largest = largest.max(d);
        }
    }
    let x = on.execution().unwrap();
    println!(
        "gliner2 schema extraction on {} {}: largest logit difference from the cpu encoder {largest:e}",
        x.backend, x.device
    );
}
