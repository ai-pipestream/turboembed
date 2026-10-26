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

fn reps() -> usize {
    std::env::var("DEVICE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20)
}

#[test]
fn schema_extraction_scores_as_on_the_cpu_encoder() {
    let limits = Limits { max_words: 64, ..Limits::default() };
    let (cpu, on) = (common::on(cpu(), limits), common::on(device(), limits));
    let schema = schema();
    let (mut largest, mut held, mut decoded) = (0f32, Vec::new(), 0usize);
    for _ in 0..reps() {
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
            let (x, y) = (common::rows(&want), common::rows(&got));
            let (x, y): (Vec<&[f32]>, Vec<&[f32]>) =
                (x.iter().map(Vec::as_slice).collect(), y.iter().map(Vec::as_slice).collect());
            held.extend(same_decisions(&x, &y, BOUND, text));
            // Decoded at a threshold half the span scores pass, none at it.
            let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
            let span_scores = want.scores().iter().filter_map(|s| match s {
                Scores::Spans { logits, .. } => Some(logits.iter().map(|&v| sig(v)).collect::<Vec<_>>()),
                Scores::Labels(_) => None,
            });
            let t = midpoint(span_scores.flatten().collect());
            let d = |p| turbo_schema_extract::decode(p, &schema, t, &turbo_schema_extract::decode::Nothing).unwrap();
            let (ea, eb) = (d(&want), d(&got));
            assert_eq!(common::picked(&ea), common::picked(&eb), "{text}: decoded at {t}");
            decoded += ea.entities.iter().map(Vec::len).sum::<usize>() + ea.records.len();
        }
    }
    varied(&held, "schema decisions");
    assert!(decoded > 0, "nothing decoded on either side");
    let x = on.execution().unwrap();
    println!(
        "gliner2 schema extraction on {} {}: {} runs, largest logit difference from the cpu encoder {largest:e}, {} \
         decisions held, {decoded} spans and records decoded alike",
        x.backend,
        x.device,
        reps(),
        held.len()
    );
}
