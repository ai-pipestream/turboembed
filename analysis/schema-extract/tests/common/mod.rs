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
