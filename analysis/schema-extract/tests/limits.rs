//! Every schema the extractor refuses, by the named limit or rule it
//! breaks, before anything is tokenized; and the requests refused by a
//! named limit once the prompt is built.

mod common;

use turbo_schema_extract::model::ExtractError;
use turbo_schema_extract::{
    Cardinality, Check, ClassificationTask, EntityType, ExtractionSchema, FieldSpec, Label, Limits, SchemaError,
    Source, StructureSpec,
};

fn entity(name: &str) -> EntityType {
    EntityType {
        name: name.into(),
        description: None,
        threshold: None,
        cardinality: Cardinality::Repeated,
        checks: Vec::new(),
    }
}

fn label(name: &str) -> Label {
    Label { name: name.into(), description: None }
}

fn task(name: &str, labels: &[&str]) -> ClassificationTask {
    ClassificationTask {
        name: name.into(),
        labels: labels.iter().map(|l| label(l)).collect(),
        multi_label: false,
        threshold: 0.5,
        top_k: None,
    }
}

fn field(name: &str) -> FieldSpec {
    FieldSpec {
        name: name.into(),
        description: None,
        threshold: None,
        cardinality: Cardinality::Single,
        choices: None,
        checks: Vec::new(),
    }
}

fn record(name: &str, fields: &[&str]) -> StructureSpec {
    StructureSpec { name: name.into(), fields: fields.iter().map(|f| field(f)).collect() }
}

fn entities(n: usize) -> ExtractionSchema {
    ExtractionSchema { entities: (0..n).map(|i| entity(&format!("e{i}"))).collect(), ..Default::default() }
}

fn over(e: Result<(), SchemaError>) -> &'static str {
    match e {
        Err(SchemaError::OverLimit { limit, .. }) => limit,
        other => panic!("not a named limit: {other:?}"),
    }
}

#[test]
fn every_named_schema_limit_is_refused_by_name() {
    let l = Limits::default();
    assert_eq!(entities(l.max_entity_types).validate(&l), Ok(()));
    assert_eq!(over(entities(l.max_entity_types + 1).validate(&l)), "max_entity_types");
    let tasks = |n: usize| ExtractionSchema {
        classifications: (0..n).map(|i| task(&format!("t{i}"), &["a"])).collect(),
        ..Default::default()
    };
    assert_eq!(over(tasks(l.max_classification_tasks + 1).validate(&l)), "max_classification_tasks");
    let names: Vec<String> = (0..=l.max_labels_per_task).map(|i| format!("l{i}")).collect();
    let many = ExtractionSchema {
        classifications: vec![task("t", &names.iter().map(String::as_str).collect::<Vec<_>>())],
        ..Default::default()
    };
    assert_eq!(over(many.validate(&l)), "max_labels_per_task");
    let records = |n: usize| ExtractionSchema {
        structures: (0..n).map(|i| record(&format!("r{i}"), &["f"])).collect(),
        ..Default::default()
    };
    assert_eq!(over(records(l.max_structures + 1).validate(&l)), "max_structures");
    let fields: Vec<String> = (0..=l.max_fields_per_structure).map(|i| format!("f{i}")).collect();
    let wide = ExtractionSchema {
        structures: vec![record("r", &fields.iter().map(String::as_str).collect::<Vec<_>>())],
        ..Default::default()
    };
    assert_eq!(over(wide.validate(&l)), "max_fields_per_structure");
    let long = ExtractionSchema { entities: vec![entity(&"x".repeat(l.max_text_bytes + 1))], ..Default::default() };
    assert_eq!(over(long.validate(&l)), "max_text_bytes");
    let mut choices = record("r", &["f"]);
    choices.fields[0].choices = Some(names.clone());
    assert_eq!(
        over(ExtractionSchema { structures: vec![choices], ..Default::default() }.validate(&l)),
        "max_labels_per_task"
    );
}

#[test]
fn malformed_schemas_are_refused_by_rule() {
    let l = Limits::default();
    let check = |s: ExtractionSchema| s.validate(&l).unwrap_err();
    assert_eq!(check(ExtractionSchema::default()), SchemaError::Empty);
    assert!(matches!(
        check(ExtractionSchema { entities: vec![entity(" ")], ..Default::default() }),
        SchemaError::Blank(_)
    ));
    assert!(matches!(
        check(ExtractionSchema { entities: vec![entity("a"), entity("a")], ..Default::default() }),
        SchemaError::Duplicate(_)
    ));
    assert!(matches!(
        check(ExtractionSchema { entities: vec![entity("a [P] b")], ..Default::default() }),
        SchemaError::Reserved(_)
    ));
    let described = EntityType { description: Some("see [DESCRIPTION]".into()), ..entity("a") };
    assert!(matches!(
        check(ExtractionSchema { entities: vec![described], ..Default::default() }),
        SchemaError::Reserved(_)
    ));
    for t in [f32::NAN, -0.1, 1.1, f32::INFINITY] {
        let e = EntityType { threshold: Some(t), ..entity("a") };
        assert!(
            matches!(check(ExtractionSchema { entities: vec![e], ..Default::default() }), SchemaError::BadValue(_)),
            "{t}"
        );
        let boost = EntityType { checks: vec![Check::Boost(Source::Pii, t)], ..entity("a") };
        assert!(
            matches!(check(ExtractionSchema { entities: vec![boost], ..Default::default() }), SchemaError::BadValue(_)),
            "{t}"
        );
    }
    let top = ClassificationTask { top_k: Some(0), ..task("t", &["a"]) };
    assert!(matches!(
        check(ExtractionSchema { classifications: vec![top], ..Default::default() }),
        SchemaError::BadValue(_)
    ));
    assert!(matches!(
        check(ExtractionSchema { classifications: vec![task("t", &[])], ..Default::default() }),
        SchemaError::Blank(_)
    ));
    assert!(matches!(
        check(ExtractionSchema { structures: vec![record("r", &[])], ..Default::default() }),
        SchemaError::Blank(_)
    ));
    let both = ExtractionSchema {
        classifications: vec![task("x", &["a"])],
        structures: vec![record("x", &["f"])],
        ..Default::default()
    };
    assert!(matches!(check(both), SchemaError::Duplicate(_)), "a task and a record share a name");
    // Names the layers join must not collide: `a` + `b.c` and `a.b` + `c`
    // are both `a.b.c`; `t` + `x: y` and `t: x` + `y` both `t: x: y`.
    let dotted =
        ExtractionSchema { structures: vec![record("a", &["b.c"]), record("a.b", &["c"])], ..Default::default() };
    assert!(matches!(check(dotted), SchemaError::Duplicate(_)), "two fields named a.b.c");
    let colons =
        ExtractionSchema { classifications: vec![task("t", &["x: y"]), task("t: x", &["y"])], ..Default::default() };
    assert!(matches!(check(colons), SchemaError::Duplicate(_)), "two labels named t: x: y");
    // A check on a closed-choice field could never apply.
    let mut checked = record("r", &["f"]);
    checked.fields[0].choices = Some(vec!["a".into()]);
    checked.fields[0].checks = vec![Check::Require(Source::Numbers)];
    assert!(matches!(
        check(ExtractionSchema { structures: vec![checked], ..Default::default() }),
        SchemaError::BadValue(_)
    ));
    // Choices that differ only in case are one choice given twice.
    let mut cased = record("r", &["f"]);
    cased.fields[0].choices = Some(vec!["Yes".into(), "yes".into()]);
    assert!(matches!(
        check(ExtractionSchema { structures: vec![cased], ..Default::default() }),
        SchemaError::Duplicate(_)
    ));
    let entities_record = ExtractionSchema { structures: vec![record("entities", &["f"])], ..Default::default() };
    assert!(matches!(check(entities_record), SchemaError::Reserved(_)), "a record named as the entity task");
}

#[test]
fn requests_over_the_tokenized_limits_are_refused_by_name() {
    // The schema prompt of one entity type, measured; then a limit at it.
    let wide = common::offline(Limits::default());
    let one = entities(1);
    let prompt = wide.prompt("a", &one).unwrap();
    let schema_len = prompt.token_first[0];
    let small = Limits { max_words: 5, max_prompt_subwords: schema_len, ..Limits::default() };
    let m = common::offline(small);
    let limit = |r: Result<_, ExtractError>| match r {
        Err(ExtractError::OverLimit { limit, .. }) => limit,
        Err(e) => panic!("not a named limit: {e}"),
        Ok(_) => panic!("accepted"),
    };
    // Five words without final punctuation: the `.` the model adds is not
    // the caller's sixth word.
    assert!(m.prompt("a b c d e", &one).is_ok());
    assert_eq!(limit(m.prompt("a b c d e f g", &one)), "max_words");
    assert_eq!(limit(m.prompt("a", &entities(4))), "max_prompt_subwords");
    let text = "word ".repeat(600);
    assert_eq!(limit(wide.prompt(&text, &one)), "max_words");
    let text = "abcdefgh ".repeat(200);
    assert_eq!(limit(wide.prompt(&text, &one)), "encoder max_sequence");
    assert!(matches!(m.prompt("a", &ExtractionSchema::default()), Err(ExtractError::Schema(SchemaError::Empty))));
}

#[test]
fn choices_differing_only_by_case_folding_are_duplicates() {
    let l = Limits::default();
    let mut cased = record("r", &["f"]);
    cased.fields[0].choices = Some(vec!["Stra\u{00DF}e".into(), "STRASSE".into()]);
    let schema = ExtractionSchema { structures: vec![cased], ..Default::default() };
    assert!(schema.validate(&l).is_err(), "folded duplicates accepted");
}
