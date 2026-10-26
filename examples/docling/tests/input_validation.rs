use serde_json::{Value, json};
use turbo_example_docling::docling::parse;

fn input() -> Value {
    serde_json::from_str(include_str!("../data/report.docling.json")).unwrap()
}

#[test]
fn maximum_heading_level_returns_error_without_panicking() {
    let mut d = input();
    d["texts"][2]["level"] = json!(u64::MAX);
    let result = std::panic::catch_unwind(|| parse(&d.to_string()));
    assert!(result.is_ok(), "untrusted heading level panicked");
    assert!(result.unwrap().is_err());
}

#[test]
fn malformed_body_children_are_not_an_empty_document() {
    let mut d = input();
    d["body"]["children"] = json!({"$ref": "#/texts/1"});
    assert!(parse(&d.to_string()).is_err(), "invalid children silently lost the whole document");
}

#[test]
fn unsupported_descendant_is_not_silently_dropped() {
    let mut d = input();
    d["texts"][3]["children"] = json!([{"$ref": "#/tables/0"}]);
    d["tables"] = json!([{"label": "table", "children": []}]);
    assert!(parse(&d.to_string()).is_err(), "a text child table was silently dropped");
}

#[test]
fn missing_body_and_invalid_levels_are_errors() {
    for level in [json!(-1), json!(1.5), json!("2"), json!(255), json!(null)] {
        let mut d = input();
        d["texts"][2]["level"] = level;
        assert!(parse(&d.to_string()).is_err());
    }
    for body in [json!(null), json!([]), json!({})] {
        let mut d = input();
        d["body"] = body;
        assert!(parse(&d.to_string()).is_err());
    }
    let mut d = input();
    d["texts"][2]["level"] = json!(254);
    assert!(parse(&d.to_string()).is_ok());
}

#[test]
fn repeated_and_cyclic_references_are_errors() {
    let mut d = input();
    d["body"]["children"].as_array_mut().unwrap().push(json!({"$ref":"#/texts/1"}));
    let error = parse(&d.to_string()).err().unwrap();
    assert!(error.contains("#/texts/1"));
    let mut d = input();
    d["groups"][0]["label"] = json!("section");
    d["groups"][0]["children"] = json!([{"$ref":"#/groups/0"}]);
    let error = parse(&d.to_string()).err().unwrap();
    assert!(error.contains("cyclic"), "{error}");
}

#[test]
fn reference_depth_is_bounded_without_rejecting_small_wrapper_trees() {
    let nested = |count: usize| {
        let mut d = input();
        d["body"]["children"] = json!([{"$ref":"#/groups/0"}]);
        d["groups"] = Value::Array(
            (0..count)
                .map(|i| {
                    let next = if i + 1 == count { "#/texts/3".to_owned() } else { format!("#/groups/{}", i + 1) };
                    json!({"label":"section", "children":[{"$ref":next}]})
                })
                .collect(),
        );
        d
    };
    let (parsed, _) = parse(&nested(2).to_string()).unwrap();
    assert_eq!(parsed.items.len(), 1);
    assert_eq!(parsed.text, input()["texts"][3]["text"].as_str().unwrap());
    let error = parse(&nested(129).to_string()).err().unwrap();
    assert!(error.contains("nesting"), "{error}");
}
