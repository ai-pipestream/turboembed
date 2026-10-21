//! Schema-driven extraction against GLiNER2 itself on the full
//! `fastino/gliner2-base-v1` checkpoint: the input ids, each span task's
//! instance count, each classification's logits, and the decoded entities,
//! classifications and records with their confidences and byte offsets,
//! from `tests/fixtures/real.json`, which `reference/build_fixture.py`
//! writes with GLiNER2's own `extract`.
//!
//! The checkpoint (Apache-2.0, 834 MB) is fetched through `turbo-zoo` by
//! digest into the zoo's home on first use, and its encoder packaged beside
//! it as the relation scorer's tests package it. Run it with
//! `cargo test --release -p turbo-schema-extract --test real -- --ignored`.

mod common;

use std::path::{Path, PathBuf};

use serde_json::Value;
use turbo_schema_extract::decode::{FieldValue, Nothing};
use turbo_schema_extract::model::Scores;
use turbo_schema_extract::{Cardinality, Gliner2Extractor, decode};
use turbo_zoo::{Acceptances, Http, Index, Policy, Zoo};

const ID: &str = "fastino-gliner2-base-v1";
const REPOSITORY: &str = "fastino/gliner2-base-v1";
const REVISION: &str = "f9634218e53580c56edf0de97ca1a7d3f1c2354e";

fn home() -> PathBuf {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    var("TURBO_ZOO_HOME")
        .or_else(|| var("XDG_DATA_HOME").map(|x| x.join("turbo-zoo")))
        .or_else(|| var("HOME").map(|h| h.join(".local/share/turbo-zoo")))
        .expect("a zoo home")
}

pub fn model() -> Gliner2Extractor {
    static ONCE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _held = ONCE.lock().unwrap_or_else(|e| e.into_inner());
    let home = home();
    let zoo = Zoo::new(Index::builtin(), home.join("artifacts"), Policy::default(), &Http);
    let user = std::env::var("USER").unwrap_or_else(|_| "test".into()).parse().unwrap();
    let dir = zoo.fetch_with(ID, &user, || Acceptances::load(&home.join("acceptances.json"))).unwrap().dir.clone();
    let package = home.join("packages").join(format!("{ID}-encoder"));
    if !package.join("deberta.json").exists() {
        std::fs::create_dir_all(package.parent().unwrap()).unwrap();
        let config: Value =
            serde_json::from_slice(&std::fs::read(dir.join("encoder_config/config.json")).unwrap()).unwrap();
        let config = turbo_bundle::deberta::from_huggingface(&config).unwrap();
        let checkpoint = std::fs::read(dir.join("model.safetensors")).unwrap();
        let staging = package.with_extension(format!("tmp-{}", std::process::id()));
        turbo_bundle::deberta::write_package(
            config,
            &checkpoint,
            "encoder.",
            (REPOSITORY, REVISION, "Apache-2.0"),
            &staging,
        )
        .unwrap();
        if std::fs::rename(&staging, &package).is_err() && package.join("deberta.json").exists() {
            std::fs::remove_dir_all(&staging).unwrap();
        }
    }
    Gliner2Extractor::load(&package, &dir, REPOSITORY, REVISION).unwrap()
}

fn reference() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn close(got: f32, want: &Value, what: &str) {
    let w = want.as_f64().unwrap() as f32;
    assert!((got - w).abs() <= 1e-4, "{what}: {got} against {w}");
}

/// A span value against the reference's `{text, confidence, start, end}`.
fn span(text: &str, got: (usize, usize, f32), want: &Value, what: &str) {
    let at = |k: &str| want[k].as_u64().unwrap() as usize;
    assert_eq!((got.0, got.1), (at("start"), at("end")), "{what}: offsets");
    assert_eq!(text[got.0..got.1].trim(), want["text"].as_str().unwrap(), "{what}: text");
    close(got.2, &want["confidence"], what);
}

/// A list or single value against the reference's list or object or null.
fn values(text: &str, got: &[(usize, usize, f32)], single: bool, want: &Value, what: &str) {
    let want: Vec<&Value> = if single {
        want.as_object().map(|_| vec![want]).unwrap_or_default()
    } else {
        want.as_array().unwrap().iter().collect()
    };
    assert_eq!(got.len(), want.len(), "{what}: count {got:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        span(text, *g, w, &format!("{what}[{i}]"));
    }
}

#[test]
#[ignore = "fetches the 834 MB checkpoint through turbo-zoo; run with --release -- --ignored"]
fn the_real_checkpoint_matches_the_reference() {
    let m = model();
    for (c, case) in reference()["cases"].as_array().unwrap().iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        let schema = common::schema(&case["schema"]);
        let pass = m.pass(text, &schema).unwrap();
        let ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(pass.prompt.ids, ids, "case {c} ids");
        let counts: Vec<usize> = pass
            .scores
            .iter()
            .filter_map(|s| match s {
                Scores::Spans { count, .. } => Some(*count),
                _ => None,
            })
            .collect();
        let want: Vec<usize> =
            case["counts"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(counts, want, "case {c} counts");
        let logits: Vec<&Vec<f32>> = pass
            .scores
            .iter()
            .filter_map(|s| match s {
                Scores::Labels(l) => Some(l),
                _ => None,
            })
            .collect();
        for (t, (g, w)) in logits.iter().zip(case["classifier"].as_array().unwrap()).enumerate() {
            for (i, (g, w)) in g.iter().zip(w.as_array().unwrap()).enumerate() {
                close(*g, w, &format!("case {c} task {t} logit {i}"));
            }
        }
        let out = decode(&pass, &schema, 0.5, &Nothing);
        let result = &case["result"];
        let triple = |v: &turbo_schema_extract::decode::SpanValue| (v.start, v.end, v.score);
        for (f, e) in schema.entities.iter().enumerate() {
            let got: Vec<_> = out.entities[f].iter().map(triple).collect();
            values(
                text,
                &got,
                e.cardinality == Cardinality::Single,
                &result["entities"][&e.name],
                &format!("case {c} entity {}", e.name),
            );
        }
        for cl in &out.classifications {
            let t = &schema.classifications[cl.task];
            let want = &result[&t.name];
            let want: Vec<&Value> = if t.multi_label { want.as_array().unwrap().iter().collect() } else { vec![want] };
            assert_eq!(cl.labels.len(), want.len(), "case {c} task {}", t.name);
            for ((l, p, _), w) in cl.labels.iter().zip(want) {
                assert_eq!(t.labels[*l].name, w["label"].as_str().unwrap(), "case {c} task {}", t.name);
                close(*p, &w["confidence"], &format!("case {c} task {}", t.name));
            }
        }
        for (s, spec) in schema.structures.iter().enumerate() {
            let records: Vec<_> = out.records.iter().filter(|r| r.structure == s).collect();
            let want = result[&spec.name].as_array().unwrap();
            assert_eq!(records.len(), want.len(), "case {c} record {}", spec.name);
            for (r, w) in records.iter().zip(want) {
                for (f, fs) in spec.fields.iter().enumerate() {
                    let what = format!("case {c} {}.{}", spec.name, fs.name);
                    let wv = &w[&fs.name];
                    if let Some(choices) = &fs.choices {
                        let got: Vec<(&str, f32)> = r.fields[f]
                            .iter()
                            .map(|v| match v {
                                FieldValue::Choice { choice, score, .. } => (choices[*choice].as_str(), *score),
                                FieldValue::Span(_) => panic!("{what}: a span for a choice field"),
                            })
                            .collect();
                        let want: Vec<&Value> = if fs.cardinality == Cardinality::Single {
                            wv.as_object().map(|_| vec![wv]).unwrap_or_default()
                        } else {
                            wv.as_array().unwrap().iter().collect()
                        };
                        assert_eq!(got.len(), want.len(), "{what}");
                        for (g, w) in got.iter().zip(want) {
                            assert_eq!(g.0, w["text"].as_str().unwrap(), "{what}");
                            close(g.1, &w["confidence"], &what);
                        }
                    } else {
                        let got: Vec<_> = r.fields[f]
                            .iter()
                            .map(|v| match v {
                                FieldValue::Span(s) => triple(s),
                                FieldValue::Choice { .. } => panic!("{what}: a choice for a span field"),
                            })
                            .collect();
                        values(text, &got, fs.cardinality == Cardinality::Single, wv, &what);
                    }
                }
            }
        }
    }
}
