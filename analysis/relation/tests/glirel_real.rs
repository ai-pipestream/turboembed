//! The native GLiREL scorer against the reference itself on the full
//! `jackboyla/glirel-large-v0` checkpoint: the words, the input ids, every
//! ordered entity pair's logit for every label and the relations above
//! 0.5, from `tests/fixtures/glirel/real.json`, which
//! `reference/glirel/build_fixture.py` writes with GLiREL's own
//! `predict_relations`. Then the scorer behind the zero-shot annotator,
//! with its model in the relations' evidence.
//!
//! The checkpoint is CC BY-NC-SA 4.0 (1.87 GB): `turbo-zoo` fetches it only
//! for a user who has accepted that licence (`turbo-zoo accept
//! CC-BY-NC-SA-4.0`). With `TURBO_GLIREL_CHECKPOINT` set to a directory
//! that already holds `pytorch_model.bin`, that copy is used and checked
//! against the index's digest instead. The tokenizer (MIT) is fetched by
//! digest, and the package `turbo-bundle glirel-convert` writes is made
//! once in the zoo's home. Run it with
//! `cargo test --release -p turbo-relation --test glirel_real -- --ignored`.

use std::path::{Path, PathBuf};

use serde_json::Value;
use turbo::deberta::package::digest;
use turbo_relation::GlirelRelations;
use turbo_relation::glirel::words::split;
use turbo_zoo::{Acceptances, Home, Http, Index, Policy, UserId, Zoo};

const ID: &str = "jackboyla-glirel-large-v0";
const TOKENIZER: &str = "microsoft-deberta-v3-large-tokenizer";

/// The package, made once from the checkpoint and the tokenizer, and the
/// tokenizer's directory.
fn model() -> GlirelRelations {
    static ONCE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _held = ONCE.lock().unwrap_or_else(|e| e.into_inner());
    let home = Home::from_env().expect("a zoo home");
    let index = Index::builtin();
    let zoo = Zoo::new(Index::builtin(), home.artifacts(), Policy::default(), &Http);
    let user = UserId::from_env().unwrap_or_else(|_| "test".parse().unwrap());
    let accepted = || Acceptances::load(&home.acceptances());
    let tokenizer = zoo.fetch_with(TOKENIZER, &user, accepted).unwrap().dir.clone();
    let entry = index.get(ID).unwrap();
    let checkpoint = match std::env::var_os("TURBO_GLIREL_CHECKPOINT") {
        Some(dir) => PathBuf::from(dir),
        None => zoo
            .fetch_with(ID, &user, accepted)
            .unwrap_or_else(|e| panic!("{e}; accept the licence with `turbo-zoo accept CC-BY-NC-SA-4.0` first"))
            .dir
            .clone(),
    };
    let package = home.dir().join("packages").join(ID);
    if !package.join("glirel.json").exists() {
        let bin = std::fs::read(checkpoint.join("pytorch_model.bin")).unwrap();
        assert_eq!(digest(&bin), entry.files[0].sha256.to_string(), "the checkpoint is the index's");
        std::fs::create_dir_all(package.parent().unwrap()).unwrap();
        let staging = package.with_extension(format!("tmp-{}", std::process::id()));
        turbo_bundle::glirel::write_package(
            &bin,
            &std::fs::read(tokenizer.join("config.json")).unwrap(),
            &std::fs::read(tokenizer.join("spm.model")).unwrap(),
            ("jackboyla/glirel-large-v0", &entry.version, "CC-BY-NC-SA-4.0"),
            &staging,
        )
        .unwrap();
        if std::fs::rename(&staging, &package).is_err() && package.join("glirel.json").exists() {
            std::fs::remove_dir_all(&staging).unwrap();
        }
    }
    GlirelRelations::load(&package, &tokenizer.join("spm.model")).unwrap()
}

fn reference() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glirel/real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn spans(v: &Value) -> Vec<(usize, usize)> {
    v.as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)).collect()
}

#[test]
fn the_words_are_pythons() {
    for case in reference()["splits"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let got: Vec<(usize, usize)> = split(text).iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(got, spans(&case["words"]), "{text:?}");
    }
}

#[test]
#[ignore = "needs the 1.87 GB CC BY-NC-SA checkpoint; run with --release -- --ignored"]
fn the_real_checkpoint_matches_the_reference() {
    let m = model();
    assert_eq!(m.model().sha256, Index::builtin().get(ID).unwrap().files[0].sha256.to_string());
    for (c, case) in reference()["cases"].as_array().unwrap().iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        let labels: Vec<&str> = case["labels"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
        let prompt = m.prompt(text, &labels).unwrap();
        let words: Vec<(usize, usize)> = prompt.words.iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(words, spans(&case["words"]), "case {c} words");
        let ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(prompt.ids, ids, "case {c} ids");
        let entities = spans(&case["ner"]);
        let pairs = spans(&case["pairs"]);
        let pass = m.pass(text, &labels).unwrap();
        let logits = m.logits(&pass, &entities, &pairs).unwrap();
        let want: Vec<f32> = case["logits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().clone())
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(logits.len(), want.len(), "case {c} logits");
        for (i, (&g, &w)) in logits.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "case {c} logit {i}: {g} against {w}");
        }
        let found = m.extract(text, &entities, &labels, 0.5).unwrap();
        let mut got: Vec<(usize, usize, usize, f32)> =
            found.iter().map(|r| (r.head, r.tail, r.label, r.score)).collect();
        let mut expected: Vec<(usize, usize, usize, f32)> = case["relations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let at = |k: &str| {
                    let p = spans(&Value::Array(vec![r[k].clone()]))[0];
                    entities.iter().position(|&e| e == (p.0, p.1 - 1)).unwrap()
                };
                (at("head"), at("tail"), r["label"].as_u64().unwrap() as usize, r["score"].as_f64().unwrap() as f32)
            })
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got.len(), expected.len(), "case {c} relations: {got:?} against {expected:?}");
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!((g.0, g.1, g.2), (e.0, e.1, e.2), "case {c}");
            assert!((g.3 - e.3).abs() <= 1e-4, "case {c} score {} against {}", g.3, e.3);
        }
    }
}

#[test]
#[ignore = "needs the 1.87 GB CC BY-NC-SA checkpoint; run with --release -- --ignored"]
fn the_scorer_scores_mentions_as_extract_scores_entities() {
    use turbo_analysis::Span;
    use turbo_relation::{Mention, RelationScorer};
    let m = model();
    let case = &reference()["cases"][1];
    let text = case["text"].as_str().unwrap();
    let labels: Vec<&str> = case["labels"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
    let words = split(text);
    let entities = spans(&case["ner"]);
    let mut mentions: Vec<Mention<'_>> = entities
        .iter()
        .map(|&(s, e)| Mention { span: Span::new(words[s].start as u32, words[e].end as u32), kind: "ENT" })
        .collect();
    // "Steve Jobs" without its first letter: not a run of words.
    mentions.push(Mention { span: Span::new(1, 10), kind: "ENT" });
    let pairs = [(0u32, 1u32), (1, 0), (3, 4), (5, 1)];
    let mut out = Vec::new();
    m.score(text, &mentions, &pairs, &labels, &mut out).unwrap();
    let pass = m.pass(text, &labels).unwrap();
    let logits = m.logits(&pass, &entities, &[(0, 1), (1, 0), (3, 4)]).unwrap();
    let k = labels.len();
    for (got, l) in out[..3 * k].iter().zip(&logits) {
        assert!((got - 1.0 / (1.0 + (-l).exp())).abs() < 1e-6);
    }
    assert!(out[3 * k..].iter().all(|&s| s == 0.0), "a partial-word mention scores 0");
    assert_eq!(m.model_ref().unwrap().id, "jackboyla/glirel-large-v0");
}
