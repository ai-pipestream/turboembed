//! The native GLiNER2 relation scorer against the reference itself on the
//! full `fastino/gliner2-base-v1` checkpoint: the input ids (word
//! splitting, schemas and SentencePiece), each relation's instance count,
//! and the extracted relations with their byte offsets and confidences,
//! from `tests/fixtures/gliner2/real.json`, which
//! `reference/gliner2/build_fixture.py` writes with GLiNER2's own
//! `extract_relations`. Then the scorer behind the zero-shot annotator,
//! with its model in the relations' evidence.
//!
//! The checkpoint (Apache-2.0, 834 MB) is fetched through `turbo-zoo` by
//! digest into the zoo's home (`$TURBO_ZOO_HOME`, else
//! `~/.local/share/turbo-zoo`) on first use, and its encoder packaged
//! beside it. Run it with
//! `cargo test --release -p turbo-relation --test gliner2_real -- --ignored`.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use turbo_analysis::{
    Analyzer, Document, Evidence, EvidenceStep, IxRange, Key, Provenance, Span, Spans, TagSet, ViewId,
};
use turbo_analysis_layers::Entities;
use turbo_relation::{Gliner2Relations, RelationLabel, Relations, ZeroShotRelationAnnotator, ZeroShotRelationConfig};
use turbo_zoo::{Acceptances, Home, Http, Index, Policy, UserId, Zoo};

const ID: &str = "fastino-gliner2-base-v1";
const REPOSITORY: &str = "fastino/gliner2-base-v1";
const REVISION: &str = "f9634218e53580c56edf0de97ca1a7d3f1c2354e";

/// The checkpoint's directory, fetched and verified, and its encoder
/// package, made once beside it.
fn model() -> Gliner2Relations {
    // The tests of this file share one fetch and one package.
    static ONCE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _held = ONCE.lock().unwrap_or_else(|e| e.into_inner());
    let home = Home::from_env().expect("a zoo home");
    let transport = Http::new();
    let zoo = Zoo::new(Index::builtin(), home.artifacts(), Policy::default(), &transport);
    let user = UserId::from_env().unwrap_or_else(|_| "test".parse().unwrap());
    let fetched = zoo.fetch_with(ID, &user, || Acceptances::load(&home.acceptances())).unwrap();
    let dir = fetched.dir.clone();
    let package = home.dir().join("packages").join(format!("{ID}-encoder"));
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
        // Another process may have packaged it meanwhile; its package is
        // the same bytes.
        if std::fs::rename(&staging, &package).is_err() && package.join("deberta.json").exists() {
            std::fs::remove_dir_all(&staging).unwrap();
        }
    }
    Gliner2Relations::load(&package, &dir, REPOSITORY, REVISION).unwrap()
}

fn reference() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gliner2/real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
}

#[test]
#[ignore = "fetches the 834 MB checkpoint through turbo-zoo; run with --release -- --ignored"]
fn the_real_checkpoint_matches_the_reference() {
    let m = model();
    let index = Index::builtin();
    let sha = index.get(ID).unwrap().files[0].sha256.to_string();
    assert_eq!(m.model().sha256, sha);
    for case in reference()["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let labels = strings(&case["labels"]);
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(m.prompt(text, &labels).ids, ids, "{text}: input ids");
        let pass = m.pass(text, &labels).unwrap();
        let counts: Vec<u64> = pass.labels.iter().map(|l| l.count as u64).collect();
        let want: Vec<u64> = case["counts"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(counts, want, "{text}: instance counts");
        let got = m.extract(text, &labels, 0.5).unwrap();
        let want = case["relations"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{text}: {got:?}");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.label as u64, w["label"].as_u64().unwrap(), "{text}");
            for (end, found) in [("head", g.head), ("tail", g.tail)] {
                let e = w[end].as_array().unwrap();
                assert_eq!(
                    (found.start as u64, found.end as u64),
                    (e[0].as_u64().unwrap(), e[1].as_u64().unwrap()),
                    "{text} {end}"
                );
                let conf = e[2].as_f64().unwrap() as f32;
                assert!((found.score - conf).abs() < 1e-4, "{text} {end}: {} against {conf}", found.score);
            }
        }
    }
}

#[test]
#[ignore = "fetches the 834 MB checkpoint through turbo-zoo; run with --release -- --ignored"]
fn the_scorer_finds_the_reference_relations_between_given_mentions() {
    let m = model();
    let sha = m.model().sha256.clone();
    let text = "Steve Jobs founded Apple in Cupertino. Tim Cook works for Apple.";
    let mut doc = Document::new(text);
    let kinds = TagSet::new("kinds", &["PERSON", "ORG", "LOC"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    let ents = [(0, 10, 0), (19, 24, 1), (28, 37, 2), (39, 47, 0), (58, 63, 1)];
    for &(a, b, _) in &ents {
        spans.push(Span::new(a, b));
    }
    let entities = Entities {
        spans,
        tokens: (0..5).map(|i| IxRange::new(i, i + 1)).collect(),
        kind: ents.iter().map(|e| turbo_analysis::Tag(e.2)).collect(),
        kinds: Arc::clone(&kinds),
        score: vec![1.0; 5],
    };
    let mut tokens = Spans::new(ViewId::SOURCE);
    for &(a, b, _) in &ents {
        tokens.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, turbo_analysis_layers::Tokens { spans: tokens, sentence: None }, Provenance::by("test"))
        .unwrap();
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();
    let typed = |name: &str, d: &str, r: &str| RelationLabel {
        domain: Some(vec![d.into()]),
        range: Some(vec![r.into()]),
        ..RelationLabel::new(name)
    };
    let labels =
        vec![typed("founded", "PERSON", "ORG"), typed("works for", "PERSON", "ORG"), typed("located in", "ORG", "LOC")];
    let config = ZeroShotRelationConfig { within_sentence: false, ..ZeroShotRelationConfig::new(labels) };
    let annotator = ZeroShotRelationAnnotator::new(config, m).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Entities>::DEFAULT)
        .add(annotator)
        .step(EvidenceStep::new(Key::<Relations>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let r = docs[0].get::<Relations>().unwrap();
    let got: Vec<(usize, usize, &str)> =
        (0..r.subject.len()).map(|i| (r.subject[i].get(), r.object[i].get(), r.kinds.label(r.kind[i]))).collect();
    for want in [(0, 1, "founded"), (3, 1, "works for"), (4, 2, "located in")] {
        assert!(got.contains(&want), "{want:?} in {got:?}");
    }
    let ev = docs[0].get::<Evidence<Relations>>().unwrap();
    let producer = ev.producer_of(0);
    assert_eq!(producer.annotator, "turbo:zero-shot-relations");
    let model = producer.model.as_ref().expect("the scorer's model in the evidence");
    assert_eq!((model.id.as_str(), model.sha256.as_str()), (REPOSITORY, sha.as_str()));
    assert_eq!(ev.verify(&docs[0]), Ok(()));
}
