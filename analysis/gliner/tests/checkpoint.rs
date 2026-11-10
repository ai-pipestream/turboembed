//! Full-width checkpoint parity for `urchade/gliner_small-v2.1` and
//! `urchade/gliner_multi-v2.1`. Fetches the permissive checkpoints and
//! tokenizers through `turbo-zoo` and packages them via `turbo-bundle` when
//! `GLINER_PACKAGE` / `GLINER_MULTI_PACKAGE` are not set.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use turbo_gliner::{Decode, ExtractionSchema, Model, package::Manifest};
use turbo_zoo::{Acceptances, Home, Http, Index, Policy, UserId, Zoo};

fn load_or_fetch(env_var: &str, id: &str, tokenizer_id: &str, repo: &str, weights_file: &str) -> Arc<Model> {
    if let Some(root) = std::env::var_os(env_var).filter(|v| !v.is_empty()) {
        return Arc::new(Manifest::load(&turbo_pins::require(&root)).unwrap());
    }
    static ONCE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _held = ONCE.lock().unwrap_or_else(|e| e.into_inner());
    let home = Home::from_env().expect("a zoo home");
    let index = Index::builtin();
    let transport = Http::new();
    let zoo = Zoo::new(Index::builtin(), home.artifacts(), Policy::default(), &transport);
    let user = UserId::from_env().unwrap_or_else(|_| "test".parse().unwrap());
    let accepted = || Acceptances::load(&home.acceptances());
    let tok_dir = zoo.fetch_with(tokenizer_id, &user, accepted).unwrap().dir.clone();
    let ckpt_dir = zoo.fetch_with(id, &user, accepted).unwrap().dir.clone();
    let entry = index.get(id).unwrap();
    let package = home.dir().join("packages").join(id);
    if !package.join("gliner.json").exists() {
        std::fs::create_dir_all(package.parent().unwrap()).unwrap();
        let staging = package.with_extension(format!("tmp-{}", std::process::id()));
        turbo_bundle::gliner::convert_files(
            &tok_dir.join("config.json"),
            &ckpt_dir.join("gliner_config.json"),
            &ckpt_dir.join(weights_file),
            &tok_dir.join("spm.model"),
            (repo, &entry.version, "Apache-2.0"),
            &staging,
        )
        .unwrap();
        if std::fs::rename(&staging, &package).is_err() && package.join("gliner.json").exists() {
            std::fs::remove_dir_all(&staging).unwrap();
        }
    }
    Arc::new(Manifest::load(&package).unwrap())
}

fn verify_reference(model: Arc<Model>, reference_path: &Path) {
    let data: Value = serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let mut maximum = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let labels = case["labels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let schema = ExtractionSchema::new(labels, 0.5, Decode::Nested).unwrap();
        let words: Vec<_> = case["words"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        let p = model.pack(&words, &schema).unwrap();
        assert_eq!(serde_json::to_value(&p.ids).unwrap(), case["ids"]);
        assert_eq!(serde_json::to_value(&p.prompts).unwrap(), case["prompts"]);
        assert_eq!(serde_json::to_value(&p.words).unwrap(), case["positions"]);
        let scores = model.score_packed(&p, &schema).unwrap();
        if !model.config().token_level {
            let (_, training_scores) = model.training_input(&words, &schema).unwrap();
            assert_eq!(training_scores, scores);
        }
        assert_eq!(scores.len(), case["logits"].as_array().unwrap().len());
        for (name, mode) in [("flat", Decode::Flat), ("nested", Decode::Nested)] {
            let schema = ExtractionSchema::new(schema.labels().to_vec(), 0.5, mode).unwrap();
            let selected = turbo_gliner::decode(scores.clone(), &schema);
            let expected = case["decoded"][name].as_array().unwrap();
            assert_eq!(selected.len(), expected.len());
            for (a, b) in selected.iter().zip(expected) {
                assert_eq!(
                    (a.start, a.end, a.label),
                    (b[0].as_u64().unwrap() as usize, b[1].as_u64().unwrap() as usize, b[2].as_u64().unwrap() as usize)
                );
                assert!((f64::from(a.score) - b[3].as_f64().unwrap()).abs() < 0.0001);
            }
        }
        for (s, r) in scores.iter().zip(case["logits"].as_array().unwrap()) {
            let error = (f64::from(s.logit) - r.as_f64().unwrap()).abs();
            maximum = maximum.max(error);
            assert!(error < 0.002, "logit {:?}: reference={r}, error={error}", s);
        }
    }
    verify_offsets_and_evidence(model);
    println!("full checkpoint maximum logit absolute error: {maximum}");
}

#[test]
#[ignore = "fetches or loads Apache-2.0 GLiNER small-v2.1 checkpoint; see docs/gliner.md"]
fn full_checkpoint_matches_python() {
    let model = load_or_fetch(
        "GLINER_PACKAGE",
        "urchade-gliner-small-v2.1",
        "microsoft-deberta-v3-small-tokenizer",
        "urchade/gliner_small-v2.1",
        "pytorch_model.bin",
    );
    let reference = std::env::var_os("GLINER_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/full-reference.json"));
    verify_reference(model, &reference);
}

#[test]
#[ignore = "fetches or loads Apache-2.0 GLiNER multi-v2.1 checkpoint; see docs/gliner.md"]
fn multi_checkpoint_matches_python() {
    let model = load_or_fetch(
        "GLINER_MULTI_PACKAGE",
        "urchade-gliner-multi-v2.1",
        "microsoft-mdeberta-v3-base-tokenizer",
        "urchade/gliner_multi-v2.1",
        "model.safetensors",
    );
    let reference = std::env::var_os("GLINER_MULTI_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/multi-reference.json"));
    verify_reference(model, &reference);
}

#[test]
#[ignore = "loads the local Apache-2.0 GLiNER small-v2.5 checkpoint; see docs/gliner.md"]
fn small_v2_5_checkpoint_matches_python() {
    let root = std::env::var_os("GLINER_V2_5_PACKAGE").expect("GLINER_V2_5_PACKAGE must name the converted package");
    let model = Arc::new(Manifest::load(&turbo_pins::require(&root)).unwrap());
    verify_reference(
        model.clone(),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/small-v2.5-reference.json").as_path(),
    );
    let schema =
        ExtractionSchema::new(vec!["person".into(), "organization".into(), "location".into()], 0.5, Decode::Flat)
            .unwrap();
    let words = ["Alice", "works", "at", "Microsoft", "in", "Paris", "."];
    let (input, candidates) = model.training_input(&words, &schema).unwrap();
    let original = std::fs::read(turbo_pins::require(&root).join("heads.safetensors")).unwrap();
    let trainer = turbo_gliner::train::OutputTrainer::load(model.config(), &original).unwrap();
    let mut grads = trainer.zero_grads();
    let gold = [(words.len() - 1, words.len(), 0)];
    let loss = trainer.gradients(&input, &candidates, &gold, &mut grads).unwrap();
    assert!(loss > 0.0, "a one-word span at the sentence end must train");
    assert!(grads[0].iter().any(|&x| x != 0.0));

    let mut boundary =
        turbo_gliner::train::OutputTrainer::load_with_boundaries(model.config(), &original, true).unwrap();
    let mut grads = boundary.zero_grads();
    let boundary_loss = boundary.gradients(&input, &candidates, &gold, &mut grads).unwrap();
    assert!((boundary_loss - loss).abs() < 1e-4);
    assert!(grads[4..].iter().flatten().any(|&x| x != 0.0));
    let mut optimizer = turbo_gliner::train::output_optimizer(
        turbo_trainer::adamw::Settings { lr: 1e-6, total: 1, ..turbo_trainer::adamw::Settings::default() },
        &boundary,
    )
    .unwrap();
    boundary.step(&mut optimizer, &mut grads).unwrap();
    let mut tuned = Manifest::load(&turbo_pins::require(&root)).unwrap();
    tuned.apply_output(&boundary).unwrap();
    let expected = tuned.training_input(&words, &schema).unwrap().1;
    let heads = boundary.write_heads(&original).unwrap();
    let out = std::env::temp_dir().join(format!("turbo-gliner-boundary-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    Manifest::write_tuned(&turbo_pins::require(&root), &out, &heads).unwrap();
    let reloaded = Manifest::load(&out).unwrap();
    assert_eq!(reloaded.training_input(&words, &schema).unwrap().1, expected);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
#[ignore = "loads the local MIT NuNER Zero-span checkpoint and generated Python reference"]
fn nuner_zero_span_checkpoint_matches_python() {
    let root = std::env::var_os("GLINER_NUNER_PACKAGE").expect("GLINER_NUNER_PACKAGE must name the converted package");
    let reference =
        std::env::var_os("GLINER_NUNER_REFERENCE").expect("GLINER_NUNER_REFERENCE must name the reference JSON");
    let model = Arc::new(Manifest::load(&turbo_pins::require(&root)).unwrap());
    verify_reference(model, Path::new(&reference));
}

#[test]
#[ignore = "loads the local Apache-2.0 token-level GLiNER checkpoint and generated Python reference"]
fn token_level_checkpoint_matches_python() {
    let root = std::env::var_os("GLINER_TOKEN_PACKAGE").expect("GLINER_TOKEN_PACKAGE must name the converted package");
    let reference =
        std::env::var_os("GLINER_TOKEN_REFERENCE").expect("GLINER_TOKEN_REFERENCE must name the reference JSON");
    let model = Arc::new(Manifest::load(&turbo_pins::require(&root)).unwrap());
    verify_reference(model, Path::new(&reference));
}

fn verify_offsets_and_evidence(model: std::sync::Arc<turbo_gliner::Model>) {
    use turbo_analysis::{Analyzer, Bound, Document, Evidence, Key, Layer, Provenance, Span, Spans, View};
    use turbo_analysis_layers::{Entities, Tokens};
    use turbo_gliner::{Decisions, GlinerAnnotator};
    let text = "Zoë visited the University of California .";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(doc.get::<View>().unwrap().id());
    let mut offset = 0;
    for word in text.split(' ') {
        spans.push(Span::new(offset, offset + word.len() as u32));
        offset += word.len() as u32 + 1;
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("fixture:tokens")).unwrap();
    let schema =
        ExtractionSchema::new(vec!["person".into(), "university".into(), "location".into()], 0.0, Decode::Nested)
            .unwrap();
    let ann = GlinerAnnotator::new(model, schema);
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(ann).write(0, "ner").write(1, "ner"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in pipeline.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let key = Key::<Entities>::named("ner");
    let entities = doc.require(&key).unwrap();
    let (a, b) = entities.spans.columns();
    assert!(a.iter().zip(b).enumerate().any(|(i, (&s, &e))| {
        a.iter().zip(b).enumerate().any(|(j, (&t, &u))| i != j && s <= t && e >= u && (s != t || e != u))
    }));
    let tokens = doc.get::<Tokens>().unwrap();
    let (st, en) = tokens.spans.columns();
    for i in 0..entities.len() {
        let r = entities.tokens[i];
        assert_eq!(
            &text[a[i] as usize..b[i] as usize],
            &text[st[r.start.get()] as usize..en[r.end.get() - 1] as usize]
        );
    }
    let ev = GlinerAnnotator::evidence(&doc, key, Key::<Decisions>::named("ner")).unwrap();
    assert_eq!(ev.verify(&doc), Ok(()));
    let mut provenance = Provenance::by("fixture:evidence");
    provenance.reads = vec![key.id(), Key::<Decisions>::named("ner").id()];
    doc.add(&Key::<Evidence<Entities>>::named("ner"), ev, provenance).unwrap();
    let mut codecs = turbo_analysis_layers::export::codecs();
    codecs.register::<Decisions>();
    let graph = turbo_analysis::export::export(&doc, &codecs).unwrap();
    let back = turbo_analysis::export::import(&graph, &codecs).unwrap();
    assert_eq!(back.require(&Key::<Evidence<Entities>>::named("ner")).unwrap().verify(&back), Ok(()));
}
