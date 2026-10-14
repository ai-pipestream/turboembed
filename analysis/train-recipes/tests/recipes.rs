//! The MASC recipes end to end on corpora written in MASC's layout: the
//! split, training, scoring both models on the held-out samples, the
//! gate, the card and the model's directory.

mod common;

use std::path::PathBuf;

use common::{SENTENCES, temp_dir, write_corpus};
use turbo_eval::model_gate::{Measure, ModelGateRecord, Task};
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::{CARD_FILE, GATE_FILE, ModelCard, write_model_dir};
use turbo_train_recipes::masc::{self, RELEASE, SPLIT, SplitCorpus};
use turbo_train_recipes::segmentation::{RECIPES, SENTENCE, TOKENS};
use turbo_train_recipes::{held_out_name, run};
use turbo_zoo::{Index, LicenceClass};

/// The pool without abbreviations, attached punctuation or contractions:
/// a model trained on it has not seen what the held-out text holds.
const PLAIN: &[&str] = &["Who called ~? Nobody did ~!", "Mr. Jones left early ~.", "It rained all day ~."];

fn split_of(root: &std::path::Path) -> (SplitCorpus, masc::CorpusDigest) {
    let digest = masc::check_corpus(root, RELEASE, true).unwrap();
    let corpus = MascCorpus::read(root).unwrap();
    (SPLIT.apply(corpus, root), digest)
}

#[test]
fn the_split_holds_out_every_tenth_document_in_read_order() {
    let root = temp_dir("split");
    write_corpus(&root, 25, SENTENCES);
    let all: Vec<PathBuf> = MascCorpus::read(&root).unwrap().documents().iter().map(|d| d.path().to_owned()).collect();
    let (split, digest) = split_of(&root);
    assert_eq!(digest.files, 25 * 5);
    assert_eq!(split.train.documents().len(), 23);
    let held: Vec<String> =
        [9, 19].iter().map(|&i| all[i].strip_prefix(&root).unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(split.held_out_paths, held);
    assert!(split.held_out_paths.iter().all(|p| !p.starts_with('/')), "{:?}", split.held_out_paths);
    assert!(split.skipped_paths.is_empty());
    // The read order is the reader's: each directory's entries by name,
    // a directory's subdirectories after its files, the last pushed first.
    assert!(split.held_out_paths[0].starts_with("written/news/"), "{:?}", split.held_out_paths);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn the_recipes_train_score_gate_and_write_their_cards() {
    let root = temp_dir("recipes");
    write_corpus(&root, 30, SENTENCES);
    let (split, digest) = split_of(&root);

    // The reference: the same recipe trained on a corpus without
    // abbreviations, contractions or attached punctuation.
    let plain_root = temp_dir("plain");
    write_corpus(&plain_root, 30, PLAIN);
    let (plain, _) = split_of(&plain_root);

    let out = temp_dir("out");
    for recipe in RECIPES {
        let weak = recipe.train(&plain.train).unwrap();
        let outcome = run(&recipe, &split, &digest, false, "plain", &weak.bytes).unwrap();
        let gate = &outcome.gate;
        gate.check().unwrap();
        println!("{gate}");
        assert_eq!(gate.task, recipe.task());
        assert_eq!(gate.measure, Measure::FMeasure);
        assert_eq!(gate.set.name, held_out_name());
        assert_eq!(gate.set.corpus_sha256, digest.sha256);
        assert_eq!(gate.candidate.id, recipe.model_id);
        assert_eq!(gate.reference.id, "plain");
        // The model trained on the corpus is at least as good as the one
        // that never saw its abbreviations and contractions, and passes.
        assert!(gate.result.difference.mean > 0.0, "{gate}");
        assert!(gate.result.pass);
        assert!(outcome.trained.samples > 0 && outcome.trained.events > 0);
        assert_eq!(outcome.trained.skipped_sentences, 0);

        // The same models with their places swapped: the weak model as
        // the candidate is below the reference on every draw and fails.
        let held_out = recipe.held_out(&split.held_out).unwrap();
        let swapped = recipe
            .gate(&weak.bytes, recipe.model_id, &outcome.trained.bytes, &held_out, &held_out_name(), &digest.sha256)
            .unwrap();
        assert!(swapped.result.difference.high < 0.0, "{swapped}");
        assert!(!swapped.result.pass);
        // A model against itself passes with no difference.
        let itself = recipe
            .gate(
                &outcome.trained.bytes,
                recipe.model_id,
                &outcome.trained.bytes,
                &held_out,
                &held_out_name(),
                &digest.sha256,
            )
            .unwrap();
        assert_eq!(itself.result.difference.mean, 0.0);
        assert!(itself.result.pass);

        // Training is deterministic: the same corpus gives the same file.
        assert_eq!(recipe.train(&split.train).unwrap().bytes, outcome.trained.bytes);

        let card = &outcome.card;
        assert_eq!(card.id, recipe.model_id);
        assert_eq!(card.class, LicenceClass::Permissive);
        assert_eq!(card.trained_on, vec![masc::training_data()]);
        assert!(card.attribution.contains("CC") || card.attribution.contains("Creative Commons"));
        assert!(!card.corpus.release_recorded);
        assert_eq!(card.corpus.held_out_documents, split.held_out_paths);
        assert_eq!(card.training.seed, None);
        assert_eq!(card.training.sentences_per_sample.is_some(), recipe.task() == Task::SentenceDetector);
        assert_eq!(card.gate.pass, gate.result.pass);

        write_model_dir(&out, card, &outcome.trained.bytes, gate).unwrap();
        let dir = out.join(recipe.model_id);
        assert_eq!(ModelCard::read(&dir.join(CARD_FILE)).unwrap(), *card);
        assert_eq!(ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap(), *gate);
        assert_eq!(std::fs::read(dir.join(&card.file.name)).unwrap(), outcome.trained.bytes);
        assert!(write_model_dir(&out, card, &outcome.trained.bytes, gate).is_err(), "nothing is replaced");

        // The model's zoo entry is permissive, shippable and names MASC.
        let url = format!("https://example.org/models/{}", card.file.name);
        let artifact = card.artifact(&url).unwrap();
        let index = Index::parse(Index::to_canonical_json(&[artifact]).as_bytes()).unwrap();
        let entry = index.get(recipe.model_id).unwrap();
        assert_eq!(entry.computed_class(), LicenceClass::Permissive);
        assert_eq!(index.shippable().count(), 1);
        assert_eq!(entry.trained_on[0].name, masc::NAME);
    }
    for dir in [root, plain_root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn a_held_out_set_of_another_component_is_refused() {
    let root = temp_dir("component");
    write_corpus(&root, 20, SENTENCES);
    let (split, _) = split_of(&root);
    let model = SENTENCE.train(&split.train).unwrap();
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert!(SENTENCE.score(&model.bytes, &tokens).is_err());
    // A sentence model is not a tokenizer model.
    let sentences = SENTENCE.held_out(&split.held_out).unwrap();
    assert!(TOKENS.score(&model.bytes, &sentences).is_err());
    let tokens_again = TOKENS.held_out(&split.held_out).unwrap();
    assert!(TOKENS.score(&model.bytes, &tokens_again).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// The gate against the published OpenNLP English UD models, when
/// `TURBO_GOLDENS_MODELS` names a directory that has them. The corpus is
/// the test corpus, not MASC, so the scores say nothing about the
/// recipes; the test checks that the published models load, match the
/// zoo's digests and are scored on the same samples.
#[test]
fn the_published_references_are_checked_and_scored_when_present() {
    let Some(dir) = std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from) else {
        eprintln!("TURBO_GOLDENS_MODELS is not set: skipping");
        return;
    };
    let root = temp_dir("published");
    write_corpus(&root, 20, SENTENCES);
    let (split, digest) = split_of(&root);
    for recipe in RECIPES {
        let path = dir.join(recipe.reference_file().unwrap());
        let Ok(reference) = std::fs::read(&path) else {
            eprintln!("{} not found: skipping", path.display());
            continue;
        };
        recipe.check_reference(&reference).unwrap();
        let outcome = run(&recipe, &split, &digest, false, recipe.reference_id, &reference).unwrap();
        outcome.gate.check().unwrap();
        println!("{}", outcome.gate);
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn the_binary_refuses_an_unrecorded_corpus_and_writes_each_model_s_directory() {
    let root = temp_dir("binary");
    write_corpus(&root, 20, SENTENCES);
    let out = temp_dir("binary-out");
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let refs = std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from).unwrap_or_else(|| out.join("none"));
    let args = |extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("masc").arg("--corpus").arg(&root).arg("--references").arg(&refs).arg("--out").arg(out.join("models"));
        c.args(extra);
        c.output().unwrap()
    };
    let refused = args(&[]);
    assert_eq!(refused.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("no recorded digest"), "{refused:?}");
    assert!(!out.join("models").exists());

    let all_present = RECIPES.iter().all(|r| refs.join(r.reference_file().unwrap()).is_file());
    let ran = args(&["--unrecorded-corpus"]);
    if all_present {
        // Every model trained on the test corpus beats the published
        // models on it, so the gates pass.
        assert_eq!(ran.status.code(), Some(0), "{ran:?}");
        for r in RECIPES {
            let dir = out.join("models").join(r.model_id);
            let card = ModelCard::read(&dir.join(CARD_FILE)).unwrap();
            assert!(!card.corpus.release_recorded);
            assert_eq!(card.gate.reference, r.reference_id);
            ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap();
        }
    } else {
        eprintln!("the published models are not under {}: only the refusal was checked", refs.display());
        assert_eq!(ran.status.code(), Some(2));
    }
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}
