#[path = "../../../tests/common/mod.rs"]
mod common;

use super::*;
use crate::namefind::selection::{GRID, select};
use crate::{RecordedReference, Reference};
use common::{SENTENCES, temp_dir, write_document};
use turbo_zoo::Sha256;

#[test]
fn real_model_gate_preserves_constants_and_existing_attempt_is_refused() {
    let root = temp_dir("name-selection-gate");
    let train_dir = root.join("train");
    let held_dir = root.join("held");
    for i in 0..10 {
        write_document(&train_dir.join("written/news"), &format!("t{i:02}"), SENTENCES, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
    for i in 0..5 {
        write_document(&held_dir.join("written/news"), &format!("h{i:02}"), SENTENCES, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
    let split = SplitCorpus {
        train: MascCorpus::read(&train_dir).unwrap(),
        held_out: MascCorpus::read(&held_dir).unwrap(),
        held_out_paths: (0..5).map(|i| format!("h{i:02}")).collect(),
        excluded: Vec::new(),
    };
    let recipe = RECIPES[0];
    let selection = select(&recipe, &split.train).unwrap();
    let chosen = selection.candidates[selection.selected].setting;
    assert!(GRID.contains(&chosen));
    let mut selected = train_selected(&recipe, &split.train, selection).unwrap();
    // A separately produced real model is the test reference. This checks
    // gate wiring, not performance against a released OpenNLP checkpoint.
    let reference = crate::namefind::PERSON.train(&split.train).unwrap().bytes;
    let pinned = Reference::Recorded(RecordedReference {
        id: "authored-namefinder-reference",
        file: "authored-namefinder-reference.bin",
        url: "https://example.invalid/authored-namefinder-reference.bin",
        size: reference.len() as u64,
        sha256: Box::leak(Sha256::of(&reference).to_string().into_boxed_str()),
    });
    selected.recipe.reference = pinned;
    let digest = masc::corpus_digest(&root).unwrap();
    let (outcome, card) = finish(selected, &split, &digest, false, &reference).unwrap();
    assert_eq!(outcome.gate.resamples, 10_000);
    assert_eq!(outcome.gate.seed, 154);
    assert_eq!(outcome.gate.candidate.id, recipe.model_id);
    assert_eq!(outcome.gate.reference.id, "authored-namefinder-reference");
    assert_eq!(outcome.gate.units.len(), 40);
    assert_eq!(outcome.card.training.parameters.iterations, chosen.iterations);
    assert_eq!(outcome.card.training.parameters.cutoff, chosen.cutoff);
    assert_eq!(outcome.card.corpus.held_out_documents, split.held_out_paths);
    assert_eq!(card.training_documents, 10);
    outcome.gate.check().unwrap();
    let out = root.join("out");
    let attempt = out.join(recipe.model_id);
    std::fs::create_dir_all(&attempt).unwrap();
    let sentinel = attempt.join("selection.json");
    std::fs::write(&sentinel, "preserved").unwrap();
    assert!(reserve_attempt(&out, &attempt).is_err());
    assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "preserved");
    let error = write_attempt(&recipe, &split, &digest, true, &reference, &root.join("wrong-count")).unwrap_err();
    assert!(error.to_string().contains("351 training and 39 held-out"));
    assert!(!root.join("wrong-count").exists());
    std::fs::remove_dir_all(root).unwrap();
}
