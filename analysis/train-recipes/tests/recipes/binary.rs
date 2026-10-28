//! The `turbo-train-recipes masc` binary.

use crate::common::{GATED_DOCUMENTS, SENTENCES, rewrite_document, temp_dir, write_corpus};
use crate::{reference_path, sentences_after, split_of, with_skipped_sentences};
use turbo_eval::model_gate::ModelGateRecord;
use turbo_train_recipes::card::{CARD_FILE, GATE_FILE, ModelCard};
use turbo_train_recipes::release::MASC_MODELS;
use turbo_train_recipes::{RECIPES, Recipe};

/// The binary reads and checks every reference a run needs, and checks
/// that none of its model directories exists, before it trains anything:
/// a run that fails on one writes nothing, and says what every problem
/// is, in the order the recipes run, `RECIPES`'s, whatever the order of
/// `--recipe`.
#[test]
fn the_binary_checks_every_reference_model_directory_and_recipe_s_samples_before_training() {
    let root = temp_dir("binary-checks");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let out = temp_dir("binary-checks-out");
    let refs = out.join("references");
    std::fs::create_dir_all(&refs).unwrap();
    let models = out.join("models");
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let run = |extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("masc").arg("--corpus").arg(&root).arg("--references").arg(&refs).arg("--out").arg(&models);
        c.arg("--unrecorded-corpus").args(extra);
        let output = c.output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        String::from_utf8_lossy(&output.stderr).into_owned()
    };
    let (sentence, tokens, pos) = (RECIPES[0], RECIPES[1], RECIPES[2]);
    let file = |r: Recipe| r.reference_file().unwrap();

    // No references: the two named are missing, in the recipes' order.
    let stderr = run(&["--recipe", pos.model_id(), "--recipe", sentence.model_id()]);
    let (at_sentence, at_pos) = (stderr.find(&file(sentence)).unwrap(), stderr.find(&file(pos)).unwrap());
    assert!(at_sentence < at_pos, "{stderr}");
    assert!(!stderr.contains(&file(tokens)), "{stderr}");
    assert!(!models.exists(), "nothing is written");

    // References that are not the recorded files, and a model directory
    // already there.
    for r in RECIPES {
        std::fs::write(refs.join(file(r)), b"not the model").unwrap();
    }
    std::fs::create_dir_all(models.join(tokens.model_id())).unwrap();
    let stderr = run(&[]);
    assert_eq!(stderr.lines().count(), RECIPES.len() + 1, "{stderr}");
    for r in RECIPES {
        assert!(stderr.contains(&format!("reference model {}: SHA-256", r.reference_id())), "{stderr}");
    }
    assert!(stderr.contains(&format!("{}: the model directory exists", tokens.model_id())), "{stderr}");
    let written: Vec<_> = std::fs::read_dir(&models).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(written, [tokens.model_id()], "nothing else is written");

    // A held-out sentence of no tokens, which the token, tagger, chunker
    // and name recipes skip: one of 320 is more than one in a thousand,
    // the token, tagger and name recipes are named with the other
    // problems, before anything is trained, and the chunker, whose limit
    // is one in ten, is not.
    let (split, _) = split_of(&root);
    let no_tokens = SENTENCES.len() + 1;
    rewrite_document(&root, &split.held_out_paths[0], &with_skipped_sentences(), &sentences_after(&[no_tokens]));
    let stderr = run(&[]);
    let names = &RECIPES[4..];
    assert_eq!(stderr.lines().count(), RECIPES.len() + 3 + names.len(), "{stderr}");
    for r in [tokens, pos].into_iter().chain(names.iter().copied()) {
        let line = format!("{}: 1 of 320 held-out sentences skipped, more than the limit of 1 in 1000", r.model_id());
        assert!(stderr.contains(&line), "{stderr}");
    }
    for r in [sentence, RECIPES[3]] {
        assert!(!stderr.contains(&format!("{}: 1 of", r.model_id())), "{stderr}");
    }
    let written: Vec<_> = std::fs::read_dir(&models).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(written, [tokens.model_id()], "nothing else is written");
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn the_binary_refuses_an_unrecorded_corpus_and_writes_each_model_s_directory() {
    let root = temp_dir("binary");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let out = temp_dir("binary-out");
    // The references present, copied into one directory under the file
    // names the binary looks for.
    let refs = out.join("references");
    std::fs::create_dir_all(&refs).unwrap();
    let mut present = Vec::new();
    for r in RECIPES {
        if let Some(path) = reference_path(&r) {
            std::fs::copy(&path, refs.join(r.reference_file().unwrap())).unwrap();
            present.push(r);
        }
    }
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let args = |extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("masc").arg("--corpus").arg(&root).arg("--references").arg(&refs).arg("--out").arg(out.join("models"));
        c.args(extra);
        c.output().unwrap()
    };
    let refused = args(&[]);
    assert_eq!(refused.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("not the recorded release's"), "{refused:?}");
    assert!(!out.join("models").exists());
    let unknown = args(&["--unrecorded-corpus", "--recipe", "turbo-en-masc-3.0.0-unknown-1"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("no recipe trains"), "{unknown:?}");

    if present.len() < RECIPES.len() {
        // Without every reference, a run of every recipe fails before it
        // trains or writes anything, and names every one missing.
        let all = args(&["--unrecorded-corpus"]);
        assert_eq!(all.status.code(), Some(2), "{all:?}");
        let stderr = String::from_utf8_lossy(&all.stderr);
        for r in RECIPES.iter().filter(|r| !present.contains(r)) {
            assert!(stderr.contains(&r.reference_file().unwrap()), "{stderr}");
        }
        assert!(!out.join("models").exists(), "a run without every reference wrote models");
    }
    if present.is_empty() {
        eprintln!("no published model is present: only the refusals were checked");
    } else {
        // The models trained on the test corpus have seen its held-out
        // sentences and beat the published models on it, so the gates
        // pass; that says nothing about the recipes on MASC.
        let mut extra = vec!["--unrecorded-corpus"];
        for r in &present {
            extra.extend(["--recipe", r.model_id()]);
        }
        let ran = args(&extra);
        assert_eq!(ran.status.code(), Some(0), "{ran:?}");
        for r in RECIPES {
            let dir = out.join("models").join(r.model_id());
            if !present.contains(&r) {
                assert!(!dir.exists(), "{} ran without --recipe", r.model_id());
                continue;
            }
            let card = ModelCard::read(&dir.join(CARD_FILE)).unwrap();
            let stdout = String::from_utf8_lossy(&ran.stdout);
            let gate = ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap();
            assert!(stdout.contains(&format!("{}\n{gate}\n", card.summary())), "{stdout}");
            assert!(!card.corpus.release_recorded);
            assert_eq!(card.gate.reference, r.reference_id());
            assert_eq!(card.gate.reference_sha256, r.reference().sha256().unwrap());
        }
        // Models trained on a corpus that is not the recorded release make
        // no zoo entries, so the index is neither printed nor checked.
        let index = std::process::Command::new(bin)
            .args(["zoo-index", "--release", MASC_MODELS.tag, "--models"])
            .arg(out.join("models"))
            .output()
            .unwrap();
        assert_eq!(index.status.code(), Some(2), "{index:?}");
        // The release's models are read in order: the first one's card,
        // when its reference was present and it ran, is refused for the
        // corpus, and otherwise its missing card is.
        let first_ran = present.contains(&MASC_MODELS.recipes()[0]);
        let expected = if first_ran { "not the recorded release" } else { CARD_FILE };
        assert!(String::from_utf8_lossy(&index.stderr).contains(expected), "{index:?}");
        assert!(index.stdout.is_empty(), "{index:?}");
    }
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}
