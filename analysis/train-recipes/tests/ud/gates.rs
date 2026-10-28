use super::*;

#[test]
fn a_tagger_s_gate_leaves_out_the_tags_the_reference_cannot_give() {
    let dir = temp_dir("left-out");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 400, "train"), &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    let r = recipe(treebank, Component::Pos, spanish(Component::Pos));
    // A reference trained without the interjection has no INTJ outcome.
    let reference = r.train(&conllu(NO_INTJ, 200, "no-intj")).unwrap();
    let tagger = r.tagger().unwrap();
    let reference_tags = tagger.reference_tags(&reference.bytes).unwrap();
    let UdSamples::Tags(samples) = r.held_out(&text).unwrap().samples else { panic!() };
    let held_out = HeldOutTags::new(samples, 0);
    let intj = TagCount { tag: "INTJ".to_owned(), tokens: 40 };
    // One in each of the 40 copies of the third sentence.
    assert_eq!(reference_tags.left_out(&held_out), std::slice::from_ref(&intj));
    r.check_data(&text, &reference.bytes).unwrap();

    let outcome = run_ud(&r, &text, &reference.bytes).unwrap();
    let record = &outcome.gate;
    record.check().unwrap();
    let tokens = held_out.tokens();
    let scoring = TagScoring { tokens, left_out: 40, reference_mappings: Vec::new() };
    assert_eq!(record.set.scoring, Scoring::Tags(scoring));
    assert_eq!(record.set.samples_sha256, held_out.sha256(&reference_tags));
    // Both models are counted on the same tokens: every one but the
    // interjections.
    let counted: u64 = record.units.iter().map(|u| u.reference.target).sum();
    assert_eq!(counted, tokens - 40);
    assert!(record.units.iter().all(|u| u.candidate.target == u.reference.target));
    let card = &outcome.card;
    let Some(TokenCoverage { tokens: on_card, reference_mappings, left_out, reference_mismatches }) =
        &card.held_out.tokens
    else {
        panic!("{:?}", card.held_out)
    };
    assert_eq!((*on_card, reference_mappings.is_empty(), left_out), (tokens, true, &vec![intj]));
    assert!(reference_mismatches.len() <= 10, "{reference_mismatches:?}");
    assert!(reference_mismatches.windows(2).all(|w| w[0].tokens >= w[1].tokens), "{reference_mismatches:?}");
    assert!(reference_mismatches.iter().all(|m| m.held_out != "INTJ"), "left-out tokens are not scored");
    assert_eq!(
        (card.gate.candidate_tag_format, card.gate.reference_tag_format),
        (Some(TagFormat::Custom), Some(TagFormat::Custom))
    );
    let summary = card.summary().to_string();
    assert!(
        summary.ends_with(&format!(
            "{}: 40 of {tokens} held-out tokens left out, their tags not the reference's: INTJ (40)",
            r.model_id
        )),
        "{summary}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_lemmatizer_s_gate_leaves_out_the_lemma_classes_the_reference_cannot_give() {
    let dir = temp_dir("left-out-lemmas");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 400, "train"), &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    let r = recipe(treebank, Component::Lemmas, spanish(Component::Lemmas));
    // A reference trained without the second sentence has no class for
    // "Vamos" to "ir" nor for "al" to "a+el".
    let without_second: Vec<usize> = (0..POOL.len()).filter(|&i| i != 1).collect();
    let reference = r.train(&conllu(&without_second, 200, "no-second")).unwrap();
    let classes = ReferenceClasses::of_model(&reference.bytes).unwrap();
    assert!(!classes.scores("Vamos", "ir") && !classes.scores("al", "a+el") && classes.scores("del", "de+el"));
    let held_out = r.held_out(&text).unwrap();
    let UdSamples::Lemmas(samples) = &held_out.samples else { panic!() };
    let tokens = held_out_tokens(&held_out);
    // One of each in each of the 40 copies of the second sentence.
    let count = |tag: &str| TagCount { tag: tag.to_owned(), tokens: 40 };
    assert_eq!(classes.left_out(samples), [count("ADP+DET"), count("VERB")]);
    r.check_data(&text, &reference.bytes).unwrap();

    let outcome = run_ud(&r, &text, &reference.bytes).unwrap();
    let record = &outcome.gate;
    record.check().unwrap();
    assert_eq!(record.set.scoring, Scoring::Tags(TagScoring { tokens, left_out: 80, reference_mappings: Vec::new() }));
    assert_eq!(record.set.samples_sha256, lemma_samples_sha256(samples, &classes));
    // Both models are counted on the same tokens: every one but those two.
    let counted: u64 = record.units.iter().map(|u| u.reference.target).sum();
    assert_eq!(counted, tokens - 80);
    assert!(record.units.iter().all(|u| u.candidate.target == u.reference.target));
    assert_eq!(
        score_lemmas(&outcome.trained.bytes, samples, &classes).unwrap(),
        record.units.iter().map(|u| u.candidate).collect::<Vec<_>>()
    );
    // Scored alone, the candidate has every class, so every token counts.
    let alone = r.score(&outcome.trained.bytes, &held_out).unwrap();
    assert_eq!(alone.iter().map(|u| u.target).sum::<u64>(), tokens);
    let card = &outcome.card;
    let Some(TokenCoverage { tokens: on_card, reference_mappings, left_out, reference_mismatches }) =
        &card.held_out.tokens
    else {
        panic!("{:?}", card.held_out)
    };
    assert_eq!(
        (*on_card, reference_mappings.is_empty(), left_out),
        (tokens, true, &vec![count("ADP+DET"), count("VERB")])
    );
    assert!(reference_mismatches.len() <= 10, "{reference_mismatches:?}");
    assert!(reference_mismatches.iter().all(|m| m.held_out != "ir" && m.held_out != "a+el"), "left out, not scored");
    let summary = card.summary().to_string();
    assert!(
        summary.ends_with(&format!(
            "{}: 80 of {tokens} held-out tokens left out, their lemmas not the reference's, by tag: ADP+DET (40), \
             VERB (40)",
            r.model_id
        )),
        "{summary}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// More held-out sentences skipped than `MAX_SKIPPED_SENTENCES` allows
/// fail the check the binary runs before training, and the run.
#[test]
fn skipped_sentences_beyond_the_limit_are_refused_before_training() {
    let dir = temp_dir("skipped");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let empty = "# sent_id = empty\n# text = _\n1.1\t_\t_\tPRON\t_\t_\t_\t_\t_\t_\n\n";
    let test = format!("{}{empty}", conllu(&all, 100, "test"));
    let treebank = write_treebank(&dir, &conllu(&all, 100, "train"), &test);
    let text = treebank.read(&dir).unwrap();
    for component in [Component::Pos, Component::Lemmas] {
        let r = recipe(treebank, component, spanish(component));
        let held_out_skipped = |result: Result<(), Error>| {
            matches!(result, Err(Error::TooManyLeftOut { what: LeftOut::HeldOutSentences, count: 1, of: 101, .. }))
        };
        assert!(held_out_skipped(r.check_data(&text, b"not read")));
        assert!(held_out_skipped(r.held_out(&text).map(|_| ())));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A card's zoo entry: only for a recipe of `UD_RECIPES`, gated against
/// that recipe's reference, and passed, and never for a treebank whose
/// text rights are not settled. The Italian models, gated against the
/// recorded local-only references, fail their gates, so they get none;
/// the Portuguese ones get none whatever their gates say.
#[test]
fn a_card_is_shippable_only_when_gated_against_its_recipe_s_reference_and_passed() {
    let dir = temp_dir("shippable");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 100, "train"), &conllu(&all, 40, "test"));
    let text = treebank.read(&dir).unwrap();
    let r = recipe(treebank, Component::Lemmas, spanish(Component::Lemmas));
    let weak = r.train(&conllu(NO_INTJ, WEAK_LEMMAS, "weak")).unwrap().bytes;
    let mut card = run_ud(&r, &text, &weak).unwrap().card;
    card.id = "turbo-es-ud-ancora-2.16-lemmas-1".to_owned();
    let refused = |card: &UdModelCard, why: &str| matches!(card.artifact("https://example.org/m.bin"), Err(Error::NotShippable { reason, .. }) if reason == why);
    assert!(refused(&card, "it was not gated against its recipe's reference"));
    let zoo = turbo_zoo::Index::builtin();
    let reference = &zoo.get("opennlp-es-ud-gsd-lemmas-1.3-2.5.4").unwrap().files[0];
    card.gate.reference = "opennlp-es-ud-gsd-lemmas-1.3-2.5.4".to_owned();
    card.gate.reference_sha256 = reference.sha256.to_string();
    card.gate.pass = false;
    assert!(refused(&card, "it failed its gate"));
    card.gate.pass = true;
    // The card names the recipe's model, but the model was trained and
    // gated on the test treebank: refused whichever of its training data
    // and its treebank still names the test treebank.
    let not_trained_there = "it was not trained and gated on its recipe's treebank";
    assert!(refused(&card, not_trained_there));
    let test_trained_on = card.trained_on.clone();
    card.trained_on = vec![SPANISH_ANCORA.training_data()];
    assert!(refused(&card, not_trained_there));
    card.treebank = TreebankCard::of(&SPANISH_ANCORA);
    card.trained_on = test_trained_on;
    assert!(refused(&card, not_trained_there));
    card.trained_on = vec![SPANISH_ANCORA.training_data()];
    card.treebank.test.sha256 = treebank.test.sha256.to_owned();
    assert!(refused(&card, not_trained_there));
    card.treebank = TreebankCard::of(&SPANISH_ANCORA);
    // Nor does a card of another component or language than its recipe's.
    let invalid = |card: &UdModelCard, expected: &str| matches!(card.artifact("https://example.org/m.bin"), Err(Error::InvalidCard { field, .. }) if field == expected);
    let mut other = card.clone();
    other.component = turbo_zoo::Component::Pos;
    assert!(invalid(&other, "component"));
    let mut other = card.clone();
    other.language = "ca".to_owned();
    assert!(invalid(&other, "language"));
    let artifact = card.artifact("https://example.org/m.bin").unwrap();
    assert_eq!(artifact.version, MODEL_VERSION);
    assert_eq!(artifact.class, LicenceClass::Permissive);
    card.class = LicenceClass::ShareAlike;
    assert!(matches!(card.artifact("https://example.org/m.bin"), Err(Error::InvalidCard { field: "class", .. })));

    let mut italian = card.clone();
    italian.class = LicenceClass::Permissive;
    italian.id = "turbo-it-ud-markit-2.16-pos-1".to_owned();
    (italian.gate.reference, italian.gate.reference_sha256) =
        (IT_UD_VIT_POS.id.to_owned(), IT_UD_VIT_POS.sha256.to_owned());
    italian.gate.pass = false;
    assert!(refused(&italian, "it failed its gate"));

    // Porttinari's text is Folha de São Paulo's, with no licence shown: a
    // Portuguese model gated against its reference and passed is refused,
    // and so is one that failed.
    let mut portuguese = card.clone();
    portuguese.class = LicenceClass::Permissive;
    portuguese.id = "turbo-pt-ud-porttinari-2.16-lemmas-1".to_owned();
    let pt = &zoo.get("opennlp-pt-ud-gsd-lemmas-1.3-2.5.4").unwrap().files[0];
    (portuguese.gate.reference, portuguese.gate.reference_sha256) =
        ("opennlp-pt-ud-gsd-lemmas-1.3-2.5.4".to_owned(), pt.sha256.to_string());
    assert!(portuguese.gate.pass);
    let unsettled = "the rights to its treebank's text are not settled";
    assert!(refused(&portuguese, unsettled));
    portuguese.gate.pass = false;
    assert!(refused(&portuguese, unsettled));

    // A card of another schema version is refused by its version alone.
    let path = dir.join("card.json");
    std::fs::write(&path, r#"{"card_version": 1, "id": "x"}"#).unwrap();
    assert!(matches!(UdModelCard::read(&path), Err(Error::CardVersion { found: 1, expected: UD_CARD_VERSION, .. })));
    // So is a card as version 3 wrote it, without the chunk and name
    // counts of the MASC card's parts, though it reads to the same fields.
    let mut three: serde_json::Value = serde_json::from_str(&card.to_json()).unwrap();
    three["card_version"] = 3.into();
    for part in ["training", "held_out"] {
        for field in ["chunks", "names"] {
            three[part].as_object_mut().unwrap().remove(field);
        }
    }
    std::fs::write(&path, serde_json::to_string_pretty(&three).unwrap()).unwrap();
    assert!(matches!(UdModelCard::read(&path), Err(Error::CardVersion { found: 3, .. })));
    std::fs::remove_dir_all(dir).unwrap();
}

/// The zoo entries of release `models-ud-2.16-1` come from the
/// directories of its models. The models here are trained on the test
/// treebank, each gated against itself, and their cards and gate records
/// rewritten to name a recipe of the release, its treebank and its
/// reference; a card left naming the test treebank makes no entry.
#[test]
fn the_ud_release_s_zoo_entries_come_from_its_models_directories() {
    let dir = temp_dir("release");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let train = conllu(&all, 100, "train");
    let treebank = write_treebank(&dir, &train, &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    let out = dir.join("models");
    let mut released = Vec::new();
    for component in COMPONENTS {
        let r = recipe(treebank, component, spanish(component));
        let model = r.train(&train).unwrap().bytes;
        let outcome = run_ud(&r, &text, &model).unwrap();
        assert!(outcome.gate.result.pass, "{}", outcome.gate);
        for u in UD_MODELS.ud_recipes().into_iter().filter(|u| u.component == component) {
            let (mut card, mut gate) = (outcome.card.clone(), outcome.gate.clone());
            card.id = u.model_id.to_owned();
            card.file.name = format!("{}.bin", u.model_id);
            card.language = u.treebank.language.to_owned();
            card.trained_on = vec![u.treebank.training_data()];
            card.treebank = TreebankCard::of(&u.treebank);
            gate.candidate.id = u.model_id.to_owned();
            gate.reference = ModelId { id: u.reference.id().to_owned(), sha256: u.reference.sha256().unwrap() };
            (card.gate.reference, card.gate.reference_sha256) =
                (gate.reference.id.clone(), gate.reference.sha256.clone());
            write_ud_model_dir(&out, &card, &outcome.trained.bytes, &gate).unwrap();
            released.push(card);
        }
    }
    assert_eq!(released.len(), UD_MODELS.models.len());

    let entries = release::entries(&out, &UD_MODELS).unwrap();
    let ids: Vec<&str> = entries.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(ids, UD_MODELS.models);
    for entry in &entries {
        let card = released.iter().find(|c| c.id == entry.id.as_str()).unwrap();
        assert_eq!(*entry, card.artifact(&UD_MODELS.url(&card.file.name)).unwrap());
    }
    // Added to the built-in index, which lists the release's models, each
    // replaces the entry of its id where it is.
    let builtin = Index::builtin();
    let index = release::index_with(&builtin, &entries);
    let replaced = Index::parse(index.as_bytes()).unwrap();
    let listed = |i: &Index| i.list().iter().map(|a| a.id.to_string()).collect::<Vec<_>>();
    assert_eq!(listed(&replaced), listed(&builtin));
    for entry in &entries {
        assert_eq!(replaced.get(entry.id.as_str()).unwrap(), entry);
    }

    // A card of the release's model that names the test treebank makes no
    // entry.
    let first = &released[0];
    let card_file = out.join(&first.id).join(CARD_FILE);
    let mut unmoved = first.clone();
    unmoved.treebank = TreebankCard::of(&treebank);
    std::fs::write(&card_file, unmoved.to_json()).unwrap();
    match release::entries(&out, &UD_MODELS) {
        Err(Error::NotShippable { id, reason }) => {
            assert_eq!(id, first.id);
            assert_eq!(reason, "it was not trained and gated on its recipe's treebank");
        }
        other => panic!("{other:?}"),
    }
    std::fs::write(&card_file, first.to_json()).unwrap();

    // The binary: the test's models are not the ones the zoo lists, so
    // `--check` exits with 1 and names each; without it the index with
    // them is printed.
    let zoo_index = |extra: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_turbo-train-recipes"))
            .args(["zoo-index", "--models"])
            .arg(&out)
            .args(["--release", UD_MODELS.tag])
            .args(extra)
            .output()
            .unwrap()
    };
    let checked = zoo_index(&["--check"]);
    assert_eq!(checked.status.code(), Some(1));
    let stdout = String::from_utf8(checked.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), UD_MODELS.models.len() + 1, "{stdout}");
    for (line, id) in lines.iter().zip(UD_MODELS.models) {
        assert!(line.starts_with(&format!("{id}: the built-in index lists another entry (")), "{line}");
    }
    assert_eq!(
        lines[UD_MODELS.models.len()],
        "the zoo's built-in index is not the one these models and release models-ud-2.16-1 give"
    );
    let printed = zoo_index(&[]);
    assert_eq!(printed.status.code(), Some(0));
    assert_eq!(String::from_utf8(printed.stdout).unwrap(), index);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_treebank_whose_files_or_licence_changed_is_refused() {
    let dir = temp_dir("changed");
    let treebank = write_treebank(&dir, &conllu(&[0], 3, "train"), &conllu(&[1], 3, "test"));
    treebank.read(&dir).unwrap();
    std::fs::write(dir.join("es_test-ud-test.conllu"), conllu(&[2], 3, "test")).unwrap();
    assert!(matches!(treebank.read(&dir), Err(Error::TreebankFile { file: "es_test-ud-test.conllu", .. })));
    std::fs::write(dir.join("LICENSE.txt"), LICENCE.replace("Attribution 4.0", "Attribution-ShareAlike 4.0")).unwrap();
    assert!(matches!(treebank.read(&dir), Err(Error::Licence { .. })));
    std::fs::remove_dir_all(dir).unwrap();
}

/// Whether CI asks for the published models to be present.
fn models_required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

/// The directory of the OpenNLP UD 1.3 models the UD recipes are gated
/// against. CI fetches only some of them, so they are read from
/// `TURBO_TEST_OPENNLP_MODELS`, as the other local-only references are:
/// without it the tests that need them are skipped, and with it a missing
/// file fails when the models are required.
fn references_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("TURBO_TEST_OPENNLP_MODELS").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("TURBO_TEST_OPENNLP_MODELS is not set: skipping the UD references");
    }
    dir
}

/// Each UD recipe's reference, where present: it matches the zoo's digest
/// (the recorded size and digest for the Italian ones) and is scored on
/// the test treebank's held-out samples of its component. The text is the
/// tests' Spanish for every language, so the scores say nothing; this
/// checks that every reference loads and is scored.
#[test]
fn the_published_references_are_checked_and_scored_when_present() {
    let Some(refs) = references_dir() else { return };
    let dir = temp_dir("published");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 10, "train"), &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    for r in UD_RECIPES {
        let reference = r.reference;
        let path = refs.join(reference.file().unwrap());
        if !path.is_file() {
            assert!(!models_required(), "{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
            eprintln!("{} not found: skipping", path.display());
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        reference.check(&bytes).unwrap();
        let test = recipe(treebank, r.component, r.reference);
        let held_out = test.samples(&text.test, treebank.test.name).unwrap();
        let counts = test.score(&bytes, &held_out).unwrap();
        assert_eq!(counts.len(), held_out.samples.len());
        let right: u64 = counts.iter().map(|c| c.true_positives).sum();
        let target: u64 = counts.iter().map(|c| c.target).sum();
        println!("{}: {right} of {target} on the test sentences", reference.id());
        assert!(right > 0, "{}", reference.id());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// The `ud` command reads and checks every reference a run needs, and
/// checks that none of its model directories exists, then reads every
/// treebank, before it trains anything, and says what every problem with
/// the references and directories is, in the order of `UD_RECIPES`.
#[test]
fn the_ud_command_checks_every_reference_and_treebank_before_training() {
    let out = temp_dir("binary");
    let models = out.join("models");
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let run = |references: &Path, extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("ud").arg("--treebanks").arg(out.join("treebanks")).arg("--references").arg(references);
        c.arg("--out").arg(&models).args(extra);
        let output = c.output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        String::from_utf8_lossy(&output.stderr).into_owned()
    };
    assert!(run(&out, &["--recipe", "turbo-en-ud-ewt-2.16-pos-1"]).contains("no recipe trains"));
    // `fetch-ud` refuses a recipe there is none of before it fetches
    // anything.
    let fetch = std::process::Command::new(bin)
        .args(["fetch-ud", "--home"])
        .arg(out.join("zoo"))
        .args(["--recipe", "turbo-es-ud-ancora-2.16-pos-1", "--recipe", "turbo-en-ud-ewt-2.16-pos-1"])
        .output()
        .unwrap();
    assert_eq!(fetch.status.code(), Some(2), "{fetch:?}");
    assert!(String::from_utf8_lossy(&fetch.stderr).contains("no recipe trains turbo-en-ud-ewt-2.16-pos-1"));
    assert!(!out.join("zoo").exists());
    // No reference in the directory: every one is named, in the recipes'
    // order, and no treebank is read.
    let stderr =
        run(&out, &["--recipe", "turbo-it-ud-markit-2.16-lemmas-1", "--recipe", "turbo-es-ud-ancora-2.16-pos-1"]);
    let (spanish, italian) =
        (stderr.find("opennlp-es-ud-gsd-pos").unwrap(), stderr.find("opennlp-it-ud-vit-lemmas").unwrap());
    assert!(spanish < italian, "{stderr}");
    assert!(!stderr.contains("treebank"), "{stderr}");
    // Wrong files under the references' names are refused by their digests,
    // the zoo's or the recorded one, and a model directory already there
    // is named too.
    let refs = out.join("references");
    std::fs::create_dir_all(&refs).unwrap();
    for file in ["opennlp-fi-ud-tdt-pos-1.3-2.5.4.bin", "opennlp-it-ud-vit-pos-1.3-2.5.4.bin"] {
        std::fs::write(refs.join(file), "not the model").unwrap();
    }
    std::fs::create_dir_all(models.join("turbo-it-ud-markit-2.16-pos-1")).unwrap();
    let stderr = run(&refs, &["--recipe", "turbo-fi-ud-ftb-2.16-pos-1", "--recipe", "turbo-it-ud-markit-2.16-pos-1"]);
    assert_eq!(stderr.lines().count(), 3, "{stderr}");
    assert!(stderr.contains("reference model opennlp-fi-ud-tdt-pos-1.3-2.5.4: SHA-256"), "{stderr}");
    assert!(stderr.contains("reference model opennlp-it-ud-vit-pos-1.3-2.5.4: SHA-256"), "{stderr}");
    assert!(stderr.contains("turbo-it-ud-markit-2.16-pos-1: the model directory exists"), "{stderr}");
    std::fs::remove_dir_all(&models).unwrap();
    // With the reference in place, the missing treebank is refused before
    // training.
    match references_dir().map(|d| d.join("opennlp-it-ud-vit-pos-1.3-2.5.4.bin")) {
        Some(path) if path.is_file() => {
            std::fs::copy(&path, refs.join("opennlp-it-ud-vit-pos-1.3-2.5.4.bin")).unwrap();
            let stderr = run(&refs, &["--recipe", "turbo-it-ud-markit-2.16-pos-1"]);
            assert!(stderr.contains("ud-italian-markit-r2.16") && stderr.contains("LICENSE.txt"), "{stderr}");
            assert!(!models.exists(), "nothing is written before every check passes");
        }
        Some(path) => {
            assert!(!models_required(), "{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
            eprintln!("{} not found: the missing treebank's refusal is not checked", path.display());
        }
        None => {}
    }
    std::fs::remove_dir_all(out).unwrap();
}
