//! The zoo entries a release's model directories make.

use turbo_eval::model_gate::{ModelGateRecord, ModelId};
use turbo_train_recipes::Error;
use turbo_train_recipes::card::{CARD_FILE, GATE_FILE, ModelCard};
use turbo_train_recipes::release::{self, MASC_MODELS};
use turbo_zoo::Index;

/// A model directory as `turbo-train-recipes masc` writes it for a card
/// that may become a zoo entry.
pub(crate) struct Released {
    pub(crate) card: ModelCard,
    pub(crate) gate: ModelGateRecord,
    pub(crate) model: Vec<u8>,
}

/// The model directories of `models` (every recipe's card that may become
/// a zoo entry, with its gate record and model) make the zoo entries of
/// [`MASC_MODELS`] from the directories of its models, in the recipes'
/// order and each at its address there, and the
/// built-in index with them added; a missing directory, a card in another
/// model's directory, a gate record that is not the card's or a model
/// file that is not its card's makes none. The binary's `zoo-index` reads
/// the same directories: with `--check` it fails, since the test's models
/// are not the ones the zoo lists, and without it prints the index.
pub(crate) fn released_models_make_the_release_s_zoo_entries(dir: &std::path::Path, models: &[Released]) {
    let models: Vec<&Released> = models.iter().filter(|r| MASC_MODELS.models.contains(&r.card.id.as_str())).collect();
    assert_eq!(models.len(), MASC_MODELS.models.len());
    let write = |r: &Released| {
        let d = dir.join(&r.card.id);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(CARD_FILE), r.card.to_json()).unwrap();
        std::fs::write(d.join(GATE_FILE), r.gate.to_json()).unwrap();
        std::fs::write(d.join(&r.card.file.name), &r.model).unwrap();
    };
    let (first, rest) = models.split_first().unwrap();
    rest.iter().for_each(|r| write(r));
    assert!(matches!(release::entries(dir, &MASC_MODELS), Err(Error::Io { .. })), "a recipe's directory is missing");
    write(first);
    let entries = release::entries(dir, &MASC_MODELS).unwrap();
    let ids: Vec<&str> = entries.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(ids, MASC_MODELS.models);
    for (entry, r) in entries.iter().zip(&models) {
        assert_eq!(*entry, r.card.artifact(&MASC_MODELS.url(&r.card.file.name)).unwrap());
        assert!(entry.files[0].url.as_str().starts_with(release::RELEASES));
    }

    // Added to an index without them, they follow its artifacts; added to
    // the built-in index, which lists the MASC models trained on the
    // recorded release, each replaces the entry of its id where it is.
    let builtin = Index::builtin();
    let one = Index::parse(Index::to_canonical_json(&builtin.list()[..1]).as_bytes()).unwrap();
    let added = Index::parse(release::index_with(&one, &entries).as_bytes()).unwrap();
    let added_ids: Vec<&str> = added.list().iter().map(|a| a.id.as_str()).collect();
    assert_eq!(added_ids[0], builtin.list()[0].id.as_str());
    assert_eq!(added_ids[1..], ids);
    let replaced = Index::parse(release::index_with(&builtin, &entries).as_bytes()).unwrap();
    assert_eq!(replaced.list().len(), builtin.list().len());
    for (now, before) in replaced.list().iter().zip(builtin.list()) {
        assert_eq!(now.id, before.id);
        match entries.iter().find(|e| e.id == now.id) {
            Some(entry) => assert_eq!(now, entry),
            None => assert_eq!(now, before),
        }
    }

    let refused_field = |expected: &str| match release::entries(dir, &MASC_MODELS) {
        Err(Error::InvalidCard { field, .. }) if field == expected => {}
        other => panic!("{expected}: {other:?}"),
    };
    // A card in another model's directory makes no entries; nor does a
    // gate record of another model, or one with another outcome than the
    // card's.
    let (second, first_dir) = (models[1], dir.join(first.card.id.as_str()));
    std::fs::write(first_dir.join(CARD_FILE), second.card.to_json()).unwrap();
    refused_field("id");
    write(first);
    std::fs::write(first_dir.join(GATE_FILE), second.gate.to_json()).unwrap();
    refused_field("gate");
    let mut other_reference = first.gate.clone();
    other_reference.reference = ModelId { id: "weak".to_owned(), sha256: first.gate.reference.sha256.clone() };
    std::fs::write(first_dir.join(GATE_FILE), other_reference.to_json()).unwrap();
    refused_field("gate");
    write(first);
    // Nor does a model file changed after its card was written.
    let mut changed = second.model.clone();
    *changed.last_mut().unwrap() ^= 1;
    write(&Released { card: second.card.clone(), gate: second.gate.clone(), model: changed });
    match release::entries(dir, &MASC_MODELS) {
        Err(Error::Zoo(turbo_zoo::ZooError::Integrity { id, .. })) => assert_eq!(id.as_str(), second.card.id),
        other => panic!("{other:?}"),
    }
    write(second);

    // The binary: the test's models are not the ones the zoo lists, so
    // `--check` exits with 1 and names each, with the size and SHA-256 of
    // both; without it the index with them is printed. A tag that names
    // no release is refused.
    let zoo_index = |extra: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_turbo-train-recipes"))
            .arg("zoo-index")
            .arg("--models")
            .arg(dir)
            .args(extra)
            .output()
            .unwrap()
    };
    let checked = zoo_index(&["--release", MASC_MODELS.tag, "--check"]);
    assert_eq!(checked.status.code(), Some(1), "{checked:?}");
    assert!(checked.stderr.is_empty(), "{checked:?}");
    let stdout = String::from_utf8(checked.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), entries.len() + 1, "{stdout}");
    for (line, entry) in lines.iter().zip(&entries) {
        let listed = &builtin.get(entry.id.as_str()).unwrap().files[0];
        let model = &entry.files[0];
        assert_eq!(
            *line,
            format!(
                "{}: the built-in index lists another entry ({} bytes, SHA-256 {}, at {}; the model has {} bytes, \
                 SHA-256 {}, at {})",
                entry.id, listed.size, listed.sha256, listed.url, model.size, model.sha256, model.url
            )
        );
    }
    assert_eq!(
        lines[entries.len()],
        format!("the zoo's built-in index is not the one these models and release {} give", MASC_MODELS.tag)
    );
    let printed = zoo_index(&["--release", MASC_MODELS.tag]);
    assert_eq!(printed.status.code(), Some(0), "{printed:?}");
    assert_eq!(String::from_utf8(printed.stdout).unwrap(), release::index_with(&builtin, &entries));
    for (extra, message) in [
        (&["--release", "../x", "--check"][..], "no release of the models is tagged \"../x\""),
        (&["--release", "--check"][..], "no release of the models is tagged \"--check\""),
        (&["--check"][..], "--release is required"),
    ] {
        let refused = zoo_index(extra);
        assert_eq!(refused.status.code(), Some(2), "{extra:?}: {refused:?}");
        assert!(String::from_utf8_lossy(&refused.stderr).contains(message), "{extra:?}: {refused:?}");
        assert!(refused.stdout.is_empty(), "{extra:?}: {refused:?}");
    }
}
