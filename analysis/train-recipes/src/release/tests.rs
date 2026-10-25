use std::collections::BTreeSet;
use std::path::Path;

use turbo_zoo::{Component, Format, Kind, LicenceClass, TrainingData};

use super::*;
use crate::card::{CARD_FILE, GATE_FILE, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION};
use crate::langdetect::{self, MODEL_ID, SENTENCES_FILE, ZOO_LANGUAGE};
use crate::langdetect_web::{self, DOCUMENTS_FILE};
use crate::treebank::TextRights;
use crate::{masc, tatoeba};

#[test]
fn every_release_is_tagged_once_and_names_recipes_in_their_order() {
    let tags: BTreeSet<&str> = MODELS_RELEASES.iter().map(|r| r.tag).collect();
    assert_eq!(tags.len(), MODELS_RELEASES.len());
    for release in MODELS_RELEASES {
        let tag = release.tag;
        assert!(tag.starts_with(|c: char| c.is_ascii_alphanumeric()), "{tag}");
        assert!(tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')), "{tag}");
        assert!(!tag.contains("..") && !tag.ends_with('.') && !tag.ends_with(".lock"), "{tag}");
        let masc = release.recipes().into_iter().map(|r| r.model_id());
        let ud = release.ud_recipes().into_iter().map(|r| r.model_id);
        let language = release.language_recipes().into_iter().map(|r| r.model_id);
        let web = release.web_language_recipes().into_iter().map(|r| r.model_id);
        let ids: Vec<&str> = masc.chain(ud).chain(language).chain(web).collect();
        assert_eq!(ids, release.models, "{tag}: a model no recipe trains, or out of order");
        assert_eq!(ModelsRelease::tagged(tag).unwrap(), release);
    }
    assert_eq!(
        MASC_MODELS.url("turbo-en-masc-3.0.0-pos-1.bin"),
        "https://github.com/ai-pipestream/turboembed/releases/download/models-masc-3.0.0-1/\
         turbo-en-masc-3.0.0-pos-1.bin"
    );
    for unknown in ["", "--check", "models-masc-3.0.0-2", "../models-masc-3.0.0-1"] {
        match ModelsRelease::tagged(unknown) {
            Err(Error::UnknownRelease { tag, known }) => {
                assert_eq!(tag, unknown);
                assert_eq!(known, [MASC_MODELS.tag, UD_MODELS.tag, TATOEBA_MODELS.tag, FINEWEB_MODELS.tag]);
            }
            other => panic!("{unknown:?}: {other:?}"),
        }
    }
}

/// The UD release holds every UD recipe's model but those of a
/// treebank whose text rights are not settled and those that failed
/// their gates, the Italian ones (analysis/train-recipes/README.md).
#[test]
fn the_ud_release_leaves_out_porttinari_and_markit() {
    for recipe in UD_RECIPES {
        let released = UD_MODELS.models.contains(&recipe.model_id);
        let settled = !matches!(recipe.treebank.text_rights, TextRights::NotSettled { .. });
        let italian = recipe.treebank.language == "it";
        assert_eq!(released, settled && !italian, "{}", recipe.model_id);
    }
}

/// The zoo lists `id`'s model at its address in `release`, as its card
/// makes the entry: permissive, needing no acceptance, trained on
/// `trained_on` and nothing else, its files the model and then
/// `extra`, each at its address in the release.
fn listed(
    index: &Index,
    release: &ModelsRelease,
    id: &str,
    component: Component,
    language: &str,
    trained_on: &[TrainingData],
    extra: &[&str],
) {
    let entry = index.get(id).unwrap();
    assert_eq!(entry.kind, Kind::Model, "{id}");
    assert_eq!(entry.component, Some(component), "{id}");
    assert_eq!(entry.language.as_ref().unwrap().as_str(), language, "{id}");
    assert_eq!(entry.format, Format::OpennlpModel, "{id}");
    assert_eq!(entry.version, MODEL_VERSION, "{id}");
    assert_eq!(entry.license, MODEL_LICENCE, "{id}");
    assert_eq!(entry.license_terms.as_str(), MODEL_LICENCE_TERMS, "{id}");
    assert_eq!(entry.class, LicenceClass::Permissive, "{id}");
    assert!(entry.licences_to_accept(turbo_zoo::Policy::default()).is_empty(), "{id}");
    assert_eq!(entry.trained_on, trained_on, "{id}");
    let files: Vec<String> =
        std::iter::once(format!("{id}.bin")).chain(extra.iter().map(|f| (*f).to_owned())).collect();
    assert_eq!(entry.files.len(), files.len(), "{id}");
    for (listed, file) in entry.files.iter().zip(&files) {
        assert_eq!(listed.path.as_str(), file, "{id}");
        assert_eq!(listed.url.as_str(), release.url(file), "{id}");
    }
    assert!(index.shippable().any(|a| a.id.as_str() == id), "{id}");
}

/// The zoo lists every model of each release at its address there, and
/// nothing else at the release's addresses. That the bytes there are
/// the recipes' is what `turbo-train-recipes zoo-index --check` checks
/// on a run.
#[test]
fn the_builtin_index_lists_every_model_at_its_release_address() {
    let index = Index::builtin();
    for recipe in MASC_MODELS.recipes() {
        listed(&index, &MASC_MODELS, recipe.model_id(), recipe.component(), "en", &[masc::training_data()], &[]);
    }
    for recipe in UD_MODELS.ud_recipes() {
        let (language, trained_on) = (recipe.treebank.language, recipe.treebank.training_data());
        listed(&index, &UD_MODELS, recipe.model_id, recipe.zoo_component(), language, &[trained_on], &[]);
    }
    let export = langdetect::EXPORT.digest().unwrap();
    let tatoeba = TrainingData {
        name: format!("Tatoeba export {}", &export[..16]),
        license: turbo_zoo::Licence::CcBy20Fr,
        license_terms: tatoeba::TERMS_URL.parse().unwrap(),
    };
    assert_eq!(langdetect::training_data(export), tatoeba);
    let (id, component) = (MODEL_ID, Component::LanguageDetector);
    listed(&index, &TATOEBA_MODELS, id, component, ZOO_LANGUAGE, std::slice::from_ref(&tatoeba), &[SENTENCES_FILE]);
    let web = langdetect_web::training_data(export);
    assert_eq!(web[0], tatoeba);
    assert_eq!(web.len(), 3);
    let id = langdetect_web::MODEL_ID;
    listed(&index, &FINEWEB_MODELS, id, component, ZOO_LANGUAGE, &web, &[SENTENCES_FILE, DOCUMENTS_FILE]);
    for release in MODELS_RELEASES {
        assert_eq!(listed_at(&index, &release), release.models, "{}", release.tag);
    }
    assert!(MASC_MODELS.recipes().iter().any(|r| r.component() == Component::Chunker));
}

/// The built-in index with its own entries of a release has no
/// differences; one with an entry changed, missing or added at the
/// release's addresses has one line for each.
#[test]
fn differences_name_every_entry_listed_otherwise_and_every_other_model_at_the_release() {
    let builtin = Index::builtin();
    let entries: Vec<Artifact> = MASC_MODELS.models.iter().map(|id| builtin.get(id).unwrap().clone()).collect();
    assert!(differences(&builtin, &entries, &MASC_MODELS).is_empty());

    // Omitting a model entry that the release publishes is flagged.
    let partial = &entries[1..];
    assert_eq!(
        differences(&builtin, partial, &MASC_MODELS),
        [format!("{}: release {} publishes it, but no entry was given", entries[0].id, MASC_MODELS.tag)]
    );

    let mut changed = entries.clone();
    changed[1].files[0].size += 1;
    let lines = differences(&builtin, &changed, &MASC_MODELS);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with(&format!("{}: the built-in index lists another entry (", entries[1].id)));

    let mut extra = entries[0].clone();
    extra.id = "turbo-en-masc-3.0.0-extra-1".parse().unwrap();
    extra.files[0].path = "turbo-en-masc-3.0.0-extra-1.bin".parse().unwrap();
    extra.files[0].url = MASC_MODELS.url("turbo-en-masc-3.0.0-extra-1.bin").parse().unwrap();
    let listed: Vec<Artifact> =
        builtin.list().iter().filter(|a| a.id != entries[0].id).cloned().chain([extra]).collect();
    let index = Index::parse(Index::to_canonical_json(&listed).as_bytes()).unwrap();
    assert_eq!(
        differences(&index, &entries, &MASC_MODELS),
        [
            format!("{}: the built-in index does not list it", entries[0].id),
            "turbo-en-masc-3.0.0-extra-1: the built-in index lists it at an address of release \
             models-masc-3.0.0-1, which does not publish it"
                .to_owned(),
        ]
    );
}

/// The workflow that publishes `release` names its tag and [`RELEASES`],
/// publishes through models-publish.yml, and fetches the references the
/// release's recipes read from the addresses `references_file` lists,
/// checked against the SHA-256s it lists.
fn published_by(release: &ModelsRelease, workflow_file: &str, references_file: &str) {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows").join(workflow_file)).unwrap();
    let lines: Vec<&str> = workflow.lines().map(str::trim).collect();
    assert!(lines.contains(&format!("TAG: {}", release.tag).as_str()), "{workflow_file}'s TAG");
    assert!(lines.contains(&format!("RELEASES: {RELEASES}").as_str()), "{workflow_file}'s RELEASES");
    assert!(workflow.contains(&format!("analysis/train-recipes/{references_file}")), "{workflow_file}");
    assert!(lines.contains(&"uses: ./.github/workflows/models-publish.yml"), "{workflow_file} publishes");

    let listed = std::fs::read_to_string(crate_dir.join(references_file)).unwrap();
    let listed: Vec<(String, String)> = listed
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (sha256, url) = l.split_once(' ').unwrap();
            (sha256.to_owned(), url.to_owned())
        })
        .collect();
    assert_eq!(listed, references(release).unwrap(), "{references_file}");
    // The workflow saves each file under the address's last segment,
    // the name a run reads it by.
    for reference in release.recipe_references() {
        let (sha256, file) = (reference.sha256().unwrap(), reference.file().unwrap());
        assert!(
            listed.iter().any(|(s, url)| *s == sha256 && url.rsplit('/').next() == Some(file.as_str())),
            "{}",
            reference.id()
        );
    }
}

#[test]
fn the_models_workflows_publish_the_releases_and_fetch_their_references() {
    published_by(&MASC_MODELS, "models.yml", "masc-references.txt");
    published_by(&UD_MODELS, "models-ud.yml", "ud-references.txt");
    published_by(&TATOEBA_MODELS, "models-tatoeba.yml", "tatoeba-references.txt");
    published_by(&FINEWEB_MODELS, "models-fineweb.yml", "fineweb-references.txt");
}

/// The `models-tatoeba` workflow names the release that keeps the
/// recorded export and reads `tatoeba-export.txt`, which lists the
/// export's two archives at their addresses in that release, each
/// under the name Tatoeba publishes it by, with a SHA-256 each; and it
/// compares the model's four files between its two runs.
#[test]
fn the_tatoeba_workflow_trains_on_the_kept_export() {
    let tag = TATOEBA_EXPORT_RELEASE;
    assert!(tag.starts_with(|c: char| c.is_ascii_alphanumeric()), "{tag}");
    assert!(tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')), "{tag}");
    assert!(MODELS_RELEASES.iter().all(|r| r.tag != tag));

    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows/models-tatoeba.yml")).unwrap();
    let lines: Vec<&str> = workflow.lines().map(str::trim).collect();
    assert!(lines.contains(&format!("DATA_TAG: {tag}").as_str()));
    assert!(workflow.contains("analysis/train-recipes/tatoeba-export.txt"));
    let listed = std::fs::read_to_string(crate_dir.join("tatoeba-export.txt")).unwrap();
    let listed: Vec<(&str, &str)> =
        listed.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(|l| l.split_once(' ').unwrap()).collect();
    let urls: Vec<String> = [tatoeba::DETAILED_ARCHIVE_URL, tatoeba::CC0_ARCHIVE_URL]
        .map(|u| format!("{RELEASES}{tag}/{}", u.rsplit('/').next().unwrap()))
        .to_vec();
    assert_eq!(listed.iter().map(|(_, url)| (*url).to_owned()).collect::<Vec<_>>(), urls);
    for (sha256, url) in &listed {
        assert!(sha256.len() == 64 && sha256.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')), "{url}");
    }
    let index = Index::builtin();
    let dataset = index.get("tatoeba-2026-10-10").unwrap();
    assert_eq!(dataset.kind, Kind::Dataset);
    assert_eq!(dataset.format, Format::TarBz2);
    let dataset_files: Vec<(String, String)> =
        dataset.files.iter().map(|f| (f.sha256.to_string(), f.url.to_string())).collect();
    let expected_files: Vec<(String, String)> =
        listed.iter().map(|(sha, url)| ((*sha).to_owned(), (*url).to_owned())).collect();
    assert_eq!(dataset_files, expected_files);
    // The runs write the model's directory under its id.
    assert!(lines.contains(&format!("id={MODEL_ID}").as_str()));
    assert!(workflow.contains(&format!("\"$id.bin\" {SENTENCES_FILE} {CARD_FILE} {GATE_FILE}")));
}
