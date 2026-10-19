use std::path::Path;

use super::*;
use crate::card::{GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, ModelFileCard};
use crate::fineweb::WebPlan;
use crate::langdetect::{
    ACCEPTED_LICENCES, CARD_FILE, ExportCard, LanguageTrainingCard, Plan, SENTENCES_FILE, SPLIT, SentencesFileCard,
    ZOO_LANGUAGE,
};
use crate::tatoeba::ExportFile;
use turbo_zoo::{Licence, LicenceClass, class_of};

#[test]
fn the_recipe_is_the_tatoeba_recipe_under_its_own_id() {
    let r = LANGDETECT_WEB;
    assert_eq!(r.tatoeba.model_id, MODEL_ID);
    assert_eq!(r.tatoeba.reference.id(), LANGDETECT.reference.id());
    assert_eq!(r.tatoeba.split, LANGDETECT.split);
    assert_eq!(r.tatoeba.parameters(), LANGDETECT.parameters());
    assert_eq!(r.tatoeba.languages, LANGDETECT.languages);
    assert_eq!(r.web, WEB);
    assert_eq!(r.compared.id(), COMPARED.id());
    let id: turbo_zoo::ArtifactId = MODEL_ID.parse().unwrap();
    assert_eq!(id.as_str(), MODEL_ID);
    assert_eq!(COMPARED.id(), langdetect::MODEL_ID);
    // The mappings over every language are the Tatoeba recipe's pairs
    // whose codes differ.
    let m: Vec<(String, String)> = r.web_mappings().into_iter().map(|m| (m.held_out, m.reference)).collect();
    assert_eq!(m.len(), 7);
    assert!(m.contains(&("est".to_owned(), "ekk".to_owned())));
    assert!(m.contains(&("zsm".to_owned(), "msa".to_owned())));
}

/// fineweb-references.txt lists the reference and the compared model,
/// each with its SHA-256 and its address, and the workflow reads it.
#[test]
fn the_references_list_names_the_reference_and_the_compared_model() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let listed = std::fs::read_to_string(crate_dir.join("fineweb-references.txt")).unwrap();
    let listed: Vec<(&str, &str)> =
        listed.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(|l| l.split_once(' ').unwrap()).collect();
    let zoo = turbo_zoo::Index::builtin();
    let compared = &zoo.get(langdetect::MODEL_ID).unwrap().files[0];
    let reference = langdetect::LANGDETECT_183;
    assert_eq!(
        listed,
        [(reference.sha256, reference.url), (compared.sha256.to_string().as_str(), compared.url.as_str()),]
    );
    // Each is saved under its address's last segment, the name the run
    // reads it by.
    assert_eq!(reference.url.rsplit('/').next(), Some(reference.file));
    assert_eq!(compared.url.as_str().rsplit('/').next(), Some(COMPARED.file().unwrap().as_str()));
    let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows/models-fineweb.yml")).unwrap();
    assert!(workflow.contains("analysis/train-recipes/fineweb-references.txt"));
    assert!(workflow.contains(&format!("id={MODEL_ID}")));
    for f in [DOCUMENTS_FILE, SENTENCES_FILE, CARD_FILE, GATE_FILE, WEB_GATE_FILE, COMPARE_TATOEBA_FILE] {
        assert!(workflow.contains(f), "{f}");
    }
    assert!(workflow.contains(COMPARE_WEB_FILE));
    // The release publishes the card and each record as
    // `<model id>.<file>`.
    let records = [CARD_FILE, GATE_FILE, WEB_GATE_FILE, COMPARE_TATOEBA_FILE, COMPARE_WEB_FILE];
    let stems = records.map(|f| f.strip_suffix(".json").unwrap()).join(" ");
    assert!(workflow.contains(&format!("for f in {stems}; do")), "{stems}");
}

#[test]
fn the_training_data_is_permissive_with_attribution() {
    let data = training_data(&"0".repeat(64));
    assert_eq!(data.len(), 3);
    assert_eq!(data[1].license, Licence::OdcBy10);
    assert_eq!(data[1].name, "FineWeb-2 af9c13333eb9");
    assert_eq!(data[2].name, "FineWeb 9bb295ddab0e");
    let class = class_of(std::iter::once(MODEL_LICENCE).chain(data.iter().map(|t| t.license)));
    assert_eq!(class, LicenceClass::Permissive);
}

#[test]
fn train_and_score_reject_mismatched_web_and_tatoeba_languages() {
    let mut r = LANGDETECT_WEB;
    r.tatoeba.languages = &langdetect::LANGUAGES[..1];
    let mut plan = Plan {
        training: Vec::new(),
        held_out: Vec::new(),
        languages: Vec::new(),
        excluded: Vec::new(),
        records: 0,
        no_language: 0,
        other_languages: 0,
        cc0_records: 0,
        cc0_differing: 0,
        cc0_not_in_detailed: 0,
    };
    let web_plan = WebPlan { training: Vec::new(), held_out: Vec::new(), files: Vec::new() };
    assert!(matches!(r.train(&plan, &web_plan, &[]), Err(Error::Model(_))));
    assert!(matches!(r.score(&[], &[], &[], &mut plan, &web_plan, "0", "0"), Err(Error::Model(_))));
}

#[test]
fn web_language_card_artifact_validates_component_and_language() {
    let sha256 = "0".repeat(64);
    let sha512 = "0".repeat(128);
    let files = fineweb::parse_files(fineweb::FILES_NAME, fineweb::FILES).unwrap();
    let ranges = fineweb::parse_ranges(fineweb::RANGES_NAME, fineweb::RANGES).unwrap();
    let lists_sha256 = fineweb::lists_digest(&files, &ranges);
    let export_file = ExportFile { name: "a.tar.bz2".to_owned(), size: 1, sha256: sha256.clone() };
    let gate = GateCard {
        record: GATE_FILE.to_owned(),
        reference: langdetect::LANGDETECT_183.id.to_owned(),
        reference_sha256: langdetect::LANGDETECT_183.sha256.to_owned(),
        candidate_tag_format: None,
        reference_tag_format: None,
        candidate_score: 1.0,
        reference_score: 0.9,
        difference_low: 0.05,
        difference_high: 0.15,
        pass: true,
    };
    let comp = ComparisonCard {
        record: COMPARE_TATOEBA_FILE.to_owned(),
        units: 1,
        candidate_score: 1.0,
        compared_score: 1.0,
        difference_low: 0.0,
        difference_high: 0.0,
    };
    let card = WebLanguageCard {
        card_version: CARD_VERSION,
        id: MODEL_ID.to_owned(),
        component: turbo_zoo::Component::LanguageDetector,
        language: ZOO_LANGUAGE.to_owned(),
        languages: vec!["eng".to_owned()],
        file: ModelFileCard {
            name: format!("{MODEL_ID}.bin"),
            size: 10,
            sha256: sha256.clone(),
            sha512: sha512.clone(),
        },
        license: MODEL_LICENCE,
        license_terms: MODEL_LICENCE_TERMS.to_owned(),
        trained_on: training_data(&sha256),
        class: LicenceClass::Permissive,
        attribution: "attr".to_owned(),
        training_sentences: SentencesFileCard {
            name: SENTENCES_FILE.to_owned(),
            sentences: 1,
            size: 2,
            sha256: sha256.clone(),
            sha512: sha512.clone(),
        },
        training_documents: DocumentsFileCard {
            name: DOCUMENTS_FILE.to_owned(),
            documents: 1,
            size: 2,
            sha256: sha256.clone(),
            sha512,
        },
        export: ExportCard {
            detailed: export_file.clone(),
            cc0: export_file,
            digest: sha256,
            recorded: true,
            records: 1,
            no_language: 0,
            other_languages: 0,
            cc0_records: 0,
            cc0_differing: 0,
            cc0_not_in_detailed: 0,
            accepted_licences: ACCEPTED_LICENCES.to_vec(),
        },
        split: SPLIT,
        per_language: Vec::new(),
        excluded_languages: Vec::new(),
        web: WebDataCard {
            datasets: Vec::new(),
            lists_sha256,
            recorded: true,
            split: fineweb::WEB_SPLIT,
            files: Vec::new(),
            files_listed: Vec::new(),
        },
        web_per_language: Vec::new(),
        compared: COMPARED.id().to_owned(),
        compared_sha256: COMPARED.sha256().unwrap(),
        compared_per_language: Vec::new(),
        training: WebTrainingCard {
            parameters: LanguageTrainingCard {
                algorithm: "MAXENT".to_owned(),
                iterations: 100,
                cutoff: 5,
                threads: 1,
                data_indexer: "OnePass".to_owned(),
                timestamp_ms: 0,
                samples: 2,
                events: 2,
                event_hash: 0,
                inexact_events: 0,
            },
            sentences: 1,
            windows: 1,
        },
        held_out_sentences: 1,
        held_out_windows: 1,
        gate: gate.clone(),
        web_gate: GateCard { record: WEB_GATE_FILE.to_owned(), ..gate },
        compared_on_sentences: comp.clone(),
        compared_on_windows: ComparisonCard { record: COMPARE_WEB_FILE.to_owned(), ..comp },
    };
    let url_m = "https://example.org/model.bin";
    let url_s = "https://example.org/training-sentences.txt";
    let url_d = "https://example.org/training-documents.txt";
    assert!(card.artifact(url_m, url_s, url_d).is_ok());

    let mut bad_comp = card.clone();
    bad_comp.component = turbo_zoo::Component::Pos;
    assert!(matches!(bad_comp.artifact(url_m, url_s, url_d), Err(Error::InvalidCard { field: "component", .. })));

    let mut bad_lang = card;
    bad_lang.language = "en".to_owned();
    assert!(matches!(bad_lang.artifact(url_m, url_s, url_d), Err(Error::InvalidCard { field: "language", .. })));
}
