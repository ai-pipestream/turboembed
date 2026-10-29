use super::*;
use crate::card::{GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, ModelFileCard};
use crate::error::Error;
use crate::tatoeba::ExportFile;
use turbo_zoo::LicenceClass;

#[test]
fn every_reference_code_is_in_one_pair_and_the_tatoeba_codes_are_unique() {
    let mut tatoeba: Vec<&str> = LANGUAGES.iter().map(|p| p.tatoeba).collect();
    let sorted = {
        let mut s = tatoeba.clone();
        s.sort_unstable();
        s
    };
    assert_eq!(tatoeba, sorted, "in Tatoeba code order");
    tatoeba.dedup();
    assert_eq!(tatoeba.len(), 100);
    let mut reference: Vec<&str> = LANGUAGES.iter().flat_map(|p| p.reference.iter().copied()).collect();
    reference.sort_unstable();
    let all = reference.len();
    reference.dedup();
    assert_eq!((all, reference.len()), (103, 103), "each of the reference's 103 codes once");
    for p in LANGUAGES {
        for code in std::iter::once(p.tatoeba).chain(p.reference.iter().copied()) {
            assert!(code.len() == 3 && code.bytes().all(|b| b.is_ascii_lowercase()), "{code}");
        }
    }
    // The pairs whose codes differ: a macrolanguage and one of its
    // individual languages.
    let differing: Vec<(&str, &[&str])> =
        LANGUAGES.iter().filter(|p| p.reference != [p.tatoeba]).map(|p| (p.tatoeba, p.reference)).collect();
    assert_eq!(
        differing,
        [
            ("est", &["est", "ekk"][..]),
            ("lvs", &["lvs", "lav"][..]),
            ("mlg", &["plt"][..]),
            ("npi", &["nep"][..]),
            ("pes", &["pes", "fas"][..]),
            ("swh", &["swa"][..]),
            ("zsm", &["msa"][..]),
        ]
    );
}

#[test]
fn the_recipe_trains_with_the_parameters_langdetect_183_was_most_likely_trained_with() {
    let ours = LANGDETECT.parameters();
    let theirs = turbo_train_components::langdetect::default_training_parameters();
    assert_eq!(ours, theirs);
    assert_eq!(LANGDETECT.reference.id(), "opennlp-langdetect-183");
    assert!(turbo_zoo::Index::builtin().get(LANGDETECT_183.id()).is_err(), "the zoo does not list the reference");
    assert!(matches!(LANGDETECT_183.sha256(), Err(Error::RetiredReference { .. })), "langdetect-183 is not pinned");
    let id: turbo_zoo::ArtifactId = MODEL_ID.parse().unwrap();
    assert_eq!(id.as_str(), MODEL_ID);
    assert!(ZOO_LANGUAGE.parse::<turbo_zoo::Language>().is_ok());
}

#[test]
fn no_language_card_is_publishable_while_its_reference_is_retired() {
    let sha256 = "0".repeat(64);
    let sha512 = "0".repeat(128);
    let export_file = ExportFile { name: "a.tar.bz2".to_owned(), size: 1, sha256: sha256.clone() };
    let card = LanguageCard {
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
        trained_on: vec![training_data(&sha256)],
        class: LicenceClass::Permissive,
        attribution: "attr".to_owned(),
        training_sentences: SentencesFileCard {
            name: SENTENCES_FILE.to_owned(),
            sentences: 1,
            size: 2,
            sha256: sha256.clone(),
            sha512: sha512.clone(),
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
        training: LanguageTrainingCard {
            algorithm: "MAXENT".to_owned(),
            iterations: 100,
            cutoff: 5,
            threads: 1,
            data_indexer: "OnePass".to_owned(),
            timestamp_ms: 0,
            samples: 1,
            events: 1,
            event_hash: 0,
            inexact_events: 0,
        },
        held_out: LanguageHeldOutCard { samples: 1, reference_mappings: Vec::new() },
        gate: GateCard {
            record: GATE_FILE.to_owned(),
            reference: LANGDETECT_183.id().to_owned(),
            reference_sha256: "0".repeat(64),
            candidate_tag_format: None,
            reference_tag_format: None,
            candidate_score: 1.0,
            reference_score: 0.9,
            difference_low: 0.05,
            difference_high: 0.15,
            pass: true,
        },
    };
    let url_m = "https://example.org/model.bin";
    let url_s = "https://example.org/training-sentences.txt";
    // The gate read langdetect-183, an OpenNLP 2.5 artifact that is not
    // pinned (#1509), so the card cannot show it was gated against it;
    // #1532 removes the gate.
    assert!(matches!(card.artifact(url_m, url_s), Err(Error::RetiredReference { id: "opennlp-langdetect-183" })));
}
