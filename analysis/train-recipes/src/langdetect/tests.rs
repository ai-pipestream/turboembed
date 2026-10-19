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
    assert!(turbo_zoo::Index::builtin().get(LANGDETECT_183.id).is_err(), "the zoo does not list the reference");
    let id: turbo_zoo::ArtifactId = MODEL_ID.parse().unwrap();
    assert_eq!(id.as_str(), MODEL_ID);
    assert!(ZOO_LANGUAGE.parse::<turbo_zoo::Language>().is_ok());
}

#[test]
fn language_card_artifact_validates_component_and_language() {
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
            reference: LANGDETECT_183.id.to_owned(),
            reference_sha256: LANGDETECT_183.sha256.to_owned(),
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
    assert!(card.artifact(url_m, url_s).is_ok());

    let mut bad_comp = card.clone();
    bad_comp.component = turbo_zoo::Component::Pos;
    assert!(matches!(bad_comp.artifact(url_m, url_s), Err(Error::InvalidCard { field: "component", .. })));

    let mut bad_lang = card;
    bad_lang.language = "en".to_owned();
    assert!(matches!(bad_lang.artifact(url_m, url_s), Err(Error::InvalidCard { field: "language", .. })));
}

#[test]
fn gate_resets_per_language_correct_counts_on_repeated_calls() {
    const PAIRS: [LanguagePair; 2] =
        [LanguagePair { tatoeba: "eng", reference: &["eng"] }, LanguagePair { tatoeba: "fra", reference: &["fra"] }];
    let recipe = LanguageRecipe {
        model_id: MODEL_ID,
        reference: Reference::Recorded(LANGDETECT_183),
        languages: &PAIRS,
        split: Split { seed: 1, held_out: 1, max_training: 2, min_sentences: 2 },
        iterations: 5,
        cutoff: 0,
    };
    let mut held_out = Vec::new();
    for i in 0..15u32 {
        held_out.push(Sample { language: 0, id: 10 + i, text: "Hello world in English.".into() });
        held_out.push(Sample { language: 1, id: 30 + i, text: "Bonjour le monde en francais.".into() });
    }
    let mut plan = Plan {
        training: vec![
            Sample { language: 0, id: 1, text: "Hello world in English.".into() },
            Sample { language: 1, id: 2, text: "Bonjour le monde en francais.".into() },
        ],
        held_out,
        languages: vec![
            LanguageCount {
                tatoeba: "eng".to_owned(),
                reference: vec!["eng".to_owned()],
                sentences: 16,
                licence_left_out: 0,
                duplicates: 0,
                training: 1,
                held_out: 15,
                candidate_correct: 99,
                reference_correct: 99,
            },
            LanguageCount {
                tatoeba: "fra".to_owned(),
                reference: vec!["fra".to_owned()],
                sentences: 16,
                licence_left_out: 0,
                duplicates: 0,
                training: 1,
                held_out: 15,
                candidate_correct: 99,
                reference_correct: 99,
            },
        ],
        excluded: Vec::new(),
        records: 32,
        no_language: 0,
        other_languages: 0,
        cc0_records: 0,
        cc0_differing: 0,
        cc0_not_in_detailed: 0,
    };
    let trained = recipe.train(&plan).unwrap();
    recipe.gate(&trained.bytes, &trained.bytes, &mut plan, "set", "0").unwrap();
    let first: Vec<(usize, usize)> =
        plan.languages.iter().map(|l| (l.candidate_correct, l.reference_correct)).collect();
    for l in &plan.languages {
        assert!(l.candidate_correct <= l.held_out);
        assert!(l.reference_correct <= l.held_out);
    }
    recipe.gate(&trained.bytes, &trained.bytes, &mut plan, "set", "0").unwrap();
    let second: Vec<(usize, usize)> =
        plan.languages.iter().map(|l| (l.candidate_correct, l.reference_correct)).collect();
    assert_eq!(first, second);
}
