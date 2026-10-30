//! Dataset validation on a Tatoeba export written
//! for the project (`tests/data/tatoeba`, 90 records in the layout of
//! `sentences_detailed.csv` and `sentences_CC0.csv`): the licence check,
//! and deterministic language splits.
//!
//! The export holds ten CC BY 2.0 FR sentences in each of eight languages
//! (eleven in English, one with a backslash; one Italian text with a tab),
//! two English CC0 sentences, a second copy of a German text, three
//! Turkish sentences, two Cantonese sentences, which the test recipe does
//! not read, and one record with no language. Its records are not in id
//! order. The CC0 file is written as if after the detailed one: it holds
//! one of the two CC0 sentences with an edited text, and a third sentence
//! the detailed file does not hold.

use std::collections::HashSet;
use std::path::PathBuf;

use turbo_jvm::random::JavaRandom;
use turbo_train_recipes::Error;
use turbo_train_recipes::langdetect::{
    EXPORT, ExcludedLanguage, ExportRecord, LanguagePair, LanguageRecipe, Split, check_export, run,
};
use turbo_train_recipes::tatoeba::{Export, SentenceLicence, TatoebaError};
use turbo_zoo::{Licence, LicenceClass};

const fn same(code: &'static [&'static str; 1]) -> LanguagePair {
    LanguagePair { tatoeba: code[0], reference: code }
}

/// The languages of the fixture, each its own code on both sides.
const TEST_LANGUAGES: [LanguagePair; 9] = [
    same(&["deu"]),
    same(&["ell"]),
    same(&["eng"]),
    same(&["fra"]),
    same(&["ita"]),
    same(&["nld"]),
    same(&["rus"]),
    same(&["spa"]),
    same(&["tur"]),
];

/// The recipe at the export's scale: four sentences held out and five
/// trained on per language, eight needed; 30 iterations with cutoff 1, as
/// 40 short training sentences have few n-grams seen five times.
const TEST: LanguageRecipe = LanguageRecipe {
    model_id: "turbo-mul-test-langdetect",
    reference: None,
    languages: &TEST_LANGUAGES,
    split: Split { seed: 429, held_out: 4, max_training: 5, min_sentences: 8 },
    iterations: 30,
    cutoff: 1,
};

/// The digest of the export in `tests/data/tatoeba`.
const EXPORT_DIGEST: &str = "cfa990b0e2685037fc0561248ec1a388957370e45e50196bd1335634d780c2d3";

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tatoeba")
}

fn export() -> Export {
    Export::read(&data()).unwrap()
}

/// A fresh directory under the system's temporary directory.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-langdetect-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_export_is_read_with_the_licence_of_each_sentence() {
    let export = export();
    assert_eq!(export.digest(), EXPORT_DIGEST);
    assert_eq!(
        (export.detailed.name.as_str(), export.cc0.name.as_str()),
        ("sentences_detailed.csv", "sentences_CC0.csv")
    );
    assert_eq!(export.detailed.sha256.to_string(), "566a3dd8cd4404a833392b9bc3a4e88feb3a6f6e22c15187bd72c05dd11ce22d");
    let sentences = export.sentences(|_| true).unwrap();
    assert_eq!((sentences.records, sentences.no_language, sentences.cc0_records), (90, 1, 3));
    // 1616 has another text in the CC0 file and is still CC0; 1651 is not
    // in the detailed file.
    assert_eq!((sentences.cc0_differing, sentences.cc0_not_in_detailed), (1, 1));
    assert_eq!(sentences.kept.len(), 89);
    // In the file's order.
    let cc0: Vec<u32> = sentences.kept.iter().filter(|s| s.licence == SentenceLicence::Cc0).map(|s| s.id).collect();
    assert_eq!(cc0, [1616, 1609]);
    let text = |id: u32| sentences.kept.iter().find(|s| s.id == id).unwrap().text.to_string();
    assert_eq!(text(1077), "The path is C:\\temp\\notes on that machine.");
    assert_eq!(text(1357), "Ho dimenticato di nuovo\tl'ombrello in ufficio.");
    assert_eq!(SentenceLicence::CcBy20Fr.licence(), Licence::CcBy20Fr);
    assert_eq!(Licence::CcBy20Fr.class(), LicenceClass::Permissive);
}

#[test]
fn the_plan_keeps_cc_by_sentences_and_splits_each_language_in_id_order() {
    let export = export();
    let plan = TEST.plan(TEST.sentences(&export).unwrap()).unwrap();
    assert_eq!((plan.records, plan.no_language, plan.other_languages), (90, 1, 2));
    assert_eq!((plan.cc0_records, plan.cc0_differing, plan.cc0_not_in_detailed), (3, 1, 1));
    let counts: Vec<(&str, usize, usize, usize, usize, usize)> = plan
        .languages
        .iter()
        .map(|l| (l.tatoeba.as_str(), l.sentences, l.licence_left_out, l.duplicates, l.training, l.held_out))
        .collect();
    assert_eq!(
        counts,
        [
            ("deu", 11, 0, 1, 5, 4),
            ("ell", 10, 0, 0, 5, 4),
            ("eng", 13, 2, 0, 5, 4),
            ("fra", 10, 0, 0, 5, 4),
            ("ita", 10, 0, 0, 5, 4),
            ("nld", 10, 0, 0, 5, 4),
            ("rus", 10, 0, 0, 5, 4),
            ("spa", 10, 0, 0, 5, 4),
        ]
    );
    assert_eq!(plan.excluded, [ExcludedLanguage { tatoeba: "tur".into(), sentences: 3, usable: 3 }]);
    assert_eq!((plan.training.len(), plan.held_out.len()), (40, 32));

    // No CC0 sentence, no second copy of a text and no sentence on both
    // sides.
    let training: HashSet<u32> = plan.training.iter().map(|s| s.id).collect();
    let held_out: HashSet<u32> = plan.held_out.iter().map(|s| s.id).collect();
    assert!(training.is_disjoint(&held_out));
    for id in [1609, 1616, 1623] {
        assert!(!training.contains(&id) && !held_out.contains(&id), "{id}");
    }

    // Each language's sentences in id order, shuffled by its own
    // java.util.Random: the first four held out, the next five trained on.
    let sentences = TEST.sentences(&export).unwrap();
    for (k, l) in plan.languages.iter().enumerate() {
        let mut ids: Vec<u32> = sentences
            .kept
            .iter()
            .filter(|s| *s.language == *l.tatoeba && s.licence == SentenceLicence::CcBy20Fr && s.id != 1623)
            .map(|s| s.id)
            .collect();
        ids.sort_unstable();
        JavaRandom::new(429).shuffle(&mut ids);
        let language = TEST_LANGUAGES.iter().position(|p| p.tatoeba == l.tatoeba).unwrap();
        let held: Vec<u32> = plan.held_out.iter().filter(|s| s.language == language).map(|s| s.id).collect();
        let trained: Vec<u32> = plan.training.iter().filter(|s| s.language == language).map(|s| s.id).collect();
        assert_eq!(held, ids[..4], "{}", l.tatoeba);
        assert_eq!(trained, ids[4..9], "{}", l.tatoeba);
        assert_eq!(&plan.held_out[4 * k].language, &language, "language by language");
    }

    // The records' order in the file does not change the plan.
    let reversed = temp_dir("reversed");
    let detailed = std::fs::read_to_string(data().join("sentences_detailed.csv")).unwrap();
    let mut lines: Vec<&str> = detailed.lines().collect();
    lines.reverse();
    std::fs::write(reversed.join("sentences_detailed.csv"), lines.join("\n") + "\n").unwrap();
    std::fs::copy(data().join("sentences_CC0.csv"), reversed.join("sentences_CC0.csv")).unwrap();
    let other = Export::read(&reversed).unwrap();
    assert_ne!(other.digest(), export.digest());
    let again = TEST.plan(TEST.sentences(&other).unwrap()).unwrap();
    assert_eq!((again.training, again.held_out), (plan.training.clone(), plan.held_out.clone()));
    std::fs::remove_dir_all(&reversed).unwrap();

    // Another seed, another split.
    let seeded = LanguageRecipe { split: Split { seed: 430, ..TEST.split }, ..TEST };
    let other = seeded.plan(seeded.sentences(&export).unwrap()).unwrap();
    assert_ne!(other.held_out, plan.held_out);

    // A threshold no language meets leaves nothing to train on.
    let strict = LanguageRecipe { split: Split { min_sentences: 12, ..TEST.split }, ..TEST };
    let err = strict.plan(strict.sentences(&export).unwrap()).unwrap_err();
    assert!(matches!(err, Error::EmptySplit { .. }), "{err:?}");
    // And one English alone meets with its CC0 sentences left out.
    let eleven = LanguageRecipe { split: Split { min_sentences: 11, ..TEST.split }, ..TEST };
    let plan = eleven.plan(eleven.sentences(&export).unwrap()).unwrap();
    assert_eq!(plan.languages.iter().map(|l| l.tatoeba.as_str()).collect::<Vec<_>>(), ["eng"]);
    assert_eq!(plan.excluded.len(), 8);
}

#[test]
fn an_export_is_a_release_only_when_it_is_the_recorded_one() {
    let export = export();
    let not_recorded = check_export(&export, ExportRecord::NotRecorded, false).unwrap_err();
    assert!(matches!(not_recorded, Error::Tatoeba(TatoebaError::NotRecorded)), "{not_recorded:?}");
    assert!(!check_export(&export, ExportRecord::NotRecorded, true).unwrap());
    assert!(check_export(&export, ExportRecord::Recorded { digest: EXPORT_DIGEST }, false).unwrap());
    let other = ExportRecord::Recorded { digest: "0000000000000000000000000000000000000000000000000000000000000000" };
    let err = check_export(&export, other, false).unwrap_err();
    assert!(
        matches!(&err, Error::Tatoeba(TatoebaError::ExportDigest { found, .. }) if found == EXPORT_DIGEST),
        "{err:?}"
    );
    assert!(!check_export(&export, other, true).unwrap());
    assert_eq!(ExportRecord::NotRecorded.digest(), None);
    // The recipe records Tatoeba's export of 2026-10-10, not this one.
    assert_eq!(EXPORT.digest(), Some("4421126b36dc0ce206b222296d696aff04a399620f1ca5dbefbfa50a1890aa92"));
    assert!(!EXPORT.is(EXPORT_DIGEST));
}

#[test]
fn classical_run_and_gate_are_retired() {
    let export = export();
    let error = run(&TEST, &export, false, &[]).unwrap_err();
    assert!(matches!(error, Error::Model(_)));

    let mut plan = TEST.plan(TEST.sentences(&export).unwrap()).unwrap();
    let error = TEST.gate(&[], &[], &mut plan, "test", "0").unwrap_err();
    assert!(matches!(error, Error::Model(_)));
}
