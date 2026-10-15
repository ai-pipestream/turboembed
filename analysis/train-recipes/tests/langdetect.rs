//! The language detector recipe end to end on a Tatoeba export written
//! for the project (`tests/data/tatoeba`, 90 records in the layout of
//! `sentences_detailed.csv` and `sentences_CC0.csv`): the licence check,
//! the split, training, the gate against the Rust language fixture, the
//! card, the model's directory and the binary.
//!
//! The export holds ten CC BY 2.0 FR sentences in each of eight languages
//! (eleven in English, one with a backslash; one Italian text with a tab),
//! two English CC0 sentences, a second copy of a German text, three
//! Turkish sentences, two Cantonese sentences, which the test recipe does
//! not read, and one record with no language. Its records are not in id
//! order.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_eval::model_gate::{Measure, ModelGateRecord, Scoring, Task};
use turbo_jvm::random::JavaRandom;
use turbo_langdetect::LanguageModel;
use turbo_maxent::ModelFile;
use turbo_train_recipes::langdetect::{
    CARD_FILE, EXPORT, ExcludedLanguage, ExportRecord, LANGDETECT, LANGDETECT_183, LANGUAGES, LanguageCard,
    LanguagePair, LanguageRecipe, MODEL_ID, Split, check_export, java_shuffle, run, write_language_dir,
};
use turbo_train_recipes::tatoeba::{Export, SentenceLicence, TatoebaError};
use turbo_train_recipes::{Error, RecordedReference, Reference, card::GATE_FILE};
use turbo_zoo::{Licence, LicenceClass};

/// `analysis/goldens/data/fixtures/langdetect-fixture-rust.bin`, trained
/// by turbo-train-components on `inputs/langdetect/samples.txt`: the test
/// recipe's reference, for its nine languages.
const FIXTURE: RecordedReference = RecordedReference {
    id: "langdetect-fixture-rust",
    file: "langdetect-fixture-rust.bin",
    url: "analysis/goldens/data/fixtures/langdetect-fixture-rust.bin",
    size: 7503,
    sha256: "bfde716d402a696348045c2225428bbeb53f9e40f6807a5716396852a2a03ccb",
};

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
    reference: Reference::Recorded(FIXTURE),
    languages: &TEST_LANGUAGES,
    split: Split { seed: 429, held_out: 4, max_training: 5, min_sentences: 8 },
    iterations: 30,
    cutoff: 1,
};

/// The digest of the export in `tests/data/tatoeba`.
const EXPORT_DIGEST: &str = "4573e384134153d7b6979379227abc68e755011affe40d2407331b69501eefe3";

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tatoeba")
}

fn export() -> Export {
    Export::read(&data()).unwrap()
}

fn fixture() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures").join(FIXTURE.file);
    std::fs::read(path).unwrap()
}

/// A fresh directory under the system's temporary directory.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-langdetect-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `langdetect-183.bin` where `turbo-goldens-tool fetch-models` writes it
/// (`TURBO_GOLDENS_MODELS`), or `None` when it is not there and the test
/// may skip; with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, as CI sets it, a
/// missing model fails.
fn langdetect_183() -> Option<PathBuf> {
    let required = std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1");
    let Some(dir) = std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from) else {
        assert!(!required, "TURBO_GOLDENS_MODELS is not set and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1");
        eprintln!("TURBO_GOLDENS_MODELS is not set: skipping {}", LANGDETECT_183.file);
        return None;
    };
    let path = dir.join(LANGDETECT_183.file);
    if path.is_file() {
        return Some(path);
    }
    assert!(!required, "{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
    eprintln!("{} not found: skipping", path.display());
    None
}

fn languages(model: &[u8]) -> Vec<String> {
    let model = LanguageModel::open(ModelFile::open(model.to_vec()).unwrap()).unwrap();
    let mut codes: Vec<String> = model.supported_languages().map(str::to_owned).collect();
    codes.sort();
    codes
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
    assert_eq!((sentences.records, sentences.no_language, sentences.cc0_records), (90, 1, 2));
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
        java_shuffle(&mut ids, &mut JavaRandom::new(429));
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
fn the_recipe_trains_the_same_bytes_twice_and_the_model_opens() {
    let export = export();
    let plan = TEST.plan(TEST.sentences(&export).unwrap()).unwrap();
    let first = TEST.train(&plan).unwrap();
    let second = TEST.train(&plan).unwrap();
    assert_eq!(first.bytes, second.bytes);
    assert_eq!((first.samples, first.events, first.inexact_events), (40, 40, 0));
    assert_eq!(languages(&first.bytes), ["deu", "ell", "eng", "fra", "ita", "nld", "rus", "spa"]);
}

#[test]
fn the_gate_scores_one_unit_per_held_out_sentence() {
    let export = export();
    let reference = fixture();
    TEST.reference.check(&reference).unwrap();
    let mut plan = TEST.plan(TEST.sentences(&export).unwrap()).unwrap();
    let trained = TEST.train(&plan).unwrap();
    let digest = export.digest();
    let record = TEST.gate(&trained.bytes, &reference, &mut plan, &TEST.held_out_name(&digest), &digest).unwrap();
    record.check().unwrap();
    assert_eq!((record.task, record.measure), (Task::LanguageDetector, Measure::Accuracy));
    assert_eq!(record.units.len(), 32);
    assert!(matches!(&record.set.scoring, Scoring::Labels(s) if s.samples == 32 && s.reference_mappings.is_empty()));
    assert_eq!(record.set.samples_sha256, TEST.samples_digest(&plan));
    assert_eq!(record.reference.sha256, FIXTURE.sha256);
    let correct = |f: fn(&turbo_train_recipes::langdetect::LanguageCount) -> usize| -> usize {
        plan.languages.iter().map(f).sum()
    };
    let (c, r) = (correct(|l| l.candidate_correct), correct(|l| l.reference_correct));
    assert_eq!(record.result.candidate.score, 100.0 * c as f64 / 32.0);
    assert_eq!(record.result.reference.score, 100.0 * r as f64 / 32.0);
    assert_eq!((c, r), (28, 26));
    let per_language: Vec<(usize, usize)> =
        plan.languages.iter().map(|l| (l.candidate_correct, l.reference_correct)).collect();
    assert_eq!(per_language, [(3, 4), (4, 4), (4, 4), (3, 2), (2, 4), (4, 3), (4, 4), (4, 1)]);
    // 32 units leave the interval too wide for the gate.
    assert_eq!((record.result.difference.low, record.result.difference.high), (-9.375, 25.0));
    assert!(!record.result.pass);

    // A code the reference does not have is refused before scoring.
    const GER: [LanguagePair; 2] =
        [LanguagePair { tatoeba: "deu", reference: &["ger"] }, LanguagePair { tatoeba: "eng", reference: &["eng"] }];
    let ger = LanguageRecipe { languages: &GER, ..TEST };
    let mut plan = ger.plan(ger.sentences(&export).unwrap()).unwrap();
    let err = ger.gate(&trained.bytes, &reference, &mut plan, "set", &digest).unwrap_err();
    assert!(err.to_string().contains("no outcome ger, which the recipe counts right for deu"), "{err}");

    // A Tatoeba language with two reference codes counts either right,
    // and the record lists the mapping.
    const WIDE: [LanguagePair; 8] = [
        LanguagePair { tatoeba: "deu", reference: &["deu", "nld"] },
        same(&["ell"]),
        same(&["eng"]),
        same(&["fra"]),
        same(&["ita"]),
        same(&["nld"]),
        same(&["rus"]),
        same(&["spa"]),
    ];
    let wide = LanguageRecipe { languages: &WIDE, ..TEST };
    let mut plan = wide.plan(wide.sentences(&export).unwrap()).unwrap();
    let narrow = {
        let mut plan = TEST.plan(TEST.sentences(&export).unwrap()).unwrap();
        TEST.gate(&trained.bytes, &reference, &mut plan, "set", &digest).unwrap();
        plan.languages[0].reference_correct
    };
    let record = wide.gate(&trained.bytes, &reference, &mut plan, "set", &digest).unwrap();
    assert!(plan.languages[0].reference_correct >= narrow);
    let Scoring::Labels(scoring) = &record.set.scoring else { panic!("{:?}", record.set.scoring) };
    assert_eq!(scoring.reference_mappings.len(), 1);
    assert_eq!(
        (scoring.reference_mappings[0].held_out.as_str(), scoring.reference_mappings[0].reference.as_str()),
        ("deu", "nld")
    );
    assert_ne!(record.set.samples_sha256, TEST.samples_digest(&TEST.plan(TEST.sentences(&export).unwrap()).unwrap()));
}

#[test]
fn the_card_records_the_run_and_is_shippable_only_from_the_recorded_export() {
    let export = export();
    let reference = fixture();
    let outcome = run(&TEST, &export, false, &reference).unwrap();
    let card = &outcome.card;
    assert_eq!(card.languages, ["deu", "ell", "eng", "fra", "ita", "nld", "rus", "spa"]);
    assert_eq!(card.excluded_languages.len(), 1);
    assert_eq!(card.split, TEST.split);
    assert_eq!(card.file.name, "turbo-mul-test-langdetect.bin");
    assert_eq!(card.file.size, outcome.trained.bytes.len() as u64);
    assert_eq!(card.trained_on.len(), 1);
    assert_eq!(card.trained_on[0].license, Licence::CcBy20Fr);
    assert_eq!(card.class, LicenceClass::Permissive);
    assert!(card.attribution.contains(&export.digest()));
    let t = &card.training;
    assert_eq!((t.algorithm.as_str(), t.iterations, t.cutoff, t.threads), ("MAXENT", 30, 1, 1));
    assert_eq!((t.data_indexer.as_str(), t.samples, t.events, t.inexact_events), ("OnePass", 40, 40, 0));
    assert_eq!(t.event_hash, outcome.trained.event_hash);
    let e = &card.export;
    assert_eq!((e.recorded, e.records, e.no_language, e.other_languages), (false, 90, 1, 2));
    assert_eq!(e.accepted_licences, [SentenceLicence::CcBy20Fr]);
    assert_eq!(card.held_out.samples, 32);
    assert_eq!(card.gate.pass, outcome.gate.result.pass);
    assert!(
        card.summary().contains("8 languages, trained on 40 sentences (40 events), held out 32"),
        "{}",
        card.summary()
    );
    assert!(card.summary().contains("tur (3)"), "{}", card.summary());

    // The directory: the model, the card and the gate record, read back.
    let out = temp_dir("card");
    write_language_dir(&out, card, &outcome.trained.bytes, &outcome.gate).unwrap();
    let dir = out.join(TEST.model_id);
    assert_eq!(std::fs::read(dir.join(&card.file.name)).unwrap(), outcome.trained.bytes);
    assert_eq!(&LanguageCard::read(&dir.join(CARD_FILE)).unwrap(), card);
    assert_eq!(ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap(), outcome.gate);
    assert!(write_language_dir(&out, card, &outcome.trained.bytes, &outcome.gate).is_err(), "the directory exists");
    let text = std::fs::read_to_string(dir.join(CARD_FILE)).unwrap();
    std::fs::write(dir.join(CARD_FILE), text.replacen("\"card_version\": 1", "\"card_version\": 2", 1)).unwrap();
    assert!(LanguageCard::read(&dir.join(CARD_FILE)).unwrap_err().to_string().contains("card version 2"));
    std::fs::write(dir.join(CARD_FILE), text.replacen("{", "{\n  \"extra\": 1,", 1)).unwrap();
    assert!(LanguageCard::read(&dir.join(CARD_FILE)).is_err(), "an unknown field");
    std::fs::remove_dir_all(&out).unwrap();

    // Each condition of a release, in the order checked.
    let url = "https://example.org/turbo-mul-tatoeba-langdetect-1.bin";
    let reason = |card: &LanguageCard| match card.artifact(url) {
        Err(Error::NotShippable { reason, .. }) => reason,
        other => panic!("{other:?}"),
    };
    let mut card = card.clone();
    assert_eq!(reason(&card), "no recipe trains it");
    card.id = MODEL_ID.to_owned();
    card.file.name = format!("{MODEL_ID}.bin");
    assert_eq!(reason(&card), "it was not gated against its recipe's reference");
    card.gate.reference = LANGDETECT_183.id.to_owned();
    card.gate.reference_sha256 = LANGDETECT_183.sha256.to_owned();
    card.gate.pass = false;
    assert_eq!(reason(&card), "it failed its gate");
    card.gate.pass = true;
    assert_eq!(reason(&card), "it was trained on a Tatoeba export that is not the recorded one");
    card.export.recorded = true;
    let artifact = card.artifact(url).unwrap();
    assert_eq!(artifact.id.as_str(), MODEL_ID);
    assert_eq!(artifact.class, LicenceClass::Permissive);
    assert_eq!(artifact.files[0].size, outcome.trained.bytes.len() as u64);
    card.class = LicenceClass::ShareAlike;
    assert!(matches!(card.artifact(url), Err(Error::InvalidCard { field: "class", .. })));
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
    // The recipe records none yet.
    assert_eq!(EXPORT, ExportRecord::NotRecorded);
}

/// The recipe's reference has exactly the codes its table maps to, and
/// is the recorded file.
#[test]
fn every_code_of_langdetect_183_is_in_the_recipe_table() {
    let Some(path) = langdetect_183() else { return };
    let bytes = std::fs::read(path).unwrap();
    LANGDETECT.reference.check(&bytes).unwrap();
    let mut table: Vec<String> = LANGUAGES.iter().flat_map(|p| p.reference.iter().map(|c| (*c).to_owned())).collect();
    table.sort();
    assert_eq!(languages(&bytes), table);
}

fn binary(export: &Path, references: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_turbo-train-recipes"));
    c.arg("tatoeba").arg("--export").arg(export).arg("--references").arg(references).arg("--out").arg(out);
    c.args(extra).output().unwrap()
}

#[test]
fn the_binary_needs_the_reference_and_the_recorded_export() {
    let root = temp_dir("binary");
    let empty = root.join("references");
    std::fs::create_dir(&empty).unwrap();
    let out = root.join("models");
    let missing = binary(&data(), &empty, &out, &[]);
    assert_eq!(missing.status.code(), Some(2), "{missing:?}");
    assert!(String::from_utf8_lossy(&missing.stderr).contains(LANGDETECT_183.file), "{missing:?}");

    let Some(path) = langdetect_183() else { return };
    let references = path.parent().unwrap();
    let refused = binary(&data(), references, &out, &[]);
    assert_eq!(refused.status.code(), Some(2), "{refused:?}");
    assert!(String::from_utf8_lossy(&refused.stderr).contains("no Tatoeba export is recorded"), "{refused:?}");
    assert!(String::from_utf8_lossy(&refused.stdout).contains(EXPORT_DIGEST), "{refused:?}");
    // With --unrecorded-export the recipe runs, and the export's few
    // sentences per language leave none of its languages.
    let small = binary(&data(), references, &out, &["--unrecorded-export"]);
    assert_eq!(small.status.code(), Some(2), "{small:?}");
    assert!(
        String::from_utf8_lossy(&small.stderr).contains("the split leaves no language with enough sentences"),
        "{small:?}"
    );
    assert!(!out.join(MODEL_ID).exists());
    std::fs::remove_dir_all(&root).unwrap();
}
