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
//! order. The CC0 file is written as if after the detailed one: it holds
//! one of the two CC0 sentences with an edited text, and a third sentence
//! the detailed file does not hold.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_eval::model_gate::{Counts, Measure, ModelGateRecord, ModelId, PairedUnit, Scoring, Task};
use turbo_jvm::random::JavaRandom;
use turbo_langdetect::LanguageModel;
use turbo_maxent::ModelFile;
use turbo_train_recipes::langdetect::{
    CARD_FILE, EXPORT, ExcludedLanguage, ExportRecord, LANGDETECT, LANGDETECT_183, LANGUAGES, LanguageCard,
    LanguagePair, LanguageRecipe, MODEL_ID, SENTENCE_URL, SENTENCES_FILE, Split, check_export, run, sentences_file,
    write_language_dir,
};
use turbo_train_recipes::release::{self, TATOEBA_MODELS};
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
const EXPORT_DIGEST: &str = "cfa990b0e2685037fc0561248ec1a388957370e45e50196bd1335634d780c2d3";

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
    assert!(card.attribution.starts_with("Trained on 40 sentences from Tatoeba"), "{}", card.attribution);
    assert!(card.attribution.contains(&format!("{SENTENCES_FILE}, shipped with the model, lists their ids")));
    assert!(card.attribution.contains("is at https://tatoeba.org/sentences/show/N."), "{}", card.attribution);
    let t = &card.training;
    assert_eq!((t.algorithm.as_str(), t.iterations, t.cutoff, t.threads), ("MAXENT", 30, 1, 1));
    assert_eq!((t.data_indexer.as_str(), t.samples, t.events, t.inexact_events), ("OnePass", 40, 40, 0));
    assert_eq!(t.event_hash, outcome.trained.event_hash);
    let e = &card.export;
    assert_eq!((e.recorded, e.records, e.no_language, e.other_languages), (false, 90, 1, 2));
    assert_eq!((e.cc0_records, e.cc0_differing, e.cc0_not_in_detailed), (3, 1, 1));
    assert_eq!(e.accepted_licences, [SentenceLicence::CcBy20Fr]);
    assert_eq!(card.held_out.samples, 32);
    assert_eq!(card.gate.pass, outcome.gate.result.pass);
    assert!(
        card.summary().contains("8 languages, trained on 40 sentences (40 events), held out 32"),
        "{}",
        card.summary()
    );
    assert!(card.summary().contains("tur (3)"), "{}", card.summary());

    // The directory: the model, the training sentences' ids, the card and
    // the gate record, read back.
    let out = temp_dir("card");
    write_language_dir(&out, &outcome).unwrap();
    let dir = out.join(TEST.model_id);
    assert_eq!(std::fs::read(dir.join(&card.file.name)).unwrap(), outcome.trained.bytes);
    assert_eq!(&LanguageCard::read(&dir.join(CARD_FILE)).unwrap(), card);
    assert_eq!(ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap(), outcome.gate);
    assert!(write_language_dir(&out, &outcome).is_err(), "the directory exists");

    // The ids file lists exactly the sentences trained on, in ascending
    // order, as the card describes it; it holds digits and line feeds only,
    // so no username.
    let ids_file = std::fs::read(dir.join(SENTENCES_FILE)).unwrap();
    assert_eq!(card.training_sentences.name, SENTENCES_FILE);
    assert_eq!(ids_file, outcome.trained.sentences);
    assert!(ids_file.iter().all(|b| b.is_ascii_digit() || *b == b'\n') && ids_file.ends_with(b"\n"));
    let listed: Vec<u32> = std::str::from_utf8(&ids_file).unwrap().lines().map(|l| l.parse().unwrap()).collect();
    let plan = TEST.plan(TEST.sentences(&export).unwrap()).unwrap();
    let mut trained: Vec<u32> = plan.training.iter().map(|s| s.id).collect();
    trained.sort_unstable();
    assert_eq!(listed, trained);
    assert_eq!(sentences_file(&plan), ids_file);
    assert_eq!((card.training_sentences.sentences, listed.len()), (40, 40));
    assert_eq!(card.training_sentences.size, ids_file.len() as u64);
    assert_eq!(card.training_sentences.sha256, turbo_zoo::Sha256::of(&ids_file).to_string());
    assert_eq!(card.training_sentences.sha512, turbo_zoo::Sha512::of(&ids_file).to_string());
    let held_out: HashSet<u32> = plan.held_out.iter().map(|s| s.id).collect();
    assert!(listed.iter().all(|id| !held_out.contains(id)), "held-out sentences are not trained on");
    assert_eq!(SENTENCE_URL, "https://tatoeba.org/sentences/show/");
    let text = std::fs::read_to_string(dir.join(CARD_FILE)).unwrap();
    std::fs::write(dir.join(CARD_FILE), text.replacen("\"card_version\": 1", "\"card_version\": 2", 1)).unwrap();
    assert!(LanguageCard::read(&dir.join(CARD_FILE)).unwrap_err().to_string().contains("card version 2"));
    std::fs::write(dir.join(CARD_FILE), text.replacen("{", "{\n  \"extra\": 1,", 1)).unwrap();
    assert!(LanguageCard::read(&dir.join(CARD_FILE)).is_err(), "an unknown field");
    std::fs::remove_dir_all(&out).unwrap();

    // Each condition of a release, in the order checked.
    let url = "https://example.org/turbo-mul-tatoeba-langdetect-1.bin";
    let ids_url = "https://example.org/turbo-mul-tatoeba-langdetect-1/training-sentences.txt";
    let reason = |card: &LanguageCard| match card.artifact(url, ids_url) {
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
    let artifact = card.artifact(url, ids_url).unwrap();
    assert_eq!(artifact.id.as_str(), MODEL_ID);
    assert_eq!(artifact.class, LicenceClass::Permissive);
    assert_eq!(artifact.files.len(), 2, "the model and the ids of its sentences");
    assert_eq!(artifact.files[0].size, outcome.trained.bytes.len() as u64);
    assert_eq!(artifact.files[1].path.to_string(), SENTENCES_FILE);
    assert_eq!(artifact.files[1].url.to_string(), ids_url);
    assert_eq!(artifact.files[1].size, card.training_sentences.size);
    assert_eq!(artifact.files[1].sha256.to_string(), card.training_sentences.sha256);
    card.class = LicenceClass::ShareAlike;
    assert!(matches!(card.artifact(url, ids_url), Err(Error::InvalidCard { field: "class", .. })));
}

/// The release's entry is made from the model's directory with both its
/// files, the model and the ids of its training sentences, each checked
/// against the card. The test model stands in for the release's: its card
/// and gate record are rewritten to name the release's model and
/// reference, with a passing gate, and to come from the recorded export.
#[test]
fn the_release_entry_checks_the_model_and_its_training_sentences() {
    let mut outcome = run(&TEST, &export(), false, &fixture()).unwrap();
    let card = &mut outcome.card;
    card.id = MODEL_ID.to_owned();
    card.file.name = format!("{MODEL_ID}.bin");
    card.export.recorded = true;
    let candidate = ModelId { id: MODEL_ID.to_owned(), sha256: card.file.sha256.clone() };
    let reference = ModelId { id: LANGDETECT_183.id.to_owned(), sha256: LANGDETECT_183.sha256.to_owned() };
    let right = PairedUnit { candidate: Counts::accuracy(1, 1), reference: Counts::accuracy(1, 1) };
    let units = vec![right; outcome.gate.units.len()];
    let gate =
        ModelGateRecord::new(Task::LanguageDetector, outcome.gate.set.clone(), candidate, reference, units).unwrap();
    assert!(gate.result.pass);
    card.gate.reference = gate.reference.id.clone();
    card.gate.reference_sha256 = gate.reference.sha256.clone();
    card.gate.candidate_score = gate.result.candidate.score;
    card.gate.reference_score = gate.result.reference.score;
    card.gate.difference_low = gate.result.difference.low;
    card.gate.difference_high = gate.result.difference.high;
    card.gate.pass = gate.result.pass;
    outcome.gate = gate;

    let out = temp_dir("release");
    write_language_dir(&out, &outcome).unwrap();
    let card = &outcome.card;
    let entries = release::entries(&out, &TATOEBA_MODELS).unwrap();
    let model_url = TATOEBA_MODELS.url(&card.file.name);
    assert_eq!(entries, [card.artifact(&model_url, &TATOEBA_MODELS.url(SENTENCES_FILE)).unwrap()]);
    let files: Vec<(&str, &str)> = entries[0].files.iter().map(|f| (f.path.as_str(), f.url.as_str())).collect();
    assert_eq!(
        files,
        [
            (card.file.name.as_str(), model_url.as_str()),
            (
                SENTENCES_FILE,
                "https://github.com/ai-pipestream/turboembed/releases/download/models-tatoeba-2026-10-10-1/training-sentences.txt"
            ),
        ]
    );

    // A changed or missing sentences file makes no entry.
    let ids = out.join(MODEL_ID).join(SENTENCES_FILE);
    let mut changed = outcome.trained.sentences.clone();
    changed[0] = if changed[0] == b'1' { b'2' } else { b'1' };
    std::fs::write(&ids, &changed).unwrap();
    match release::entries(&out, &TATOEBA_MODELS) {
        Err(Error::Zoo(turbo_zoo::ZooError::Integrity { id, .. })) => assert_eq!(id.as_str(), MODEL_ID),
        other => panic!("{other:?}"),
    }
    std::fs::remove_file(&ids).unwrap();
    assert!(matches!(release::entries(&out, &TATOEBA_MODELS), Err(Error::Io { .. })));
    std::fs::remove_dir_all(&out).unwrap();
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
    let expected = format!("the export's digest is {EXPORT_DIGEST}, not the recorded {}", EXPORT.digest().unwrap());
    assert!(String::from_utf8_lossy(&refused.stderr).contains(&expected), "{refused:?}");
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
