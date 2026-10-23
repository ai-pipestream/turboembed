//! `TokenNameFinderCrossValidator` and `LanguageDetectorCrossValidator`
//! against the JVM's goldens under
//! `analysis/goldens/data/cross-validate/<run>/`: per fold the documents
//! and samples on each side, the training events and their checksum, the
//! counts and scores, and every prediction; then the merged precision,
//! recall and F-measure (name finder) and accuracy (language detector),
//! bit for bit. The runs are `namefind-bio` and `namefind-bilou` over
//! `inputs/tagged/names.txt`, and `langdetect-default` (cutoff 5) and
//! `langdetect-cutoff1` over `inputs/langdetect/samples.txt`, 3 folds each.

mod common;

use std::path::Path;

use common::{golden_files, layer};
use turbo_formats::native::{LanguageReader, NameReader};
use turbo_formats::{LanguageSample, NameSample};
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape, data_dir};
use turbo_sequence::namefind::NameCodec;
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    LanguageCrossValidation, LanguageDetectorCrossValidator, LanguageDetectorFactory, default_training_parameters,
};
use turbo_train_components::namefind::{NameCrossValidation, TokenNameFinderCrossValidator, TokenNameFinderFactory};
use turbo_unicode::WhitespaceMode;

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;
const STR: ColumnType = ColumnType::Str;

const LAYERS: &[(&str, &[ColumnSpec])] = &[
    (
        "turbo:name-cv-folds",
        &[
            col("fold", U32),
            col("training-documents", U32),
            col("training-samples", U32),
            col("test-documents", U32),
            col("test-samples", U32),
            col("events", U32),
            col("crc32c", U32),
            col("selected", U32),
            col("target", U32),
            col("true-positives", U32),
            col("correct", U32),
            col("precision", F64),
            col("recall", F64),
            col("f-measure", F64),
        ],
    ),
    (
        "turbo:name-cv-names",
        &[col("fold", U32), col("sample", U32), col("start", U32), col("end", U32), col("type", STR)],
    ),
    (
        "turbo:name-cv-merged",
        &[
            col("selected", U32),
            col("target", U32),
            col("true-positives", U32),
            col("precision", F64),
            col("recall", F64),
            col("f-measure", F64),
        ],
    ),
    (
        "turbo:language-cv-folds",
        &[
            col("fold", U32),
            col("training-samples", U32),
            col("test-samples", U32),
            col("events", U32),
            col("crc32c", U32),
            col("correct", U32),
            col("documents", U32),
            col("accuracy", F64),
        ],
    ),
    ("turbo:language-cv-predictions", &[col("fold", U32), col("sample", U32), col("language", STR)]),
    ("turbo:language-cv-merged", &[col("documents", U32), col("accuracy", F64)]),
];

fn read(path: &Path) -> Golden {
    let shapes: Vec<Shape> = LAYERS
        .iter()
        .map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns })
        .collect();
    Golden::read_with(path, &shapes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn config<'g>(at: &str, g: &'g Golden, key: &str) -> &'g str {
    common::config(g, key).unwrap_or_else(|| panic!("{at}: no config {key}"))
}

fn input(name: &str) -> String {
    let path = data_dir().join("inputs").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn name_samples(text: &str) -> Vec<NameSample<'_>> {
    NameReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect()
}

fn language_samples(text: &str) -> Vec<LanguageSample<'_>> {
    LanguageReader::new(text).unwrap().map(|s| s.unwrap()).collect()
}

fn runs(prefix: &str) -> Vec<(String, Golden)> {
    let mut files = golden_files(&data_dir().join("cross-validate"));
    files.sort();
    files
        .into_iter()
        .filter(|p| {
            p.parent()
                .and_then(Path::parent)
                .and_then(Path::file_name)
                .is_some_and(|n| n.to_str().is_some_and(|n| n.starts_with(prefix)))
        })
        .map(|p| (p.display().to_string(), read(&p)))
        .collect()
}

fn name_cross_validation(at: &str, g: &Golden, samples: &[NameSample<'_>]) -> NameCrossValidation {
    let codec = match config(at, g, "codec") {
        "bio" => NameCodec::Bio,
        "bilou" => NameCodec::Bilou,
        other => panic!("{at}: codec {other}"),
    };
    assert_eq!(config(at, g, "language"), "en", "{at}");
    let folds: usize = config(at, g, "folds").parse().unwrap();
    let factory = TokenNameFinderFactory::new("en", None, codec).unwrap();
    TokenNameFinderCrossValidator::new(&factory, None, TrainingParameters::new())
        .evaluate(samples, folds, &mut TrainingConfiguration::default(), Stamp::at(0))
        .unwrap_or_else(|e| panic!("{at}: {e}"))
}

fn language_cross_validation(at: &str, g: &Golden, samples: &[LanguageSample<'_>]) -> LanguageCrossValidation {
    let mut params = default_training_parameters();
    assert_eq!(config(at, g, "Algorithm"), "MAXENT", "{at}");
    assert_eq!(config(at, g, "Iterations"), "100", "{at}");
    params.put(Key::CUTOFF, config(at, g, "Cutoff").parse::<i32>().unwrap());
    let folds: usize = config(at, g, "folds").parse().unwrap();
    LanguageDetectorCrossValidator::new(&LanguageDetectorFactory::new(), params)
        .evaluate(samples, folds, &mut TrainingConfiguration::default(), Stamp::at(0))
        .unwrap_or_else(|e| panic!("{at}: {e}"))
}

fn bits(at: &str, what: &str, ours: f64, theirs: f64) {
    assert_eq!(ours.to_bits(), theirs.to_bits(), "{at}: {what}: {ours} against the JVM's {theirs}");
}

#[test]
fn name_finder_folds_and_merged_scores_match_the_jvm() {
    let text = input("tagged/names.txt");
    let samples = name_samples(&text);
    let runs = runs("namefind-");
    assert_eq!(runs.len(), 2, "the namefind-bio and namefind-bilou goldens");
    for (at, g) in &runs {
        let cv = name_cross_validation(at, g, &samples);
        let l = layer(g, "turbo:name-cv-folds");
        assert_eq!(cv.folds.len(), l.len(), "{at}: folds");
        let u = |c: &str, i: usize| u64::from(l.u32s(c).unwrap()[i]);
        for (i, f) in cv.folds.iter().enumerate() {
            let at = format!("{at} fold {i}");
            assert_eq!(f.index as u64, u("fold", i), "{at}");
            assert_eq!(f.training_documents as u64, u("training-documents", i), "{at}: training documents");
            assert_eq!(f.training_samples as u64, u("training-samples", i), "{at}: training samples");
            assert_eq!(f.test_documents as u64, u("test-documents", i), "{at}: test documents");
            assert_eq!(f.test_samples as u64, u("test-samples", i), "{at}: test samples");
            assert_eq!(f.events as u64, u("events", i), "{at}: events");
            assert_eq!(u64::from(f.event_checksum), u("crc32c", i), "{at}: event checksum");
            assert_eq!(f.fmeasure.selected(), u("selected", i), "{at}: selected");
            assert_eq!(f.fmeasure.target(), u("target", i), "{at}: target");
            assert_eq!(f.fmeasure.true_positives(), u("true-positives", i), "{at}: true positives");
            assert_eq!(f.correct_samples as u64, u("correct", i), "{at}: correct samples");
            bits(&at, "precision", f.fmeasure.precision(), l.f64s("precision").unwrap()[i]);
            bits(&at, "recall", f.fmeasure.recall(), l.f64s("recall").unwrap()[i]);
            bits(&at, "f-measure", f.fmeasure.f_measure(), l.f64s("f-measure").unwrap()[i]);
        }
        let names = layer(g, "turbo:name-cv-names");
        let ours: Vec<(u32, u32, u32, u32, String)> = cv
            .folds
            .iter()
            .flat_map(|f| {
                f.predictions.iter().enumerate().flat_map(move |(s, found)| {
                    found.iter().map(move |n| {
                        (f.index as u32, s as u32, n.start as u32, n.end as u32, n.kind.clone().unwrap_or_default())
                    })
                })
            })
            .collect();
        let theirs: Vec<(u32, u32, u32, u32, String)> = (0..names.len())
            .map(|r| {
                let c = |n: &str| names.u32s(n).unwrap()[r];
                (c("fold"), c("sample"), c("start"), c("end"), names.strs("type").unwrap().get(r).to_owned())
            })
            .collect();
        assert_eq!(ours, theirs, "{at}: predicted names");
        let m = layer(g, "turbo:name-cv-merged");
        assert_eq!(cv.fmeasure.selected(), u64::from(m.u32s("selected").unwrap()[0]), "{at}: merged selected");
        assert_eq!(cv.fmeasure.target(), u64::from(m.u32s("target").unwrap()[0]), "{at}: merged target");
        assert_eq!(cv.fmeasure.true_positives(), u64::from(m.u32s("true-positives").unwrap()[0]), "{at}");
        bits(at, "merged precision", cv.fmeasure.precision(), m.f64s("precision").unwrap()[0]);
        bits(at, "merged recall", cv.fmeasure.recall(), m.f64s("recall").unwrap()[0]);
        bits(at, "merged f-measure", cv.fmeasure.f_measure(), m.f64s("f-measure").unwrap()[0]);
    }
}

#[test]
fn language_detector_folds_and_merged_accuracy_match_the_jvm() {
    let text = input("langdetect/samples.txt");
    let samples = language_samples(&text);
    let runs = runs("langdetect-");
    assert_eq!(runs.len(), 2, "the langdetect-default and langdetect-cutoff1 goldens");
    for (at, g) in &runs {
        let cv = language_cross_validation(at, g, &samples);
        let l = layer(g, "turbo:language-cv-folds");
        assert_eq!(cv.folds.len(), l.len(), "{at}: folds");
        let u = |c: &str, i: usize| u64::from(l.u32s(c).unwrap()[i]);
        for (i, f) in cv.folds.iter().enumerate() {
            let at = format!("{at} fold {i}");
            assert_eq!(f.index as u64, u("fold", i), "{at}");
            assert_eq!(f.training_samples as u64, u("training-samples", i), "{at}: training samples");
            assert_eq!(f.test_samples as u64, u("test-samples", i), "{at}: test samples");
            assert_eq!(f.test_samples as u64, u("documents", i), "{at}: document count");
            assert_eq!(f.events as u64, u("events", i), "{at}: events");
            assert_eq!(u64::from(f.event_checksum), u("crc32c", i), "{at}: event checksum");
            assert_eq!(f.inexact_events, 0, "{at}: the samples hold no tree bucket ties");
            assert_eq!(f.correct as u64, u("correct", i), "{at}: correct");
            bits(&at, "accuracy", f.accuracy, l.f64s("accuracy").unwrap()[i]);
        }
        let p = layer(g, "turbo:language-cv-predictions");
        let ours: Vec<(u32, u32, &str)> = cv
            .folds
            .iter()
            .flat_map(|f| f.predictions.iter().enumerate().map(move |(s, c)| (f.index as u32, s as u32, c.as_str())))
            .collect();
        let theirs: Vec<(u32, u32, &str)> = (0..p.len())
            .map(|r| (p.u32s("fold").unwrap()[r], p.u32s("sample").unwrap()[r], p.strs("language").unwrap().get(r)))
            .collect();
        assert_eq!(ours, theirs, "{at}: predicted languages");
        let m = layer(g, "turbo:language-cv-merged");
        assert_eq!(cv.document_count() as u64, u64::from(m.u32s("documents").unwrap()[0]), "{at}: documents");
        let theirs = m.f64s("accuracy").unwrap()[0];
        bits(at, "merged accuracy (JVM parity)", cv.jvm_accuracy(), theirs);
        let correct: usize = cv.folds.iter().map(|f| f.correct).sum();
        assert_eq!(cv.accuracy(), correct as f64 / cv.document_count() as f64, "{at}: exact accuracy");
        assert!((cv.accuracy() - theirs).abs() <= f64::EPSILON, "{at}: exact and JVM accuracy agree to rounding");
    }
}

#[test]
fn the_same_input_gives_the_same_folds_twice() {
    let text = input("tagged/names.txt");
    let samples = name_samples(&text);
    for (at, g) in &runs("namefind-bio") {
        assert_eq!(name_cross_validation(at, g, &samples), name_cross_validation(at, g, &samples), "{at}");
    }
    let text = input("langdetect/samples.txt");
    let samples = language_samples(&text);
    for (at, g) in &runs("langdetect-default") {
        assert_eq!(language_cross_validation(at, g, &samples), language_cross_validation(at, g, &samples), "{at}");
    }
}
