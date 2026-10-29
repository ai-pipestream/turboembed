//! Every run by name, and the one that opens it.

use super::language::Detector;
use super::normalize::{NormalizerRun, RUN_NAMES as NORMALIZERS};
use super::numeric::{MoneyLayer, Numeric};
use super::pii::PiiRun;
use super::sentences::{MaxentSentences, NewlineSentences};
use super::subwords::{VOCABULARY, WordPiece};
use super::tagging::{DictionaryRun, Tagger};
use super::temporal::Temporal;
use super::tokens::{Lattice, Maxent, Plain, Uax29, Unigram};
use super::{Run, split_options};
use crate::resources::{Resources, Unavailable};

/// The annotator directories the generator writes that the differential
/// does not cover: the training stack (`train`, `train-events`,
/// `train-langdetect`, `train-liblinear`), the format readers (`formats`,
/// `masc`, `ner`, `corpora`, `ad`) and the parse and head rule readers (`parse`,
/// `head-rules`) compare files, not layers of a text; the evaluators and the
/// cross-validators (`cross-validate`) compare scores; the detokenizer
/// runs on token lists; the BPE trainer learns merges from a whole set; the
/// gazetteers answer queries line by line and load whole tables, and the
/// place profiles load whole tables; the
/// sequence runs (`string-pattern`, `features`, `codec`, `beam`,
/// `tag-dictionary`, `tag-format`), the name finders, the
/// Hunspell runs, the SVM document categorizer (`svm-doccat`) read input sets of their own,
/// which turbo-sequence, turbo-namefind, turbo-hunspell
/// and turbo-doccat compare with; the other
/// `namefind` runs train name finders from name samples and find names in
/// them, and the sentiment runs (`sentiment`) train models on a corpus of
/// their own and score their own documents, which turbo-train-components
/// compares with; the maxent document categorizer runs (`doccat`) train
/// models on a downloaded corpus and categorize documents of their own; the
/// PII rewrites (`pii-rewrite`) compare whole rewritten
/// texts, labels and audit reports rather than layers, which turbo-pii
/// compares with; the neural part-of-speech taggers (`postag-neural`) train
/// models of their own and tag token lists rather than texts; the CoNLL-U
/// dependency and CorefUD readers (`conllu`) and the OntoNotes and French
/// Treebank readers (`ontonotes`, `frenchtreebank`) read files of their own,
/// which turbo-formats compares with; the sentence vector runs
/// (`sentvectors`) need an encoder bundle, and turbo-analysis-embed's
/// `sentence_vectors_dl` test compares with them.
pub const NOT_COVERED: [&str; 27] = [
    "train",
    "train-events",
    "train-langdetect",
    "train-liblinear",
    "formats",
    "evaluate",
    "cross-validate",
    "detokenize",
    "bpe",
    "masc",
    "ner",
    "ad",
    "string-pattern",
    "features",
    "codec",
    "beam",
    "tag-dictionary",
    "tag-format",
    "namefind",
    "svm-doccat",
    "sentiment",
    "doccat",
    "pii-rewrite",
    "postag-neural",
    "conllu",
    "corpora",
    "sentvectors",
];

/// Historical directory inventory, including retired classical runs.
/// Native execution uses [`native_runs`]; retirement never opens resources.
/// The directory inventory remains until its owning snapshot migration lands.
pub const RUNS: &[(&str, &str)] = &[
    ("sentdetect", "en-abbrev"),
    ("sentdetect", "en-plain"),
    ("sentdetect", "en-plain-perceptron-rust"),
    ("sentdetect", "en-plain-qn-rust"),
    ("sentdetect", "en-plain-rust"),
    ("sentdetect", "newline"),
    ("tokenize", "simple"),
    ("tokenize", "simple+keep-new-lines"),
    ("tokenize", "tokenizer-abbreviations"),
    ("tokenize", "uax29"),
    ("tokenize", "whitespace"),
    ("tokenize", "whitespace+keep-new-lines"),
    ("unigram", "jieba-dict-subset"),
    ("unigram", "zh-unigram-fixture"),
    ("unigram", "zh-unigram-modes"),
    ("lattice", "mecab-fixture-ja"),
    ("lattice", "mecab-fixture-ko"),
    ("lattice", "mecab-fixture-partial"),
    ("lattice", "mecab-ipadic-2.7.0-20070801"),
    ("lattice", "mecab-ko-dic-2.1.1-20180720"),
    ("lattice", "unidic-mecab-2.1.2"),
    ("langdetect", "langdetect-fixture-rust"),
    ("langdetect", "langdetect-fixture-rust+probing"),
    ("langdetect", "langdetect-fixture-rust+probing-chunk50"),
    ("normalize", "whitespace"),
    ("normalize", "whitespace-keep-lines"),
    ("normalize", "whitespace-keep-paragraphs"),
    ("normalize", "dash"),
    ("normalize", "quote"),
    ("normalize", "bullet"),
    ("normalize", "ellipsis"),
    ("normalize", "digit"),
    ("normalize", "invisible"),
    ("normalize", "dehyphenation"),
    ("normalize", "nfc"),
    ("normalize", "nfkc"),
    ("normalize", "case-fold"),
    ("normalize", "case-fold-tr"),
    ("normalize", "full-case-fold"),
    ("normalize", "accent-fold"),
    ("normalize", "german-umlaut"),
    ("normalize", "confusable-skeleton"),
    ("normalize", "url"),
    ("normalize", "bounded-url"),
    ("normalize", "social-media"),
    ("normalize", "number"),
    ("normalize", "shrink"),
    ("normalize", "english-contraction"),
    ("normalize", "emoji"),
    ("normalize", "emoji-to-emoticon"),
    ("normalize", "emoticon-to-emoji"),
    ("normalize", "symbol-joiner"),
    ("normalize", "default-chain"),
    ("normalize", "umlaut-full-case-whitespace-chain"),
    ("numbers", "latin-eu"),
    ("numbers", "latin-us"),
    ("money", "latin-eu"),
    ("money", "latin-us"),
    ("money-convert", "latin-eu"),
    ("money-convert", "latin-us"),
    ("quantity", "latin-eu"),
    ("quantity", "latin-us"),
    ("temporal", "dateline"),
    ("temporal", "fixed"),
    ("pii", "payment"),
    ("pii", "contact"),
    ("pii", "network"),
    ("pii", "secrets"),
    ("pii", "crypto"),
    ("pii", "us-identity"),
    ("pii", "eu-identity"),
    ("pii", "ca-identity"),
    ("pii", "device"),
    ("pii", "all-structured"),
    ("subwords", VOCABULARY),
];

/// Classical archive consumers retained only in the historical inventory.
/// Unknown names within a retired family are refused before resource lookup too.
pub fn is_retired(annotator: &str, run: &str) -> bool {
    let base = run.split('+').next().unwrap_or("");
    match annotator {
        "sentdetect" => !matches!(base, "newline" | "rule" | "paragraph"),
        "tokenize" => !matches!(base, "simple" | "whitespace" | "uax29"),
        "postag" | "lemmatize" | "chunk" | "langdetect" => true,
        _ => false,
    }
}

/// Retained native runs, including rule sentences and paragraph structure.
/// Historical JVM expected outputs play no part in this execution registry.
pub fn native_runs() -> impl Iterator<Item = (&'static str, &'static str)> {
    RUNS.iter().copied().filter(|(annotator, run)| !is_retired(annotator, run)).chain([
        ("sentdetect", "rule"),
        ("sentdetect", "paragraph"),
        ("lemmatize-dictionary", "lemmatizer-dictionary"),
    ])
}

/// Opens the Rust side of `<annotator>/<run>`.
pub fn open(annotator: &str, run: &str, res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
    if is_retired(annotator, run) {
        return Err(Unavailable::NotCovered(format!("retired classical run {annotator}/{run}")));
    }
    let (base, options) = split_options(run);
    let not_covered = || Unavailable::NotCovered(format!("{annotator}/{run}"));
    if annotator == "tokenize" && base == "uax29" && !options.is_empty() {
        return Err(not_covered());
    }
    match annotator {
        "sentdetect" if matches!(base, "newline" | "rule" | "paragraph") => {
            NewlineSentences::open_native(base, &options)
        }
        "sentdetect" => MaxentSentences::open(base, &options, res),
        "tokenize" => match base {
            "simple" => Plain::open(true, &options),
            "whitespace" => Plain::open(false, &options),
            "uax29" if options.is_empty() => Uax29::open(),
            _ => Maxent::open(base, &options, res),
        },
        "unigram" => Unigram::open(base, &options, res),
        "lattice" => Lattice::open(base, &options, res),
        "postag" => Tagger::open(base, &options, res),
        "lemmatize-dictionary" if run == "lemmatizer-dictionary" => DictionaryRun::open(res),
        "langdetect" => Detector::open(base, &options, res),
        "normalize" if NORMALIZERS.contains(&run) => NormalizerRun::open(run),
        "numbers" => Numeric::open_numbers(run),
        "money" => Numeric::open_money(run, MoneyLayer::Extracted, res),
        "money-convert" => Numeric::open_money(run, MoneyLayer::Converted, res),
        "quantity" => Numeric::open_quantities(run),
        "temporal" => Temporal::open(run),
        "pii" => PiiRun::open(run),
        "subwords" if run == VOCABULARY => WordPiece::open(res),
        _ => Err(not_covered()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Published OpenNLP 2.5 UD / language-detector run directories still on
    /// disk until #1644 to #1648 delete them. Each must stay off `RUNS` and
    /// must open as [`Unavailable::NotCovered`].
    const RETIRED_PUBLISHED_OPENNLP_RUNS: &[(&str, &str)] = &[
        ("sentdetect", "opennlp-de-ud-gsd-sentence-1.3-2.5.4"),
        ("sentdetect", "opennlp-en-ud-ewt-sentence-1.3-2.5.4"),
        ("sentdetect", "opennlp-pt-ud-gsd-sentence-1.3-2.5.4"),
        ("tokenize", "opennlp-de-ud-gsd-tokens-1.3-2.5.4"),
        ("tokenize", "opennlp-de-ud-gsd-tokens-1.3-2.5.4+alphanumeric"),
        ("tokenize", "opennlp-en-ud-ewt-tokens-1.3-2.5.4"),
        ("tokenize", "opennlp-en-ud-ewt-tokens-1.3-2.5.4+alphanumeric"),
        ("tokenize", "opennlp-en-ud-ewt-tokens-1.3-2.5.4+english-dictionary"),
        ("tokenize", "opennlp-es-ud-gsd-tokens-1.3-2.5.4"),
        ("tokenize", "opennlp-es-ud-gsd-tokens-1.3-2.5.4+alphanumeric"),
        ("tokenize", "opennlp-fr-ud-gsd-tokens-1.3-2.5.4"),
        ("tokenize", "opennlp-fr-ud-gsd-tokens-1.3-2.5.4+alphanumeric"),
        ("tokenize", "opennlp-pt-ud-gsd-tokens-1.3-2.5.4"),
        ("tokenize", "opennlp-pt-ud-gsd-tokens-1.3-2.5.4+alphanumeric"),
        ("postag", "opennlp-de-ud-gsd-pos-1.3-2.5.4"),
        ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4"),
        ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4+beam1"),
        ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4+beam10"),
        ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4+penn"),
        ("lemmatize", "opennlp-de-ud-gsd-lemmas-1.3-2.5.4"),
        ("lemmatize", "opennlp-en-ud-ewt-lemmas-1.3-2.5.4"),
        ("langdetect", "langdetect-183"),
        ("langdetect", "langdetect-183+probing"),
        ("langdetect", "langdetect-183+probing-chunk50"),
    ];

    fn is_retired_published_opennlp_run(run: &str) -> bool {
        let base = run.split('+').next().unwrap_or(run);
        base.contains("1.3-2.5.4") || base == "langdetect-183"
    }

    #[test]
    fn the_catalog_names_every_run_directory_the_differential_covers() {
        let data = turbo_goldens::data_dir();
        let mut active = Vec::new();
        let mut retired_on_disk = Vec::new();
        for annotator in std::fs::read_dir(&data).unwrap() {
            let annotator = annotator.unwrap();
            let name = annotator.file_name().into_string().unwrap();
            if !annotator.file_type().unwrap().is_dir() || NOT_COVERED.contains(&name.as_str()) {
                continue;
            }
            // Inputs and test data, not golden runs.
            if ["inputs", "fixtures", "train-fixtures", "rates", "conformance"].contains(&name.as_str()) {
                continue;
            }
            for run in std::fs::read_dir(annotator.path()).unwrap() {
                let run = run.unwrap().file_name().into_string().unwrap();
                if is_retired_published_opennlp_run(&run) {
                    retired_on_disk.push((name.clone(), run));
                } else {
                    active.push((name.clone(), run));
                }
            }
        }
        active.sort();
        retired_on_disk.sort();
        let mut listed: Vec<(String, String)> = RUNS.iter().map(|(a, r)| (a.to_string(), r.to_string())).collect();
        listed.sort();
        // Native runs may outlive their JVM golden trees; skip families
        // whose data directory is not checked in yet.
        for family in ["normalize", "numbers", "money", "money-convert", "quantity", "temporal", "pii"] {
            if !data.join(family).is_dir() {
                listed.retain(|(a, _)| a.as_str() != family);
            }
        }
        assert_eq!(listed, active);

        // Every retired published-model directory still on disk must be on
        // the denylist, and every denylist entry for a covered annotator that
        // still exists must appear here (until #1644 to #1648 delete them).
        let denylist: Vec<(String, String)> =
            RETIRED_PUBLISHED_OPENNLP_RUNS.iter().map(|(a, r)| (a.to_string(), r.to_string())).collect();
        for (annotator, run) in &retired_on_disk {
            assert!(
                denylist.iter().any(|(a, r)| a == annotator && r == run),
                "on-disk retired published OpenNLP run {annotator}/{run} is not on RETIRED_PUBLISHED_OPENNLP_RUNS"
            );
            assert!(
                is_retired_published_opennlp_run(run),
                "{annotator}/{run} must match the retired published OpenNLP denylist predicate"
            );
        }
        for (annotator, run) in RETIRED_PUBLISHED_OPENNLP_RUNS {
            if NOT_COVERED.contains(annotator) {
                continue;
            }
            let path = data.join(annotator).join(run);
            if path.is_dir() {
                assert!(
                    retired_on_disk.iter().any(|(a, r)| a == annotator && r == *run),
                    "denylist entry {annotator}/{run} exists on disk but was not collected"
                );
            }
        }
    }

    #[test]
    fn the_catalog_lists_no_retired_opennlp_published_model() {
        for (annotator, run) in RUNS {
            assert!(
                !is_retired_published_opennlp_run(run),
                "catalog still names retired published model {annotator}/{run}"
            );
            assert!(
                !RETIRED_PUBLISHED_OPENNLP_RUNS.contains(&(annotator, run)),
                "catalog still names denylisted run {annotator}/{run}"
            );
        }
    }

    #[test]
    fn retired_published_opennlp_runs_are_not_covered() {
        let res = Resources::new("/nonexistent");
        for &(annotator, run) in RETIRED_PUBLISHED_OPENNLP_RUNS {
            assert!(is_retired_published_opennlp_run(run), "{annotator}/{run} must match the denylist predicate");
            assert!(
                matches!(open(annotator, run, &res), Err(Unavailable::NotCovered(_))),
                "{annotator}/{run} must not open from the catalog"
            );
        }
    }

    #[test]
    fn runs_without_a_model_open_from_the_fixtures_alone() {
        let res = Resources::new("/nonexistent");
        for (annotator, run) in [
            ("sentdetect", "rule"),
            ("sentdetect", "newline"),
            ("sentdetect", "paragraph"),
            ("tokenize", "simple+keep-new-lines"),
            ("tokenize", "uax29"),
            ("normalize", "default-chain"),
            ("normalize", "umlaut-full-case-whitespace-chain"),
            ("money-convert", "latin-eu"),
            ("pii", "all-structured"),
        ] {
            open(annotator, run, &res).unwrap_or_else(|e| panic!("{annotator}/{run}: {e}"));
        }
        assert!(matches!(open("tokenize", "tokenizer-abbreviations", &res), Err(Unavailable::NotCovered(_))));
        assert!(matches!(open("train", "anything", &res), Err(Unavailable::NotCovered(_))));
        // There is one whitespace definition, so no run takes a mode.
        for (annotator, run) in [("sentdetect", "en-abbrev+legacy"), ("tokenize", "simple+legacy")] {
            assert!(matches!(open(annotator, run, &res), Err(Unavailable::NotCovered(_))), "{run}");
        }
        assert!(open("normalize", "case-fold-tr", &res).is_ok());
    }
}
