//! Every run by name, and the one that opens it.

use super::language::Detector;
use super::normalize::{NormalizerRun, RUN_NAMES as NORMALIZERS};
use super::numeric::{MoneyLayer, Numeric};
use super::pii::PiiRun;
use super::sentences::{MaxentSentences, NewlineSentences};
use super::subwords::{VOCABULARY, WordPiece};
use super::tagging::{Chunker, FIXTURE_CHUNKERS, Lemmatizer, Tagger};
use super::temporal::Temporal;
use super::tokens::{Lattice, Maxent, Plain, Uax29, Unigram};
use super::{Run, split_options};
use crate::resources::{Resources, Unavailable};

/// The annotator directories the generator writes that the differential
/// does not cover: the training stack (`train`, `train-events`,
/// `train-langdetect`, `train-liblinear`), the format readers (`formats`, `masc`, `ner`, `ad`) and
/// the parse and head rule readers (`parse`, `head-rules`) compare files,
/// not layers of a text; the evaluators compare scores; the detokenizer
/// runs on token lists; the BPE trainer learns merges from a whole set; the
/// gazetteers answer queries line by line and load whole tables, and the
/// place profiles load whole tables; the
/// sequence runs (`string-pattern`, `features`, `codec`, `beam`,
/// `tag-dictionary`, `tag-format`), the name finders, the stemmers, the
/// Hunspell runs, the stopword lists, the glossary matchers and readers and
/// the SVM document categorizer (`svm-doccat`) read input sets of their own,
/// which turbo-sequence, turbo-namefind, turbo-stem, turbo-hunspell,
/// turbo-stopword, turbo-glossary and turbo-doccat compare with; the other
/// `namefind` runs train name finders from name samples and find names in
/// them, and the sentiment runs (`sentiment`) train models on a corpus of
/// their own and score their own documents, which turbo-train-components
/// compares with; the PII rewrites (`pii-rewrite`) compare whole rewritten
/// texts, labels and audit reports rather than layers, which turbo-pii
/// compares with.
pub const NOT_COVERED: [&str; 29] = [
    "train",
    "train-events",
    "train-langdetect",
    "train-liblinear",
    "formats",
    "evaluate",
    "detokenize",
    "bpe",
    "masc",
    "ner",
    "ad",
    "parse",
    "head-rules",
    "gazetteer",
    "place-profiles",
    "string-pattern",
    "features",
    "codec",
    "beam",
    "tag-dictionary",
    "tag-format",
    "namefind",
    "stem",
    "hunspell",
    "stopword",
    "glossary",
    "svm-doccat",
    "sentiment",
    "pii-rewrite",
];

/// Every run the differential covers, as `(annotator, run)`: the golden
/// directories under `analysis/goldens/data`. A run whose model is not
/// present opens as `Unavailable::Missing`.
pub const RUNS: &[(&str, &str)] = &[
    ("sentdetect", "en-abbrev"),
    ("sentdetect", "en-abbrev+legacy"),
    ("sentdetect", "en-plain"),
    ("sentdetect", "en-plain+legacy"),
    ("sentdetect", "en-plain-perceptron-rust"),
    ("sentdetect", "en-plain-perceptron-rust+legacy"),
    ("sentdetect", "en-plain-qn-rust"),
    ("sentdetect", "en-plain-qn-rust+legacy"),
    ("sentdetect", "en-plain-rust"),
    ("sentdetect", "en-plain-rust+legacy"),
    ("sentdetect", "newline"),
    ("sentdetect", "newline+legacy"),
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
    ("tokenize", "simple"),
    ("tokenize", "simple+keep-new-lines"),
    ("tokenize", "simple+keep-new-lines+legacy"),
    ("tokenize", "simple+legacy"),
    ("tokenize", "tokenizer-abbreviations"),
    ("tokenize", "uax29"),
    ("tokenize", "whitespace"),
    ("tokenize", "whitespace+keep-new-lines"),
    ("tokenize", "whitespace+keep-new-lines+legacy"),
    ("tokenize", "whitespace+legacy"),
    ("unigram", "jieba-dict-subset"),
    ("unigram", "jieba-dict-subset+legacy"),
    ("unigram", "zh-unigram-fixture"),
    ("unigram", "zh-unigram-fixture+legacy"),
    ("unigram", "zh-unigram-modes"),
    ("unigram", "zh-unigram-modes+legacy"),
    ("lattice", "mecab-fixture-ja"),
    ("lattice", "mecab-fixture-ja+legacy"),
    ("lattice", "mecab-fixture-ko"),
    ("lattice", "mecab-fixture-partial"),
    ("lattice", "mecab-ipadic-2.7.0-20070801"),
    ("lattice", "mecab-ko-dic-2.1.1-20180720"),
    ("postag", "opennlp-de-ud-gsd-pos-1.3-2.5.4"),
    ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4"),
    ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4+beam1"),
    ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4+beam10"),
    ("postag", "opennlp-en-ud-ewt-pos-1.3-2.5.4+penn"),
    ("postag", "pos-fixture"),
    ("postag", "pos-fixture-rust"),
    ("lemmatize", "lemma-fixture-rust"),
    ("lemmatize", "opennlp-de-ud-gsd-lemmas-1.3-2.5.4"),
    ("lemmatize", "opennlp-en-ud-ewt-lemmas-1.3-2.5.4"),
    ("lemmatize-dictionary", "lemmatizer-dictionary"),
    ("chunk", "chunker-fixture"),
    ("chunk", "chunker-fixture+beam1"),
    ("chunk", "chunker-fixture-rust"),
    ("langdetect", "langdetect-183"),
    ("langdetect", "langdetect-fixture-rust"),
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

/// Opens the Rust side of `<annotator>/<run>`.
pub fn open(annotator: &str, run: &str, res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
    let (base, options) = split_options(run);
    let not_covered = || Unavailable::NotCovered(format!("{annotator}/{run}"));
    match annotator {
        "sentdetect" if base == "newline" => NewlineSentences::open(&options),
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
        "lemmatize" if options.is_empty() => Lemmatizer::open_maxent(base, res),
        "lemmatize-dictionary" if run == "lemmatizer-dictionary" => Lemmatizer::open_dictionary(res),
        "chunk" if FIXTURE_CHUNKERS.contains(&base) => Chunker::open(base, &options, res),
        "langdetect" if options.is_empty() => Detector::open(base, res),
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

    #[test]
    fn the_catalog_names_every_run_directory_the_differential_covers() {
        let data = turbo_goldens::data_dir();
        let mut on_disk = Vec::new();
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
                on_disk.push((name.clone(), run));
            }
        }
        on_disk.sort();
        let mut listed: Vec<(String, String)> = RUNS.iter().map(|(a, r)| (a.to_string(), r.to_string())).collect();
        listed.sort();
        // The PII goldens come with the first `generate` that knows the
        // packs; until they are on disk, their runs have nothing to match.
        if !data.join("pii").is_dir() {
            listed.retain(|(a, _)| a != "pii");
        }
        // One run has goldens the differential leaves out: the Turkish
        // locale case fold, which the port has no counterpart of.
        on_disk.retain(|(a, r)| !(a == "normalize" && r == "case-fold-tr"));
        assert_eq!(listed, on_disk);
    }

    #[test]
    fn runs_without_a_model_open_from_the_fixtures_alone() {
        let res = Resources::new("/nonexistent");
        for (annotator, run) in [
            ("sentdetect", "en-abbrev+legacy"),
            ("tokenize", "simple+keep-new-lines"),
            ("tokenize", "uax29"),
            ("normalize", "default-chain"),
            ("money-convert", "latin-eu"),
            ("chunk", "chunker-fixture+beam1"),
            ("chunk", "chunker-fixture-rust"),
            ("pii", "all-structured"),
        ] {
            open(annotator, run, &res).unwrap_or_else(|e| panic!("{annotator}/{run}: {e}"));
        }
        assert!(matches!(open("tokenize", "opennlp-en-ud-ewt-tokens-1.3-2.5.4", &res), Err(Unavailable::Missing(_))));
        assert!(matches!(open("train", "anything", &res), Err(Unavailable::NotCovered(_))));
        assert!(matches!(open("normalize", "case-fold-tr", &res), Err(Unavailable::NotCovered(_))));
    }
}
