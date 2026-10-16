//! The sentence, language and document corpus readers against the JVM:
//! every `corpora` golden of `analysis/goldens` (written by
//! `cargo run -p turbo-goldens-tool -- generate`, see
//! analysis/goldens-tool/README.md) is what one of OpenNLP's streams read
//! from one input of the `corpora` set. Each test reads the same input
//! with the reader here and rebuilds every layer of the golden from its
//! samples: the samples as written, the errors and where they came, the
//! texts with the byte spans of their sentences or tokens, the labels and
//! the tokens, and for 20 Newsgroups the order of the files.
//!
//! Beside the comparison, every sentence and token source is checked to
//! be a span of the input (or of the bundled file it was read from).

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use turbo_formats::FormatError;
use turbo_formats::corpora::{
    IrishSentenceBank, LeipzigCorpus, LeipzigParams, LetsmtDocument, LetsmtReader, MosesReader, NewsgroupDocument,
    NewsgroupFile, NkjpReader, NkjpSegmentation, NkjpText, newsgroup_listing, newsgroup_order,
};
use turbo_formats::ner::RuleTokenizer;
use turbo_formats::sample::{SentenceSample, TokenSample};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary, DictionaryOperation};
use turbo_tokenize::{TokenizerOptions, WhitespaceMode};

mod bundle;
use bundle::write_bundle;

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;

fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TEXTS: &[ColumnSpec] = &[ColumnSpec::new("text", STR)];
    const SPANS: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("start", U32), ColumnSpec::new("end", U32)];
    const LABELS: &[ColumnSpec] = &[ColumnSpec::new("label", STR)];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const FILES: &[ColumnSpec] = &[ColumnSpec::new("file", STR)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc("turbo:sample-texts", TEXTS),
        doc("turbo:sample-spans", SPANS),
        doc("turbo:sample-labels", LABELS),
        doc("turbo:sample-tokens", TOKENS),
        doc("turbo:listing", FILES),
        doc("turbo:files", FILES),
    ]
}

/// The layers rebuilt from the Rust samples, by id, each as rows of
/// cells written as text.
#[derive(Default)]
struct Rebuilt(BTreeMap<&'static str, Vec<Vec<String>>>);

impl Rebuilt {
    fn row(&mut self, id: &'static str, cells: Vec<String>) {
        self.0.entry(id).or_default().push(cells);
    }

    /// Declares a layer that may have no rows.
    fn layer(&mut self, id: &'static str) {
        self.0.entry(id).or_default();
    }

    fn error(&mut self, after: usize, e: &dyn ErrorLike) {
        self.row("turbo:read-errors", vec![after.to_string(), e.class(), e.message()]);
    }
}

/// An error with the class and message of the exception Java throws.
trait ErrorLike {
    fn class(&self) -> String;
    fn message(&self) -> String;
}

impl ErrorLike for FormatError {
    fn class(&self) -> String {
        self.java_class().unwrap_or_else(|| panic!("{self} has no Java class")).to_owned()
    }

    fn message(&self) -> String {
        self.to_string()
    }
}

fn cells(layer: &GoldenLayer) -> Vec<Vec<String>> {
    (0..layer.len())
        .map(|i| {
            layer
                .columns
                .iter()
                .map(|c| match &c.data {
                    ColumnData::U32(v) => v[i].to_string(),
                    ColumnData::Str(v) => v.get(i).to_owned(),
                    ColumnData::F64(v) => v[i].to_string(),
                })
                .collect()
        })
        .collect()
}

/// Reads every item of a reader, keeping the errors with the number of
/// samples before each, as the generator records them.
fn drain<S>(items: impl Iterator<Item = Result<S, FormatError>>, out: &mut Rebuilt) -> Vec<S> {
    let mut samples = Vec::new();
    for item in items {
        match item {
            Ok(s) => samples.push(s),
            Err(e) => out.error(samples.len(), &e),
        }
    }
    samples
}

fn written<S: Display>(samples: &[S], out: &mut Rebuilt) {
    for s in samples {
        out.row("turbo:samples", vec![s.to_string()]);
    }
}

/// The text and span layers of sentence samples, checking that each
/// sentence's source is a span of one of `sources`.
fn sentences(samples: &[SentenceSample<'_>], sources: &[&str], out: &mut Rebuilt) {
    written(samples, out);
    out.layer("turbo:sample-texts");
    out.layer("turbo:sample-spans");
    for (i, s) in samples.iter().enumerate() {
        out.row("turbo:sample-texts", vec![s.document().to_owned()]);
        for (span, source) in s.sentences().iter().zip(s.sources()) {
            assert!(sources.iter().any(|t| t.get(source.range()).is_some()), "a sentence source outside the input");
            out.row("turbo:sample-spans", vec![i.to_string(), span.start.to_string(), span.end.to_string()]);
        }
    }
}

/// The text and span layers of token samples.
fn tokens(samples: &[TokenSample<'_>], input: &str, out: &mut Rebuilt) {
    written(samples, out);
    out.layer("turbo:sample-texts");
    out.layer("turbo:sample-spans");
    for (i, s) in samples.iter().enumerate() {
        out.row("turbo:sample-texts", vec![s.text().to_owned()]);
        for (span, source) in s.tokens().iter().zip(s.sources()) {
            assert!(input.get(source.range()).is_some(), "a token source outside the input");
            out.row("turbo:sample-spans", vec![i.to_string(), span.start.to_string(), span.end.to_string()]);
        }
    }
}

/// Writes the bundle `input` under a fresh `root`.
fn unbundle(input: &str, root: &Path) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root).unwrap();
    write_bundle(input, root);
}

/// The detokenizer dictionary of the `detokenizer` config record: tokens
/// and operations, separated by spaces.
fn dictionary(record: &str) -> DetokenizerDictionary {
    let mut d = DetokenizerDictionary::new();
    let parts: Vec<&str> = record.split(' ').collect();
    for pair in parts.chunks(2) {
        d.insert(pair[0], DictionaryOperation::from_attribute(pair[1]).unwrap());
    }
    d
}

fn config<'g>(golden: &'g Golden, key: &str) -> &'g str {
    let entry = golden.header.config.iter().find(|c| c.key == key);
    &entry.unwrap_or_else(|| panic!("no config {key}")).value
}

fn rebuild(run: &str, golden: &Golden, input: &str, scratch: &Path) -> Rebuilt {
    let mut out = Rebuilt::default();
    out.layer("turbo:samples");
    out.layer("turbo:read-errors");
    let base = run.split('+').next().unwrap();
    let mode = if run.ends_with("+legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
    match base {
        "moses" => {
            let samples = drain(MosesReader::new(input).unwrap(), &mut out);
            sentences(&samples, &[input], &mut out);
        }
        "letsmt" | "letsmt-detokenized" => {
            let dict = (base == "letsmt-detokenized").then(|| dictionary(config(golden, "detokenizer")));
            let samples = match LetsmtDocument::parse(input) {
                Err(e) => {
                    out.error(0, &e);
                    Vec::new()
                }
                Ok(document) => {
                    let detokenizer = dict.as_ref().map(|d| (Detokenizer::new(d), mode));
                    drain(LetsmtReader::new(document, detokenizer), &mut out)
                }
            };
            sentences(&samples, &[input], &mut out);
        }
        "nkjp" => {
            unbundle(input, scratch);
            let segmentation = std::fs::read_to_string(scratch.join("ann_segmentation.xml")).unwrap();
            let text = std::fs::read_to_string(scratch.join("text.xml")).unwrap();
            let parsed = NkjpSegmentation::parse(&segmentation).and_then(|s| Ok((s, NkjpText::parse(&text)?)));
            let samples = match &parsed {
                Err(e) => {
                    out.error(0, e);
                    Vec::new()
                }
                Ok((s, t)) => drain(NkjpReader::new(s.clone(), t), &mut out),
            };
            sentences(&samples, &[&segmentation], &mut out);
        }
        "irish-sentences" | "irish-tokens" => match IrishSentenceBank::parse(input) {
            Err(e) => {
                out.error(0, &e);
                if base == "irish-tokens" {
                    tokens(&[], input, &mut out);
                } else {
                    sentences(&[], &[input], &mut out);
                }
            }
            Ok(bank) => {
                if base == "irish-tokens" {
                    let samples = drain(bank.token_samples(), &mut out);
                    tokens(&samples, input, &mut out);
                } else {
                    let samples = drain(bank.sentence_sample().into_iter(), &mut out);
                    sentences(&samples, &[input], &mut out);
                }
            }
        },
        "leipzig-2x3" | "leipzig-1x4-skip2" => {
            unbundle(input, scratch);
            let number = |key| config(golden, key).parse().unwrap();
            let params = LeipzigParams {
                sentences_per_sample: number("sentencesPerSample"),
                samples_per_language: number("samplesPerLanguage"),
                samples_to_skip: number("samplesToSkip"),
            };
            let corpus = LeipzigCorpus::open(scratch).unwrap();
            out.layer("turbo:sample-labels");
            out.layer("turbo:sample-texts");
            match corpus.samples(params, |p| std::fs::read_to_string(p)) {
                Err(turbo_formats::corpora::DirectoryError::Format(e)) => {
                    let corpus = scratch.to_str().unwrap();
                    let row = vec!["0".to_owned(), e.class(), e.message().replace(corpus, "<corpus>")];
                    out.row("turbo:read-errors", row);
                }
                Err(e) => panic!("{e}"),
                Ok(samples) => {
                    written(&samples, &mut out);
                    for s in &samples {
                        let text = std::fs::read_to_string(&corpus.files()[s.file]).unwrap();
                        for span in &s.sentences {
                            assert!(text.get(span.range()).is_some(), "a sentence outside its file");
                        }
                        let l = s.language_sample();
                        out.row("turbo:sample-labels", vec![l.language.as_str().to_owned()]);
                        out.row("turbo:sample-texts", vec![l.context.as_str().to_owned()]);
                    }
                }
            }
        }
        "20newsgroup-whitespace" | "20newsgroup-simple" => {
            unbundle(input, scratch);
            let options = TokenizerOptions { whitespace: mode, keep_newlines: false };
            let tokenizer = if base == "20newsgroup-simple" {
                RuleTokenizer::Simple(options)
            } else {
                RuleTokenizer::Whitespace(options)
            };
            // The JVM's listing order, which depends on its file system;
            // the files are the ones the Rust listing finds.
            let listed: Vec<String> =
                golden.layer("turbo:listing").unwrap().strs("file").unwrap().iter().map(str::to_owned).collect();
            let mut ours: Vec<String> = newsgroup_listing(scratch)
                .unwrap()
                .iter()
                .map(|f| f.path.strip_prefix(scratch).unwrap().to_str().unwrap().to_owned())
                .collect();
            ours.sort();
            let mut theirs = listed.clone();
            theirs.sort();
            assert_eq!(ours, theirs, "the files listed");
            for f in &listed {
                out.row("turbo:listing", vec![f.clone()]);
            }
            let listing: Vec<NewsgroupFile> = listed
                .iter()
                .map(|f| NewsgroupFile { category: f.split('/').next().unwrap().to_owned(), path: scratch.join(f) })
                .collect();
            let ordered = newsgroup_order(Path::new(config(golden, "dataDir")), listing);
            out.layer("turbo:files");
            out.layer("turbo:sample-labels");
            out.layer("turbo:sample-tokens");
            let documents: Vec<NewsgroupDocument> =
                ordered.iter().map(|f| NewsgroupDocument::read(f).unwrap()).collect();
            let samples: Vec<_> = documents.iter().map(|d| d.sample(&tokenizer).unwrap()).collect();
            written(&samples, &mut out);
            for (i, (f, s)) in ordered.iter().zip(&samples).enumerate() {
                out.row("turbo:files", vec![f.path.strip_prefix(scratch).unwrap().to_str().unwrap().to_owned()]);
                out.row("turbo:sample-labels", vec![s.category.as_str().to_owned()]);
                for t in &s.tokens {
                    assert_eq!(&documents[i].text[t.span.range()], t.as_str(), "a token's span names other bytes");
                    out.row("turbo:sample-tokens", vec![i.to_string(), t.as_str().to_owned()]);
                }
            }
        }
        _ => panic!("no reader for the corpora run {run}"),
    }
    out
}

fn goldens(root: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for run in std::fs::read_dir(root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("corpora")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

#[test]
fn every_corpora_golden_matches() {
    let data = turbo_goldens::data_dir();
    let root = data.join("corpora");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; make the goldens with `{GENERATE}`", root.display());
        return;
    }
    let shapes = shapes();
    let all = goldens(&root);
    assert!(all.len() >= 44, "only {} corpora goldens", all.len());
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("corpora-goldens");
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let mut rebuilt = rebuild(&run, &golden, &input, &scratch).0;
        for layer in &golden.layers {
            let ours =
                rebuilt.remove(layer.id.as_str()).unwrap_or_else(|| panic!("{}: no {}", path.display(), layer.id));
            let theirs = cells(layer);
            for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
                assert_eq!(a, b, "{}: {} row {i}", path.display(), layer.id);
            }
            assert_eq!(ours.len(), theirs.len(), "{}: {} rows", path.display(), layer.id);
        }
        assert!(rebuilt.is_empty(), "{}: layers the golden lacks: {:?}", path.display(), rebuilt.keys());
    }
}
