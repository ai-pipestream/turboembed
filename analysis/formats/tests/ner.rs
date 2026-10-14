//! The named entity corpus readers against the JVM: every `ner` golden of
//! `analysis/goldens` (written by `turbo-goldens generate`, see
//! analysis/goldens-tool/README.md) is what one of OpenNLP's streams read
//! from one input of the `ner` set. Each test reads the same input with
//! the reader here and rebuilds every layer of the golden from its
//! samples: the samples as written, the errors and where they came, the
//! tokens and the names.
//!
//! Beside the comparison, every token is checked to be the bytes of the
//! input its span names, inside the sample's source.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use turbo_formats::FormatError;
use turbo_formats::ner::brat::{AnnotationConfiguration, BratParser, BratSamples, read_brat_corpus};
use turbo_formats::ner::{
    BioNlpReader, BioNlpTypes, CensusReader, EvalitaLanguage, EvalitaReader, EvalitaTypes, GermEvalLayer,
    GermEvalReader, GermEvalTypes, MucSamples, NewlineSentences, RuleTokenizer, muc_files,
};
use turbo_formats::sample::NameSample;
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_tokenize::{TokenizerOptions, WhitespaceMode};

mod bundle;
use bundle::write_bundle;

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;

fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const SPANS: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("type", STR),
    ];
    const CONFIG: &[ColumnSpec] =
        &[ColumnSpec::new("file", STR), ColumnSpec::new("type", STR), ColumnSpec::new("class", STR)];
    const CONFIG_ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("file", STR), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc("turbo:sample-tokens", TOKENS),
        doc("turbo:names", SPANS),
        doc("turbo:brat-config", CONFIG),
        doc("turbo:brat-config-errors", CONFIG_ERRORS),
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
    drain_with(items, out, |m| m)
}

/// [`drain`] with each message passed through `message`.
fn drain_with<S>(
    items: impl Iterator<Item = Result<S, FormatError>>,
    out: &mut Rebuilt,
    message: impl Fn(String) -> String,
) -> Vec<S> {
    let mut samples = Vec::new();
    out.layer("turbo:read-errors");
    for item in items {
        match item {
            Ok(s) => samples.push(s),
            Err(e) => {
                let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class")).to_owned();
                out.row("turbo:read-errors", vec![samples.len().to_string(), class, message(e.to_string())]);
            }
        }
    }
    samples
}

fn written<S: Display>(samples: &[S], out: &mut Rebuilt) {
    out.layer("turbo:samples");
    for s in samples {
        out.row("turbo:samples", vec![s.to_string()]);
    }
}

/// The token and name layers of name samples, checking that every token
/// is the bytes of the input its span names, inside the sample's source.
fn names(input: &str, samples: &[NameSample<'_>], out: &mut Rebuilt) {
    written(samples, out);
    out.layer("turbo:sample-tokens");
    out.layer("turbo:names");
    for (i, s) in samples.iter().enumerate() {
        for t in s.tokens() {
            assert!(t.is_borrowed() && &input[t.span.range()] == t.as_str(), "a token's span names other bytes");
            assert!(s.source().start <= t.span.start && t.span.end <= s.source().end, "a token outside its source");
            out.row("turbo:sample-tokens", vec![i.to_string(), t.as_str().to_owned()]);
        }
        for n in s.names() {
            let row = vec![i.to_string(), n.start.to_string(), n.end.to_string(), n.kind.unwrap().to_owned()];
            out.row("turbo:names", row);
        }
    }
}

/// The texts of the `.sgm` files of the bundle `input`, written under
/// `root`, in the order the MUC factory reads them.
fn muc_documents(input: &str, root: &Path) -> Vec<String> {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root).unwrap();
    write_bundle(input, root);
    let files = muc_files(root).unwrap();
    files.iter().map(|f| String::from_utf8(std::fs::read(f).unwrap()).unwrap()).collect()
}

/// [`names`] for samples of several texts: each token is the bytes of
/// one of `texts` its span names.
fn names_in(texts: &[&str], samples: &[NameSample<'_>], out: &mut Rebuilt) {
    written(samples, out);
    out.layer("turbo:sample-tokens");
    out.layer("turbo:names");
    for (i, s) in samples.iter().enumerate() {
        for t in s.tokens() {
            let named = texts.iter().any(|d| d.get(t.span.range()) == Some(t.as_str()));
            assert!(t.is_borrowed() && named, "a token's span names other bytes");
            out.row("turbo:sample-tokens", vec![i.to_string(), t.as_str().to_owned()]);
        }
        for n in s.names() {
            let row = vec![i.to_string(), n.start.to_string(), n.end.to_string(), n.kind.unwrap().to_owned()];
            out.row("turbo:names", row);
        }
    }
}

fn rebuild(run: &str, input: &str, scratch: &Path) -> Rebuilt {
    let mut out = Rebuilt::default();
    match run {
        "brat-config" => {
            let _ = std::fs::remove_dir_all(scratch);
            std::fs::create_dir_all(scratch).unwrap();
            write_bundle(input, scratch);
            let mut confs: Vec<_> = std::fs::read_dir(scratch)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .filter(|n| n.ends_with(".conf"))
                .collect();
            confs.sort();
            out.layer("turbo:brat-config");
            out.layer("turbo:brat-config-errors");
            for name in confs {
                let text = String::from_utf8(std::fs::read(scratch.join(&name)).unwrap()).unwrap();
                match AnnotationConfiguration::parse(&text, WhitespaceMode::Unicode) {
                    Ok(c) => {
                        for (kind, class) in c.types() {
                            let row = vec![name.clone(), kind.to_owned(), class.java_name().to_owned()];
                            out.row("turbo:brat-config", row);
                        }
                    }
                    Err(e) => {
                        let row = vec![name.clone(), e.java_class().unwrap().to_owned(), e.to_string()];
                        out.row("turbo:brat-config-errors", row);
                    }
                }
            }
        }
        "brat-whitespace" | "brat-simple" | "brat-recursive-person-location" | "brat-whitespace+legacy" => {
            let mode = if run.ends_with("+legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
            let options = TokenizerOptions { whitespace: mode, keep_newlines: false };
            let tokenizer =
                if run == "brat-simple" { RuleTokenizer::Simple(options) } else { RuleTokenizer::Whitespace(options) };
            let recursive = run == "brat-recursive-person-location";
            let types = recursive
                .then(|| BratParser::<NewlineSentences, RuleTokenizer>::name_types_from_param("Person , Location"))
                .flatten();
            let _ = std::fs::remove_dir_all(scratch);
            std::fs::create_dir_all(scratch).unwrap();
            write_bundle(input, scratch);
            let config = std::fs::read_to_string(scratch.join("annotation.conf")).unwrap();
            AnnotationConfiguration::parse(&config, mode).unwrap();
            let documents = read_brat_corpus(scratch, recursive, mode).unwrap();
            let parser = BratParser::new(NewlineSentences(mode), tokenizer, types, mode).unwrap();
            let corpus = scratch.to_str().unwrap().to_owned();
            let samples =
                drain_with(BratSamples::new(&documents, &parser), &mut out, |m| m.replace(&corpus, "<corpus>"));
            names_in(
                &documents.iter().filter_map(|d| d.as_ref().ok()).map(|d| d.text()).collect::<Vec<_>>(),
                &samples,
                &mut out,
            );
        }
        "muc6-whitespace" | "muc6-simple" | "muc6-whitespace+legacy" => {
            let mode = if run.ends_with("+legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
            let options = TokenizerOptions { whitespace: mode, keep_newlines: false };
            let tokenizer =
                if run == "muc6-simple" { RuleTokenizer::Simple(options) } else { RuleTokenizer::Whitespace(options) };
            let documents = muc_documents(input, scratch);
            let samples = drain(MucSamples::new(documents.iter().map(String::as_str), &tokenizer, mode), &mut out);
            names_in(&documents.iter().map(String::as_str).collect::<Vec<_>>(), &samples, &mut out);
        }
        "germeval-outer" | "germeval-inner" | "germeval-outer-loc-misc" => {
            let types =
                GermEvalTypes::from_param(if run.ends_with("loc-misc") { "loc,misc" } else { "per,loc,org,misc" });
            let layer =
                GermEvalLayer::from_param(Some(if run == "germeval-inner" { "inner" } else { "outer" })).unwrap();
            let samples = drain(GermEvalReader::new(input, types, layer).unwrap(), &mut out);
            names(input, &samples, &mut out);
        }
        "bionlp" | "bionlp-cell-line" | "bionlp-dna-cell-type" => {
            let types = match run {
                "bionlp-cell-line" => "cell_line",
                "bionlp-dna-cell-type" => "DNA,cell_type",
                _ => "DNA,protein,cell_type,cell_line,RNA",
            };
            let samples = drain(BioNlpReader::new(input, BioNlpTypes::from_param(types)).unwrap(), &mut out);
            names(input, &samples, &mut out);
        }
        "evalita" | "evalita-per-gpe" => {
            let types = EvalitaTypes::from_param(if run == "evalita" { "per,org,loc,gpe" } else { "per,gpe" });
            let language = EvalitaLanguage::from_code("it").unwrap();
            let samples = drain(EvalitaReader::new(input, language, types).unwrap(), &mut out);
            names(input, &samples, &mut out);
        }
        "census" => {
            let samples = drain(CensusReader::new(input).unwrap(), &mut out);
            let listed: Vec<String> = samples.iter().map(|f| format!("[{}]", f.as_str())).collect();
            written(&listed, &mut out);
            out.layer("turbo:sample-tokens");
            for (i, f) in samples.iter().enumerate() {
                assert!(!input[f.span.range()].contains(' '), "a census name's span covers a space");
                out.row("turbo:sample-tokens", vec![i.to_string(), f.as_str().to_owned()]);
            }
        }
        _ => panic!("no reader for the ner run {run}"),
    }
    out
}

fn goldens() -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join("ner");
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("ner")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

#[test]
fn every_ner_golden_matches() {
    let data = turbo_goldens::data_dir();
    let shapes = shapes();
    let all = goldens();
    assert!(all.len() >= 32, "only {} ner goldens", all.len());
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ner-goldens");
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let mut rebuilt = rebuild(&run, &input, &scratch).0;
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
