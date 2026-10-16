//! The Arvores Deitadas readers against the JVM: every `ad` golden of
//! `analysis/goldens` (written by `cargo run -p turbo-goldens-tool --
//! generate`, see analysis/goldens-tool/README.md) is what one of
//! OpenNLP's Floresta streams read from one input of the `ad` set. Each
//! test reads the same input with the reader here and rebuilds every
//! layer of the golden from its samples: the samples as written, the
//! errors and where they came, the tokens, tags, chunk tags, names,
//! phrases, and the texts with the byte spans of their sentences or
//! tokens.
//!
//! Beside the comparison, every token and tag is checked against the
//! input: a field that borrows the input is the bytes its span names, and
//! one with a text of its own (a joined contraction, a tag with features,
//! a chunk tag) lies inside its sample's source.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_formats::ad::{AD_FORMAT, AdChunkReader, AdChunks, AdNameReader, AdPosReader, AdSentenceReader};
use turbo_formats::convert::ToTokenSamples;
use turbo_formats::registry::{ChunkSamples, Registry};
use turbo_formats::{Field, FormatError, ReadOptions};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary};

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;
const SAMPLE_TOKENS: &str = "turbo:sample-tokens";

fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const TAG: &[ColumnSpec] = &[ColumnSpec::new("tag", STR)];
    const CHUNK: &[ColumnSpec] = &[ColumnSpec::new("chunk", STR)];
    const SPANS: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("type", STR),
    ];
    const PHRASE_ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TEXT: &[ColumnSpec] = &[ColumnSpec::new("text", STR)];
    const BYTES: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("start", U32), ColumnSpec::new("end", U32)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    let aligned =
        |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Aligned(SAMPLE_TOKENS.into()), columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc(SAMPLE_TOKENS, TOKENS),
        aligned("turbo:sample-tags", TAG),
        aligned("turbo:sample-chunks", CHUNK),
        doc("turbo:names", SPANS),
        doc("turbo:phrases", SPANS),
        doc("turbo:phrase-errors", PHRASE_ERRORS),
        doc("turbo:sample-texts", TEXT),
        doc("turbo:sample-spans", BYTES),
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
/// samples before each, as the generator records them, and the samples
/// as written.
fn drain<S: Display>(items: impl Iterator<Item = Result<S, FormatError>>, out: &mut Rebuilt) -> Vec<S> {
    let mut samples = Vec::new();
    out.layer("turbo:read-errors");
    for item in items {
        match item {
            Ok(s) => samples.push(s),
            Err(e) => {
                let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class")).to_owned();
                out.row("turbo:read-errors", vec![samples.len().to_string(), class, e.to_string()]);
            }
        }
    }
    out.layer("turbo:samples");
    for s in &samples {
        out.row("turbo:samples", vec![s.to_string()]);
    }
    samples
}

/// Checks a field against the input: its bytes when it borrows them,
/// else that its span lies inside the sample's source.
fn check(input: &str, source: Span, fields: &[Field<'_>]) {
    for f in fields {
        if f.is_borrowed() {
            assert_eq!(&input[f.span.range()], f.as_str(), "a field's span names other bytes");
        }
        assert!(source.start <= f.span.start && f.span.end <= source.end, "{f:?} outside its sample {source:?}");
    }
}

fn tokens<'s, 'a: 's>(per_sample: impl Iterator<Item = &'s [Field<'a>]>, out: &mut Rebuilt) {
    out.layer(SAMPLE_TOKENS);
    for (i, fields) in per_sample.enumerate() {
        for f in fields {
            out.row(SAMPLE_TOKENS, vec![i.to_string(), f.as_str().to_owned()]);
        }
    }
}

fn column<'s, 'a: 's>(id: &'static str, per_sample: impl Iterator<Item = &'s [Field<'a>]>, out: &mut Rebuilt) {
    out.layer(id);
    for fields in per_sample {
        for f in fields {
            out.row(id, vec![f.as_str().to_owned()]);
        }
    }
}

fn spans(id: &'static str, sample: usize, start: u32, end: u32, kind: &str, out: &mut Rebuilt) {
    out.row(id, vec![sample.to_string(), start.to_string(), end.to_string(), kind.to_owned()]);
}

fn texts<'s>(per_sample: impl Iterator<Item = (&'s str, &'s [Span])>, out: &mut Rebuilt) {
    out.layer("turbo:sample-texts");
    out.layer("turbo:sample-spans");
    for (i, (text, spans)) in per_sample.enumerate() {
        out.row("turbo:sample-texts", vec![text.to_owned()]);
        for s in spans {
            out.row("turbo:sample-spans", vec![i.to_string(), s.start.to_string(), s.end.to_string()]);
        }
    }
}

fn rebuild(run: &str, input: &str, dictionary: &DetokenizerDictionary) -> Rebuilt {
    let mut out = Rebuilt::default();
    match run {
        "names" | "names-nosplit" => {
            let samples = drain(AdNameReader::new(input, run == "names").unwrap(), &mut out);
            for s in &samples {
                check(input, s.source(), s.tokens());
            }
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            out.layer("turbo:names");
            for (i, s) in samples.iter().enumerate() {
                for n in s.names() {
                    spans("turbo:names", i, n.start, n.end, n.kind.unwrap(), &mut out);
                }
            }
        }
        "pos" | "pos-expand-features" => {
            let expand = run == "pos-expand-features";
            let samples = drain(AdPosReader::new(input, expand, expand).unwrap(), &mut out);
            for s in &samples {
                check(input, s.source(), s.tokens());
                check(input, s.source(), s.tags());
            }
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
        }
        "chunk" | "chunk-1-3" => {
            let (start, end) = if run == "chunk" { (None, None) } else { (Some(1), Some(3)) };
            let samples = drain(AdChunkReader::new(input, start, end).unwrap(), &mut out);
            for s in &samples {
                check(input, s.source(), s.tokens());
                check(input, s.source(), s.tags());
                check(input, s.source(), s.chunks());
            }
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
            column("turbo:sample-chunks", samples.iter().map(|s| s.chunks()), &mut out);
            out.layer("turbo:phrases");
            out.layer("turbo:phrase-errors");
            for (i, s) in samples.iter().enumerate() {
                for p in s.phrases().unwrap() {
                    spans("turbo:phrases", i, p.start, p.end, p.kind.as_str(), &mut out);
                }
            }
        }
        "sentence" | "sentence-titles" => {
            let samples = drain(AdSentenceReader::new(input, run == "sentence-titles").unwrap(), &mut out);
            for s in &samples {
                for &source in s.sources() {
                    assert!(s.source().start <= source.start && source.end <= s.source().end);
                }
            }
            texts(samples.iter().map(|s| (s.document(), s.sentences())), &mut out);
        }
        "token" => {
            let names = AdNameReader::new(input, true).unwrap();
            let samples = drain(ToTokenSamples::new(names, Detokenizer::new(dictionary)), &mut out);
            for s in &samples {
                for &source in s.sources() {
                    assert!(s.source().start <= source.start && source.end <= s.source().end);
                }
            }
            texts(samples.iter().map(|s| (s.text(), s.tokens())), &mut out);
        }
        _ => panic!("no reader for the ad run {run}"),
    }
    out
}

fn goldens(root: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for run in std::fs::read_dir(root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("ad")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

fn dictionary(data: &Path) -> DetokenizerDictionary {
    DetokenizerDictionary::read(&std::fs::read(data.join("fixtures/pt-detokenizer.xml")).unwrap()).unwrap()
}

#[test]
fn every_ad_golden_matches() {
    let data = turbo_goldens::data_dir();
    let root = data.join("ad");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; make the goldens with `{GENERATE}`", root.display());
        return;
    }
    let dictionary = dictionary(&data);
    let shapes = shapes();
    let all = goldens(&root);
    assert!(all.len() >= 97, "only {} ad goldens", all.len());
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let mut rebuilt = rebuild(&run, &input, &dictionary).0;
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

/// The inputs the sentence runs leave out read the same through the
/// reader here, which ends where Java's stream throws on every read.
#[test]
fn a_missing_first_text_ends_the_sentence_reader() {
    let input =
        "<s>\n&&\nUTT:n(\"nada\")\tnada\n</s>\n<s>\nSOURCE: a\n8 Depois.\nUTT:adv(\"depois\")\tDepois\n.\n</s>\n";
    let items: Vec<_> = AdSentenceReader::new(input, false).unwrap().collect();
    let errors: Vec<String> = items.iter().map(|i| i.as_ref().unwrap_err().to_string()).collect();
    assert_eq!(
        errors,
        [
            "Cannot invoke \"String.length()\" because \"meta\" is null",
            "Cannot invoke \"String.trim()\" because \"text\" is null"
        ]
    );
}

/// The registry opens the chunk reader with its range.
#[test]
fn the_registry_reads_a_range_of_chunks() {
    let input =
        "<s>\nSOURCE: a\n1 Um.\nUTT:n(\"um\")\tUm\n</s>\n<s>\nSOURCE: b\n1 Dois.\nUTT:n(\"dois\")\tDois\n</s>\n";
    let mut r = Registry::<ChunkSamples>::with_native();
    r.register(AdChunks { start: Some(1), end: None }).unwrap();
    let samples: Vec<String> =
        r.open(AD_FORMAT, input, ReadOptions::default()).unwrap().map(|s| s.unwrap().to_string()).collect();
    assert_eq!(samples, ["Dois n B-NP\n"]);
}
