//! The readers against the JVM: every `formats` golden of
//! `analysis/goldens` (written by `turbo-goldens generate`, see
//! analysis/goldens-tool/README.md) is what one of OpenNLP's sample
//! streams read from one input of the `formats` set. Each test reads the
//! same input with the reader here and rebuilds every layer of the golden
//! from its samples: the samples as written, the errors and where they
//! came, the tokens, tags, lemmas, chunk tags, names, phrases, sample
//! texts and spans, and labels.
//!
//! Beside the comparison, every token, tag and lemma is checked to be the
//! bytes of the input its span names, and every sample of a native format
//! is written and read back to the same written form.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_formats::conllu::{ConlluLemmaReader, ConlluPosReader, ConlluSentenceReader, ConlluTokenReader, Tagset};
use turbo_formats::convert::{ChunkSize, ToSentenceSamples, ToTokenSamples};
use turbo_formats::native::{
    ChunkReader, DocumentReader, LanguageReader, LemmaReader, NameReader, SentenceReader, TokenReader, WordTagReader,
};
use turbo_formats::{Field, FormatError, SentenceSample, TokenSample};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary};
use turbo_unicode::WhitespaceMode;

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;
const SAMPLE_TOKENS: &str = "turbo:sample-tokens";

fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const TAG: &[ColumnSpec] = &[ColumnSpec::new("tag", STR)];
    const LEMMA: &[ColumnSpec] = &[ColumnSpec::new("lemma", STR)];
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
    const LABEL: &[ColumnSpec] = &[ColumnSpec::new("label", STR)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    let aligned =
        |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Aligned(SAMPLE_TOKENS.into()), columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc(SAMPLE_TOKENS, TOKENS),
        aligned("turbo:sample-tags", TAG),
        aligned("turbo:sample-lemmas", LEMMA),
        aligned("turbo:sample-chunks", CHUNK),
        doc("turbo:names", SPANS),
        doc("turbo:phrases", SPANS),
        doc("turbo:phrase-errors", PHRASE_ERRORS),
        doc("turbo:sample-texts", TEXT),
        doc("turbo:sample-spans", BYTES),
        doc("turbo:sample-labels", LABEL),
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
    samples
}

fn written<S: Display>(samples: &[S], out: &mut Rebuilt) {
    out.layer("turbo:samples");
    for s in samples {
        out.row("turbo:samples", vec![s.to_string()]);
    }
}

/// A CoNLL-U column: each value is a field of the input, or the values
/// of a multiword token's words joined with `+`, each of them inside the
/// joined field's span.
fn check_joined(input: &str, fields: &[Field<'_>]) {
    for f in fields {
        let bytes = &input[f.span.range()];
        if f.is_borrowed() {
            assert_eq!(bytes, f.as_str(), "a field's span names other bytes");
        } else if f.as_str().is_empty() {
            assert!(f.span.is_empty());
        } else {
            assert!(f.as_str().split('+').all(|part| bytes.contains(part)), "{f:?} is not joined from {bytes:?}");
        }
    }
}

fn check_fields<'f, 'a: 'f>(input: &str, fields: impl IntoIterator<Item = &'f Field<'a>>) {
    for f in fields {
        assert!(f.is_borrowed() && &input[f.span.range()] == f.as_str(), "a field's span names other bytes");
    }
}

fn tokens<'a>(per_sample: impl Iterator<Item = &'a [Field<'a>]>, out: &mut Rebuilt) {
    out.layer(SAMPLE_TOKENS);
    for (i, fields) in per_sample.enumerate() {
        for f in fields {
            out.row(SAMPLE_TOKENS, vec![i.to_string(), f.as_str().to_owned()]);
        }
    }
}

fn column<'a>(id: &'static str, per_sample: impl Iterator<Item = &'a [Field<'a>]>, out: &mut Rebuilt) {
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

/// Writes one sample and reads it back with `read`, which must give one
/// sample written the same, or none when `vanishes`.
fn round_trip<S: Display>(sample: &S, expect: Back, read: impl Fn(&str) -> Vec<Result<String, FormatError>>) {
    let mut text = String::new();
    turbo_formats::write_samples([sample], &mut text).unwrap();
    let back = read(&text);
    match expect {
        Back::Same => assert_eq!(back, [Ok(sample.to_string())], "{text:?} read back otherwise"),
        Back::Nothing => assert!(back.is_empty(), "{text:?} read back as {back:?}"),
        Back::Error => assert!(matches!(back.as_slice(), [Err(_)]), "{text:?} read back as {back:?}"),
    }
}

/// What a sample written and read back gives.
#[derive(Copy, Clone, Debug)]
enum Back {
    /// The same sample.
    Same,
    /// Nothing: a name sample with no tokens and no names is written as
    /// an empty line.
    Nothing,
    /// An error: a name sample with an empty name after its last token is
    /// written without that name's start tag, as `NameSample.toString`
    /// writes start tags only before tokens.
    Error,
}

/// The written forms of what a reader read.
fn written_of<S: Display>(items: impl Iterator<Item = Result<S, FormatError>>) -> Vec<Result<String, FormatError>> {
    items.map(|s| s.map(|s| s.to_string())).collect()
}

fn token_texts(samples: &[TokenSample<'_>], out: &mut Rebuilt) {
    texts(samples.iter().map(|s| (s.text(), s.tokens())), out);
}

fn sentence_texts(samples: &[SentenceSample<'_>], out: &mut Rebuilt) {
    texts(samples.iter().map(|s| (s.document(), s.sentences())), out);
}

fn rebuild(run: &str, input: &str, dictionary: &DetokenizerDictionary) -> Rebuilt {
    let (kind, mode) = match run.strip_suffix("+legacy") {
        Some(k) => (k, WhitespaceMode::Legacy),
        None => (run, WhitespaceMode::Unicode),
    };
    let mut out = Rebuilt::default();
    match kind {
        "name" => {
            let samples = drain(NameReader::new(input, mode).unwrap(), &mut out);
            written(&samples, &mut out);
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            out.layer("turbo:names");
            for (i, s) in samples.iter().enumerate() {
                check_fields(input, s.tokens());
                for n in s.names() {
                    spans("turbo:names", i, n.start, n.end, n.kind.unwrap(), &mut out);
                    assert!(input[n.source.range()].starts_with("<START"));
                    assert!(input[n.source.range()].ends_with("<END>"));
                }
                let n = s.tokens().len() as u32;
                let back = if s.names().iter().any(|m| m.start == n) {
                    Back::Error
                } else if n == 0 && s.names().is_empty() {
                    Back::Nothing
                } else {
                    Back::Same
                };
                round_trip(s, back, |t| written_of(NameReader::new(t, mode).unwrap()));
            }
        }
        "wordtag" => {
            let samples = drain(WordTagReader::new(input, mode).unwrap(), &mut out);
            written(&samples, &mut out);
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
            for s in &samples {
                check_fields(input, s.tokens());
                check_fields(input, s.tags());
                round_trip(s, Back::Same, |t| written_of(WordTagReader::new(t, mode).unwrap()));
            }
        }
        "lemma" => {
            let samples = drain(LemmaReader::new(input).unwrap(), &mut out);
            written(&samples, &mut out);
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
            column("turbo:sample-lemmas", samples.iter().map(|s| s.lemmas()), &mut out);
            for s in &samples {
                check_fields(input, s.tokens());
                check_fields(input, s.tags());
                check_fields(input, s.lemmas());
                round_trip(s, Back::Same, |t| written_of(LemmaReader::new(t).unwrap()));
            }
        }
        "chunk" => {
            let samples = drain(ChunkReader::new(input).unwrap(), &mut out);
            written(&samples, &mut out);
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
            column("turbo:sample-chunks", samples.iter().map(|s| s.chunks()), &mut out);
            out.layer("turbo:phrases");
            out.layer("turbo:phrase-errors");
            for (i, s) in samples.iter().enumerate() {
                check_fields(input, s.tokens());
                check_fields(input, s.tags());
                check_fields(input, s.chunks());
                match s.phrases() {
                    Ok(phrases) => {
                        for p in phrases {
                            check_fields(input, [&p.kind]);
                            spans("turbo:phrases", i, p.start, p.end, p.kind.as_str(), &mut out);
                        }
                    }
                    Err(e) => out.row(
                        "turbo:phrase-errors",
                        vec![i.to_string(), e.java_class().unwrap().to_owned(), e.to_string()],
                    ),
                }
                round_trip(s, Back::Same, |t| written_of(ChunkReader::new(t).unwrap()));
            }
        }
        "token" => {
            let samples = drain(TokenReader::new(input, mode).unwrap(), &mut out);
            written(&samples, &mut out);
            token_texts(&samples, &mut out);
            for s in &samples {
                for (i, src) in s.sources().iter().enumerate() {
                    assert_eq!(&input[src.range()], s.token(i));
                }
                round_trip(s, Back::Same, |t| written_of(TokenReader::new(t, mode).unwrap()));
            }
        }
        "sentence" => {
            let samples = drain(SentenceReader::new(input).unwrap(), &mut out);
            written(&samples, &mut out);
            sentence_texts(&samples, &mut out);
            for s in &samples {
                for (i, src) in s.sources().iter().enumerate() {
                    let unescaped = input[src.range()].replace("<LF>", "\n").replace("<CR>", "\r");
                    assert_eq!(unescaped, s.sentence(i));
                }
                round_trip(s, Back::Same, |t| written_of(SentenceReader::new(t).unwrap()));
            }
        }
        "doccat" => {
            let samples = drain(DocumentReader::new(input, mode).unwrap(), &mut out);
            written(&samples, &mut out);
            out.layer("turbo:sample-labels");
            for s in &samples {
                out.row("turbo:sample-labels", vec![s.category.as_str().to_owned()]);
                check_fields(input, [&s.category]);
                check_fields(input, &s.tokens);
                round_trip(s, Back::Same, |t| written_of(DocumentReader::new(t, mode).unwrap()));
            }
            tokens(samples.iter().map(|s| s.tokens.as_slice()), &mut out);
        }
        "language" => {
            let samples = drain(LanguageReader::new(input).unwrap(), &mut out);
            written(&samples, &mut out);
            out.layer("turbo:sample-labels");
            out.layer("turbo:sample-texts");
            for s in &samples {
                out.row("turbo:sample-labels", vec![s.language.as_str().to_owned()]);
                out.row("turbo:sample-texts", vec![s.context.as_str().to_owned()]);
                check_fields(input, [&s.language, &s.context]);
                round_trip(s, Back::Same, |t| written_of(LanguageReader::new(t).unwrap()));
            }
        }
        "name-to-token" | "pos-to-token" | "name-to-sentence" | "pos-to-sentence" => {
            let detokenizer = Detokenizer::new(dictionary);
            let chunk = ChunkSize::Samples(std::num::NonZeroUsize::new(2).unwrap());
            let names = || NameReader::new(input, mode).unwrap();
            let pos = || WordTagReader::new(input, mode).unwrap();
            if kind.ends_with("token") {
                let samples = if kind == "name-to-token" {
                    drain(ToTokenSamples::new(names(), detokenizer), &mut out)
                } else {
                    drain(ToTokenSamples::new(pos(), detokenizer), &mut out)
                };
                written(&samples, &mut out);
                token_texts(&samples, &mut out);
                for s in &samples {
                    for (i, src) in s.sources().iter().enumerate() {
                        assert_eq!(&input[src.range()], s.token(i));
                    }
                }
            } else {
                let samples = if kind == "name-to-sentence" {
                    drain(ToSentenceSamples::new(names(), detokenizer, chunk), &mut out)
                } else {
                    drain(ToSentenceSamples::new(pos(), detokenizer, chunk), &mut out)
                };
                written(&samples, &mut out);
                sentence_texts(&samples, &mut out);
            }
        }
        "conllu-pos-u" | "conllu-pos-x" => {
            let tagset = if kind.ends_with('u') { Tagset::U } else { Tagset::X };
            let samples = drain(ConlluPosReader::new(input, tagset).unwrap(), &mut out);
            written(&samples, &mut out);
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
            for s in &samples {
                check_fields(input, s.tokens());
                check_joined(input, s.tags());
            }
        }
        "conllu-lemma-u" | "conllu-lemma-x" => {
            let tagset = if kind.ends_with('u') { Tagset::U } else { Tagset::X };
            let samples = drain(ConlluLemmaReader::new(input, tagset).unwrap(), &mut out);
            written(&samples, &mut out);
            tokens(samples.iter().map(|s| s.tokens()), &mut out);
            column("turbo:sample-tags", samples.iter().map(|s| s.tags()), &mut out);
            column("turbo:sample-lemmas", samples.iter().map(|s| s.lemmas()), &mut out);
            for s in &samples {
                check_fields(input, s.tokens());
                check_joined(input, s.tags());
                check_joined(input, s.lemmas());
            }
        }
        "conllu-sentence" => {
            let samples = drain(ConlluSentenceReader::new(input, 2).unwrap(), &mut out);
            written(&samples, &mut out);
            sentence_texts(&samples, &mut out);
            for s in &samples {
                for (i, src) in s.sources().iter().enumerate() {
                    // A sentence without a text comment is `null`, with the
                    // sentence's lines as its source.
                    assert!(s.sentence(i) == "null" || input[src.range()] == *s.sentence(i));
                }
            }
        }
        "conllu-token" => {
            let samples = drain(ConlluTokenReader::new(input, mode).unwrap(), &mut out);
            written(&samples, &mut out);
            token_texts(&samples, &mut out);
            for s in &samples {
                for (i, src) in s.sources().iter().enumerate() {
                    let token = s.token(i);
                    let bytes = &input[src.range()];
                    assert!(token.contains(bytes), "{token:?} does not hold its source {bytes:?}");
                }
            }
        }
        other => panic!("no reader for the run {other}"),
    }
    out
}

fn goldens() -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join("formats");
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("formats")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

fn dictionary(data: &Path) -> DetokenizerDictionary {
    DetokenizerDictionary::read(&std::fs::read(data.join("fixtures/en-detokenizer.xml")).unwrap()).unwrap()
}

#[test]
fn every_format_golden_matches() {
    let data = turbo_goldens::data_dir();
    let dictionary = dictionary(&data);
    let shapes = shapes();
    let all = goldens();
    assert!(all.len() >= 53, "only {} formats goldens", all.len());
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
