//! The MASC readers against the JVM: every `masc` golden of
//! `analysis/goldens` (written by `turbo-goldens generate`, see
//! analysis/goldens-tool/README.md) is what OpenNLP's MASC classes read
//! from one bundle of the `formats` set, a small corpus directory written
//! as one text (the format is described in `MascGolden.java`). Each test
//! writes the bundle out as the generator does, reads it with the readers
//! here and rebuilds every layer of the golden: what each document holds
//! and what parsing it threw, the documents `MascDocumentStream` keeps,
//! and the samples, errors, tokens, tags, names, texts and spans of each
//! sample stream over two passes with a `reset` between them.
//!
//! Beside the comparison, every token, tag and sentence is checked to be
//! the bytes of the document its span names.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use turbo_formats::masc::{
    MascCorpus, MascError, MascNameReader, MascPosReader, MascSentenceReader, MascTokenReader, read_document,
};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;
const SAMPLE_TOKENS: &str = "turbo:sample-tokens";

/// The sentences per sample of the sentence stream (`MascGolden.SENTENCES_PER_SAMPLE`).
const SENTENCES_PER_SAMPLE: usize = 2;

/// The reads after which a stream is taken not to end (`MascGolden.MAX_READS`).
const MAX_READS: usize = 100_000;

fn shapes() -> Vec<Shape> {
    const DOCUMENTS: &[ColumnSpec] = &[
        ColumnSpec::new("path", STR),
        ColumnSpec::new("class", STR),
        ColumnSpec::new("message", STR),
        ColumnSpec::new("abort", U32),
        ColumnSpec::new("penn", U32),
        ColumnSpec::new("ne", U32),
    ];
    const SENTENCES: &[ColumnSpec] = &[
        ColumnSpec::new("doc", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("quarks", STR),
        ColumnSpec::new("tokenized", U32),
    ];
    const TOKENS: &[ColumnSpec] = &[
        ColumnSpec::new("sentence", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("id", U32),
        ColumnSpec::new("pos", STR),
        ColumnSpec::new("base", STR),
        ColumnSpec::new("nulls", U32),
        ColumnSpec::new("quarks", STR),
    ];
    const NAMES: &[ColumnSpec] = &[
        ColumnSpec::new("sentence", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("type", STR),
        ColumnSpec::new("nulls", U32),
    ];
    const PATH: &[ColumnSpec] = &[ColumnSpec::new("path", STR)];
    const FAILURE: &[ColumnSpec] = &[ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("pass", U32), ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] = &[
        ColumnSpec::new("pass", U32),
        ColumnSpec::new("after", U32),
        ColumnSpec::new("class", STR),
        ColumnSpec::new("message", STR),
    ];
    const SAMPLE_TEXT: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const TAG: &[ColumnSpec] = &[ColumnSpec::new("tag", STR)];
    const SAMPLE_NAMES: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("type", STR),
        ColumnSpec::new("nulls", U32),
    ];
    const TEXT: &[ColumnSpec] = &[ColumnSpec::new("text", STR)];
    const BYTES: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("start", U32), ColumnSpec::new("end", U32)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:masc-documents", DOCUMENTS),
        doc("turbo:masc-sentences", SENTENCES),
        doc("turbo:masc-tokens", TOKENS),
        doc("turbo:masc-names", NAMES),
        doc("turbo:masc-stream", PATH),
        doc("turbo:masc-stream-errors", FAILURE),
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc("turbo:open-errors", FAILURE),
        doc(SAMPLE_TOKENS, SAMPLE_TEXT),
        Shape { id: "turbo:sample-tags".into(), anchor: GoldenAnchor::Aligned(SAMPLE_TOKENS.into()), columns: TAG },
        doc("turbo:names", SAMPLE_NAMES),
        doc("turbo:sample-texts", TEXT),
        doc("turbo:sample-spans", BYTES),
    ]
}

/// The layers rebuilt from the Rust readers, by id, each as rows of cells
/// written as text.
#[derive(Default)]
struct Rebuilt(BTreeMap<&'static str, Vec<Vec<String>>>);

impl Rebuilt {
    fn row(&mut self, id: &'static str, cells: Vec<String>) {
        self.0.entry(id).or_default().push(cells);
    }

    /// Declares layers that may have no rows.
    fn layers(&mut self, ids: &[&'static str]) {
        for id in ids {
            self.0.entry(id).or_default();
        }
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

// ---- the bundle ----

/// Writes the files of a bundle under `root`, as `MascGolden.writeBundle`
/// does, refusing what it refuses.
fn write_bundle(text: &str, root: &Path) {
    let mut pos = 0;
    while pos < text.len() {
        let eol = text[pos..].find('\n').map(|i| pos + i);
        let line = &text[pos..eol.unwrap_or(text.len())];
        let next = eol.map_or(text.len(), |e| e + 1);
        if let Some((hex, path)) = line
            .strip_prefix("@@ file ")
            .map(|p| (false, p))
            .or_else(|| line.strip_prefix("@@ hex ").map(|p| (true, p)))
        {
            let file = resolve(root, path);
            let (end, after) = if text[next..].starts_with("@@ end") {
                (next, next + 6)
            } else {
                let end = next + text[next..].find("\n@@ end").unwrap_or_else(|| panic!("{line} has no @@ end"));
                (end, end + 7)
            };
            assert!(after >= text.len() || text.as_bytes()[after] == b'\n', "text after @@ end");
            let content = &text[next..end];
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            let bytes = if hex { hex_bytes(content) } else { content.as_bytes().to_vec() };
            std::fs::write(&file, bytes).unwrap();
            pos = (after + 1).min(text.len());
        } else if let Some(path) = line.strip_prefix("@@ dir ") {
            std::fs::create_dir_all(resolve(root, path)).unwrap();
            pos = next;
        } else {
            assert!(line.is_empty() || line.starts_with('#'), "a line outside a file: {line}");
            pos = next;
        }
    }
}

fn resolve(root: &Path, relative: &str) -> PathBuf {
    assert!(
        !relative.is_empty() && !relative.starts_with('/') && !relative.split('/').any(|p| p == ".." || p == "."),
        "a path outside the corpus: {relative}"
    );
    root.join(relative)
}

fn hex_bytes(content: &str) -> Vec<u8> {
    let digits: Vec<u8> = content.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert!(digits.len().is_multiple_of(2), "an odd number of hex digits");
    digits.chunks(2).map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap()).collect()
}

/// The `.hdr` files in the order the stream finds them in a bundle's
/// layout: a directory's own, then its one subdirectory's.
fn headers(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut dir = Some(root.to_owned());
    while let Some(d) = dir.take() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().path()).collect();
        entries.sort();
        let (dirs, files): (Vec<PathBuf>, Vec<PathBuf>) = entries.into_iter().partition(|p| p.is_dir());
        assert!(dirs.len() <= 1, "{} has more than one directory", d.display());
        let hdrs: Vec<PathBuf> = files.into_iter().filter(|p| p.extension().is_some_and(|e| e == "hdr")).collect();
        assert!(hdrs.len() <= 1, "{} has more than one .hdr file", d.display());
        out.extend(hdrs);
        dir = dirs.into_iter().next();
    }
    out
}

// ---- the runs ----

/// An error as the generator writes it: its Java class and its message
/// with the corpus directory written `<corpus>`.
fn failure(e: &MascError, root: &Path) -> (String, String) {
    let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class")).to_owned();
    (class, e.to_string().replace(root.to_str().unwrap(), "<corpus>"))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap().to_str().unwrap().to_owned()
}

fn ids(ids: impl Iterator<Item = i32>) -> String {
    ids.map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
}

fn flag(b: bool) -> String {
    u8::from(b).to_string()
}

fn documents(root: &Path, out: &mut Rebuilt) {
    out.layers(&[
        "turbo:masc-documents",
        "turbo:masc-sentences",
        "turbo:masc-tokens",
        "turbo:masc-names",
        "turbo:masc-stream",
        "turbo:masc-stream-errors",
    ]);
    let mut sentence_index = 0usize;
    for (doc_index, hdr) in headers(root).iter().enumerate() {
        let (class, message, abort, doc) = match read_document(hdr) {
            Ok(doc) => (String::new(), String::new(), false, Some(doc)),
            Err(e) => {
                let (class, message) = failure(&e, root);
                (class, message, !e.skips_document(), None)
            }
        };
        let penn = doc.as_ref().is_some_and(|d| d.has_penn_tags());
        let ne = doc.as_ref().is_some_and(|d| d.has_named_entities());
        out.row("turbo:masc-documents", vec![relative(root, hdr), class, message, flag(abort), flag(penn), flag(ne)]);
        let Some(doc) = doc else {
            continue;
        };
        for sentence in doc.sentences() {
            out.row(
                "turbo:masc-sentences",
                vec![
                    doc_index.to_string(),
                    sentence.start.to_string(),
                    sentence.end.to_string(),
                    ids(sentence.quarks().iter().map(|&q| doc.quarks()[q].id)),
                    flag(sentence.tokens().is_some()),
                ],
            );
            for t in sentence.tokens().unwrap_or_default() {
                let nulls = u8::from(t.tag.is_none()) | u8::from(t.base.is_none()) << 1;
                out.row(
                    "turbo:masc-tokens",
                    vec![
                        sentence_index.to_string(),
                        t.start.to_string(),
                        t.end.to_string(),
                        t.id.to_string(),
                        t.tag.clone().unwrap_or_default(),
                        t.base.clone().unwrap_or_default(),
                        nulls.to_string(),
                        ids(t.quarks.iter().map(|&q| doc.quarks()[q].id)),
                    ],
                );
            }
            for n in sentence.names() {
                out.row(
                    "turbo:masc-names",
                    vec![
                        sentence_index.to_string(),
                        n.start.to_string(),
                        n.end.to_string(),
                        n.kind.clone().unwrap_or_default(),
                        flag(n.kind.is_none()),
                    ],
                );
            }
            sentence_index += 1;
        }
    }
    match MascCorpus::read(root) {
        Ok(corpus) => {
            for d in corpus.documents() {
                out.row("turbo:masc-stream", vec![relative(root, d.path())]);
            }
        }
        Err(e) => {
            let (class, message) = failure(&e, root);
            out.row("turbo:masc-stream-errors", vec![class, message]);
        }
    }
}

/// A sample stream's `read` and `reset`, for the passes.
trait Stream {
    type Sample;
    fn read(&mut self) -> Result<Option<Self::Sample>, MascError>;
    fn reset(&mut self);
}

macro_rules! stream {
    ($reader:ident, $sample:ty) => {
        impl<'c> Stream for $reader<'c> {
            type Sample = $sample;
            fn read(&mut self) -> Result<Option<$sample>, MascError> {
                $reader::read(self)
            }
            fn reset(&mut self) {
                $reader::reset(self)
            }
        }
    };
}

stream!(MascPosReader, turbo_formats::masc::MascSample<turbo_formats::PosSample<'c>>);
stream!(MascTokenReader, turbo_formats::masc::MascSample<turbo_formats::TokenSample<'c>>);
stream!(MascNameReader, turbo_formats::masc::MascSample<turbo_formats::NameSample<'c>>);
stream!(MascSentenceReader, turbo_formats::masc::MascSentenceSample<'c>);

/// Reads a stream as the generator does: to the end or a
/// `NullPointerException`, then `reset` and again, recording each error.
fn passes<S: Stream>(
    mut stream: S,
    root: &Path,
    out: &mut Rebuilt,
    show: impl Fn(&S::Sample) -> String,
) -> Vec<S::Sample> {
    let mut samples = Vec::new();
    for pass in 0..2 {
        if pass == 1 {
            stream.reset();
        }
        for reads in 0.. {
            assert!(reads <= MAX_READS, "the stream does not end");
            match stream.read() {
                Ok(Some(s)) => {
                    out.row("turbo:samples", vec![pass.to_string(), show(&s)]);
                    samples.push(s);
                }
                Ok(None) => break,
                Err(e) => {
                    let (class, message) = failure(&e, root);
                    let npe = class == "java.lang.NullPointerException";
                    out.row("turbo:read-errors", vec![pass.to_string(), samples.len().to_string(), class, message]);
                    if npe {
                        break;
                    }
                }
            }
        }
    }
    samples
}

fn written<S: Display>(s: &S) -> String {
    s.to_string()
}

fn texts<'s>(per_sample: impl Iterator<Item = (&'s str, &'s [turbo_analysis::Span])>, out: &mut Rebuilt) {
    out.layers(&["turbo:sample-texts", "turbo:sample-spans"]);
    for (i, (text, spans)) in per_sample.enumerate() {
        out.row("turbo:sample-texts", vec![text.to_owned()]);
        for s in spans {
            out.row("turbo:sample-spans", vec![i.to_string(), s.start.to_string(), s.end.to_string()]);
        }
    }
}

fn samples(run: &str, root: &Path, out: &mut Rebuilt) {
    out.layers(&["turbo:samples", "turbo:read-errors", "turbo:open-errors"]);
    let open_error = |e: MascError, out: &mut Rebuilt| {
        let (class, message) = failure(&e, root);
        out.row("turbo:open-errors", vec![class, message]);
    };
    let corpus = match MascCorpus::read(root) {
        Ok(c) => Some(c),
        Err(e) => {
            open_error(e, out);
            None
        }
    };
    match run {
        "pos" => {
            out.layers(&[SAMPLE_TOKENS, "turbo:sample-tags"]);
            let Some(reader) = corpus.as_ref().map(MascPosReader::new).transpose().unwrap_or_else(|e| {
                open_error(e, out);
                None
            }) else {
                return;
            };
            let samples = passes(reader, root, out, |s| written(&s.sample));
            let corpus = corpus.as_ref().unwrap();
            for (i, s) in samples.iter().enumerate() {
                let text = corpus.documents()[s.document].text();
                for (token, tag) in s.sample.tokens().iter().zip(s.sample.tags()) {
                    assert_eq!(&text[token.span.range()], token.as_str());
                    assert_eq!(tag.span, token.span);
                    out.row(SAMPLE_TOKENS, vec![i.to_string(), token.as_str().to_owned()]);
                    out.row("turbo:sample-tags", vec![tag.as_str().to_owned()]);
                }
            }
        }
        "name" => {
            out.layers(&[SAMPLE_TOKENS, "turbo:names"]);
            let Some(reader) = corpus.as_ref().map(MascNameReader::new).transpose().unwrap_or_else(|e| {
                open_error(e, out);
                None
            }) else {
                return;
            };
            let samples = passes(reader, root, out, |s| written(&s.sample));
            let corpus = corpus.as_ref().unwrap();
            for (i, s) in samples.iter().enumerate() {
                let text = corpus.documents()[s.document].text();
                for token in s.sample.tokens() {
                    assert_eq!(&text[token.span.range()], token.as_str());
                    out.row(SAMPLE_TOKENS, vec![i.to_string(), token.as_str().to_owned()]);
                }
                for n in s.sample.names() {
                    out.row(
                        "turbo:names",
                        vec![
                            i.to_string(),
                            n.start.to_string(),
                            n.end.to_string(),
                            n.kind.unwrap_or_default().to_owned(),
                            flag(n.kind.is_none()),
                        ],
                    );
                }
            }
        }
        "token" => {
            let Some(reader) = corpus.as_ref().map(MascTokenReader::new).transpose().unwrap_or_else(|e| {
                open_error(e, out);
                None
            }) else {
                out.layers(&["turbo:sample-texts", "turbo:sample-spans"]);
                return;
            };
            let samples = passes(reader, root, out, |s| written(&s.sample));
            let corpus = corpus.as_ref().unwrap();
            for s in &samples {
                let text = corpus.documents()[s.document].text();
                assert_eq!(&text[s.sample.source().range()], s.sample.text());
                for (i, src) in s.sample.sources().iter().enumerate() {
                    assert_eq!(&text[src.range()], s.sample.token(i));
                }
            }
            texts(samples.iter().map(|s| (s.sample.text(), s.sample.tokens())), out);
        }
        "sentence" => {
            let Some(corpus) = corpus.as_ref() else {
                out.layers(&["turbo:sample-texts", "turbo:sample-spans"]);
                return;
            };
            let samples =
                passes(MascSentenceReader::new(corpus, SENTENCES_PER_SAMPLE), root, out, |s| written(&s.sample));
            for s in &samples {
                assert_eq!(s.documents.len(), s.sample.sentences().len());
                for (i, (src, &d)) in s.sample.sources().iter().zip(&s.documents).enumerate() {
                    assert_eq!(&corpus.documents()[d].text()[src.range()], s.sample.sentence(i));
                }
            }
            texts(samples.iter().map(|s| (s.sample.document(), s.sample.sentences())), out);
        }
        other => panic!("no reader for the run {other}"),
    }
}

fn goldens() -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join("masc");
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

#[test]
fn every_masc_golden_matches() {
    let data = turbo_goldens::data_dir();
    let shapes = shapes();
    let all = goldens();
    assert!(all.len() >= 40, "only {} masc goldens", all.len());
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("masc-goldens");
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let root = scratch.join(format!("{run}-{}", path.file_stem().unwrap().to_str().unwrap()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        write_bundle(&input, &root);
        let mut rebuilt = Rebuilt::default();
        if run == "documents" {
            documents(&root, &mut rebuilt);
        } else {
            samples(&run, &root, &mut rebuilt);
        }
        let mut rebuilt = rebuilt.0;
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
        std::fs::remove_dir_all(&root).unwrap();
    }
}
