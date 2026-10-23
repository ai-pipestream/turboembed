//! What the OntoNotes and French Treebank golden tests share: the layer
//! shapes, the rebuilt layers and their comparison with a golden, and the
//! layers `ParserGolden` writes for parses.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_parser::{Parse, ParserChunkSample, ParserPosSample};

/// The command that writes the goldens.
pub const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;
const SAMPLE_TOKENS: &str = "turbo:sample-tokens";

pub fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TEXTS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const NODES: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("depth", U32),
        ColumnSpec::new("type", STR),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("head-start", U32),
        ColumnSpec::new("head-end", U32),
        ColumnSpec::new("own-head", U32),
        ColumnSpec::new("head-index", U32),
    ];
    const POS: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("token", STR), ColumnSpec::new("tag", STR)];
    const CHUNKS: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("token", STR),
        ColumnSpec::new("tag", STR),
        ColumnSpec::new("pred", STR),
    ];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const TAG: &[ColumnSpec] = &[ColumnSpec::new("tag", STR)];
    const SPANS: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("type", STR),
    ];
    const LINES: &[ColumnSpec] = &[ColumnSpec::new("line", STR)];
    const FILES: &[ColumnSpec] = &[ColumnSpec::new("file", STR)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    let aligned =
        |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Aligned(SAMPLE_TOKENS.into()), columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc("turbo:parse-texts", TEXTS),
        doc("turbo:parse-nodes", NODES),
        doc("turbo:pos-samples", POS),
        doc("turbo:chunk-samples", CHUNKS),
        doc(SAMPLE_TOKENS, TOKENS),
        aligned("turbo:sample-tags", TAG),
        doc("turbo:names", SPANS),
        doc("turbo:lines", LINES),
        doc("turbo:listing", FILES),
    ]
}

/// The layers rebuilt from the Rust readers, by id, each as rows of cells
/// written as text. A cell of `None` is not compared.
#[derive(Default)]
pub struct Rebuilt(pub BTreeMap<&'static str, Vec<Vec<Option<String>>>>);

impl Rebuilt {
    /// Declares a layer that may have no rows.
    pub fn layer(&mut self, id: &'static str) {
        self.0.entry(id).or_default();
    }

    pub fn row(&mut self, id: &'static str, cells: Vec<Option<String>>) {
        self.0.entry(id).or_default().push(cells);
    }

    /// An error and the number of samples before it. `message` is `None`
    /// where the message is not this crate's to reproduce.
    pub fn error(&mut self, after: usize, class: Option<&str>, message: Option<String>) {
        let class = class.unwrap_or_else(|| panic!("an error without a Java class: {message:?}"));
        self.row("turbo:read-errors", vec![s(after), s(class), message]);
    }

    /// The files of the corpus, relative to `root`.
    pub fn listing(&mut self, root: &Path, files: &[PathBuf]) {
        self.layer("turbo:listing");
        for f in files {
            self.row(
                "turbo:listing",
                vec![s(f.strip_prefix(root).unwrap().to_str().unwrap().replace(std::path::MAIN_SEPARATOR, "/"))],
            );
        }
    }

    /// The layers `ParserGolden` writes for `parses`.
    pub fn parses(&mut self, parses: &[&Parse]) {
        for id in [
            "turbo:samples",
            "turbo:read-errors",
            "turbo:parse-texts",
            "turbo:parse-nodes",
            "turbo:pos-samples",
            "turbo:chunk-samples",
        ] {
            self.layer(id);
        }
        for (i, p) in parses.iter().enumerate() {
            self.row("turbo:samples", vec![s(p.to_string_penn_treebank())]);
            self.row("turbo:parse-texts", vec![s(i), s(p.text())]);
            self.nodes(p, 0, i);
            let pos = ParserPosSample::from_parse(p);
            for (token, tag) in pos.sentence.iter().zip(&pos.tags) {
                self.row("turbo:pos-samples", vec![s(i), s(token), s(tag)]);
            }
            let chunk = ParserChunkSample::from_parse(p);
            for ((token, tag), pred) in chunk.sentence.iter().zip(&chunk.tags).zip(&chunk.preds) {
                self.row("turbo:chunk-samples", vec![s(i), s(token), s(tag), s(pred)]);
            }
        }
    }

    fn nodes(&mut self, node: &Parse, depth: usize, sample: usize) {
        let span = node.span();
        let head = node.head();
        self.row(
            "turbo:parse-nodes",
            vec![
                s(sample),
                s(depth),
                s(node.kind()),
                s(span.start),
                s(span.end),
                s(head.span.start),
                s(head.span.end),
                s(u32::from(node.is_own_head())),
                s(node.head_index()),
            ],
        );
        for child in node.children() {
            self.nodes(child, depth + 1, sample);
        }
    }
}

pub fn s(v: impl ToString) -> Option<String> {
    Some(v.to_string())
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

/// The goldens of `annotator`, as (run, path), or `None` with a note when
/// they have not been generated.
pub fn goldens(annotator: &str) -> Option<Vec<(String, PathBuf)>> {
    let root = turbo_goldens::data_dir().join(annotator);
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; make the goldens with `{GENERATE}`", root.display());
        return None;
    }
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join(annotator)).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    Some(out)
}

/// Compares every layer of `golden` with the rebuilt one.
pub fn compare(path: &Path, golden: &Golden, mut rebuilt: Rebuilt) {
    assert!(golden.header.error.is_none(), "{}: an error record", path.display());
    for layer in &golden.layers {
        let ours = rebuilt.0.remove(layer.id.as_str()).unwrap_or_else(|| panic!("{}: no {}", path.display(), layer.id));
        let theirs = cells(layer);
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(a.len(), b.len(), "{}: {} row {i}", path.display(), layer.id);
            for (ca, cb) in a.iter().zip(b) {
                if let Some(ca) = ca {
                    assert_eq!(ca, cb, "{}: {} row {i}: {a:?} against {b:?}", path.display(), layer.id);
                }
            }
        }
        assert_eq!(ours.len(), theirs.len(), "{}: {} rows", path.display(), layer.id);
    }
    assert!(rebuilt.0.is_empty(), "{}: layers the golden lacks: {:?}", path.display(), rebuilt.0.keys());
}

/// A fresh scratch directory for one golden's corpus.
pub fn scratch(name: &str, golden: &Path) -> PathBuf {
    let stem = golden.file_stem().unwrap().to_str().unwrap();
    let run = golden.parent().unwrap().parent().unwrap().file_name().unwrap().to_str().unwrap();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name).join(run).join(stem);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
