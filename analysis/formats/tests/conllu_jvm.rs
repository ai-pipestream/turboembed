//! The CoNLL-U dependency and CorefUD readers against the JVM: every
//! golden under `analysis/goldens/data/conllu` (written by `turbo-goldens
//! generate`, see analysis/goldens-tool/README.md) is what
//! `ConlluDependencySampleStream` or `ConlluCorefDocumentStream` read from
//! one input of the `conllu` set. Each test reads the same input with the
//! reader here and rebuilds every layer of the golden: the samples as
//! written, the errors and where they came, the tokens, tags, heads and
//! relations; or the documents' texts, sentences, tokens, tags, mentions
//! with their chains, and speakers.
//!
//! Beside the comparison, every token, tag and relation is checked to be
//! the bytes of the input its span names, and every mention's source to
//! run from its first token's form to its last's.

use std::collections::BTreeMap;
use std::path::PathBuf;

use turbo_formats::FormatError;
use turbo_formats::conllu::{ConlluCorefReader, ConlluDependencyReader, Tagset};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_unicode::WhitespaceMode;

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;

fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const TAG: &[ColumnSpec] = &[ColumnSpec::new("tag", STR)];
    const HEAD: &[ColumnSpec] = &[ColumnSpec::new("head", U32)];
    const RELATION: &[ColumnSpec] = &[ColumnSpec::new("relation", STR)];
    const TEXT: &[ColumnSpec] = &[ColumnSpec::new("text", STR)];
    const SPANNED: &[ColumnSpec] = &[
        ColumnSpec::new("document", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("text", STR),
    ];
    const TAGS: &[ColumnSpec] =
        &[ColumnSpec::new("start", U32), ColumnSpec::new("end", U32), ColumnSpec::new("tag", STR)];
    const MENTIONS: &[ColumnSpec] = &[
        ColumnSpec::new("document", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("chain", U32),
        ColumnSpec::new("kind", STR),
        ColumnSpec::new("entity", STR),
    ];
    const SPEAKERS: &[ColumnSpec] = &[
        ColumnSpec::new("document", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("label", STR),
    ];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    let aligned =
        |id: &str, to: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Aligned(to.into()), columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc("turbo:sample-tokens", TOKENS),
        aligned("turbo:sample-tags", "turbo:sample-tokens", TAG),
        aligned("turbo:sample-heads", "turbo:sample-tokens", HEAD),
        aligned("turbo:sample-relations", "turbo:sample-tokens", RELATION),
        doc("turbo:coref-texts", TEXT),
        doc("turbo:coref-sentences", SPANNED),
        doc("turbo:coref-tokens", SPANNED),
        aligned("turbo:coref-tags", "turbo:coref-tokens", TAGS),
        doc("turbo:coref-mentions", MENTIONS),
        doc("turbo:coref-speakers", SPEAKERS),
    ]
}

/// The layers rebuilt from the Rust readers, by id, each as rows of
/// cells written as text.
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

    fn error(&mut self, after: usize, e: &FormatError) {
        let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class"));
        self.row("turbo:read-errors", vec![after.to_string(), class.to_owned(), e.to_string()]);
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

fn tagset(run: &str) -> Tagset {
    if run.contains("-x") { Tagset::X } else { Tagset::U }
}

fn dependencies(run: &str, input: &str) -> Rebuilt {
    let mode = if run.ends_with("+legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
    let mut out = Rebuilt::default();
    out.layers(&[
        "turbo:samples",
        "turbo:read-errors",
        "turbo:sample-tokens",
        "turbo:sample-tags",
        "turbo:sample-heads",
        "turbo:sample-relations",
    ]);
    let mut samples = Vec::new();
    for item in ConlluDependencyReader::new(input, tagset(run), mode).unwrap() {
        match item {
            Ok(s) => samples.push(s),
            Err(e) => out.error(samples.len(), &e),
        }
    }
    for (i, s) in samples.iter().enumerate() {
        out.row("turbo:samples", vec![s.to_string()]);
        for ((token, tag), arc) in s.tokens().iter().zip(s.tags()).zip(s.tree().arcs()) {
            assert_eq!(&input[token.span.range()], token.as_str());
            assert_eq!(&input[tag.span.range()], tag.as_str());
            assert_eq!(&input[arc.relation.span.range()], arc.relation.as_str());
            assert!(s.source().start <= token.span.start && arc.relation.span.end <= s.source().end);
            out.row("turbo:sample-tokens", vec![i.to_string(), token.to_string()]);
            out.row("turbo:sample-tags", vec![tag.to_string()]);
            out.row("turbo:sample-heads", vec![arc.head.column().to_string()]);
            out.row("turbo:sample-relations", vec![arc.relation.to_string()]);
        }
    }
    out
}

fn coref(run: &str, input: &str) -> Rebuilt {
    let mut out = Rebuilt::default();
    out.layers(&[
        "turbo:read-errors",
        "turbo:coref-texts",
        "turbo:coref-sentences",
        "turbo:coref-tokens",
        "turbo:coref-tags",
        "turbo:coref-mentions",
        "turbo:coref-speakers",
    ]);
    let mut documents = Vec::new();
    for item in ConlluCorefReader::new(input, tagset(run)).unwrap() {
        match item {
            Ok(d) => documents.push(d),
            Err(e) => out.error(documents.len(), &e),
        }
    }
    for (d, doc) in documents.iter().enumerate() {
        let text = doc.text();
        let n = d.to_string();
        out.row("turbo:coref-texts", vec![text.to_owned()]);
        for s in doc.sentences() {
            let (start, end) = (s.text.start.to_string(), s.text.end.to_string());
            out.row(
                "turbo:coref-sentences",
                vec![n.clone(), start.clone(), end.clone(), text[s.text.range()].to_owned()],
            );
            if let Some(label) = &s.speaker {
                assert_eq!(&input[label.span.range()], label.as_str());
                out.row("turbo:coref-speakers", vec![n.clone(), start, end, label.to_string()]);
            }
        }
        for t in doc.tokens() {
            assert_eq!(&input[t.form.span.range()], t.form.as_str());
            assert_eq!(&input[t.tag.span.range()], t.tag.as_str());
            assert_eq!(&text[t.text.range()], t.form.as_str());
            let (start, end) = (t.text.start.to_string(), t.text.end.to_string());
            out.row("turbo:coref-tokens", vec![n.clone(), start.clone(), end.clone(), t.form.to_string()]);
            out.row("turbo:coref-tags", vec![start, end, t.tag.to_string()]);
        }
        for m in doc.mentions() {
            let (first, last) = (&doc.tokens()[m.tokens.start as usize], &doc.tokens()[m.tokens.end as usize - 1]);
            assert_eq!(m.source.start, first.form.span.start);
            assert_eq!(m.source.end, last.form.span.end);
            assert_eq!(&input[m.open.entity.span.range()], m.entity().as_str());
            let row = vec![
                n.clone(),
                m.text.start.to_string(),
                m.text.end.to_string(),
                m.chain.to_string(),
                "gold".to_owned(),
                "-1".to_owned(),
            ];
            out.row("turbo:coref-mentions", row);
        }
    }
    out
}

fn goldens() -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join("conllu");
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap_or_else(|e| panic!("{}: {e}", root.display())) {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("conllu")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

#[test]
fn every_conllu_golden_matches() {
    let data = turbo_goldens::data_dir();
    let shapes = shapes();
    let all = goldens();
    let runs: std::collections::BTreeSet<&str> = all.iter().map(|(r, _)| r.as_str()).collect();
    assert_eq!(
        runs.into_iter().collect::<Vec<_>>(),
        ["corefud-u", "corefud-x", "dependency-u", "dependency-u+legacy", "dependency-x"]
    );
    // Three dependency runs over two inputs and two CorefUD runs over three.
    assert_eq!(all.len(), 12, "conllu goldens");
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let mut rebuilt =
            if run.starts_with("dependency") { dependencies(&run, &input) } else { coref(&run, &input) }.0;
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
