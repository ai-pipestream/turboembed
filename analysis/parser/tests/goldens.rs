//! The parse reader and writer and the head rules against the JVM: every `parse` and `head-rules`
//! golden of `analysis/goldens` (written by `turbo-goldens generate`, see
//! analysis/goldens-tool/README.md and `ParserGolden.java`).
//!
//! A `parse` golden is what OpenNLP made of the lines of one input of the `parse` set in one run:
//! the parses written back (`toStringPennTreebank`), the exceptions, each parse's text, every node
//! with its span and head, and the samples `PosSampleStream` and `ChunkSampleStream` make of it.
//! A `head-rules` golden is what a rule reader made of one rule file: the rules written back, or
//! the exception. Each test does the same here and rebuilds every layer. Exception messages are
//! compared only where this crate writes Java's message (`IOException` from the rule readers);
//! the others are the JVM's own texts.
//!
//! The `parse-treebank` runs of `ParserTrainGolden.java`, which share the `parse` directory, are
//! compared in `goldens/treebank.rs`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_parser::{
    AncoraSpanishHeadRules, EnglishHeadRules, GapLabeler, HeadRules, HeadRulesError, Parse, ParseError,
    ParseSampleStream, ParserChunkSample, ParserPosSample,
};

#[path = "goldens/treebank.rs"]
mod treebank;

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;

fn shapes() -> Vec<Shape> {
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
    const LINES: &[ColumnSpec] = &[ColumnSpec::new("line", STR)];
    const RULE_ERRORS: &[ColumnSpec] = &[ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc("turbo:parse-texts", TEXTS),
        doc("turbo:parse-nodes", NODES),
        doc("turbo:pos-samples", POS),
        doc("turbo:chunk-samples", CHUNKS),
        doc("turbo:head-rules", LINES),
        doc("turbo:rules-errors", RULE_ERRORS),
    ]
}

/// The layers rebuilt here, by id, each as rows of cells written as text. A cell of `None` is
/// not compared.
#[derive(Default)]
struct Rebuilt(BTreeMap<&'static str, Vec<Vec<Option<String>>>>);

impl Rebuilt {
    fn layer(&mut self, id: &'static str) {
        self.0.entry(id).or_default();
    }

    fn row(&mut self, id: &'static str, cells: Vec<Option<String>>) {
        self.0.entry(id).or_default().push(cells);
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

fn s(v: impl ToString) -> Option<String> {
    Some(v.to_string())
}

/// What a `parse` run does to each parse after reading it.
enum Steps {
    /// Nothing: the `treebank` runs.
    None,
    /// `updateHeads(rules)`.
    Heads(Box<dyn Rules>),
    /// `pruneParse`, `fixPossesives`, then `updateHeads(rules)`.
    PossessivesAndHeads(Box<dyn Rules>),
}

/// Head rules that are also the gap labeler, as both OpenNLP classes are.
trait Rules: HeadRules + GapLabeler {
    fn labeler(&self) -> &dyn GapLabeler;
    fn rules(&self) -> &dyn HeadRules;
}

impl<T: HeadRules + GapLabeler> Rules for T {
    fn labeler(&self) -> &dyn GapLabeler {
        self
    }
    fn rules(&self) -> &dyn HeadRules {
        self
    }
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(turbo_goldens::data_dir().join("fixtures").join(name)).unwrap()
}

fn run(name: &str) -> (bool, Steps) {
    let english = || Box::new(EnglishHeadRules::from_str(&fixture("en-head-rules.txt")).unwrap()) as Box<dyn Rules>;
    let spanish = |f: &str| Box::new(AncoraSpanishHeadRules::from_str(&fixture(f)).unwrap()) as Box<dyn Rules>;
    match name {
        "treebank" => (false, Steps::None),
        "treebank+function-tags" => (true, Steps::None),
        "en-head-rules" => (false, Steps::Heads(english())),
        "en-head-rules+possessives" => (false, Steps::PossessivesAndHeads(english())),
        "es-head-rules" => (false, Steps::Heads(spanish("es-head-rules.txt"))),
        "es-regex-head-rules" => (false, Steps::Heads(spanish("es-regex-head-rules.txt"))),
        _ => panic!("unknown parse run {name}"),
    }
}

fn read_error(out: &mut Rebuilt, after: usize, e: &ParseError) {
    let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class"));
    out.row("turbo:read-errors", vec![s(after), s(class), None]);
}

fn rebuild_parses(run_name: &str, input: &str) -> Rebuilt {
    let (function_tags, steps) = run(run_name);
    let labeler = match &steps {
        Steps::None => None,
        Steps::Heads(r) | Steps::PossessivesAndHeads(r) => Some(r.labeler()),
    };
    let mut out = Rebuilt::default();
    out.layer("turbo:read-errors");
    let mut parses = Vec::new();
    for item in ParseSampleStream::with_options(input, function_tags, labeler) {
        let done = item.and_then(|mut p| {
            match &steps {
                Steps::None => {}
                Steps::Heads(r) => p.update_heads(r.rules()),
                Steps::PossessivesAndHeads(r) => {
                    Parse::prune_parse(&mut p)?;
                    Parse::fix_possessives(&mut p);
                    p.update_heads(r.rules());
                }
            }
            Ok(p)
        });
        match done {
            Ok(p) => parses.push(p),
            Err(e) => read_error(&mut out, parses.len(), &e),
        }
    }
    for id in ["turbo:samples", "turbo:parse-texts", "turbo:parse-nodes", "turbo:pos-samples", "turbo:chunk-samples"] {
        out.layer(id);
    }
    for (i, p) in parses.iter().enumerate() {
        out.row("turbo:samples", vec![s(p.to_string_penn_treebank())]);
        out.row("turbo:parse-texts", vec![s(i), s(p.text())]);
        nodes(p, 0, i, &mut out);
        let pos = ParserPosSample::from_parse(p);
        for (token, tag) in pos.sentence.iter().zip(&pos.tags) {
            out.row("turbo:pos-samples", vec![s(i), s(token), s(tag)]);
        }
        let chunk = ParserChunkSample::from_parse(p);
        for ((token, tag), pred) in chunk.sentence.iter().zip(&chunk.tags).zip(&chunk.preds) {
            out.row("turbo:chunk-samples", vec![s(i), s(token), s(tag), s(pred)]);
        }
    }
    out
}

fn nodes(node: &Parse, depth: usize, sample: usize, out: &mut Rebuilt) {
    let span = node.span();
    let head = node.head();
    assert_eq!(&node.text()[span.range()], node.covered_text());
    out.row(
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
        nodes(child, depth + 1, sample, out);
    }
}

fn rebuild_rules(run_name: &str, input: &str) -> Rebuilt {
    let read: Result<String, HeadRulesError> = match run_name {
        "en" => EnglishHeadRules::from_str(input).map(|r| r.serialize_to_string()),
        "es" => AncoraSpanishHeadRules::from_str(input).map(|r| r.serialize_to_string()),
        _ => panic!("unknown head-rules run {run_name}"),
    };
    let mut out = Rebuilt::default();
    out.layer("turbo:head-rules");
    out.layer("turbo:rules-errors");
    match read {
        Ok(written) => {
            let lines = written.strip_suffix('\n').map_or(Vec::new(), |w| w.split('\n').collect());
            for line in lines {
                out.row("turbo:head-rules", vec![s(line)]);
            }
        }
        Err(e) => {
            let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class"));
            let message = matches!(e, HeadRulesError::InvalidTagCount { .. }).then(|| e.to_string());
            out.row("turbo:rules-errors", vec![s(class), message]);
        }
    }
    out
}

/// The goldens of the runs under `annotator` on its own input set, the one of the same name; the
/// `parse-treebank` runs beside them are `treebank`'s.
fn goldens(annotator: &str) -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join(annotator);
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        let set = run.join(annotator);
        if !set.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(set).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

fn compare(path: &std::path::Path, golden: &Golden, mut rebuilt: BTreeMap<&'static str, Vec<Vec<Option<String>>>>) {
    assert!(golden.header.error.is_none(), "{}: an error record", path.display());
    for layer in &golden.layers {
        let ours = rebuilt.remove(layer.id.as_str()).unwrap_or_else(|| panic!("{}: no {}", path.display(), layer.id));
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
    assert!(rebuilt.is_empty(), "{}: layers the golden lacks: {:?}", path.display(), rebuilt.keys());
}

#[test]
fn every_parse_golden_matches() {
    let data = turbo_goldens::data_dir();
    let shapes = shapes();
    let all = goldens("parse");
    assert!(all.len() >= 24, "only {} parse goldens", all.len());
    for (run_name, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        compare(&path, &golden, rebuild_parses(&run_name, &input).0);
    }
}

#[test]
fn every_head_rules_golden_matches() {
    let data = turbo_goldens::data_dir();
    let shapes = shapes();
    let all = goldens("head-rules");
    assert!(all.len() >= 26, "only {} head-rules goldens", all.len());
    for (run_name, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        compare(&path, &golden, rebuild_rules(&run_name, &input).0);
    }
}
