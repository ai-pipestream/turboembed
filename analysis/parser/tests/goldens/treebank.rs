//! The parsers' training streams, context generators, models and output against the JVM: the
//! goldens `ParserTrainGolden.java` writes under `analysis/goldens/data/parse/<run>/parse-treebank`,
//! one per run and input of the `parse-treebank` set (`treebank`, the training corpus, and
//! `held-out`), 13 runs and 26 files.
//!
//! A sample is a non-blank line of the input, numbered from 0, read with `Parse::parse` for each
//! use, as the generator reads it fresh with `Parse.parseParse` for every call. The head rules are
//! the `en-head-rules.txt` fixture every header names. Each run is rebuilt here and every layer
//! compared:
//!
//! - `dictionary-cutoff-1` and `dictionary-cutoff-5`: `build_dictionary` over the input's samples,
//!   the n-grams in `String.compareTo` order, token by token.
//! - The eight `events-*` runs: the events one `ParserEventStream` makes of each sample, chunking
//!   or tree insertion, each with its outcome and its contexts in order.
//! - `generators`: every context generator called on each sample's chunk array (pruned, heads
//!   updated, initial chunks, punctuation collapsed), as built and with every chunk labeled
//!   complete.
//! - `model-chunking` and `model-treeinsert`: the model the JVM trained, read from the bytes in the
//!   `treebank` golden, the parses `ParserTool.parseLine` gives for each sample's tokens in each
//!   search, and the evaluation.
//!
//! Rows are matched by the columns that name them (a sample, a state, an index, a search) and are
//! in file order within those, so the order the generator loops in does not matter. Every `f64` is
//! compared by its bits. An exception's class is always compared and its message where this crate
//! can word it as Java does.
//!
//! One JVM behaviour is not reproduced. OpenNLP's chunking `BuildContextGenerator` reads the label
//! of the chunks before the one it is asked about in its bracket, quote, comma and end of sentence
//! checks, and throws `NullPointerException` on a chunk without one. Only the unlabeled state has
//! such chunks (the parser and the event stream label them first), and the Rust generator takes a
//! missing label as not a start. Where Java's walk meets a chunk without a label, the call is
//! expected to be that exception and its context is not compared.
//!
//! Until the goldens are checked in, the tests that read them say so once and pass. The others
//! check the comparison itself: goldens written from this side's own rows, for every run, read
//! back and compare clean, and a changed cell is found.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use turbo_goldens::hex::to_hex;
use turbo_goldens::{
    AnnotatorRef, Column, ColumnData, ColumnSpec, ColumnType, ConfigEntry, Generator, Golden, GoldenAnchor,
    GoldenLayer, Header, Input, Jdk, Model, OpenNlp, Shape, UnicodeVersion, sha256, sha512,
};
use turbo_maxent::{Properties, Zip};
use turbo_parser::model::{CHUNKER_TAGGER_MODEL_ENTRY_NAME, PARSER_TAGGER_MODEL_ENTRY_NAME, PARSER_TYPE};
use turbo_parser::treeinsert::{self, TreeInsertError};
use turbo_parser::{
    COMPLETE, ChunkContextGenerator, EnglishHeadRules, HeadRules, HeadRulesArtifact, Parse, ParseError,
    ParserChunkSample, ParserEvaluator, ParserEventType, ParserFactory, ParserModel, ParserTool,
    ParserTrainingParameters, ParserType, START, build_dictionary, chunking, collapse_punctuation,
};

/// The input set of every run.
const SET: &str = "parse-treebank";

/// The inputs of the set, by id: the training corpus first, as the model runs need it first.
const INPUTS: [&str; 2] = ["treebank", "held-out"];

/// What every header names: the head rules fixture and its class.
const RULES_FILE: &str = "en-head-rules.txt";
const RULES_CLASS: &str = "opennlp.tools.parser.lang.en.HeadRules";

/// The searches each sample is parsed with, as `(beam, advance)`, and the parse counts asked for.
const SEARCHES: [(u32, f64); 2] = [(1, 0.5), (3, 0.95)];
const PARSE_COUNTS: [u32; 2] = [1, 3];

/// The searches the evaluation runs, the last one `ParserFactory.create`'s defaults.
const EVALUATIONS: [(u32, f64); 3] = [(1, 0.5), (3, 0.95), (20, 0.95)];

/// The states each generator call runs in: the chunk array as built, and with every chunk labeled
/// complete.
const STATES: [&str; 2] = ["unlabeled", "complete"];

const DICTIONARY: &str = "turbo:dictionary";
const EVENTS: &str = "turbo:events";
const SAMPLE_ERRORS: &str = "turbo:sample-errors";
const PARSE_TEXTS: &str = "turbo:parse-texts";
const CHUNKS: &str = "turbo:chunks";
const CHUNKER_CONTEXTS: &str = "turbo:chunker-contexts";
const CHUNKING_BUILD: &str = "turbo:chunking-build";
const CHUNKING_BUILD_DICTIONARY: &str = "turbo:chunking-build-dictionary";
const CHUNKING_CHECK: &str = "turbo:chunking-check";
const TREEINSERT_BUILD: &str = "turbo:treeinsert-build";
const TREEINSERT_ATTACH: &str = "turbo:treeinsert-attach";
const TREEINSERT_CHECK: &str = "turbo:treeinsert-check";
const GENERATOR_ERRORS: &str = "turbo:generator-errors";
const MODEL_FILE: &str = "turbo:model-file";
const MODEL_HEX: &str = "turbo:model-hex";
const MODEL_ENTRIES: &str = "turbo:model-entries";
const MANIFEST: &str = "turbo:manifest";
const PARSES: &str = "turbo:parses";
const PARSE_NODES: &str = "turbo:parse-nodes";
const PARSE_ERRORS: &str = "turbo:parse-errors";
const EVALUATION: &str = "turbo:evaluation";

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;
const STR: ColumnType = ColumnType::Str;

const CONTEXTS: &[ColumnSpec] =
    &[col("sample", U32), col("state", STR), col("index", U32), col("size", U32), col("context", STR)];

/// Every layer of the 26 goldens: its id, how many of its leading columns name a row (rows with
/// the same names are compared in order), and its columns.
const LAYERS: &[(&str, usize, &[ColumnSpec])] = &[
    (DICTIONARY, 1, &[col("ngram", U32), col("token", STR)]),
    (EVENTS, 1, &[col("sample", U32), col("outcome", STR), col("size", U32), col("context", STR)]),
    (SAMPLE_ERRORS, 1, &[col("sample", U32), col("class", STR), col("message", STR)]),
    (PARSE_TEXTS, 1, &[col("sample", U32), col("text", STR)]),
    (
        CHUNKS,
        2,
        &[
            col("sample", U32),
            col("index", U32),
            col("type", STR),
            col("start", U32),
            col("end", U32),
            col("head-start", U32),
            col("head-end", U32),
            col("head-index", U32),
        ],
    ),
    (CHUNKER_CONTEXTS, 3, CONTEXTS),
    (CHUNKING_BUILD, 3, CONTEXTS),
    (CHUNKING_BUILD_DICTIONARY, 3, CONTEXTS),
    (
        CHUNKING_CHECK,
        5,
        &[
            col("sample", U32),
            col("state", STR),
            col("type", STR),
            col("start", U32),
            col("end", U32),
            col("size", U32),
            col("context", STR),
        ],
    ),
    (TREEINSERT_BUILD, 3, CONTEXTS),
    (
        TREEINSERT_ATTACH,
        4,
        &[
            col("sample", U32),
            col("state", STR),
            col("index", U32),
            col("frontier-index", U32),
            col("size", U32),
            col("context", STR),
        ],
    ),
    (
        TREEINSERT_CHECK,
        4,
        &[
            col("sample", U32),
            col("state", STR),
            col("index", U32),
            col("trim", U32),
            col("size", U32),
            col("context", STR),
        ],
    ),
    (
        GENERATOR_ERRORS,
        4,
        &[
            col("layer", STR),
            col("sample", U32),
            col("state", STR),
            col("index", U32),
            col("class", STR),
            col("message", STR),
        ],
    ),
    (MODEL_FILE, 0, &[col("sha256", STR), col("length", U32)]),
    (MODEL_HEX, 0, &[col("hex", STR)]),
    (MODEL_ENTRIES, 1, &[col("path", STR), col("length", U32), col("sha256", STR)]),
    (MANIFEST, 2, &[col("package", STR), col("key", STR), col("value", STR)]),
    (
        PARSES,
        5,
        &[
            col("sample", U32),
            col("beam", U32),
            col("advance", F64),
            col("k", U32),
            col("rank", U32),
            col("prob", F64),
            col("show", STR),
        ],
    ),
    (
        PARSE_NODES,
        5,
        &[
            col("sample", U32),
            col("beam", U32),
            col("advance", F64),
            col("k", U32),
            col("rank", U32),
            col("depth", U32),
            col("type", STR),
            col("start", U32),
            col("end", U32),
            col("head-start", U32),
            col("head-end", U32),
            col("head-index", U32),
        ],
    ),
    (
        PARSE_ERRORS,
        4,
        &[
            col("sample", U32),
            col("beam", U32),
            col("advance", F64),
            col("k", U32),
            col("class", STR),
            col("message", STR),
        ],
    ),
    (
        EVALUATION,
        2,
        &[col("beam", U32), col("advance", F64), col("precision", F64), col("recall", F64), col("f-measure", F64)],
    ),
];

fn shapes() -> Vec<Shape> {
    LAYERS.iter().map(|&(id, _, columns)| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns }).collect()
}

fn key_width(id: &str) -> usize {
    LAYERS.iter().find(|l| l.0 == id).unwrap_or_else(|| panic!("no layer {id}")).1
}

/// Which event stream a run reads.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stream {
    Chunking,
    TreeInsert,
}

/// What a run computes.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// `buildDictionary` at a cutoff.
    Dictionary(usize),
    /// A `ParserEventStream` of one type, with the input's own dictionary at cutoff 1 or without.
    Events(Stream, ParserEventType, bool),
    /// The context generators.
    Generators,
    /// A trained model and its parses.
    Model(ParserType),
}

/// One run: its directory, the annotator and config its headers hold besides the head rules, and
/// what it computes.
struct Run {
    name: &'static str,
    class: &'static str,
    method: &'static str,
    config: &'static [(&'static str, &'static str)],
    kind: Kind,
}

const CHUNKING_STREAM: &str = "opennlp.tools.parser.chunking.ParserEventStream";
const TREEINSERT_STREAM: &str = "opennlp.tools.parser.treeinsert.ParserEventStream";

/// An event stream run, `config` its `etype` and, with `dictionary`, its `dictionary-cutoff`.
const fn events(
    name: &'static str,
    stream: Stream,
    etype: ParserEventType,
    dictionary: bool,
    config: &'static [(&'static str, &'static str)],
) -> Run {
    let class = match stream {
        Stream::Chunking => CHUNKING_STREAM,
        Stream::TreeInsert => TREEINSERT_STREAM,
    };
    Run { name, class, method: "read", config, kind: Kind::Events(stream, etype, dictionary) }
}

/// The 13 runs, by directory name.
const RUNS: [Run; 13] = [
    Run {
        name: "dictionary-cutoff-1",
        class: "opennlp.tools.parser.AbstractBottomUpParser",
        method: "buildDictionary",
        config: &[("cutoff", "1")],
        kind: Kind::Dictionary(1),
    },
    Run {
        name: "dictionary-cutoff-5",
        class: "opennlp.tools.parser.AbstractBottomUpParser",
        method: "buildDictionary",
        config: &[("cutoff", "5")],
        kind: Kind::Dictionary(5),
    },
    events("events-chunking-build", Stream::Chunking, ParserEventType::Build, false, &[("etype", "BUILD")]),
    events(
        "events-chunking-build+dictionary",
        Stream::Chunking,
        ParserEventType::Build,
        true,
        &[("etype", "BUILD"), ("dictionary-cutoff", "1")],
    ),
    events("events-chunking-check", Stream::Chunking, ParserEventType::Check, false, &[("etype", "CHECK")]),
    events("events-treeinsert-build", Stream::TreeInsert, ParserEventType::Build, false, &[("etype", "BUILD")]),
    events("events-treeinsert-check", Stream::TreeInsert, ParserEventType::Check, false, &[("etype", "CHECK")]),
    events("events-treeinsert-attach", Stream::TreeInsert, ParserEventType::Attach, false, &[("etype", "ATTACH")]),
    events("events-chunk", Stream::Chunking, ParserEventType::Chunk, false, &[("etype", "CHUNK")]),
    events("events-tag", Stream::Chunking, ParserEventType::Tag, false, &[("etype", "TAG")]),
    Run {
        name: "generators",
        class: "opennlp.tools.parser.AbstractContextGenerator",
        method: "getContext",
        config: &[("check-types", "NP,VP,S"), ("dictionary-cutoff", "1")],
        kind: Kind::Generators,
    },
    Run {
        name: "model-chunking",
        class: "opennlp.tools.parser.chunking.Parser",
        method: "train",
        config: &[],
        kind: Kind::Model(ParserType::Chunking),
    },
    Run {
        name: "model-treeinsert",
        class: "opennlp.tools.parser.treeinsert.Parser",
        method: "train",
        config: &[],
        kind: Kind::Model(ParserType::TreeInsert),
    },
];

/// The check types of the generators run.
const CHECK_TYPES: [&str; 3] = ["NP", "VP", "S"];

/// One row rebuilt here: a cell per column, `None` where the golden's cell is not compared. An
/// optional row may be missing from the golden.
#[derive(Clone, Debug)]
struct Row {
    cells: Vec<Option<String>>,
    optional: bool,
}

/// The layers rebuilt for one golden, by id.
#[derive(Clone, Debug, Default)]
struct Rebuilt(BTreeMap<&'static str, Vec<Row>>);

impl Rebuilt {
    fn layers(&mut self, ids: &[&'static str]) {
        for id in ids {
            self.0.entry(id).or_default();
        }
    }

    fn row(&mut self, id: &'static str, cells: Vec<Option<String>>) {
        self.0.entry(id).or_default().push(Row { cells, optional: false });
    }

    fn optional_row(&mut self, id: &'static str, cells: Vec<Option<String>>) {
        self.0.entry(id).or_default().push(Row { cells, optional: true });
    }

    /// A context as its golden row has it: the call's key, the count, the strings joined by a
    /// space.
    fn context(&mut self, id: &'static str, mut key: Vec<Option<String>>, context: &[String]) {
        key.push(n(context.len()));
        key.push(s(context.join(" ")));
        self.row(id, key);
    }
}

fn n(v: impl Display) -> Option<String> {
    Some(v.to_string())
}

fn s(v: impl Into<String>) -> Option<String> {
    Some(v.into())
}

/// An `f64` cell: its bits, which are compared, and the value for a person.
fn f(x: f64) -> Option<String> {
    Some(f64_cell(x))
}

fn f64_cell(x: f64) -> String {
    format!("{:016x} ({x:?})", x.to_bits())
}

/// The golden's rows, each cell written as `Rebuilt` writes its own.
fn cells(layer: &GoldenLayer) -> Vec<Vec<String>> {
    (0..layer.len())
        .map(|i| {
            layer
                .columns
                .iter()
                .map(|c| match &c.data {
                    ColumnData::U32(v) => v[i].to_string(),
                    ColumnData::F64(v) => f64_cell(v[i]),
                    ColumnData::Str(v) => v.get(i).to_owned(),
                })
                .collect()
        })
        .collect()
}

/// The differences between a golden and the layers rebuilt for it, each a line naming the layer
/// and the row; at most `SHOWN` per layer are spelled out.
fn compare(golden: &Golden, rebuilt: &Rebuilt) -> Vec<String> {
    const SHOWN: usize = 5;
    let mut out = Vec::new();
    for layer in &golden.layers {
        let Some(ours) = rebuilt.0.get(layer.id.as_str()) else {
            out.push(format!("{}: in the golden, not rebuilt here", layer.id));
            continue;
        };
        let k = key_width(&layer.id);
        let theirs = cells(layer);
        let mut golden_groups: BTreeMap<Vec<&str>, Vec<&Vec<String>>> = BTreeMap::new();
        for row in &theirs {
            golden_groups.entry(row[..k].iter().map(String::as_str).collect()).or_default().push(row);
        }
        let mut our_groups: BTreeMap<Vec<&str>, Vec<&Row>> = BTreeMap::new();
        for row in ours {
            let key = row.cells[..k].iter().map(|c| c.as_deref().expect("a key cell is always compared")).collect();
            our_groups.entry(key).or_default().push(row);
        }
        let keys: BTreeSet<&Vec<&str>> = golden_groups.keys().chain(our_groups.keys()).collect();
        let mut differ = Vec::new();
        for key in keys {
            let found = match (golden_groups.get(key), our_groups.get(key)) {
                (Some(g), None) => Some(format!("{key:?}: only in the golden: {:?}", g[0])),
                (None, Some(o)) if o.iter().all(|r| r.optional) => None,
                (None, Some(o)) => Some(format!("{key:?}: only here: {:?}", o[0].cells)),
                (Some(g), Some(o)) => {
                    let o: Vec<&&Row> = o.iter().filter(|r| !r.optional || g.iter().any(|g| matches(r, g))).collect();
                    match g.iter().zip(&o).position(|(g, o)| !matches(o, g)) {
                        Some(i) => Some(format!("{key:?} row {i}: golden {:?}, here {:?}", g[i], o[i].cells)),
                        None if g.len() != o.len() => {
                            Some(format!("{key:?}: {} rows in the golden, {} here", g.len(), o.len()))
                        }
                        None => None,
                    }
                }
                (None, None) => unreachable!("a key of one of them"),
            };
            differ.extend(found);
        }
        if !differ.is_empty() {
            out.push(format!("{}: {} of {} row groups differ", layer.id, differ.len(), golden_groups.len()));
            out.extend(differ.iter().take(SHOWN).map(|d| format!("  {}: {d}", layer.id)));
        }
    }
    for id in rebuilt.0.keys().filter(|id| golden.layer(id).is_none()) {
        out.push(format!("{id}: rebuilt here, not in the golden"));
    }
    out
}

fn matches(ours: &Row, golden: &[String]) -> bool {
    ours.cells.len() == golden.len() && ours.cells.iter().zip(golden).all(|(o, g)| o.as_ref().is_none_or(|o| o == g))
}

/// The samples: the non-blank lines (as Java's `String.trim().isEmpty()` judges them), in order.
fn sample_lines(input: &str) -> Vec<&str> {
    input.lines().filter(|l| !l.trim_matches(|c: char| c <= ' ').is_empty()).collect()
}

/// The samples that read, as `buildDictionary`, the trainer and the evaluator take them.
fn parsed(lines: &[&str]) -> Vec<Parse> {
    lines.iter().filter_map(|l| Parse::parse(l).ok()).collect()
}

/// The exception OpenNLP throws where this crate returns `e`; the message is not compared.
fn parse_error(e: &ParseError) -> (String, Option<String>) {
    (e.java_class().unwrap_or_else(|| panic!("{e}: Java throws nothing here")).to_owned(), None)
}

/// The exception OpenNLP throws where tree insertion returns `e` on `sample`, with Java's message
/// where this crate can word it.
fn tree_insert_error(e: &TreeInsertError, sample: &Parse) -> (String, Option<String>) {
    const RUNTIME: &str = "java.lang.RuntimeException";
    match e {
        TreeInsertError::Parse(e) => parse_error(e),
        TreeInsertError::ParentNotInFrontier => (RUNTIME.to_owned(), None),
        TreeInsertError::NoAttachment { span } => {
            (RUNTIME.to_owned(), Some(format!("No Attachment: {}", &sample.text()[span.range()])))
        }
        TreeInsertError::MissingParent => ("java.lang.NullPointerException".to_owned(), None),
        // OpenNLP calls `System.exit(1)`, which no golden records, so this differs from any row.
        TreeInsertError::FrontierMismatch { .. } => ("System.exit".to_owned(), Some(e.to_string())),
        TreeInsertError::MissingAttachModel => panic!("{e}"),
    }
}

/// The n-gram of a `build_dictionary` key: each token's byte length, `:` and the token.
fn ngram(key: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = key;
    while !rest.is_empty() {
        let (len, after) = rest.split_once(':').unwrap_or_else(|| panic!("{key:?}: no length"));
        let len: usize = len.parse().unwrap_or_else(|_| panic!("{key:?}: bad length"));
        out.push(after[..len].to_owned());
        rest = &after[len..];
    }
    out
}

/// The generator's n-gram order: `String.compareTo` (UTF-16 units) token by token, a shorter
/// n-gram first when it is the start of the other.
fn java_order(a: &[String], b: &[String]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        match x.encode_utf16().cmp(y.encode_utf16()) {
            Ordering::Equal => {}
            o => return o,
        }
    }
    a.len().cmp(&b.len())
}

fn rebuild_dictionary(lines: &[&str], rules: &EnglishHeadRules, cutoff: usize) -> Rebuilt {
    let mut ngrams: Vec<Vec<String>> =
        build_dictionary(&parsed(lines), rules, cutoff).iter().map(|k| ngram(k)).collect();
    ngrams.sort_by(|a, b| java_order(a, b));
    let mut out = Rebuilt::default();
    out.layers(&[DICTIONARY]);
    for (i, tokens) in ngrams.iter().enumerate() {
        for token in tokens {
            out.row(DICTIONARY, vec![n(i), s(token)]);
        }
    }
    out
}

fn rebuild_events(
    lines: &[&str],
    rules: &EnglishHeadRules,
    stream: Stream,
    etype: ParserEventType,
    dictionary: bool,
) -> Rebuilt {
    let dict = dictionary.then(|| build_dictionary(&parsed(lines), rules, 1));
    let mut out = Rebuilt::default();
    out.layers(&[EVENTS, SAMPLE_ERRORS]);
    for (i, line) in lines.iter().enumerate() {
        let made = Parse::parse(line).map_err(|e| parse_error(&e)).and_then(|p| match stream {
            Stream::Chunking => {
                chunking::ParserEventStream::generate_events(std::slice::from_ref(&p), rules, etype, dict.as_ref())
                    .map_err(|e| parse_error(&e))
            }
            Stream::TreeInsert => {
                treeinsert::ParserEventStream::generate_events(std::slice::from_ref(&p), rules, etype, None)
                    .map_err(|e| tree_insert_error(&e, &p))
            }
        });
        match made {
            Ok(events) => {
                for e in events {
                    let context: Vec<String> = e.context().iter().map(|c| c.to_string_lossy()).collect();
                    let outcome = e.outcome().to_string_lossy();
                    out.row(EVENTS, vec![n(i), s(outcome), n(context.len()), s(context.join(" "))]);
                }
            }
            Err((class, message)) => out.row(SAMPLE_ERRORS, vec![n(i), s(class), message]),
        }
    }
    out
}

/// Whether OpenNLP's chunking `BuildContextGenerator.getContext(constituents, index)` reads the
/// label of a chunk without one: its checks for a closing bracket or quote, a comma, and a last
/// `.` walk back from `index - 1`, stop at the matching opening type (the closing checks only)
/// and otherwise read the chunk's label.
fn reads_missing_label(constituents: &[Parse], index: usize) -> bool {
    let open = match constituents[index].kind() {
        "-RRB-" => Some("-LRB-"),
        "-RCB-" => Some("-LCB-"),
        "''" => Some("``"),
        "'" => Some("`"),
        "," => Some(","),
        "." if index + 1 == constituents.len() => None,
        _ => return false,
    };
    for p in constituents[..index].iter().rev() {
        if open.is_some_and(|o| p.kind() == o) {
            return false;
        }
        match p.label() {
            None => return true,
            Some(l) if l.starts_with(START) => return false,
            Some(_) => {}
        }
    }
    false
}

fn rebuild_generators(lines: &[&str], rules: &EnglishHeadRules) -> Rebuilt {
    let punct = rules.punctuation_tags();
    let build = chunking::BuildContextGenerator::new();
    let build_dict = chunking::BuildContextGenerator::with_dictionary(build_dictionary(&parsed(lines), rules, 1));
    let check = chunking::CheckContextGenerator::new();
    let ti_build = treeinsert::BuildContextGenerator::new();
    let ti_attach = treeinsert::AttachContextGenerator::new(punct.clone());
    let ti_check = treeinsert::CheckContextGenerator::new(punct.clone());
    let mut out = Rebuilt::default();
    out.layers(&[
        PARSE_TEXTS,
        SAMPLE_ERRORS,
        CHUNKS,
        CHUNKER_CONTEXTS,
        CHUNKING_BUILD,
        CHUNKING_BUILD_DICTIONARY,
        CHUNKING_CHECK,
        TREEINSERT_BUILD,
        TREEINSERT_ATTACH,
        TREEINSERT_CHECK,
        GENERATOR_ERRORS,
    ]);
    for (i, line) in lines.iter().enumerate() {
        let mut p = match Parse::parse(line) {
            Ok(p) => p,
            Err(e) => {
                let (class, message) = parse_error(&e);
                out.row(SAMPLE_ERRORS, vec![n(i), s(class), message]);
                continue;
            }
        };
        out.row(PARSE_TEXTS, vec![n(i), s(p.text())]);
        if let Err(e) = Parse::prune_parse(&mut p) {
            let (class, message) = parse_error(&e);
            out.row(SAMPLE_ERRORS, vec![n(i), s(class), message]);
            continue;
        }
        p.update_heads(rules);

        // The chunker's contexts, from the initial chunks before punctuation is collapsed.
        let sample = ParserChunkSample::from_parse(&p);
        let words: Vec<&str> = sample.sentence.iter().map(String::as_str).collect();
        let tags: Vec<&str> = sample.tags.iter().map(String::as_str).collect();
        let preds: Vec<&str> = sample.preds.iter().map(String::as_str).collect();
        for t in 0..words.len() {
            let context = ChunkContextGenerator::new().get_context(t, &words, &tags, &preds);
            out.context(CHUNKER_CONTEXTS, vec![n(i), s(STATES[0]), n(t)], &context);
        }

        let initial: Vec<Parse> = p.initial_chunks().into_iter().cloned().collect();
        let built = collapse_punctuation(&initial, punct);
        for (ci, c) in built.iter().enumerate() {
            let (span, head) = (c.span(), c.head());
            out.row(
                CHUNKS,
                vec![
                    n(i),
                    n(ci),
                    s(c.kind()),
                    n(span.start),
                    n(span.end),
                    n(head.span.start),
                    n(head.span.end),
                    n(c.head_index()),
                ],
            );
        }

        for state in STATES {
            let mut cs = built.clone();
            if state == "complete" {
                for c in &mut cs {
                    c.set_label(Some(COMPLETE.to_owned()));
                }
            }
            let refs: Vec<&Parse> = cs.iter().collect();
            let java_error =
                |out: &mut Rebuilt, layer: &'static str, index: usize, (class, message): (String, Option<String>)| {
                    out.row(GENERATOR_ERRORS, vec![s(layer), n(i), s(state), n(index), s(class), message]);
                };
            for index in 0..cs.len() {
                let key = || vec![n(i), s(state), n(index)];
                for (layer, generator) in [(CHUNKING_BUILD, &build), (CHUNKING_BUILD_DICTIONARY, &build_dict)] {
                    if state == STATES[0] && reads_missing_label(&cs, index) {
                        java_error(&mut out, layer, index, ("java.lang.NullPointerException".to_owned(), None));
                    } else {
                        out.context(layer, key(), &generator.get_context(&cs, index));
                    }
                }
                out.context(TREEINSERT_BUILD, key(), &ti_build.get_context(&refs, index));
                // The frontier is the chunks before `index`, nearest first.
                let frontier: Vec<&Parse> = refs[..index].iter().rev().copied().collect();
                for fi in 0..frontier.len() {
                    let mut key = key();
                    key.push(n(fi));
                    out.context(TREEINSERT_ATTACH, key, &ti_attach.get_context(&refs, index, &frontier, fi));
                }
                if !cs[index].is_pos_tag() {
                    for trim in [false, true] {
                        match ti_check.get_context(&cs[index], &refs, index, trim) {
                            Ok(context) => {
                                let mut key = key();
                                key.push(n(u32::from(trim)));
                                out.context(TREEINSERT_CHECK, key, &context);
                            }
                            Err(TreeInsertError::ParentNotInFrontier) => {
                                let rf = treeinsert::Parser::get_right_frontier(&cs[0], punct);
                                let rf: Vec<String> = rf.iter().map(ToString::to_string).collect();
                                let message =
                                    format!("Parent not found in right frontier:{} rf=[{}]", cs[index], rf.join(", "));
                                let class = "java.lang.RuntimeException".to_owned();
                                java_error(&mut out, TREEINSERT_CHECK, index, (class, Some(message)));
                            }
                            Err(e) => java_error(&mut out, TREEINSERT_CHECK, index, tree_insert_error(&e, &p)),
                        }
                    }
                }
            }
            for start in 0..cs.len() {
                for end in start..cs.len().min(start + 3) {
                    for kind in CHECK_TYPES {
                        let key = vec![n(i), s(state), s(kind), n(start), n(end)];
                        out.context(CHUNKING_CHECK, key, &check.get_context(&cs, kind, start, end));
                    }
                }
            }
        }
    }
    out
}

/// The bytes of the model the JVM trained, from its hex rows.
fn model_bytes(golden: &Golden) -> Vec<u8> {
    let hex = golden.layer(MODEL_HEX).and_then(|l| l.strs("hex")).expect("the model's hex rows");
    let digit = |c: u8| match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("{:?} is not a lowercase hex digit", c as char),
    };
    let mut out = Vec::new();
    for row in hex.iter() {
        let (pairs, rest) = row.as_bytes().as_chunks::<2>();
        assert!(rest.is_empty(), "a hex row of odd length");
        out.extend(pairs.iter().map(|&[hi, lo]| (digit(hi) << 4) | digit(lo)));
    }
    out
}

fn entry_names<B: AsRef<[u8]>>(zip: &Zip<B>) -> Vec<String> {
    zip.names().map(|n| String::from_utf8(n.to_vec()).expect("a UTF-8 entry name")).collect()
}

/// The layers that describe the model file: its digest and length, its bytes in rows of 64, every
/// entry with those of the nested tagger and chunker packages (`<entry>/<nested entry>`), and
/// every manifest. A nested package's own row may be missing from the golden.
fn model_layers(out: &mut Rebuilt, bytes: &[u8]) {
    out.layers(&[MODEL_FILE, MODEL_HEX, MODEL_ENTRIES, MANIFEST]);
    out.row(MODEL_FILE, vec![s(to_hex(&sha256(bytes))), n(bytes.len())]);
    for row in bytes.chunks(64) {
        out.row(MODEL_HEX, vec![s(to_hex(row))]);
    }
    let manifest = |out: &mut Rebuilt, package: &str, data: &[u8]| {
        let properties = Properties::load(data).unwrap_or_else(|e| panic!("{package:?} manifest: {e}"));
        for (key, value) in properties.iter() {
            out.row(MANIFEST, vec![s(package), s(key), s(value)]);
        }
    };
    let zip = Zip::new(bytes).expect("the model is a zip");
    for name in entry_names(&zip) {
        let data = zip.entry(&name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let row = vec![s(&name), n(data.len()), s(to_hex(&sha256(&data)))];
        if name == "manifest.properties" {
            manifest(out, "", &data);
        }
        if !data.starts_with(b"PK\x03\x04") {
            out.row(MODEL_ENTRIES, row);
            continue;
        }
        out.optional_row(MODEL_ENTRIES, row);
        let nested = Zip::new(&data[..]).unwrap_or_else(|e| panic!("{name}: {e}"));
        for inner in entry_names(&nested) {
            let inner_data = nested.entry(&inner).unwrap_or_else(|e| panic!("{name}/{inner}: {e}"));
            let path = format!("{name}/{inner}");
            out.row(MODEL_ENTRIES, vec![s(path), n(inner_data.len()), s(to_hex(&sha256(&inner_data)))]);
            if inner == "manifest.properties" {
                manifest(out, &name, &inner_data);
            }
        }
    }
}

/// A node and the nodes below it, in preorder.
fn nodes(out: &mut Rebuilt, key: &[Option<String>], node: &Parse, depth: usize) {
    let (span, head) = (node.span(), node.head());
    let mut row = key.to_vec();
    row.extend([
        n(depth),
        s(node.kind()),
        n(span.start),
        n(span.end),
        n(head.span.start),
        n(head.span.end),
        n(node.head_index()),
    ]);
    out.row(PARSE_NODES, row);
    for child in node.children() {
        nodes(out, key, child, depth + 1);
    }
}

fn rebuild_model(lines: &[&str], bytes: &[u8], with_model: bool) -> Rebuilt {
    let model = Arc::new(ParserModel::from_bytes(bytes.to_vec()).unwrap_or_else(|e| panic!("the model: {e}")));
    let mut out = Rebuilt::default();
    if with_model {
        model_layers(&mut out, model.as_bytes());
    }
    out.layers(&[PARSE_TEXTS, PARSES, PARSE_NODES, PARSE_ERRORS, EVALUATION]);
    let references: Vec<Parse> =
        lines.iter().map(|l| Parse::parse(l).unwrap_or_else(|e| panic!("{l:?}: {e}"))).collect();
    for (i, reference) in references.iter().enumerate() {
        let tokens: Vec<&str> = reference.token_nodes().into_iter().map(Parse::covered_text).collect();
        let line = tokens.join(" ");
        out.row(PARSE_TEXTS, vec![n(i), s(&line)]);
        for (beam, advance) in SEARCHES {
            let parser = ParserFactory::create_with_params(Arc::clone(&model), beam as usize, advance);
            for k in PARSE_COUNTS {
                let key = vec![n(i), n(beam), f(advance), n(k)];
                match ParserTool::parse_line(&line, &parser, k as usize) {
                    Ok(parses) => {
                        for (rank, p) in parses.iter().enumerate() {
                            let mut key = key.clone();
                            key.push(n(rank));
                            let mut show = String::new();
                            p.show(&mut show);
                            let mut row = key.clone();
                            row.extend([f(p.prob()), s(show)]);
                            out.row(PARSES, row);
                            nodes(&mut out, &key, p, 0);
                        }
                    }
                    Err(e) => {
                        let (class, message) = parse_error(&e);
                        let mut row = key;
                        row.extend([s(class), message]);
                        out.row(PARSE_ERRORS, row);
                    }
                }
            }
        }
    }
    for (beam, advance) in EVALUATIONS {
        let parser = ParserFactory::create_with_params(Arc::clone(&model), beam as usize, advance);
        let mut evaluator = ParserEvaluator::new(&parser);
        let row = match evaluator.evaluate(&references) {
            Ok(()) => {
                let fm = evaluator.fmeasure();
                vec![n(beam), f(advance), f(fm.precision()), f(fm.recall()), f(fm.f_measure())]
            }
            // The golden has scores, so this row differs and says why.
            Err(e) => vec![n(beam), f(advance), s(format!("{}: {e}", parse_error(&e).0)), None, None],
        };
        out.row(EVALUATION, row);
    }
    out
}

/// Everything `run` computes from `input`, the input with this id; `model` is the trained model's
/// bytes for a model run.
fn rebuild(run: &Run, id: &str, input: &str, rules: &EnglishHeadRules, model: Option<&[u8]>) -> Rebuilt {
    let lines = sample_lines(input);
    match run.kind {
        Kind::Dictionary(cutoff) => rebuild_dictionary(&lines, rules, cutoff),
        Kind::Events(stream, etype, dictionary) => rebuild_events(&lines, rules, stream, etype, dictionary),
        Kind::Generators => rebuild_generators(&lines, rules),
        Kind::Model(_) => rebuild_model(&lines, model.expect("a model run's model"), id == INPUTS[0]),
    }
}

/// Checks what the header says against the run: the input, the head rules, the annotator and the
/// config.
fn check_header(what: &str, golden: &Golden, run: &Run, id: &str, rules_text: &str) {
    let h = &golden.header;
    if let Some(e) = &h.error {
        panic!("{what}: the run threw {}: {}", e.class, e.message);
    }
    assert_eq!((h.input.set.as_str(), h.input.id.as_str()), (SET, id), "{what}: the input");
    assert_eq!(h.models.len(), 1, "{what}: one model record, the head rules");
    assert_eq!(h.models[0].file, RULES_FILE, "{what}: the head rules file");
    assert!(h.models[0].sha512 == sha512(rules_text.as_bytes()), "{what}: {RULES_FILE} is not the file the JVM read");
    assert_eq!((h.annotator.class.as_str(), h.annotator.method.as_str()), (run.class, run.method), "{what}");
    let config = |key: &str| h.config.iter().find(|c| c.key == key).map(|c| c.value.as_str());
    assert_eq!(config("rules"), Some(RULES_CLASS), "{what}: config rules");
    for &(key, value) in run.config {
        assert_eq!(config(key), Some(value), "{what}: config {key}");
    }
}

fn rules_text() -> String {
    super::fixture(RULES_FILE)
}

/// The directory the run directories are in, or `None`, said once, when none of the 13 is
/// checked in yet.
fn checked_in() -> Option<PathBuf> {
    let root = turbo_goldens::data_dir().join("parse");
    if RUNS.iter().any(|r| root.join(r.name).join(SET).exists()) {
        return Some(root);
    }
    eprintln!(
        "skipped: {} is absent, as are the other 12 parse-treebank run directories; \
         ParserTrainGolden.java writes them",
        root.join(RUNS[0].name).join(SET).display()
    );
    None
}

/// Compares every golden of the runs `select` picks, failing with every difference found.
fn check_runs(select: impl Fn(&Kind) -> bool) {
    let Some(root) = checked_in() else { return };
    let rules_text = rules_text();
    let rules = EnglishHeadRules::from_str(&rules_text).unwrap();
    let shapes = shapes();
    let mut problems = Vec::new();
    let mut checked = 0;
    for run in RUNS.iter().filter(|r| select(&r.kind)) {
        let mut model = None;
        for id in INPUTS {
            let path = root.join(run.name).join(SET).join(format!("{id}.golden"));
            let what = format!("{}/{SET}/{id}", run.name);
            assert!(path.is_file(), "{} is missing", path.display());
            let golden = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
            check_header(&what, &golden, run, id, &rules_text);
            let input = golden.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
            if matches!(run.kind, Kind::Model(_)) && model.is_none() {
                model = Some(model_bytes(&golden));
            }
            let rebuilt = rebuild(run, id, &input, &rules, model.as_deref());
            problems.extend(compare(&golden, &rebuilt).into_iter().map(|p| format!("{what}: {p}")));
            checked += 1;
        }
    }
    assert!(checked > 0, "no run selected");
    assert!(problems.is_empty(), "{} differences from the JVM:\n{}", problems.len(), problems.join("\n"));
}

#[test]
fn the_parse_treebank_goldens_are_the_26_expected_files() {
    let Some(root) = checked_in() else { return };
    let expected: BTreeSet<String> =
        RUNS.iter().flat_map(|r| INPUTS.map(|id| format!("{}/{SET}/{id}.golden", r.name))).collect();
    assert_eq!(expected.len(), 26);
    let mut found = BTreeSet::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let dir = run.join(SET);
        if !dir.exists() {
            continue;
        }
        let mut stack = vec![dir];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    found.insert(path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    let missing: Vec<&String> = expected.difference(&found).collect();
    let extra: Vec<&String> = found.difference(&expected).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "under {}: missing {missing:?}, not expected {extra:?}",
        root.display()
    );
    for id in INPUTS {
        let input = turbo_goldens::data_dir().join("inputs").join(SET).join(format!("{id}.txt"));
        assert!(input.is_file(), "{} is missing", input.display());
    }
}

#[test]
fn the_dictionary_goldens_match() {
    check_runs(|k| matches!(k, Kind::Dictionary(_)));
}

#[test]
fn the_event_goldens_match() {
    check_runs(|k| matches!(k, Kind::Events(..)));
}

#[test]
fn the_generator_goldens_match() {
    check_runs(|k| matches!(k, Kind::Generators));
}

#[test]
fn the_chunking_model_goldens_match() {
    check_runs(|k| matches!(k, Kind::Model(ParserType::Chunking)));
}

#[test]
fn the_tree_insertion_model_goldens_match() {
    check_runs(|k| matches!(k, Kind::Model(ParserType::TreeInsert)));
}

/// The parameters of the model runs: every model MAXENT with 100 iterations and cutoff 1, and the
/// dictionary at cutoff 1. The run gives the tagger no beam size.
fn model_run_parameters() -> ParserTrainingParameters {
    let mut params = ParserTrainingParameters::with_iterations_and_cutoff(100, 1);
    params.tagger = params.chunker.clone();
    params
}

/// A model trained here on the corpus the JVM trained on, with the same parameters, is the model
/// the JVM wrote: the same manifests (the parser type under `parser-type`, each maxent model's
/// training report under its name, the serializer classes, and the tagger's and chunker's own
/// entries) and the same bytes in every other entry. The JVM's manifests have no `Timestamp`,
/// which this side writes, so the manifest files and the nested packages holding them are
/// compared by their entries only.
#[test]
fn a_model_trained_here_is_the_model_the_jvm_wrote() {
    let Some(root) = checked_in() else { return };
    let rules = EnglishHeadRules::from_str(&rules_text()).unwrap();
    let shapes = shapes();
    let nested = [PARSER_TAGGER_MODEL_ENTRY_NAME, CHUNKER_TAGGER_MODEL_ENTRY_NAME];
    let keep = |id: &str, row: &[String]| match id {
        MANIFEST => row[1] != "Timestamp",
        _ => !row[0].ends_with("manifest.properties") && !nested.contains(&row[0].as_str()),
    };
    for run in RUNS.iter().filter(|r| matches!(r.kind, Kind::Model(_))) {
        let path = root.join(run.name).join(SET).join(format!("{}.golden", INPUTS[0]));
        let golden = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
        let input = golden.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let samples = parsed(&sample_lines(&input));
        let artifact = HeadRulesArtifact::English(rules.clone());
        let params = model_run_parameters();
        let model = match run.kind {
            Kind::Model(ParserType::Chunking) => chunking::Parser::train("en", &samples, artifact, &params),
            Kind::Model(ParserType::TreeInsert) => treeinsert::Parser::train("en", &samples, artifact, &params),
            _ => unreachable!("a model run"),
        }
        .unwrap_or_else(|e| panic!("{}: {e}", run.name));
        let mut rebuilt = Rebuilt::default();
        model_layers(&mut rebuilt, model.as_bytes());
        for id in [MANIFEST, MODEL_ENTRIES] {
            let theirs: Vec<Vec<String>> =
                cells(golden.layer(id).unwrap()).into_iter().filter(|row| keep(id, row)).collect();
            let ours: Vec<Vec<String>> = rebuilt.0[id]
                .iter()
                .map(|row| row.cells.iter().map(|c| c.clone().unwrap()).collect::<Vec<String>>())
                .filter(|row| keep(id, row))
                .collect();
            assert_eq!(ours, theirs, "{}: {id}", run.name);
        }
        assert_eq!(model.file().manifest().get(PARSER_TYPE), Some(model.parser_type().as_str()), "{}", run.name);
    }
}

/// A project-written corpus for the self-test, with brackets so that the unlabeled chunking build
/// context reads a missing label.
const SELF_TEST_TREEBANK: &str = "\
(TOP (S (NP (DT The) (JJ quick) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .)))

(TOP (S (NP (DT The) (NN dog)) (VP (VBD slept) (PP (IN under) (NP (DT the) (JJ old) (NN tree)))) (. .)))
(TOP (S (NP (NNP Alice)) (VP (VBD wrote) (NP (DT a) (NN program)) (PP (IN in) (NP (NNP Rust)))) (. .)))
(TOP (S (NP (PRP She)) (VP (VBD saw) (NP (DT the) (NN ship)) (, ,) (CC and) (VP (VBD waved))) (. .)))
(TOP (S (PP (IN On) (NP (NNP Monday))) (, ,) (NP (DT the) (NN team)) (VP (VBD shipped) (NP (DT the) (NN release))) (. .)))
(TOP (S (NP (NNP Bob)) (-LRB- -LRB-) (NP (NNP Rust)) (-RRB- -RRB-) (VP (VBD won)) (. .)))
(TOP (S (NP (NP (DT The) (NN man)) (PP (IN with) (NP (DT a) (NN hat)))) (VP (VBD left) (NP (DT the) (NN room))) (. .)))
(TOP (S (NP (NNP Eve)) (VP (VP (VBD read) (NP (DT a) (NN book))) (CC and) (VP (VBD slept))) (. .)))
";

const SELF_TEST_HELD_OUT: &str = "\
(TOP (S (NP (DT The) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat)))) (. .)))
(TOP (S (NP (NNP Eve)) (VP (VBD read) (NP (DT a) (NN book))) (. .)))
";

/// The header the generator would write for `run` on `input`, the input with this id.
fn header(run: &Run, id: &str, input: &str, rules_text: &str) -> Header {
    let text_field = || "self-test".to_owned();
    let mut config: Vec<ConfigEntry> = run
        .config
        .iter()
        .chain(&[("rules", RULES_CLASS)])
        .map(|&(key, value)| ConfigEntry { key: key.into(), value: value.into() })
        .collect();
    config.sort_by(|a, b| a.key.cmp(&b.key));
    Header {
        generator: Generator { name: text_field(), version: 1 },
        opennlp: OpenNlp { commit: [0; 20], version: text_field() },
        jdk: Jdk { version: text_field(), vm_version: text_field(), vendor: text_field() },
        unicode: UnicodeVersion { major: 15, minor: 0 },
        input: Input {
            set: SET.into(),
            id: id.into(),
            utf8_len: input.len() as u32,
            utf16_len: input.encode_utf16().count() as u32,
            sha256: sha256(input.as_bytes()),
        },
        models: vec![Model { file: RULES_FILE.into(), sha512: sha512(rules_text.as_bytes()) }],
        annotator: AnnotatorRef { class: run.class.into(), method: run.method.into() },
        config,
        error: None,
    }
}

/// A golden with this header holding `rebuilt`.
fn to_golden(header: Header, rebuilt: &Rebuilt, shapes: &[Shape]) -> Golden {
    let layers = rebuilt
        .0
        .iter()
        .map(|(&id, rows)| {
            let shape = shapes.iter().find(|s| s.id == id).unwrap_or_else(|| panic!("no shape for {id}"));
            let mut columns: Vec<Column> =
                shape.columns.iter().map(|c| Column { name: c.name.into(), data: ColumnData::empty(c.ty) }).collect();
            for row in rows {
                assert_eq!(row.cells.len(), columns.len(), "{id}: {:?}", row.cells);
                for (cell, column) in row.cells.iter().zip(&mut columns) {
                    let cell = cell.as_deref().unwrap_or("");
                    match &mut column.data {
                        ColumnData::U32(v) => v.push(cell.parse().unwrap_or_else(|_| panic!("{id}: {cell:?}"))),
                        ColumnData::F64(v) => v.push(f64::from_bits(u64::from_str_radix(&cell[..16], 16).unwrap())),
                        ColumnData::Str(v) => v.push(cell),
                    }
                }
            }
            GoldenLayer { id: id.into(), anchor: GoldenAnchor::Document, columns }
        })
        .collect();
    Golden { header, layers }
}

/// `golden` with the last cell of the first row of its first layer that has rows and no cell left
/// uncompared changed, if it has such a layer.
fn with_one_cell_changed(golden: &Golden) -> Option<Golden> {
    let mut changed = golden.clone();
    let layer = changed
        .layers
        .iter_mut()
        .find(|l| !l.is_empty() && ![SAMPLE_ERRORS, GENERATOR_ERRORS, PARSE_ERRORS].contains(&l.id.as_str()))?;
    let last = layer.columns.last_mut().unwrap();
    last.data = match &last.data {
        ColumnData::U32(v) => ColumnData::U32([vec![v[0] + 1], v[1..].to_vec()].concat()),
        ColumnData::F64(v) => ColumnData::F64([vec![f64::from_bits(v[0].to_bits() ^ 1)], v[1..].to_vec()].concat()),
        ColumnData::Str(v) => {
            let mut strs = turbo_analysis_layers::Strs::default();
            strs.push(&format!("{}x", v.get(0)));
            for i in 1..v.len() {
                strs.push(v.get(i));
            }
            ColumnData::Str(strs)
        }
    };
    Some(changed)
}

/// The comparison itself, on goldens written from this side's own rows for every run: each reads
/// back with the shapes here and compares clean, and changing one cell makes it differ. The model
/// runs read models trained here on a small corpus.
#[test]
fn the_comparison_reads_back_what_it_rebuilds_and_finds_a_changed_cell() {
    let rules_text = rules_text();
    let rules = EnglishHeadRules::from_str(&rules_text).unwrap();
    let shapes = shapes();
    let samples = parsed(&sample_lines(SELF_TEST_TREEBANK));
    let params = ParserTrainingParameters::with_iterations_and_cutoff(10, 1);
    let artifact = || HeadRulesArtifact::English(rules.clone());
    let chunking_model = chunking::Parser::train("en", &samples, artifact(), &params).unwrap();
    let treeinsert_model = treeinsert::Parser::train("en", &samples, artifact(), &params).unwrap();
    let mut layers_seen = BTreeSet::new();
    let mut changes = 0;
    for run in &RUNS {
        let model = match run.kind {
            Kind::Model(ParserType::Chunking) => Some(chunking_model.as_bytes()),
            Kind::Model(ParserType::TreeInsert) => Some(treeinsert_model.as_bytes()),
            _ => None,
        };
        for (id, input) in INPUTS.into_iter().zip([SELF_TEST_TREEBANK, SELF_TEST_HELD_OUT]) {
            let what = format!("{}/{id}", run.name);
            let rebuilt = rebuild(run, id, input, &rules, model);
            let written = to_golden(header(run, id, input, &rules_text), &rebuilt, &shapes);
            let text = written.to_text().unwrap();
            let golden = Golden::parse_with(&text, &shapes).unwrap_or_else(|e| panic!("{what}: {e}"));
            check_header(&what, &golden, run, id, &rules_text);
            if let Some(bytes) = model.filter(|_| id == INPUTS[0]) {
                assert_eq!(model_bytes(&golden), bytes, "{what}");
            }
            assert_eq!(compare(&golden, &rebuilt), Vec::<String>::new(), "{what}");
            if let Some(changed) = with_one_cell_changed(&golden) {
                assert!(!compare(&changed, &rebuilt).is_empty(), "{what}: a changed cell");
                changes += 1;
            }
            layers_seen.extend(golden.layers.iter().filter(|l| !l.is_empty()).map(|l| l.id.clone()));
        }
    }
    // Every layer has rows in some run but the sample and parse errors, which this corpus does
    // not cause.
    let expected: BTreeSet<String> =
        LAYERS.iter().map(|l| l.0).filter(|id| ![SAMPLE_ERRORS, PARSE_ERRORS].contains(id)).map(String::from).collect();
    assert_eq!(layers_seen, expected);
    // Only the held-out input of `dictionary-cutoff-5` has no layer with rows to change.
    assert_eq!(changes, 2 * RUNS.len() - 1);
}

/// A model golden's entries may leave out the nested packages themselves, but nothing else.
#[test]
fn only_a_nested_package_row_may_be_missing() {
    let mut rebuilt = Rebuilt::default();
    rebuilt.row(MODEL_ENTRIES, vec![s("build.model"), n(3), s("ab")]);
    rebuilt.optional_row(MODEL_ENTRIES, vec![s("parsertager.postagger"), n(5), s("cd")]);
    rebuilt.row(MODEL_ENTRIES, vec![s("parsertager.postagger/pos.model"), n(2), s("ef")]);
    let shapes = shapes();
    let golden = |rows: &[(&str, u32, &str)]| {
        let mut listed = Rebuilt::default();
        listed.layers(&[MODEL_ENTRIES]);
        for &(path, length, sha) in rows {
            listed.row(MODEL_ENTRIES, vec![s(path), n(length), s(sha)]);
        }
        let run = RUNS.iter().find(|r| r.name == "model-chunking").unwrap();
        to_golden(header(run, "treebank", "", ""), &listed, &shapes)
    };
    let both = [("build.model", 3, "ab"), ("parsertager.postagger/pos.model", 2, "ef")];
    assert_eq!(compare(&golden(&both), &rebuilt), Vec::<String>::new());
    let all =
        [("build.model", 3, "ab"), ("parsertager.postagger", 5, "cd"), ("parsertager.postagger/pos.model", 2, "ef")];
    assert_eq!(compare(&golden(&all), &rebuilt), Vec::<String>::new());
    assert!(!compare(&golden(&both[..1]), &rebuilt).is_empty(), "a nested entry missing");
    let wrong =
        [("build.model", 3, "ab"), ("parsertager.postagger", 6, "cd"), ("parsertager.postagger/pos.model", 2, "ef")];
    assert!(!compare(&golden(&wrong), &rebuilt).is_empty(), "a nested package listed with the wrong length");
}

#[test]
fn ngrams_are_in_string_compare_order() {
    let key = |tokens: &[&str]| tokens.iter().map(|t| format!("{}:{t}", t.len())).collect::<String>();
    assert_eq!(ngram(&key(&["1:2", "a"])), ["1:2", "a"]);
    let mut ngrams: Vec<Vec<String>> = [&["b"][..], &["a", "b"], &["a"], &["\u{ff61}"], &["\u{1f600}"], &["B"]]
        .iter()
        .map(|t| t.iter().map(|s| s.to_string()).collect())
        .collect();
    ngrams.sort_by(|a, b| java_order(a, b));
    // UTF-16 order puts a surrogate pair before U+FF61, where UTF-8 order puts it after.
    let order: Vec<String> = ngrams.iter().map(|n| n.join(" ")).collect();
    assert_eq!(order, ["B", "a", "a b", "b", "\u{1f600}", "\u{ff61}"]);
}

#[test]
fn a_missing_label_is_read_where_java_reads_it() {
    let chunks = |kinds: &[&str]| -> Vec<Parse> {
        let text: String = kinds.iter().map(|_| "x ").collect();
        let text: Arc<str> = text.into();
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                Parse::new(
                    Arc::clone(&text),
                    turbo_analysis::Span::new(2 * i as u32, 2 * i as u32 + 1),
                    *k,
                    1.0,
                    i as i32,
                )
            })
            .collect()
    };
    let cs = chunks(&["NP", "-LRB-", "NP", "-RRB-", "VP", "."]);
    assert!(!reads_missing_label(&cs, 0));
    assert!(reads_missing_label(&cs, 3), "the NP before the bracket has no label");
    assert!(reads_missing_label(&cs, 5), "a last `.` reads the label before it");
    let cs = chunks(&["NP", "-LRB-", "-RRB-"]);
    assert!(!reads_missing_label(&cs, 2), "the opening bracket comes first");
    let mut cs = chunks(&["NP", "NP", "-RRB-"]);
    cs[1].set_label(Some("S-NP".into()));
    assert!(!reads_missing_label(&cs, 2), "a start stops the walk");
    cs[1].set_label(Some(COMPLETE.into()));
    assert!(reads_missing_label(&cs, 2), "the walk goes on past a complete chunk");
}
