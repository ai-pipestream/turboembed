//! A small reader for the golden files the JVM generators write (format in
//! analysis/golden/jvm/README.md and tests/fixtures/README.md): header
//! lines, the escaped input, and `span:` lines with UTF-8 and UTF-16
//! offsets and, for the maxent detector, a probability; and the context
//! files, with every candidate's features, probabilities and decision. A
//! line this reader does not know fails the test.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_sentdetect::WhitespaceMode;
use turbo_unicode::modified_utf8::push_modified_utf8;

/// One span as Java reported it, with both kinds of offsets and, when the
/// component has one, its probability.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GoldenSpan {
    pub utf8: (u32, u32),
    pub utf16: (usize, usize),
    pub prob: Option<f64>,
}

/// One golden file.
#[derive(Debug)]
pub struct Golden {
    pub component: String,
    pub whitespace: WhitespaceMode,
    pub model: Option<String>,
    pub model_sha256: Option<String>,
    pub input: String,
    pub spans: Vec<GoldenSpan>,
}

/// The directory of this crate's golden files.
pub fn golden_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// The directory of this crate's fixtures.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The files of one golden directory of this crate, sorted by name.
pub fn files(dir: &str) -> Vec<PathBuf> {
    files_in(&golden_root().join(dir))
}

/// The `.txt` files of a directory, sorted by name.
pub fn files_in(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    out.sort();
    out
}

pub fn read(path: &Path) -> Golden {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn parse(text: &str) -> Result<Golden, String> {
    let (mut component, mut whitespace, mut input, mut model, mut model_sha256) = (None, None, None, None, None);
    let mut spans = Vec::new();
    for line in text.lines() {
        let (key, value) = split(line)?;
        match key {
            "component" => component = Some(value.to_string()),
            "whitespace" => whitespace = Some(mode(value)?),
            "model" => model = Some(value.to_string()),
            "model-sha256" => model_sha256 = Some(value.to_string()),
            "input" => input = Some(unescape(value)?),
            "span" => spans.push(span(value)?),
            "opennlp" | "jdk" => {}
            other => return Err(format!("unknown line {other:?}")),
        }
    }
    Ok(Golden {
        component: component.ok_or("no component")?,
        whitespace: whitespace.ok_or("no whitespace mode")?,
        model,
        model_sha256,
        input: input.ok_or("no input")?,
        spans,
    })
}

fn split(line: &str) -> Result<(&str, &str), String> {
    line.split_once(": ").or_else(|| line.strip_suffix(':').map(|k| (k, ""))).ok_or_else(|| line.to_string())
}

fn mode(value: &str) -> Result<WhitespaceMode, String> {
    match value {
        "unicode" => Ok(WhitespaceMode::Unicode),
        "legacy" => Ok(WhitespaceMode::Legacy),
        other => Err(format!("whitespace mode {other:?}")),
    }
}

/// `<utf8 start> <utf8 end> <utf16 start> <utf16 end> [<probability>]`.
fn span(value: &str) -> Result<GoldenSpan, String> {
    let fields: Vec<&str> = value.split(' ').collect();
    let n = |i: usize| fields[i].parse::<usize>().map_err(|_| format!("span {value:?}"));
    let prob = match fields.len() {
        4 => None,
        5 => Some(fields[4].parse::<f64>().map_err(|_| format!("probability in {value:?}"))?),
        _ => return Err(format!("span {value:?} needs four offsets and an optional probability")),
    };
    Ok(GoldenSpan { utf8: (n(0)? as u32, n(1)? as u32), utf16: (n(2)?, n(3)?), prob })
}

/// One candidate of a context file.
#[derive(Debug)]
pub struct GoldenCandidate {
    /// The UTF-16 index of the end character.
    pub at: usize,
    /// Each feature as the model looks it up: modified UTF-8.
    pub features: Vec<Vec<u8>>,
    /// P(s) and P(n).
    pub probs: (f64, f64),
    /// `split`, `no-split` or `vetoed`.
    pub decision: String,
}

/// The candidates of one input of a context file.
#[derive(Debug)]
pub struct ContextInput {
    pub input: String,
    pub candidates: Vec<GoldenCandidate>,
}

/// A context file.
#[derive(Debug)]
pub struct ContextFile {
    pub component: String,
    pub whitespace: WhitespaceMode,
    pub model: String,
    pub inputs: Vec<ContextInput>,
}

pub fn read_contexts(path: &Path) -> ContextFile {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse_contexts(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn parse_contexts(text: &str) -> Result<ContextFile, String> {
    let (mut component, mut whitespace, mut model) = (None, None, None);
    let mut inputs: Vec<ContextInput> = Vec::new();
    let mut candidate: Option<GoldenCandidate> = None;
    for line in text.lines() {
        let (key, value) = split(line)?;
        match key {
            "component" => component = Some(value.to_string()),
            "whitespace" => whitespace = Some(mode(value)?),
            "model" => model = Some(value.to_string()),
            "outcomes" if value == "s n" => {}
            "input" => inputs.push(ContextInput { input: unescape(value)?, candidates: Vec::new() }),
            "at" => {
                let at = value.parse().map_err(|_| format!("at {value:?}"))?;
                candidate =
                    Some(GoldenCandidate { at, features: Vec::new(), probs: (0.0, 0.0), decision: String::new() });
            }
            "features" => {
                let c = candidate.as_mut().ok_or("features before at")?;
                for f in value.split('\t') {
                    let mut bytes = Vec::new();
                    for u in unescape_units(f)? {
                        push_modified_utf8(&mut bytes, u32::from(u));
                    }
                    c.features.push(bytes);
                }
            }
            "probs" => {
                let c = candidate.as_mut().ok_or("probs before at")?;
                let (s, n) = value.split_once(' ').ok_or("two probabilities")?;
                c.probs = (s.parse().map_err(|_| "P(s)")?, n.parse().map_err(|_| "P(n)")?);
            }
            "decision" => {
                let mut c = candidate.take().ok_or("decision before at")?;
                c.decision = value.to_string();
                inputs.last_mut().ok_or("a candidate before any input")?.candidates.push(c);
            }
            "opennlp" | "jdk" | "model-sha256" => {}
            other => return Err(format!("unknown line {other:?}")),
        }
    }
    if candidate.is_some() {
        return Err("a candidate without a decision".into());
    }
    Ok(ContextFile {
        component: component.ok_or("no component")?,
        whitespace: whitespace.ok_or("no whitespace mode")?,
        model: model.ok_or("no model")?,
        inputs,
    })
}

/// Undoes the generator's escapes: `\\`, `\n`, `\r`, `\t` and `\uXXXX`
/// per UTF-16 code unit, a character outside the BMP being two of them.
pub fn unescape(s: &str) -> Result<String, String> {
    String::from_utf16(&unescape_units(s)?).map_err(|e| e.to_string())
}

/// [`unescape`] to UTF-16 code units, which may hold a lone surrogate (a
/// feature can).
pub fn unescape_units(s: &str) -> Result<Vec<u16>, String> {
    let mut units: Vec<u16> = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0; 2];
            units.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(u16::from(b'\\')),
            Some('n') => units.push(u16::from(b'\n')),
            Some('r') => units.push(u16::from(b'\r')),
            Some('t') => units.push(u16::from(b'\t')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).map_err(|_| format!("escape \\u{hex}"))?);
            }
            other => return Err(format!("escape \\{other:?}")),
        }
    }
    Ok(units)
}

#[test]
fn reader_unescapes_and_parses() {
    let g = parse(
        "component: newline\nopennlp: x\njdk: 21\nwhitespace: legacy\ninput: a\\\\b\\r\\n\\t\\u00A0\\uD83D\\uDE00\nspan: 0 3 0 3\n",
    )
    .unwrap();
    assert_eq!(g.component, "newline");
    assert_eq!(g.whitespace, WhitespaceMode::Legacy);
    assert_eq!(g.input, "a\\b\r\n\t\u{A0}\u{1F600}");
    assert_eq!(g.spans, [GoldenSpan { utf8: (0, 3), utf16: (0, 3), prob: None }]);
    let g = parse("component: x\nwhitespace: unicode\nmodel: m.bin\nmodel-sha256: ab\ninput: a\nspan: 0 1 0 1 0.25\n")
        .unwrap();
    assert_eq!(g.model.as_deref(), Some("m.bin"));
    assert_eq!(g.spans, [GoldenSpan { utf8: (0, 1), utf16: (0, 1), prob: Some(0.25) }]);
    assert_eq!(parse("component: newline\nwhitespace: unicode\ninput: \n").unwrap().input, "");
    assert!(parse("component: newline\nwhitespace: unicode\ninput: \\uD83D\n").is_err());
    assert!(parse("component: newline\nwhitespace: unicode\ninput: x\nerror: boom\n").is_err());
}

#[test]
fn reader_parses_contexts() {
    let c = parse_contexts(
        "component: c\nwhitespace: legacy\nmodel: m.bin\noutcomes: s n\ninput: a.\\uD83D\\uDE00\nat: 1\nfeatures: eos=.\ts=\\uD83D\nprobs: 0.75 0.25\ndecision: split\n",
    )
    .unwrap();
    assert_eq!(c.whitespace, WhitespaceMode::Legacy);
    assert_eq!(c.inputs.len(), 1);
    assert_eq!(c.inputs[0].input, "a.\u{1F600}");
    let k = &c.inputs[0].candidates[0];
    assert_eq!((k.at, k.probs, k.decision.as_str()), (1, (0.75, 0.25), "split"));
    // A lone surrogate is kept, as modified UTF-8.
    assert_eq!(k.features, [b"eos=.".to_vec(), b"s=\xED\xA0\xBD".to_vec()]);
    assert!(parse_contexts("component: c\nwhitespace: unicode\nmodel: m\ninput: x\nat: 0\n").is_err());
}
