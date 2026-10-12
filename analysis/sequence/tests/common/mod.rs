//! Reading the files under `golden/`, which the Java programs in
//! `golden/src` wrote (see the crate README).

#![allow(dead_code)]

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use turbo_sequence::featuregen::descriptor::Resources;

pub fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("golden")
}

pub fn read(relative: &str) -> String {
    let path = golden().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The UTF-16 code units of a string escaped by `Common.escape`.
pub fn unescape_units(s: &str) -> Vec<u16> {
    let mut out = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => out.push(0x5C),
            Some('t') => out.push(0x09),
            Some('n') => out.push(0x0A),
            Some('r') => out.push(0x0D),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {s}"),
        }
    }
    out
}

/// A string escaped by `Common.escape`, which must be well formed UTF-16.
pub fn unescape(s: &str) -> String {
    String::from_utf16(&unescape_units(s)).unwrap_or_else(|_| panic!("lone surrogate in {s}"))
}

/// The tab-separated escaped strings of a line, none for an empty line.
pub fn unescape_list(s: &str) -> Vec<String> {
    if s.is_empty() { Vec::new() } else { s.split('\t').map(unescape).collect() }
}

/// A string as `Common.escape` writes it.
pub fn escape(s: &str) -> String {
    let mut out = String::new();
    for u in s.encode_utf16() {
        turbo_sequence::features::escape_unit(&mut out, u);
    }
    out
}

/// The value after `key: ` on a line, failing when the line is not that.
pub fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.strip_prefix(key)
        .and_then(|rest| rest.strip_prefix(": ").or_else(|| rest.strip_prefix(":")))
        .unwrap_or_else(|| panic!("expected {key}: in {line:?}"))
}

/// The lines after the three header lines, which must name `component`.
pub fn body<'a>(text: &'a str, component: &str) -> std::str::Lines<'a> {
    let mut lines = text.lines();
    assert_eq!(field(lines.next().unwrap(), "component"), component);
    field(lines.next().unwrap(), "opennlp");
    field(lines.next().unwrap(), "jdk");
    lines
}

/// The resources a descriptor names, read from `golden/resources`.
pub struct DirResources(pub PathBuf);

impl Resources for DirResources {
    fn resource(&self, name: &str) -> turbo_sequence::Result<Option<Cow<'_, [u8]>>> {
        match std::fs::read(self.0.join(name)) {
            Ok(bytes) => Ok(Some(Cow::Owned(bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => panic!("{name}: {e}"),
        }
    }
}

/// The escaped features of `out`, one string per feature.
pub fn escaped(out: &turbo_sequence::Features) -> Vec<String> {
    (0..out.len()).map(|i| out.escaped(i)).collect()
}

/// One sentence of a feature dump.
pub struct DumpSentence {
    pub tokens: Vec<String>,
    pub outcomes: Vec<String>,
    /// The escaped features of each token.
    pub features: Vec<Vec<String>>,
    /// Whether `updateAdaptiveData` was called after the sentence.
    pub update: bool,
}

/// A `component: features` file.
pub fn read_dump(relative: &str) -> (String, Vec<DumpSentence>) {
    let text = read(relative);
    let mut lines = body(&text, "features").peekable();
    let descriptor = field(lines.next().unwrap(), "descriptor").to_string();
    let mut sentences = Vec::new();
    while let Some(line) = lines.next() {
        field(line, "sentence");
        let tokens = unescape_list(field(lines.next().unwrap(), "tokens"));
        let outcomes = unescape_list(field(lines.next().unwrap(), "outcomes"));
        let mut features = Vec::new();
        while lines.peek().is_some_and(|l| l.starts_with("at:")) {
            let at: usize = field(lines.next().unwrap(), "at").parse().unwrap();
            assert_eq!(at, features.len());
            let f = field(lines.next().unwrap(), "features");
            features.push(if f.is_empty() { Vec::new() } else { f.split('\t').map(str::to_string).collect() });
        }
        let update = lines.peek() == Some(&"update");
        if update {
            lines.next();
        }
        sentences.push(DumpSentence { tokens, outcomes, features, update });
    }
    (descriptor, sentences)
}
