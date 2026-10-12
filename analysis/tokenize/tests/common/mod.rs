//! A reader for the golden files the OpenNLP JVM wrote (the format is in
//! analysis/golden/jvm/README.md). It is small on purpose: it reads the
//! header, the escaped input and the result lines, and checks nothing
//! else.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// One span as Java reported it.
#[derive(Clone, Debug, PartialEq)]
pub enum GoldenSpan {
    /// `span:` UTF-8 and UTF-16 offsets, and the optional probability and
    /// type that follow.
    Valid { utf8: (u32, u32), utf16: (u32, u32), extra: Vec<String> },
    /// `span-invalid:` an offset between the halves of a surrogate pair,
    /// so only UTF-16 offsets.
    Invalid { utf16: (u32, u32), extra: Vec<String> },
}

/// One golden file.
#[derive(Debug)]
pub struct Golden {
    pub path: PathBuf,
    pub component: String,
    pub whitespace: String,
    pub input: String,
    pub spans: Vec<GoldenSpan>,
    /// The `error:` line, when the Java component threw.
    pub error: Option<String>,
}

impl Golden {
    pub fn name(&self) -> &str {
        self.path.file_stem().and_then(|s| s.to_str()).unwrap()
    }
}

/// The golden directory of this crate.
pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// Every golden file of directory `dir`, sorted by name.
pub fn read_dir(dir: &str) -> Vec<Golden> {
    let dir = golden_dir().join(dir);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no golden files in {}", dir.display());
    paths.into_iter().map(|p| read(&p)).collect()
}

/// Reads one golden file.
pub fn read(path: &Path) -> Golden {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut g = Golden {
        path: path.to_path_buf(),
        component: String::new(),
        whitespace: String::new(),
        input: String::new(),
        spans: Vec::new(),
        error: None,
    };
    let mut saw_input = false;
    for line in text.lines() {
        let (key, value) = line.split_once(": ").unwrap_or_else(|| (line.trim_end_matches(':'), ""));
        match key {
            "component" => g.component = value.to_string(),
            "whitespace" => g.whitespace = value.to_string(),
            "input" => {
                g.input = unescape(value);
                saw_input = true;
            }
            "span" => {
                let f: Vec<&str> = value.split(' ').collect();
                let n = |i: usize| f[i].parse::<u32>().unwrap();
                let extra = f[4..].iter().map(|s| s.to_string()).collect();
                g.spans.push(GoldenSpan::Valid { utf8: (n(0), n(1)), utf16: (n(2), n(3)), extra });
            }
            "span-invalid" => {
                let f: Vec<&str> = value.split(' ').collect();
                let n = |i: usize| f[i].parse::<u32>().unwrap();
                let extra = f[2..].iter().map(|s| s.to_string()).collect();
                g.spans.push(GoldenSpan::Invalid { utf16: (n(0), n(1)), extra });
            }
            "error" => g.error = Some(unescape(value)),
            _ => {}
        }
    }
    assert!(saw_input, "{}: no input line", path.display());
    g
}

/// Undoes the input escapes: `\\`, `\n`, `\r`, `\t` and `\uXXXX` over
/// UTF-16 code units, so a surrogate pair is two escapes.
pub fn unescape(s: &str) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
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
                units.push(u16::from_str_radix(&hex, 16).unwrap_or_else(|_| panic!("bad escape \\u{hex}")));
            }
            other => panic!("bad escape \\{other:?}"),
        }
    }
    String::from_utf16(&units).expect("an input of whole characters")
}
