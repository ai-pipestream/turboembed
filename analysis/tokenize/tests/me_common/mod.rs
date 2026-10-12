//! Reading the golden files the JVM wrote for TokenizerME
//! (analysis/golden/jvm/README.md has the format), and the one rule that
//! turns a Java split between surrogate halves into what the Rust port
//! produces.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_tokenize::me::TokenizerModel;
use turbo_unicode::utf16::utf16_to_byte;

pub fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

pub fn golden_dir() -> PathBuf {
    crate_dir().join("tests/golden")
}

/// The test model file of that name.
pub fn model(name: &str) -> TokenizerModel {
    let path = crate_dir().join("tests/models").join(name);
    TokenizerModel::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Undoes the escaping of `input:` and `features:` values: `\\`, `\n`,
/// `\r`, `\t` and `\uXXXX`, giving UTF-16 code units.
pub fn unescape(s: &str) -> Vec<u16> {
    let mut out = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => out.push(u16::from(b'\\')),
            Some('n') => out.push(u16::from(b'\n')),
            Some('r') => out.push(u16::from(b'\r')),
            Some('t') => out.push(u16::from(b'\t')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&hex, 16).unwrap_or_else(|_| panic!("bad escape \\u{hex}")));
            }
            other => panic!("bad escape \\{other:?}"),
        }
    }
    out
}

/// An escaped value as a Rust string; it must hold no lone surrogate.
pub fn unescape_str(s: &str) -> String {
    String::from_utf16(&unescape(s)).expect("an input without lone surrogates")
}

/// Code units as Java's modified UTF-8, the bytes a feature is compared as.
pub fn modified_utf8(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for &u in units {
        turbo_unicode::modified_utf8::push_modified_utf8(&mut out, u32::from(u));
    }
    out
}

/// One span line: UTF-8 offsets when the span is valid, UTF-16 offsets
/// always, and the probability.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenSpan {
    pub utf8: Option<(u32, u32)>,
    pub utf16: (usize, usize),
    pub prob: f64,
}

/// A tokenizer golden file.
#[derive(Debug)]
pub struct Golden {
    pub path: PathBuf,
    pub headers: Vec<(String, String)>,
    pub input: String,
    pub spans: Vec<GoldenSpan>,
}

impl Golden {
    pub fn read(path: &Path) -> Golden {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut headers = Vec::new();
        let mut input = None;
        let mut spans = Vec::new();
        for line in text.lines() {
            let (key, value) = line.split_once(": ").or_else(|| line.strip_suffix(':').map(|k| (k, ""))).unwrap();
            match key {
                "input" => input = Some(unescape_str(value)),
                "span" | "span-invalid" => {
                    let f: Vec<&str> = value.split(' ').collect();
                    let n = |s: &str| s.parse::<usize>().unwrap();
                    spans.push(if key == "span" {
                        GoldenSpan {
                            utf8: Some((n(f[0]) as u32, n(f[1]) as u32)),
                            utf16: (n(f[2]), n(f[3])),
                            prob: f[4].parse().unwrap(),
                        }
                    } else {
                        GoldenSpan { utf8: None, utf16: (n(f[0]), n(f[1])), prob: f[2].parse().unwrap() }
                    });
                }
                "error" => panic!("{}: the JVM failed: {value}", path.display()),
                _ => headers.push((key.to_string(), value.to_string())),
            }
        }
        Golden { path: path.to_path_buf(), headers, input: input.expect("an input line"), spans }
    }

    pub fn header(&self, key: &str) -> &str {
        self.headers.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).unwrap_or_else(|| panic!("no {key}"))
    }
}

/// What the Rust port produces for spans the JVM wrote: a span whose end
/// falls between the two halves of a pair is joined with the span after
/// it, whose start is the same point, and their probabilities multiply,
/// since the port does not take that split and goes on multiplying the
/// token's decisions. Gives (UTF-8 start, UTF-8 end, probability).
pub fn expected_spans(input: &str, spans: &[GoldenSpan]) -> Vec<(u32, u32, f64)> {
    let byte = |unit: usize| utf16_to_byte(input, unit).map(|b| b as u32);
    let mut out = Vec::new();
    let mut open: Option<(u32, f64)> = None;
    for s in spans {
        let joined = open.is_some();
        let (start, prob) = match open.take() {
            Some((start, prob)) => {
                assert!(byte(s.utf16.0).is_none(), "a joined span continues where the last one stopped");
                (start, prob * s.prob)
            }
            None => (byte(s.utf16.0).expect("a span start between two halves without a split before it"), s.prob),
        };
        match byte(s.utf16.1) {
            Some(end) => {
                if let (Some(utf8), false) = (s.utf8, joined) {
                    assert_eq!(utf8, (start, end), "the UTF-8 offsets agree with the UTF-16 ones");
                }
                out.push((start, end, prob));
            }
            None => open = Some((start, prob)),
        }
    }
    assert!(open.is_none(), "no span ends between two halves at the end of the text");
    out
}

/// One unit of a contexts file: its UTF-16 range in the input and, per
/// position, the index, the features as code units and P(T), P(F).
#[derive(Debug)]
pub struct ContextUnit {
    pub utf16: (usize, usize),
    pub positions: Vec<(usize, Vec<Vec<u16>>, [f64; 2])>,
}

/// A contexts file: the model, the outcome names in the order the
/// probabilities are written, and each input with its units.
#[derive(Debug)]
pub struct Contexts {
    pub headers: Vec<(String, String)>,
    pub outcomes: Vec<String>,
    pub inputs: Vec<(String, Vec<ContextUnit>)>,
}

impl Contexts {
    pub fn read(path: &Path) -> Contexts {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut c = Contexts { headers: Vec::new(), outcomes: Vec::new(), inputs: Vec::new() };
        for line in text.lines() {
            let (key, value) = line.split_once(": ").unwrap_or_else(|| (line.trim_end_matches(':'), ""));
            match key {
                "outcomes" => c.outcomes = value.split(' ').map(String::from).collect(),
                "input" => c.inputs.push((unescape_str(value), Vec::new())),
                "unit" => {
                    let (a, b) = value.split_once(' ').unwrap();
                    let unit = ContextUnit { utf16: (a.parse().unwrap(), b.parse().unwrap()), positions: Vec::new() };
                    c.inputs.last_mut().unwrap().1.push(unit);
                }
                "at" => {
                    let unit = c.inputs.last_mut().unwrap().1.last_mut().unwrap();
                    unit.positions.push((value.parse().unwrap(), Vec::new(), [f64::NAN; 2]));
                }
                "features" => {
                    let position = c.inputs.last_mut().unwrap().1.last_mut().unwrap().positions.last_mut().unwrap();
                    position.1 = value.split('\t').map(unescape).collect();
                }
                "probs" => {
                    let position = c.inputs.last_mut().unwrap().1.last_mut().unwrap().positions.last_mut().unwrap();
                    let (a, b) = value.split_once(' ').unwrap();
                    position.2 = [a.parse().unwrap(), b.parse().unwrap()];
                }
                _ => c.headers.push((key.to_string(), value.to_string())),
            }
        }
        c
    }

    pub fn header(&self, key: &str) -> &str {
        self.headers.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).unwrap_or_else(|| panic!("no {key}"))
    }
}

pub const LANGUAGES: [&str; 5] = ["en", "de", "es", "fr", "pt"];

/// Asserts two probabilities agree within 1e-9.
#[track_caller]
pub fn assert_close(got: f64, want: f64, what: &dyn std::fmt::Display) {
    assert!((got - want).abs() <= 1e-9, "{what}: probability {got}, the JVM gave {want}");
}
