//! A small reader for the golden files the JVM generator writes (format in
//! analysis/golden/jvm/README.md): header lines, the escaped input, and
//! `span:` lines with UTF-8 and UTF-16 offsets. Only what the newline
//! files hold is read; a line this reader does not know fails the test.

use std::path::{Path, PathBuf};

use turbo_sentdetect::WhitespaceMode;

/// One span as Java reported it, with both kinds of offsets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GoldenSpan {
    pub utf8: (u32, u32),
    pub utf16: (usize, usize),
}

/// One golden file.
#[derive(Debug)]
pub struct Golden {
    pub component: String,
    pub whitespace: WhitespaceMode,
    pub input: String,
    pub spans: Vec<GoldenSpan>,
}

/// The files of one golden directory of this crate, sorted by name.
pub fn files(dir: &str) -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(dir);
    let mut out: Vec<PathBuf> = std::fs::read_dir(&root)
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
    let (mut component, mut whitespace, mut input) = (None, None, None);
    let mut spans = Vec::new();
    for line in text.lines() {
        let (key, value) = line.split_once(": ").or_else(|| line.strip_suffix(':').map(|k| (k, ""))).ok_or(line)?;
        match key {
            "component" => component = Some(value.to_string()),
            "whitespace" => {
                whitespace = Some(match value {
                    "unicode" => WhitespaceMode::Unicode,
                    "legacy" => WhitespaceMode::Legacy,
                    other => return Err(format!("whitespace mode {other:?}")),
                })
            }
            "input" => input = Some(unescape(value)?),
            "span" => {
                let n: Vec<usize> = value
                    .split(' ')
                    .map(|x| x.parse().map_err(|_| format!("span {value:?}")))
                    .collect::<Result<_, _>>()?;
                let [a, b, c, d] = n[..] else { return Err(format!("span {value:?} needs four offsets")) };
                spans.push(GoldenSpan { utf8: (a as u32, b as u32), utf16: (c, d) });
            }
            "opennlp" | "jdk" => {}
            other => return Err(format!("unknown line {other:?}")),
        }
    }
    Ok(Golden {
        component: component.ok_or("no component")?,
        whitespace: whitespace.ok_or("no whitespace mode")?,
        input: input.ok_or("no input")?,
        spans,
    })
}

/// Undoes the generator's escapes: `\\`, `\n`, `\r`, `\t` and `\uXXXX`
/// per UTF-16 code unit, a character outside the BMP being two of them.
pub fn unescape(s: &str) -> Result<String, String> {
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
    String::from_utf16(&units).map_err(|e| e.to_string())
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
    assert_eq!(g.spans, [GoldenSpan { utf8: (0, 3), utf16: (0, 3) }]);
    assert_eq!(parse("component: newline\nwhitespace: unicode\ninput: \n").unwrap().input, "");
    assert!(parse("component: newline\nwhitespace: unicode\ninput: \\uD83D\n").is_err());
    assert!(parse("component: newline\nwhitespace: unicode\ninput: x\nerror: boom\n").is_err());
}
