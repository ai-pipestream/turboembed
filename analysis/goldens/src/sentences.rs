//! The sentence segmentation conformance suites under
//! `data/conformance/sentences` (README.md there names the sources and
//! their licences), read into typed cases.
//!
//! - [`SentenceCase`]: a text and the sentences a reference segmenter's
//!   test expects, from the Pragmatic Segmenter specs and the pySBD tests.
//!   A `.cases` file holds a `#` comment header, then one block per case:
//!
//!   ```text
//!   case pragmatic-segmenter/english_spec/001
//!   lang en
//!   name Golden Rules: Simple period to end sentence #001
//!   option doc_type=pdf
//!   text Hello World. My name is Jonas.
//!   sentence Hello World.
//!   sentence My name is Jonas.
//!   end
//!   ```
//!
//!   `name` and `option` are optional and `option` may repeat; the values
//!   of `name`, `text` and `sentence` use the escapes of [`crate::escape`].
//! - [`BreakTestCase`]: a line of Unicode's `SentenceBreakTest.txt`, the
//!   code points and the positions UAX #29 breaks at.
//! - [`ExpectedFailure`]: a case a detector is known to get wrong, with the
//!   reason, from a `.failures` file: `<case id><TAB><reason>` per line.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::data_dir;
use crate::escape::unescape;

/// One case of a sentence suite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentenceCase {
    /// `<suite>/<source file>/<number>`, unique across the suites.
    pub id: String,
    /// The language tag the reference segmenter was given.
    pub lang: String,
    /// What the source calls the test, when it names it.
    pub name: String,
    /// The segmenter options the source's test passes, such as
    /// `doc_type=pdf` or `clean=true`, and `skipped-upstream` or
    /// `xfail-upstream` for a test the source skips or expects to fail.
    pub options: Vec<String>,
    pub text: String,
    /// The expected sentences, as the source writes them.
    pub sentences: Vec<String>,
}

impl SentenceCase {
    /// Whether the expected sentences are the text cut into pieces, every
    /// character of the text in one of them, once whitespace is ignored.
    /// A case whose expected output also rewrites the text (Pragmatic
    /// Segmenter's cleaning removes dot leaders, for example) is not.
    pub fn is_extractive(&self) -> bool {
        let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        strip(&self.text) == self.sentences.iter().map(|s| strip(s)).collect::<String>()
    }
}

/// A sentence suite file that does not read, with the line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuiteError {
    pub path: PathBuf,
    pub line: usize,
    pub message: String,
}

impl fmt::Display for SuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.path.display(), self.line, self.message)
    }
}

impl std::error::Error for SuiteError {}

/// The conformance data, `data/conformance/sentences`.
pub fn sentences_dir() -> PathBuf {
    data_dir().join("conformance").join("sentences")
}

/// Reads one `.cases` file.
pub fn read_cases(path: &Path) -> Result<Vec<SentenceCase>, SuiteError> {
    let text = std::fs::read_to_string(path).map_err(|e| SuiteError {
        path: path.to_owned(),
        line: 0,
        message: e.to_string(),
    })?;
    parse_cases(&text).map_err(|(line, message)| SuiteError { path: path.to_owned(), line, message })
}

/// Parses the text of a `.cases` file; an error carries the line number.
pub fn parse_cases(text: &str) -> Result<Vec<SentenceCase>, (usize, String)> {
    let mut out = Vec::new();
    let mut open: Option<SentenceCase> = None;
    for (n, line) in text.split('\n').enumerate() {
        let n = n + 1;
        if line.is_empty() || line.starts_with('#') {
            if open.is_some() && line.starts_with('#') {
                return Err((n, "a comment inside a case".into()));
            }
            continue;
        }
        let (key, value) = line.split_once(' ').unwrap_or((line, ""));
        let value_text = || unescape(value).map_err(|e| (n, format!("{key}: {e}")));
        match (key, open.as_mut()) {
            ("case", None) => {
                if value.is_empty() {
                    return Err((n, "a case without an id".into()));
                }
                open = Some(SentenceCase {
                    id: value.to_owned(),
                    lang: String::new(),
                    name: String::new(),
                    options: Vec::new(),
                    text: String::new(),
                    sentences: Vec::new(),
                });
            }
            ("case", Some(c)) => return Err((n, format!("case {} has no end", c.id))),
            ("lang", Some(c)) if c.lang.is_empty() && !value.is_empty() => c.lang = value.to_owned(),
            ("name", Some(c)) if c.name.is_empty() => c.name = value_text()?,
            ("option", Some(c)) if !value.is_empty() => c.options.push(value.to_owned()),
            ("text", Some(c)) if c.text.is_empty() => c.text = value_text()?,
            ("sentence", Some(c)) => c.sentences.push(value_text()?),
            ("end", Some(_)) if value.is_empty() => {
                let c = open.take().expect("open");
                if c.lang.is_empty() {
                    return Err((n, format!("case {} has no lang", c.id)));
                }
                if out.iter().any(|o: &SentenceCase| o.id == c.id) {
                    return Err((n, format!("case {} twice", c.id)));
                }
                out.push(c);
            }
            _ => return Err((n, format!("unexpected line {line:?}"))),
        }
    }
    match open {
        Some(c) => Err((text.split('\n').count(), format!("case {} has no end", c.id))),
        None => Ok(out),
    }
}

/// Every case of a suite (`pragmatic-segmenter`, `pysbd`), its files in
/// name order.
pub fn sentence_suite(suite: &str) -> Vec<SentenceCase> {
    let dir = sentences_dir().join(suite);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "cases"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no cases in {}", dir.display());
    files.iter().flat_map(|p| read_cases(p).unwrap_or_else(|e| panic!("{e}"))).collect()
}

/// One line of `SentenceBreakTest.txt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BreakTestCase {
    /// The line number in the file.
    pub line: usize,
    pub text: String,
    /// The byte offsets of the breaks inside the text: the one at its
    /// start and the one at its end, which every line has, are left out.
    pub breaks: Vec<u32>,
    /// The rule that decides each position, from the line's comment: one
    /// per character boundary, the start and the end of the text included
    /// (`1.0`, then one per position between characters, then `2.0`).
    pub rules: Vec<String>,
}

impl BreakTestCase {
    /// The rule that decides the position at byte offset `at`.
    pub fn rule_at(&self, at: u32) -> &str {
        let index = self.text[..at as usize].chars().count();
        self.rules.get(index).map_or("?", String::as_str)
    }
}

/// Reads `SentenceBreakTest.txt` (or `WordBreakTest.txt`, which has the
/// same format): `÷` and `×` between code points in hex, `#` comments.
pub fn read_break_test(path: &Path) -> Result<Vec<BreakTestCase>, SuiteError> {
    let err = |line, message: String| SuiteError { path: path.to_owned(), line, message };
    let text = std::fs::read_to_string(path).map_err(|e| err(0, e.to_string()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let (content, comment) = line.split_once('#').unwrap_or((line, ""));
        let content = content.trim();
        if content.is_empty() {
            continue;
        }
        let rules = comment.split('[').skip(1).filter_map(|r| r.split_once(']').map(|(id, _)| id.to_owned())).collect();
        let mut case = BreakTestCase { line: n + 1, text: String::new(), breaks: Vec::new(), rules };
        let mut tokens = content.split_whitespace().peekable();
        if tokens.next() != Some("÷") {
            return Err(err(n + 1, "a line that does not start with ÷".into()));
        }
        while let Some(cp) = tokens.next() {
            let c = u32::from_str_radix(cp, 16)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| err(n + 1, format!("not a code point: {cp}")))?;
            case.text.push(c);
            match tokens.next() {
                Some("÷") if tokens.peek().is_some() => case.breaks.push(case.text.len() as u32),
                Some("÷") => {}
                Some("×") if tokens.peek().is_some() => {}
                other => return Err(err(n + 1, format!("expected ÷ or × after {cp}, found {other:?}"))),
            }
        }
        if case.rules.len() != case.text.chars().count() + 1 {
            return Err(err(
                n + 1,
                format!("{} rules in the comment for {} positions", case.rules.len(), case.text.chars().count() + 1),
            ));
        }
        out.push(case);
    }
    Ok(out)
}

/// A case a detector is known to get wrong, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedFailure {
    pub id: String,
    pub reason: String,
}

/// Reads a `.failures` file: `#` comments, then `<id><TAB><reason>` per
/// line. A missing file is an empty list.
pub fn read_expected_failures(path: &Path) -> Result<Vec<ExpectedFailure>, SuiteError> {
    let err = |line, message: String| SuiteError { path: path.to_owned(), line, message };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(err(0, e.to_string())),
    };
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (id, reason) = line.split_once('\t').ok_or_else(|| err(n + 1, "no tab between id and reason".into()))?;
        if id.is_empty() || reason.is_empty() {
            return Err(err(n + 1, "an empty id or reason".into()));
        }
        out.push(ExpectedFailure { id: id.to_owned(), reason: reason.to_owned() });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_parse() {
        let text = "# header\n\ncase s/f/001\nlang en\nname A \\u00e9 b\noption clean=true\ntext One. Two\\n\
                    \\u0020\nsentence One.\nsentence Two\nend\n";
        let cases = parse_cases(text).unwrap();
        assert_eq!(cases.len(), 1);
        let c = &cases[0];
        assert_eq!((c.id.as_str(), c.lang.as_str(), c.name.as_str()), ("s/f/001", "en", "A é b"));
        assert_eq!(c.options, ["clean=true"]);
        assert_eq!(c.text, "One. Two\n ");
        assert_eq!(c.sentences, ["One.", "Two"]);
        assert!(c.is_extractive());
    }

    #[test]
    fn malformed_cases_name_their_line() {
        assert_eq!(parse_cases("case a\nlang en\n").unwrap_err().1, "case a has no end");
        assert_eq!(parse_cases("case a\ntext x\nend\n").unwrap_err(), (3, "case a has no lang".into()));
        assert_eq!(parse_cases("case a\nlang en\nbogus\nend").unwrap_err().0, 3);
        assert_eq!(parse_cases("case a\nlang en\ntext \\q\nend").unwrap_err().0, 3);
        assert!(parse_cases("case a\nlang en\nend\ncase a\nlang en\nend\n").unwrap_err().1.contains("twice"));
        assert_eq!(parse_cases("lang en\n").unwrap_err().0, 1);
    }

    #[test]
    fn rewriting_cases_are_not_extractive() {
        let c = SentenceCase {
            id: "x".into(),
            lang: "en".into(),
            name: String::new(),
            options: Vec::new(),
            text: "About Me......5".into(),
            sentences: vec!["About Me".into()],
        };
        assert!(!c.is_extractive());
    }

    #[test]
    fn the_checked_in_suites_read() {
        let pragmatic = sentence_suite("pragmatic-segmenter");
        let pysbd = sentence_suite("pysbd");
        assert_eq!((pragmatic.len(), pysbd.len()), (585, 471));
        let uax = read_break_test(&sentences_dir().join("uax29/SentenceBreakTest-18.0.0.txt")).unwrap();
        assert_eq!(uax.len(), 512);
        assert_eq!(uax[0].text, "\r\r");
        assert_eq!(uax[0].breaks, [1]);
        assert_eq!(uax[0].rules, ["1.0", "4.0", "2.0"]);
        assert_eq!(uax[0].rule_at(1), "4.0");
    }
}
