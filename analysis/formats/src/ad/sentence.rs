//! The sentences of an Arvores Deitadas text: `ADSentenceStream` and
//! `SentenceParser.parse`.
//!
//! A sentence is the lines between an `<s ...>` line and `</s>` (or
//! `</ext>`), without the lines that start with `A1`. Among them, a line
//! that starts with `SOURCE` gives the metadata's source part, and the
//! line after it holds an id, a space and the sentence's text; the tree
//! lines follow, up to an empty line, `</s>` or `&&`. A `&&` before the
//! `SOURCE` line marks another analysis of the sentence before: it keeps
//! that sentence's text and metadata.

use std::rc::Rc;

use turbo_analysis::Span;

use super::tree::{Tree, TreeBuilder, is_ws};
use crate::source::{Field, Line, Lines, Piece, java_trim};

const SENTENCE_TAG: &str = "s";
const TEXT_TAG: &str = "ext";
const TITLE_TAG: &str = "t";
const BOX_TAG: &str = "caixa";
const PARAGRAPH_TAG: &str = "p";
const CLOSING_GUILLEMET: char = '\u{BB}';

/// One sentence: its text, metadata and tree. The text and the metadata
/// are what the parser kept last, so they can be another sentence's, or
/// missing (Java's `null`) before any sentence set them.
#[derive(Clone, Debug)]
pub(crate) struct AdSentence<'a> {
    pub text: Option<Field<'a>>,
    /// The id before the text, ` p=` and the paragraph number, ` title`
    /// inside a title, ` box` inside a box, and the `SOURCE` line after
    /// its first seven UTF-16 units.
    pub metadata: Option<Rc<str>>,
    pub tree: Tree<'a>,
    /// The sentence's lines, from the first to the end of the last.
    pub source: Span,
    /// The number of its first line.
    pub line: u32,
}

/// What one read of the sentence stream gives.
#[derive(Clone, Debug)]
pub(crate) enum SentenceRead<'a> {
    Sentence(AdSentence<'a>),
    /// A sentence the parser gave up on: no `SOURCE` line, a `SOURCE` line
    /// of six characters, or no line after it. Java's stream returns
    /// `null`, which its readers take for the end of the text; it logs
    /// "Caught exception for the given sentence" for the last two.
    Failed,
}

/// `ADSentenceStream` over the lines of a text.
pub(crate) struct SentenceStream<'a> {
    lines: Lines<'a>,
    paragraph: i32,
    title: bool,
    in_box: bool,
    /// The text and metadata the parser kept last (its `text` and `meta`
    /// fields).
    text: Option<Field<'a>>,
    metadata: Option<Rc<str>>,
    buffer: Vec<Line<'a>>,
}

impl<'a> SentenceStream<'a> {
    pub fn new(lines: Lines<'a>) -> SentenceStream<'a> {
        SentenceStream {
            lines,
            paragraph: 0,
            title: false,
            in_box: false,
            text: None,
            metadata: None,
            buffer: Vec::new(),
        }
    }

    /// The next sentence, or `None` at the end of the text. A sentence cut
    /// off by the end of the text is read with the lines it has.
    pub fn read(&mut self) -> Option<SentenceRead<'a>> {
        self.buffer.clear();
        let mut started = false;
        loop {
            let Some(line) = self.lines.next() else {
                return (started && !self.buffer.is_empty()).then(|| self.parse());
            };
            let text = line.text();
            if started {
                if is_closing_tag(text, SENTENCE_TAG) || is_closing_tag(text, TEXT_TAG) {
                    started = false;
                } else if !text.starts_with("A1") {
                    self.buffer.push(line);
                }
            } else if is_opening_tag(text, SENTENCE_TAG) {
                started = true;
            } else if is_opening_tag(text, PARAGRAPH_TAG) {
                self.paragraph = self.paragraph.wrapping_add(1);
            } else if is_opening_tag(text, TITLE_TAG) {
                self.title = true;
            } else if is_closing_tag(text, TITLE_TAG) {
                self.title = false;
            } else if is_opening_tag(text, TEXT_TAG) {
                self.paragraph = 0;
            } else if is_opening_tag(text, BOX_TAG) {
                self.in_box = true;
            } else if is_closing_tag(text, BOX_TAG) {
                self.in_box = false;
            }
            if !started && !self.buffer.is_empty() {
                return Some(self.parse());
            }
        }
    }

    /// `SentenceParser.parse` of the buffered lines.
    fn parse(&mut self) -> SentenceRead<'a> {
        let first = self.buffer[0];
        let last = *self.buffer.last().expect("not empty");
        let mut lines = self.buffer.iter().copied();
        let mut line = lines.next().expect("not empty");
        let mut same = false;
        while !line.text().starts_with("SOURCE") {
            if line.text() == "&&" {
                same = true;
                break;
            }
            match lines.next() {
                Some(l) => line = l,
                None => return SentenceRead::Failed,
            }
        }
        if !same {
            let Some(from_source) = after_seven_units(line.text()) else {
                return SentenceRead::Failed;
            };
            let Some(text_line) = lines.next() else {
                return SentenceRead::Failed;
            };
            let t = text_line.text();
            let space = t.find(' ');
            let after = match space {
                Some(s) => text_line.piece.slice(s + 1, t.len()),
                None => text_line.piece,
            };
            self.text = Some(fix_punctuation(java_trim(after)));
            if let Some(s) = space.filter(|&s| s > 0) {
                let title = if self.title { " title" } else { "" };
                let in_box = if self.in_box { " box" } else { "" };
                let meta = format!("{} p={}{title}{in_box}{from_source}", &t[..s], self.paragraph);
                self.metadata = Some(meta.into());
            }
        }
        let mut next = lines.next();
        while next.is_some_and(|l| l.text().starts_with("###")) {
            next = lines.next();
        }
        let mut tree = TreeBuilder::new();
        while let Some(l) = next {
            let t = l.text();
            if t.is_empty() || t.starts_with("</s>") || t == "&&" {
                break;
            }
            tree.line(l);
            next = lines.next();
        }
        SentenceRead::Sentence(AdSentence {
            text: self.text.clone(),
            metadata: self.metadata.clone(),
            tree: tree.finish(),
            source: Span::new(first.span().start, last.span().end),
            line: first.number,
        })
    }
}

/// `line.substring(7)` of a `SOURCE` line, or `None` when it has fewer
/// than seven UTF-16 units. A cut through a character outside the Basic
/// Multilingual Plane, which leaves Java half of it, leaves U+FFFD here.
fn after_seven_units(line: &str) -> Option<std::borrow::Cow<'_, str>> {
    let mut units = 0;
    for (i, c) in line.char_indices() {
        if units == 7 {
            return Some(line[i..].into());
        }
        units += c.len_utf16();
        if units > 7 {
            return Some(format!("\u{FFFD}{}", &line[i + c.len_utf8()..]).into());
        }
    }
    (units == 7).then(|| "".into())
}

/// `fixPunctuation`: the whitespace between `»` and a `.` or `,` after it
/// removed.
fn fix_punctuation(text: Piece<'_>) -> Field<'_> {
    let t = text.text;
    if !t.contains(CLOSING_GUILLEMET) {
        return text.field();
    }
    let mut fixed = String::with_capacity(t.len());
    let mut i = 0;
    while let Some(c) = t[i..].chars().next() {
        fixed.push(c);
        i += c.len_utf8();
        if c == CLOSING_GUILLEMET {
            let next = t[i..].char_indices().find(|&(_, c)| !is_ws(c)).map_or(t.len(), |(o, _)| i + o);
            if next > i && matches!(t.as_bytes().get(next), Some(b'.' | b',')) {
                i = next;
            }
        }
    }
    Field::made(fixed, text.span)
}

/// `isOpeningTag`: the whole line is `<name>` or `<name` followed by
/// whitespace and attributes without `>`, then `>`.
fn is_opening_tag(line: &str, name: &str) -> bool {
    let b = line.as_bytes();
    let after = name.len() + 1;
    if b.len() < after + 1 || b[0] != b'<' || !line[1..].starts_with(name) || b[b.len() - 1] != b'>' {
        return false;
    }
    let last = b.len() - 1;
    if after < last && !line[after..].chars().next().is_some_and(is_ws) {
        return false;
    }
    b[after..].iter().position(|&c| c == b'>') == Some(last - after)
}

/// `isClosingTag`: the whole line is `</name>`.
fn is_closing_tag(line: &str, name: &str) -> bool {
    line.len() == name.len() + 3 && line.starts_with("</") && line[2..].starts_with(name) && line.ends_with('>')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markup_tags() {
        assert!(is_opening_tag("<s>", "s"));
        assert!(is_opening_tag("<s id=\"1\">", "s"));
        assert!(is_opening_tag("<s\u{a0}x>", "s"));
        assert!(!is_opening_tag("<sx>", "s"));
        assert!(!is_opening_tag("<s a=\">\">", "s"));
        assert!(!is_opening_tag("<s", "s"));
        assert!(is_closing_tag("</caixa>", "caixa"));
        assert!(!is_closing_tag("</s> ", "s"));
    }

    #[test]
    fn source_cut_and_punctuation() {
        assert_eq!(after_seven_units("SOURCE: x").as_deref(), Some(" x"));
        assert_eq!(after_seven_units("SOURCE:").as_deref(), Some(""));
        assert_eq!(after_seven_units("SOURCE"), None);
        assert_eq!(after_seven_units("SOURCE\u{1F600}y").as_deref(), Some("\u{FFFD}y"));
        let t = "«a» , b» .";
        let f = fix_punctuation(Piece { text: t, span: Span::new(0, t.len() as u32) });
        assert_eq!(f.as_str(), "«a», b».");
    }

    #[test]
    fn sentences_keep_text_and_metadata() {
        let src = "<ext>\n<p>\n<s>\nSOURCE: ref=x\nCP1-2 Um texto.\nA1\nSTA:fcl\n=H:n('texto' M S) texto\n</s>\n<s>\n&&\nSTA:fcl\n</s>\n";
        let mut s = SentenceStream::new(Lines::new(src).unwrap());
        let Some(SentenceRead::Sentence(a)) = s.read() else { panic!() };
        assert_eq!(a.text.as_ref().unwrap().as_str(), "Um texto.");
        assert_eq!(a.metadata.as_deref(), Some("CP1-2 p=1 ref=x"));
        assert_eq!(a.tree.leaves().count(), 1);
        let Some(SentenceRead::Sentence(b)) = s.read() else { panic!() };
        assert_eq!(b.metadata.as_deref(), Some("CP1-2 p=1 ref=x"));
        assert_eq!(b.tree.leaves().count(), 0);
        assert!(s.read().is_none());
    }
}
