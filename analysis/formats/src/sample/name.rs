//! Name samples: `opennlp.tools.namefind.NameSample`.

use std::cmp::Ordering;
use std::fmt::{self, Write as _};

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::source::Field;

/// The type the name format gives a name whose start tag has none
/// (`NameSample.DEFAULT_TYPE`).
pub const DEFAULT_NAME_TYPE: &str = "default";

/// One name of a [`NameSample`]: tokens `start..end` of the sample, and
/// the name's type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Name<'a> {
    /// The first token of the name.
    pub start: u32,
    /// One past the last token; equal to `start` for an empty name, which
    /// the format allows (`<START> <END>`).
    pub end: u32,
    /// The type, `None` only for a name built without one; a name the
    /// format reads always has one.
    pub kind: Option<&'a str>,
    /// The bytes of the source from the start of the start tag to the end
    /// of the end tag; empty, at the start of the first token, for a name
    /// not read from a source.
    pub source: Span,
}

impl<'a> Name<'a> {
    /// A name of tokens `start..end`, not read from a source.
    pub fn new(start: u32, end: u32, kind: Option<&'a str>) -> Name<'a> {
        assert!(start <= end, "a name ends at or after its start");
        Name { start, end, kind, source: Span::default() }
    }

    /// The number of tokens.
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// `Span.compareTo`: by start, then the longer name first, then by
    /// type in Java's string order, a name with a type before one
    /// without.
    pub fn java_cmp(&self, other: &Name<'_>) -> Ordering {
        self.start.cmp(&other.start).then(other.end.cmp(&self.end)).then_with(|| match (self.kind, other.kind) {
            (None, None) => Ordering::Equal,
            (Some(a), Some(b)) => turbo_jvm::chars::compare(a, b),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
        })
    }

    /// `Span.toString`: `[start..end)`, then a space and the type when
    /// there is one.
    pub fn java_string(&self) -> String {
        match self.kind {
            Some(k) => format!("[{}..{}) {k}", self.start, self.end),
            None => format!("[{}..{})", self.start, self.end),
        }
    }
}

/// A sentence with its names: the tokens and the names over them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameSample<'a> {
    tokens: Vec<Field<'a>>,
    names: Vec<Name<'a>>,
    clear_adaptive_data: bool,
    source: Span,
}

impl<'a> NameSample<'a> {
    /// A sample of `tokens` with `names`, which are sorted as
    /// `NameSample`'s constructor sorts them ([`Name::java_cmp`], stable).
    /// `source` is the bytes the sample was read from.
    ///
    /// # Errors
    ///
    /// [`FormatError::OverlappingNames`] when a name starts before the one
    /// sorted ahead of it ends, as the constructor throws; `line` is the
    /// source line to report. [`FormatError::NameOutsideTokens`] when a name
    /// reaches past the tokens.
    pub fn new(
        tokens: Vec<Field<'a>>,
        mut names: Vec<Name<'a>>,
        clear_adaptive_data: bool,
        source: Span,
        line: u32,
    ) -> Result<NameSample<'a>, FormatError> {
        names.sort_by(|a, b| a.java_cmp(b));
        for pair in names.windows(2) {
            if pair[1].start < pair[0].end {
                return Err(FormatError::OverlappingNames {
                    line,
                    first: pair[0].java_string(),
                    second: pair[1].java_string(),
                    sentence: java_list(tokens.iter().map(Field::as_str)),
                });
            }
        }
        if let Some(n) = names.iter().find(|n| n.end as usize > tokens.len()) {
            return Err(FormatError::NameOutsideTokens { name: n.java_string(), tokens: tokens.len() });
        }
        Ok(NameSample { tokens, names, clear_adaptive_data, source })
    }

    pub fn tokens(&self) -> &[Field<'a>] {
        &self.tokens
    }

    /// The names, in the order `NameSample` keeps them.
    pub fn names(&self) -> &[Name<'a>] {
        &self.names
    }

    /// Whether the sample follows an empty line, which tells the name
    /// finder's feature generators to clear their adaptive data.
    pub fn clear_adaptive_data(&self) -> bool {
        self.clear_adaptive_data
    }

    /// The bytes of the source the sample was read from: its line.
    pub fn source(&self) -> Span {
        self.source
    }

    /// The texts of the tokens of `name`.
    pub fn name_tokens(&self, name: &Name<'_>) -> &[Field<'a>] {
        &self.tokens[name.start as usize..name.end as usize]
    }
}

/// `List.toString` of strings: `[a, b]`.
pub(crate) fn java_list<'s>(items: impl Iterator<Item = &'s str>) -> String {
    let mut out = String::from("[");
    for (i, s) in items.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(s);
    }
    out.push(']');
    out
}

/// `NameSample.toString`, the name format's line: an empty line first
/// when the sample clears adaptive data, then the tokens separated by
/// spaces with `<START:type>` and `<END>` around the names.
///
/// It reproduces two details of OpenNLP's: a sample of one token keeps
/// the space after it (`x ` and, with a name over it, `<START:t> x  <END>`
/// with two spaces), and the tags of a token are written name by name in
/// the sorted order, so an empty name and a longer one starting at the
/// same token write the longer one's start tag first.
impl fmt::Display for NameSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        if self.clear_adaptive_data {
            out.push('\n');
        }
        for (i, token) in self.tokens.iter().enumerate() {
            let i = i as u32;
            for name in &self.names {
                if name.start == i {
                    match name.kind {
                        None => out.push_str("<START> "),
                        Some(k) => {
                            let _ = write!(out, "<START:{k}> ");
                        }
                    }
                }
                if name.end == i {
                    out.push_str("<END> ");
                }
            }
            out.push_str(&token.text);
            out.push(' ');
        }
        if self.tokens.len() > 1 {
            out.pop();
        }
        for name in &self.names {
            if name.end as usize == self.tokens.len() {
                out.push_str(" <END>");
            }
        }
        f.write_str(&out)
    }
}
