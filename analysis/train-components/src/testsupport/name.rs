//! The name format: one sentence per line, tokens separated by
//! whitespace, each name between a `<START:type>` (or `<START>`) token
//! and an `<END>` token. An empty line marks the next sample as the
//! first of a new document.

use turbo_analysis::Span;

use super::{java_trim, whitespace_tokens};
use turbo_formats::sample::{DEFAULT_NAME_TYPE, Name, NameSample};
use turbo_formats::{FormatError, Line, Lines, Piece};

const END_TAG: &str = "<END>";

/// Reads name samples from the name format.
#[derive(Clone, Debug)]
pub struct NameReader<'a> {
    lines: Lines<'a>,
    default_type: &'a str,
}

impl<'a> NameReader<'a> {
    /// A reader of `source`, splitting lines at Unicode whitespace.
    pub fn new(source: &'a str) -> Result<NameReader<'a>, FormatError> {
        Ok(NameReader { lines: Lines::new(source)?, default_type: DEFAULT_NAME_TYPE })
    }
}

impl<'a> Iterator for NameReader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut clear = false;
        loop {
            let line = self.lines.next()?;
            if java_trim(line.piece).text.is_empty() {
                clear = true;
                continue;
            }
            return Some(parse_line(line, clear, self.default_type));
        }
    }
}

/// One line of the name format.
fn parse_line<'a>(
    line: Line<'a>,
    clear_adaptive_data: bool,
    default_type: &'a str,
) -> Result<NameSample<'a>, FormatError> {
    let parts: Vec<Piece<'a>> = whitespace_tokens(line.piece).collect();
    let mut tokens = Vec::with_capacity(parts.len());
    let mut names = Vec::new();
    let mut name_type = default_type;
    // The token index and source offset of the open name's start tag.
    let mut open: Option<(u32, u32)> = None;
    for (pi, part) in parts.iter().enumerate() {
        if let Some(kind) = start_tag(part.text) {
            if open.is_some() {
                return Err(FormatError::NestedName { line: line.number, context: context(&parts, pi) });
            }
            open = Some((tokens.len() as u32, part.span.start));
            if let Some(kind) = kind {
                if kind.is_empty() {
                    return Err(FormatError::MissingNameType { line: line.number, context: context(&parts, pi) });
                }
                name_type = kind;
            }
        } else if part.text == END_TAG {
            let Some((start, from)) = open.take() else {
                return Err(FormatError::UnexpectedEnd { line: line.number, context: context(&parts, pi) });
            };
            names.push(Name {
                start,
                end: tokens.len() as u32,
                kind: Some(name_type),
                source: Span::new(from, part.span.end),
            });
        } else {
            tokens.push(part.field());
        }
    }
    NameSample::new(tokens, names, clear_adaptive_data, line.span(), line.number)
}

/// Whether `token` is a whole `<START>` or `<START:type>` tag: `None` if
/// not, `Some(None)` for `<START>` and `Some(Some(type))` for a typed tag,
/// whose type may be empty.
fn start_tag(token: &str) -> Option<Option<&str>> {
    let inner = token.strip_prefix("<START")?.strip_suffix('>')?;
    if inner.is_empty() {
        return Some(None);
    }
    let kind = inner.strip_prefix(':')?;
    let excluded = |c: char| matches!(c, ':' | '>' | ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r');
    (!kind.contains(excluded)).then_some(Some(kind))
}

/// Up to two parts before the one at `index`, that part between `###`,
/// and up to two parts after it.
fn context(parts: &[Piece<'_>], index: usize) -> String {
    let mut out = String::new();
    if index > 1 {
        out.push_str(parts[index - 2].text);
        out.push(' ');
    }
    if index > 0 {
        out.push_str(parts[index - 1].text);
        out.push(' ');
    }
    out.push_str("###");
    out.push_str(parts[index].text);
    out.push_str("### ");
    if index + 1 < parts.len() {
        out.push_str(parts[index + 1].text);
        out.push(' ');
    }
    if index + 2 < parts.len() {
        out.push_str(parts[index + 2].text);
    }
    out
}
