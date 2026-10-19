//! Penn Treebank bracket reading and token escaping (`Parse.parseParse`, `Parse.encodeToken`).

use std::borrow::Cow;
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::head_rules::GapLabeler;

use super::error::ParseError;
use super::{INC_NODE, Parse, TOK_NODE, TOP_NODE, TreeBuilder, checked_text_len};

/// Left round bracket token (`Parse.BRACKET_LRB`).
pub const BRACKET_LRB: &str = "(";
/// Right round bracket token (`Parse.BRACKET_RRB`).
pub const BRACKET_RRB: &str = ")";
/// Left curly bracket token (`Parse.BRACKET_LCB`).
pub const BRACKET_LCB: &str = "{";
/// Right curly bracket token (`Parse.BRACKET_RCB`).
pub const BRACKET_RCB: &str = "}";
/// Left square bracket token (`Parse.BRACKET_LSB`).
pub const BRACKET_LSB: &str = "[";
/// Right square bracket token (`Parse.BRACKET_RSB`).
pub const BRACKET_RSB: &str = "]";

/// The labels `Parse.getType` returns whole when a label starts with them, in the order it tests
/// them, before it looks for `-` or `=`.
pub(super) const ESCAPED_TYPES: [&str; 7] = ["-LCB-", "-RCB-", "-LRB-", "-RRB-", "-RSB-", "-LSB-", "-NONE-"];

/// Encodes a bracket token into its Penn Treebank escape (`Parse.encodeToken`).
#[inline]
pub fn encode_token(token: &str) -> &str {
    match token {
        "(" => "-LRB-",
        ")" => "-RRB-",
        "{" => "-LCB-",
        "}" => "-RCB-",
        "[" => "-LSB-",
        "]" => "-RSB-",
        _ => token,
    }
}

/// Decodes a Penn Treebank bracket escape back into its character (`Parse.decodeToken`).
#[inline]
pub fn decode_token(token: &str) -> &str {
    match token {
        "-LRB-" => BRACKET_LRB,
        "-RRB-" => BRACKET_RRB,
        "-LCB-" => BRACKET_LCB,
        "-RCB-" => BRACKET_RCB,
        "-LSB-" => BRACKET_LSB,
        "-RSB-" => BRACKET_RSB,
        _ => token,
    }
}

/// Separates round and curly brackets from adjacent non-whitespace characters with a single ASCII
/// space (`ParserTool.spaceUntokenizedParens`), leaving whitespace already there alone.
///
/// `mode` decides what counts as whitespace, as the active `WhitespaceMode` does for
/// `StringUtil.isWhitespace` in OpenNLP.
pub fn space_untokenized_parens(line: &str, mode: WhitespaceMode) -> Cow<'_, str> {
    let is_ws = |c: char| mode.is_whitespace(c as u32);
    let Some(first) = line.find(['(', ')', '{', '}']) else {
        return Cow::Borrowed(line);
    };
    let mut spaced = String::with_capacity(line.len() + 8);
    spaced.push_str(&line[..first]);
    let mut prev_ws = spaced.chars().next_back().is_none_or(is_ws);
    let mut chars = line[first..].chars().peekable();
    while let Some(c) = chars.next() {
        if matches!(c, '(' | ')' | '{' | '}') {
            if !spaced.is_empty() && !prev_ws {
                spaced.push(' ');
            }
            spaced.push(c);
            prev_ws = false;
            if chars.peek().is_some_and(|&next| !is_ws(next)) {
                spaced.push(' ');
                prev_ws = true;
            }
        } else {
            spaced.push(c);
            prev_ws = is_ws(c);
        }
    }
    Cow::Owned(spaced)
}

/// A constituent read while scanning a Penn Treebank bracketed parse
/// (`opennlp.tools.parser.Constituent`).
///
/// The label is `None` where `Parse.getType` found no type in the bracket (Java's `null`), as for
/// the unlabelled outer bracket of `( (S ...) )`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constituent<'a> {
    label: Option<Cow<'a, str>>,
    span: Span,
}

impl<'a> Constituent<'a> {
    /// Creates a constituent with `label` and `span`.
    pub fn new(label: Option<Cow<'a, str>>, span: Span) -> Constituent<'a> {
        Constituent { label, span }
    }

    /// Returns the constituent label (`Constituent.getLabel`).
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns `true` when the label is `expected`, or [`ParseError::MissingType`] (at input byte
    /// `at`) when there is no label, where Java's `getLabel().equals(expected)` dereferences
    /// `null`.
    pub fn label_is(&self, expected: &str, at: usize) -> Result<bool, ParseError> {
        self.label().map(|l| l == expected).ok_or(ParseError::MissingType { byte_offset: at })
    }

    /// Updates the constituent label (`Constituent.setLabel`).
    pub fn set_label(&mut self, label: impl Into<Cow<'a, str>>) {
        self.label = Some(label.into());
    }

    /// Returns the byte span of the constituent in the sentence text.
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Where one token of a parse read by [`Parse::parse_with_sources`] came from: its span in the
/// parse's sentence text and, in the bracketed input, the token as written (before an escape such
/// as `-LRB-` is decoded) and the type of the bracket around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenSource {
    /// The token's span in the sentence text ([`Parse::text`]).
    pub text: Span,
    /// The token's bytes in the input.
    pub token: Span,
    /// The bytes in the input of the type of the bracket that holds the token: the type alone,
    /// without a function tag, unless function tags are kept. Empty, after the `(`, for a bracket
    /// without a type.
    pub label: Span,
}

/// [`java_type`] for the text after each `(` of an input in turn, in time linear in the input.
///
/// The type ends at the first space, `=` or `-` after the `(`, so the types of the brackets of
/// `((((x` each run to the end; looking for that character again for each `(` takes time in the
/// square of the input's length, as Java's regular expression does. The scan keeps the first such
/// character at or after the last place it looked, which is the first at or after any later place
/// up to it, and the same for the function tag after a `-`.
#[derive(Default)]
pub(super) struct TypeScan {
    stop: usize,
    tag_stop: usize,
}

impl TypeScan {
    /// The first byte at or after `from` that ends a type, or the input's length.
    pub(super) fn next_stop(bytes: &[u8], from: usize) -> usize {
        from + bytes[from..].iter().position(|&b| matches!(b, b' ' | b'=' | b'-')).unwrap_or(bytes.len() - from)
    }

    /// `java_type(&input[at..], use_function_tags)`, for `at` that never goes down between calls.
    pub(super) fn kind<'a>(&mut self, input: &'a str, at: usize, use_function_tags: bool) -> Option<&'a str> {
        let rest = &input[at..];
        if let Some(escaped) = ESCAPED_TYPES.iter().find(|t| rest.starts_with(**t)) {
            return Some(escaped);
        }
        let bytes = input.as_bytes();
        if self.stop < at {
            self.stop = Self::next_stop(bytes, at);
        }
        let n = self.stop - at;
        if n == 0 {
            return None;
        }
        if use_function_tags && bytes.get(self.stop) == Some(&b'-') {
            let from = self.stop + 1;
            if self.tag_stop < from {
                self.tag_stop = Self::next_stop(bytes, from);
            }
            let m = self.tag_stop - from;
            if m > 0 {
                return Some(&rest[..n + 1 + m]);
            }
        }
        Some(&rest[..n])
    }
}

/// `Parse.getToken(rest)` before decoding: the token of `^[^ ()]+ ([^ ()]+)\s*\)`, a run of
/// characters other than space and brackets after the label and exactly one space, followed by
/// optional Java `\s` whitespace and `)`. Backtracking cannot make that pattern match where this
/// does not, since giving characters of the token back to `\s*` leads to the same `)` test.
fn java_token(rest: &str) -> Option<&str> {
    let bytes = rest.as_bytes();
    let run = |from: usize| bytes[from..].iter().take_while(|&&b| !matches!(b, b' ' | b'(' | b')')).count();
    let label = run(0);
    if label == 0 || bytes.get(label) != Some(&b' ') {
        return None;
    }
    let start = label + 1;
    let len = run(start);
    if len == 0 {
        return None;
    }
    let after = start + len;
    let ws =
        bytes[after..].iter().take_while(|&&b| matches!(b, b' ' | b'\t' | b'\n' | b'\x0B' | b'\x0C' | b'\r')).count();
    (bytes.get(after + ws) == Some(&b')')).then(|| &rest[start..after])
}

impl Parse {
    /// Builds the flat initial parse tree (`INC` root with `TK` children) from pre-tokenized
    /// strings (`Parse.createFromTokens`).
    pub fn from_tokens<S: AsRef<str>>(tokens: &[S]) -> Result<Parse, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::EmptyTokens);
        }
        let total_len: usize = tokens.iter().map(|t| t.as_ref().len()).sum::<usize>() + tokens.len() - 1;
        let text_len = checked_text_len(total_len)?;
        let mut text = String::with_capacity(total_len);
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(tok.as_ref());
        }
        let text: Arc<str> = text.into();
        let mut root = TreeBuilder::new(Parse::new(Arc::clone(&text), Span::new(0, text_len), INC_NODE, 0.0, 0));
        let mut start = 0usize;
        for (i, tok) in tokens.iter().enumerate() {
            let len = tok.as_ref().len();
            root.insert(Parse::new(
                Arc::clone(&text),
                Span::new(start as u32, (start + len) as u32),
                TOK_NODE,
                0.0,
                i as i32,
            ))?;
            start += len + 1;
        }
        Ok(root.finish())
    }

    /// Reads a Penn Treebank bracketed parse string without function tags or gap labeling
    /// (`Parse.parseParse(parse)` with `Parse.useFunctionTags(false)`).
    #[allow(clippy::self_named_constructors)]
    pub fn parse(input: &str) -> Result<Parse, ParseError> {
        Self::parse_with_options(input, false, None)
    }

    /// Reads a Penn Treebank bracketed parse string with a [`GapLabeler`]
    /// (`Parse.parseParse(parse, gl)`).
    pub fn parse_with_gaps(input: &str, gap_labeler: &dyn GapLabeler) -> Result<Parse, ParseError> {
        Self::parse_with_options(input, false, Some(gap_labeler))
    }

    /// Reads a Penn Treebank bracketed parse string (`Parse.parseParse`). `use_function_tags`
    /// stands for OpenNLP's static `Parse.useFunctionTags`.
    ///
    /// This reads what `parseParse` reads, leniency included:
    ///
    /// - A bracket's type is the run of characters after `(` up to a space, `=` or `-`, unless the
    ///   label starts with an escaped bracket type or `-NONE-`, which is taken alone (`-NONE-1`
    ///   gives `-NONE-`). Only those three characters end it, so `((S` has the type `(S`.
    /// - With function tags, the first tag after the type is kept (`NP-SBJ-TMP` gives `NP-SBJ`,
    ///   and `WHNP-1` stays `WHNP-1`).
    /// - A bracket holds a token only when its label is followed by exactly one space, the token,
    ///   optional whitespace and `)`. The token runs to the next space or bracket, so it may hold a
    ///   tab, and `(NN  word)`, with two spaces, holds none.
    /// - Text between brackets is ignored, and brackets still open at the end are dropped.
    ///
    /// The sentence text is the tokens, each followed by a space, and spans are UTF-8 byte offsets
    /// into it where Java's are UTF-16 indices.
    ///
    /// It fails where OpenNLP throws: on a `)` without an open bracket, and on a bracket without a
    /// type that is closed around tokens (the outer bracket of `( (S ...) )`) or whose label a gap
    /// labeler reads.
    pub fn parse_with_options(
        input: &str,
        use_function_tags: bool,
        gap_labeler: Option<&dyn GapLabeler>,
    ) -> Result<Parse, ParseError> {
        Self::read_brackets(input, use_function_tags, gap_labeler, None)
    }

    /// Reads a Penn Treebank bracketed parse string as [`Parse::parse_with_options`] does, without
    /// a gap labeler, and also gives where each token of the parse came from in `input`, one
    /// [`TokenSource`] per token node in the order of the sentence text.
    pub fn parse_with_sources(input: &str, use_function_tags: bool) -> Result<(Parse, Vec<TokenSource>), ParseError> {
        let mut sources = Vec::new();
        let parse = Self::read_brackets(input, use_function_tags, None, Some(&mut sources))?;
        Ok((parse, sources))
    }

    fn read_brackets(
        input: &str,
        use_function_tags: bool,
        gap_labeler: Option<&dyn GapLabeler>,
        mut sources: Option<&mut Vec<TokenSource>>,
    ) -> Result<Parse, ParseError> {
        let mut text = String::new();
        let mut offset = 0u32;
        let mut types = TypeScan::default();
        let mut stack: Vec<Constituent<'_>> = Vec::new();
        let mut cons: Vec<Constituent<'_>> = Vec::new();
        for (ci, &c) in input.as_bytes().iter().enumerate() {
            if c == b'(' {
                let rest = &input[ci + 1..];
                let kind = types.kind(input, ci + 1, use_function_tags);
                let token = java_token(rest);
                stack.push(Constituent::new(kind.map(Cow::Borrowed), Span::new(offset, offset)));
                if let Some(token) = token {
                    match gap_labeler {
                        Some(gl) if kind == Some("-NONE-") => gl.label_gaps(&mut stack, ci)?,
                        _ => {
                            let decoded = decode_token(token);
                            // `offset` fits u32 here; the guard keeps one more
                            // token and its space inside the limit.
                            if offset as usize + decoded.len() + 1 > u32::MAX as usize {
                                return Err(ParseError::TooLong { len: input.len() });
                            }
                            let len = decoded.len() as u32;
                            if let Some(sources) = sources.as_deref_mut() {
                                let at = token.as_ptr() as usize - input.as_ptr() as usize;
                                let label = kind.map_or(0, str::len);
                                sources.push(TokenSource {
                                    text: Span::new(offset, offset + len),
                                    token: Span::new(at as u32, (at + token.len()) as u32),
                                    label: Span::new(ci as u32 + 1, (ci + 1 + label) as u32),
                                });
                            }
                            let tk = Constituent::new(Some(Cow::Borrowed(TOK_NODE)), Span::new(offset, offset + len));
                            cons.push(tk);
                            text.push_str(decoded);
                            text.push(' ');
                            offset += len + 1;
                        }
                    }
                }
            } else if c == b')' {
                let con = stack.pop().ok_or(ParseError::UnmatchedClose { byte_offset: ci })?;
                let start = con.span.start;
                if start < offset {
                    cons.push(Constituent::new(con.label, Span::new(start, offset - 1)));
                }
            }
        }

        let txt: Arc<str> = text.into();
        let mut token_index = -1i32;
        let mut tree = TreeBuilder::new(Parse::new(
            Arc::clone(&txt),
            Span::new(0, checked_text_len(txt.len())?),
            TOP_NODE,
            1.0,
            0,
        ));
        for con in cons {
            let kind = con.label.ok_or(ParseError::MissingType { byte_offset: input.len() })?;
            if kind != TOP_NODE {
                if kind == TOK_NODE {
                    token_index += 1;
                }
                tree.insert(Parse::new(Arc::clone(&txt), con.span, kind.into_owned(), 1.0, token_index))?;
            }
        }
        Ok(tree.finish())
    }
}
