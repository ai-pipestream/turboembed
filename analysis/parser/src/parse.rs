//! Constituency parse trees and Penn Treebank bracket I/O (`opennlp.tools.parser.Parse`).

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_jvm::strict_math;
use turbo_unicode::WhitespaceMode;

use crate::head_rules::{GapLabeler, HeadRules};

/// The constituent label of the root node (`Parser.TOP_NODE`).
pub const TOP_NODE: &str = "TOP";

/// The constituent label of an incomplete root node (`Parser.INC_NODE`).
pub const INC_NODE: &str = "INC";

/// The constituent label of a token leaf node (`Parser.TOK_NODE`).
pub const TOK_NODE: &str = "TK";

/// Prefix for outcomes starting a constituent (`AbstractBottomUpParser.START`).
pub const START: &str = "S-";

/// Prefix for outcomes continuing a constituent (`AbstractBottomUpParser.CONT`).
pub const CONT: &str = "C-";

/// Outcome for a token not contained in a basal constituent (`AbstractBottomUpParser.OTHER`).
pub const OTHER: &str = "O";

/// Outcome used when a constituent is complete (`AbstractBottomUpParser.COMPLETE`).
pub const COMPLETE: &str = "c";

/// Outcome used when a constituent is incomplete (`AbstractBottomUpParser.INCOMPLETE`).
pub const INCOMPLETE: &str = "i";

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
const ESCAPED_TYPES: [&str; 7] = ["-LCB-", "-RCB-", "-LRB-", "-RRB-", "-RSB-", "-LSB-", "-NONE-"];

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

/// An error raised while reading or changing a [`Parse`]. Each variant stands for the exception
/// OpenNLP throws in the same place ([`ParseError::java_class`]), except
/// [`ParseError::NotCharBoundary`], which has none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A bracket without a type was closed around tokens, or a gap labeler read the label of one
    /// (`NullPointerException`). `byte_offset` is the input offset of the `-NONE-` bracket the
    /// gap labeler was called for, or the input's length for a closed bracket.
    MissingType { byte_offset: usize },
    /// A `)` with no open bracket (`EmptyStackException`), at `byte_offset` in the input.
    UnmatchedClose { byte_offset: usize },
    /// `Parse::from_tokens` was given an empty token slice (`IllegalArgumentException`).
    EmptyTokens,
    /// `Parse::insert` was given a constituent whose span is outside the receiving node's span
    /// (`IllegalArgumentException`).
    NotContained { parent: Span, child: Span },
    /// A span that does not start and end on character boundaries of the sentence text. Java
    /// indexes UTF-16 units and has no such case.
    NotCharBoundary { span: Span },
    /// `add` or `adjoin` got no head from the head rules, as when the first child is a token node
    /// (`NullPointerException` on the head's index).
    MissingHead,
    /// `prune_parse` would replace the root, which has no parent (`NullPointerException`).
    MissingParent,
    /// `add_names` was given a token index outside the tokens (`ArrayIndexOutOfBoundsException`).
    TokenIndex { index: i64, len: usize },
    /// `add_names` was given a name whose span would end before it starts
    /// (`IllegalArgumentException` from `Span`).
    ReversedName { name: Span },
}

impl ParseError {
    /// The class of the exception OpenNLP throws in the same place, if it throws one.
    pub fn java_class(&self) -> Option<&'static str> {
        match self {
            ParseError::MissingType { .. } | ParseError::MissingHead | ParseError::MissingParent => {
                Some("java.lang.NullPointerException")
            }
            ParseError::UnmatchedClose { .. } => Some("java.util.EmptyStackException"),
            ParseError::EmptyTokens | ParseError::NotContained { .. } | ParseError::ReversedName { .. } => {
                Some("java.lang.IllegalArgumentException")
            }
            ParseError::TokenIndex { .. } => Some("java.lang.ArrayIndexOutOfBoundsException"),
            ParseError::NotCharBoundary { .. } => None,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MissingType { byte_offset } => {
                write!(f, "constituent without a type (input byte {byte_offset})")
            }
            ParseError::UnmatchedClose { byte_offset } => {
                write!(f, "closing bracket without an open one (input byte {byte_offset})")
            }
            ParseError::EmptyTokens => f.write_str("tokens must not be empty"),
            ParseError::NotContained { .. } => f.write_str("Inserting constituent not contained in the sentence!"),
            ParseError::NotCharBoundary { span } => {
                write!(f, "span {}..{} is not on character boundaries", span.start, span.end)
            }
            ParseError::MissingHead => f.write_str("the head rules gave no head"),
            ParseError::MissingParent => f.write_str("the root has no parent to take its only child"),
            ParseError::TokenIndex { index, len } => write!(f, "Index {index} out of bounds for length {len}"),
            ParseError::ReversedName { name } => {
                write!(f, "name {}..{} ends before it starts", name.start, name.end)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// The node a parse takes as its head (`Parse.getHead`), recorded as the head node's span and
/// head token index.
///
/// OpenNLP keeps a reference to the head node. Its users read only the head's covered text and
/// head index, so those are kept instead, and they stay right when the tree is rearranged around
/// the head. They are what the head node had when it was chosen: a later change to that node's
/// span is not seen, where Java would see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Head {
    /// The head node's span in the sentence text.
    pub span: Span,
    /// The head node's head token index (`Parse.getHeadIndex`).
    pub head_index: i32,
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

/// Builds a tree by inserting constituents into its root as [`Parse::insert`] does, in less time
/// when they come in the order a bracketed tree closes them: each after the constituents inside it
/// and after those to its left.
///
/// `insert` looks at every child of the root for each constituent, so a flat sentence of `n`
/// tokens takes time in `n` squared. While the constituents come in that order, the root's
/// children are sorted, do not overlap and are not empty, and a new constituent that ends at or
/// after the last child and starts at or after the end of the child before the first one it
/// covers takes the children from that one on and goes last, which is what `insert` does with
/// it. The builder finds that child by binary search and checks the conditions; the first
/// constituent that does not meet them, and every one after it, goes through `insert`.
#[derive(Debug)]
pub struct TreeBuilder {
    root: Parse,
    ordered: bool,
}

impl TreeBuilder {
    /// A builder over `root`, which has no children.
    pub fn new(root: Parse) -> TreeBuilder {
        let ordered = root.parts.is_empty();
        TreeBuilder { root, ordered }
    }

    /// Inserts `constituent` into the tree, as [`Parse::insert`] does.
    pub fn insert(&mut self, mut constituent: Parse) -> Result<(), ParseError> {
        let ic = constituent.span;
        if self.ordered && ic.start < ic.end && self.root.span.contains(ic) {
            let parts = &mut self.root.parts;
            let first = parts.partition_point(|p| p.span.start < ic.start);
            let after_left = first == 0 || parts[first - 1].span.end <= ic.start;
            let covers_rest = parts.last().is_none_or(|p| p.span.end <= ic.end);
            if after_left && covers_rest {
                if !self.root.text.is_char_boundary(ic.start as usize)
                    || !self.root.text.is_char_boundary(ic.end as usize)
                {
                    return Err(ParseError::NotCharBoundary { span: ic });
                }
                constituent.parts.extend(parts.drain(first..));
                parts.push(constituent);
                return Ok(());
            }
        }
        self.ordered = false;
        self.root.insert(constituent)
    }

    /// The tree.
    pub fn finish(self) -> Parse {
        self.root
    }
}

/// A constituency parse node (`opennlp.tools.parser.Parse`).
///
/// All nodes in the same sentence share an `Arc<str>` holding the space-joined sentence text, and
/// `span` stores UTF-8 byte offsets into that text. A node is its own head until head rules give it
/// another ([`Head`]).
///
/// `Clone` copies the whole subtree. Java's `clone` copies the node and shares its children, and a
/// clone of a node that is its own head keeps the original as its head; here the clone is its own
/// head.
#[derive(Clone, Debug)]
pub struct Parse {
    text: Arc<str>,
    span: Span,
    kind: String,
    parts: Vec<Parse>,
    /// `None` when the node is its own head (Java's `head == this`).
    head: Option<Head>,
    head_index: i32,
    label: Option<String>,
    prob: f64,
    is_chunk: bool,
    derivation: Option<String>,
    prev_punct: Vec<Parse>,
    next_punct: Vec<Parse>,
}

impl Parse {
    /// Creates a parse node that is its own head (`new Parse(text, span, type, p, index)`).
    ///
    /// `span` must start and end on character boundaries of `text`; [`Parse::insert`] checks this,
    /// and [`Parse::show`] and [`Parse::covered_text`] panic on a node that breaks it.
    pub fn new(text: impl Into<Arc<str>>, span: Span, kind: impl Into<String>, prob: f64, head_index: i32) -> Parse {
        Parse {
            text: text.into(),
            span,
            kind: kind.into(),
            parts: Vec::new(),
            head: None,
            head_index,
            label: None,
            prob,
            is_chunk: false,
            derivation: None,
            prev_punct: Vec::new(),
            next_punct: Vec::new(),
        }
    }

    /// Creates a parse node with `head` as its head and head index, or its own head with index 0
    /// when `head` is `None` (`new Parse(text, span, type, p, h)`).
    pub fn with_head(
        text: impl Into<Arc<str>>,
        span: Span,
        kind: impl Into<String>,
        prob: f64,
        head: Option<Head>,
    ) -> Parse {
        let mut p = Parse::new(text, span, kind, prob, 0);
        if let Some(h) = head {
            p.head = Some(h);
            p.head_index = h.head_index;
        }
        p
    }

    /// Builds the flat initial parse tree (`INC` root with `TK` children) from pre-tokenized
    /// strings (`Parse.createFromTokens`).
    pub fn from_tokens<S: AsRef<str>>(tokens: &[S]) -> Result<Parse, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::EmptyTokens);
        }
        let total_len: usize = tokens.iter().map(|t| t.as_ref().len()).sum::<usize>() + tokens.len() - 1;
        let mut text = String::with_capacity(total_len);
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(tok.as_ref());
        }
        let text: Arc<str> = text.into();
        let mut root =
            TreeBuilder::new(Parse::new(Arc::clone(&text), Span::new(0, text.len() as u32), INC_NODE, 0.0, 0));
        let mut start = 0u32;
        for (i, tok) in tokens.iter().enumerate() {
            let len = tok.as_ref().len() as u32;
            root.insert(Parse::new(Arc::clone(&text), Span::new(start, start + len), TOK_NODE, 0.0, i as i32))?;
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
                            if let Some(sources) = sources.as_deref_mut() {
                                let at = token.as_ptr() as usize - input.as_ptr() as usize;
                                let label = kind.map_or(0, str::len);
                                sources.push(TokenSource {
                                    text: Span::new(offset, offset + decode_token(token).len() as u32),
                                    token: Span::new(at as u32, (at + token.len()) as u32),
                                    label: Span::new(ci as u32 + 1, (ci + 1 + label) as u32),
                                });
                            }
                            let token = decode_token(token);
                            let len = token.len() as u32;
                            let tk = Constituent::new(Some(Cow::Borrowed(TOK_NODE)), Span::new(offset, offset + len));
                            cons.push(tk);
                            text.push_str(token);
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
        let mut tree = TreeBuilder::new(Parse::new(Arc::clone(&txt), Span::new(0, txt.len() as u32), TOP_NODE, 1.0, 0));
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

    /// Inserts `constituent` into this parse tree based on its text span (`Parse.insert`).
    ///
    /// The constituent goes into the deepest node with a child that contains it, and takes the
    /// children of that node that it contains, a child with the same span included.
    pub fn insert(&mut self, mut constituent: Parse) -> Result<(), ParseError> {
        let ic = constituent.span;
        if !self.span.contains(ic) {
            return Err(ParseError::NotContained { parent: self.span, child: ic });
        }
        if !self.text.is_char_boundary(ic.start as usize) || !self.text.is_char_boundary(ic.end as usize) {
            return Err(ParseError::NotCharBoundary { span: ic });
        }
        let mut node = self;
        'descend: loop {
            let mut pi = 0;
            while pi < node.parts.len() {
                let sp = node.parts[pi].span;
                if sp.start >= ic.end {
                    break;
                } else if ic.contains(sp) {
                    constituent.parts.push(node.parts.remove(pi));
                } else if sp.contains(ic) {
                    node = &mut node.parts[pi];
                    continue 'descend;
                } else {
                    pi += 1;
                }
            }
            node.parts.insert(pi, constituent);
            return Ok(());
        }
    }

    /// Appends the Penn Treebank bracketed representation of this parse to `out` (`Parse.show`).
    ///
    /// Text of a node that its children do not cover is written as a token, with brackets
    /// escaped; a parse read by [`Parse::parse`] has the space after its last token inside the
    /// root, so the root closes with ` )`.
    pub fn show(&self, out: &mut String) {
        let mut start = self.span.start as usize;
        if self.kind != TOK_NODE {
            out.push('(');
            out.push_str(&self.kind);
            out.push(' ');
        }
        for c in &self.parts {
            let s_start = c.span.start as usize;
            if start < s_start {
                out.push_str(encode_token(&self.text[start..s_start]));
            }
            c.show(out);
            start = c.span.end as usize;
        }
        let end = self.span.end as usize;
        if start < end {
            out.push_str(encode_token(&self.text[start..end]));
        }
        if self.kind != TOK_NODE {
            out.push(')');
        }
    }

    /// Returns the Penn Treebank bracketed representation of this parse (`Parse.toStringPennTreebank`).
    pub fn to_string_penn_treebank(&self) -> String {
        let mut out = String::with_capacity(self.text.len() * 4);
        self.show(&mut out);
        out
    }

    /// Returns the log probability sum of the POS-tag sequence under this parse
    /// (`Parse.getTagSequenceProb`), using `turbo_jvm::strict_math::log` (`StrictMath.log`).
    pub fn tag_sequence_prob(&self) -> f64 {
        if self.parts.len() == 1 && self.parts[0].kind == TOK_NODE {
            strict_math::log(self.prob)
        } else if self.parts.is_empty() {
            0.0
        } else {
            let mut sum = 0.0;
            for part in &self.parts {
                sum += part.tag_sequence_prob();
            }
            sum
        }
    }

    /// Returns `true` if this parse has a single top-level child (`Parse.complete`).
    pub fn complete(&self) -> bool {
        self.parts.len() == 1
    }

    /// Returns the substring of `text` covered by this node's `span` (`Parse.getCoveredText`).
    pub fn covered_text(&self) -> &str {
        &self.text[self.span.range()]
    }

    /// Returns the shared sentence text (`Parse.getText`).
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the shared `Arc<str>` holding the sentence text.
    pub fn text_arc(&self) -> &Arc<str> {
        &self.text
    }

    /// Returns the UTF-8 byte span of this constituent in `text` (`Parse.getSpan`).
    pub fn span(&self) -> Span {
        self.span
    }

    /// Returns the syntactic constituent type (`Parse.getType`).
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Sets the syntactic constituent type (`Parse.setType`).
    pub fn set_kind(&mut self, kind: impl Into<String>) {
        self.kind = kind.into();
    }

    /// Returns the probability or log-probability score associated with this node (`Parse.getProb`).
    pub fn prob(&self) -> f64 {
        self.prob
    }

    /// Adds `log_prob` to this node's probability score (`Parse.addProb`).
    pub fn add_prob(&mut self, log_prob: f64) {
        self.prob += log_prob;
    }

    /// Returns the child constituents (`Parse.getChildren`).
    pub fn children(&self) -> &[Parse] {
        &self.parts
    }

    /// Returns a mutable slice of the child constituents.
    pub fn children_mut(&mut self) -> &mut [Parse] {
        &mut self.parts
    }

    /// Returns the number of child constituents (`Parse.getChildCount`).
    pub fn child_count(&self) -> usize {
        self.parts.len()
    }

    /// Replaces the label of the child at `index` (`Parse.setChild`).
    pub fn set_child(&mut self, index: usize, label: impl Into<String>) {
        self.parts[index].label = Some(label.into());
    }

    /// Appends `daughter`, after copies of its preceding punctuation, to this node and updates
    /// its span and head (`Parse.add`). The daughter keeps its punctuation set, as in Java.
    ///
    /// Fails with [`ParseError::MissingHead`] when `rules` give no head, after the children and
    /// span have changed, as Java does.
    pub fn add(&mut self, daughter: Parse, rules: &dyn HeadRules) -> Result<(), ParseError> {
        self.parts.extend(daughter.prev_punct.iter().cloned());
        let end = daughter.span.end;
        self.parts.push(daughter);
        self.span = Span::new(self.span.start, end);
        self.take_head_from_children(rules)
    }

    /// Removes the child at `index` and updates the span when the first or last child is removed
    /// (`Parse.remove`).
    pub fn remove(&mut self, index: usize) {
        self.parts.remove(index);
        if !self.parts.is_empty() && (index == 0 || index == self.parts.len()) {
            self.update_span();
        }
    }

    /// Adjoins `node` to the child at `parse_index` and returns the node made to hold both
    /// (`Parse.adjoinRoot`). That node has the child's type and the head `rules` give for the
    /// child and `node`, and holds copies of `node`'s preceding punctuation between them. This
    /// node's span is left as it is.
    pub fn adjoin_root(&mut self, node: Parse, rules: &dyn HeadRules, parse_index: usize) -> &mut Parse {
        Self::adjoin_pair(&mut self.parts[parse_index], node, rules);
        &mut self.parts[parse_index]
    }

    /// Sister-adjoins this node's last child and `sister` under a node made with the last child's
    /// type, as [`Parse::adjoin_root`] does, then updates this node's span and head
    /// (`Parse.adjoin`). Returns the node holding both.
    ///
    /// Fails with [`ParseError::MissingHead`] when `rules` give no head for this node's children,
    /// after the children and span have changed, as Java does.
    pub fn adjoin(&mut self, sister: Parse, rules: &dyn HeadRules) -> Result<&mut Parse, ParseError> {
        let last = self.parts.len() - 1;
        let sister_end = sister.span.end;
        Self::adjoin_pair(&mut self.parts[last], sister, rules);
        self.span = Span::new(self.span.start, sister_end);
        self.take_head_from_children(rules)?;
        Ok(&mut self.parts[last])
    }

    /// Replaces `slot` by a node holding its old content and `node`, as `adjoin` and
    /// `adjoin_root` build it.
    fn adjoin_pair(slot: &mut Parse, node: Parse, rules: &dyn HeadRules) {
        let placeholder = Parse::new(Arc::clone(&slot.text), slot.span, String::new(), 0.0, 0);
        let pair = [std::mem::replace(slot, placeholder), node];
        let head = rules.get_head(&pair, &pair[0].kind);
        let [last_child, node] = pair;
        let span = Span::new(last_child.span.start, node.span.end);
        let mut adj = Parse::with_head(Arc::clone(&last_child.text), span, last_child.kind.clone(), 1.0, head);
        adj.parts.reserve(node.prev_punct.len() + 2);
        adj.parts.push(last_child);
        adj.parts.extend(node.prev_punct.iter().cloned());
        adj.parts.push(node);
        *slot = adj;
    }

    /// Moves all children of `self` other than the child at `root_index` into that child, those
    /// before it in front of its own children and those after it behind them, and updates that
    /// child's span (`Parse.expandTopNode`). This node's span and head are left as they are.
    pub fn expand_top_node(&mut self, root_index: usize) {
        let mut after = self.parts.split_off(root_index + 1);
        let mut root = self.parts.pop().expect("root_index is a child of this node");
        let mut before = std::mem::take(&mut self.parts);
        before.reserve(root.parts.len() + after.len());
        before.append(&mut root.parts);
        before.append(&mut after);
        root.parts = before;
        root.update_span();
        self.parts.push(root);
    }

    /// Returns the head of this node (`Parse.getHead`): the node itself until head rules give it
    /// another.
    pub fn head(&self) -> Head {
        self.head.unwrap_or(Head { span: self.span, head_index: self.head_index })
    }

    /// Returns this node as the head of another node: its own span and head index, as Java's head
    /// rules return a constituent itself.
    pub fn as_head(&self) -> Head {
        Head { span: self.span, head_index: self.head_index }
    }

    /// Returns `true` when this node is its own head (Java's `getHead() == this`).
    pub fn is_own_head(&self) -> bool {
        self.head.is_none()
    }

    /// Returns the text the head covers (`getHead().getCoveredText()`).
    pub fn head_text(&self) -> &str {
        &self.text[self.head().span.range()]
    }

    /// Returns the sentence token index of the head token (`Parse.getHeadIndex`).
    pub fn head_index(&self) -> i32 {
        self.head_index
    }

    /// Returns the parser stage label assigned to this node (`Parse.getLabel`).
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Assigns a parser stage label to this node (`Parse.setLabel`).
    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
    }

    /// Computes heads bottom-up for this parse and all sub-parses (`Parse.updateHeads`). A node
    /// whose rules give no head (a part-of-speech node over its token) becomes its own head and
    /// keeps its head index.
    pub fn update_heads(&mut self, rules: &dyn HeadRules) {
        if self.parts.is_empty() {
            self.head = None;
            return;
        }
        for c in &mut self.parts {
            c.update_heads(rules);
        }
        match rules.get_head(&self.parts, &self.kind) {
            Some(h) => {
                self.head = Some(h);
                self.head_index = h.head_index;
            }
            None => self.head = None,
        }
    }

    /// Sets the head `rules` give for the children, as `add` and `adjoin` do.
    fn take_head_from_children(&mut self, rules: &dyn HeadRules) -> Result<(), ParseError> {
        let h = rules.get_head(&self.parts, &self.kind).ok_or(ParseError::MissingHead)?;
        self.head = Some(h);
        self.head_index = h.head_index;
        Ok(())
    }

    /// Recomputes `span` from the first and last child (`Parse.updateSpan`).
    pub fn update_span(&mut self) {
        if let (Some(first), Some(last)) = (self.parts.first(), self.parts.last()) {
            self.span = Span::new(first.span.start, last.span.end);
        }
    }

    /// Replaces every node whose only child has the node's own type by that child
    /// (`Parse.pruneParse`), repeatedly, so a chain `A -> A -> A` keeps its last node.
    ///
    /// Fails with [`ParseError::MissingParent`], changing nothing, when `parse` itself has a single
    /// child of its own type, where Java dereferences the root's missing parent.
    pub fn prune_parse(parse: &mut Parse) -> Result<(), ParseError> {
        if parse.parts.len() == 1 && parse.kind == parse.parts[0].kind {
            return Err(ParseError::MissingParent);
        }
        Self::prune_children(parse);
        Ok(())
    }

    fn prune_children(node: &mut Parse) {
        for child in &mut node.parts {
            while child.parts.len() == 1 && child.kind == child.parts[0].kind {
                let grand_child = child.parts.pop().expect("one child");
                *child = grand_child;
            }
            Self::prune_children(child);
        }
    }

    /// Wraps the tag nodes after a possessive (`POS`) in an `NP` whose head is the first of them
    /// (`Parse.fixPossesives`): when the tag after a `POS` node is a child of the `POS` node's
    /// grandparent, it and the following tags that share its parent are put under an added `NP`.
    ///
    /// OpenNLP compares the tag nodes' current parents, so an `NP` inserted for one possessive
    /// changes the parents seen for the next; the tag paths are read again after every insert.
    pub fn fix_possessives(parse: &mut Parse) {
        let mut tags: Vec<TagPath> = Vec::new();
        let mut path = Vec::new();
        Self::collect_tag_paths(parse, &mut path, &mut tags);
        for ti in 0..tags.len() {
            if tags[ti].kind != "POS" || ti + 1 >= tags.len() {
                continue;
            }
            let grandparent = tags[ti].path.len().checked_sub(2).map(|n| &tags[ti].path[..n]);
            if grandparent != Some(tags[ti + 1].parent()) {
                continue;
            }
            let start = tags[ti + 1].span.start;
            let mut end = tags[ti + 1].span.end;
            for npi in (ti + 2)..tags.len() {
                if tags[npi].parent() == tags[npi - 1].parent() {
                    end = tags[npi].span.end;
                } else {
                    break;
                }
            }
            let head = Head { span: tags[ti + 1].span, head_index: tags[ti + 1].head_index };
            let np = Parse::with_head(Arc::clone(&parse.text), Span::new(start, end), "NP", 1.0, Some(head));
            parse.insert(np).expect("tag nodes lie inside the root");
            tags.clear();
            Self::collect_tag_paths(parse, &mut path, &mut tags);
        }
    }

    /// Lists the tag nodes as `getTagNodes` does, each with its path. Inserting a node over whole
    /// tag nodes, as `fix_possessives` and `add_names` do, never changes which nodes are tags or
    /// their order, so listing them again finds the same nodes as Java's array holds.
    fn collect_tag_paths(node: &Parse, path: &mut Vec<usize>, out: &mut Vec<TagPath>) {
        for (i, child) in node.parts.iter().enumerate() {
            path.push(i);
            if child.is_pos_tag() {
                out.push(TagPath {
                    kind: child.kind.clone(),
                    span: child.span,
                    head_index: child.head_index,
                    path: path.clone(),
                });
            } else {
                Self::collect_tag_paths(child, path, out);
            }
            path.pop();
        }
    }

    /// Returns `true` if this node is a POS-tag preterminal (`Parse.isPosTag`).
    pub fn is_pos_tag(&self) -> bool {
        self.parts.len() == 1 && self.parts[0].kind == TOK_NODE
    }

    /// Returns `true` if every child of this node is a POS-tag preterminal (`Parse.isFlat`).
    pub fn is_flat(&self) -> bool {
        self.parts.iter().all(Parse::is_pos_tag)
    }

    /// Marks whether this node was built during the chunking phase (`Parse.isChunk(boolean)`).
    pub fn set_is_chunk(&mut self, is_chunk: bool) {
        self.is_chunk = is_chunk;
    }

    /// Returns whether this node was built during the chunking phase (`Parse.isChunk()`).
    pub fn is_chunk(&self) -> bool {
        self.is_chunk
    }

    /// Returns all POS-tag nodes under this parse in left-to-right order (`Parse.getTagNodes`).
    pub fn tag_nodes(&self) -> Vec<&Parse> {
        let mut tags = Vec::new();
        self.collect_tag_nodes(&mut tags);
        tags
    }

    fn collect_tag_nodes<'a>(&'a self, out: &mut Vec<&'a Parse>) {
        for part in &self.parts {
            if part.is_pos_tag() {
                out.push(part);
            } else {
                part.collect_tag_nodes(out);
            }
        }
    }

    /// Returns all `TK` token leaf nodes under this parse in left-to-right order
    /// (`Parse.getTokenNodes`).
    pub fn token_nodes(&self) -> Vec<&Parse> {
        let mut tokens = Vec::new();
        self.collect_token_nodes(&mut tokens);
        tokens
    }

    fn collect_token_nodes<'a>(&'a self, out: &mut Vec<&'a Parse>) {
        for part in &self.parts {
            if part.kind == TOK_NODE {
                out.push(part);
            } else {
                part.collect_token_nodes(out);
            }
        }
    }

    /// Returns the basal phrase chunks and unchunked POS nodes under this parse in left-to-right
    /// order (`ChunkSampleStream.getInitialChunks`). A node without children counts as a node
    /// whose children are all tags, as in Java.
    pub fn initial_chunks(&self) -> Vec<&Parse> {
        let mut chunks = Vec::new();
        self.collect_initial_chunks(&mut chunks);
        chunks
    }

    fn collect_initial_chunks<'a>(&'a self, out: &mut Vec<&'a Parse>) {
        if self.is_pos_tag() || self.parts.iter().all(Parse::is_pos_tag) {
            out.push(self);
        } else {
            for kid in &self.parts {
                kid.collect_initial_chunks(out);
            }
        }
    }

    /// Inserts a node of type `tag` for each name (`Parse.addNames`). Each name is a range of
    /// indices into the tag nodes of `parse` (the `getTagNodes` array OpenNLP's callers pass as
    /// `tokens`), end exclusive.
    ///
    /// A name's node goes under the common parent of its first and last token when the name's span
    /// is that parent's span or crosses none of its children. When it crosses a child of an `NP`
    /// and covers the last child of that `NP`'s first child, the node spans the whole `NP`
    /// instead; otherwise the name is skipped. The tokens are the same nodes for every name, so
    /// inserting one name's node can move the tokens of the next, as in Java.
    ///
    /// Fails, after inserting the names before it, on a token index outside the tag nodes or a
    /// name that ends before it starts.
    pub fn add_names(tag: &str, names: &[Span], parse: &mut Parse) -> Result<(), ParseError> {
        let mut tokens: Vec<TagPath> = Vec::new();
        let mut path = Vec::new();
        Self::collect_tag_paths(parse, &mut path, &mut tokens);
        for name in names {
            let token = |index: i64| {
                usize::try_from(index)
                    .ok()
                    .and_then(|i| tokens.get(i))
                    .ok_or(ParseError::TokenIndex { index, len: tokens.len() })
            };
            let first = token(i64::from(name.start))?;
            let last = token(i64::from(name.end) - 1)?;
            if first.span.start > last.span.end {
                return Err(ParseError::ReversedName { name: *name });
            }
            let common = common_parent_path(&first.path, &last.path);
            let name_span = Span::new(first.span.start, last.span.end);
            let end_head_index = last.head_index;
            let parent = parse.node_at_path_mut(&common);
            let node = if name_span == parent.span || !parent.parts.iter().any(|kid| spans_cross(name_span, kid.span)) {
                Some(Parse::new(Arc::clone(&parent.text), name_span, tag, 1.0, end_head_index))
            } else if parent.kind == "NP"
                && let Some(grand_kids) = parent.parts.first().map(|kid| &kid.parts)
                && grand_kids.len() > 1
                && name_span.contains(grand_kids[grand_kids.len() - 1].span)
            {
                Some(Parse::new(Arc::clone(&parent.text), parent.span, tag, 1.0, parent.head_index))
            } else {
                None
            };
            if let Some(node) = node {
                parent.insert(node).expect("a name lies inside its common parent");
                tokens.clear();
                Self::collect_tag_paths(parse, &mut path, &mut tokens);
            }
        }
        Ok(())
    }

    fn node_at_path_mut(&mut self, path: &[usize]) -> &mut Parse {
        let mut cur = self;
        for &idx in path {
            cur = &mut cur.parts[idx];
        }
        cur
    }

    /// Returns the punctuation parses immediately preceding this node, in the order of
    /// `Parse.getPreviousPunctuationSet`.
    pub fn previous_punctuation(&self) -> &[Parse] {
        &self.prev_punct
    }

    /// Adds a punctuation parse preceding this node (`Parse.addPreviousPunctuation`).
    ///
    /// OpenNLP keeps the set in a `TreeSet` ordered by [`Parse::cmp_prob`], so the marks are in
    /// order of descending probability and a mark with the same probability as one already there
    /// is dropped. Every node `Parse::parse` makes has probability 1, so only the first mark added
    /// to such a node is kept.
    pub fn add_previous_punctuation(&mut self, punct: Parse) {
        insert_by_prob(&mut self.prev_punct, punct);
    }

    /// Sets the punctuation parses preceding this node, in the order given
    /// (`Parse.setPrevPunctuation`).
    pub fn set_previous_punctuation(&mut self, punct: Vec<Parse>) {
        self.prev_punct = punct;
    }

    /// Returns the punctuation parses immediately following this node, in the order of
    /// `Parse.getNextPunctuationSet`.
    pub fn next_punctuation(&self) -> &[Parse] {
        &self.next_punct
    }

    /// Adds a punctuation parse following this node (`Parse.addNextPunctuation`), ordered and
    /// dropped as [`Parse::add_previous_punctuation`] describes.
    pub fn add_next_punctuation(&mut self, punct: Parse) {
        insert_by_prob(&mut self.next_punct, punct);
    }

    /// Sets the punctuation parses following this node, in the order given
    /// (`Parse.setNextPunctuation`).
    pub fn set_next_punctuation(&mut self, punct: Vec<Parse>) {
        self.next_punct = punct;
    }

    /// Returns the derivation string, if any (`Parse.getDerivation`).
    pub fn derivation(&self) -> Option<&str> {
        self.derivation.as_deref()
    }

    /// Sets the derivation string (`Parse.setDerivation`).
    pub fn set_derivation(&mut self, derivation: Option<String>) {
        self.derivation = derivation;
    }

    /// Compares two parses by descending probability (`Parse.compareTo`, which uses
    /// `Double.compare`).
    pub fn cmp_prob(&self, other: &Parse) -> Ordering {
        java_double_compare(other.prob, self.prob)
    }
}

/// `Double.compare`: numeric order with `-0.0` before `0.0`, and every NaN equal to every other
/// and after everything else.
fn java_double_compare(a: f64, b: f64) -> Ordering {
    match a.partial_cmp(&b) {
        Some(Ordering::Equal) | None => {
            let bits = |x: f64| if x.is_nan() { 0x7ff8_0000_0000_0000_i64 } else { x.to_bits() as i64 };
            bits(a).cmp(&bits(b))
        }
        Some(o) => o,
    }
}

/// `TreeSet.add` under `Parse.compareTo`: keeps `set` sorted and drops `punct` when an element
/// compares equal to it.
fn insert_by_prob(set: &mut Vec<Parse>, punct: Parse) {
    if let Err(pos) = set.binary_search_by(|p| p.cmp_prob(&punct)) {
        set.insert(pos, punct);
    }
}

/// `Span.crosses`.
#[inline]
fn spans_cross(a: Span, b: Span) -> bool {
    !a.contains(b)
        && !b.contains(a)
        && ((a.start <= b.start && b.start < a.end) || (b.start <= a.start && a.start < b.end))
}

/// The path of `Parse.getCommonParent` for the nodes at paths `a` and `b`: the parent when they
/// are the same node, otherwise their deepest common ancestor (one of them, when it contains the
/// other).
fn common_parent_path(a: &[usize], b: &[usize]) -> Vec<usize> {
    if a == b {
        return a[..a.len().saturating_sub(1)].to_vec();
    }
    let lcp = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    a[..lcp].to_vec()
}

/// `Parse.equals`: the label, span, text and children, but not the type, so `(NP x)` and `(VP x)`
/// over the same text are equal.
impl PartialEq for Parse {
    fn eq(&self, other: &Parse) -> bool {
        self.label == other.label && self.span == other.span && *self.text == *other.text && self.parts == other.parts
    }
}

impl Eq for Parse {}

/// `Parse.hashCode`: the span and text.
impl Hash for Parse {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.span.hash(state);
        (*self.text).hash(state);
    }
}

/// `Parse.toString`: the covered text.
impl fmt::Display for Parse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.covered_text())
    }
}

/// [`java_type`] for the text after each `(` of an input in turn, in time linear in the input.
///
/// The type ends at the first space, `=` or `-` after the `(`, so the types of the brackets of
/// `((((x` each run to the end; looking for that character again for each `(` takes time in the
/// square of the input's length, as Java's regular expression does. The scan keeps the first such
/// character at or after the last place it looked, which is the first at or after any later place
/// up to it, and the same for the function tag after a `-`.
#[derive(Default)]
struct TypeScan {
    stop: usize,
    tag_stop: usize,
}

impl TypeScan {
    /// The first byte at or after `from` that ends a type, or the input's length.
    fn next_stop(bytes: &[u8], from: usize) -> usize {
        from + bytes[from..].iter().position(|&b| matches!(b, b' ' | b'=' | b'-')).unwrap_or(bytes.len() - from)
    }

    /// `java_type(&input[at..], use_function_tags)`, for `at` that never goes down between calls.
    fn kind<'a>(&mut self, input: &'a str, at: usize, use_function_tags: bool) -> Option<&'a str> {
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

/// `Parse.getType(rest)` for the text after a `(`: an escaped bracket type or `-NONE-` when the
/// text starts with one, otherwise the leading run of characters other than space, `=` and `-`
/// (`^([^ =-]+)`), and with function tags the first tag after it (`^[^ =-]+-([^ =-]+)`). The
/// type is always a prefix of `rest`, so it is returned as a slice of it. [`TypeScan`] gives the
/// same for each `(` of an input in turn.
#[cfg(test)]
fn java_type(rest: &str, use_function_tags: bool) -> Option<&str> {
    if let Some(escaped) = ESCAPED_TYPES.iter().find(|t| rest.starts_with(**t)) {
        return Some(escaped);
    }
    let bytes = rest.as_bytes();
    let run = |from: usize| bytes[from..].iter().take_while(|&&b| !matches!(b, b' ' | b'=' | b'-')).count();
    let n = run(0);
    if n == 0 {
        return None;
    }
    if use_function_tags && bytes.get(n) == Some(&b'-') {
        let m = run(n + 1);
        if m > 0 {
            return Some(&rest[..n + 1 + m]);
        }
    }
    Some(&rest[..n])
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

/// A tag node found by walking the tree: its type, span, head index and the child indices that
/// lead to it from the root.
struct TagPath {
    kind: String,
    span: Span,
    head_index: i32,
    path: Vec<usize>,
}

impl TagPath {
    /// The path of the node's parent.
    fn parent(&self) -> &[usize] {
        &self.path[..self.path.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_type_scan_matches_java_type_at_every_bracket() {
        let inputs = [
            "(TOP (S (NP-SBJ-1 (NNP A)) (VP (VB b))))",
            "((((((x y",
            "(A-((((B-C= (-LRB- -LRB-) (-NONE-1 *) (=X) (-Y) (A--B) (A-",
            "(NP-SBJ=2 (WHNP-1 x)) ( (S (A b)))",
        ];
        for input in inputs {
            for tags in [false, true] {
                let mut scan = TypeScan::default();
                for (ci, _) in input.match_indices('(') {
                    assert_eq!(scan.kind(input, ci + 1, tags), java_type(&input[ci + 1..], tags), "{input} at {ci}");
                }
            }
        }
    }

    #[test]
    fn the_tree_builder_matches_insert() {
        let text: Arc<str> = "a b c d e ".into();
        let spans = [(0, 1), (2, 3), (0, 3), (0, 3), (4, 5), (6, 7), (4, 7), (8, 9), (0, 9), (3, 5), (8, 9), (8, 9)];
        let root = || Parse::new(Arc::clone(&text), Span::new(0, 10), TOP_NODE, 1.0, 0);
        let mut slow = root();
        let mut fast = TreeBuilder::new(root());
        for (i, (s, e)) in spans.into_iter().enumerate() {
            let node = || Parse::new(Arc::clone(&text), Span::new(s, e), format!("N{i}"), 1.0, i as i32);
            slow.insert(node()).unwrap();
            fast.insert(node()).unwrap();
            let (mut a, mut b) = (String::new(), String::new());
            slow.show(&mut a);
            fast.root.show(&mut b);
            assert_eq!(a, b, "after {i}");
        }
    }
}
