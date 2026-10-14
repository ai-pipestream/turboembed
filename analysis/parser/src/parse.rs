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

#[inline]
fn is_unicode_whitespace(c: char) -> bool {
    WhitespaceMode::Unicode.is_whitespace(c as u32)
}

#[inline]
fn spans_cross(a: Span, b: Span) -> bool {
    !a.contains(b)
        && !b.contains(a)
        && ((a.start <= b.start && b.start < a.end) || (b.start <= a.start && a.start < b.end))
}

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
/// space (`ParserTool.spaceUntokenizedParens`), leaving existing Unicode whitespace untouched.
pub fn space_untokenized_parens(line: &str) -> Cow<'_, str> {
    let Some(first) = line.find(['(', ')', '{', '}']) else {
        return Cow::Borrowed(line);
    };
    let mut spaced = String::with_capacity(line.len() + 16);
    spaced.push_str(&line[..first]);
    let mut prev_ws = spaced.chars().next_back().is_none_or(is_unicode_whitespace);
    let mut chars = line[first..].chars().peekable();
    while let Some(c) = chars.next() {
        if matches!(c, '(' | ')' | '{' | '}') {
            if !spaced.is_empty() && !prev_ws {
                spaced.push(' ');
            }
            spaced.push(c);
            prev_ws = false;
            if chars.peek().is_some_and(|&next| !is_unicode_whitespace(next)) {
                spaced.push(' ');
                prev_ws = true;
            }
        } else {
            spaced.push(c);
            prev_ws = is_unicode_whitespace(c);
        }
    }
    Cow::Owned(spaced)
}

/// A constituent read while scanning a Penn Treebank bracketed parse (`opennlp.tools.parser.Constituent`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constituent {
    label: String,
    span: Span,
}

impl Constituent {
    /// Creates a constituent with `label` and `span`.
    pub fn new(label: impl Into<String>, span: Span) -> Constituent {
        Constituent { label: label.into(), span }
    }

    /// Returns the constituent label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Updates the constituent label (`Constituent.setLabel`).
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
    }

    /// Returns the byte span of the constituent in the sentence text.
    pub fn span(&self) -> Span {
        self.span
    }
}

/// An error raised while reading or constructing a [`Parse`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A malformed Penn Treebank bracketed string, with its UTF-16 offset (`index`), UTF-8 byte
    /// offset (`byte_offset`), and reason string matching OpenNLP's `malformedParse`.
    MalformedTreebank { index: usize, byte_offset: usize, reason: &'static str },
    /// `Parse::from_tokens` was given an empty token slice.
    EmptyTokens,
    /// `Parse::insert` was given a constituent whose span is outside the receiving node's span.
    NotContained { parent: Span, child: Span },
}

impl ParseError {
    #[inline]
    fn malformed(index: usize, byte_offset: usize, reason: &'static str) -> ParseError {
        ParseError::MalformedTreebank { index, byte_offset, reason }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MalformedTreebank { index, reason, .. } => {
                write!(f, "Malformed treebank parse at index {index}: {reason}")
            }
            ParseError::EmptyTokens => write!(f, "tokens must not be empty"),
            ParseError::NotContained { .. } => {
                write!(f, "Inserting constituent not contained in the sentence!")
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// A constituency parse node (`opennlp.tools.parser.Parse`).
///
/// All nodes in the same sentence share an `Arc<str>` holding the space-joined sentence text, and
/// `span` stores UTF-8 byte offsets into that text.
#[derive(Clone, Debug)]
pub struct Parse {
    text: Arc<str>,
    span: Span,
    kind: String,
    parts: Vec<Parse>,
    /// Which child in `parts` provides this node's head, and whether `head()` follows that child's
    /// own `head()` (`true`, as in English head rules and fallbacks) or stops at the child itself
    /// (`false`, as in AnCora Spanish tag-pattern matches). `None` means `self` is its own head.
    head_child: Option<(usize, bool)>,
    head_index: i32,
    label: Option<String>,
    prob: f64,
    is_chunk: bool,
    derivation: Option<String>,
    prev_punct: Vec<Parse>,
    next_punct: Vec<Parse>,
}

impl Parse {
    /// Creates a parse node (`new Parse(text, span, type, p, index)`).
    pub fn new(text: impl Into<Arc<str>>, span: Span, kind: impl Into<String>, prob: f64, head_index: i32) -> Parse {
        Parse {
            text: text.into(),
            span,
            kind: kind.into(),
            parts: Vec::new(),
            head_child: None,
            head_index,
            label: None,
            prob,
            is_chunk: false,
            derivation: None,
            prev_punct: Vec::new(),
            next_punct: Vec::new(),
        }
    }

    /// Creates a parse node taking its head index from `head` (`new Parse(text, span, type, p, h)`).
    pub fn with_head(
        text: impl Into<Arc<str>>,
        span: Span,
        kind: impl Into<String>,
        prob: f64,
        head: Option<&Parse>,
    ) -> Parse {
        let head_index = head.map_or(0, |h| h.head_index);
        Parse::new(text, span, kind, prob, head_index)
    }

    /// Builds the flat initial parse tree (`INC` root with `TK` children) from pre-tokenized
    /// strings (`Parse.createFromTokens`).
    pub fn from_tokens<S: AsRef<str>>(tokens: &[S]) -> Result<Parse, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::EmptyTokens);
        }
        let total_len: usize = tokens.iter().map(|t| t.as_ref().len()).sum::<usize>() + tokens.len().saturating_sub(1);
        let mut text = String::with_capacity(total_len);
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(tok.as_ref());
        }
        let text: Arc<str> = text.into();
        let mut root = Parse::new(Arc::clone(&text), Span::new(0, text.len() as u32), INC_NODE, 0.0, 0);
        let mut start = 0u32;
        for (i, tok) in tokens.iter().enumerate() {
            let len = tok.as_ref().len() as u32;
            root.insert(Parse::new(Arc::clone(&text), Span::new(start, start + len), TOK_NODE, 0.0, i as i32))?;
            start += len + 1;
        }
        Ok(root)
    }

    /// Reads a Penn Treebank bracketed parse string without function tags or gap labeling
    /// (`Parse.parseParse(parse)`).
    #[allow(clippy::self_named_constructors)]
    pub fn parse(input: &str) -> Result<Parse, ParseError> {
        Self::parse_with_options(input, false, None)
    }

    /// Reads a Penn Treebank bracketed parse string with a [`GapLabeler`]
    /// (`Parse.parseParse(parse, gl)`).
    pub fn parse_with_gaps(input: &str, gap_labeler: &dyn GapLabeler) -> Result<Parse, ParseError> {
        Self::parse_with_options(input, false, Some(gap_labeler))
    }

    /// Reads a Penn Treebank bracketed parse string with explicit `use_function_tags` and optional
    /// [`GapLabeler`].
    pub fn parse_with_options(
        input: &str,
        use_function_tags: bool,
        gap_labeler: Option<&dyn GapLabeler>,
    ) -> Result<Parse, ParseError> {
        let mut text = String::new();
        let mut offset = 0u32;
        let mut roots = 0usize;
        let mut stack: Vec<Constituent> = Vec::new();
        let mut cons: Vec<Constituent> = Vec::new();
        let mut scanner = Scanner::new(input);

        while let Some((b_pos, u_pos, c)) = scanner.next_char() {
            if c == '(' {
                if stack.is_empty() {
                    roots += 1;
                    if roots > 1 {
                        return Err(ParseError::malformed(u_pos, b_pos, "multiple root constituents"));
                    }
                }
                scanner.skip_whitespace();
                let raw_label = scanner.scan_token();
                let file_wrapper = raw_label.is_empty() && scanner.peek() == Some('(') && stack.is_empty();
                if raw_label.is_empty() && !file_wrapper {
                    return Err(ParseError::malformed(u_pos, b_pos, "missing constituent label"));
                }
                let kind = if file_wrapper {
                    TOP_NODE
                } else {
                    extract_type(raw_label, use_function_tags)
                        .ok_or_else(|| ParseError::malformed(u_pos, b_pos, "invalid constituent label"))?
                };
                stack.push(Constituent::new(kind, Span::new(offset, offset)));

                if !file_wrapper {
                    scanner.skip_whitespace();
                }
                let Some(next_ch) = scanner.peek() else {
                    return Err(ParseError::malformed(u_pos, b_pos, "unterminated constituent"));
                };
                if next_ch != '(' && next_ch != ')' {
                    let raw_token = scanner.scan_token();
                    let token_end_byte = scanner.byte_pos;
                    let token_end_u16 = scanner.u16_pos;
                    let token = decode_token(raw_token);
                    scanner.skip_whitespace();
                    if scanner.peek() != Some(')') {
                        return Err(ParseError::malformed(
                            token_end_u16,
                            token_end_byte,
                            "terminal must end after one token",
                        ));
                    }
                    if kind == "-NONE-"
                        && let Some(gl) = gap_labeler
                    {
                        gl.label_gaps(&mut stack);
                    } else {
                        let tok_len = token.len() as u32;
                        cons.push(Constituent::new(TOK_NODE, Span::new(offset, offset + tok_len)));
                        text.push_str(token);
                        text.push(' ');
                        offset += tok_len + 1;
                    }
                }
            } else if c == ')' {
                let Some(con) = stack.pop() else {
                    return Err(ParseError::malformed(u_pos, b_pos, "unmatched closing parenthesis"));
                };
                let start = con.span.start;
                if start < offset {
                    cons.push(Constituent::new(con.label, Span::new(start, offset - 1)));
                }
            } else if !is_unicode_whitespace(c) {
                return Err(ParseError::malformed(u_pos, b_pos, "content outside a constituent"));
            }
        }

        if !stack.is_empty() {
            return Err(ParseError::malformed(scanner.u16_pos, scanner.byte_pos, "unclosed constituent"));
        }
        if roots == 0 {
            return Err(ParseError::malformed(0, 0, "missing root constituent"));
        }

        let txt: Arc<str> = text.into();
        let mut token_index = -1i32;
        let mut p = Parse::new(Arc::clone(&txt), Span::new(0, txt.len() as u32), TOP_NODE, 1.0, 0);
        for con in cons {
            if con.label != TOP_NODE {
                if con.label == TOK_NODE {
                    token_index += 1;
                }
                let c = Parse::new(Arc::clone(&txt), con.span, con.label, 1.0, token_index);
                p.insert(c)?;
            }
        }
        Ok(p)
    }

    /// Inserts `constituent` into this parse tree based on its text span (`Parse.insert`).
    pub fn insert(&mut self, mut constituent: Parse) -> Result<(), ParseError> {
        let ic = constituent.span;
        if !self.span.contains(ic) {
            return Err(ParseError::NotContained { parent: self.span, child: ic });
        }
        let mut pi = 0;
        while pi < self.parts.len() {
            let sp = self.parts[pi].span;
            if sp.start >= ic.end {
                break;
            } else if ic.contains(sp) {
                let sub_part = self.parts.remove(pi);
                constituent.parts.push(sub_part);
            } else if sp.contains(ic) {
                return self.parts[pi].insert(constituent);
            } else {
                pi += 1;
            }
        }
        self.parts.insert(pi, constituent);
        Ok(())
    }

    /// Appends the Penn Treebank bracketed representation of this parse to `out` (`Parse.show`).
    pub fn show(&self, out: &mut String) {
        let mut start = self.span.start as usize;
        if self.kind != TOK_NODE {
            out.push('(');
            out.push_str(&self.kind);
            out.push(' ');
        }
        for c in &self.parts {
            let s = c.span;
            let s_start = s.start as usize;
            if start < s_start {
                out.push_str(encode_token(&self.text[start..s_start]));
            }
            c.show(out);
            start = s.end as usize;
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

    /// Appends `daughter` (and its preceding punctuation) to this node and updates its span and
    /// head (`Parse.add`).
    pub fn add(&mut self, mut daughter: Parse, rules: &dyn HeadRules) {
        let prev = std::mem::take(&mut daughter.prev_punct);
        self.parts.extend(prev);
        let end = daughter.span.end;
        self.parts.push(daughter);
        self.span = Span::new(self.span.start, end);
        self.resolve_head_from_children(rules);
    }

    /// Removes the child at `index` and updates the span when the first or last child is removed
    /// (`Parse.remove`).
    pub fn remove(&mut self, index: usize) {
        self.parts.remove(index);
        if !self.parts.is_empty() && (index == 0 || index == self.parts.len()) {
            self.update_span();
        }
        if let Some((hc, follow)) = self.head_child {
            if hc == index {
                self.head_child = None;
            } else if hc > index {
                self.head_child = Some((hc - 1, follow));
            }
        }
    }

    /// Adjoins `node` to the child at `parse_index` and returns a mutable reference to the new
    /// adjoined parent node (`Parse.adjoinRoot`).
    pub fn adjoin_root(&mut self, mut node: Parse, rules: &dyn HeadRules, parse_index: usize) -> &mut Parse {
        let last_child = self.parts[parse_index].clone();
        let adj_span = Span::new(last_child.span.start, node.span.end);
        let adj_type = last_child.kind.clone();
        let prev = std::mem::take(&mut node.prev_punct);
        let mut adj_node = Parse::new(Arc::clone(&self.text), adj_span, adj_type, 1.0, 0);
        adj_node.parts.push(last_child);
        adj_node.parts.extend(prev);
        adj_node.parts.push(node);
        // In OpenNLP, adjoinRoot computes rules.getHead(new Parse[]{lastChild, node}, lastChild.getType()).
        adj_node.resolve_head_pair(0, adj_node.parts.len() - 1, rules);
        self.parts[parse_index] = adj_node;
        &mut self.parts[parse_index]
    }

    /// Sister-adjoins this node's last child and `sister` and returns a mutable reference to their
    /// new parent node (`Parse.adjoin`).
    pub fn adjoin(&mut self, mut sister: Parse, rules: &dyn HeadRules) -> &mut Parse {
        let last_idx = self.parts.len() - 1;
        let last_child = self.parts[last_idx].clone();
        let adj_span = Span::new(last_child.span.start, sister.span.end);
        let adj_type = last_child.kind.clone();
        let prev = std::mem::take(&mut sister.prev_punct);
        let sister_end = sister.span.end;
        let mut adj_node = Parse::new(Arc::clone(&self.text), adj_span, adj_type, 1.0, 0);
        adj_node.parts.push(last_child);
        adj_node.parts.extend(prev);
        adj_node.parts.push(sister);
        adj_node.resolve_head_pair(0, adj_node.parts.len() - 1, rules);
        self.parts[last_idx] = adj_node;
        self.span = Span::new(self.span.start, sister_end);
        self.resolve_head_from_children(rules);
        &mut self.parts[last_idx]
    }

    /// Moves all children of `self` other than the child at `root_index` into that child and
    /// updates its span (`Parse.expandTopNode`).
    pub fn expand_top_node(&mut self, root_index: usize) {
        let mut drained = std::mem::take(&mut self.parts);
        let mut root = drained.remove(root_index);
        let (before, after) = drained.split_at_mut(root_index);
        let mut new_parts = Vec::with_capacity(before.len() + root.parts.len() + after.len());
        new_parts.extend(before.iter_mut().map(Parse::std_mem_take));
        new_parts.append(&mut root.parts);
        new_parts.extend(after.iter_mut().map(Parse::std_mem_take));
        root.parts = new_parts;
        root.update_span();
        self.parts.push(root);
    }

    fn std_mem_take(p: &mut Parse) -> Parse {
        std::mem::replace(p, Parse::new(Arc::clone(&p.text), Span::new(0, 0), "", 0.0, 0))
    }

    /// Returns the head constituent associated with this node (`Parse.getHead`).
    pub fn head(&self) -> &Parse {
        let mut cur = self;
        while let Some((idx, follow)) = cur.head_child {
            let child = &cur.parts[idx];
            if !follow {
                return child;
            }
            cur = child;
        }
        cur
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

    /// Computes head constituents bottom-up for this parse and all sub-parses (`Parse.updateHeads`).
    pub fn update_heads(&mut self, rules: &dyn HeadRules) {
        if !self.parts.is_empty() {
            for c in &mut self.parts {
                c.update_heads(rules);
            }
            self.resolve_head_from_children(rules);
        } else {
            self.head_child = None;
        }
    }

    fn resolve_head_from_children(&mut self, rules: &dyn HeadRules) {
        let found = rules.get_head(&self.parts, &self.kind).and_then(|h| {
            self.parts.iter().enumerate().find_map(|(i, c)| {
                if std::ptr::eq(h, c) {
                    Some((i, false, h.head_index))
                } else if std::ptr::eq(h, c.head()) {
                    Some((i, true, h.head_index))
                } else {
                    None
                }
            })
        });
        match found {
            Some((idx, follow, head_index)) => {
                self.head_child = Some((idx, follow));
                self.head_index = head_index;
            }
            None => {
                self.head_child = None;
            }
        }
    }

    fn resolve_head_pair(&mut self, first_idx: usize, second_idx: usize, rules: &dyn HeadRules) {
        let pair = [self.parts[first_idx].clone(), self.parts[second_idx].clone()];
        let mapped = rules.get_head(&pair, &self.kind).and_then(|h| {
            [first_idx, second_idx].into_iter().enumerate().find_map(|(pi, real_idx)| {
                if std::ptr::eq(h, &pair[pi]) {
                    Some((real_idx, false, h.head_index))
                } else if std::ptr::eq(h, pair[pi].head()) {
                    Some((real_idx, true, h.head_index))
                } else {
                    None
                }
            })
        });
        if let Some((idx, follow, head_index)) = mapped {
            self.head_child = Some((idx, follow));
            self.head_index = head_index;
        } else {
            self.head_child = None;
        }
    }

    /// Recomputes `span` from the first and last child (`Parse.updateSpan`).
    pub fn update_span(&mut self) {
        if let (Some(first), Some(last)) = (self.parts.first(), self.parts.last()) {
            self.span = Span::new(first.span.start, last.span.end);
        }
    }

    /// Prunes vacuous unary productions (`A -> A`) in place (`Parse.pruneParse`).
    pub fn prune_parse(parse: &mut Parse) {
        for child in &mut parse.parts {
            while child.parts.len() == 1 && child.kind == child.parts[0].kind {
                let grand_child = child.parts.remove(0);
                *child = grand_child;
            }
            Self::prune_parse(child);
        }
    }

    /// Wraps post-possessive sibling tag sequences in an `NP` constituent (`Parse.fixPossesives`).
    pub fn fix_possessives(parse: &mut Parse) {
        let mut tags: Vec<TagPathEntry> = Vec::new();
        let mut path = Vec::new();
        Self::collect_tag_paths(parse, &mut path, &mut tags);
        for ti in 0..tags.len() {
            if tags[ti].kind == "POS"
                && ti + 1 < tags.len()
                && tags[ti].grandparent.as_deref() == Some(tags[ti + 1].parent.as_slice())
            {
                let start = tags[ti + 1].span.start;
                let mut end = tags[ti + 1].span.end;
                for npi in (ti + 2)..tags.len() {
                    if tags[npi].parent == tags[npi - 1].parent {
                        end = tags[npi].span.end;
                    } else {
                        break;
                    }
                }
                let np_pos = Parse::with_head(
                    Arc::clone(&parse.text),
                    Span::new(start, end),
                    "NP",
                    1.0,
                    Some(&tags[ti + 1].node),
                );
                let _ = parse.insert(np_pos);
            }
        }
    }

    fn collect_tag_paths(node: &Parse, path: &mut Vec<usize>, out: &mut Vec<TagPathEntry>) {
        if node.is_pos_tag() {
            let parent = if path.is_empty() { Vec::new() } else { path[..path.len() - 1].to_vec() };
            let grandparent = if path.len() >= 2 { Some(path[..path.len() - 2].to_vec()) } else { None };
            out.push(TagPathEntry {
                kind: node.kind.clone(),
                span: node.span,
                node: node.clone(),
                parent,
                grandparent,
            });
            return;
        }
        for (i, child) in node.parts.iter().enumerate() {
            path.push(i);
            Self::collect_tag_paths(child, path, out);
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
    /// order (`ChunkSampleStream.getInitialChunks`).
    pub fn initial_chunks(&self) -> Vec<&Parse> {
        let mut chunks = Vec::new();
        self.collect_initial_chunks(&mut chunks);
        chunks
    }

    fn collect_initial_chunks<'a>(&'a self, out: &mut Vec<&'a Parse>) {
        if self.is_pos_tag() || (!self.parts.is_empty() && self.parts.iter().all(Parse::is_pos_tag)) {
            out.push(self);
        } else {
            for kid in &self.parts {
                kid.collect_initial_chunks(out);
            }
        }
    }

    /// Inserts named-entity spans (expressed as token index ranges `[start..end)`) into `parse`
    /// under `tag` (`Parse.addNames`).
    pub fn add_names(tag: &str, names: &[Span], parse: &mut Parse) {
        for name_tok_span in names {
            let mut tok_paths: Vec<(Span, i32, Vec<usize>)> = Vec::new();
            let mut path = Vec::new();
            Self::collect_token_paths(parse, &mut path, &mut tok_paths);
            let start_idx = name_tok_span.start as usize;
            let end_idx = name_tok_span.end as usize;
            if end_idx == 0 || end_idx > tok_paths.len() || start_idx >= end_idx {
                continue;
            }
            let (start_span, _, ref start_path) = tok_paths[start_idx];
            let (end_span, end_head_idx, ref end_path) = tok_paths[end_idx - 1];
            let common_path = common_parent_path(start_path, end_path);
            let common_parent = parse.node_at_path_mut(&common_path);
            let name_span = Span::new(start_span.start, end_span.end);
            if name_span == common_parent.span {
                let node = Parse::new(Arc::clone(&common_parent.text), name_span, tag, 1.0, end_head_idx);
                let _ = common_parent.insert(node);
            } else {
                let crossing = common_parent.parts.iter().any(|kid| spans_cross(name_span, kid.span));
                if !crossing {
                    let node = Parse::new(Arc::clone(&common_parent.text), name_span, tag, 1.0, end_head_idx);
                    let _ = common_parent.insert(node);
                } else if common_parent.kind == "NP"
                    && let Some(first_kid) = common_parent.parts.first()
                    && first_kid.parts.len() > 1
                    && name_span.contains(first_kid.parts.last().unwrap().span)
                {
                    let node = Parse::new(
                        Arc::clone(&common_parent.text),
                        common_parent.span,
                        tag,
                        1.0,
                        common_parent.head_index,
                    );
                    let _ = common_parent.insert(node);
                }
            }
        }
    }

    fn collect_token_paths(node: &Parse, path: &mut Vec<usize>, out: &mut Vec<(Span, i32, Vec<usize>)>) {
        for (i, part) in node.parts.iter().enumerate() {
            path.push(i);
            if part.is_pos_tag() || part.kind == TOK_NODE {
                out.push((part.span, part.head_index, path.clone()));
            } else {
                Self::collect_token_paths(part, path, out);
            }
            path.pop();
        }
    }

    fn node_at_path_mut(&mut self, path: &[usize]) -> &mut Parse {
        let mut cur = self;
        for &idx in path {
            cur = &mut cur.parts[idx];
        }
        cur
    }

    /// Returns the punctuation parses immediately preceding this node (`Parse.getPreviousPunctuationSet`).
    pub fn previous_punctuation(&self) -> &[Parse] {
        &self.prev_punct
    }

    /// Adds a punctuation parse preceding this node (`Parse.addPreviousPunctuation`).
    pub fn add_previous_punctuation(&mut self, punct: Parse) {
        if let Err(pos) = self.prev_punct.binary_search_by(|p| cmp_punct(p, &punct)) {
            self.prev_punct.insert(pos, punct);
        }
    }

    /// Sets the punctuation parses preceding this node (`Parse.setPrevPunctuation`).
    pub fn set_previous_punctuation(&mut self, punct: Vec<Parse>) {
        self.prev_punct = punct;
    }

    /// Returns the punctuation parses immediately following this node (`Parse.getNextPunctuationSet`).
    pub fn next_punctuation(&self) -> &[Parse] {
        &self.next_punct
    }

    /// Adds a punctuation parse following this node (`Parse.addNextPunctuation`).
    pub fn add_next_punctuation(&mut self, punct: Parse) {
        if let Err(pos) = self.next_punct.binary_search_by(|p| cmp_punct(p, &punct)) {
            self.next_punct.insert(pos, punct);
        }
    }

    /// Sets the punctuation parses following this node (`Parse.setNextPunctuation`).
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

    /// Compares two parses by descending probability (`Parse.compareTo`).
    pub fn cmp_prob(&self, other: &Parse) -> Ordering {
        other.prob.total_cmp(&self.prob)
    }
}

fn cmp_punct(a: &Parse, b: &Parse) -> Ordering {
    a.span.cmp(&b.span).then_with(|| a.kind.cmp(&b.kind))
}

fn common_parent_path(a: &[usize], b: &[usize]) -> Vec<usize> {
    if a == b {
        return a[..a.len().saturating_sub(1)].to_vec();
    }
    let lcp = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    a[..lcp].to_vec()
}

impl PartialEq for Parse {
    fn eq(&self, other: &Parse) -> bool {
        self.label == other.label
            && self.span == other.span
            && self.kind == other.kind
            && *self.text == *other.text
            && self.parts == other.parts
    }
}

impl Eq for Parse {}

impl Hash for Parse {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.span.hash(state);
        (*self.text).hash(state);
    }
}

impl fmt::Display for Parse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.covered_text())
    }
}

/// Extracts the constituent category and, when `use_function_tags` is `true`, all non-numeric
/// function tags before any numeric coindex (`Parse.getType` on `OPENNLP-XXXX-treebank-scanner`).
fn extract_type(label: &str, use_function_tags: bool) -> Option<&str> {
    if matches!(label, "-LCB-" | "-RCB-" | "-LRB-" | "-RRB-" | "-RSB-" | "-LSB-" | "-NONE-") {
        return Some(label);
    }
    let bytes = label.as_bytes();
    let mut type_end = 0;
    while type_end < bytes.len() && bytes[type_end] != b'-' && bytes[type_end] != b'=' {
        type_end += 1;
    }
    if type_end == 0 {
        return None;
    }
    let mut end = type_end;
    if use_function_tags {
        while end < bytes.len() && bytes[end] == b'-' {
            let start = end + 1;
            let mut next = start;
            let mut numeric = true;
            while next < bytes.len() && bytes[next] != b'-' && bytes[next] != b'=' {
                numeric &= bytes[next].is_ascii_digit();
                next += 1;
            }
            if next == start || numeric {
                break;
            }
            end = next;
        }
    }
    Some(&label[..end])
}

/// Single-pass UTF-8 and UTF-16 cursor over a Penn Treebank parse string.
struct Scanner<'a> {
    src: &'a str,
    byte_pos: usize,
    u16_pos: usize,
}

impl<'a> Scanner<'a> {
    #[inline]
    fn new(src: &'a str) -> Scanner<'a> {
        Scanner { src, byte_pos: 0, u16_pos: 0 }
    }

    #[inline]
    fn peek(&self) -> Option<char> {
        self.src[self.byte_pos..].chars().next()
    }

    #[inline]
    fn next_char(&mut self) -> Option<(usize, usize, char)> {
        let ch = self.peek()?;
        let b_pos = self.byte_pos;
        let u_pos = self.u16_pos;
        self.byte_pos += ch.len_utf8();
        self.u16_pos += ch.len_utf16();
        Some((b_pos, u_pos, ch))
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if !is_unicode_whitespace(ch) {
                break;
            }
            self.byte_pos += ch.len_utf8();
            self.u16_pos += ch.len_utf16();
        }
    }

    #[inline]
    fn scan_token(&mut self) -> &'a str {
        let start = self.byte_pos;
        while let Some(ch) = self.peek() {
            if is_unicode_whitespace(ch) || ch == '(' || ch == ')' {
                break;
            }
            self.byte_pos += ch.len_utf8();
            self.u16_pos += ch.len_utf16();
        }
        &self.src[start..self.byte_pos]
    }
}

struct TagPathEntry {
    kind: String,
    span: Span,
    node: Parse,
    parent: Vec<usize>,
    grandparent: Option<Vec<usize>>,
}
