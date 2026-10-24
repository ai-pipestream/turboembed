//! Compiled POS-tag and token-literal automaton (`TagPatternRule`) over
//! `(tag, token_surface)` pairs without string concatenation or regexes.

pub mod parser;

use std::cell::RefCell;
use std::fmt;

use turbo_jvm::TagPattern;
use turbo_sequence::chunker::TokenTag;
use turbo_unicode::casefold::fold;

thread_local! {
    /// Reused fold scratch for case-insensitive word matching.
    static FOLD_SCRATCH: RefCell<String> = const { RefCell::new(String::new()) };
}

/// The full case folding of `word` (so `strasse` matches `Straße`).
pub(crate) fn folded(word: &str) -> String {
    let mut out = String::new();
    fold(word, &mut out);
    out
}

use self::parser::{Expr, parse_pattern};

/// Maximum upper bound accepted in a `{m,n}` quantifier to prevent state blowup.
pub(crate) const MAX_QUANTIFIER_BOUND: usize = 256;
/// Maximum total expanded NFA state count permitted for a single compiled pattern rule.
pub(crate) const MAX_AUTOMATON_STATES: usize = 1024;
/// Maximum group nesting depth permitted in a pattern expression.
pub(crate) const MAX_NESTING_DEPTH: usize = 32;

/// Error returned when compiling a [`TagPatternRule`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternError {
    /// Phrase chunk kind label is empty or whitespace-only.
    EmptyKind,
    /// Pattern expression is empty or whitespace-only.
    EmptyPattern,
    /// Pattern can only match zero tokens.
    ZeroLengthPattern,
    /// Unclosed `<...>` tag specification at byte offset `pos`.
    UnclosedTag { pos: usize },
    /// Empty `<>` tag specification at byte offset `pos`.
    EmptyTag { pos: usize },
    /// Unclosed quoted word literal at byte offset `pos`.
    UnclosedQuote { pos: usize },
    /// Unclosed parenthesized group `(` at byte offset `pos`.
    UnclosedGroup { pos: usize },
    /// Unexpected character or token at byte offset `pos`.
    UnexpectedSyntax { pos: usize, detail: String },
    /// Invalid `{m,n}` quantifier at byte offset `pos`.
    InvalidQuantifier { pos: usize, detail: String },
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::EmptyKind => f.write_str("pattern rule chunk kind must not be empty"),
            PatternError::EmptyPattern => f.write_str("pattern expression must not be empty"),
            PatternError::ZeroLengthPattern => f.write_str("pattern expression must match at least one token"),
            PatternError::UnclosedTag { pos } => write!(f, "unclosed '<' tag matcher at byte {pos}"),
            PatternError::EmptyTag { pos } => write!(f, "empty '<>' tag matcher at byte {pos}"),
            PatternError::UnclosedQuote { pos } => write!(f, "unclosed quoted literal at byte {pos}"),
            PatternError::UnclosedGroup { pos } => write!(f, "unclosed '(' group at byte {pos}"),
            PatternError::UnexpectedSyntax { pos, detail } => write!(f, "syntax error at byte {pos}: {detail}"),
            PatternError::InvalidQuantifier { pos, detail } => write!(f, "invalid quantifier at byte {pos}: {detail}"),
        }
    }
}

impl std::error::Error for PatternError {}

/// Single POS-tag matching condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TagCond {
    /// Matches any POS tag (`<.*>` or `<*>`).
    Any,
    /// Matches a POS tag equal to the given string (`<NN>`).
    Exact(String),
    /// Matches a POS tag starting with the given prefix (`<NN.*>` or `<NN*>`).
    Prefix(String),
    /// Matches a POS tag ending with the given suffix (`<.*NN>`).
    Suffix(String),
    /// A Java-style whole-tag pattern (`<JJ[RS]?>`, `<.*NN.*>`) via [`TagPattern`]; `source` is
    /// the pattern text, kept for identity and diagnostics.
    Pattern { source: String, pattern: Box<TagPattern> },
}

impl TagCond {
    #[inline]
    pub fn matches(&self, tag: &str) -> bool {
        match self {
            TagCond::Any => true,
            TagCond::Exact(expected) => tag == expected,
            TagCond::Prefix(prefix) => tag.starts_with(prefix.as_str()),
            TagCond::Suffix(suffix) => tag.ends_with(suffix.as_str()),
            TagCond::Pattern { pattern, .. } => pattern.matches(tag),
        }
    }
}

/// Single token surface-form matching condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WordCond {
    /// Case-sensitive exact token match (`"word"`).
    Exact(String),
    /// Case-insensitive token match (`"word"i`).
    CaseInsensitive(String),
}

impl WordCond {
    #[inline]
    pub fn matches(&self, word: &str) -> bool {
        match self {
            WordCond::Exact(expected) => word == expected,
            WordCond::CaseInsensitive(expected_folded) => {
                if word.is_ascii() && expected_folded.is_ascii() {
                    word.eq_ignore_ascii_case(expected_folded)
                } else {
                    FOLD_SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.clear();
                        fold(word, &mut scratch);
                        *scratch == *expected_folded
                    })
                }
            }
        }
    }
}

/// Predicate evaluated on a single `(POS tag, token surface)` pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMatcher {
    pub tag_alts: Vec<TagCond>,
    pub negate_tag: bool,
    pub word_alts: Vec<WordCond>,
}

impl TokenMatcher {
    /// Matches a single exact POS tag with no surface constraint.
    pub fn exact_tag(tag: impl Into<String>) -> Self {
        Self { tag_alts: vec![TagCond::Exact(tag.into())], negate_tag: false, word_alts: Vec::new() }
    }

    /// Matches any POS tag with the given prefix.
    pub fn prefix_tag(prefix: impl Into<String>) -> Self {
        Self { tag_alts: vec![TagCond::Prefix(prefix.into())], negate_tag: false, word_alts: Vec::new() }
    }

    /// Matches any token whose surface form equals `word` (case-sensitive).
    pub fn exact_word(word: impl Into<String>) -> Self {
        Self { tag_alts: Vec::new(), negate_tag: false, word_alts: vec![WordCond::Exact(word.into())] }
    }

    /// Matches any token whose surface form equals `word` (case-insensitive).
    pub fn word_ignore_case(word: impl Into<String>) -> Self {
        Self {
            tag_alts: Vec::new(),
            negate_tag: false,
            word_alts: vec![WordCond::CaseInsensitive(folded(&word.into()))],
        }
    }

    /// Adds a case-sensitive surface constraint to this matcher.
    pub fn with_word(mut self, word: impl Into<String>) -> Self {
        self.word_alts.push(WordCond::Exact(word.into()));
        self
    }

    /// Adds a case-insensitive surface constraint to this matcher.
    pub fn with_word_ignore_case(mut self, word: impl Into<String>) -> Self {
        self.word_alts.push(WordCond::CaseInsensitive(folded(&word.into())));
        self
    }

    #[inline]
    pub fn matches(&self, tag: &str, word: &str) -> bool {
        let tag_ok = if self.tag_alts.is_empty() {
            true
        } else {
            let hit = self.tag_alts.iter().any(|c| c.matches(tag));
            if self.negate_tag { !hit } else { hit }
        };
        if !tag_ok {
            return false;
        }
        self.word_alts.is_empty() || self.word_alts.iter().any(|w| w.matches(word))
    }
}

#[derive(Clone, Debug)]
enum StepOp {
    Match { matcher: TokenMatcher, out: usize },
    Split { out1: usize, out2: usize },
    Jump { out: usize },
    Accept,
}

/// A compiled declarative POS-tag and token-literal rule that emits chunks of `kind`.
#[derive(Clone, Debug)]
pub struct TagPatternRule {
    kind: String,
    pattern: String,
    ops: Vec<StepOp>,
    start_state: usize,
    accept_state: usize,
    fingerprint: u64,
}

impl TagPatternRule {
    /// Compiles a declarative pattern string (for example `"<DT>?<JJ.*>*<NN.*>+"` or
    /// `"<VB.*><RP|IN>"`) into a token-level automaton for chunk label `kind`.
    pub fn compile(kind: impl Into<String>, pattern: impl Into<String>) -> Result<Self, PatternError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(PatternError::EmptyKind);
        }
        let pattern = pattern.into();
        if pattern.trim().is_empty() {
            return Err(PatternError::EmptyPattern);
        }
        let expr = parse_pattern(&pattern)?;
        Self::from_expr(kind, pattern, &expr)
    }

    /// Builds a rule from an explicit sequence of `(TokenMatcher, min, max)` steps without parsing.
    /// Useful when constructing rules programmatically over arbitrary tag strings.
    pub fn from_steps<I>(kind: impl Into<String>, steps: I) -> Result<Self, PatternError>
    where
        I: IntoIterator<Item = (TokenMatcher, usize, Option<usize>)>,
    {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(PatternError::EmptyKind);
        }
        let mut seq = Vec::new();
        for (idx, (matcher, min, max)) in steps.into_iter().enumerate() {
            if let Some(hi) = max
                && (min > hi || hi == 0 || hi > MAX_QUANTIFIER_BOUND)
            {
                return Err(PatternError::InvalidQuantifier {
                    pos: idx,
                    detail: format!("bounds {{{min},{hi}}} out of range"),
                });
            }
            if min > MAX_QUANTIFIER_BOUND {
                return Err(PatternError::InvalidQuantifier {
                    pos: idx,
                    detail: format!("lower bound {min} exceeds {MAX_QUANTIFIER_BOUND}"),
                });
            }
            seq.push(Expr::Repeat { inner: Box::new(Expr::Atom(matcher)), min, max });
        }
        if seq.is_empty() {
            return Err(PatternError::EmptyPattern);
        }
        let expr = Expr::Seq(seq);
        if !expr.is_valid_non_empty() {
            return Err(PatternError::ZeroLengthPattern);
        }
        if expr.estimated_states().is_none_or(|n| n > MAX_AUTOMATON_STATES) {
            return Err(PatternError::InvalidQuantifier {
                pos: 0,
                detail: format!("compiled automaton exceeds maximum state budget of {MAX_AUTOMATON_STATES}"),
            });
        }
        let desc = format!("programmatic:{kind}:{}", expr_fingerprint(&expr));
        Self::from_expr(kind, desc, &expr)
    }

    fn from_expr(kind: String, pattern: String, expr: &Expr) -> Result<Self, PatternError> {
        let mut ops = vec![StepOp::Accept];
        let accept_state = 0;
        let start_state = compile_expr(expr, accept_state, &mut ops)?;
        let fp = compute_rule_fingerprint(&kind, &pattern, expr);
        Ok(Self { kind, pattern, ops, start_state, accept_state, fingerprint: fp })
    }

    /// Phrase chunk kind label produced by this rule (for example `"NP"`, `"VP"`, `"PP"`).
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Pattern string used to build this rule.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Deterministic 64-bit fingerprint of this rule.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Returns the end token index (`> start`) of the longest match starting at `start`,
    /// or `None` if this rule does not match at `start`.
    pub fn longest_match_at(&self, sentence: &[TokenTag<'_>], start: usize) -> Option<usize> {
        self.longest_match_unclaimed(sentence, start, &[])
    }

    /// Returns the end token index (`> start`) of the longest match starting at `start`
    /// that does not consume any token marked `true` in `occupied`.
    pub fn longest_match_unclaimed(&self, sentence: &[TokenTag<'_>], start: usize, occupied: &[bool]) -> Option<usize> {
        if start >= sentence.len() || occupied.get(start).copied().unwrap_or(false) {
            return None;
        }
        let n_states = self.ops.len();
        let mut current = vec![false; n_states];
        let mut next = vec![false; n_states];
        self.add_epsilon_closure(&mut current, self.start_state);

        let mut best_end = None;
        for (idx, tok) in sentence.iter().enumerate().skip(start) {
            if occupied.get(idx).copied().unwrap_or(false) {
                break;
            }
            next.fill(false);
            let mut any_active = false;
            for (state, &is_active) in current.iter().enumerate() {
                if !is_active {
                    continue;
                }
                if let StepOp::Match { matcher, out } = &self.ops[state]
                    && matcher.matches(tok.tag, tok.token)
                {
                    self.add_epsilon_closure(&mut next, *out);
                    any_active = true;
                }
            }
            if !any_active {
                break;
            }
            if next[self.accept_state] {
                best_end = Some(idx + 1);
            }
            std::mem::swap(&mut current, &mut next);
        }
        best_end
    }

    fn add_epsilon_closure(&self, active: &mut [bool], state: usize) {
        if active[state] {
            return;
        }
        active[state] = true;
        match &self.ops[state] {
            StepOp::Split { out1, out2 } => {
                self.add_epsilon_closure(active, *out1);
                self.add_epsilon_closure(active, *out2);
            }
            StepOp::Jump { out } => {
                self.add_epsilon_closure(active, *out);
            }
            StepOp::Match { .. } | StepOp::Accept => {}
        }
    }
}

fn push_op(ops: &mut Vec<StepOp>, op: StepOp) -> Result<usize, PatternError> {
    if ops.len() >= MAX_AUTOMATON_STATES {
        return Err(PatternError::InvalidQuantifier {
            pos: 0,
            detail: format!("compiled automaton exceeds maximum state budget of {MAX_AUTOMATON_STATES}"),
        });
    }
    let id = ops.len();
    ops.push(op);
    Ok(id)
}

fn compile_expr(expr: &Expr, target: usize, ops: &mut Vec<StepOp>) -> Result<usize, PatternError> {
    match expr {
        Expr::Atom(matcher) => push_op(ops, StepOp::Match { matcher: matcher.clone(), out: target }),
        Expr::Seq(items) => {
            let mut next = target;
            for item in items.iter().rev() {
                next = compile_expr(item, next, ops)?;
            }
            Ok(next)
        }
        Expr::Alt(branches) => {
            let mut iter = branches.iter().rev();
            let Some(last) = iter.next() else {
                return Ok(target);
            };
            let mut acc = compile_expr(last, target, ops)?;
            for b in iter {
                let b_start = compile_expr(b, target, ops)?;
                acc = push_op(ops, StepOp::Split { out1: b_start, out2: acc })?;
            }
            Ok(acc)
        }
        Expr::Repeat { inner, min, max } => {
            let tail = match max {
                None => {
                    let loop_id = push_op(ops, StepOp::Jump { out: target })?;
                    let body_start = compile_expr(inner, loop_id, ops)?;
                    ops[loop_id] = StepOp::Split { out1: body_start, out2: target };
                    loop_id
                }
                Some(hi) => {
                    let mut next = target;
                    for _ in *min..*hi {
                        let copy_start = compile_expr(inner, next, ops)?;
                        next = push_op(ops, StepOp::Split { out1: copy_start, out2: target })?;
                    }
                    next
                }
            };
            let mut entry = tail;
            for _ in 0..*min {
                entry = compile_expr(inner, entry, ops)?;
            }
            Ok(entry)
        }
    }
}

fn expr_fingerprint(expr: &Expr) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    hash_expr(expr, &mut h);
    h
}

fn compute_rule_fingerprint(kind: &str, pattern: &str, expr: &Expr) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    mix_bytes(&mut h, kind.as_bytes());
    mix_bytes(&mut h, pattern.as_bytes());
    hash_expr(expr, &mut h);
    h
}

fn mix_bytes(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    *h = (*h ^ 0xff).wrapping_mul(0x0000_0100_0000_01b3);
}

fn hash_expr(expr: &Expr, h: &mut u64) {
    match expr {
        Expr::Atom(m) => {
            mix_bytes(h, &[1, u8::from(m.negate_tag)]);
            mix_bytes(h, &(m.tag_alts.len() as u64).to_le_bytes());
            for tc in &m.tag_alts {
                let (disc, payload) = match tc {
                    TagCond::Any => (10, "".as_bytes()),
                    TagCond::Exact(s) => (11, s.as_bytes()),
                    TagCond::Prefix(s) => (12, s.as_bytes()),
                    TagCond::Suffix(s) => (13, s.as_bytes()),
                    TagCond::Pattern { source, .. } => (14, source.as_bytes()),
                };
                mix_bytes(h, &[disc]);
                mix_bytes(h, payload);
            }
            mix_bytes(h, &(m.word_alts.len() as u64).to_le_bytes());
            for wc in &m.word_alts {
                match wc {
                    WordCond::Exact(s) => {
                        mix_bytes(h, &[20]);
                        mix_bytes(h, s.as_bytes());
                    }
                    WordCond::CaseInsensitive(s) => {
                        mix_bytes(h, &[21]);
                        mix_bytes(h, s.as_bytes());
                    }
                }
            }
        }
        Expr::Seq(items) => {
            mix_bytes(h, &[2]);
            mix_bytes(h, &(items.len() as u64).to_le_bytes());
            for it in items {
                hash_expr(it, h);
            }
        }
        Expr::Alt(branches) => {
            mix_bytes(h, &[3]);
            mix_bytes(h, &(branches.len() as u64).to_le_bytes());
            for b in branches {
                hash_expr(b, h);
            }
        }
        Expr::Repeat { inner, min, max } => {
            mix_bytes(h, &[4]);
            mix_bytes(h, &(*min as u64).to_le_bytes());
            mix_bytes(h, &(max.unwrap_or(usize::MAX) as u64).to_le_bytes());
            hash_expr(inner, h);
        }
    }
}
