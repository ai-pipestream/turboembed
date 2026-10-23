//! Declarative POS-tag and token-literal pattern parser.

use turbo_jvm::TagPattern;

use super::{
    MAX_AUTOMATON_STATES, MAX_NESTING_DEPTH, MAX_QUANTIFIER_BOUND, PatternError, TagCond, TokenMatcher, WordCond,
    folded,
};

#[derive(Clone, Debug)]
pub(super) enum Expr {
    Atom(TokenMatcher),
    Seq(Vec<Expr>),
    Alt(Vec<Expr>),
    Repeat { inner: Box<Expr>, min: usize, max: Option<usize> },
}

impl Expr {
    pub(super) fn min_tokens(&self) -> usize {
        match self {
            Expr::Atom(_) => 1,
            Expr::Seq(items) => items.iter().map(Expr::min_tokens).fold(0usize, usize::saturating_add),
            Expr::Alt(branches) => branches.iter().map(Expr::min_tokens).min().unwrap_or(0),
            Expr::Repeat { inner, min, .. } => min.saturating_mul(inner.min_tokens()),
        }
    }

    pub(super) fn is_valid_non_empty(&self) -> bool {
        self.min_tokens() > 0 && !self.has_nullable_repeat()
    }

    pub(super) fn estimated_states(&self) -> Option<usize> {
        match self {
            Expr::Atom(_) => Some(1),
            Expr::Seq(items) => items.iter().try_fold(0usize, |acc, it| acc.checked_add(it.estimated_states()?)),
            Expr::Alt(branches) => {
                branches.iter().try_fold(branches.len(), |acc, b| acc.checked_add(b.estimated_states()?))
            }
            Expr::Repeat { inner, min, max } => {
                let sub = inner.estimated_states()?;
                let copies = match max {
                    Some(hi) => *hi,
                    None => min.checked_add(1)?,
                };
                let extra_splits = max.map_or(1usize, |hi| hi.saturating_sub(*min));
                copies.checked_mul(sub)?.checked_add(extra_splits)
            }
        }
    }

    fn has_nullable_repeat(&self) -> bool {
        match self {
            Expr::Atom(_) => false,
            Expr::Seq(items) => items.iter().any(Expr::has_nullable_repeat),
            Expr::Alt(branches) => branches.iter().any(Expr::has_nullable_repeat),
            Expr::Repeat { inner, max, .. } => {
                (max.is_none() && inner.min_tokens() == 0) || inner.has_nullable_repeat()
            }
        }
    }
}

pub(super) fn parse_pattern(pattern: &str) -> Result<Expr, PatternError> {
    let mut parser = Parser { src: pattern, pos: 0, depth: 0 };
    let expr = parser.parse_alt()?;
    parser.skip_ws();
    if parser.pos < parser.src.len() {
        return Err(PatternError::UnexpectedSyntax {
            pos: parser.pos,
            detail: format!("unexpected trailing input {:?}", &parser.src[parser.pos..]),
        });
    }
    if !expr.is_valid_non_empty() {
        return Err(PatternError::ZeroLengthPattern);
    }
    if expr.estimated_states().is_none_or(|n| n > MAX_AUTOMATON_STATES) {
        return Err(PatternError::InvalidQuantifier {
            pos: 0,
            detail: format!("compiled automaton exceeds maximum state budget of {MAX_AUTOMATON_STATES}"),
        });
    }
    Ok(expr)
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn parse_alt(&mut self) -> Result<Expr, PatternError> {
        let mut branches = vec![self.parse_seq()?];
        loop {
            self.skip_ws();
            if self.peek() == Some('|') {
                self.pos += 1;
                branches.push(self.parse_seq()?);
            } else {
                break;
            }
        }
        if branches.len() == 1 { Ok(branches.pop().unwrap()) } else { Ok(Expr::Alt(branches)) }
    }

    fn parse_seq(&mut self) -> Result<Expr, PatternError> {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None | Some(')') | Some('|') => break,
                Some('<') => {
                    let atom = self.parse_tag_atom()?;
                    items.push(self.parse_quantified(Expr::Atom(atom))?);
                }
                Some('"') | Some('\'') => {
                    let atom = self.parse_word_atom()?;
                    items.push(self.parse_quantified(Expr::Atom(atom))?);
                }
                Some('(') => {
                    let open_pos = self.pos;
                    if self.depth >= MAX_NESTING_DEPTH {
                        return Err(PatternError::InvalidQuantifier {
                            pos: open_pos,
                            detail: format!("group nesting depth exceeds {MAX_NESTING_DEPTH}"),
                        });
                    }
                    self.pos += 1;
                    self.depth += 1;
                    let inner = self.parse_alt()?;
                    self.depth -= 1;
                    self.skip_ws();
                    if self.peek() != Some(')') {
                        return Err(PatternError::UnclosedGroup { pos: open_pos });
                    }
                    self.pos += 1;
                    items.push(self.parse_quantified(inner)?);
                }
                Some(ch) => {
                    return Err(PatternError::UnexpectedSyntax {
                        pos: self.pos,
                        detail: format!("unexpected character {ch:?}"),
                    });
                }
            }
        }
        if items.is_empty() {
            return Err(PatternError::UnexpectedSyntax { pos: self.pos, detail: "empty sequence branch".into() });
        }
        if items.len() == 1 { Ok(items.pop().unwrap()) } else { Ok(Expr::Seq(items)) }
    }

    fn parse_quantified(&mut self, inner: Expr) -> Result<Expr, PatternError> {
        self.skip_ws();
        let q_pos = self.pos;
        match self.peek() {
            Some('?') => {
                self.pos += 1;
                Ok(Expr::Repeat { inner: Box::new(inner), min: 0, max: Some(1) })
            }
            Some('*') => {
                self.pos += 1;
                Ok(Expr::Repeat { inner: Box::new(inner), min: 0, max: None })
            }
            Some('+') => {
                self.pos += 1;
                Ok(Expr::Repeat { inner: Box::new(inner), min: 1, max: None })
            }
            Some('{') => {
                self.pos += 1;
                let (min, max) = self.parse_brace_bounds(q_pos)?;
                Ok(Expr::Repeat { inner: Box::new(inner), min, max })
            }
            _ => Ok(inner),
        }
    }

    fn parse_brace_bounds(&mut self, open_pos: usize) -> Result<(usize, Option<usize>), PatternError> {
        let close_rel = self.src[self.pos..]
            .find('}')
            .ok_or_else(|| PatternError::InvalidQuantifier { pos: open_pos, detail: "unclosed '{'".into() })?;
        let body = self.src[self.pos..self.pos + close_rel].trim();
        self.pos += close_rel + 1;
        if body.is_empty() {
            return Err(PatternError::InvalidQuantifier { pos: open_pos, detail: "empty '{}' quantifier".into() });
        }
        let (min, max) = if let Some((lo_s, hi_s)) = body.split_once(',') {
            let lo: usize = lo_s.trim().parse().map_err(|_| PatternError::InvalidQuantifier {
                pos: open_pos,
                detail: format!("invalid lower bound {:?}", lo_s.trim()),
            })?;
            let hi_trimmed = hi_s.trim();
            let hi = if hi_trimmed.is_empty() {
                None
            } else {
                Some(hi_trimmed.parse::<usize>().map_err(|_| PatternError::InvalidQuantifier {
                    pos: open_pos,
                    detail: format!("invalid upper bound {hi_trimmed:?}"),
                })?)
            };
            (lo, hi)
        } else {
            let exact: usize = body.parse().map_err(|_| PatternError::InvalidQuantifier {
                pos: open_pos,
                detail: format!("invalid repeat count {body:?}"),
            })?;
            (exact, Some(exact))
        };
        if min > MAX_QUANTIFIER_BOUND {
            return Err(PatternError::InvalidQuantifier {
                pos: open_pos,
                detail: format!("lower bound {min} exceeds {MAX_QUANTIFIER_BOUND}"),
            });
        }
        if let Some(hi) = max {
            if hi > MAX_QUANTIFIER_BOUND {
                return Err(PatternError::InvalidQuantifier {
                    pos: open_pos,
                    detail: format!("upper bound {hi} exceeds {MAX_QUANTIFIER_BOUND}"),
                });
            }
            if min > hi || hi == 0 {
                return Err(PatternError::InvalidQuantifier {
                    pos: open_pos,
                    detail: format!("invalid bounds {{{min},{hi}}}"),
                });
            }
        }
        Ok((min, max))
    }

    fn parse_word_atom(&mut self) -> Result<TokenMatcher, PatternError> {
        let wc = self.parse_quoted_word()?;
        Ok(TokenMatcher { tag_alts: Vec::new(), negate_tag: false, word_alts: vec![wc] })
    }

    fn parse_quoted_word(&mut self) -> Result<WordCond, PatternError> {
        let quote_pos = self.pos;
        let quote = self.peek().unwrap();
        self.pos += quote.len_utf8();
        let mut buf = String::new();
        let mut closed = false;
        while let Some(c) = self.peek() {
            self.pos += c.len_utf8();
            if c == '\\' {
                let esc = self.peek().ok_or(PatternError::UnclosedQuote { pos: quote_pos })?;
                self.pos += esc.len_utf8();
                buf.push(esc);
            } else if c == quote {
                closed = true;
                break;
            } else {
                buf.push(c);
            }
        }
        if !closed {
            return Err(PatternError::UnclosedQuote { pos: quote_pos });
        }
        if self.peek() == Some('i') {
            self.pos += 1;
            Ok(WordCond::CaseInsensitive(folded(&buf)))
        } else {
            Ok(WordCond::Exact(buf))
        }
    }

    fn parse_tag_atom(&mut self) -> Result<TokenMatcher, PatternError> {
        let open_pos = self.pos;
        self.pos += 1;
        let mut raw = String::new();
        let mut closed = false;
        while let Some(c) = self.peek() {
            self.pos += c.len_utf8();
            if c == '\\' {
                let esc = self.peek().ok_or(PatternError::UnclosedTag { pos: open_pos })?;
                self.pos += esc.len_utf8();
                raw.push('\\');
                raw.push(esc);
            } else if c == '>' {
                closed = true;
                break;
            } else {
                raw.push(c);
            }
        }
        if !closed {
            return Err(PatternError::UnclosedTag { pos: open_pos });
        }
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(PatternError::EmptyTag { pos: open_pos });
        }

        let (tag_part, word_part) = split_unescaped_once(trimmed, &[':', '=']);
        let (negate_tag, tag_body) = if let Some(rest) = tag_part.strip_prefix('!')
            && !rest.trim().is_empty()
        {
            (true, rest.trim())
        } else {
            (false, tag_part.trim())
        };

        let mut tag_alts = Vec::new();
        for piece in split_unescaped(tag_body, '|') {
            let p = piece.trim();
            if p.is_empty() {
                return Err(PatternError::EmptyTag { pos: open_pos });
            }
            tag_alts.push(parse_tag_cond(p, open_pos)?);
        }

        let mut word_alts = Vec::new();
        if let Some(wp) = word_part {
            let mut sub = Parser { src: wp, pos: 0, depth: 0 };
            loop {
                sub.skip_ws();
                if matches!(sub.peek(), Some('"') | Some('\'')) {
                    word_alts.push(sub.parse_quoted_word()?);
                } else {
                    return Err(PatternError::UnexpectedSyntax {
                        pos: open_pos,
                        detail: "expected quoted word literal after ':' or '=' in tag matcher".into(),
                    });
                }
                sub.skip_ws();
                if sub.peek() == Some('|') {
                    sub.pos += 1;
                } else {
                    sub.skip_ws();
                    if sub.pos < sub.src.len() {
                        return Err(PatternError::UnexpectedSyntax {
                            pos: open_pos,
                            detail: format!("unexpected trailing text {:?} in tag word matcher", &sub.src[sub.pos..]),
                        });
                    }
                    break;
                }
            }
        }

        Ok(TokenMatcher { tag_alts, negate_tag, word_alts })
    }
}

fn split_unescaped_once<'a>(s: &'a str, delims: &[char]) -> (&'a str, Option<&'a str>) {
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if delims.contains(&c) {
            return (&s[..i], Some(&s[i + c.len_utf8()..]));
        }
    }
    (s, None)
}

fn split_unescaped(s: &str, delim: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == delim {
            out.push(&s[start..i]);
            start = i + c.len_utf8();
        }
    }
    out.push(&s[start..]);
    out
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(esc) = chars.next() {
                out.push(esc);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// One tag alternative. Plain tags and the documented `.*` / `*`
/// shorthands keep their fast forms; anything holding another
/// metacharacter compiles through the shared Java tag-pattern
/// reader, and what that reader refuses is an error at the tag's
/// position, never a silent literal.
fn parse_tag_cond(raw: &str, pos: usize) -> Result<TagCond, PatternError> {
    if raw == ".*" || raw == "*" {
        return Ok(TagCond::Any);
    }
    let plain = |s: &str| {
        !s.chars().any(|c| matches!(c, '*' | '[' | ']' | '?' | '+' | '.' | '(' | ')' | '{' | '}' | '^' | '$' | '\\'))
    };
    if let Some(prefix) = raw.strip_suffix(".*").filter(|p| plain(p)) {
        return Ok(TagCond::Prefix(unescape(prefix)));
    }
    if let Some(prefix) = raw.strip_suffix('*').filter(|p| !p.is_empty() && plain(p)) {
        return Ok(TagCond::Prefix(unescape(prefix)));
    }
    if let Some(suffix) = raw.strip_prefix(".*").filter(|s| plain(s)) {
        return Ok(TagCond::Suffix(unescape(suffix)));
    }
    if plain(raw) {
        return Ok(TagCond::Exact(unescape(raw)));
    }
    TagPattern::compile(raw)
        .map(|pattern| TagCond::Pattern { source: raw.into(), pattern: Box::new(pattern) })
        .map_err(|detail| PatternError::UnexpectedSyntax { pos, detail: format!("tag pattern {raw:?}: {detail}") })
}
