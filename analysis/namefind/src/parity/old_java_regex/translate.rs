//! Expression, sequence, quantifier, and group translation for [`super::JavaRegex`].

use std::fmt::Write;

use crate::error::{Error, Feature};

use super::{Flags, Limits};

/// What a parsed piece of pattern can do, for the checks on quantifiers.
#[derive(Copy, Clone, Debug)]
pub(super) struct Piece {
    /// Can match the empty string.
    pub nullable: bool,
    /// Matches no characters, only a position.
    pub assertion: bool,
}

pub(super) const CHAR: Piece = Piece { nullable: false, assertion: false };
pub(super) const ASSERTION: Piece = Piece { nullable: true, assertion: true };

/// Translates a Java pattern, returning the `regex` pattern and its limits.
pub(super) fn translate(pattern: &str, flags: Flags) -> Result<(String, Limits), Error> {
    let mut t = Translator {
        source: pattern,
        p: remove_qe_quoting(pattern.chars().map(|c| c as u32).collect()),
        at: 0,
        out: String::with_capacity(pattern.len() * 2),
        ci: flags.case_insensitive,
        dotall: flags.dotall,
        limits: Limits::default(),
        names: Vec::new(),
        anchored_empty: false,
    };
    let piece = t.expr()?;
    if t.at < t.p.len() {
        // Only an unbalanced ')' stops the top-level expression early.
        return Err(t.syntax("unmatched closing ')'"));
    }
    t.limits.mid_pair |= piece.nullable && t.anchored_empty;
    Ok((t.out, t.limits))
}

/// `Pattern.RemoveQEQuoting`: rewrites `\Q..\E` as escaped characters.
fn remove_qe_quoting(p: Vec<u32>) -> Vec<u32> {
    let len = p.len();
    let mut i = 0;
    while i + 1 < len {
        if p[i] != '\\' as u32 {
            i += 1;
        } else if p[i + 1] != 'Q' as u32 {
            i += 2;
        } else {
            break;
        }
    }
    if i + 1 >= len {
        return p;
    }
    let mut out: Vec<u32> = p[..i].to_vec();
    i += 2;
    let mut in_quote = true;
    let mut begin_quote = true;
    let backslash = '\\' as u32;
    while i < len {
        let c = p[i];
        i += 1;
        if c >= 0x80 || (c as u8).is_ascii_alphabetic() {
            out.push(c);
        } else if (c as u8).is_ascii_digit() {
            if begin_quote {
                // Java writes \x3 first, so a digit cannot extend an escape
                // before the quote.
                out.extend(['\\' as u32, 'x' as u32, '3' as u32]);
            }
            out.push(c);
        } else if c != backslash {
            if in_quote {
                out.push(backslash);
            }
            out.push(c);
        } else if in_quote {
            if p.get(i) == Some(&('E' as u32)) {
                i += 1;
                in_quote = false;
            } else {
                out.extend([backslash, backslash]);
            }
        } else if p.get(i) == Some(&('Q' as u32)) {
            i += 1;
            in_quote = true;
            begin_quote = true;
            continue;
        } else {
            out.push(c);
            if i != len {
                out.push(p[i]);
                i += 1;
            }
        }
        begin_quote = false;
    }
    out
}

pub(super) struct Translator<'s> {
    pub source: &'s str,
    /// The pattern as code points, after `\Q..\E` is rewritten.
    pub p: Vec<u32>,
    pub at: usize,
    pub out: String,
    pub ci: bool,
    pub dotall: bool,
    pub limits: Limits,
    /// Named groups so far, to refuse a duplicate as Java does.
    pub names: Vec<String>,
    /// Uses `\b`, `^` or `\A`, which hold at the start of a character
    /// outside the BMP but not between its halves.
    pub anchored_empty: bool,
}

impl Translator<'_> {
    pub(super) fn syntax(&self, message: &'static str) -> Error {
        Error::Syntax { pattern: self.source.into(), at: self.at, message }
    }

    pub(super) fn unsupported(&self, feature: Feature) -> Error {
        Error::Unsupported { pattern: self.source.into(), at: self.at, feature }
    }

    pub(super) fn peek(&self) -> Option<u32> {
        self.p.get(self.at).copied()
    }

    pub(super) fn peek_is(&self, c: char) -> bool {
        self.peek() == Some(c as u32)
    }

    pub(super) fn peek_at(&self, ahead: usize) -> Option<u32> {
        self.p.get(self.at + ahead).copied()
    }

    pub(super) fn read(&mut self) -> Option<u32> {
        let c = self.peek();
        if c.is_some() {
            self.at += 1;
        }
        c
    }

    /// An alternation: `sequence ('|' sequence)*`.
    fn expr(&mut self) -> Result<Piece, Error> {
        let mut nullable = self.sequence()?.nullable;
        while self.peek_is('|') {
            self.at += 1;
            self.out.push('|');
            nullable |= self.sequence()?.nullable;
        }
        Ok(Piece { nullable, assertion: false })
    }

    /// Atoms and their quantifiers, up to `|`, `)` or the end.
    fn sequence(&mut self) -> Result<Piece, Error> {
        let mut nullable = true;
        while let Some(c) = self.peek() {
            let start = self.out.len();
            let piece = match char::from_u32(c).unwrap_or('\0') {
                '|' | ')' => break,
                '(' => match self.group()? {
                    Some(piece) => piece,
                    // An inline flag group takes no quantifier.
                    None => continue,
                },
                '[' => {
                    self.at += 1;
                    self.class()?
                }
                '\\' => self.escape()?,
                '^' => {
                    self.at += 1;
                    self.anchored_empty = true;
                    self.out.push_str("\\A");
                    ASSERTION
                }
                '$' => {
                    self.at += 1;
                    self.limits.dollar = true;
                    self.out.push_str("\\z");
                    ASSERTION
                }
                '.' => {
                    self.at += 1;
                    if self.dotall {
                        self.out.push_str("(?s:.)");
                    } else {
                        self.out.push_str("[^\\n\\r\\x{85}\\x{2028}\\x{2029}]");
                    }
                    CHAR
                }
                '?' | '*' | '+' => return Err(self.syntax("dangling meta character")),
                // Java's atom is empty here, and the quantifier that must
                // follow repeats the empty string.
                '{' => Piece { nullable: true, assertion: false },
                _ => {
                    self.at += 1;
                    self.literal(c);
                    CHAR
                }
            };
            let piece = self.quantifier(piece, start)?;
            nullable &= piece.nullable;
        }
        Ok(Piece { nullable, assertion: false })
    }

    /// A literal character, folded as Java folds it.
    pub(super) fn literal(&mut self, c: u32) {
        let ascii = c < 0x80 && (c as u8).is_ascii_alphabetic();
        if self.ci && ascii {
            let _ = write!(self.out, "[{}{}]", (c as u8 & !0x20) as char, (c as u8 | 0x20) as char);
        } else if c < 0x80 && (c as u8).is_ascii_alphanumeric() {
            self.out.push(c as u8 as char);
        } else {
            let _ = write!(self.out, "\\x{{{c:X}}}");
        }
    }

    /// `closure`: a quantifier after the piece written from `start`.
    fn quantifier(&mut self, piece: Piece, start: usize) -> Result<Piece, Error> {
        let (min, max) = match self.peek().and_then(char::from_u32) {
            Some('?') => {
                self.at += 1;
                (0, Some(1))
            }
            Some('*') => {
                self.at += 1;
                (0, None)
            }
            Some('+') => {
                self.at += 1;
                (1, None)
            }
            Some('{') => self.counted()?,
            _ => return Ok(piece),
        };
        if piece.assertion {
            return Err(self.unsupported(Feature::QuantifiedAssertion));
        }
        if piece.nullable && max.is_none_or(|m| m > 1) {
            return Err(self.unsupported(Feature::EmptyLoop));
        }
        if self.out.len() == start {
            // The empty atom before a counted quantifier.
            self.out.push_str("(?:)");
        }
        match (min, max) {
            (0, Some(1)) => self.out.push('?'),
            (0, None) => self.out.push('*'),
            (1, None) => self.out.push('+'),
            (min, None) => {
                let _ = write!(self.out, "{{{min},}}");
            }
            (min, Some(max)) => {
                let _ = write!(self.out, "{{{min},{max}}}");
            }
        }
        match self.peek().and_then(char::from_u32) {
            Some('?') => {
                self.at += 1;
                self.out.push('?');
            }
            Some('+') => return Err(self.unsupported(Feature::PossessiveQuantifier)),
            _ => {}
        }
        Ok(Piece { nullable: piece.nullable || min == 0, assertion: false })
    }

    /// `{n}`, `{n,}` or `{n,m}`, at the `{`.
    fn counted(&mut self) -> Result<(u32, Option<u32>), Error> {
        self.at += 1;
        if !self.peek().is_some_and(is_digit) {
            return Err(self.syntax("illegal repetition"));
        }
        let min = self.number()?;
        let max = if self.peek_is(',') {
            self.at += 1;
            if self.peek_is('}') { None } else { Some(self.number()?) }
        } else {
            Some(min)
        };
        if !self.peek_is('}') {
            return Err(self.syntax("unclosed counted closure"));
        }
        self.at += 1;
        if max.is_some_and(|m| m < min) {
            return Err(self.syntax("illegal repetition range"));
        }
        Ok((min, max))
    }

    /// Decimal digits as a Java `int`; none reads as 0, as in Java.
    fn number(&mut self) -> Result<u32, Error> {
        let mut n: i32 = 0;
        while let Some(d) = self.peek().filter(|&d| is_digit(d)) {
            n = n
                .checked_mul(10)
                .and_then(|n| n.checked_add((d - '0' as u32) as i32))
                .ok_or_else(|| self.syntax("illegal repetition range"))?;
            self.at += 1;
        }
        Ok(n as u32)
    }

    /// `group0`, at the `(`. `None` for an inline flag group.
    fn group(&mut self) -> Result<Option<Piece>, Error> {
        self.at += 1;
        let (ci, dotall) = (self.ci, self.dotall);
        if self.peek_is('?') {
            self.at += 1;
            let kind = self.peek().and_then(char::from_u32);
            if matches!(kind, Some(':' | '=' | '!' | '>' | '<' | '$' | '@')) {
                self.at += 1;
            }
            match kind {
                Some(':') => {}
                Some('=' | '!') => return Err(self.unsupported(Feature::Lookahead)),
                Some('>') => return Err(self.unsupported(Feature::AtomicGroup)),
                Some('<') => match self.peek().and_then(char::from_u32) {
                    Some('=' | '!') => return Err(self.unsupported(Feature::Lookbehind)),
                    _ => self.group_name()?,
                },
                Some('$' | '@') => return Err(self.syntax("unknown group type")),
                _ => {
                    self.inline_flags()?;
                    match self.read().and_then(char::from_u32) {
                        Some(')') => return Ok(None),
                        Some(':') => {}
                        _ => return Err(self.syntax("unknown inline modifier")),
                    }
                }
            }
        }
        self.out.push_str("(?:");
        let inner = self.expr()?;
        if !self.peek_is(')') {
            return Err(self.syntax("unclosed group"));
        }
        self.at += 1;
        self.out.push(')');
        self.ci = ci;
        self.dotall = dotall;
        Ok(Some(Piece { nullable: inner.nullable, assertion: false }))
    }

    /// `groupname`: a Latin letter, then letters and digits, then `>`.
    fn group_name(&mut self) -> Result<(), Error> {
        let mut name = String::new();
        match self.read() {
            Some(c) if c < 0x80 && (c as u8).is_ascii_alphabetic() => name.push(c as u8 as char),
            _ => return Err(self.syntax("capturing group name does not start with a Latin letter")),
        }
        loop {
            match self.read() {
                Some(c) if c < 0x80 && (c as u8).is_ascii_alphanumeric() => name.push(c as u8 as char),
                Some(c) if c == '>' as u32 => break,
                _ => return Err(self.syntax("named capturing group is missing trailing '>'")),
            }
        }
        if self.names.contains(&name) {
            return Err(self.syntax("named capturing group is already defined"));
        }
        self.names.push(name);
        Ok(())
    }

    /// `addFlag` and `subFlag`.
    fn inline_flags(&mut self) -> Result<(), Error> {
        let mut on = true;
        while let Some(c) = self.peek().and_then(char::from_u32) {
            match c {
                'i' => self.ci = on,
                's' => self.dotall = on,
                'm' | 'd' | 'u' | 'c' | 'x' | 'U' => return Err(self.unsupported(Feature::Flag(c))),
                '-' if on => on = false,
                _ => return Ok(()),
            }
            self.at += 1;
        }
        Ok(())
    }
}

fn is_digit(c: u32) -> bool {
    (0x30..=0x39).contains(&c)
}
