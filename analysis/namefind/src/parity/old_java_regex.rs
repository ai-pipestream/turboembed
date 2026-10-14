//! `java_regex.rs` as it was before the bytecode matcher (main at
//! 1fc349f7): the Java pattern translated to the `regex` crate. Kept
//! unchanged apart from this header and the trimmed tests, as the
//! reference the matcher is checked against.

#![allow(dead_code)]

use std::fmt::Write;

use turbo_unicode::GeneralCategory;
use turbo_unicode::java::code_point::{general_category, is_letter_or_digit};

use crate::error::{Error, Feature, TextLimit};

/// The `Pattern.compile` flags this port supports.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `Pattern.CASE_INSENSITIVE`: ASCII letters match either case.
    pub case_insensitive: bool,
    /// `Pattern.DOTALL`: `.` matches line terminators too.
    pub dotall: bool,
}

impl Flags {
    pub const NONE: Flags = Flags { case_insensitive: false, dotall: false };
    pub const CASE_INSENSITIVE: Flags = Flags { case_insensitive: true, dotall: false };
}

/// What a text must not hold for this pattern to match as Java's does.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Limits {
    /// Uses `\b` or `\B`.
    pub word_boundary: bool,
    /// Uses `$` or `\Z`.
    pub dollar: bool,
    /// Can match differently where Java starts at the second half of a
    /// surrogate pair.
    pub mid_pair: bool,
}

/// A Java regular expression, compiled to the `regex` crate.
#[derive(Clone, Debug)]
pub struct JavaRegex {
    source: Box<str>,
    flags: Flags,
    regex: regex::Regex,
    limits: Limits,
}

impl JavaRegex {
    /// `Pattern.compile(pattern)`.
    pub fn compile(pattern: &str) -> Result<JavaRegex, Error> {
        JavaRegex::compile_with(pattern, Flags::NONE)
    }

    /// `Pattern.compile(pattern, flags)`.
    pub fn compile_with(pattern: &str, flags: Flags) -> Result<JavaRegex, Error> {
        let (translated, limits) = translate(pattern, flags)?;
        let regex = regex::RegexBuilder::new(&translated)
            .size_limit(1 << 28)
            .build()
            .map_err(|e| Error::Compile { pattern: pattern.into(), message: e.to_string() })?;
        Ok(JavaRegex { source: pattern.into(), flags, regex, limits })
    }

    /// The Java pattern, as given.
    pub fn as_str(&self) -> &str {
        &self.source
    }

    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// The `regex` crate pattern it was translated to.
    pub fn translated(&self) -> &str {
        self.regex.as_str()
    }

    pub(crate) fn regex(&self) -> &regex::Regex {
        &self.regex
    }

    #[cfg(test)]
    pub(crate) fn limits(&self) -> Limits {
        self.limits
    }

    /// The first limit `facts` breaks, as an error.
    pub(crate) fn check(&self, facts: &TextFacts) -> Result<(), Error> {
        let limit = if self.limits.word_boundary && facts.mark_at_boundary {
            TextLimit::MarkAtBoundary
        } else if self.limits.dollar && facts.final_line_terminator {
            TextLimit::FinalLineTerminator
        } else if self.limits.mid_pair && facts.supplementary {
            TextLimit::Supplementary
        } else {
            return Ok(());
        };
        Err(Error::Text { pattern: self.source.clone(), limit })
    }
}

/// What the limits ask of a text, found in one pass. A nonspacing mark
/// counts at a boundary as `Pattern.Bound` counts it (`hasBaseCharacter`):
/// walking back over UTF-16 units from the mark, a letter or digit comes
/// before anything but nonspacing marks. A mark outside the BMP counts
/// when it follows a letter or digit, since the walk starts at its first
/// unit, though Java sees it as a word character only on its right side.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct TextFacts {
    pub mark_at_boundary: bool,
    pub final_line_terminator: bool,
    pub supplementary: bool,
}

impl TextFacts {
    pub fn of(text: &str) -> TextFacts {
        let mut facts = TextFacts {
            final_line_terminator: text.ends_with(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']),
            ..TextFacts::default()
        };
        if text.is_ascii() {
            return facts;
        }
        // Whether a walk back from the next character would find a letter
        // or digit before anything but nonspacing marks.
        let mut based = false;
        for c in text.chars() {
            let cp = c as u32;
            let mark = is_nonspacing_mark(cp);
            facts.mark_at_boundary |= mark && based;
            facts.supplementary |= cp > 0xFFFF;
            if cp > 0xFFFF {
                // The walk reads this character's second unit as a lone
                // surrogate, which ends it.
                based = false;
            } else if !mark {
                based = is_letter_or_digit(cp);
            }
        }
        facts
    }
}

/// `Character.getType(cp) == NON_SPACING_MARK` in JDK 21: U+1171E became a
/// spacing mark in Unicode 16.0, after the JDK's Unicode 15.0.
fn is_nonspacing_mark(cp: u32) -> bool {
    general_category(cp) == GeneralCategory::Mn || cp == 0x1171E
}

/// The largest code point.
const MAX: u32 = 0x10FFFF;

/// A set of code points as sorted, disjoint, inclusive ranges.
#[derive(Clone, Debug, Default)]
struct Set(Vec<(u32, u32)>);

impl Set {
    fn add(&mut self, lo: u32, hi: u32) {
        self.0.push((lo, hi));
    }

    /// Adds `[lo, hi]`, and with `ci` each ASCII letter whose other case is
    /// in it: Java's `CIRange` and `BitClass.add` without `UNICODE_CASE`.
    fn add_range(&mut self, lo: u32, hi: u32, ci: bool) {
        self.add(lo, hi);
        if ci {
            for c in (b'A'..=b'Z').chain(b'a'..=b'z') {
                let other = (c ^ 0x20) as u32;
                if (lo..=hi).contains(&other) {
                    self.add(c as u32, c as u32);
                }
            }
        }
    }

    fn add_all(&mut self, ranges: &[(u32, u32)]) {
        self.0.extend_from_slice(ranges);
    }

    fn add_complement(&mut self, ranges: &[(u32, u32)]) {
        let mut next = 0;
        for &(lo, hi) in ranges {
            if lo > next {
                self.add(next, lo - 1);
            }
            next = hi + 1;
        }
        if next <= MAX {
            self.add(next, MAX);
        }
    }

    fn normalize(&mut self) {
        self.0.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(self.0.len());
        for &(lo, hi) in &self.0 {
            match out.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                _ => out.push((lo, hi)),
            }
        }
        self.0 = out;
    }

    fn negate(&mut self) {
        self.normalize();
        let ranges = std::mem::take(&mut self.0);
        self.add_complement(&ranges);
    }

    fn intersects(&self, lo: u32, hi: u32) -> bool {
        self.0.iter().any(|&(a, b)| a <= hi && lo <= b)
    }

    fn covers(&self, lo: u32, hi: u32) -> bool {
        self.0.iter().any(|&(a, b)| a <= lo && hi <= b)
    }

    /// Writes the set as a `regex` class. Surrogates are left out: no
    /// `&str` holds one.
    fn write(&self, out: &mut String) {
        let mut any = false;
        out.push('[');
        for &(lo, hi) in &self.0 {
            for (lo, hi) in [(lo, hi.min(0xD7FF)), (lo.max(0xE000), hi)] {
                if lo <= hi {
                    any = true;
                    if lo == hi {
                        let _ = write!(out, "\\x{{{lo:X}}}");
                    } else {
                        let _ = write!(out, "\\x{{{lo:X}}}-\\x{{{hi:X}}}");
                    }
                }
            }
        }
        if !any {
            // A class that matches nothing.
            out.push_str("^\\x{0}-\\x{10FFFF}");
        }
        out.push(']');
    }
}

const DIGIT: &[(u32, u32)] = &[(0x30, 0x39)];
const WORD: &[(u32, u32)] = &[(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)];
const SPACE: &[(u32, u32)] = &[(0x09, 0x0D), (0x20, 0x20)];
const HORIZONTAL: &[(u32, u32)] = &[
    (0x09, 0x09),
    (0x20, 0x20),
    (0xA0, 0xA0),
    (0x1680, 0x1680),
    (0x180E, 0x180E),
    (0x2000, 0x200A),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
];
const VERTICAL: &[(u32, u32)] = &[(0x0A, 0x0D), (0x85, 0x85), (0x2028, 0x2029)];

/// The class a predefined escape letter stands for, and whether negated.
fn predefined(letter: u32) -> Option<(&'static [(u32, u32)], bool)> {
    let class = match char::from_u32(letter)? {
        'd' | 'D' => DIGIT,
        'w' | 'W' => WORD,
        's' | 'S' => SPACE,
        'h' | 'H' => HORIZONTAL,
        'v' | 'V' => VERTICAL,
        _ => return None,
    };
    Some((class, (letter as u8).is_ascii_uppercase()))
}

/// What a parsed piece of pattern can do, for the checks on quantifiers.
#[derive(Copy, Clone, Debug)]
struct Piece {
    /// Can match the empty string.
    nullable: bool,
    /// Matches no characters, only a position.
    assertion: bool,
}

const CHAR: Piece = Piece { nullable: false, assertion: false };
const ASSERTION: Piece = Piece { nullable: true, assertion: true };

/// Translates a Java pattern, returning the `regex` pattern and its limits.
fn translate(pattern: &str, flags: Flags) -> Result<(String, Limits), Error> {
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

struct Translator<'s> {
    source: &'s str,
    /// The pattern as code points, after `\Q..\E` is rewritten.
    p: Vec<u32>,
    at: usize,
    out: String,
    ci: bool,
    dotall: bool,
    limits: Limits,
    /// Named groups so far, to refuse a duplicate as Java does.
    names: Vec<String>,
    /// Uses `\b`, `^` or `\A`, which hold at the start of a character
    /// outside the BMP but not between its halves.
    anchored_empty: bool,
}

impl Translator<'_> {
    fn syntax(&self, message: &'static str) -> Error {
        Error::Syntax { pattern: self.source.into(), at: self.at, message }
    }

    fn unsupported(&self, feature: Feature) -> Error {
        Error::Unsupported { pattern: self.source.into(), at: self.at, feature }
    }

    fn peek(&self) -> Option<u32> {
        self.p.get(self.at).copied()
    }

    fn peek_is(&self, c: char) -> bool {
        self.peek() == Some(c as u32)
    }

    fn peek_at(&self, ahead: usize) -> Option<u32> {
        self.p.get(self.at + ahead).copied()
    }

    fn read(&mut self) -> Option<u32> {
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
    fn literal(&mut self, c: u32) {
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

    /// An escape outside a class, at the backslash.
    fn escape(&mut self) -> Result<Piece, Error> {
        self.at += 1;
        let Some(c) = self.read() else {
            return Err(self.syntax("unexpected internal error"));
        };
        let letter = char::from_u32(c).unwrap_or('\0');
        let assertion = |t: &mut Self, text: &str| {
            t.out.push_str(text);
            Ok(ASSERTION)
        };
        match letter {
            '1'..='9' | 'k' => Err(self.unsupported(Feature::Backreference)),
            'p' | 'P' => Err(self.unsupported(Feature::Property)),
            'A' => {
                self.anchored_empty = true;
                assertion(self, "\\A")
            }
            'z' => assertion(self, "\\z"),
            'Z' => {
                self.limits.dollar = true;
                assertion(self, "\\z")
            }
            'b' => {
                if self.peek_is('{') {
                    return Err(self.unsupported(Feature::Escape('b')));
                }
                self.limits.word_boundary = true;
                self.anchored_empty = true;
                assertion(self, "(?-u:\\b)")
            }
            'B' => {
                self.limits.word_boundary = true;
                self.limits.mid_pair = true;
                assertion(self, "(?-u:\\B)")
            }
            'G' | 'R' | 'X' | 'N' => Err(self.unsupported(Feature::Escape(letter))),
            _ => {
                if let Some((class, negated)) = predefined(c) {
                    let mut set = Set::default();
                    if negated {
                        set.add_complement(class);
                    } else {
                        set.add_all(class);
                    }
                    set.write(&mut self.out);
                    return Ok(CHAR);
                }
                self.at -= 1;
                let c = self.escaped_char()?;
                self.literal(c);
                Ok(CHAR)
            }
        }
    }

    /// The character an escape that names one stands for, at the letter
    /// after the backslash: `escape` for the cases that return a value.
    fn escaped_char(&mut self) -> Result<u32, Error> {
        let Some(c) = self.read() else {
            return Err(self.syntax("unexpected internal error"));
        };
        Ok(match char::from_u32(c).unwrap_or('\0') {
            '0' => self.octal()?,
            'a' => 0x07,
            'e' => 0x1B,
            'f' => 0x0C,
            'n' => 0x0A,
            'r' => 0x0D,
            't' => 0x09,
            'c' => match self.read() {
                Some(x) => x ^ 64,
                None => return Err(self.syntax("illegal control escape sequence")),
            },
            'x' => self.hex()?,
            'u' => self.unicode()?,
            l if l.is_ascii_alphanumeric() => return Err(self.syntax("illegal/unsupported escape sequence")),
            _ => c,
        })
    }

    /// `o()`: one to three octal digits after `\0`, the first of three at
    /// most 3.
    fn octal(&mut self) -> Result<u32, Error> {
        let is_octal = |c: Option<u32>| c.filter(|c| (0x30..=0x37).contains(c)).map(|c| c - 0x30);
        let Some(n) = is_octal(self.peek()) else {
            return Err(self.syntax("illegal octal escape sequence"));
        };
        self.at += 1;
        let Some(m) = is_octal(self.peek()) else { return Ok(n) };
        self.at += 1;
        match is_octal(self.peek()) {
            Some(o) if n <= 3 => {
                self.at += 1;
                Ok(n * 64 + m * 8 + o)
            }
            _ => Ok(n * 8 + m),
        }
    }

    /// `x()`: `\xhh` or `\x{h..}`.
    fn hex(&mut self) -> Result<u32, Error> {
        let hex = |c: Option<u32>| c.and_then(char::from_u32).and_then(|c| c.to_digit(16));
        if let Some(a) = hex(self.peek()) {
            self.at += 1;
            if let Some(b) = hex(self.peek()) {
                self.at += 1;
                return Ok(a * 16 + b);
            }
            return Err(self.syntax("illegal hexadecimal escape sequence"));
        }
        if self.peek_is('{') && hex(self.peek_at(1)).is_some() {
            self.at += 1;
            let mut cp = 0;
            while let Some(d) = hex(self.peek()) {
                self.at += 1;
                cp = (cp << 4) + d;
                if cp > MAX {
                    return Err(self.syntax("hexadecimal codepoint is too big"));
                }
            }
            if !self.peek_is('}') {
                return Err(self.syntax("unclosed hexadecimal escape sequence"));
            }
            self.at += 1;
            if (0xD800..=0xDFFF).contains(&cp) {
                return Err(self.unsupported(Feature::Surrogate));
            }
            return Ok(cp);
        }
        Err(self.syntax("illegal hexadecimal escape sequence"))
    }

    /// `\uhhhh`, joined with a following `\uhhhh` when the two make a
    /// surrogate pair, as Java joins them.
    fn unicode(&mut self) -> Result<u32, Error> {
        let unit = self.four_hex()?;
        if (0xD800..0xDC00).contains(&unit) && self.peek_is('\\') && self.peek_at(1) == Some('u' as u32) {
            let save = self.at;
            self.at += 2;
            let low = self.four_hex()?;
            if (0xDC00..0xE000).contains(&low) {
                return Ok(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00));
            }
            self.at = save;
        }
        if (0xD800..=0xDFFF).contains(&unit) {
            return Err(self.unsupported(Feature::Surrogate));
        }
        Ok(unit)
    }

    fn four_hex(&mut self) -> Result<u32, Error> {
        let mut n = 0;
        for _ in 0..4 {
            match self.read().and_then(char::from_u32).and_then(|c| c.to_digit(16)) {
                Some(d) => n = n * 16 + d,
                None => return Err(self.syntax("illegal Unicode escape sequence")),
            }
        }
        Ok(n)
    }

    /// `clazz`, after the `[`.
    fn class(&mut self) -> Result<Piece, Error> {
        let mut set = Set::default();
        let negated = self.peek_is('^');
        if negated {
            self.at += 1;
        }
        let mut empty = true;
        loop {
            let Some(c) = self.peek() else {
                return Err(self.syntax("unclosed character class"));
            };
            match char::from_u32(c).unwrap_or('\0') {
                '[' => return Err(self.unsupported(Feature::NestedClass)),
                '&' if self.peek_at(1) == Some('&' as u32) => {
                    return Err(self.unsupported(Feature::ClassIntersection));
                }
                ']' if !empty => {
                    self.at += 1;
                    break;
                }
                _ => self.class_item(&mut set)?,
            }
            empty = false;
        }
        if negated {
            set.negate();
        } else {
            set.normalize();
        }
        // Java's matcher can try the second half of a surrogate pair; a
        // class that matches it there but not the whole character at the
        // first half matches where the `regex` crate cannot.
        if set.intersects(0xDC00, 0xDFFF) && !set.covers(0x10000, MAX) {
            self.limits.mid_pair = true;
        }
        set.write(&mut self.out);
        Ok(CHAR)
    }

    /// `range`: one character, range or predefined class of a class.
    fn class_item(&mut self, set: &mut Set) -> Result<(), Error> {
        let start = match self.class_char()? {
            ClassChar::Char(c) => c,
            ClassChar::Class(class, negated) => {
                if negated {
                    set.add_complement(class);
                } else {
                    set.add_all(class);
                }
                return Ok(());
            }
        };
        if self.peek_is('-') {
            match self.peek_at(1).and_then(char::from_u32) {
                Some('[' | ']') | None => {}
                Some(_) => {
                    self.at += 1;
                    let end = match self.class_char_in_range()? {
                        Some(end) if end >= start => end,
                        _ => return Err(self.syntax("illegal character range")),
                    };
                    set.add_range(start, end, self.ci);
                    return Ok(());
                }
            }
        }
        set.add_range(start, start, self.ci);
        Ok(())
    }

    /// One character of a class, or a predefined class.
    fn class_char(&mut self) -> Result<ClassChar, Error> {
        let c = self.read().unwrap_or(0);
        if c != '\\' as u32 {
            return Ok(ClassChar::Char(c));
        }
        let Some(letter) = self.peek() else {
            return Err(self.syntax("unexpected internal error"));
        };
        if letter == 'p' as u32 || letter == 'P' as u32 {
            return Err(self.unsupported(Feature::Property));
        }
        // Java reads \v as U+000B when a '-' follows it, so [\v-] holds
        // U+000B and '-'.
        if letter == 'v' as u32 && self.peek_at(1) == Some('-' as u32) {
            self.at += 1;
            return Ok(ClassChar::Char(0x0B));
        }
        if let Some((class, negated)) = predefined(letter) {
            self.at += 1;
            return Ok(ClassChar::Class(class, negated));
        }
        self.class_escape()
    }

    /// The end of a range: a character, or `None` for a predefined class,
    /// which Java refuses there.
    fn class_char_in_range(&mut self) -> Result<Option<u32>, Error> {
        let c = self.read().unwrap_or(0);
        if c != '\\' as u32 {
            return Ok(Some(c));
        }
        let Some(letter) = self.peek() else {
            return Err(self.syntax("unexpected internal error"));
        };
        if letter == 'v' as u32 {
            self.at += 1;
            return Ok(Some(0x0B));
        }
        if predefined(letter).is_some() {
            self.at += 1;
            return Ok(None);
        }
        match self.class_escape()? {
            ClassChar::Char(c) => Ok(Some(c)),
            ClassChar::Class(..) => Ok(None),
        }
    }

    /// An escape inside a class, at the letter after the backslash.
    fn class_escape(&mut self) -> Result<ClassChar, Error> {
        let letter = self.peek().and_then(char::from_u32).unwrap_or('\0');
        match letter {
            // Assertions and references are not escapes inside a class.
            '1'..='9' | 'A' | 'B' | 'G' | 'R' | 'X' | 'Z' | 'b' | 'k' | 'z' => {
                Err(self.syntax("illegal/unsupported escape sequence"))
            }
            'N' => Err(self.unsupported(Feature::Escape('N'))),
            _ => self.escaped_char().map(ClassChar::Char),
        }
    }
}

enum ClassChar {
    Char(u32),
    Class(&'static [(u32, u32)], bool),
}

fn is_digit(c: u32) -> bool {
    (0x30..=0x39).contains(&c)
}
