//! `java_regex.rs` as it was before the bytecode matcher (main at
//! 1fc349f7): the Java pattern translated to the `regex` crate. Kept
//! unchanged apart from this header and the trimmed tests, as the
//! reference the matcher is checked against.

#![allow(dead_code)]

mod escape;
mod translate;

use std::fmt::Write;

use turbo_unicode::GeneralCategory;
use turbo_unicode::java::code_point::{general_category, is_letter_or_digit};

use crate::error::{Error, TextLimit};

use self::translate::translate;

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
