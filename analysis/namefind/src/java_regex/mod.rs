//! `java.util.regex.Pattern` syntax, compiled to a direct bytecode matcher.
//!
//! A user's pattern is written for Java, and [`JavaRegex::compile`] gives
//! it Java's meaning or refuses it. The parse follows `Pattern`'s own
//! (`RemoveQEQuoting`, `sequence`, `closure`, `clazz`, `range`, `escape`
//! and `addFlag` in JDK 21), and the output is a compact bytecode program
//! whose every construct means what the Java one does on any text:
//!
//! - `\d`, `\w`, `\s`, `\h`, `\v` and their negations use the classes Java
//!   gives them by default: `\d` is `[0-9]`, `\w` is `[a-zA-Z_0-9]` and
//!   `\s` is `[ \t\n\x0B\f\r]`, ASCII only.
//! - `.` excludes `\n`, `\r`, U+0085, U+2028 and U+2029, as Java's does,
//!   unless `DOTALL` (`(?s)`) is on.
//! - `CASE_INSENSITIVE` (`(?i)`) folds ASCII letters only, as Java does
//!   without `UNICODE_CASE`; each letter and range is expanded to the
//!   characters Java accepts.
//! - `^` is `\A`; `$` and `\Z` also match before a final line terminator,
//!   while `\z` requires the end. `\b` and `\B` use ASCII word characters
//!   plus nonspacing marks with a letter or digit base, as JDK 21 does.
//! - Groups do not report sub-captures.
//!
//! Refused when compiled, with [`Error::Unsupported`]: lookahead,
//! lookbehind, atomic groups, backreferences, possessive quantifiers,
//! nested classes and `&&`, `\p{..}` and `\P{..}`, `\G`, `\R`, `\X`,
//! `\N{..}`, `\b{g}`, surrogate code points written as escapes, a
//! quantifier on an assertion, a loop on a group that can match the empty
//! string, and the flags other than `i` and `s` (`m`, `x`, `u`, `U`, `d`
//! and `c`). A pattern Java itself refuses is [`Error::Syntax`].
//!
//! Some constructs can expose Java's UTF-16 search positions and are
//! refused on supplementary text with [`Error::Text`]:
//!
//! - Java's matcher tries every UTF-16 position, including the second half
//!   of a surrogate pair. `\B` holds there, a class that holds a surrogate
//!   but not every character outside the BMP can match the lone half
//!   there and not the whole character before it, and after an empty
//!   match made with `\b`, `^` or `\A` the next search starts there. Such
//!   a pattern on a text with a character outside the Basic Multilingual
//!   Plane is refused.

mod assertions;
mod class;
mod compile;
mod escape;
mod parse;
mod pike;
#[cfg(test)]
mod tests;

use crate::error::{Error, TextLimit};

use self::class::CharClass;
use self::compile::{Assertion, Compiler, Inst};
use self::parse::parse;
use self::pike::Pike;

/// Maximum number of bytecode instructions a compiled pattern may emit.
const MAX_INSTS: usize = 1 << 20;

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

/// Maximum nesting depth of groups. The parser, the compiler and the
/// syntax tree's drop all recurse once per level, so a deeper pattern is
/// refused with [`Error::Compile`] rather than overflowing the stack.
const MAX_DEPTH: usize = 200;

/// Largest `(pc, pos)` table, in bits (32 MiB), that a backtracking search
/// may use. A search whose table would be larger runs on [`Pike`] instead.
const MAX_MEMO_BITS: usize = 1 << 28;

/// Backtracks at one start position before failed states are recorded.
const MEMO_AFTER_BACKTRACKS: usize = 64;

/// Reusable scratch buffers for [`JavaRegex`] matching.
#[derive(Default)]
pub(crate) struct Scratch {
    stack: Vec<(u32, usize)>,
    /// The `(pc, pos)` states known to fail, for positions at or after the
    /// `from` of the current [`JavaRegex::find_at`]. Without captures a
    /// state fails whatever start reached it, so the set is kept across the
    /// start positions of one search.
    visited: Vec<u64>,
    /// The words of `visited` that hold a bit, so that the next search
    /// clears only what was set rather than the whole table.
    touched: Vec<usize>,
    /// Whether `visited` is in use for the current search.
    memo: bool,
    pike: Pike,
}

/// Which matcher [`JavaRegex::find_at_with`] runs. Only the tests choose
/// one; the finders use `Auto`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum Engine {
    /// Backtracking while the `(pc, pos)` table fits [`MAX_MEMO_BITS`],
    /// the Pike VM otherwise.
    Auto,
    Backtrack,
    Pike,
}

/// A Java regular expression, compiled to a direct bytecode program.
#[derive(Clone, Debug)]
pub struct JavaRegex {
    source: Box<str>,
    flags: Flags,
    insts: Box<[Inst]>,
    classes: Box<[CharClass]>,
    limits: Limits,
}

impl JavaRegex {
    /// `Pattern.compile(pattern)`.
    pub fn compile(pattern: &str) -> Result<JavaRegex, Error> {
        JavaRegex::compile_with(pattern, Flags::NONE)
    }

    /// `Pattern.compile(pattern, flags)`.
    pub fn compile_with(pattern: &str, flags: Flags) -> Result<JavaRegex, Error> {
        let (node, classes, limits) = parse(pattern, flags)?;
        let insts = Compiler::compile(pattern, &node)?;
        Ok(JavaRegex { source: pattern.into(), flags, insts, classes: classes.into_boxed_slice(), limits })
    }

    /// The Java pattern, as given.
    pub fn as_str(&self) -> &str {
        &self.source
    }

    pub fn flags(&self) -> Flags {
        self.flags
    }

    #[cfg(test)]
    pub(crate) fn limits(&self) -> Limits {
        self.limits
    }

    /// The first limit `facts` breaks, as an error.
    pub(crate) fn check(&self, facts: &TextFacts) -> Result<(), Error> {
        let limit = if self.limits.mid_pair && facts.supplementary {
            TextLimit::Supplementary
        } else {
            return Ok(());
        };
        Err(Error::Text { pattern: self.source.clone(), limit })
    }

    /// Finds the leftmost match in `text` starting at or after byte offset
    /// `from`, using reusable `scratch` buffers. The match is the one
    /// `Matcher.find` reports: the first start where the pattern matches,
    /// and there the first match in Java's backtracking order.
    ///
    /// A search takes time and memory linear in the pattern's instructions
    /// times the length of `text` after `from`, apart from Java's backward
    /// nonspacing-mark walks at word assertions; it is never exponential.
    pub(crate) fn find_at(&self, text: &str, from: usize, scratch: &mut Scratch) -> Option<(usize, usize)> {
        self.find_at_with(text, from, scratch, Engine::Auto)
    }

    /// [`JavaRegex::find_at`] on the given matcher. Both find the same
    /// match; the choice only trades speed for memory.
    pub(crate) fn find_at_with(
        &self,
        text: &str,
        from: usize,
        scratch: &mut Scratch,
        engine: Engine,
    ) -> Option<(usize, usize)> {
        if from > text.len() || !text.is_char_boundary(from) {
            return None;
        }
        let anchored = matches!(self.insts.first(), Some(Inst::Assert(Assertion::StartText)));
        if anchored && from > 0 {
            return None;
        }
        let pike = match engine {
            Engine::Auto => !self.table_fits(text.len() - from),
            Engine::Backtrack => false,
            Engine::Pike => true,
        };
        if pike {
            return scratch.pike.find(self, text, from, anchored);
        }
        scratch.memo = false;
        let mut start = from;
        loop {
            if let Some(end) = self.match_at(text, from, start, scratch) {
                return Some((start, end));
            }
            if anchored {
                return None;
            }
            start = next_char(text, start)?.1;
        }
    }

    /// Whether a backtracking search over `len` bytes may use its full
    /// `(pc, pos)` table.
    fn table_fits(&self, len: usize) -> bool {
        len.checked_add(1)
            .and_then(|stride| self.insts.len().checked_mul(stride))
            .is_some_and(|bits| bits <= MAX_MEMO_BITS)
    }

    /// The end of the match at `start_pos`, part of a search from `from`:
    /// a backtracking walk that, once it has backtracked
    /// [`MEMO_AFTER_BACKTRACKS`] times, records each `(pc, pos)` it enters
    /// and never enters one twice. Patterns have no captures and no loop
    /// that can repeat without consuming, so a state entered before has
    /// either failed or is still being explored above, and either way
    /// cannot lead to an earlier match.
    fn match_at(&self, text: &str, from: usize, start_pos: usize, scratch: &mut Scratch) -> Option<usize> {
        let insts = &self.insts;
        let rem = text.len() - from;
        let stride = rem + 1;
        scratch.stack.clear();
        let mut backtracks = 0usize;
        let mut pc = 0usize;
        let mut pos = start_pos;

        loop {
            if scratch.memo {
                let idx = pc * stride + (pos - from);
                let word = idx >> 6;
                let mask = 1u64 << (idx & 63);
                let bits = scratch.visited[word];
                if bits & mask != 0 {
                    let (next_pc, next_pos) = scratch.stack.pop()?;
                    pc = next_pc as usize;
                    pos = next_pos;
                    continue;
                }
                if bits == 0 {
                    scratch.touched.push(word);
                }
                scratch.visited[word] = bits | mask;
            }

            let ok = match insts[pc] {
                Inst::Match => return Some(pos),
                Inst::Jump(target) => {
                    pc = target as usize;
                    continue;
                }
                Inst::Split(prefer, fallback) => {
                    scratch.stack.push((fallback, pos));
                    pc = prefer as usize;
                    continue;
                }
                Inst::Assert(assertion) => {
                    let pass = assertion.holds(text, pos);
                    if pass {
                        pc += 1;
                    }
                    pass
                }
                inst => match next_char(text, pos) {
                    Some((c, next_pos)) if self.consumes(inst, c) => {
                        pos = next_pos;
                        pc += 1;
                        true
                    }
                    _ => false,
                },
            };

            if !ok {
                let (next_pc, next_pos) = scratch.stack.pop()?;
                backtracks += 1;
                if !scratch.memo && backtracks >= MEMO_AFTER_BACKTRACKS && self.table_fits(rem) {
                    scratch.start_memo((insts.len() * stride).div_ceil(64));
                }
                pc = next_pc as usize;
                pos = next_pos;
            }
        }
    }

    /// Whether the character-consuming `inst` accepts `c`.
    #[inline]
    fn consumes(&self, inst: Inst, c: char) -> bool {
        match inst {
            Inst::Literal(want) => c == want,
            Inst::AsciiCase(upper) => c.is_ascii() && (c as u8 & !0x20) == upper,
            Inst::AnyChar => true,
            Inst::NotLineTerminator => !is_line_terminator(c),
            Inst::Class(idx) => self.classes[idx as usize].matches(c),
            Inst::Match | Inst::Jump(_) | Inst::Split(..) | Inst::Assert(_) => false,
        }
    }
}

impl Scratch {
    /// Turns on the `(pc, pos)` table for the rest of the current search,
    /// with `words` words, all clear.
    fn start_memo(&mut self, words: usize) {
        for w in self.touched.drain(..) {
            self.visited[w] = 0;
        }
        if self.visited.len() < words {
            self.visited.resize(words, 0);
        }
        self.memo = true;
    }
}

#[inline]
fn next_char(text: &str, pos: usize) -> Option<(char, usize)> {
    let b = *text.as_bytes().get(pos)?;
    if b < 0x80 {
        Some((b as char, pos + 1))
    } else {
        let c = text.get(pos..)?.chars().next()?;
        Some((c, pos + c.len_utf8()))
    }
}

#[inline]
fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// Whether a text can expose Java's UTF-16 mid-pair search positions.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct TextFacts {
    pub supplementary: bool,
}

impl TextFacts {
    pub fn of(text: &str) -> TextFacts {
        TextFacts { supplementary: !text.is_ascii() && text.chars().any(|c| c as u32 > 0xFFFF) }
    }
}
