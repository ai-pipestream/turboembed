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
//! - `^` is `\A` and `$` is `\z` (see the limits below); `\b` and `\B` are
//!   ASCII word boundaries.
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
//! Three constructs match as Java's do on most texts but not all, and the
//! texts where they differ are refused when matched, with
//! [`Error::Text`]:
//!
//! - `\b` and `\B` treat a nonspacing mark that follows a letter or digit
//!   as a word character in Java (`Pattern.Bound`), even though `\w` does
//!   not match it. A text with such a mark is refused.
//! - `$` and `\Z` also match before a line terminator that ends the text.
//!   A text that ends with one is refused.
//! - Java's matcher tries every UTF-16 position, including the second half
//!   of a surrogate pair. `\B` holds there, a class that holds a surrogate
//!   but not every character outside the BMP can match the lone half
//!   there and not the whole character before it, and after an empty
//!   match made with `\b`, `^` or `\A` the next search starts there. Such
//!   a pattern on a text with a character outside the Basic Multilingual
//!   Plane is refused.

use turbo_unicode::GeneralCategory;
use turbo_unicode::java::code_point::{general_category, is_letter_or_digit};

use crate::error::{Error, Feature, TextLimit};

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

    /// Finds the leftmost match in `text` starting at or after byte offset
    /// `from`, using reusable `scratch` buffers. The match is the one
    /// `Matcher.find` reports: the first start where the pattern matches,
    /// and there the first match in Java's backtracking order.
    ///
    /// A search takes time and memory linear in the pattern's instructions
    /// times the length of `text` after `from`, never exponential.
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
        if from > text.len() {
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
        self.insts.len().saturating_mul(len + 1) <= MAX_MEMO_BITS
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
        let stride = text.len() - from + 1;
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
                    let pass = assertion.holds(text.as_bytes(), pos);
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
                if !scratch.memo && backtracks >= MEMO_AFTER_BACKTRACKS {
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

/// A Pike VM over the same program: every thread advances one character
/// at a time, at most one thread per instruction, kept in priority order,
/// so the first thread to reach [`Inst::Match`] is the match Java's
/// backtracking finds. It needs memory linear in the pattern only, for
/// searches whose `(pc, pos)` table would be too large.
#[derive(Default)]
struct Pike {
    current: Threads,
    next: Threads,
    stack: Vec<u32>,
}

/// A sparse set of instructions, each with the start of its thread, in
/// the order they were added.
#[derive(Default)]
struct Threads {
    dense: Vec<(u32, usize)>,
    sparse: Vec<u32>,
}

impl Threads {
    fn reset(&mut self, insts: usize) {
        self.dense.clear();
        if self.sparse.len() < insts {
            self.sparse.resize(insts, 0);
        }
    }

    #[inline]
    fn contains(&self, pc: u32) -> bool {
        let i = self.sparse[pc as usize] as usize;
        i < self.dense.len() && self.dense[i].0 == pc
    }

    #[inline]
    fn insert(&mut self, pc: u32, start: usize) {
        self.sparse[pc as usize] = self.dense.len() as u32;
        self.dense.push((pc, start));
    }
}

impl Pike {
    fn find(&mut self, re: &JavaRegex, text: &str, from: usize, anchored: bool) -> Option<(usize, usize)> {
        let insts = &re.insts;
        self.current.reset(insts.len());
        self.next.reset(insts.len());
        let mut matched = None;
        let mut pos = from;
        loop {
            // A new start has a lower priority than every thread already
            // running, and none is needed once a match is known.
            if matched.is_none() && !(anchored && pos > 0) {
                Pike::add(insts, &mut self.current, &mut self.stack, 0, pos, text.as_bytes(), pos);
            }
            if self.current.dense.is_empty() && (matched.is_some() || anchored) {
                break;
            }
            let step = next_char(text, pos);
            for i in 0..self.current.dense.len() {
                let (pc, start) = self.current.dense[i];
                match insts[pc as usize] {
                    // Threads after this one have a lower priority.
                    Inst::Match => {
                        matched = Some((start, pos));
                        break;
                    }
                    inst => {
                        if let Some((c, next_pos)) = step
                            && re.consumes(inst, c)
                        {
                            Pike::add(insts, &mut self.next, &mut self.stack, pc + 1, start, text.as_bytes(), next_pos);
                        }
                    }
                }
            }
            let Some((_, next_pos)) = step else { break };
            std::mem::swap(&mut self.current, &mut self.next);
            self.next.dense.clear();
            pos = next_pos;
        }
        matched
    }

    /// Adds the thread at `pc` and everything it reaches without consuming
    /// a character at `pos`, in priority order.
    fn add(insts: &[Inst], list: &mut Threads, stack: &mut Vec<u32>, pc: u32, start: usize, bytes: &[u8], pos: usize) {
        stack.clear();
        stack.push(pc);
        while let Some(mut pc) = stack.pop() {
            loop {
                if list.contains(pc) {
                    break;
                }
                list.insert(pc, start);
                match insts[pc as usize] {
                    Inst::Jump(target) => pc = target,
                    Inst::Split(prefer, fallback) => {
                        stack.push(fallback);
                        pc = prefer;
                    }
                    Inst::Assert(a) if a.holds(bytes, pos) => pc += 1,
                    _ => break,
                }
            }
        }
    }
}

impl Assertion {
    #[inline]
    fn holds(self, bytes: &[u8], pos: usize) -> bool {
        let boundary = || {
            let prev = pos > 0 && is_ascii_word(bytes[pos - 1]);
            let next = pos < bytes.len() && is_ascii_word(bytes[pos]);
            prev != next
        };
        match self {
            Assertion::StartText => pos == 0,
            Assertion::EndText => pos == bytes.len(),
            Assertion::AsciiWordBoundary => boundary(),
            Assertion::AsciiNonWordBoundary => !boundary(),
        }
    }
}

#[inline]
fn next_char(text: &str, pos: usize) -> Option<(char, usize)> {
    let b = *text.as_bytes().get(pos)?;
    if b < 0x80 {
        Some((b as char, pos + 1))
    } else {
        let c = text[pos..].chars().next()?;
        Some((c, pos + c.len_utf8()))
    }
}

#[inline]
fn is_ascii_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
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

/// A compiled character class with an O(1) ASCII bitmap and sorted
/// disjoint non-ASCII code point ranges.
#[derive(Clone, Debug)]
struct CharClass {
    ascii: u128,
    ranges: Box<[(u32, u32)]>,
}

impl CharClass {
    #[inline]
    fn matches(&self, c: char) -> bool {
        let cp = c as u32;
        if cp < 0x80 {
            (self.ascii & (1u128 << cp)) != 0
        } else {
            self.ranges
                .binary_search_by(|&(lo, hi)| {
                    if cp < lo {
                        std::cmp::Ordering::Greater
                    } else if cp > hi {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .is_ok()
        }
    }
}

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

    /// Converts the normalized set into a [`CharClass`]. Surrogates are
    /// excluded because no `&str` holds one.
    fn into_char_class(mut self) -> CharClass {
        self.normalize();
        let mut ascii = 0u128;
        let mut ranges = Vec::new();
        for (lo, hi) in self.0 {
            if lo < 0x80 {
                let ascii_hi = hi.min(0x7F);
                for cp in lo..=ascii_hi {
                    ascii |= 1u128 << cp;
                }
            }
            for (r_lo, r_hi) in [(lo.max(0x80), hi.min(0xD7FF)), (lo.max(0xE000), hi)] {
                if r_lo <= r_hi {
                    ranges.push((r_lo, r_hi));
                }
            }
        }
        CharClass { ascii, ranges: ranges.into_boxed_slice() }
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Assertion {
    StartText,
    EndText,
    AsciiWordBoundary,
    AsciiNonWordBoundary,
}

#[derive(Copy, Clone, Debug)]
enum Inst {
    Match,
    Jump(u32),
    Split(u32, u32),
    Assert(Assertion),
    Literal(char),
    AsciiCase(u8),
    AnyChar,
    NotLineTerminator,
    Class(u32),
}

#[derive(Clone, Debug)]
enum Node {
    Empty,
    Assert(Assertion),
    Literal(char),
    AsciiCase(u8),
    AnyChar,
    NotLineTerminator,
    Class(u32),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Rep { body: Box<Node>, min: u32, max: Option<u32>, greedy: bool },
}

struct Compiler<'s> {
    pattern: &'s str,
    insts: Vec<Inst>,
}

impl<'s> Compiler<'s> {
    fn compile(pattern: &'s str, root: &Node) -> Result<Box<[Inst]>, Error> {
        let mut c = Compiler { pattern, insts: Vec::new() };
        c.emit(root)?;
        c.push(Inst::Match)?;
        Ok(c.insts.into_boxed_slice())
    }

    fn pc(&self) -> u32 {
        self.insts.len() as u32
    }

    fn push(&mut self, inst: Inst) -> Result<u32, Error> {
        if self.insts.len() >= MAX_INSTS {
            return Err(Error::Compile {
                pattern: self.pattern.into(),
                message: format!("compiled pattern exceeds {MAX_INSTS} instructions"),
            });
        }
        let pc = self.insts.len() as u32;
        self.insts.push(inst);
        Ok(pc)
    }

    fn emit(&mut self, node: &Node) -> Result<(), Error> {
        match node {
            Node::Empty => Ok(()),
            Node::Assert(a) => {
                self.push(Inst::Assert(*a))?;
                Ok(())
            }
            Node::Literal(c) => {
                self.push(Inst::Literal(*c))?;
                Ok(())
            }
            Node::AsciiCase(u) => {
                self.push(Inst::AsciiCase(*u))?;
                Ok(())
            }
            Node::AnyChar => {
                self.push(Inst::AnyChar)?;
                Ok(())
            }
            Node::NotLineTerminator => {
                self.push(Inst::NotLineTerminator)?;
                Ok(())
            }
            Node::Class(idx) => {
                self.push(Inst::Class(*idx))?;
                Ok(())
            }
            Node::Seq(items) => {
                for item in items {
                    self.emit(item)?;
                }
                Ok(())
            }
            Node::Alt(branches) => {
                // `Translator::expr` makes an `Alt` of two or more branches.
                let mut end_jumps = Vec::with_capacity(branches.len() - 1);
                for branch in &branches[..branches.len() - 1] {
                    let split_pc = self.push(Inst::Split(0, 0))?;
                    let branch_start = self.pc();
                    self.emit(branch)?;
                    let jump_pc = self.push(Inst::Jump(0))?;
                    end_jumps.push(jump_pc);
                    let next_pc = self.pc();
                    self.insts[split_pc as usize] = Inst::Split(branch_start, next_pc);
                }
                self.emit(&branches[branches.len() - 1])?;
                let end_pc = self.pc();
                for jp in end_jumps {
                    self.insts[jp as usize] = Inst::Jump(end_pc);
                }
                Ok(())
            }
            Node::Rep { body, min, max, greedy } => match *max {
                Some(0) => Ok(()),
                Some(m) => {
                    for _ in 0..*min {
                        self.emit(body)?;
                    }
                    if m > *min {
                        let count = (m - *min) as usize;
                        if count > MAX_INSTS {
                            return Err(Error::Compile {
                                pattern: self.pattern.into(),
                                message: format!("repetition count {m} exceeds limit"),
                            });
                        }
                        let mut splits = Vec::with_capacity(count);
                        for _ in *min..m {
                            let split_pc = self.push(Inst::Split(0, 0))?;
                            let body_start = self.pc();
                            self.emit(body)?;
                            splits.push((split_pc, body_start));
                        }
                        let end_pc = self.pc();
                        for (split_pc, body_start) in splits {
                            self.insts[split_pc as usize] =
                                if *greedy { Inst::Split(body_start, end_pc) } else { Inst::Split(end_pc, body_start) };
                        }
                    }
                    Ok(())
                }
                None => {
                    if *min == 0 {
                        let split_pc = self.push(Inst::Split(0, 0))?;
                        let body_start = self.pc();
                        self.emit(body)?;
                        self.push(Inst::Jump(split_pc))?;
                        let end_pc = self.pc();
                        self.insts[split_pc as usize] =
                            if *greedy { Inst::Split(body_start, end_pc) } else { Inst::Split(end_pc, body_start) };
                    } else {
                        for _ in 0..(*min - 1) {
                            self.emit(body)?;
                        }
                        let body_start = self.pc();
                        self.emit(body)?;
                        let split_pc = self.push(Inst::Split(0, 0))?;
                        let end_pc = self.pc();
                        self.insts[split_pc as usize] =
                            if *greedy { Inst::Split(body_start, end_pc) } else { Inst::Split(end_pc, body_start) };
                    }
                    Ok(())
                }
            },
        }
    }
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

/// Parses a Java pattern into an AST node, character classes, and text limits.
fn parse(pattern: &str, flags: Flags) -> Result<(Node, Vec<CharClass>, Limits), Error> {
    let mut t = Translator {
        source: pattern,
        p: remove_qe_quoting(pattern.chars().map(|c| c as u32).collect()),
        at: 0,
        classes: Vec::new(),
        ci: flags.case_insensitive,
        dotall: flags.dotall,
        limits: Limits::default(),
        names: Vec::new(),
        anchored_empty: false,
        depth: 0,
    };
    let (node, piece) = t.expr()?;
    if t.at < t.p.len() {
        // Only an unbalanced ')' stops the top-level expression early.
        return Err(t.syntax("unmatched closing ')'"));
    }
    t.limits.mid_pair |= piece.nullable && t.anchored_empty;
    Ok((node, t.classes, t.limits))
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
    classes: Vec<CharClass>,
    ci: bool,
    dotall: bool,
    limits: Limits,
    /// Named groups so far, to refuse a duplicate as Java does.
    names: Vec<String>,
    /// Uses `\b`, `^` or `\A`, which hold at the start of a character
    /// outside the BMP but not between its halves.
    anchored_empty: bool,
    /// Groups open at `at`.
    depth: usize,
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

    fn add_class(&mut self, set: Set) -> Node {
        let idx = self.classes.len() as u32;
        self.classes.push(set.into_char_class());
        Node::Class(idx)
    }

    /// An alternation: `sequence ('|' sequence)*`.
    fn expr(&mut self) -> Result<(Node, Piece), Error> {
        let (first, first_piece) = self.sequence()?;
        if !self.peek_is('|') {
            return Ok((first, first_piece));
        }
        let mut branches = vec![first];
        let mut nullable = first_piece.nullable;
        while self.peek_is('|') {
            self.at += 1;
            let (seq, piece) = self.sequence()?;
            branches.push(seq);
            nullable |= piece.nullable;
        }
        Ok((Node::Alt(branches), Piece { nullable, assertion: false }))
    }

    /// Atoms and their quantifiers, up to `|`, `)` or the end.
    fn sequence(&mut self) -> Result<(Node, Piece), Error> {
        let mut nullable = true;
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            let (node, piece) = match char::from_u32(c).unwrap_or('\0') {
                '|' | ')' => break,
                '(' => match self.group()? {
                    Some(res) => res,
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
                    (Node::Assert(Assertion::StartText), ASSERTION)
                }
                '$' => {
                    self.at += 1;
                    self.limits.dollar = true;
                    (Node::Assert(Assertion::EndText), ASSERTION)
                }
                '.' => {
                    self.at += 1;
                    let node = if self.dotall { Node::AnyChar } else { Node::NotLineTerminator };
                    (node, CHAR)
                }
                '?' | '*' | '+' => return Err(self.syntax("dangling meta character")),
                // Java's atom is empty here, and the quantifier that must
                // follow repeats the empty string.
                '{' => (Node::Empty, Piece { nullable: true, assertion: false }),
                _ => {
                    self.at += 1;
                    (self.literal(c), CHAR)
                }
            };
            let (node, piece) = self.quantifier(node, piece)?;
            if !matches!(node, Node::Empty) {
                items.push(node);
            }
            nullable &= piece.nullable;
        }
        let node = match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap(),
            _ => Node::Seq(items),
        };
        Ok((node, Piece { nullable, assertion: false }))
    }

    /// A literal character, folded as Java folds it.
    fn literal(&self, c: u32) -> Node {
        let ascii = c < 0x80 && (c as u8).is_ascii_alphabetic();
        if self.ci && ascii {
            Node::AsciiCase((c as u8) & !0x20)
        } else {
            Node::Literal(char::from_u32(c).expect("surrogates are refused"))
        }
    }

    /// `closure`: an optional quantifier after `node`.
    fn quantifier(&mut self, node: Node, piece: Piece) -> Result<(Node, Piece), Error> {
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
            _ => return Ok((node, piece)),
        };
        if piece.assertion {
            return Err(self.unsupported(Feature::QuantifiedAssertion));
        }
        if piece.nullable && max.is_none_or(|m| m > 1) {
            return Err(self.unsupported(Feature::EmptyLoop));
        }
        let greedy = match self.peek().and_then(char::from_u32) {
            Some('?') => {
                self.at += 1;
                false
            }
            Some('+') => return Err(self.unsupported(Feature::PossessiveQuantifier)),
            _ => true,
        };
        Ok((
            Node::Rep { body: Box::new(node), min, max, greedy },
            Piece { nullable: piece.nullable || min == 0, assertion: false },
        ))
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
    fn group(&mut self) -> Result<Option<(Node, Piece)>, Error> {
        if self.depth == MAX_DEPTH {
            return Err(Error::Compile {
                pattern: self.source.into(),
                message: format!("groups nested deeper than {MAX_DEPTH}"),
            });
        }
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
        self.depth += 1;
        let (inner_node, inner) = self.expr()?;
        self.depth -= 1;
        if !self.peek_is(')') {
            return Err(self.syntax("unclosed group"));
        }
        self.at += 1;
        self.ci = ci;
        self.dotall = dotall;
        Ok(Some((inner_node, Piece { nullable: inner.nullable, assertion: false })))
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
    fn escape(&mut self) -> Result<(Node, Piece), Error> {
        self.at += 1;
        let Some(c) = self.read() else {
            return Err(self.syntax("unexpected internal error"));
        };
        let letter = char::from_u32(c).unwrap_or('\0');
        match letter {
            '1'..='9' | 'k' => Err(self.unsupported(Feature::Backreference)),
            'p' | 'P' => Err(self.unsupported(Feature::Property)),
            'A' => {
                self.anchored_empty = true;
                Ok((Node::Assert(Assertion::StartText), ASSERTION))
            }
            'z' => Ok((Node::Assert(Assertion::EndText), ASSERTION)),
            'Z' => {
                self.limits.dollar = true;
                Ok((Node::Assert(Assertion::EndText), ASSERTION))
            }
            'b' => {
                if self.peek_is('{') {
                    return Err(self.unsupported(Feature::Escape('b')));
                }
                self.limits.word_boundary = true;
                self.anchored_empty = true;
                Ok((Node::Assert(Assertion::AsciiWordBoundary), ASSERTION))
            }
            'B' => {
                self.limits.word_boundary = true;
                self.limits.mid_pair = true;
                Ok((Node::Assert(Assertion::AsciiNonWordBoundary), ASSERTION))
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
                    let node = self.add_class(set);
                    return Ok((node, CHAR));
                }
                self.at -= 1;
                let c = self.escaped_char()?;
                Ok((self.literal(c), CHAR))
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
    fn class(&mut self) -> Result<(Node, Piece), Error> {
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
        // first half matches where UTF-8 character scanning cannot.
        if set.intersects(0xDC00, 0xDFFF) && !set.covers(0x10000, MAX) {
            self.limits.mid_pair = true;
        }
        let node = self.add_class(set);
        Ok((node, CHAR))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn find(p: &str, text: &str) -> Option<(usize, usize)> {
        let r = JavaRegex::compile(p).unwrap();
        let mut scratch = Scratch::default();
        r.find_at(text, 0, &mut scratch)
    }

    fn feature(p: &str) -> Feature {
        match JavaRegex::compile(p) {
            Err(Error::Unsupported { feature, .. }) => feature,
            other => panic!("{p}: {other:?}"),
        }
    }

    fn syntax(p: &str) {
        assert!(matches!(JavaRegex::compile(p), Err(Error::Syntax { .. })), "{p}");
    }

    #[test]
    fn classes_are_ascii() {
        assert_eq!(find(r"\d", "٠7"), Some(("٠".len(), "٠7".len())));
        assert_eq!(find(r"[\w-]", "é-"), Some(("é".len(), "é-".len())));
        assert_eq!(find("a.b", "a\nb a\u{85}b axb"), Some((9, 12)));
        assert_eq!(find("(?s).", "\n"), Some((0, 1)));
    }

    #[test]
    fn case_folds_ascii_letters_only() {
        assert_eq!(find("(?i)aé[x-z]", "AéZ"), Some((0, 4)));
        assert_eq!(find("(?i)aé[x-z]", "AÉZ"), None);
        let r = JavaRegex::compile_with("[k]", Flags::CASE_INSENSITIVE).unwrap();
        let mut scratch = Scratch::default();
        assert_eq!(r.find_at("K", 0, &mut scratch), Some((0, 1)));
        assert_eq!(r.find_at("\u{212A}", 0, &mut scratch), None);
        // Flags end with their group.
        assert_eq!(find("((?i)a)a", "Aa"), Some((0, 2)));
        assert_eq!(find("((?i)a)a", "AA"), None);
    }

    #[test]
    fn quoting_and_escapes() {
        assert_eq!(find(r"\Qa.b\E*", "a.bb"), Some((0, 4)));
        assert_eq!(find(r"\Qa.b\E*", "axb"), None);
        assert_eq!(find(r"[\Q]\E]", "]"), Some((0, 1)));
        assert_eq!(find(r"\0101\x41\u0041\cA", "AAA\x01"), Some((0, 4)));
        assert_eq!(find(r"\0400", " 0"), Some((0, 2)));
        assert_eq!(find(r"\uD83D\uDE00", "😀"), Some((0, 4)));
        assert_eq!(find(r"[\v-]", "\x0B-"), Some((0, 1)));
        assert_eq!(find("[]a]", "]"), Some((0, 1)));
        assert_eq!(find("{1}", "abc"), Some((0, 0)));
    }

    #[test]
    fn refusals() {
        assert_eq!(feature("(?=a)"), Feature::Lookahead);
        assert_eq!(feature("(?<!a)b"), Feature::Lookbehind);
        assert_eq!(feature("(?>a)"), Feature::AtomicGroup);
        assert_eq!(feature(r"(a)\1"), Feature::Backreference);
        assert_eq!(feature("a*+"), Feature::PossessiveQuantifier);
        assert_eq!(feature("(a*)*"), Feature::EmptyLoop);
        assert_eq!(feature("^*"), Feature::QuantifiedAssertion);
        assert_eq!(feature("[a[b]]"), Feature::NestedClass);
        assert_eq!(feature("[a&&b]"), Feature::ClassIntersection);
        assert_eq!(feature(r"\p{L}"), Feature::Property);
        assert_eq!(feature(r"\uD800"), Feature::Surrogate);
        assert_eq!(feature("(?m)a"), Feature::Flag('m'));
        assert_eq!(feature(r"\R"), Feature::Escape('R'));
        for p in ["{", "a{", "a{1", "x{,2}", "a{2,1}", "[]", "(?i)*", "a**", r"\y", r"\E", "(?<a>x)(?<a>y)"] {
            syntax(p);
        }
        for p in [r"\0", r"\08", r"\c", r"\x4", r"\x{110000}", "(?<1a>x)", "x{2147483648}", "(a", "a)", "[a-\\d]"] {
            syntax(p);
        }
    }

    #[test]
    fn limits() {
        assert_eq!(JavaRegex::compile("a$").unwrap().limits(), Limits { dollar: true, ..Limits::default() });
        let b = JavaRegex::compile(r"\bx").unwrap().limits();
        assert!(b.word_boundary && !b.mid_pair);
        assert!(JavaRegex::compile(r"\b|x").unwrap().limits().mid_pair);
        assert!(JavaRegex::compile(r"\Bx").unwrap().limits().mid_pair);
        assert!(JavaRegex::compile(r"[\x00-\uFFFF]").unwrap().limits().mid_pair);
        assert!(!JavaRegex::compile(r"[^a]").unwrap().limits().mid_pair);
        assert!(JavaRegex::compile(r"[^\x{10000}-\x{10FFFF}]").unwrap().limits().mid_pair);
    }

    #[test]
    fn compile_limit_and_backtracking_memoization() {
        assert!(matches!(JavaRegex::compile("a{2000000}"), Err(Error::Compile { .. })));
        // Triggers the visited bitset (> 64 backtracks at a single start position) without exponential blowup.
        assert_eq!(find("(a|a)+b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaac"), None);
        assert_eq!(find("(a|a)+b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"), Some((0, 31)));
        // The failed states are kept across start positions, so a search
        // that fails at every start is linear, not quadratic, in the text.
        let long = "a".repeat(20_000);
        assert_eq!(find("(a|aa)*c", &long), None);
        assert_eq!(find(".*x", &long), None);
        assert_eq!(find("(a|aa)*c|a$", &long), Some((19_999, 20_000)));
        let mut scratch = Scratch::default();
        for (p, want) in [("(a|aa)*c", None), (".*x", None), ("(a|aa)*c|a$", Some((19_999, 20_000)))] {
            let r = JavaRegex::compile(p).unwrap();
            assert_eq!(r.find_at_with(&long, 0, &mut scratch, Engine::Pike), want, "{p}");
        }
    }

    #[test]
    fn large_tables_run_on_the_pike_vm() {
        // 200,001 instructions over 2,001 positions is past the table's
        // limit, so the search keeps no table at all.
        let r = JavaRegex::compile("a{0,100000}!").unwrap();
        assert!(r.table_fits(1_000) && !r.table_fits(2_000));
        let text = "x".repeat(2_000);
        let mut scratch = Scratch::default();
        assert_eq!(r.find_at(&text, 0, &mut scratch), None);
        assert!(scratch.visited.is_empty());
        let text = format!("{}aa!", "x".repeat(2_000));
        assert_eq!(r.find_at(&text, 0, &mut scratch), Some((2_000, 2_003)));
        assert_eq!(r.find_at(&text, 2_001, &mut scratch), Some((2_001, 2_003)));
    }

    #[test]
    fn the_table_is_cleared_between_searches() {
        // Each search turns the table on; a state left from the one before
        // would hide the later matches.
        let r = JavaRegex::compile("(a|a)+b").unwrap();
        let text = format!("{}c aab ", "a".repeat(30)).repeat(3);
        let mut scratch = Scratch::default();
        let mut from = 0;
        let mut found = Vec::new();
        while let Some((s, e)) = r.find_at(&text, from, &mut scratch) {
            found.push(s);
            from = e;
        }
        assert_eq!(found, [32, 68, 104]);
        assert!(!scratch.touched.is_empty());
    }

    #[test]
    fn nesting_is_limited() {
        let nested = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
        assert_eq!(find(&nested(MAX_DEPTH), "a"), Some((0, 1)));
        assert!(matches!(JavaRegex::compile(&nested(MAX_DEPTH + 1)), Err(Error::Compile { .. })));
        assert!(matches!(JavaRegex::compile(&"(".repeat(100_000)), Err(Error::Compile { .. })));
    }
}
