//! The features of one candidate: OpenNLP's `DefaultSDContextGenerator`.
//!
//! The Java generator indexes the text by UTF-16 code unit, and so does
//! this port, through a [`Utf16Cursor`] that walks the text in place. For
//! a candidate end character it finds:
//!
//! - the prefix: the text from the previous whitespace run, or from the
//!   nearest end character before the candidate (inclusive), up to the
//!   candidate;
//! - the previous word: the text from the whitespace run before that, up
//!   to the start of the prefix;
//! - the suffix: the text after the candidate, up to the last character of
//!   the next whitespace run or the next end character;
//! - the next word: the text after that, up to the following whitespace.
//!
//! Each is trimmed as `String.trim` trims (every code unit at or below
//! U+0020 at either end), which is not the whitespace definition used to
//! find them. Whitespace is the configured [`WhitespaceMode`], since the
//! Java uses `StringUtil.isWhitespace`. The features are then `sp` and
//! `sn` (whitespace before and after the candidate), `eos=` and the
//! character, `x=` the prefix with its length in code units, `xcap` when
//! its first code unit is upper case and `xabbrev` when the prefix and the
//! end character are an abbreviation, and `v=`, `s=`, `n=` for the other
//! three words with their own `cap` and `abbrev` flags.
//!
//! The words can end between the two halves of a character outside the
//! BMP (the Java stops one code unit before the end of the text), so they
//! are written per code unit as modified UTF-8, lone surrogates included,
//! exactly as the JVM looks them up in the model.

use turbo_maxent::Dictionary;
use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};
use turbo_unicode::{Utf16Cursor, WhitespaceMode, java::code_point as java};

use crate::eos::{EndOfSentence, Position, Scanned};
use crate::features::Features;
use crate::units::{AsciiCursor, Units};

/// Buffers for abbreviation lookups, reused across candidates.
#[derive(Clone, Debug, Default)]
pub struct Lookup {
    /// The probe of the `xabbrev` test: the prefix and the end character.
    probe: String,
    /// The lowercased probe of a case-insensitive dictionary.
    lower: String,
}

/// The context generator for one model and whitespace mode.
#[derive(Copy, Clone, Debug)]
pub struct SdContext<'m> {
    eos: &'m EndOfSentence,
    abbreviations: Option<&'m Dictionary>,
    whitespace: WhitespaceMode,
    limits: Limits,
}

/// The longest words worth writing or looking up, in code units. A word
/// feature longer than every predicate of the model adds nothing to the
/// evaluation, and a word longer than every dictionary entry is not an
/// abbreviation, so neither is built. Every code unit is at least one
/// byte of modified UTF-8, and lowercasing per code point keeps the
/// number of code units, so a count of code units is a safe bound.
#[derive(Copy, Clone, Debug)]
pub struct Limits {
    /// The longest predicate, in bytes of modified UTF-8.
    pub feature: usize,
    /// The longest dictionary entry, in code units.
    pub abbreviation: usize,
}

impl Limits {
    /// No limit: every feature is written as the Java writes it.
    pub const NONE: Limits = Limits { feature: usize::MAX, abbreviation: usize::MAX };
}

/// What the whitespace searches of earlier candidates of one text found,
/// so that later candidates do not walk the same stretch again. Without
/// it a text with no whitespace costs a walk back to its start and one on
/// to its end per candidate. Each search returns exactly what the plain
/// walk returns; `Walks::plain` turns the memory off, for tests.
#[derive(Copy, Clone, Debug)]
pub struct Walks {
    enabled: bool,
    /// The last `previous_space` answered with this memory was for the
    /// unit index `back_seek`: the last whitespace unit in
    /// `[1, back_seek)` is at `back_ws` (0 when there is none), and the
    /// answer was `back_answer`. The answer depends only on that unit.
    back_seek: u32,
    back_ws: u32,
    back_answer: Position,
    /// The first whitespace unit at or after unit index `fwd_from` and the
    /// last unit of its run, or `None` when there is no whitespace from
    /// there on. `fwd_from` is `u32::MAX` while nothing is known.
    fwd_from: u32,
    fwd_ws: Option<(Position, Position)>,
}

impl Walks {
    /// The memory for a new text.
    pub fn new() -> Walks {
        Walks {
            enabled: true,
            back_seek: 0,
            back_ws: 0,
            back_answer: Position::default(),
            fwd_from: u32::MAX,
            fwd_ws: None,
        }
    }

    /// No memory: every search walks as the Java does.
    #[cfg(test)]
    pub(crate) fn plain() -> Walks {
        Walks { enabled: false, ..Walks::new() }
    }
}

impl Default for Walks {
    fn default() -> Walks {
        Walks::new()
    }
}

#[inline]
fn position<'t, C: Units<'t>>(c: &C) -> Position {
    Position { byte: c.byte_offset() as u32, unit: c.unit_index() as u32 }
}

/// A stretch of the text between two cursors, `[start, end)` in code units.
#[derive(Clone)]
struct Word<C> {
    start: C,
    end: C,
}

impl<'t, C: Units<'t>> Word<C> {
    /// The stretch trimmed as `String.trim` trims.
    fn trimmed(start: &C, end: &C) -> Word<C> {
        let (mut start, mut end) = (start.clone(), end.clone());
        while start.unit_index() < end.unit_index() && start.unit().is_some_and(|u| u <= 0x20) {
            start.step_forward();
        }
        while end.unit_index() > start.unit_index() && end.peek_prev_unit().is_some_and(|u| u <= 0x20) {
            end.step_back();
        }
        Word { start, end }
    }

    fn len(&self) -> u32 {
        (self.end.unit_index() - self.start.unit_index()) as u32
    }

    fn is_empty(&self) -> bool {
        self.end.unit_index() == self.start.unit_index()
    }

    /// Whether the first code unit is upper case (`Character.isUpperCase`
    /// on a `char`: a high surrogate never is).
    fn first_upper(&self) -> bool {
        self.start.unit().is_some_and(|u| java::is_upper_case(u32::from(u)))
    }

    /// The word as a `&str`, or `None` when it starts or ends inside a
    /// character, and so holds a lone surrogate.
    fn as_str(&self) -> Option<&'t str> {
        (self.start.char_start() && self.end.char_start())
            .then(|| &self.start.text()[self.start.byte_offset()..self.end.byte_offset()])
    }

    /// Appends the code units as modified UTF-8.
    fn push_to(&self, out: &mut Vec<u8>) {
        if self.is_empty() {
            return;
        }
        let text = self.start.text();
        let mut from = self.start.byte_offset();
        if !self.start.char_start() {
            // Starts on a low surrogate: write it alone, then go on from the
            // next character.
            push_modified_utf8(out, u32::from(self.start.unit().expect("not at the end")));
            from += 4;
        }
        let to = self.end.byte_offset();
        if to > from {
            push_str_modified_utf8(out, &text[from..to]);
        }
        if !self.end.char_start() {
            // Ends before a low surrogate: the high one is the last unit.
            push_modified_utf8(out, u32::from(self.end.peek_prev_unit().expect("a unit before")));
        }
    }
}

impl<'m> SdContext<'m> {
    /// `DefaultSDContextGenerator(abbreviations, eosCharacters)` with the
    /// whitespace definition of `whitespace`. `Limits::NONE` builds every
    /// feature the Java builds.
    pub fn new(
        eos: &'m EndOfSentence,
        abbreviations: Option<&'m Dictionary>,
        whitespace: WhitespaceMode,
        limits: Limits,
    ) -> SdContext<'m> {
        SdContext { eos, abbreviations, whitespace, limits }
    }

    #[inline]
    fn is_ws(&self, unit: Option<u16>) -> bool {
        unit.is_some_and(|u| self.whitespace.is_whitespace(u32::from(u)))
    }

    /// Replaces `out` with the features of the candidate at `at`, which
    /// must be a position [`EndOfSentence::positions`] listed for this
    /// text with this context's end set. `walks` must be new for the text
    /// and see its candidates in text order. A text that is all ASCII is
    /// walked by byte.
    pub fn context(&self, text: Scanned<'_>, at: Position, walks: &mut Walks, out: &mut Features, lookup: &mut Lookup) {
        text.debug_check(at);
        if text.is_ascii() {
            self.build(AsciiCursor::at(text.text(), at.byte as usize), text, walks, out, lookup);
        } else {
            self.build(Utf16Cursor::at(text.text(), at.byte as usize, at.unit as usize), text, walks, out, lookup);
        }
    }

    fn build<'t, C: Units<'t>>(
        &self,
        pos: C,
        text: Scanned<'t>,
        walks: &mut Walks,
        out: &mut Features,
        lookup: &mut Lookup,
    ) {
        out.clear();
        let eos = pos.unit().expect("a candidate is a character of the text");
        debug_assert!(self.eos.contains(eos), "a candidate holds an end character");
        let len16 = text.len16() as usize;
        let last_index = len16 - 1;

        if self.is_ws(pos.peek_prev_unit()) {
            out.push(b"sp");
        }
        if self.is_ws(pos.peek_next_unit()) {
            out.push(b"sn");
        }
        out.buf().extend_from_slice(b"eos=");
        match eos {
            0x0A => out.buf().extend_from_slice(b"<LF>"),
            0x0D => out.buf().extend_from_slice(b"<CR>"),
            u => push_modified_utf8(out.buf(), u32::from(u)),
        }
        out.finish();

        // The prefix runs from the previous whitespace run, or from the
        // nearest end character after it, to the candidate.
        let mut prefix_start = self.previous_space(&pos, walks);
        let mut c = pos.clone();
        while c.step_back() && c.unit_index() > prefix_start.unit_index() {
            if self.eos.contains_opt(c.unit()) {
                prefix_start = c;
                break;
            }
        }
        let prefix = Word::trimmed(&prefix_start, &pos);
        let previous = Word::trimmed(&self.previous_space(&prefix_start, walks), &prefix_start);

        let (suffix, next) = if pos.unit_index() == last_index {
            (None, None)
        } else {
            // The suffix runs to the last character of the next whitespace
            // run, or to the next end character.
            let mut suffix_end = self.next_space(&pos, last_index, text, walks);
            let mut c = pos.clone();
            while c.step_forward() && c.unit_index() < suffix_end.unit_index() {
                if self.eos.contains_opt(c.unit()) {
                    suffix_end = c;
                    break;
                }
            }
            let mut after = pos.clone();
            after.step_forward();
            let suffix = Word::trimmed(&after, &suffix_end);
            let mut next_start = suffix_end;
            next_start.step_forward();
            let next_end = self.next_space(&next_start, len16, text, walks);
            (Some(suffix), Some(Word::trimmed(&next_start, &next_end)))
        };

        self.word(out, b"x=", Some(&prefix));
        if !prefix.is_empty() {
            out.push_number(prefix.len());
            if prefix.first_upper() {
                out.push(b"xcap");
            }
            if let Some(dict) = self.abbreviations
                && (prefix.len() as usize) < self.limits.abbreviation
                && let Some(p) = prefix.as_str()
            {
                lookup.probe.clear();
                lookup.probe.push_str(p);
                lookup.probe.push(char::from_u32(u32::from(eos)).expect("not a surrogate"));
                if dict.contains_with(&lookup.probe, &mut lookup.lower) {
                    out.push(b"xabbrev");
                }
            }
        }
        self.flagged(out, b"v=", Some(&previous), b"vcap", b"vabbrev", lookup);
        self.flagged(out, b"s=", suffix.as_ref(), b"scap", b"sabbrev", lookup);
        self.flagged(out, b"n=", next.as_ref(), b"ncap", b"nabbrev", lookup);
    }

    /// Writes `name` and the word (nothing for an absent word), unless the
    /// feature would be longer than every predicate of the model.
    fn word<'t, C: Units<'t>>(&self, out: &mut Features, name: &[u8], word: Option<&Word<C>>) {
        let len = word.map_or(0, |w| w.len() as usize);
        if name.len() + len > self.limits.feature {
            return;
        }
        out.buf().extend_from_slice(name);
        if let Some(w) = word {
            w.push_to(out.buf());
        }
        out.finish();
    }

    /// Writes `name` and the word, then its capitalization and abbreviation
    /// flags when it is not empty.
    fn flagged<'t, C: Units<'t>>(
        &self,
        out: &mut Features,
        name: &[u8],
        word: Option<&Word<C>>,
        cap: &[u8],
        abbrev: &[u8],
        lookup: &mut Lookup,
    ) {
        self.word(out, name, word);
        let Some(w) = word.filter(|w| !w.is_empty()) else { return };
        if w.first_upper() {
            out.push(cap);
        }
        if let Some(dict) = self.abbreviations
            && w.len() as usize <= self.limits.abbreviation
            && let Some(s) = w.as_str()
            && dict.contains_with(s, &mut lookup.lower)
        {
            out.push(abbrev);
        }
    }

    /// `previousSpaceIndex`: from the unit before `seek`, back to the
    /// nearest whitespace, and then to the first unit of that whitespace
    /// run; the start of the text when there is none (whitespace at index
    /// 0 is never looked at). The answer depends only on the last
    /// whitespace unit in `[1, seek)`, which `walks` remembers between
    /// calls, so a walk stops where the previous one started.
    fn previous_space<'t, C: Units<'t>>(&self, seek: &C, walks: &mut Walks) -> C {
        let s = seek.unit_index() as u32;
        if !walks.enabled {
            return self.previous_space_from(seek, 1).0;
        }
        if s < walks.back_seek {
            return if walks.back_ws < s {
                C::place(seek.text(), walks.back_answer)
            } else {
                // Whitespace between `seek` and the remembered seek: rare
                // (a prefix that starts at a whitespace run), and bounded
                // by the word before it.
                self.previous_space_from(seek, 1).0
            };
        }
        let floor = walks.back_seek.max(1) as usize;
        let (c, ws) = self.previous_space_from(seek, floor);
        let answer = match ws {
            Some(ws) => {
                walks.back_ws = ws as u32;
                walks.back_answer = position(&c);
                c
            }
            None => C::place(seek.text(), walks.back_answer),
        };
        walks.back_seek = s;
        answer
    }

    /// The plain walk of `previousSpaceIndex`, looking for whitespace no
    /// lower than unit index `floor` (at least 1). Returns the answer and
    /// the index of the whitespace unit found, or, when there is none down
    /// to `floor`, a cursor at or below `floor` and `None`; with a floor of
    /// 1 that cursor is at index 0, the Java's answer.
    fn previous_space_from<'t, C: Units<'t>>(&self, seek: &C, floor: usize) -> (C, Option<usize>) {
        debug_assert!(floor >= 1);
        let mut c = seek.clone();
        c.step_back();
        while c.unit_index() >= floor && !self.is_ws(c.unit()) {
            c.step_back();
        }
        if c.unit_index() < floor {
            return (c, None);
        }
        let ws = c.unit_index();
        while c.unit_index() > 0 && self.is_ws(c.peek_prev_unit()) {
            c.step_back();
        }
        (c, Some(ws))
    }

    /// `nextSpaceIndex`: from the unit after `seek`, forward to the nearest
    /// whitespace before index `last`, and then to the last unit of that
    /// whitespace run; index `last` when there is none. The first
    /// whitespace at or after a point is remembered in `walks`, so a walk
    /// that reaches a point already searched from stops there.
    fn next_space<'t, C: Units<'t>>(&self, seek: &C, last: usize, text: Scanned<'t>, walks: &mut Walks) -> C {
        if !walks.enabled {
            return self.next_space_plain(seek, last);
        }
        let mut c = seek.clone();
        c.step_forward();
        let start = c.unit_index() as u32;
        let known = walks.fwd_from <= start && walks.fwd_ws.is_none_or(|(ws, _)| ws.unit >= start);
        if !known {
            // Walk to the first whitespace, or to the point the last walk
            // started from, beyond which the answer is known.
            let found = loop {
                if c.unit_index() as u32 == walks.fwd_from {
                    break walks.fwd_ws;
                }
                if self.is_ws(c.unit()) {
                    let ws = position(&c);
                    while self.is_ws(c.peek_next_unit()) {
                        c.step_forward();
                    }
                    break Some((ws, position(&c)));
                }
                if !c.step_forward() {
                    break None;
                }
            };
            walks.fwd_from = start;
            walks.fwd_ws = found;
        }
        self.next_space_answer(text, walks.fwd_ws, last, seek)
    }

    /// The plain walk of `nextSpaceIndex`, as the Java does it.
    fn next_space_plain<'t, C: Units<'t>>(&self, seek: &C, last: usize) -> C {
        let mut c = seek.clone();
        c.step_forward();
        while c.unit_index() < last {
            if self.is_ws(c.unit()) {
                while self.is_ws(c.peek_next_unit()) {
                    c.step_forward();
                }
                return c;
            }
            c.step_forward();
        }
        // The Java returns `last` even when it started past it.
        while c.unit_index() > last {
            c.step_back();
        }
        c
    }

    /// The answer of `nextSpaceIndex` given the first whitespace at or
    /// after the unit after the seek: the end of its run when it is before
    /// `last`, else index `last` (even when the walk started past it).
    fn next_space_answer<'t, C: Units<'t>>(
        &self,
        text: Scanned<'t>,
        ws: Option<(Position, Position)>,
        last: usize,
        seek: &C,
    ) -> C {
        match ws {
            Some((ws, run_end)) if (ws.unit as usize) < last => C::place(seek.text(), run_end),
            _ => {
                let len16 = text.len16() as usize;
                debug_assert!(last == len16 || last + 1 == len16);
                let mut c = C::place(seek.text(), Position { byte: text.text().len() as u32, unit: len16 as u32 });
                if last < len16 {
                    c.step_back();
                }
                c
            }
        }
    }
}

#[cfg(test)]
mod tests;
