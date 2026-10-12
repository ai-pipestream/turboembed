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
pub(crate) struct SdContext<'m> {
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
pub(crate) struct Limits {
    /// The longest predicate, in bytes of modified UTF-8.
    pub(crate) feature: usize,
    /// The longest dictionary entry, in code units.
    pub(crate) abbreviation: usize,
}

impl Limits {
    /// No limit: every feature is written as the Java writes it.
    pub(crate) const NONE: Limits = Limits { feature: usize::MAX, abbreviation: usize::MAX };
}

/// What the whitespace searches of earlier candidates of one text found,
/// so that later candidates do not walk the same stretch again. Without
/// it a text with no whitespace costs a walk back to its start and one on
/// to its end per candidate. Each search returns exactly what the plain
/// walk returns; `Walks::plain` turns the memory off, for tests.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Walks {
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
    pub(crate) fn new() -> Walks {
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
    pub(crate) fn new(
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
    pub(crate) fn context(
        &self,
        text: Scanned<'_>,
        at: Position,
        walks: &mut Walks,
        out: &mut Features,
        lookup: &mut Lookup,
    ) {
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
mod tests {
    use super::*;

    /// The features of the candidate at UTF-16 index `unit`, as strings
    /// (lone surrogates shown as `\uXXXX`).
    fn features(text: &str, unit: usize, dict: Option<&Dictionary>, mode: WhitespaceMode) -> Vec<String> {
        let eos = EndOfSentence::default();
        let ctx = SdContext::new(&eos, dict, mode, Limits::NONE);
        let mut positions = Vec::new();
        let scanned = eos.positions(text, &mut positions);
        let at = *positions.iter().find(|p| p.unit as usize == unit).expect("a candidate at that index");
        let mut out = Features::new();
        ctx.context(scanned, at, &mut Walks::new(), &mut out, &mut Lookup::default());
        out.iter().map(show).collect()
    }

    /// Modified UTF-8 back to a string, a surrogate written as an escape.
    fn show(bytes: &[u8]) -> String {
        let mut s = String::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0xED && i + 2 < bytes.len() && bytes[i + 1] >= 0xA0 {
                let u = (u32::from(bytes[i] & 0x0F) << 12)
                    | (u32::from(bytes[i + 1] & 0x3F) << 6)
                    | u32::from(bytes[i + 2] & 0x3F);
                s.push_str(&format!("\\u{u:04X}"));
                i += 3;
            } else {
                let len = match bytes[i] {
                    0..0x80 => 1,
                    0xC0..0xE0 => 2,
                    _ => 3,
                };
                s.push_str(std::str::from_utf8(&bytes[i..i + len]).unwrap());
                i += len;
            }
        }
        s
    }

    fn dictionary() -> Dictionary {
        Dictionary::read(
            br#"<dictionary case_sensitive="false"><entry><token>Mr.</token></entry><entry><token>U.S.</token></entry></dictionary>"#,
        )
        .unwrap()
    }

    const UNICODE: WhitespaceMode = WhitespaceMode::Unicode;

    #[test]
    fn plain_sentence_end() {
        // Worked from the Java: "Fine." at the end of the text.
        assert_eq!(
            features("It was fine. Then", 11, None, UNICODE),
            ["sn", "eos=.", "x=fine", "4", "v=was", "s=", "n=Then", "ncap"]
        );
        assert_eq!(
            features("sharp. Fine.", 11, None, UNICODE),
            ["eos=.", "x=Fine", "4", "xcap", "v=sharp.", "s=", "n="]
        );
    }

    #[test]
    fn abbreviations_from_the_dictionary() {
        let dict = dictionary();
        // From the fixture context file: the prefix with the end character
        // is looked up, and the next word as it stands.
        assert_eq!(
            features("MR. SMITH", 2, Some(&dict), UNICODE),
            ["sn", "eos=.", "x=MR", "2", "xcap", "xabbrev", "v=", "s=", "n=SMITH", "ncap"]
        );
        assert_eq!(
            features("jones met. Mr. X", 9, Some(&dict), UNICODE),
            ["sn", "eos=.", "x=met", "3", "v=jones", "s=", "n=Mr.", "ncap", "nabbrev"]
        );
        // Without a dictionary there are no abbreviation flags.
        assert_eq!(
            features("MR. SMITH", 2, None, UNICODE),
            ["sn", "eos=.", "x=MR", "2", "xcap", "v=", "s=", "n=SMITH", "ncap"]
        );
    }

    #[test]
    fn runs_of_end_characters() {
        // The prefix starts at the nearest end character before the
        // candidate, and includes it; the suffix stops at the next one.
        assert_eq!(
            features("at 9 A.M. sharp", 6, None, UNICODE),
            ["eos=.", "x=A", "1", "xcap", "v=9", "s=M", "scap", "n=sharp"]
        );
        assert_eq!(
            features("at 9 A.M. sharp", 8, None, UNICODE),
            ["sn", "eos=.", "x=.M", "2", "v=A", "vcap", "s=", "n=sharp"]
        );
        // The previous word is everything back to the whitespace before
        // the prefix, end characters and all.
        assert_eq!(
            features("A.M. sharp. Fine.", 10, None, UNICODE),
            ["sn", "eos=.", "x=sharp", "5", "v=A.M.", "vcap", "s=", "n=Fine.", "ncap"]
        );
    }

    #[test]
    fn line_breaks_are_escaped() {
        let eos = EndOfSentence::from_chars("\n.").unwrap();
        let ctx = SdContext::new(&eos, None, UNICODE, Limits::NONE);
        let mut positions = Vec::new();
        let scanned = eos.positions("a\nb", &mut positions);
        let mut out = Features::new();
        ctx.context(scanned, positions[0], &mut Walks::new(), &mut out, &mut Lookup::default());
        // The Java computes the next word up to one unit before the end of
        // the text here, so the "b" is in neither the suffix nor the next word.
        assert_eq!(out.iter().map(show).collect::<Vec<_>>(), ["eos=<LF>", "x=a", "1", "v=", "s=", "n="]);
    }

    #[test]
    fn a_lone_high_surrogate_after_the_last_period() {
        // From the fixture context file: the suffix stops one code unit
        // before the end of the text, inside the emoji.
        assert_eq!(
            features("here. Last.\u{1F600}", 10, None, UNICODE),
            ["eos=.", "x=Last", "4", "xcap", "v=here.", "s=\\uD83D", "n="]
        );
        // Further from the end the whole character is in the suffix, as its
        // two surrogates in modified UTF-8.
        assert_eq!(
            features("It ended.\u{1F600} Next", 8, None, UNICODE),
            ["eos=.", "x=ended", "5", "v=It", "vcap", "s=\\uD83D\\uDE00", "n=Next", "ncap"]
        );
    }

    #[test]
    fn whitespace_mode_changes_words() {
        // U+001C is whitespace only in the legacy mode; U+0085 only in the
        // Unicode mode. From the fixture context files.
        let text = "Alpha.\u{1C}Beta.\u{1F} Gamma.";
        assert_eq!(
            features(text, 11, None, UNICODE),
            ["eos=.", "x=.\u{1C}Beta", "6", "v=Alpha", "vcap", "s=", "n=Gamma.", "ncap"]
        );
        assert_eq!(
            features(text, 11, None, WhitespaceMode::Legacy),
            ["sn", "eos=.", "x=Beta", "4", "xcap", "v=Alpha.", "vcap", "s=", "n=Gamma.", "ncap"]
        );
        let text = "First line.\u{85}Second line.";
        assert_eq!(
            features(text, 10, None, UNICODE),
            ["sn", "eos=.", "x=line", "4", "v=First", "vcap", "s=", "n=Second", "ncap"]
        );
        assert_eq!(
            features(text, 10, None, WhitespaceMode::Legacy),
            ["eos=.", "x=line", "4", "v=First", "vcap", "s=\u{85}Second", "n=line."]
        );
    }

    #[test]
    fn both_walks_give_the_same_features() {
        // An ASCII text takes the byte walk; the general walk on the same
        // text must give the same features at every candidate.
        let dict = dictionary();
        let eos = EndOfSentence::default();
        for mode in [UNICODE, WhitespaceMode::Legacy] {
            let ctx = SdContext::new(&eos, Some(&dict), mode, Limits::NONE);
            let text = " Mr. Smith  met U.S. people.\tA.M. z.B.  x. \x1c? end!";
            let mut positions = Vec::new();
            let scanned = eos.positions(text, &mut positions);
            assert!(scanned.is_ascii());
            let (mut fast, mut general) = (Features::new(), Features::new());
            let mut lookup = Lookup::default();
            let (mut fast_walks, mut general_walks) = (Walks::new(), Walks::new());
            for &at in &positions {
                ctx.context(scanned, at, &mut fast_walks, &mut fast, &mut lookup);
                let cursor = Utf16Cursor::at(text, at.byte as usize, at.unit as usize);
                ctx.build(cursor, scanned, &mut general_walks, &mut general, &mut lookup);
                assert!(fast.iter().eq(general.iter()), "at {at:?}");
            }
        }
    }

    /// Texts on which the remembered whitespace searches are compared with
    /// the plain walks: long stretches without whitespace, whitespace runs
    /// at either end, end characters that are whitespace, characters
    /// outside the BMP, and whitespace that only one mode has.
    fn walk_texts() -> Vec<String> {
        let mut texts = vec![
            "A.".repeat(300),
            "x.Y".repeat(200),
            format!("  {}  ", "ab.c?D!".repeat(80)),
            format!("{} tail. end", "q.".repeat(150)),
            format!("Lead. {}. x", "w".repeat(400)),
            "He thought hard.Bekannt. Sie kam.\n\nNew. Line.\nA.B.C. d. e.".repeat(6),
            "\u{1F600}.\u{10400}x. \u{1F600}.\u{85}y.\u{1C}z.\u{3000}.".repeat(30),
            " . . ..  .\t.\n".repeat(20),
            ".".repeat(50),
            "\n.\n.x\n\n.".repeat(20),
        ];
        texts.push(texts.concat());
        texts
    }

    /// Every candidate of every text in order, with remembered searches and
    /// with plain walks, in both modes and both cursor kinds, with and
    /// without the dictionary.
    #[test]
    fn remembered_searches_match_the_plain_walks() {
        let dict = dictionary();
        for eos in [EndOfSentence::default(), EndOfSentence::from_chars("\n.!?").unwrap()] {
            for mode in [UNICODE, WhitespaceMode::Legacy] {
                for dict in [None, Some(&dict)] {
                    let ctx = SdContext::new(&eos, dict, mode, Limits::NONE);
                    let mut positions = Vec::new();
                    let (mut remembered, mut plain) = (Features::new(), Features::new());
                    let mut lookup = Lookup::default();
                    for text in walk_texts() {
                        let scanned = eos.positions(&text, &mut positions);
                        let (mut walks, mut ascii_walks) = (Walks::new(), Walks::new());
                        for &at in &positions {
                            let cursor = Utf16Cursor::at(&text, at.byte as usize, at.unit as usize);
                            ctx.build(cursor.clone(), scanned, &mut Walks::plain(), &mut plain, &mut lookup);
                            ctx.build(cursor, scanned, &mut walks, &mut remembered, &mut lookup);
                            assert!(plain.iter().eq(remembered.iter()), "{mode:?} at {at:?} of {text:?}");
                            if scanned.is_ascii() {
                                let cursor = AsciiCursor::at(&text, at.byte as usize);
                                ctx.build(cursor, scanned, &mut ascii_walks, &mut remembered, &mut lookup);
                                assert!(plain.iter().eq(remembered.iter()), "{mode:?} at {at:?} of {text:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn limits_leave_out_only_words_that_cannot_match() {
        let dict = dictionary();
        let eos = EndOfSentence::default();
        let text = "Somewhat long. Mr. Smith left. Abbreviations.";
        let mut positions = Vec::new();
        let scanned = eos.positions(text, &mut positions);
        let at = positions[2];
        let mut out = Features::new();
        let mut get = |limits: Limits| {
            let ctx = SdContext::new(&eos, Some(&dict), UNICODE, limits);
            ctx.context(scanned, at, &mut Walks::new(), &mut out, &mut Lookup::default());
            out.iter().map(show).collect::<Vec<_>>()
        };
        assert_eq!(
            get(Limits::NONE),
            ["sn", "eos=.", "x=left", "4", "v=Smith", "vcap", "s=", "n=Abbreviations.", "ncap"]
        );
        // "n=Abbreviations." is 16 bytes and "v=Smith" 7: with 7 the first
        // is left out and its flags stay; with 6 the second goes too.
        assert_eq!(
            get(Limits { feature: 7, abbreviation: 3 }),
            ["sn", "eos=.", "x=left", "4", "v=Smith", "vcap", "s=", "ncap"]
        );
        assert_eq!(get(Limits { feature: 6, abbreviation: 3 }), ["sn", "eos=.", "x=left", "4", "vcap", "s=", "ncap"]);
        // The abbreviation limit counts the end character with the prefix:
        // "Mr." is three units, so a limit of 2 skips the lookup.
        let at = positions[1];
        let mut get = |abbreviation: usize| {
            let ctx = SdContext::new(&eos, Some(&dict), UNICODE, Limits { feature: usize::MAX, abbreviation });
            ctx.context(scanned, at, &mut Walks::new(), &mut out, &mut Lookup::default());
            out.iter().any(|f| f == b"xabbrev")
        };
        assert!(get(3));
        assert!(!get(2));
    }

    #[test]
    fn a_lone_period() {
        assert_eq!(features(".", 0, None, UNICODE), ["eos=.", "x=", "v=", "s=", "n="]);
        assert_eq!(features(" . ", 1, None, UNICODE), ["sp", "sn", "eos=.", "x=", "v=", "s=", "n="]);
    }
}
