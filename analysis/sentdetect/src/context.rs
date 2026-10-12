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

use crate::eos::{EndOfSentence, Position};
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
    pub fn new(
        eos: &'m EndOfSentence,
        abbreviations: Option<&'m Dictionary>,
        whitespace: WhitespaceMode,
    ) -> SdContext<'m> {
        SdContext { eos, abbreviations, whitespace }
    }

    #[inline]
    fn is_ws(&self, unit: Option<u16>) -> bool {
        unit.is_some_and(|u| self.whitespace.is_whitespace(u32::from(u)))
    }

    /// Replaces `out` with the features of the candidate at `at`. `len16`
    /// is the text's length in code units, and `at` must hold a character
    /// of the end set. A text whose length in code units is its length in
    /// bytes is all ASCII, and is walked by byte.
    pub fn context(&self, text: &str, len16: u32, at: Position, out: &mut Features, lookup: &mut Lookup) {
        if len16 as usize == text.len() {
            self.build(AsciiCursor::at(text, at.byte as usize), len16, out, lookup);
        } else {
            self.build(Utf16Cursor::at(text, at.byte as usize, at.unit as usize), len16, out, lookup);
        }
    }

    fn build<'t, C: Units<'t>>(&self, pos: C, len16: u32, out: &mut Features, lookup: &mut Lookup) {
        out.clear();
        let eos = pos.unit().expect("a candidate is a character of the text");
        let last_index = len16 as usize - 1;

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
        let mut prefix_start = self.previous_space(&pos);
        let mut c = pos.clone();
        while c.step_back() && c.unit_index() > prefix_start.unit_index() {
            if self.eos.contains_opt(c.unit()) {
                prefix_start = c;
                break;
            }
        }
        let prefix = Word::trimmed(&prefix_start, &pos);
        let previous = Word::trimmed(&self.previous_space(&prefix_start), &prefix_start);

        let (suffix, next) = if pos.unit_index() == last_index {
            (None, None)
        } else {
            // The suffix runs to the last character of the next whitespace
            // run, or to the next end character.
            let mut suffix_end = self.next_space(&pos, last_index);
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
            let next_end = self.next_space(&next_start, len16 as usize);
            (Some(suffix), Some(Word::trimmed(&next_start, &next_end)))
        };

        self.word(out, b"x=", Some(&prefix));
        if !prefix.is_empty() {
            out.push_number(prefix.len());
            if prefix.first_upper() {
                out.push(b"xcap");
            }
            if let Some(dict) = self.abbreviations
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

    /// Writes `name` and the word (nothing for an absent word).
    fn word<'t, C: Units<'t>>(&self, out: &mut Features, name: &[u8], word: Option<&Word<C>>) {
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
            && let Some(s) = w.as_str()
            && dict.contains_with(s, &mut lookup.lower)
        {
            out.push(abbrev);
        }
    }

    /// `previousSpaceIndex`: from the unit before `seek`, back to the
    /// nearest whitespace, and then to the first unit of that whitespace
    /// run; the start of the text when there is none.
    fn previous_space<'t, C: Units<'t>>(&self, seek: &C) -> C {
        let mut c = seek.clone();
        c.step_back();
        while c.unit_index() > 0 && !self.is_ws(c.unit()) {
            c.step_back();
        }
        if c.unit_index() > 0 {
            while c.unit_index() > 0 && self.is_ws(c.peek_prev_unit()) {
                c.step_back();
            }
        }
        c
    }

    /// `nextSpaceIndex`: from the unit after `seek`, forward to the nearest
    /// whitespace before index `last`, and then to the last unit of that
    /// whitespace run; index `last` when there is none.
    fn next_space<'t, C: Units<'t>>(&self, seek: &C, last: usize) -> C {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The features of the candidate at UTF-16 index `unit`, as strings
    /// (lone surrogates shown as `\uXXXX`).
    fn features(text: &str, unit: usize, dict: Option<&Dictionary>, mode: WhitespaceMode) -> Vec<String> {
        let eos = EndOfSentence::default();
        let ctx = SdContext::new(&eos, dict, mode);
        let mut positions = Vec::new();
        let len16 = eos.positions(text, &mut positions);
        let at = *positions.iter().find(|p| p.unit as usize == unit).expect("a candidate at that index");
        let mut out = Features::new();
        ctx.context(text, len16, at, &mut out, &mut Lookup::default());
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
        let ctx = SdContext::new(&eos, None, UNICODE);
        let mut positions = Vec::new();
        let len16 = eos.positions("a\nb", &mut positions);
        let mut out = Features::new();
        ctx.context("a\nb", len16, positions[0], &mut out, &mut Lookup::default());
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
            let ctx = SdContext::new(&eos, Some(&dict), mode);
            let text = " Mr. Smith  met U.S. people.\tA.M. z.B.  x. \x1c? end!";
            let mut positions = Vec::new();
            let len16 = eos.positions(text, &mut positions);
            assert_eq!(len16 as usize, text.len());
            let (mut fast, mut general) = (Features::new(), Features::new());
            let mut lookup = Lookup::default();
            for &at in &positions {
                ctx.context(text, len16, at, &mut fast, &mut lookup);
                ctx.build(Utf16Cursor::at(text, at.byte as usize, at.unit as usize), len16, &mut general, &mut lookup);
                assert!(fast.iter().eq(general.iter()), "at {at:?}");
            }
        }
    }

    #[test]
    fn a_lone_period() {
        assert_eq!(features(".", 0, None, UNICODE), ["eos=.", "x=", "v=", "s=", "n="]);
        assert_eq!(features(" . ", 1, None, UNICODE), ["sp", "sn", "eos=.", "x=", "v=", "s=", "n="]);
    }
}
