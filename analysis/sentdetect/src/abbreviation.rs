//! The abbreviation veto: `SentenceDetectorME.AbbreviationIndex`.
//!
//! When the model says a candidate ends a sentence, the split is refused
//! if an entry of the abbreviation dictionary covers the candidate and
//! starts at the start of the current segment, or right after whitespace
//! (`Character.isWhitespace`, whatever the whitespace mode), an
//! apostrophe (`'`, `` ` ``, U+00B4) or an opening round bracket. Only the
//! first token of a multi-token entry counts, as in the Java.
//!
//! The Java looks at a window of the text around the candidate, as wide
//! as the longest entry on either side, lowercased per code point with
//! `Character.toLowerCase` for a case-insensitive dictionary, and probes a
//! hash set with every substring of an entry's length that starts at or
//! before the candidate and reaches it. This port does the same over
//! UTF-16 code units, so substrings that cut a character outside the BMP
//! are probed as the Java probes them (and never match, as no entry holds
//! a lone surrogate).

use std::collections::{BTreeSet, HashSet};

use turbo_maxent::Dictionary;
use turbo_unicode::{Utf16Cursor, java};

use crate::eos::Position;

/// The dictionary's first tokens as UTF-16, bucketed by length.
#[derive(Clone, Debug)]
pub struct AbbreviationIndex {
    entries: HashSet<Box<[u16]>>,
    /// The distinct entry lengths in code units, ascending.
    lengths: Vec<u32>,
    max_len: u32,
    case_sensitive: bool,
}

impl AbbreviationIndex {
    /// Indexes the first token of every entry; lowercased unless the
    /// dictionary is case-sensitive.
    pub fn new(dictionary: &Dictionary) -> AbbreviationIndex {
        let case_sensitive = dictionary.is_case_sensitive();
        let mut entries = HashSet::new();
        let mut lengths = BTreeSet::new();
        let mut units = Vec::new();
        for token in dictionary.as_set() {
            units.clear();
            fold_into(&mut units, token, case_sensitive);
            lengths.insert(units.len() as u32);
            entries.insert(units.as_slice().into());
        }
        let lengths: Vec<u32> = lengths.into_iter().collect();
        let max_len = lengths.last().copied().unwrap_or(0);
        AbbreviationIndex { entries, lengths, max_len, case_sensitive }
    }

    /// The number of distinct first tokens.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `allowsBreak`: whether a split at `candidate` is acceptable in the
    /// segment that starts at `from`. `len16` is the text's length in code
    /// units; `window` is a buffer reused across calls.
    pub fn allows_break(
        &self,
        text: &str,
        len16: u32,
        from: Position,
        candidate: Position,
        window: &mut Vec<u16>,
    ) -> bool {
        if self.lengths.is_empty() || candidate.unit < from.unit || candidate.unit > len16 {
            return true;
        }
        let cand = Utf16Cursor::at(text, candidate.byte as usize, candidate.unit as usize);
        let scan_start = from.unit;

        // The window holds every position an entry can start at and the
        // longest entry after the candidate, widened to whole characters.
        let low = scan_start.max(candidate.unit.saturating_sub(self.max_len)) as usize;
        let mut start = cand.clone();
        while start.unit_index() > low && start.step_back() {}
        if !start.char_start() {
            start.step_back();
        }
        let high = len16.min(candidate.unit.saturating_add(self.max_len)) as usize;
        let mut end = cand.clone();
        while end.unit_index() < high && end.step_forward() {}
        if !end.char_start() {
            end.step_forward();
        }
        window.clear();
        fold_into(window, &text[start.byte_offset()..end.byte_offset()], self.case_sensitive);
        let window_start = start.unit_index() as u32;
        debug_assert_eq!(window.len(), end.unit_index() - start.unit_index(), "lowercasing keeps the length");

        for &len in &self.lengths {
            let first = scan_start.max(candidate.unit.saturating_sub(len));
            for pos in first..=candidate.unit {
                let end_pos = pos + len;
                if end_pos > len16 {
                    break;
                }
                let key = &window[(pos - window_start) as usize..(end_pos - window_start) as usize];
                if !self.entries.contains(key) {
                    continue;
                }
                if pos == from.unit && end_pos == candidate.unit + 1 {
                    return false;
                }
                let prev = unit_at(&cand, if pos == from.unit { pos } else { pos - 1 });
                let prev = u32::from(prev);
                if java::is_whitespace(prev) || matches!(prev, 0x27 | 0x60 | 0xB4 | 0x28) {
                    return false;
                }
            }
        }
        true
    }
}

/// The code unit at index `unit`, at or before the cursor's position.
fn unit_at(cursor: &Utf16Cursor<'_>, unit: u32) -> u16 {
    let mut c = cursor.clone();
    while c.unit_index() > unit as usize {
        c.step_back();
    }
    c.unit().expect("an index inside the text")
}

/// Appends the UTF-16 code units of `s`, each code point lowercased with
/// `Character.toLowerCase` unless `case_sensitive`.
fn fold_into(out: &mut Vec<u16>, s: &str, case_sensitive: bool) {
    for c in s.chars() {
        let cp = if case_sensitive { c as u32 } else { java::to_lower_case(c as u32) };
        if cp < 0x1_0000 {
            out.push(cp as u16);
        } else {
            out.push((0xD800 + ((cp - 0x1_0000) >> 10)) as u16);
            out.push((0xDC00 + ((cp - 0x1_0000) & 0x3FF)) as u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::EndOfSentence;

    fn index(case_sensitive: bool, entries: &[&[&str]]) -> AbbreviationIndex {
        let mut xml = format!(r#"<dictionary case_sensitive="{case_sensitive}">"#);
        for e in entries {
            xml.push_str("<entry>");
            for t in *e {
                xml.push_str(&format!("<token>{t}</token>"));
            }
            xml.push_str("</entry>");
        }
        xml.push_str("</dictionary>");
        AbbreviationIndex::new(&Dictionary::read(xml.as_bytes()).unwrap())
    }

    /// Whether a split is allowed at the end character at UTF-16 index
    /// `cand`, in the segment starting at UTF-16 index `from`.
    fn allows(idx: &AbbreviationIndex, text: &str, from: usize, cand: usize) -> bool {
        let mut positions = Vec::new();
        let len16 = EndOfSentence::from_chars(".!?)").unwrap().positions(text, &mut positions);
        let candidate = *positions.iter().find(|p| p.unit as usize == cand).expect("a candidate");
        let from =
            Position { byte: turbo_unicode::utf16::utf16_to_byte(text, from).unwrap() as u32, unit: from as u32 };
        idx.allows_break(text, len16, from, candidate, &mut Vec::new())
    }

    #[test]
    fn entries_and_lengths() {
        let idx = index(false, &[&["Mr."], &["U.S."], &["et", "al."], &["DR."]]);
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.lengths, [2, 3, 4]);
        assert!(idx.entries.contains(&[u16::from(b'd'), u16::from(b'r'), u16::from(b'.')][..]));
        // Only the first token of a multi-token entry is indexed.
        assert!(idx.entries.contains(&[u16::from(b'e'), u16::from(b't')][..]));
        // An empty dictionary allows every break.
        let empty = index(false, &[]);
        assert!(empty.is_empty());
        assert!(empty.allows_break("a. b", 4, Position::default(), Position { byte: 1, unit: 1 }, &mut Vec::new()));
    }

    #[test]
    fn vetoes_after_whitespace_apostrophe_and_bracket() {
        let idx = index(false, &[&["Mr."], &["Dr."]]);
        // From the fixture context file: "(Mr. Smith) came." and "'Dr. Jones'".
        assert!(!allows(&idx, "(Mr. Smith) came.", 0, 3));
        assert!(!allows(&idx, "x. 'Dr. Jones' left.", 2, 6));
        assert!(!allows(&idx, "Then mr. Smith", 0, 7));
        // At the start of the segment: the entry exactly fills it.
        assert!(!allows(&idx, "Dr. Who", 0, 2));
        // A letter right before the entry: "Hamr." is not "mr.".
        assert!(allows(&idx, "Hamr. Then", 0, 4));
        // Case matters in a case-sensitive dictionary.
        let cs = index(true, &[&["Mr."]]);
        assert!(allows(&cs, "Then MR. Smith", 0, 7));
        assert!(!allows(&cs, "Then Mr. Smith", 0, 7));
    }

    #[test]
    fn the_segment_start_bounds_the_search() {
        let idx = index(false, &[&["U.S."]]);
        // The whole entry from the segment start.
        assert!(!allows(&idx, "U.S. report", 0, 3));
        // The segment starts inside the entry, so it no longer counts.
        assert!(allows(&idx, "U.S. report", 2, 3));
        // An entry starting at the segment start but not ending at the
        // candidate is judged by its own first character, which is "U".
        assert!(allows(&idx, "U.S.) x", 0, 4));
    }

    #[test]
    fn the_window_never_cuts_a_character() {
        // A character outside the BMP right before the entry: the window
        // start is moved back to the start of the pair.
        let idx = index(false, &[&["etc."], &["ab."]]);
        assert!(allows(&idx, "\u{1F600}etc. x", 0, 5));
        assert!(!allows(&idx, "\u{1F600} etc. x", 0, 6));
        // Lowercasing works per code point, outside the BMP too: U+10400
        // lowercases to U+10428.
        let idx = index(false, &[&["\u{10428}."]]);
        assert!(!allows(&idx, "x \u{10400}. y", 0, 4));
    }
}
