//! The features of one position in a unit: OpenNLP's
//! `DefaultTokenContextGenerator`.
//!
//! For position `index` of a unit (a UTF-16 index, from 1 to one before
//! the unit's length), the generator lists, in this order:
//!
//! - `p=` the prefix before the position and `s=` the suffix from it;
//! - the character before (`p1`) and the one before that (`p2`, or
//!   `p2=bok` at index 1), with their classes, then `p21=` the two of them
//!   and `p1f1=` the characters on both sides of the position;
//! - the character at the position (`f1`) and the one after (`f2`, or
//!   `f2=bok` on the last one), with their classes, then `f12=` the two;
//! - `cc` when the unit starts with `&` and ends with `;`, an entity;
//! - `pabb` at the last position of a unit the abbreviation dictionary
//!   lists.
//!
//! A character's classes are `_alpha` (a letter), `_caps` (an upper case
//! letter), `_num` (a decimal digit), `_ws` (whitespace in the configured
//! mode), `_eos` (`.` `?` `!`), `_quote` (`` ` `` `"` `'`), `_lp` (`[` `{`
//! `(`) and `_rp` (`]` `}` `)`). "Character" means a Java `char`: a UTF-16
//! code unit, so a character outside the Basic Multilingual Plane is two,
//! each a surrogate in no class, and a prefix can end on a high
//! surrogate. The features are written as Java modified UTF-8, the
//! encoding of the model's predicate names, so such a feature matches
//! what the JVM looked up.

use turbo_unicode::java;
use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};
use turbo_unicode::{Utf16Cursor, WhitespaceMode};

/// The features of one context, by column: their bytes back to back and
/// the end of each. Cleared and refilled for every position, so a unit's
/// loop allocates nothing once the buffers have grown.
#[derive(Clone, Debug, Default)]
pub struct Features {
    bytes: Vec<u8>,
    ends: Vec<u32>,
}

impl Features {
    pub fn new() -> Features {
        Features::default()
    }

    /// Removes every feature, keeping the buffers.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.ends.clear();
    }

    /// The number of features.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Feature `i` as modified UTF-8.
    pub fn get(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }

    /// The features in order, for `MaxentModel::eval_features`.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> + '_ {
        let mut start = 0;
        self.ends.iter().map(move |&end| {
            let feature = &self.bytes[start..end as usize];
            start = end as usize;
            feature
        })
    }

    /// Ends the feature written since the previous one.
    #[inline]
    fn end(&mut self) {
        self.ends.push(self.bytes.len() as u32);
    }

    /// A feature that is a fixed string.
    #[inline]
    fn fixed(&mut self, feature: &[u8]) {
        self.bytes.extend_from_slice(feature);
        self.end();
    }

    /// `key=c` and the classes of code unit `c`.
    fn char_features(&mut self, key: &[u8], c: u16, mode: WhitespaceMode) {
        self.bytes.extend_from_slice(key);
        self.bytes.push(b'=');
        push_modified_utf8(&mut self.bytes, u32::from(c));
        self.end();
        let cp = u32::from(c);
        let class: &[u8] = if java::is_letter(cp) {
            self.with_suffix(key, b"_alpha");
            if !java::is_upper_case(cp) {
                return;
            }
            b"_caps"
        } else if java::is_digit(cp) {
            b"_num"
        } else if mode.is_whitespace(cp) {
            b"_ws"
        } else {
            match c {
                0x2E | 0x3F | 0x21 => b"_eos",
                0x60 | 0x22 | 0x27 => b"_quote",
                0x5B | 0x7B | 0x28 => b"_lp",
                0x5D | 0x7D | 0x29 => b"_rp",
                _ => return,
            }
        };
        self.with_suffix(key, class);
    }

    #[inline]
    fn with_suffix(&mut self, key: &[u8], suffix: &[u8]) {
        self.bytes.extend_from_slice(key);
        self.bytes.extend_from_slice(suffix);
        self.end();
    }

    /// `key=` followed by two code units.
    #[inline]
    fn pair(&mut self, key: &[u8], a: u16, b: u16) {
        self.bytes.extend_from_slice(key);
        push_modified_utf8(&mut self.bytes, u32::from(a));
        push_modified_utf8(&mut self.bytes, u32::from(b));
        self.end();
    }
}

/// What the generator knows about a unit as a whole, worked out once per
/// unit rather than at every position.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UnitFacts {
    /// The unit starts with `&` and ends with `;`.
    pub entity: bool,
    /// The abbreviation dictionary lists the unit.
    pub abbreviation: bool,
}

impl UnitFacts {
    /// The facts of `unit`, with `abbreviation` from a dictionary lookup.
    pub fn of(unit: &str, abbreviation: bool) -> UnitFacts {
        let b = unit.as_bytes();
        UnitFacts { entity: b.first() == Some(&b'&') && b.last() == Some(&b';'), abbreviation }
    }
}

/// Writes the features of the position `at` of its unit into `out`,
/// replacing what was there. `at` walks the unit's text; its position is
/// the index `getContext` is given, which must be at least 1 and before
/// the unit's last code unit's end (the tokenizer never asks for index 0,
/// where the Java would add `p1=bok`, nor for the end).
pub fn token_context(at: &Utf16Cursor<'_>, facts: UnitFacts, mode: WhitespaceMode, out: &mut Features) {
    let unit = at.text();
    let f1 = at.unit().expect("a position before the end of the unit");
    let p1 = at.peek_prev_unit().expect("a position after the start of the unit");
    out.clear();

    // The prefix and the suffix. On the low half of a pair the prefix ends
    // with the high half and the suffix starts with the low one.
    let byte = at.byte_offset();
    out.bytes.extend_from_slice(b"p=");
    push_str_modified_utf8(&mut out.bytes, &unit[..byte]);
    let suffix_from = if at.char_start() {
        byte
    } else {
        push_modified_utf8(&mut out.bytes, u32::from(p1));
        byte + 4
    };
    out.end();
    out.bytes.extend_from_slice(b"s=");
    if !at.char_start() {
        push_modified_utf8(&mut out.bytes, u32::from(f1));
    }
    push_str_modified_utf8(&mut out.bytes, &unit[suffix_from..]);
    out.end();

    out.char_features(b"p1", p1, mode);
    let mut back = at.clone();
    back.step_back();
    match back.peek_prev_unit() {
        Some(p2) => {
            out.char_features(b"p2", p2, mode);
            out.pair(b"p21=", p2, p1);
        }
        None => out.fixed(b"p2=bok"),
    }
    out.pair(b"p1f1=", p1, f1);

    out.char_features(b"f1", f1, mode);
    let next = at.peek_next_unit();
    match next {
        Some(f2) => {
            out.char_features(b"f2", f2, mode);
            out.pair(b"f12=", f1, f2);
        }
        None => out.fixed(b"f2=bok"),
    }
    if facts.entity {
        out.fixed(b"cc");
    }
    if next.is_none() && facts.abbreviation {
        out.fixed(b"pabb");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The features at UTF-16 index `index` of `unit`, as strings with
    /// each lone surrogate written as `<XXXX>`.
    fn context(unit: &str, index: usize, abbreviation: bool) -> Vec<String> {
        let mut at = Utf16Cursor::new(unit);
        for _ in 0..index {
            at.step_forward();
        }
        let mut f = Features::new();
        token_context(&at, UnitFacts::of(unit, abbreviation), WhitespaceMode::Unicode, &mut f);
        f.iter().map(show).collect()
    }

    /// Decodes modified UTF-8 three-byte surrogates as `<XXXX>`.
    fn show(bytes: &[u8]) -> String {
        let mut s = String::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0xED && i + 2 < bytes.len() && bytes[i + 1] >= 0xA0 {
                let v = (u32::from(bytes[i] & 0x0F) << 12)
                    | (u32::from(bytes[i + 1] & 0x3F) << 6)
                    | u32::from(bytes[i + 2] & 0x3F);
                s.push_str(&format!("<{v:04X}>"));
                i += 3;
            } else {
                let len = match bytes[i] {
                    0..0x80 => 1,
                    0xC0..0xE0 => 2,
                    0xE0..0xF0 => 3,
                    _ => 4,
                };
                s.push_str(std::str::from_utf8(&bytes[i..i + len]).unwrap());
                i += len;
            }
        }
        s
    }

    #[test]
    fn first_position() {
        // The first context of the golden contexts file for English.
        assert_eq!(
            context("Qu'est-ce", 1, false),
            [
                "p=Q",
                "s=u'est-ce",
                "p1=Q",
                "p1_alpha",
                "p1_caps",
                "p2=bok",
                "p1f1=Qu",
                "f1=u",
                "f1_alpha",
                "f2='",
                "f2_quote",
                "f12=u'"
            ]
        );
    }

    #[test]
    fn last_position_and_abbreviation() {
        assert_eq!(
            context("Dr.", 2, true),
            [
                "p=Dr", "s=.", "p1=r", "p1_alpha", "p2=D", "p2_alpha", "p2_caps", "p21=Dr", "p1f1=r.", "f1=.",
                "f1_eos", "f2=bok", "pabb"
            ]
        );
        // Not at the last position, no pabb.
        assert!(!context("e.g.", 2, true).contains(&"pabb".to_string()));
        assert!(!context("Dr.", 2, false).contains(&"pabb".to_string()));
    }

    #[test]
    fn classes() {
        let f = context("(9)", 1, false);
        assert_eq!(
            f,
            ["p=(", "s=9)", "p1=(", "p1_lp", "p2=bok", "p1f1=(9", "f1=9", "f1_num", "f2=)", "f2_rp", "f12=9)"]
        );
        let f = context("`!\"", 1, false);
        assert_eq!(f[2..4], ["p1=`", "p1_quote"]);
        assert_eq!(f[6..8], ["f1=!", "f1_eos"]);
        assert_eq!(f[8..10], ["f2=\"", "f2_quote"]);
        // A character in no class has its own feature only.
        let f = context("a-b", 1, false);
        assert_eq!(f[6..8], ["f1=-", "f2=b"]);
        // Greek capitals are letters and upper case; a Devanagari digit is a digit.
        let f = context("\u{3a9}\u{967}", 1, false);
        assert_eq!(f[2..5], ["p1=\u{3a9}", "p1_alpha", "p1_caps"]);
        assert_eq!(f[7..9], ["f1=\u{967}", "f1_num"]);
    }

    #[test]
    fn entity() {
        let f = context("&amp;", 2, false);
        assert_eq!(f.last().unwrap(), "cc");
        assert!(!context("&amp", 2, false).contains(&"cc".to_string()));
    }

    #[test]
    fn surrogate_halves_are_code_units() {
        // "ab" U+1F600 "cd": at index 3 the position is between the halves.
        let unit = "ab\u{1f600}cd";
        assert_eq!(
            context(unit, 3, false),
            [
                "p=ab<D83D>",
                "s=<DE00>cd",
                "p1=<D83D>",
                "p2=b",
                "p2_alpha",
                "p21=b<D83D>",
                "p1f1=<D83D><DE00>",
                "f1=<DE00>",
                "f2=c",
                "f2_alpha",
                "f12=<DE00>c"
            ]
        );
        // At index 4, after the pair, the prefix holds both halves.
        let f = context(unit, 4, false);
        assert_eq!(f[0], "p=ab<D83D><DE00>");
        assert_eq!(f[1], "s=cd");
        assert_eq!(f[2], "p1=<DE00>");
        assert_eq!(f[3], "p2=<D83D>");
        assert_eq!(f[4], "p21=<D83D><DE00>");
        // At index 2 the position is on the high half, and the next unit
        // is the low half; at index 1 the next unit is the high half.
        assert_eq!(context(unit, 2, false).last().unwrap(), "f12=<D83D><DE00>");
        assert_eq!(context(unit, 1, false).last().unwrap(), "f12=b<D83D>");
    }

    #[test]
    fn nul_is_two_bytes() {
        let unit = "a\0b";
        let mut at = Utf16Cursor::new(unit);
        at.step_forward();
        let mut f = Features::new();
        token_context(&at, UnitFacts::of(unit, false), WhitespaceMode::Unicode, &mut f);
        assert_eq!(f.get(1), b"s=\xC0\x80b");
        assert_eq!(f.get(5), b"p1f1=a\xC0\x80");
        assert_eq!(f.get(6), b"f1=\xC0\x80");
    }

    #[test]
    fn features_iterate_as_they_index() {
        let unit = "abc";
        let mut at = Utf16Cursor::new(unit);
        at.step_forward();
        let mut f = Features::new();
        token_context(&at, UnitFacts::of(unit, false), WhitespaceMode::Unicode, &mut f);
        let n = f.len();
        assert!(!f.is_empty());
        let by_index: Vec<&[u8]> = (0..n).map(|i| f.get(i)).collect();
        assert_eq!(f.iter().collect::<Vec<_>>(), by_index);
        f.clear();
        assert!(f.is_empty());
    }
}
