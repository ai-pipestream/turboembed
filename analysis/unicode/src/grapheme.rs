//! Extended grapheme clusters, UAX #29 rules GB1 to GB999 of Unicode 18.0,
//! over the `Grapheme_Cluster_Break` property (`GraphemeBreakProperty.txt`),
//! the `Indic_Conjunct_Break` property (`DerivedCoreProperties.txt`) and
//! `Extended_Pictographic` (`emoji-data.txt`).
//!
//! In Unicode 18.0, rule GB9c reads `ConjunctLinker ConjunctExtender* ×
//! LinkingConsonant`: a consonant joins the cluster when a linker (a
//! virama, which Unicode 18.0 assigns in Myanmar and Khmer too) comes
//! before it with nothing but `Indic_Conjunct_Break=Extend` marks after the
//! linker. The boundaries are checked against Unicode's
//! `GraphemeBreakTest.txt`, every case.

use crate::tables::grapheme_break::{BLOCKS, INCB_NAMES, INCB_SHIFT, INDEX, NAMES};
use crate::tables::props::{self, EXTENDED_PICTOGRAPHIC};
use crate::trie::lookup_code_point;

/// A UAX #29 `Grapheme_Cluster_Break` property value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GraphemeBreak {
    /// No `Grapheme_Cluster_Break` value: every code point the property file does not list.
    #[default]
    Other = 0,
    /// Carriage return (U+000D).
    Cr = 1,
    /// Line feed (U+000A).
    Lf = 2,
    /// Control and format characters that always stand alone.
    Control = 3,
    /// Combining marks and other characters that extend the cluster before them.
    Extend = 4,
    /// Zero width joiner (U+200D).
    Zwj = 5,
    /// Regional indicator symbols, used in pairs for flags.
    RegionalIndicator = 6,
    /// Characters that join the cluster after them, such as some Arabic number signs.
    Prepend = 7,
    /// Spacing marks that join the cluster before them.
    SpacingMark = 8,
    /// Hangul leading consonants (choseong).
    L = 9,
    /// Hangul vowels (jungseong).
    V = 10,
    /// Hangul trailing consonants (jongseong).
    T = 11,
    /// Hangul LV syllables.
    Lv = 12,
    /// Hangul LVT syllables.
    Lvt = 13,
}

impl GraphemeBreak {
    /// Every value, in discriminant order, which is also the order of the
    /// numbers the generated table stores.
    pub const ALL: [GraphemeBreak; 14] = [
        GraphemeBreak::Other,
        GraphemeBreak::Cr,
        GraphemeBreak::Lf,
        GraphemeBreak::Control,
        GraphemeBreak::Extend,
        GraphemeBreak::Zwj,
        GraphemeBreak::RegionalIndicator,
        GraphemeBreak::Prepend,
        GraphemeBreak::SpacingMark,
        GraphemeBreak::L,
        GraphemeBreak::V,
        GraphemeBreak::T,
        GraphemeBreak::Lv,
        GraphemeBreak::Lvt,
    ];

    /// The value name as `GraphemeBreakProperty.txt` writes it, such as `SpacingMark`.
    pub const fn property_name(self) -> &'static str {
        NAMES[self as usize]
    }
}

/// An `Indic_Conjunct_Break` property value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IndicConjunctBreak {
    /// Every code point `DerivedCoreProperties.txt` does not list.
    #[default]
    None = 0,
    /// Viramas that link two consonants into one cluster.
    Linker = 1,
    /// Consonants that a linker joins to the cluster before them.
    Consonant = 2,
    /// Marks that may stand between a linker and the consonant it links.
    Extend = 3,
}

impl IndicConjunctBreak {
    /// Every value, in discriminant order.
    pub const ALL: [IndicConjunctBreak; 4] = [
        IndicConjunctBreak::None,
        IndicConjunctBreak::Linker,
        IndicConjunctBreak::Consonant,
        IndicConjunctBreak::Extend,
    ];

    /// The value name as `DerivedCoreProperties.txt` writes it, such as `Linker`.
    pub const fn property_name(self) -> &'static str {
        INCB_NAMES[self as usize]
    }
}

#[inline]
fn stored(cp: u32) -> u8 {
    lookup_code_point(&INDEX, &BLOCKS, cp)
}

/// The `Grapheme_Cluster_Break` value of a code point; [`GraphemeBreak::Other`]
/// for every code point the property file does not list and for values
/// above U+10FFFF.
#[inline]
pub fn grapheme_break(cp: u32) -> GraphemeBreak {
    GraphemeBreak::ALL[(stored(cp) & 0x0F) as usize]
}

/// The `Indic_Conjunct_Break` value of a code point.
#[inline]
pub fn indic_conjunct_break(cp: u32) -> IndicConjunctBreak {
    IndicConjunctBreak::ALL[(stored(cp) >> INCB_SHIFT) as usize]
}

/// What the rules that look further back than one character need to know
/// about the text before a position.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
enum Pictographic {
    /// The text before does not end in `ExtPict Extend*` or `ExtPict Extend* ZWJ`.
    #[default]
    No,
    /// It ends in `ExtPict Extend*`.
    Sequence,
    /// It ends in `ExtPict Extend* ZWJ`: GB11 keeps a following `ExtPict`.
    Joined,
}

/// The extended grapheme cluster boundaries of a text, as UTF-8 byte
/// offsets in increasing order: 0, every boundary inside the text, and
/// the length of the text. An empty text has the one boundary 0.
///
/// The iterator is `Copy`, so a caller can keep a position and come back
/// to it.
#[derive(Copy, Clone, Debug)]
pub struct GraphemeBoundaries<'a> {
    text: &'a str,
    /// The byte offset of the next character to read.
    at: usize,
    /// Whether the boundary at 0 has been returned.
    started: bool,
    /// Whether the boundary at the end has been returned.
    done: bool,
    /// The class of the character before `at`.
    previous: GraphemeBreak,
    /// Whether the text before `at` ends in `ConjunctLinker ConjunctExtender*` (GB9c).
    linked: bool,
    pictographic: Pictographic,
    /// Whether the text before `at` ends in an odd number of regional indicators (GB12, GB13).
    odd_regional: bool,
}

/// The extended grapheme cluster boundaries of `text`.
pub fn grapheme_boundaries(text: &str) -> GraphemeBoundaries<'_> {
    GraphemeBoundaries {
        text,
        at: 0,
        started: false,
        done: false,
        previous: GraphemeBreak::Other,
        linked: false,
        pictographic: Pictographic::No,
        odd_regional: false,
    }
}

impl GraphemeBoundaries<'_> {
    /// Reads the character at `self.at`, says whether a boundary falls
    /// before it, and moves past it.
    #[inline]
    fn step(&mut self) -> bool {
        let rest = &self.text[self.at..];
        let c = rest.chars().next().expect("step is called inside the text");
        let cp = c as u32;
        let value = stored(cp);
        let class = GraphemeBreak::ALL[(value & 0x0F) as usize];
        let incb = IndicConjunctBreak::ALL[(value >> INCB_SHIFT) as usize];
        let pictographic = lookup_code_point(&props::INDEX, &props::BLOCKS, cp) & EXTENDED_PICTOGRAPHIC != 0;
        let boundary = self.at > 0 && self.rules(class, incb, pictographic);
        self.at += c.len_utf8();
        self.linked = match incb {
            IndicConjunctBreak::Linker => true,
            IndicConjunctBreak::Extend => self.linked,
            _ => false,
        };
        self.pictographic = if pictographic {
            Pictographic::Sequence
        } else {
            match (self.pictographic, class) {
                (Pictographic::Sequence, GraphemeBreak::Extend) => Pictographic::Sequence,
                (Pictographic::Sequence, GraphemeBreak::Zwj) => Pictographic::Joined,
                _ => Pictographic::No,
            }
        };
        self.odd_regional = class == GraphemeBreak::RegionalIndicator && !self.odd_regional;
        self.previous = class;
        boundary
    }

    /// Rules GB3 to GB999: whether a boundary falls before a character of
    /// class `current`, given the state of the text before it.
    #[inline]
    fn rules(&self, current: GraphemeBreak, incb: IndicConjunctBreak, pictographic: bool) -> bool {
        use GraphemeBreak::*;
        let previous = self.previous;
        if previous == Cr && current == Lf {
            return false; // GB3
        }
        if matches!(previous, Control | Cr | Lf) || matches!(current, Control | Cr | Lf) {
            return true; // GB4, GB5
        }
        match (previous, current) {
            (L, L | V | Lv | Lvt) | (Lv | V, V | T) | (Lvt | T, T) => return false, // GB6, GB7, GB8
            _ => {}
        }
        if matches!(current, Extend | Zwj | SpacingMark) || previous == Prepend {
            return false; // GB9, GB9a, GB9b
        }
        if self.linked && incb == IndicConjunctBreak::Consonant {
            return false; // GB9c
        }
        if self.pictographic == Pictographic::Joined && pictographic {
            return false; // GB11
        }
        if previous == RegionalIndicator && current == RegionalIndicator && self.odd_regional {
            return false; // GB12, GB13
        }
        true // GB999
    }
}

impl Iterator for GraphemeBoundaries<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            self.done = self.text.is_empty();
            return Some(0);
        }
        while self.at < self.text.len() {
            let before = self.at;
            if self.step() {
                return Some(before);
            }
        }
        self.done = true;
        Some(self.text.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundaries(text: &str) -> Vec<usize> {
        grapheme_boundaries(text).collect()
    }

    #[test]
    fn empty_and_single() {
        assert_eq!(boundaries(""), [0]);
        assert_eq!(boundaries("a"), [0, 1]);
        assert_eq!(boundaries("ab"), [0, 1, 2]);
        assert_eq!(boundaries("\r\n"), [0, 2]);
        assert_eq!(boundaries("a\u{0301}b"), [0, 3, 4]);
    }

    #[test]
    fn values() {
        assert_eq!(grapheme_break(0x000D), GraphemeBreak::Cr);
        assert_eq!(grapheme_break(0x0E33), GraphemeBreak::SpacingMark);
        assert_eq!(grapheme_break(0x0E31), GraphemeBreak::Extend);
        assert_eq!(grapheme_break(0x110000), GraphemeBreak::Other);
        assert_eq!(indic_conjunct_break(0x094D), IndicConjunctBreak::Linker);
        assert_eq!(indic_conjunct_break(0x1039), IndicConjunctBreak::Linker);
        assert_eq!(indic_conjunct_break(0x17D2), IndicConjunctBreak::Linker);
        assert_eq!(indic_conjunct_break(0x0915), IndicConjunctBreak::Consonant);
        assert_eq!(GraphemeBreak::Lvt.property_name(), "LVT");
        assert_eq!(IndicConjunctBreak::Extend.property_name(), "Extend");
    }

    #[test]
    fn clones_resume_where_they_were() {
        let text = "ab\u{0E01}\u{0E31}c";
        let mut it = grapheme_boundaries(text);
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(1));
        let saved = it;
        assert_eq!(it.collect::<Vec<_>>(), [2, 8, 9]);
        assert_eq!(saved.collect::<Vec<_>>(), [2, 8, 9]);
    }
}
