//! The byte offset scanner against a plain transcription of the Java over
//! a `Vec<char>`, on generated texts. The transcription indexes
//! characters, reads a space outside the text as the Java's `charAt`
//! does, and builds the value from a normalized string as the Java builds
//! its `BigDecimal`; it shares only the character classes with the
//! scanner, so a disagreement points at an offset or a step the scanner
//! took differently.

mod common;

use common::Rng;
use turbo_analysis::{Layer, ViewId};
use turbo_numeric::{Multiplier, NumberNotation, NumberScanner, Numbers, ScaleMarker};
use turbo_unicode::GeneralCategory::*;
use turbo_unicode::category::general_category;
use turbo_unicode::{WhitespaceMode, java};

/// One mention: character offsets, the normalized value and the marker.
#[derive(Debug, PartialEq)]
struct Mention {
    start: usize,
    end: usize,
    numeral_end: usize,
    negative: bool,
    coefficient: String,
    scale: u32,
    multiplier: Multiplier,
    marker: ScaleMarker,
}

struct Reference {
    c: Vec<char>,
    group: char,
    decimal: char,
    apply_scale: bool,
}

fn letter(c: Option<char>) -> bool {
    c.is_some_and(java::is_letter)
}

fn mark(c: char) -> bool {
    matches!(general_category(c), Mn | Mc | Me)
}

fn continuation(c: Option<char>) -> bool {
    c.is_some_and(|c| matches!(general_category(c), Lu | Ll | Lt | Lm | Lo | Nd | Nl | No | Mn | Mc | Me | Pc))
}

fn unsupported(c: char) -> bool {
    ['\u{A0}', '\u{2009}', '\u{202F}', '\'', '\u{2019}', '\u{66B}', '\u{66C}'].contains(&c)
}

impl Reference {
    fn at(&self, i: isize) -> char {
        self.cp(i).unwrap_or(' ')
    }

    fn cp(&self, i: isize) -> Option<char> {
        usize::try_from(i).ok().and_then(|i| self.c.get(i).copied())
    }

    fn digit(&self, i: isize) -> bool {
        self.at(i).is_ascii_digit()
    }

    fn base_before(&self, index: isize) -> Option<char> {
        let mut p = index;
        while p > 0 {
            let c = self.c[p as usize - 1];
            if !mark(c) {
                return Some(c);
            }
            p -= 1;
        }
        None
    }

    fn boundary_after(&self, i: isize) -> bool {
        !continuation(self.cp(i))
    }

    fn continues(&self, index: isize) -> bool {
        let before = self.at(index - 1);
        if (before == self.group || before == self.decimal || unsupported(before)) && self.digit(index - 2) {
            return true;
        }
        let e = index - if before == '+' || before == '-' { 2 } else { 1 };
        if self.at(e) != 'e' && self.at(e) != 'E' {
            return false;
        }
        let m = self.at(e - 1);
        m.is_ascii_digit() || ((m == self.group || m == self.decimal) && self.digit(e - 2))
    }

    fn exponent(&self, s: isize) -> bool {
        if self.at(s) != 'e' && self.at(s) != 'E' {
            return false;
        }
        let sign = self.at(s + 1) == '+' || self.at(s + 1) == '-';
        self.digit(s + if sign { 2 } else { 1 })
    }

    fn word_end(&self, start: isize, max: usize) -> Option<isize> {
        let units = |i: isize| (start..i).map(|k| self.at(k).len_utf16()).sum::<usize>();
        let mut i = start;
        while letter(self.cp(i)) && units(i) < max {
            i += 1;
        }
        (i != start && units(i) <= max && self.boundary_after(i)).then_some(i)
    }

    /// `NumberScan.parse`: the end, the numeral end, the normalized value
    /// string, and the scale.
    fn parse(&self, start: isize) -> Option<(isize, isize, String, Multiplier, ScaleMarker)> {
        let mut i = start;
        let mut normalized = String::new();
        while self.digit(i) {
            normalized.push(self.at(i));
            i += 1;
        }
        if i == start {
            return None;
        }
        let three = |s: isize| self.digit(s) && self.digit(s + 1) && self.digit(s + 2) && !self.digit(s + 3);
        if self.at(i) == self.group && i - start <= 3 {
            while self.at(i) == self.group && three(i + 1) {
                (1..4).for_each(|k| normalized.push(self.at(i + k)));
                i += 4;
            }
        }
        if self.at(i) == self.decimal && self.digit(i + 1) {
            normalized.push('.');
            i += 1;
            while self.digit(i) {
                normalized.push(self.at(i));
                i += 1;
            }
        }
        let next = self.at(i);
        if (next == self.group || next == self.decimal || unsupported(next)) && self.digit(i + 1) {
            return None;
        }
        if self.exponent(i) || ((next == self.group || next == self.decimal) && self.exponent(i + 1)) {
            return None;
        }
        if !self.apply_scale {
            return Some((i, i, normalized, Multiplier::One, ScaleMarker::None));
        }
        if letter(self.cp(i)) {
            let s = self.at(i);
            let bn = (s == 'b' || s == 'B') && (self.at(i + 1) == 'n' || self.at(i + 1) == 'N');
            let end = i + if bn { 2 } else { 1 };
            let m = match s {
                'k' | 'K' => Multiplier::Thousand,
                'm' | 'M' => Multiplier::Million,
                'b' | 'B' => Multiplier::Billion,
                _ => return None,
            };
            return self.boundary_after(end).then_some((end, i, normalized, m, ScaleMarker::Suffix));
        }
        if WhitespaceMode::Unicode.is_whitespace(next as u32)
            && let Some(end) = self.word_end(i + 1, 8)
        {
            let word: String = (i + 1..end).map(|k| java::to_lower_char(self.at(k))).collect();
            let m = match word.as_str() {
                "thousand" => Some(Multiplier::Thousand),
                "million" => Some(Multiplier::Million),
                "billion" => Some(Multiplier::Billion),
                "trillion" => Some(Multiplier::Trillion),
                _ => None,
            };
            if let Some(m) = m {
                return Some((end, i, normalized, m, ScaleMarker::Word));
            }
        }
        Some((i, i, normalized, Multiplier::One, ScaleMarker::None))
    }

    /// `CursorQuantityExtractor.extract` and `matchAt`, without money and
    /// units.
    fn extract(&self) -> Vec<Mention> {
        let mut out = Vec::new();
        let mut i = 0isize;
        while (i as usize) < self.c.len() {
            let start = i;
            let negative = self.at(i) == '-';
            let first = start + isize::from(negative);
            let boundary = match negative {
                true => {
                    let b = self.base_before(start);
                    !continuation(b) && b != Some('%') && b.is_none_or(|c| general_category(c) != Sc)
                }
                false => !continuation(self.base_before(start)),
            };
            if self.digit(first)
                && !self.continues(first)
                && boundary
                && let Some((end, numeral_end, normalized, multiplier, marker)) = self.parse(first)
            {
                let (int, frac) = normalized.split_once('.').unwrap_or((&normalized, ""));
                let mut coefficient = format!("{int}{frac}").trim_start_matches('0').to_string();
                if coefficient.is_empty() {
                    coefficient.push('0');
                } else {
                    coefficient.extend(std::iter::repeat_n('0', multiplier.zeros() as usize));
                }
                out.push(Mention {
                    start: start as usize,
                    end: end as usize,
                    numeral_end: numeral_end as usize,
                    negative,
                    coefficient,
                    scale: frac.len() as u32,
                    multiplier,
                    marker,
                });
                i = end;
            } else {
                i = start + 1;
            }
        }
        out
    }
}

/// The scanner's mentions, with byte offsets turned into character ones.
fn scanned(text: &str, scanner: NumberScanner) -> Vec<Mention> {
    let mut n = Numbers::new(ViewId::SOURCE);
    scanner.scan(text, &mut n);
    assert_eq!(n.check(), Ok(()), "{text:?}");
    let chars = |b: u32| text[..b as usize].chars().count();
    n.spans
        .iter()
        .map(|(ix, s)| {
            let i = ix.get();
            Mention {
                start: chars(s.start),
                end: chars(s.end),
                numeral_end: chars(n.numeral_end[i]),
                negative: n.negative[i],
                coefficient: n.coefficient.get(i).to_string(),
                scale: n.scale[i],
                multiplier: n.multiplier[i],
                marker: n.marker[i],
            }
        })
        .collect()
}

/// Pieces of numbers and of what surrounds them: separators of both
/// notations and the unsupported ones, exponents, scale letters and words,
/// signs, marks, letters and digits of other scripts, characters outside
/// the Basic Multilingual Plane, and whitespace of several widths.
const POOL: &[&str] = &[
    "0",
    "1",
    "2",
    "3",
    "5",
    "7",
    "9",
    "0",
    "4",
    "6",
    "8",
    "12",
    "123",
    "1234",
    "000",
    ",",
    ".",
    ",",
    ".",
    "-",
    "-",
    "+",
    "e",
    "E",
    "k",
    "m",
    "b",
    "n",
    "K",
    "M",
    "B",
    "N",
    " ",
    " ",
    " ",
    "\u{A0}",
    "\u{2009}",
    "\u{202F}",
    "'",
    "\u{2019}",
    "\u{66B}",
    "\u{66C}",
    "%",
    "$",
    "\u{20AC}",
    "a",
    "x",
    "_",
    "\u{301}",
    "\u{903}",
    "\u{661}",
    "\u{B2}",
    "\u{2160}",
    "\u{1D48E}",
    "\u{1D7D9}",
    "\u{1D165}",
    "\u{1F4E6}",
    "\u{130}",
    "\u{212A}",
    "\t",
    "\u{85}",
    "\u{3000}",
    "\u{200B}",
    "thousand",
    "million",
    "billion",
    "trillion",
    "MILLION",
    "m\u{130}ll\u{130}on",
    "millions",
    "bn",
    "kg",
    "\u{E9}",
    "\u{4E00}",
    "(",
    ")",
    "\u{10400}",
    "\u{10400}\u{10401}\u{10402}\u{10403}",
];

#[test]
fn the_scanner_agrees_with_a_transcription_over_chars() {
    let mut rng = Rng(0x5eed_2170);
    let mut text = String::new();
    let mut mentions = 0;
    for _ in 0..20_000 {
        text.clear();
        for _ in 0..rng.below(14) {
            text.push_str(POOL[rng.below(POOL.len())]);
        }
        for notation in [NumberNotation::LatinUs, NumberNotation::LatinEu] {
            for apply_scale in [true, false] {
                let reference = Reference {
                    c: text.chars().collect(),
                    group: notation.group_separator(),
                    decimal: notation.decimal_separator(),
                    apply_scale,
                };
                let expected = reference.extract();
                mentions += expected.len();
                let got = scanned(&text, NumberScanner { notation, apply_scale, opennlp_parity: true });
                assert_eq!(got, expected, "{text:?} in {notation:?}, scale {apply_scale}");
            }
        }
    }
    // The texts hold enough numbers for the comparison to mean something.
    assert!(mentions > 20_000, "{mentions}");
}
