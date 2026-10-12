//! The alphanumeric check: OpenNLP's `AlphaNumericCheck`.
//!
//! With `useAlphaNumericOptimization` on, a whitespace unit that matches
//! the model's alphanumeric pattern is a token as it stands and the model
//! is not consulted. Every pattern OpenNLP ships has the form
//! `^[characters]+$`, where the characters are single characters and
//! ranges `a-z`. [`AlphaNumeric::parse`] reads exactly that form into a
//! set of code points, as `AlphaNumericCheck.parseCharacterClass` does,
//! and [`AlphaNumeric::test`] checks a unit code point by code point,
//! which is what `java.util.regex` does with such a pattern. A pattern of
//! any other form is refused with the pattern in the error rather than
//! approximated: there is no regular expression engine here.

use std::fmt;

/// `Factory.DEFAULT_ALPHANUMERIC`, the pattern TokenizerME uses when the
/// model has none.
pub const DEFAULT_PATTERN: &str = "^[A-Za-z0-9]+$";

/// The per-language patterns of OpenNLP's `tokenize.lang.Factory`, keyed
/// by the ISO 639 codes `Factory.getAlphanumeric` accepts.
const LANGUAGE_PATTERNS: [(&[&str], &str); 8] = [
    (&["es", "spa"], "^[0-9a-záéíóúüýñA-ZÁÉÍÓÚÝÑ]+$"),
    (&["it", "ita"], "^[0-9a-zàèéìîíòóùüA-ZÀÈÉÌÎÍÒÓÙÜ]+$"),
    (&["pt", "por"], "^[0-9a-záãâàéêíóõôúüçA-ZÁÃÂÀÉÊÍÓÕÔÚÜÇ]+$"),
    (&["ca", "cat"], "^[0-9a-zàèéíïòóúüçA-ZÀÈÉÍÏÒÓÚÜÇ]+$"),
    (&["pl", "pol"], "^[A-Za-z0-9żźćńółęąśŻŹĆĄŚĘŁÓŃ]+$"),
    (&["de", "deu", "ger"], "^[A-Za-z0-9äéöüÄÉÖÜß]+$"),
    (&["fr", "fre", "fra"], "^[a-zA-Z0-9àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+$"),
    (&["nl", "nld", "dut"], "^[A-Za-z0-9äöüëèéïĳÄÖÜËÉÈÏĲ]+$"),
];

/// `Factory.getAlphanumeric(languageCode)`: the language's pattern, or
/// [`DEFAULT_PATTERN`] for a language it does not list.
///
/// TokenizerME itself does not use this for a model it loads: the
/// factory OpenNLP builds for a loaded model never learns the language
/// before it picks the pattern, so a model without `alphaNumericPattern`
/// gets [`DEFAULT_PATTERN`] whatever its language (see
/// [`TokenizerModel`](super::TokenizerModel)).
pub fn language_pattern(language: &str) -> &'static str {
    LANGUAGE_PATTERNS.iter().find(|(codes, _)| codes.contains(&language)).map_or(DEFAULT_PATTERN, |&(_, p)| p)
}

/// A pattern that is not of the form `^[characters]+$`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternError {
    /// The pattern as the model gave it.
    pub pattern: String,
    /// What in it is not supported.
    pub reason: &'static str,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alphanumeric pattern \"{}\": {}", self.pattern, self.reason)
    }
}

impl std::error::Error for PatternError {}

/// The set of code points an alphanumeric pattern accepts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlphaNumeric {
    /// One bit per ASCII code point, for the common case.
    ascii: u128,
    /// The other members as sorted, merged, inclusive ranges.
    ranges: Vec<(u32, u32)>,
}

impl AlphaNumeric {
    /// Reads a pattern of the form `^[characters]+$`.
    ///
    /// The characters are single characters and ranges `x-y` with `x` at
    /// most `y`; a `-` that cannot start a range is a literal, as in the
    /// Java. Refused: any other form, an empty class, the characters
    /// `[ ] \ ^ &` inside the class (escapes such as `\p{L}`, nested
    /// classes, negation, intersection), and a character outside the Basic
    /// Multilingual Plane. Every other character stands for itself, as in
    /// `AlphaNumericCheck.isLiteral`: inside a class `?`, `*`, `+`, `.`,
    /// `(`, `{`, `|` and `$` are plain characters. A range that spans
    /// the surrogates leaves them out, as OpenNLP 3 does.
    pub fn parse(pattern: &str) -> Result<AlphaNumeric, PatternError> {
        let fail = |reason| PatternError { pattern: pattern.to_string(), reason };
        let body = pattern
            .strip_prefix("^[")
            .and_then(|p| p.strip_suffix("]+$"))
            .ok_or_else(|| fail("not of the form ^[characters]+$"))?;
        if body.is_empty() {
            return Err(fail("the character class is empty"));
        }
        let chars: Vec<char> = body.chars().collect();
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let literal = |c: char| -> Result<u32, PatternError> {
            match c {
                '[' | ']' | '\\' | '^' | '&' => {
                    Err(fail("the class holds [, ], \\, ^ or &, which are not plain characters"))
                }
                c if c as u32 > 0xFFFF => Err(fail("the class holds a character outside the Basic Multilingual Plane")),
                c => Ok(c as u32),
            }
        };
        let mut i = 0;
        while i < chars.len() {
            let from = literal(chars[i])?;
            if i + 2 < chars.len() && chars[i + 1] == '-' {
                let to = literal(chars[i + 2])?;
                if to < from {
                    return Err(fail("a range ends before it starts"));
                }
                ranges.push((from, to));
                i += 3;
            } else {
                ranges.push((from, from));
                i += 1;
            }
        }
        Ok(AlphaNumeric::from_ranges(ranges))
    }

    /// The set of `ranges`, which may overlap and be in any order.
    fn from_ranges(mut ranges: Vec<(u32, u32)>) -> AlphaNumeric {
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for (a, b) in ranges {
            match merged.last_mut() {
                Some(last) if a <= last.1.saturating_add(1) => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        let mut ascii = 0u128;
        for &(a, b) in &merged {
            for cp in a..=b.min(0x7F) {
                ascii |= 1 << cp;
            }
        }
        // The surrogates are never members: no `&str` holds one, and
        // OpenNLP 3 takes them out of every range.
        let ranges = merged
            .into_iter()
            .filter(|&(_, b)| b >= 0x80)
            .map(|(a, b)| (a.max(0x80), b))
            .flat_map(|(a, b)| {
                let below = (a < 0xD800).then(|| (a, b.min(0xD7FF)));
                let above = (b > 0xDFFF).then(|| (a.max(0xE000), b));
                below.into_iter().chain(above)
            })
            .collect();
        AlphaNumeric { ascii, ranges }
    }

    /// Whether the set holds `cp`.
    #[inline]
    pub fn contains(&self, cp: u32) -> bool {
        if cp < 0x80 {
            return self.ascii & (1 << cp) != 0;
        }
        let i = self.ranges.partition_point(|&(_, b)| b < cp);
        self.ranges.get(i).is_some_and(|&(a, _)| a <= cp)
    }

    /// Whether every character of `unit` is in the set: the pattern
    /// matches the whole unit. An empty unit does not match.
    #[inline]
    pub fn test(&self, unit: &str) -> bool {
        if unit.is_empty() {
            return false;
        }
        if unit.is_ascii() {
            return unit.bytes().all(|b| self.ascii & (1 << b) != 0);
        }
        unit.chars().all(|c| self.contains(c as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pattern() {
        let a = AlphaNumeric::parse(DEFAULT_PATTERN).unwrap();
        assert!(a.test("abc"));
        assert!(a.test("ABC123"));
        assert!(a.test("Z"));
        assert!(!a.test(""));
        assert!(!a.test("don't"));
        assert!(!a.test("a-b"));
        assert!(!a.test("caf\u{e9}"));
        assert!(!a.test("a b"));
        assert!(!a.test("_"));
    }

    #[test]
    fn every_language_pattern_parses() {
        for (codes, pattern) in LANGUAGE_PATTERNS {
            let a = AlphaNumeric::parse(pattern).unwrap_or_else(|e| panic!("{e}"));
            assert!(a.test("Abc0"), "{codes:?}");
            assert!(!a.test("a.b"), "{codes:?}");
            for code in codes {
                assert_eq!(language_pattern(code), pattern);
            }
        }
        assert_eq!(language_pattern("en"), DEFAULT_PATTERN);
        assert_eq!(language_pattern(""), DEFAULT_PATTERN);
        assert_eq!(language_pattern("ES"), DEFAULT_PATTERN, "codes are compared exactly, as in the Java");
    }

    #[test]
    fn language_letters_match_their_pattern() {
        let de = AlphaNumeric::parse(language_pattern("de")).unwrap();
        assert!(de.test("Stra\u{df}e"));
        assert!(de.test("Caf\u{e9}"));
        assert!(!de.test("gar\u{e7}on"));
        let fr = AlphaNumeric::parse(language_pattern("fr")).unwrap();
        assert!(fr.test("gar\u{e7}on"));
        assert!(fr.test("c\u{153}ur"));
        assert!(fr.test("\u{178}"));
        let nl = AlphaNumeric::parse(language_pattern("nl")).unwrap();
        assert!(nl.test("\u{133}s"));
        let pl = AlphaNumeric::parse(language_pattern("pl")).unwrap();
        assert!(pl.test("\u{17c}\u{f3}\u{142}w"));
        assert!(!pl.test("\u{e4}"));
    }

    #[test]
    fn the_golden_manifest_patterns() {
        // As the manifests store them after `\uXXXX` unescaping.
        let es = AlphaNumeric::parse(
            "^[0-9a-z\u{e1}\u{e9}\u{ed}\u{f3}\u{fa}\u{fc}\u{fd}\u{f1}A-Z\u{c1}\u{c9}\u{cd}\u{d3}\u{da}\u{dd}\u{d1}]+$",
        )
        .unwrap();
        assert!(es.test("Espa\u{f1}a"));
        assert!(!es.test("\u{e0}"));
    }

    #[test]
    fn ranges_and_literal_hyphens() {
        let a = AlphaNumeric::parse("^[a-c-]+$").unwrap();
        assert!(a.test("abc-"));
        assert!(!a.test("d"));
        let a = AlphaNumeric::parse("^[-x]+$").unwrap();
        assert!(a.test("-x-"));
        // "a-" at the end is two literals, as in the Java.
        let a = AlphaNumeric::parse("^[a-]+$").unwrap();
        assert!(a.test("a-"));
        assert!(!a.test("b"));
        // "!--" is the range from ! to -.
        let a = AlphaNumeric::parse("^[!--]+$").unwrap();
        assert!(a.test("!#,-"));
        assert!(!a.test("."));
        // Overlapping and adjacent ranges merge.
        let a = AlphaNumeric::parse("^[a-mf-zA]+$").unwrap();
        assert_eq!(a.ranges, []);
        assert!(a.test("azA"));
    }

    #[test]
    fn ranges_beyond_ascii_and_around_the_surrogates() {
        let a = AlphaNumeric::parse("^[\u{3b1}-\u{3c9}]+$").unwrap();
        assert!(a.test("\u{3b1}\u{3b2}\u{3c9}"));
        assert!(!a.test("\u{391}"));
        let a = AlphaNumeric::parse("^[\u{d000}-\u{ffff}]+$").unwrap();
        assert_eq!(a.ranges, [(0xD000, 0xD7FF), (0xE000, 0xFFFF)]);
        assert!(a.contains(0xD7FF));
        assert!(!a.contains(0xD800));
        assert!(!a.contains(0xDFFF));
        assert!(a.contains(0xE000));
        assert!(!a.test("\u{1f600}"));
    }

    #[test]
    fn regex_metacharacters_are_plain_inside_the_class() {
        let a = AlphaNumeric::parse("^[a-z.*+?(){}|$]+$").unwrap();
        let mut members: Vec<u32> = ('a'..='z').map(|c| c as u32).collect();
        members.extend(".*+?(){}|$".chars().map(|c| c as u32));
        members.sort_unstable();
        let all: Vec<u32> = (0..0x80).filter(|&cp| a.contains(cp)).collect();
        assert_eq!(all, members);
        assert!(a.ranges.is_empty());
        assert!(a.test("a.b*c?"));
        assert!(a.test("${x}|(y)+"));
        assert!(!a.test("a-b"));
        assert!(!a.test("A"));
    }

    #[test]
    fn unsupported_patterns_are_refused_with_the_pattern() {
        let refused = [
            ("[A-Za-z]+", "not of the form ^[characters]+$"),
            ("^[A-Za-z]*$", "not of the form ^[characters]+$"),
            ("^[A-Za-z]+", "not of the form ^[characters]+$"),
            ("^\\w+$", "not of the form ^[characters]+$"),
            ("^[]+$", "the character class is empty"),
            ("^[\\p{L}0-9]+$", "the class holds [, ], \\, ^ or &, which are not plain characters"),
            ("^[^a-z]+$", "the class holds [, ], \\, ^ or &, which are not plain characters"),
            ("^[a-z&&[^x]]+$", "the class holds [, ], \\, ^ or &, which are not plain characters"),
            ("^[z-a]+$", "a range ends before it starts"),
            ("^[a\u{1f600}]+$", "the class holds a character outside the Basic Multilingual Plane"),
        ];
        for (pattern, reason) in refused {
            let e = AlphaNumeric::parse(pattern).unwrap_err();
            assert_eq!(e, PatternError { pattern: pattern.into(), reason }, "{pattern}");
            assert!(e.to_string().contains(pattern));
        }
    }
}
