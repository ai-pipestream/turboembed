//! Generators that read one token: its text, its class, its prefixes and
//! suffixes, its character n-grams.

use std::collections::HashSet;

use turbo_jvm::chars::utf16_len;
use turbo_unicode::java::code_point::to_lower_case;

use super::FeatureGenerator;
use crate::features::Features;
use crate::pattern::{PatternMode, token_feature};

/// `TokenFeatureGenerator`: `w=` and the token, lowercased by
/// `StringUtil.toLowerCase` unless `lowercase` is false.
pub struct TokenGenerator {
    lowercase: bool,
}

impl TokenGenerator {
    pub fn new(lowercase: bool) -> TokenGenerator {
        TokenGenerator { lowercase }
    }
}

impl FeatureGenerator for TokenGenerator {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        out.push_str("w=");
        if self.lowercase {
            out.push_lower(tokens[index]);
        } else {
            out.push_str(tokens[index]);
        }
        out.finish();
    }
}

/// `TokenClassFeatureGenerator`: `wc=` and the token's class
/// ([`token_feature`]), and with `word_and_class` also `w&c=`, the
/// lowercased token, a comma and the class.
///
/// The Java constructor's default is `false` and the descriptor factory's
/// default is `true`; [`descriptor`](super::descriptor) uses the factory's.
pub struct TokenClass {
    word_and_class: bool,
    mode: PatternMode,
}

impl TokenClass {
    pub fn new(word_and_class: bool, mode: PatternMode) -> TokenClass {
        TokenClass { word_and_class, mode }
    }
}

impl FeatureGenerator for TokenClass {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let class = token_feature(token, self.mode);
        out.push_str("wc=").push_str(class).finish();
        if self.word_and_class {
            out.push_str("w&c=").push_lower(token).push_str(",").push_str(class).finish();
        }
    }
}

/// `PrefixFeatureGenerator`: `pre=` and each of the token's first
/// 1, 2, ... `length` UTF-16 code units, as many as the token has.
pub struct Prefix {
    length: usize,
}

impl Prefix {
    /// The default length, `PrefixFeatureGenerator.DEFAULT_MAX_LENGTH`.
    pub const DEFAULT_LENGTH: usize = 4;

    pub fn new(length: usize) -> Prefix {
        Prefix { length }
    }
}

impl FeatureGenerator for Prefix {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let n = self.length.min(utf16_len(token));
        for len in 1..=n {
            out.push_str("pre=").push_units(token, 0, len).finish();
        }
    }
}

/// `SuffixFeatureGenerator`: `suf=` and each of the token's last
/// 1, 2, ... `length` UTF-16 code units, as many as the token has.
pub struct Suffix {
    length: usize,
}

impl Suffix {
    /// The default length, `SuffixFeatureGenerator.DEFAULT_MAX_LENGTH`.
    pub const DEFAULT_LENGTH: usize = 4;

    pub fn new(length: usize) -> Suffix {
        Suffix { length }
    }
}

impl FeatureGenerator for Suffix {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let units = utf16_len(token);
        let n = self.length.min(units);
        for len in 1..=n {
            out.push_str("suf=").push_units(token, units - len, units).finish();
        }
    }
}

/// `CharacterNgramFeatureGenerator`: `ng=` and every distinct substring of
/// the lowercased token from `min` to `max` UTF-16 code units long,
/// shortest first and then from left to right, each written once where
/// it first occurs (`NGramModel` keeps them in a `LinkedHashMap`).
pub struct CharacterNgram {
    min: usize,
    max: usize,
    lowered: Vec<u16>,
}

impl CharacterNgram {
    pub fn new(min: usize, max: usize) -> CharacterNgram {
        CharacterNgram { min, max, lowered: Vec::new() }
    }
}

impl FeatureGenerator for CharacterNgram {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        // `StringUtil.toLowerCaseCharBuffer`: each code point lowercased,
        // as UTF-16.
        self.lowered.clear();
        let mut buf = [0u16; 2];
        for c in tokens[index].chars() {
            let lower = to_lower_case(c as u32);
            let lower = char::from_u32(lower).expect("a lowercase mapping is a character");
            self.lowered.extend_from_slice(lower.encode_utf16(&mut buf));
        }
        // Java bounds the loops by the token's length and reads the
        // lowercased buffer; the simple lowercase mapping never changes
        // the number of UTF-16 units.
        let length = utf16_len(tokens[index]);
        debug_assert_eq!(length, self.lowered.len());
        let mut seen: HashSet<&[u16]> = HashSet::new();
        for len in self.min..=self.max {
            if len > length {
                // No n-gram is longer than the token, and the lengths only
                // grow from here.
                break;
            }
            for start in 0..=(length - len) {
                let gram = &self.lowered[start..start + len];
                if seen.insert(gram) {
                    out.push_str("ng=").push_lower_utf16(gram).finish();
                }
            }
        }
    }
}
