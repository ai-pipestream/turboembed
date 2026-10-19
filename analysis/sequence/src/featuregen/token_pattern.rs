//! `TokenPatternFeatureGenerator`.

use turbo_jvm::chars::Utf16Units;
use turbo_unicode::WhitespaceMode;
use turbo_unicode::java::code_point::{is_alphabetic, is_digit};

use super::FeatureGenerator;
use crate::features::Features;
use crate::pattern::{PatternMode, token_feature_utf16};

/// `TokenPatternFeatureGenerator` with its default tokenizer,
/// `SimpleTokenizer.INSTANCE`: splits the token into runs of letters,
/// digits and single kinds of other character, then writes
///
/// - for a token of one piece, `st=` and the lowercased token, and
///   nothing else;
/// - otherwise `stn=` and the number of pieces; for each piece `pt2=` with
///   the classes of it and the next piece, `pt3=` with the classes of it
///   and the next two, and `st=` with the lowercased piece unless it holds
///   something other than an ASCII letter; and last `pta=` with the
///   classes of all the pieces.
///
/// `SimpleTokenizer` classifies UTF-16 code units, so a character outside
/// the Basic Multilingual Plane is two pieces of one lone surrogate each;
/// they are kept as code units here so their features come out as Java's
/// do. Whitespace is OpenNLP's default (`WhitespaceMode.UNICODE`).
pub struct TokenPattern {
    mode: PatternMode,
    units: Vec<u16>,
    pieces: Vec<(usize, usize)>,
    pattern: String,
}

impl TokenPattern {
    pub fn new(mode: PatternMode) -> TokenPattern {
        TokenPattern { mode, units: Vec::new(), pieces: Vec::new(), pattern: String::new() }
    }

    /// `SimpleTokenizer.tokenizePos` on `self.units`, without new lines
    /// kept.
    fn split(&mut self) {
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum Class {
            Whitespace,
            Alphabetic,
            Numeric,
            Other,
        }
        self.pieces.clear();
        let mut state = Class::Whitespace;
        let mut class = Class::Whitespace;
        let mut start = 0;
        let mut previous = 0u16;
        for (i, &c) in self.units.iter().enumerate() {
            let cp = u32::from(c);
            class = if WhitespaceMode::Unicode.is_whitespace(cp) {
                Class::Whitespace
            } else if is_alphabetic(cp) {
                Class::Alphabetic
            } else if is_digit(cp) {
                Class::Numeric
            } else {
                Class::Other
            };
            if state == Class::Whitespace {
                if class != Class::Whitespace {
                    start = i;
                }
            } else if class != state || (class == Class::Other && c != previous) {
                self.pieces.push((start, i));
                start = i;
            }
            state = class;
            previous = c;
        }
        if class != Class::Whitespace {
            self.pieces.push((start, self.units.len()));
        }
    }

    fn class(&self, piece: usize) -> &'static str {
        let (start, end) = self.pieces[piece];
        token_feature_utf16(&self.units[start..end], self.mode)
    }
}

impl FeatureGenerator for TokenPattern {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        self.units.clear();
        self.units.extend(Utf16Units::new(token).map(|u| u.unit));
        self.split();
        let n = self.pieces.len();
        if n == 1 {
            out.push_str("st=").push_lower(token).finish();
            return;
        }
        out.push_str("stn=").push_int(n).finish();
        self.pattern.clear();
        for i in 0..n {
            if i + 1 < n {
                out.push_str("pt2=").push_str(self.class(i)).push_str(self.class(i + 1)).finish();
            }
            if i + 2 < n {
                out.push_str("pt3=")
                    .push_str(self.class(i))
                    .push_str(self.class(i + 1))
                    .push_str(self.class(i + 2))
                    .finish();
            }
            let class = self.class(i);
            self.pattern.push_str(class);
            let (start, end) = self.pieces[i];
            let piece = &self.units[start..end];
            if piece.iter().all(|&u| matches!(u, 0x41..=0x5A | 0x61..=0x7A)) {
                out.push_str("st=").push_lower_utf16(piece).finish();
            }
        }
        out.push_str("pta=").push_str(&self.pattern).finish();
    }
}
