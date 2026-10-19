//! The single-byte candidates of the detector, ported from chardetng
//! 0.1.17 (`src/lib.rs`, Copyright Mozilla Foundation, licensed under
//! Apache-2.0 or MIT; see NOTICE). Each scores byte pairs by their classes
//! in `data` and disqualifies itself on a byte its encoding leaves
//! unmapped.

mod other;

pub(super) use other::{ArabicFrenchCandidate, CaselessCandidate, LogicalCandidate, VisualCandidate};

use super::data::*;
use super::scores::*;

// For Latin, we only penalize pairwise bad transitions
// if one participant is non-ASCII. This avoids violating
// the principle that ASCII pairs never contribute to the
// score. (Maybe that's a bad principle, though!)
#[derive(PartialEq)]
pub(super) enum LatinCaseState {
    Space,
    Upper,
    Lower,
    AllCaps,
}

// Fon non-Latin, we calculate case-related penalty
// or bonus on a per-non-Latin-word basis.
#[derive(PartialEq)]
enum NonLatinCaseState {
    Space,
    Upper,
    Lower,
    UpperLower,
    AllCaps,
    Mix,
}

pub(super) struct NonLatinCasedCandidate {
    pub(super) data: &'static SingleByteData,
    prev: u8,
    case_state: NonLatinCaseState,
    prev_ascii: bool,
    current_word_len: u64,
    pub(super) longest_word: u64,
    ibm866: bool,
    prev_was_a0: bool, // Only used with IBM866
}

impl NonLatinCasedCandidate {
    pub(super) fn new(data: &'static SingleByteData) -> Self {
        NonLatinCasedCandidate {
            data,
            prev: 0,
            case_state: NonLatinCaseState::Space,
            prev_ascii: true,
            current_word_len: 0,
            longest_word: 0,
            ibm866: data == &SINGLE_BYTE_DATA[IBM866_INDEX],
            prev_was_a0: false,
        }
    }

    pub(super) fn feed(&mut self, buffer: &[u8]) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let class = self.data.classify(b);
            if class == 255 {
                return None;
            }
            let caseless_class = class & 0x7F;

            let ascii = b < 0x80;
            let ascii_pair = self.prev_ascii && ascii;

            let non_ascii_alphabetic = self.data.is_non_latin_alphabetic(caseless_class, false);

            // The purpose of this state machine is to avoid misdetecting Greek as
            // Cyrillic by:
            //
            // * Giving a small bonus to words that start with an upper-case letter
            //   and are lower-case for the rest.
            // * Giving a large penalty to start with one lower-case letter followed
            //   by all upper-case (obviously upper and lower case inverted, which
            //   unfortunately is possible due to KOI8-U).
            // * Giving a small per-word penalty to all-uppercase KOI8-U (to favor
            //   all-lowercase Greek over all-caps KOI8-U).
            // * Giving large penalties for mixed-case other than initial upper-case.
            //   This also helps relative to non-cased encodings.

            // ASCII doesn't participate in non-Latin casing.
            if caseless_class == LATIN_LETTER {
                // Latin
                // Mark this word as a mess. If there end up being non-Latin
                // letters in this word, the ASCII-adjacency penalty gets
                // applied to Latin/non-Latin pairs and the mix penalty
                // to non-Latin/non-Latin pairs.
                // XXX Apply penalty here
                self.case_state = NonLatinCaseState::Mix;
            } else if !non_ascii_alphabetic {
                // Space
                match self.case_state {
                    NonLatinCaseState::Space | NonLatinCaseState::Upper | NonLatinCaseState::Lower => {}
                    NonLatinCaseState::UpperLower => {
                        // Intentionally applied only once per word.
                        score += NON_LATIN_CAPITALIZATION_BONUS;
                    }
                    NonLatinCaseState::AllCaps => {
                        // Intentionally applied only once per word.
                        if self.data == &SINGLE_BYTE_DATA[KOI8_U_INDEX] {
                            // Apply only to KOI8-U.
                            score += NON_LATIN_ALL_CAPS_PENALTY;
                        }
                    }
                    NonLatinCaseState::Mix => {
                        // Per letter
                        score += NON_LATIN_MIXED_CASE_PENALTY * (self.current_word_len as i64);
                    }
                }
                self.case_state = NonLatinCaseState::Space;
            } else if (class >> 7) == 0 {
                // Lower case
                match self.case_state {
                    NonLatinCaseState::Space => {
                        self.case_state = NonLatinCaseState::Lower;
                    }
                    NonLatinCaseState::Upper => {
                        self.case_state = NonLatinCaseState::UpperLower;
                    }
                    NonLatinCaseState::Lower | NonLatinCaseState::UpperLower | NonLatinCaseState::Mix => {}
                    NonLatinCaseState::AllCaps => {
                        self.case_state = NonLatinCaseState::Mix;
                    }
                }
            } else {
                // Upper case
                match self.case_state {
                    NonLatinCaseState::Space => {
                        self.case_state = NonLatinCaseState::Upper;
                    }
                    NonLatinCaseState::Upper => {
                        self.case_state = NonLatinCaseState::AllCaps;
                    }
                    NonLatinCaseState::Lower | NonLatinCaseState::UpperLower => {
                        self.case_state = NonLatinCaseState::Mix;
                    }
                    NonLatinCaseState::AllCaps | NonLatinCaseState::Mix => {}
                }
            }

            // XXX Apply penalty if > 16
            if non_ascii_alphabetic {
                self.current_word_len += 1;
            } else {
                if self.current_word_len > self.longest_word {
                    self.longest_word = self.current_word_len;
                }
                self.current_word_len = 0;
            }

            let is_a0 = b == 0xA0;
            if !ascii_pair {
                // 0xA0 is no-break space in many other encodings, so avoid
                // assigning score to IBM866 when 0xA0 occurs next to itself
                // or a space-like byte.
                if !(self.ibm866
                    && ((is_a0 && (self.prev_was_a0 || self.prev == 0)) || caseless_class == 0 && self.prev_was_a0))
                {
                    score += self.data.score(caseless_class, self.prev, false);
                }

                if (self.prev == LATIN_LETTER && non_ascii_alphabetic)
                    || (caseless_class == LATIN_LETTER && self.data.is_non_latin_alphabetic(self.prev, false))
                {
                    score += LATIN_ADJACENCY_PENALTY;
                }
            }

            self.prev_ascii = ascii;
            self.prev = caseless_class;
            self.prev_was_a0 = is_a0;
        }
        Some(score)
    }
}

enum OrdinalState {
    Other,
    Space,
    PeriodAfterN,
    OrdinalExpectingSpace,
    OrdinalExpectingSpaceUndoImplausibility,
    OrdinalExpectingSpaceOrDigit,
    OrdinalExpectingSpaceOrDigitUndoImplausibily,
    UpperN,
    LowerN,
    FeminineAbbreviationStartLetter,
    Digit,
    Roman,
    Copyright,
}

pub(super) struct LatinCandidate {
    pub(super) data: &'static SingleByteData,
    prev: u8,
    case_state: LatinCaseState,
    prev_non_ascii: u32,
    ordinal_state: OrdinalState, // Used only when `windows1252 == true`
    windows1252: bool,
}

impl LatinCandidate {
    pub(super) fn new(data: &'static SingleByteData) -> Self {
        LatinCandidate {
            data,
            prev: 0,
            case_state: LatinCaseState::Space,
            prev_non_ascii: 0,
            ordinal_state: OrdinalState::Space,
            windows1252: data == &SINGLE_BYTE_DATA[WINDOWS_1252_INDEX],
        }
    }

    pub(super) fn feed(&mut self, buffer: &[u8]) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let class = self.data.classify(b);
            if class == 255 {
                return None;
            }
            let caseless_class = class & 0x7F;

            let ascii = b < 0x80;
            let ascii_pair = self.prev_non_ascii == 0 && ascii;

            let non_ascii_penalty = match self.prev_non_ascii {
                0..=2 => 0,
                3 => -5,
                4 => -20,
                _ => -200,
            };
            score += non_ascii_penalty;
            // XXX if has Vietnamese-only characters and word length > 7,
            // apply penalty

            if !self.data.is_latin_alphabetic(caseless_class) {
                self.case_state = LatinCaseState::Space;
            } else if (class >> 7) == 0 {
                // Penalizing lower case after two upper case
                // is important for avoiding misdetecting
                // windows-1250 as windows-1252 (byte 0x9F).
                if self.case_state == LatinCaseState::AllCaps && !ascii_pair {
                    score += IMPLAUSIBLE_LATIN_CASE_TRANSITION_PENALTY;
                }
                self.case_state = LatinCaseState::Lower;
            } else {
                match self.case_state {
                    LatinCaseState::Space => {
                        self.case_state = LatinCaseState::Upper;
                    }
                    LatinCaseState::Upper | LatinCaseState::AllCaps => {
                        self.case_state = LatinCaseState::AllCaps;
                    }
                    LatinCaseState::Lower => {
                        if !ascii_pair {
                            // XXX How bad is this for Irish Gaelic?
                            score += IMPLAUSIBLE_LATIN_CASE_TRANSITION_PENALTY;
                        }
                        self.case_state = LatinCaseState::Upper;
                    }
                }
            }

            // Treat pairing space-like, which can be non-ASCII, with ASCII as
            // ASCIIish enough not to get a score in order to avoid giving
            // ASCII i and I in windows-1254 next to windows-125x apostrophe/quote
            // a score. This avoids detecting English I’ as Turkish.
            let ascii_ish_pair =
                ascii_pair || (ascii && self.prev == 0) || (caseless_class == 0 && self.prev_non_ascii == 0);

            if !ascii_ish_pair {
                score += self.data.score(caseless_class, self.prev, false);
            }

            if self.windows1252 {
                // This state machine assigns score to the sequences
                // * " º " (Spanish)
                // * " ª " (Spanish)
                // * ".ª " (Spanish)
                // * ".º " (Spanish)
                // * "n.º1" (Spanish)
                // * " Mª " (Spanish)
                // * " Dª " (Spanish)
                // * " Nª " (Spanish)
                // * " Sª " (Spanish)
                // * " 3º " (Italian, where 3 is an ASCII digit)
                // * " 3ª " (Italian, where 3 is an ASCII digit)
                // * " Xº " (Italian, where X is a small Roman numeral)
                // * " Xª " (Italian, where X is a small Roman numeral)
                // * " Nº1" (Italian, where 1 is an ASCII digit)
                // * " Nº " (Italian)
                // * " © " (otherwise ASCII-only)
                // which are problematic to deal with by pairwise scoring
                // without messing up Romanian detection.
                // Initial sc
                match self.ordinal_state {
                    OrdinalState::Other => {
                        if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        }
                    }
                    OrdinalState::Space => {
                        if caseless_class == 0 {
                            // pass
                        } else if b == 0xAA || b == 0xBA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpace;
                        } else if b == b'M' || b == b'D' || b == b'S' {
                            self.ordinal_state = OrdinalState::FeminineAbbreviationStartLetter;
                        } else if b == b'N' {
                            // numero or Nuestra
                            self.ordinal_state = OrdinalState::UpperN;
                        } else if b == b'n' {
                            // numero
                            self.ordinal_state = OrdinalState::LowerN;
                        } else if caseless_class == (ASCII_DIGIT as u8) {
                            self.ordinal_state = OrdinalState::Digit;
                        } else if caseless_class == 9 /* I */ || caseless_class == 22 /* V */ || caseless_class == 24
                        /* X */
                        {
                            self.ordinal_state = OrdinalState::Roman;
                        } else if b == 0xA9 {
                            self.ordinal_state = OrdinalState::Copyright;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::OrdinalExpectingSpace => {
                        if caseless_class == 0 {
                            score += ORDINAL_BONUS;
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::OrdinalExpectingSpaceUndoImplausibility => {
                        if caseless_class == 0 {
                            score += ORDINAL_BONUS - IMPLAUSIBILITY_PENALTY;
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::OrdinalExpectingSpaceOrDigit => {
                        if caseless_class == 0 {
                            score += ORDINAL_BONUS;
                            self.ordinal_state = OrdinalState::Space;
                        } else if caseless_class == (ASCII_DIGIT as u8) {
                            score += ORDINAL_BONUS;
                            // Deliberately set to `Other`
                            self.ordinal_state = OrdinalState::Other;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::OrdinalExpectingSpaceOrDigitUndoImplausibily => {
                        if caseless_class == 0 {
                            score += ORDINAL_BONUS - IMPLAUSIBILITY_PENALTY;
                            self.ordinal_state = OrdinalState::Space;
                        } else if caseless_class == (ASCII_DIGIT as u8) {
                            score += ORDINAL_BONUS - IMPLAUSIBILITY_PENALTY;
                            // Deliberately set to `Other`
                            self.ordinal_state = OrdinalState::Other;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::UpperN => {
                        if b == 0xAA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpaceUndoImplausibility;
                        } else if b == 0xBA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpaceOrDigitUndoImplausibily;
                        } else if b == b'.' {
                            self.ordinal_state = OrdinalState::PeriodAfterN;
                        } else if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::LowerN => {
                        if b == 0xBA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpaceOrDigitUndoImplausibily;
                        } else if b == b'.' {
                            self.ordinal_state = OrdinalState::PeriodAfterN;
                        } else if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::FeminineAbbreviationStartLetter => {
                        if b == 0xAA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpaceUndoImplausibility;
                        } else if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::Digit => {
                        if b == 0xAA || b == 0xBA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpace;
                        } else if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        } else if caseless_class == (ASCII_DIGIT as u8) {
                            // pass
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::Roman => {
                        if b == 0xAA || b == 0xBA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpaceUndoImplausibility;
                        } else if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        } else if caseless_class == 9 /* I */ || caseless_class == 22 /* V */ || caseless_class == 24
                        /* X */
                        {
                            // pass
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::PeriodAfterN => {
                        if b == 0xBA {
                            self.ordinal_state = OrdinalState::OrdinalExpectingSpaceOrDigit;
                        } else if caseless_class == 0 {
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                    OrdinalState::Copyright => {
                        if caseless_class == 0 {
                            score += COPYRIGHT_BONUS;
                            self.ordinal_state = OrdinalState::Space;
                        } else {
                            self.ordinal_state = OrdinalState::Other;
                        }
                    }
                }
            }

            if ascii {
                self.prev_non_ascii = 0;
            } else {
                self.prev_non_ascii += 1;
            }
            self.prev = caseless_class;
        }
        Some(score)
    }
}
