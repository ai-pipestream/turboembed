//! Arabic/French, caseless, and logical/visual Hebrew single-byte candidates,
//! ported from chardetng 0.1.17 (`src/lib.rs`, Copyright Mozilla Foundation,
//! licensed under Apache-2.0 or MIT; see NOTICE).

use super::LatinCaseState;
use crate::detect::data::*;
use crate::detect::scores::*;

pub(in crate::detect) struct ArabicFrenchCandidate {
    pub(in crate::detect) data: &'static SingleByteData,
    prev: u8,
    case_state: LatinCaseState,
    prev_ascii: bool,
    current_word_len: u64,
    pub(in crate::detect) longest_word: u64,
}

impl ArabicFrenchCandidate {
    pub(in crate::detect) fn new(data: &'static SingleByteData) -> Self {
        ArabicFrenchCandidate {
            data,
            prev: 0,
            case_state: LatinCaseState::Space,
            prev_ascii: true,
            current_word_len: 0,
            longest_word: 0,
        }
    }

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8]) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let class = self.data.classify(b);
            if class == 255 {
                return None;
            }
            let caseless_class = class & 0x7F;

            let ascii = b < 0x80;
            let ascii_pair = self.prev_ascii && ascii;

            if caseless_class != LATIN_LETTER {
                // We compute case penalties for French only
                self.case_state = LatinCaseState::Space;
            } else if (class >> 7) == 0 {
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
                            score += IMPLAUSIBLE_LATIN_CASE_TRANSITION_PENALTY;
                        }
                        self.case_state = LatinCaseState::Upper;
                    }
                }
            }

            // Count only Arabic word length and ignore French
            let non_ascii_alphabetic = self.data.is_non_latin_alphabetic(caseless_class, true);
            // XXX apply penalty if > 23
            if non_ascii_alphabetic {
                self.current_word_len += 1;
            } else {
                if self.current_word_len > self.longest_word {
                    self.longest_word = self.current_word_len;
                }
                self.current_word_len = 0;
            }

            if !ascii_pair {
                score += self.data.score(caseless_class, self.prev, true);

                if (self.prev == LATIN_LETTER && non_ascii_alphabetic)
                    || (caseless_class == LATIN_LETTER && self.data.is_non_latin_alphabetic(self.prev, true))
                {
                    score += LATIN_ADJACENCY_PENALTY;
                }
            }

            self.prev_ascii = ascii;
            self.prev = caseless_class;
        }
        Some(score)
    }
}

pub(in crate::detect) struct CaselessCandidate {
    pub(in crate::detect) data: &'static SingleByteData,
    prev: u8,
    prev_ascii: bool,
    current_word_len: u64,
    pub(in crate::detect) longest_word: u64,
}

impl CaselessCandidate {
    pub(in crate::detect) fn new(data: &'static SingleByteData) -> Self {
        CaselessCandidate { data, prev: 0, prev_ascii: true, current_word_len: 0, longest_word: 0 }
    }

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8]) -> Option<i64> {
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
            // Apply penalty if > 23 and not Thai
            if non_ascii_alphabetic {
                self.current_word_len += 1;
            } else {
                if self.current_word_len > self.longest_word {
                    self.longest_word = self.current_word_len;
                }
                self.current_word_len = 0;
            }

            if !ascii_pair {
                score += self.data.score(caseless_class, self.prev, false);

                if (self.prev == LATIN_LETTER && non_ascii_alphabetic)
                    || (caseless_class == LATIN_LETTER && self.data.is_non_latin_alphabetic(self.prev, false))
                {
                    score += LATIN_ADJACENCY_PENALTY;
                }
            }

            self.prev_ascii = ascii;
            self.prev = caseless_class;
        }
        Some(score)
    }
}

fn is_ascii_punctuation(byte: u8) -> bool {
    matches!(byte, b'.' | b',' | b':' | b';' | b'?' | b'!')
}

pub(in crate::detect) struct LogicalCandidate {
    pub(in crate::detect) data: &'static SingleByteData,
    prev: u8,
    prev_ascii: bool,
    pub(in crate::detect) plausible_punctuation: u64,
    current_word_len: u64,
    pub(in crate::detect) longest_word: u64,
}

impl LogicalCandidate {
    pub(in crate::detect) fn new(data: &'static SingleByteData) -> Self {
        LogicalCandidate {
            data,
            prev: 0,
            prev_ascii: true,
            plausible_punctuation: 0,
            current_word_len: 0,
            longest_word: 0,
        }
    }

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8]) -> Option<i64> {
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
            // XXX apply penalty if > 22
            if non_ascii_alphabetic {
                self.current_word_len += 1;
            } else {
                if self.current_word_len > self.longest_word {
                    self.longest_word = self.current_word_len;
                }
                self.current_word_len = 0;
            }

            if !ascii_pair {
                score += self.data.score(caseless_class, self.prev, false);

                let prev_non_ascii_alphabetic = self.data.is_non_latin_alphabetic(self.prev, false);
                if caseless_class == 0 && prev_non_ascii_alphabetic && is_ascii_punctuation(b) {
                    self.plausible_punctuation += 1;
                }

                if (self.prev == LATIN_LETTER && non_ascii_alphabetic)
                    || (caseless_class == LATIN_LETTER && prev_non_ascii_alphabetic)
                {
                    score += LATIN_ADJACENCY_PENALTY;
                }
            }

            self.prev_ascii = ascii;
            self.prev = caseless_class;
        }
        Some(score)
    }
}

pub(in crate::detect) struct VisualCandidate {
    pub(in crate::detect) data: &'static SingleByteData,
    prev: u8,
    prev_ascii: bool,
    prev_punctuation: bool,
    pub(in crate::detect) plausible_punctuation: u64,
    current_word_len: u64,
    pub(in crate::detect) longest_word: u64,
}

impl VisualCandidate {
    pub(in crate::detect) fn new(data: &'static SingleByteData) -> Self {
        VisualCandidate {
            data,
            prev: 0,
            prev_ascii: true,
            prev_punctuation: false,
            plausible_punctuation: 0,
            current_word_len: 0,
            longest_word: 0,
        }
    }

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8]) -> Option<i64> {
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
            // XXX apply penalty if > 22
            if non_ascii_alphabetic {
                self.current_word_len += 1;
            } else {
                if self.current_word_len > self.longest_word {
                    self.longest_word = self.current_word_len;
                }
                self.current_word_len = 0;
            }

            if !ascii_pair {
                score += self.data.score(caseless_class, self.prev, false);

                if non_ascii_alphabetic && self.prev_punctuation {
                    self.plausible_punctuation += 1;
                }

                if (self.prev == LATIN_LETTER && non_ascii_alphabetic)
                    || (caseless_class == LATIN_LETTER && self.data.is_non_latin_alphabetic(self.prev, false))
                {
                    score += LATIN_ADJACENCY_PENALTY;
                }
            }

            self.prev_ascii = ascii;
            self.prev = caseless_class;
            self.prev_punctuation = caseless_class == 0 && is_ascii_punctuation(b);
        }
        Some(score)
    }
}
