//! Shift_JIS and EUC-JP Japanese CJK candidates, ported from chardetng 0.1.17
//! (`src/lib.rs`, Copyright Mozilla Foundation, licensed under Apache-2.0 or
//! MIT; see NOTICE).

use super::{HalfWidthKatakana, LatinCj, cjk_extra_score, problematic_lead};
use crate::detect::data;
use crate::detect::scores::*;
use crate::detect::stream::{EucJpStream, ShiftJisStream};

pub(in crate::detect) struct ShiftJisCandidate {
    decoder: ShiftJisStream,
    half_width_katakana_seen: bool,
    half_width_katakana_state: HalfWidthKatakana,
    prev: LatinCj,
    prev_byte: u8,
    pending_score: Option<i64>,
}

impl ShiftJisCandidate {
    pub(in crate::detect) fn new() -> Self {
        ShiftJisCandidate {
            decoder: ShiftJisStream::default(),
            half_width_katakana_seen: false,
            half_width_katakana_state: HalfWidthKatakana::DakutenForbidden,
            prev: LatinCj::Other,
            prev_byte: 0,
            pending_score: None,
        }
    }

    fn maybe_set_as_pending(&mut self, s: i64) -> i64 {
        assert!(self.pending_score.is_none());
        if self.prev == LatinCj::Cj || !problematic_lead(self.prev_byte) {
            s
        } else {
            self.pending_score = Some(s);
            0
        }
    }

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let fed = self.decoder.feed(b);
            let (dst, written) = (fed.units, fed.written);
            if written > 0 {
                let half_width_katakana_state = self.half_width_katakana_state;
                self.half_width_katakana_state = HalfWidthKatakana::DakutenForbidden;
                let u = dst[0];
                if (u >= u16::from(b'a') && u <= u16::from(b'z')) || (u >= u16::from(b'A') && u <= u16::from(b'Z')) {
                    self.pending_score = None; // Discard pending score
                    if self.prev == LatinCj::Cj {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::AsciiLetter;
                } else if (0xFF61..=0xFF9F).contains(&u) {
                    if !self.half_width_katakana_seen {
                        self.half_width_katakana_seen = true;
                        // To avoid misdetecting title-length inputs
                        score += SHIFT_JIS_INITIAL_HALF_WIDTH_KATAKANA_PENALTY;
                    }
                    self.pending_score = None; // Discard pending score
                    score += HALF_WIDTH_KATAKANA_SCORE;

                    if (0xFF76..=0xFF84).contains(&u) || u == 0xFF73 {
                        self.half_width_katakana_state = HalfWidthKatakana::DakutenAllowed;
                    } else if (0xFF8A..=0xFF8E).contains(&u) {
                        self.half_width_katakana_state = HalfWidthKatakana::DakutenOrHandakutenAllowed;
                    } else if u == 0xFF9E {
                        if half_width_katakana_state == HalfWidthKatakana::DakutenForbidden {
                            score += IMPLAUSIBILITY_PENALTY;
                        } else {
                            score += HALF_WIDTH_KATAKANA_VOICING_SCORE;
                        }
                    } else if u == 0xFF9F {
                        if half_width_katakana_state != HalfWidthKatakana::DakutenOrHandakutenAllowed {
                            score += IMPLAUSIBILITY_PENALTY;
                        } else {
                            score += HALF_WIDTH_KATAKANA_VOICING_SCORE;
                        }
                    }

                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else if (0x3040..0x3100).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    score += SHIFT_JIS_SCORE_PER_KANA;
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else if (0x3400..0xA000).contains(&u) || (0xF900..0xFB00).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    if self.prev_byte < 0x98 || (self.prev_byte == 0x98 && b < 0x73) {
                        score += self.maybe_set_as_pending(
                            SHIFT_JIS_SCORE_PER_LEVEL_1_KANJI + cjk_extra_score(u, &data::DETECTOR_DATA.frequent_kanji),
                        );
                    } else {
                        score += self.maybe_set_as_pending(SHIFT_JIS_SCORE_PER_LEVEL_2_KANJI);
                    }
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else if (0xE000..0xF900).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    score += SHIFT_JIS_PUA_PENALTY;
                    self.prev = LatinCj::Other;
                } else {
                    match u {
                        0x3000 // Distinct from Korean, space
                        | 0x3001 // Distinct from Korean, enumeration comma
                        | 0x3002 // Distinct from Korean, full stop
                        | 0xFF08 // Distinct from Korean, parenthesis
                        | 0xFF09 // Distinct from Korean, parenthesis
                        => {
                            if let Some(pending) = self.pending_score {
                                score += pending;
                                self.pending_score = None;
                            }
                            // Not really needed for CJK distinction
                            // but let's give non-zero score for these
                            // common byte pairs anyway.
                            score += CJ_PUNCTUATION;
                        }
                        0..=0x7F => {
                            self.pending_score = None; // Discard pending score
                        }
                        0x80 => {
                            // This is a control character that overlaps euro
                            // in windows-1252 and happens to be a non-error
                            // is Shift_JIS.
                            self.pending_score = None; // Discard pending score
                            score += IMPLAUSIBILITY_PENALTY;
                        }
                        _ => {
                            if let Some(pending) = self.pending_score {
                                score += pending;
                                self.pending_score = None;
                            }
                            score += CJK_OTHER;
                        }
                    }
                    self.prev = LatinCj::Other;
                }
            }
            match fed.malformed {
                None => {}
                Some(malformed_len) => {
                    if (((self.prev_byte >= 0x81 && self.prev_byte <= 0x9F)
                        || (self.prev_byte >= 0xE0 && self.prev_byte <= 0xFC))
                        && ((0x40..=0x7E).contains(&b) || (0x80..=0xFC).contains(&b)))
                        && !((self.prev_byte == 0x82 && b >= 0xFA)
                            || (self.prev_byte == 0x84 && ((0xDD..=0xE4).contains(&b) || b >= 0xFB))
                            || (self.prev_byte == 0x86 && (0xF2..=0xFA).contains(&b))
                            || (self.prev_byte == 0x87 && (0x77..=0x7D).contains(&b))
                            || (self.prev_byte == 0xFC && b >= 0xF5))
                    {
                        // Shift_JIS2004 or MacJapanese
                        if let Some(pending) = self.pending_score {
                            score += pending;
                            self.pending_score = None;
                        }
                        score += SHIFT_JIS_EXTENSION_PENALTY;
                        // Approximate boundary
                        if self.prev_byte < 0x87 {
                            self.prev = LatinCj::Other;
                        } else {
                            if self.prev == LatinCj::AsciiLetter {
                                score += CJK_LATIN_ADJACENCY_PENALTY;
                            }
                            self.prev = LatinCj::Cj;
                        }
                    } else if malformed_len == 1 && (b == 0xA0 || b >= 0xFD) {
                        self.pending_score = None; // Just in case
                        score += SHIFT_JIS_SINGLE_BYTE_EXTENSION_PENALTY;
                        self.prev = LatinCj::Other;
                    } else {
                        return None;
                    }
                }
            }
            self.prev_byte = b;
        }
        if last && !self.decoder.finish() {
            return None;
        }
        Some(score)
    }
}

pub(in crate::detect) struct EucJpCandidate {
    decoder: EucJpStream,
    non_ascii_seen: bool,
    half_width_katakana_state: HalfWidthKatakana,
    prev: LatinCj,
    prev_byte: u8,
    prev_prev_byte: u8,
}

impl EucJpCandidate {
    pub(in crate::detect) fn new() -> Self {
        EucJpCandidate {
            decoder: EucJpStream::default(),
            non_ascii_seen: false,
            half_width_katakana_state: HalfWidthKatakana::DakutenForbidden,
            prev: LatinCj::Other,
            prev_byte: 0,
            prev_prev_byte: 0,
        }
    }

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let fed = self.decoder.feed(b);
            let (dst, written) = (fed.units, fed.written);
            if written > 0 {
                let half_width_katakana_state = self.half_width_katakana_state;
                self.half_width_katakana_state = HalfWidthKatakana::DakutenForbidden;
                let u = dst[0];
                if !self.non_ascii_seen && u >= 0x80 {
                    self.non_ascii_seen = true;
                    if (0xFF61..=0xFF9F).contains(&u) {
                        // return None;
                    }
                    if (0x3040..0x3100).contains(&u) {
                        // Remove the kana advantage over initial Big5
                        // hanzi.
                        score += EUC_JP_INITIAL_KANA_PENALTY;
                    }
                }
                if (u >= u16::from(b'a') && u <= u16::from(b'z')) || (u >= u16::from(b'A') && u <= u16::from(b'Z')) {
                    if self.prev == LatinCj::Cj {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::AsciiLetter;
                } else if (0xFF61..=0xFF9F).contains(&u) {
                    score += HALF_WIDTH_KATAKANA_SCORE;

                    if (0xFF76..=0xFF84).contains(&u) || u == 0xFF73 {
                        self.half_width_katakana_state = HalfWidthKatakana::DakutenAllowed;
                    } else if (0xFF8A..=0xFF8E).contains(&u) {
                        self.half_width_katakana_state = HalfWidthKatakana::DakutenOrHandakutenAllowed;
                    } else if u == 0xFF9E {
                        if half_width_katakana_state == HalfWidthKatakana::DakutenForbidden {
                            score += IMPLAUSIBILITY_PENALTY;
                        } else {
                            score += HALF_WIDTH_KATAKANA_VOICING_SCORE;
                        }
                    } else if u == 0xFF9F {
                        if half_width_katakana_state != HalfWidthKatakana::DakutenOrHandakutenAllowed {
                            score += IMPLAUSIBILITY_PENALTY;
                        } else {
                            score += HALF_WIDTH_KATAKANA_VOICING_SCORE;
                        }
                    }

                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Other;
                } else if (0x3041..=0x3093).contains(&u) || (0x30A1..=0x30F6).contains(&u) {
                    match u {
                        0x3090 // hiragana wi
                        | 0x3091 // hiragana we
                        | 0x30F0 // katakana wi
                        | 0x30F1 // katakana we
                        => {
                            // Remove advantage over Big5 Hanzi
                            score += EUC_JP_SCORE_PER_NEAR_OBSOLETE_KANA;
                        }
                        _ => {
                            score += EUC_JP_SCORE_PER_KANA;
                        }
                    }
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else if (0x3400..0xA000).contains(&u) || (0xF900..0xFB00).contains(&u) {
                    if self.prev_prev_byte == 0x8F {
                        score += EUC_JP_SCORE_PER_OTHER_KANJI;
                    } else if self.prev_byte < 0xD0 {
                        score += EUC_JP_SCORE_PER_LEVEL_1_KANJI;
                        score += cjk_extra_score(u, &data::DETECTOR_DATA.frequent_kanji);
                    } else {
                        score += EUC_JP_SCORE_PER_LEVEL_2_KANJI;
                    }
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else {
                    match u {
                        0x3000 // Distinct from Korean, space
                        | 0x3001 // Distinct from Korean, enumeration comma
                        | 0x3002 // Distinct from Korean, full stop
                        | 0xFF08 // Distinct from Korean, parenthesis
                        | 0xFF09 // Distinct from Korean, parenthesis
                        => {
                            score += CJ_PUNCTUATION;
                        }
                        0..=0x7F => {}
                        _ => {
                            score += CJK_OTHER;
                        }
                    }
                    self.prev = LatinCj::Other;
                }
            }
            match fed.malformed {
                None => {}
                Some(_) => {
                    if (0xA1..=0xFE).contains(&b)
                        && self.prev_byte >= 0xA1
                        && self.prev_byte <= 0xFE
                        && ((self.prev_prev_byte != 0x8F
                            && !(self.prev_byte == 0xA8 && (0xDF..=0xE6).contains(&b))
                            && !(self.prev_byte == 0xAC && (0xF4..=0xFC).contains(&b))
                            && !(self.prev_byte == 0xAD && (0xD8..=0xDE).contains(&b)))
                            || (self.prev_prev_byte == 0x8F
                                && self.prev_byte != 0xA2
                                && self.prev_byte != 0xA6
                                && self.prev_byte != 0xA7
                                && self.prev_byte != 0xA9
                                && self.prev_byte != 0xAA
                                && self.prev_byte != 0xAB
                                && self.prev_byte != 0xED
                                && !(self.prev_byte == 0xFE && b >= 0xF7)))
                    {
                        score += EUC_JP_EXTENSION_PENALTY;
                        if self.prev == LatinCj::AsciiLetter {
                            score += CJK_LATIN_ADJACENCY_PENALTY;
                        }
                        self.prev = LatinCj::Cj;
                    } else {
                        return None;
                    }
                }
            }
            self.prev_prev_byte = self.prev_byte;
            self.prev_byte = b;
        }
        if last && !self.decoder.finish() {
            return None;
        }
        Some(score)
    }
}
