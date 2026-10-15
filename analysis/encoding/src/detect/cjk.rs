//! The CJK candidates of the detector, ported from chardetng 0.1.17
//! (`src/lib.rs`, Copyright Mozilla Foundation, licensed under Apache-2.0
//! or MIT; see NOTICE). chardetng feeds the encoding_rs decoders one byte
//! at a time and scores what each byte writes; the decoders here are
//! `stream`'s, which give the same results for that kind of feeding.

use super::data;
use super::stream::{Big5Stream, EucJpStream, EucKrStream, GbkStream, ShiftJisStream};
use super::*;

#[derive(PartialEq)]
enum LatinCj {
    AsciiLetter,
    Cj,
    Other,
}

// The variant names are chardetng's.
#[allow(clippy::enum_variant_names)]
#[derive(PartialEq, Copy, Clone)]
enum HalfWidthKatakana {
    DakutenForbidden,
    DakutenAllowed,
    DakutenOrHandakutenAllowed,
}

#[derive(PartialEq)]
enum LatinKorean {
    AsciiLetter,
    Hangul,
    Hanja,
    Other,
}

fn cjk_extra_score(u: u16, table: &'static [u16; 128]) -> i64 {
    if let Some(pos) = table.iter().position(|&x| x == u) { ((128 - pos) / 16) as i64 } else { 0 }
}

pub(super) struct GbkCandidate {
    decoder: GbkStream,
    prev_byte: u8,
    prev: LatinCj,
    pending_score: Option<i64>,
}

impl GbkCandidate {
    pub(super) fn new() -> Self {
        GbkCandidate { decoder: GbkStream::default(), prev_byte: 0, prev: LatinCj::Other, pending_score: None }
    }

    fn maybe_set_as_pending(&mut self, s: i64) -> i64 {
        assert!(self.pending_score.is_none());
        if self.prev == LatinCj::Cj || !more_problematic_lead(self.prev_byte) {
            s
        } else {
            self.pending_score = Some(s);
            0
        }
    }

    pub(super) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let fed = self.decoder.feed(b);
            let (dst, written) = (fed.units, fed.written);
            if written == 1 {
                let u = dst[0];
                if (u >= u16::from(b'a') && u <= u16::from(b'z')) || (u >= u16::from(b'A') && u <= u16::from(b'Z')) {
                    self.pending_score = None; // Discard pending score
                    if self.prev == LatinCj::Cj {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::AsciiLetter;
                } else if u == 0x20AC {
                    // euro sign
                    self.pending_score = None; // Discard pending score
                    // Should there even be a penalty?
                    self.prev = LatinCj::Other;
                } else if (0x4E00..=0x9FA5).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    if (0xA1..=0xFE).contains(&b) {
                        match self.prev_byte {
                            0xA1..=0xD7 => {
                                score += GBK_SCORE_PER_LEVEL_1;
                                score += cjk_extra_score(u, &data::DETECTOR_DATA.frequent_simplified);
                            }
                            0xD8..=0xFE => score += GBK_SCORE_PER_LEVEL_2,
                            _ => {
                                score += GBK_SCORE_PER_NON_EUC;
                            }
                        }
                    } else {
                        score += self.maybe_set_as_pending(GBK_SCORE_PER_NON_EUC);
                    }
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else if (0x3400..0xA000).contains(&u) || (0xF900..0xFB00).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    // XXX score?
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else if (0xE000..0xF900).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    // Treat the GB18030-required PUA mappings as non-EUC ideographs.
                    match u {
                        0xE78D..=0xE796
                        | 0xE816..=0xE818
                        | 0xE81E
                        | 0xE826
                        | 0xE82B
                        | 0xE82C
                        | 0xE831
                        | 0xE832
                        | 0xE83B
                        | 0xE843
                        | 0xE854
                        | 0xE855
                        | 0xE864 => {
                            score += GBK_SCORE_PER_NON_EUC;
                            if self.prev == LatinCj::AsciiLetter {
                                score += CJK_LATIN_ADJACENCY_PENALTY;
                            }
                            self.prev = LatinCj::Cj;
                        }
                        _ => {
                            score += GBK_PUA_PENALTY;
                            self.prev = LatinCj::Other;
                        }
                    }
                } else {
                    match u {
                        0x3000 // Distinct from Korean, space
                        | 0x3001 // Distinct from Korean, enumeration comma
                        | 0x3002 // Distinct from Korean, full stop
                        | 0xFF08 // Distinct from Korean, parenthesis
                        | 0xFF09 // Distinct from Korean, parenthesis
                        | 0xFF01 // Distinct from Japanese, exclamation
                        | 0xFF0C // Distinct from Japanese, comma
                        | 0xFF1B // Distinct from Japanese, semicolon
                        | 0xFF1F // Distinct from Japanese, question
                        => {
                            if let Some(pending) = self.pending_score {
                                score += pending;
                                self.pending_score = None;
                            }
                            score += CJ_PUNCTUATION;
                        }
                        0..=0x7F => {
                            self.pending_score = None; // Discard pending score
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
            } else if written == 2 {
                if let Some(pending) = self.pending_score {
                    score += pending;
                    self.pending_score = None;
                }
                let u = dst[0];
                if (0xDB80..=0xDBFF).contains(&u) {
                    score += GBK_PUA_PENALTY;
                    self.prev = LatinCj::Other;
                } else if (0xD480..0xD880).contains(&u) {
                    score += GBK_SCORE_PER_NON_EUC;
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                } else {
                    score += CJK_OTHER;
                    self.prev = LatinCj::Other;
                }
            }
            match fed.malformed {
                None => {}
                Some(malformed_len) => {
                    if (self.prev_byte == 0xA0 || self.prev_byte == 0xFE || self.prev_byte == 0xFD)
                        && (b < 0x80 || b == 0xFF)
                    {
                        // Mac OS Chinese Simplified single-byte that conflicts with code page GBK lead byte
                        // followed by ASCII or a non-conflicting single-byte extension.
                        self.pending_score = None; // Just in case
                        score += GBK_SINGLE_BYTE_EXTENSION_PENALTY;
                        if b.is_ascii_lowercase() || b.is_ascii_uppercase() {
                            self.prev = LatinCj::AsciiLetter;
                        } else if b == 0xFF {
                            score += GBK_SINGLE_BYTE_EXTENSION_PENALTY;
                            self.prev = LatinCj::Other;
                        } else {
                            self.prev = LatinCj::Other;
                        }
                        // The GBK decoder has the pending ASCII concept, which is
                        // a problem with this trickery, so let's reset the state.
                        self.decoder = GbkStream::default();
                    } else if malformed_len == 1 && b == 0xFF {
                        // Mac OS Chinese Simplified single-byte extension that doesn't conflict with lead bytes
                        self.pending_score = None; // Just in case
                        score += GBK_SINGLE_BYTE_EXTENSION_PENALTY;
                        self.prev = LatinCj::Other;
                        // The GBK decoder has the pending ASCII concept, which is
                        // a problem with this trickery, so let's reset the state.
                        self.decoder = GbkStream::default();
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

// Shift_JIS and Big5
fn problematic_lead(b: u8) -> bool {
    matches!(b, 0x91..=0x97 | 0x9A | 0x8A | 0x9B | 0x8B | 0x9E | 0x8E | 0xB0)
}

// GBK and EUC-KR
fn more_problematic_lead(b: u8) -> bool {
    problematic_lead(b) || b == 0x82 || b == 0x84 || b == 0x85 || b == 0xA0
}

pub(super) struct ShiftJisCandidate {
    decoder: ShiftJisStream,
    half_width_katakana_seen: bool,
    half_width_katakana_state: HalfWidthKatakana,
    prev: LatinCj,
    prev_byte: u8,
    pending_score: Option<i64>,
}

impl ShiftJisCandidate {
    pub(super) fn new() -> Self {
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

    pub(super) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
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

pub(super) struct EucJpCandidate {
    decoder: EucJpStream,
    non_ascii_seen: bool,
    half_width_katakana_state: HalfWidthKatakana,
    prev: LatinCj,
    prev_byte: u8,
    prev_prev_byte: u8,
}

impl EucJpCandidate {
    pub(super) fn new() -> Self {
        EucJpCandidate {
            decoder: EucJpStream::default(),
            non_ascii_seen: false,
            half_width_katakana_state: HalfWidthKatakana::DakutenForbidden,
            prev: LatinCj::Other,
            prev_byte: 0,
            prev_prev_byte: 0,
        }
    }

    pub(super) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
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

pub(super) struct Big5Candidate {
    decoder: Big5Stream,
    prev: LatinCj,
    prev_byte: u8,
    pending_score: Option<i64>,
}

impl Big5Candidate {
    pub(super) fn new() -> Self {
        Big5Candidate { decoder: Big5Stream::default(), prev: LatinCj::Other, prev_byte: 0, pending_score: None }
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

    pub(super) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let fed = self.decoder.feed(b);
            let (dst, written) = (fed.units, fed.written);
            if written == 1 {
                let u = dst[0];
                if (u >= u16::from(b'a') && u <= u16::from(b'z')) || (u >= u16::from(b'A') && u <= u16::from(b'Z')) {
                    self.pending_score = None; // Discard pending score
                    if self.prev == LatinCj::Cj {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::AsciiLetter;
                } else if (0x3400..0xA000).contains(&u) || (0xF900..0xFB00).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    match self.prev_byte {
                        0xA4..=0xC6 => {
                            score += self.maybe_set_as_pending(BIG5_SCORE_PER_LEVEL_1_HANZI);
                            // score += cjk_extra_score(u, &data::DETECTOR_DATA.frequent_traditional);
                        }
                        _ => {
                            score += self.maybe_set_as_pending(BIG5_SCORE_PER_OTHER_HANZI);
                        }
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
                        | 0xFF01 // Distinct from Japanese, exclamation
                        | 0xFF0C // Distinct from Japanese, comma
                        | 0xFF1B // Distinct from Japanese, semicolon
                        | 0xFF1F // Distinct from Japanese, question
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
            } else if written == 2 {
                if let Some(pending) = self.pending_score {
                    score += pending;
                    self.pending_score = None;
                }
                if dst[0] == 0xCA || dst[0] == 0xEA {
                    score += CJK_OTHER;
                    self.prev = LatinCj::Other;
                } else {
                    debug_assert!(dst[0] >= 0xD480 && dst[0] < 0xD880);
                    score += self.maybe_set_as_pending(BIG5_SCORE_PER_OTHER_HANZI);
                    if self.prev == LatinCj::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinCj::Cj;
                }
            }
            match fed.malformed {
                None => {}
                Some(malformed_len) => {
                    if self.prev_byte >= 0x81
                        && self.prev_byte <= 0xFE
                        && ((0x40..=0x7E).contains(&b) || (0xA1..=0xFE).contains(&b))
                    {
                        // The byte pair is in the Big5 range but unmapped.
                        // Treat as PUA to avoid rejecting Big5-UAO, etc.
                        // We don't reprocess `b` even if ASCII, since it's
                        // logically part of the pair.
                        if let Some(pending) = self.pending_score {
                            score += pending;
                            self.pending_score = None;
                        }
                        score += BIG5_PUA_PENALTY;
                        // Assume Hanzi semantics
                        if self.prev == LatinCj::AsciiLetter {
                            score += CJK_LATIN_ADJACENCY_PENALTY;
                        }
                        self.prev = LatinCj::Cj;
                    } else if (self.prev_byte == 0xA0 || self.prev_byte == 0xFD || self.prev_byte == 0xFE)
                        && (b < 0x80 || b == 0xFF)
                    {
                        // Mac OS Chinese Traditional single-byte that conflicts with code page Big5 lead byte
                        // followed by ASCII or a non-conflicting single-byte extension.
                        self.pending_score = None; // Just in case
                        score += BIG5_SINGLE_BYTE_EXTENSION_PENALTY;
                        if b.is_ascii_lowercase() || b.is_ascii_uppercase() {
                            self.prev = LatinCj::AsciiLetter;
                        } else if b == 0xFF {
                            score += BIG5_SINGLE_BYTE_EXTENSION_PENALTY;
                            self.prev = LatinCj::Other;
                        } else {
                            self.prev = LatinCj::Other;
                        }
                    } else if malformed_len == 1 && b == 0xFF {
                        // Mac OS Chinese Traditional single-byte extension that doesn't conflict with lead bytes
                        self.pending_score = None; // Just in case
                        score += BIG5_SINGLE_BYTE_EXTENSION_PENALTY;
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

pub(super) struct EucKrCandidate {
    decoder: EucKrStream,
    prev_byte: u8,
    prev_was_euc_range: bool,
    prev: LatinKorean,
    current_word_len: u64,
    pending_score: Option<i64>,
}

impl EucKrCandidate {
    pub(super) fn new() -> Self {
        EucKrCandidate {
            decoder: EucKrStream::default(),
            prev_byte: 0,
            prev_was_euc_range: false,
            prev: LatinKorean::Other,
            current_word_len: 0,
            pending_score: None,
        }
    }

    fn maybe_set_as_pending(&mut self, s: i64) -> i64 {
        assert!(self.pending_score.is_none());
        if self.prev == LatinKorean::Hangul || !more_problematic_lead(self.prev_byte) {
            s
        } else {
            self.pending_score = Some(s);
            0
        }
    }

    pub(super) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        let mut score = 0i64;
        for &b in buffer {
            let in_euc_range = (0xA1..=0xFE).contains(&b);
            let fed = self.decoder.feed(b);
            let (dst, written) = (fed.units, fed.written);
            if written > 0 {
                let u = dst[0];
                if (u >= u16::from(b'a') && u <= u16::from(b'z')) || (u >= u16::from(b'A') && u <= u16::from(b'Z')) {
                    self.pending_score = None; // Discard pending score
                    match self.prev {
                        LatinKorean::Hangul | LatinKorean::Hanja => {
                            score += CJK_LATIN_ADJACENCY_PENALTY;
                        }
                        _ => {}
                    }
                    self.prev = LatinKorean::AsciiLetter;
                    self.current_word_len = 0;
                } else if (0xAC00..=0xD7A3).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    if self.prev_was_euc_range && in_euc_range {
                        score += EUC_KR_SCORE_PER_EUC_HANGUL;
                        score += cjk_extra_score(u, &data::DETECTOR_DATA.frequent_hangul);
                    } else {
                        score += self.maybe_set_as_pending(EUC_KR_SCORE_PER_NON_EUC_HANGUL);
                    }
                    if self.prev == LatinKorean::AsciiLetter {
                        score += CJK_LATIN_ADJACENCY_PENALTY;
                    }
                    self.prev = LatinKorean::Hangul;
                    self.current_word_len += 1;
                    if self.current_word_len > 5 {
                        score += EUC_KR_LONG_WORD_PENALTY;
                    }
                } else if (0x4E00..0xAC00).contains(&u) || (0xF900..=0xFA0B).contains(&u) {
                    if let Some(pending) = self.pending_score {
                        score += pending;
                        self.pending_score = None;
                    }
                    score += EUC_KR_SCORE_PER_HANJA;
                    match self.prev {
                        LatinKorean::AsciiLetter => {
                            score += CJK_LATIN_ADJACENCY_PENALTY;
                        }
                        LatinKorean::Hangul => {
                            score += EUC_KR_HANJA_AFTER_HANGUL_PENALTY;
                        }
                        _ => {}
                    }
                    self.prev = LatinKorean::Hanja;
                    self.current_word_len += 1;
                    if self.current_word_len > 5 {
                        score += EUC_KR_LONG_WORD_PENALTY;
                    }
                } else {
                    if u >= 0x80 {
                        if let Some(pending) = self.pending_score {
                            score += pending;
                            self.pending_score = None;
                        }
                        score += CJK_OTHER;
                    } else {
                        self.pending_score = None; // Discard pending score
                    }
                    self.prev = LatinKorean::Other;
                    self.current_word_len = 0;
                }
            }
            match fed.malformed {
                None => {}
                Some(malformed_len) => {
                    if (self.prev_byte == 0xC9 || self.prev_byte == 0xFE) && (0xA1..=0xFE).contains(&b) {
                        if let Some(pending) = self.pending_score {
                            score += pending;
                            self.pending_score = None;
                        }
                        // The byte pair is in code page 949 EUDC range
                        score += EUC_KR_PUA_PENALTY;
                        // Assume Hanja semantics
                        match self.prev {
                            LatinKorean::AsciiLetter => {
                                score += CJK_LATIN_ADJACENCY_PENALTY;
                            }
                            LatinKorean::Hangul => {
                                score += EUC_KR_HANJA_AFTER_HANGUL_PENALTY;
                            }
                            _ => {}
                        }
                        self.prev = LatinKorean::Hanja;
                        self.current_word_len += 1;
                        if self.current_word_len > 5 {
                            score += EUC_KR_LONG_WORD_PENALTY;
                        }
                    } else if (self.prev_byte == 0xA1
                        || (self.prev_byte >= 0xA3 && self.prev_byte <= 0xA8)
                        || (self.prev_byte >= 0xAA && self.prev_byte <= 0xAD))
                        && (0x7B..=0x7D).contains(&b)
                    {
                        if let Some(pending) = self.pending_score {
                            score += pending;
                            self.pending_score = None;
                        }
                        // MacKorean symbols in range not part of code page 949
                        score += EUC_KR_MAC_KOREAN_PENALTY;
                        self.prev = LatinKorean::Other;
                        self.current_word_len = 0;
                    } else if (self.prev_byte >= 0x81 && self.prev_byte <= 0x84) && (b <= 0x80 || b == 0xFF) {
                        // MacKorean single-byte that conflicts with code page 949 lead byte
                        // followed by ASCII or a non-conflicting single-byte extension.
                        self.pending_score = None; // Just in case
                        score += EUC_KR_SINGLE_BYTE_EXTENSION_PENALTY;
                        if b.is_ascii_lowercase() || b.is_ascii_uppercase() {
                            self.prev = LatinKorean::AsciiLetter;
                        } else if b == 0x80 || b == 0xFF {
                            score += EUC_KR_SINGLE_BYTE_EXTENSION_PENALTY;
                            self.prev = LatinKorean::Other;
                        } else {
                            self.prev = LatinKorean::Other;
                        }
                        self.current_word_len = 0;
                    } else if malformed_len == 1 && (b == 0x80 || b == 0xFF) {
                        // MacKorean single-byte extensions that don't conflict with lead bytes
                        self.pending_score = None; // Just in case
                        score += EUC_KR_SINGLE_BYTE_EXTENSION_PENALTY;
                        self.prev = LatinKorean::Other;
                        self.current_word_len = 0;
                    } else {
                        return None;
                    }
                }
            }
            self.prev_was_euc_range = in_euc_range;
            self.prev_byte = b;
        }
        if last && !self.decoder.finish() {
            return None;
        }
        Some(score)
    }
}
