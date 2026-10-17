//! The CJK candidates of the detector, ported from chardetng 0.1.17
//! (`src/lib.rs`, Copyright Mozilla Foundation, licensed under Apache-2.0
//! or MIT; see NOTICE). chardetng feeds the encoding_rs decoders one byte
//! at a time and scores what each byte writes; the decoders here are
//! `stream`'s, which give the same results for that kind of feeding.

mod chinese;
mod japanese;

pub(super) use chinese::{Big5Candidate, GbkCandidate};
pub(super) use japanese::{EucJpCandidate, ShiftJisCandidate};

use super::data;
use super::scores::*;
use super::stream::EucKrStream;

#[derive(PartialEq)]
pub(super) enum LatinCj {
    AsciiLetter,
    Cj,
    Other,
}

// The variant names are chardetng's.
#[allow(clippy::enum_variant_names)]
#[derive(PartialEq, Copy, Clone)]
pub(super) enum HalfWidthKatakana {
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

pub(super) fn cjk_extra_score(u: u16, table: &'static [u16; 128]) -> i64 {
    if let Some(pos) = table.iter().position(|&x| x == u) { ((128 - pos) / 16) as i64 } else { 0 }
}

// Shift_JIS and Big5
pub(super) fn problematic_lead(b: u8) -> bool {
    matches!(b, 0x91..=0x97 | 0x9A | 0x8A | 0x9B | 0x8B | 0x9E | 0x8E | 0xB0)
}

// GBK and EUC-KR
pub(super) fn more_problematic_lead(b: u8) -> bool {
    problematic_lead(b) || b == 0x82 || b == 0x84 || b == 0x85 || b == 0xA0
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
