//! GBK and Big5 CJK candidates, ported from chardetng 0.1.17 (`src/lib.rs`,
//! Copyright Mozilla Foundation, licensed under Apache-2.0 or MIT; see NOTICE).

use super::{LatinCj, cjk_extra_score, more_problematic_lead, problematic_lead};
use crate::detect::data;
use crate::detect::scores::*;
use crate::detect::stream::{Big5Stream, GbkStream};

pub(in crate::detect) struct GbkCandidate {
    decoder: GbkStream,
    prev_byte: u8,
    prev: LatinCj,
    pending_score: Option<i64>,
}

impl GbkCandidate {
    pub(in crate::detect) fn new() -> Self {
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

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
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

pub(in crate::detect) struct Big5Candidate {
    decoder: Big5Stream,
    prev: LatinCj,
    prev_byte: u8,
    pending_score: Option<i64>,
}

impl Big5Candidate {
    pub(in crate::detect) fn new() -> Self {
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

    pub(in crate::detect) fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
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
