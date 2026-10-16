//! Guessing the encoding of bytes that carry no label: a port of
//! chardetng 0.1.17, the detector Firefox uses for unlabelled legacy
//! pages (Copyright Mozilla Foundation, Apache-2.0 or MIT; its statistics
//! in `data` are CC0; see NOTICE).
//!
//! The detector feeds the bytes to 27 candidates at once: UTF-8,
//! ISO-2022-JP, the five CJK multi-byte encodings and twenty single-byte
//! ones. A candidate that sees a byte its encoding cannot have drops out;
//! the others add up scores for plausible and implausible pairs of
//! characters. The guess is the best score, with two rules ahead of it:
//! valid UTF-8 wins over every legacy encoding, and ASCII with ISO-2022-JP
//! escapes is ISO-2022-JP.
//!
//! What differs from chardetng:
//!
//! - there is no top-level domain hint, so every guess is the one
//!   chardetng makes for a generic domain;
//! - the CJK candidates decode with `stream`'s decoders instead of
//!   encoding_rs, giving the same result per byte;
//! - [`Detection`] adds a confidence derived from the score margin.
//!
//! The scores are deterministic: the same bytes give the same guess
//! however they are split across calls to [`EncodingDetector::feed`].

mod cjk;
mod data;
mod single_byte;
mod stream;

use turbo_encoding_layers::Encoding;

use crate::decode::{Iso2022Jp, Step, Stepper};
use cjk::{Big5Candidate, EucJpCandidate, EucKrCandidate, GbkCandidate, ShiftJisCandidate};
use data::*;
use single_byte::{
    ArabicFrenchCandidate, CaselessCandidate, LatinCandidate, LogicalCandidate, NonLatinCasedCandidate, VisualCandidate,
};
use stream::Utf8Stream;

// The scores below are chardetng's, with its comments.

const LATIN_ADJACENCY_PENALTY: i64 = -50;

const IMPLAUSIBILITY_PENALTY: i64 = -220;

const ORDINAL_BONUS: i64 = 300;

/// Must match the ISO-8859-2 score for " Š ". Note: There
/// are four Slovenian Wikipedia list page titles where the
/// list is split by letter so that Š stands alone for the
/// list part for Š. Let's assume that's a special case not
/// worth detecting even though the copyright sign detection
/// makes Slovenian title detection round to one percentage
/// point worse.
const COPYRIGHT_BONUS: i64 = 222;

const IMPLAUSIBLE_LATIN_CASE_TRANSITION_PENALTY: i64 = -180;

const NON_LATIN_CAPITALIZATION_BONUS: i64 = 40;

const NON_LATIN_ALL_CAPS_PENALTY: i64 = -40;

const NON_LATIN_MIXED_CASE_PENALTY: i64 = -20;

// Manually calibrated relative to windows-1256 Arabic
const CJK_BASE_SCORE: i64 = 41;

const CJK_SECONDARY_BASE_SCORE: i64 = 20;

const SHIFT_JIS_SCORE_PER_KANA: i64 = 20;

const SHIFT_JIS_SCORE_PER_LEVEL_1_KANJI: i64 = CJK_BASE_SCORE;

const SHIFT_JIS_SCORE_PER_LEVEL_2_KANJI: i64 = CJK_SECONDARY_BASE_SCORE;

// Manually calibrated relative to windows-1256 Persian and Urdu
const SHIFT_JIS_INITIAL_HALF_WIDTH_KATAKANA_PENALTY: i64 = -75;

const HALF_WIDTH_KATAKANA_SCORE: i64 = 1;

// Unclear if this is a good idea; seems not harmful, but can't be sure.
const HALF_WIDTH_KATAKANA_VOICING_SCORE: i64 = 10;

const SHIFT_JIS_PUA_PENALTY: i64 = -(CJK_BASE_SCORE * 10);

const SHIFT_JIS_EXTENSION_PENALTY: i64 = SHIFT_JIS_PUA_PENALTY * 2;

const SHIFT_JIS_SINGLE_BYTE_EXTENSION_PENALTY: i64 = SHIFT_JIS_EXTENSION_PENALTY;

const EUC_JP_SCORE_PER_KANA: i64 = CJK_BASE_SCORE + (CJK_BASE_SCORE / 3); // Relative to Big5

const EUC_JP_SCORE_PER_NEAR_OBSOLETE_KANA: i64 = CJK_BASE_SCORE - 1;

const EUC_JP_SCORE_PER_LEVEL_1_KANJI: i64 = CJK_BASE_SCORE;

const EUC_JP_SCORE_PER_LEVEL_2_KANJI: i64 = CJK_SECONDARY_BASE_SCORE;

const EUC_JP_SCORE_PER_OTHER_KANJI: i64 = CJK_SECONDARY_BASE_SCORE / 4;

const EUC_JP_INITIAL_KANA_PENALTY: i64 = -((CJK_BASE_SCORE / 3) + 1);

// Needs to be more severe than for Shift_JIS to avoid misdetecting EUC-KR!
const EUC_JP_EXTENSION_PENALTY: i64 = -(CJK_BASE_SCORE * 50);

const BIG5_SCORE_PER_LEVEL_1_HANZI: i64 = CJK_BASE_SCORE;

const BIG5_SCORE_PER_OTHER_HANZI: i64 = CJK_SECONDARY_BASE_SCORE;

// More severe than other PUA penalties to avoid misdetecting EUC-KR! (25 as
// the multiplier is too little)
const BIG5_PUA_PENALTY: i64 = -(CJK_BASE_SCORE * 30);

const BIG5_SINGLE_BYTE_EXTENSION_PENALTY: i64 = -(CJK_BASE_SCORE * 40);

const EUC_KR_SCORE_PER_EUC_HANGUL: i64 = CJK_BASE_SCORE + 1;

const EUC_KR_SCORE_PER_NON_EUC_HANGUL: i64 = CJK_SECONDARY_BASE_SCORE / 5;

const EUC_KR_SCORE_PER_HANJA: i64 = CJK_SECONDARY_BASE_SCORE / 2;

const EUC_KR_HANJA_AFTER_HANGUL_PENALTY: i64 = -(CJK_BASE_SCORE * 10);

const EUC_KR_LONG_WORD_PENALTY: i64 = -6;

const EUC_KR_PUA_PENALTY: i64 = GBK_PUA_PENALTY - 1; // Break tie in favor of GBK

const EUC_KR_MAC_KOREAN_PENALTY: i64 = EUC_KR_PUA_PENALTY * 2;

const EUC_KR_SINGLE_BYTE_EXTENSION_PENALTY: i64 = EUC_KR_MAC_KOREAN_PENALTY;

const GBK_SCORE_PER_LEVEL_1: i64 = CJK_BASE_SCORE;

const GBK_SCORE_PER_LEVEL_2: i64 = CJK_SECONDARY_BASE_SCORE;

const GBK_SCORE_PER_NON_EUC: i64 = CJK_SECONDARY_BASE_SCORE / 4;

const GBK_PUA_PENALTY: i64 = -(CJK_BASE_SCORE * 10);

const GBK_SINGLE_BYTE_EXTENSION_PENALTY: i64 = GBK_PUA_PENALTY * 4;

const CJK_LATIN_ADJACENCY_PENALTY: i64 = -CJK_BASE_SCORE; // smaller penalty than LATIN_ADJACENCY_PENALTY

const CJ_PUNCTUATION: i64 = CJK_BASE_SCORE / 2;

const CJK_OTHER: i64 = CJK_SECONDARY_BASE_SCORE / 4;

/// Latin letter caseless class
const LATIN_LETTER: u8 = 1;

/// The ISO-2022-JP candidate: valid or not, no score.
#[derive(Default)]
struct Iso2022Candidate {
    decoder: Iso2022Jp,
}

impl Iso2022Candidate {
    fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        for &b in buffer {
            if let Step::Error { .. } = self.decoder.step(b) {
                return None;
            }
        }
        if last && self.decoder.eof().is_some() {
            return None;
        }
        Some(0)
    }
}

enum InnerCandidate {
    Latin(LatinCandidate),
    NonLatinCased(NonLatinCasedCandidate),
    Caseless(CaselessCandidate),
    ArabicFrench(ArabicFrenchCandidate),
    Logical(LogicalCandidate),
    Visual(VisualCandidate),
    Utf8(Utf8Stream),
    Iso2022(Iso2022Candidate),
    Shift(ShiftJisCandidate),
    EucJp(EucJpCandidate),
    EucKr(EucKrCandidate),
    Big5(Big5Candidate),
    Gbk(GbkCandidate),
}

impl InnerCandidate {
    fn feed(&mut self, buffer: &[u8], last: bool) -> Option<i64> {
        // The single-byte candidates treat the end of the input as a space.
        fn with_end(last: bool, mut feed: impl FnMut(&[u8]) -> Option<i64>, buffer: &[u8]) -> Option<i64> {
            let score = feed(buffer)?;
            if last { Some(score + feed(b" ")?) } else { Some(score) }
        }
        match self {
            InnerCandidate::Latin(c) => with_end(last, |b| c.feed(b), buffer),
            InnerCandidate::NonLatinCased(c) => with_end(last, |b| c.feed(b), buffer),
            InnerCandidate::Caseless(c) => with_end(last, |b| c.feed(b), buffer),
            InnerCandidate::ArabicFrench(c) => with_end(last, |b| c.feed(b), buffer),
            InnerCandidate::Logical(c) => with_end(last, |b| c.feed(b), buffer),
            InnerCandidate::Visual(c) => with_end(last, |b| c.feed(b), buffer),
            InnerCandidate::Utf8(c) => c.feed(buffer, last).then_some(0),
            InnerCandidate::Iso2022(c) => c.feed(buffer, last),
            InnerCandidate::Shift(c) => c.feed(buffer, last),
            InnerCandidate::EucJp(c) => c.feed(buffer, last),
            InnerCandidate::EucKr(c) => c.feed(buffer, last),
            InnerCandidate::Big5(c) => c.feed(buffer, last),
            InnerCandidate::Gbk(c) => c.feed(buffer, last),
        }
    }
}

struct Candidate {
    inner: InnerCandidate,
    /// None once the candidate has dropped out.
    score: Option<i64>,
}

impl Candidate {
    fn new(inner: InnerCandidate) -> Candidate {
        Candidate { inner, score: Some(0) }
    }

    fn feed(&mut self, buffer: &[u8], last: bool) {
        if let Some(old_score) = self.score {
            self.score = self.inner.feed(buffer, last).map(|new_score| old_score + new_score);
        }
    }

    /// The score that competes, with chardetng's rule for a generic
    /// domain: a candidate for a script written in words must have seen a
    /// word of two letters or more.
    fn score(&self) -> Option<i64> {
        let longest_word = match &self.inner {
            InnerCandidate::NonLatinCased(c) => Some(c.longest_word),
            InnerCandidate::Caseless(c) => Some(c.longest_word),
            InnerCandidate::ArabicFrench(c) => Some(c.longest_word),
            InnerCandidate::Logical(c) => Some(c.longest_word),
            InnerCandidate::Visual(c) => Some(c.longest_word),
            _ => None,
        };
        if longest_word.is_some_and(|w| w < 2) {
            return None;
        }
        self.score
    }

    fn plausible_punctuation(&self) -> u64 {
        match &self.inner {
            InnerCandidate::Logical(c) => c.plausible_punctuation,
            InnerCandidate::Visual(c) => c.plausible_punctuation,
            _ => unreachable!("only the Hebrew candidates count punctuation"),
        }
    }

    fn encoding(&self) -> Encoding {
        match &self.inner {
            InnerCandidate::Latin(c) => c.data.encoding,
            InnerCandidate::NonLatinCased(c) => c.data.encoding,
            InnerCandidate::Caseless(c) => c.data.encoding,
            InnerCandidate::ArabicFrench(c) => c.data.encoding,
            InnerCandidate::Logical(c) => c.data.encoding,
            InnerCandidate::Visual(c) => c.data.encoding,
            InnerCandidate::Shift(_) => Encoding::ShiftJis,
            InnerCandidate::EucJp(_) => Encoding::EucJp,
            InnerCandidate::Big5(_) => Encoding::Big5,
            InnerCandidate::EucKr(_) => Encoding::EucKr,
            InnerCandidate::Gbk(_) => Encoding::Gbk,
            InnerCandidate::Utf8(_) => Encoding::Utf8,
            InnerCandidate::Iso2022(_) => Encoding::Iso2022Jp,
        }
    }
}

/// Up to two ASCII bytes seen before the first non-ASCII byte, which the
/// candidates need as context (Spanish "n.º").
#[derive(Clone, Copy, Default)]
struct BeforeNonAscii {
    bytes: [u8; 2],
    len: usize,
}

impl BeforeNonAscii {
    fn as_slice(&self) -> &[u8] {
        &self.bytes[2 - self.len..]
    }

    fn push(&mut self, buffer: &[u8]) {
        for &b in buffer.iter().rev().take(2).rev() {
            self.bytes = [self.bytes[1], b];
            self.len = (self.len + 1).min(2);
        }
    }
}

/// A guess and how sure it is.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Detection {
    pub encoding: Encoding,
    /// In [0, 1]. 1 for valid UTF-8 (ASCII included) when UTF-8 is
    /// allowed, and for ASCII with valid ISO-2022-JP escapes; 0 for the
    /// windows-1252 a valid UTF-8 input gets when it is not. Otherwise a function of how far
    /// the guess's score is ahead of the best score of another encoding,
    /// `margin / (margin + 200)`, so 0 when no candidate scored above
    /// zero (the guess is then the windows-1252 default). It is a
    /// heuristic for ranking guesses, not a calibrated probability.
    pub confidence: f32,
    /// Whether the input had any non-ASCII byte.
    pub non_ascii: bool,
}

/// The margin at which confidence reaches one half.
const HALF_CONFIDENCE_MARGIN: f32 = 200.0;

/// The detector. Feed it the bytes, in as many calls as convenient, the
/// last with `last` set, then ask for the guess.
pub struct EncodingDetector {
    candidates: [Candidate; 27],
    non_ascii_seen: u64,
    last_before_non_ascii: BeforeNonAscii,
    esc_seen: bool,
    closed: bool,
}

impl Default for EncodingDetector {
    fn default() -> Self {
        EncodingDetector::new()
    }
}

impl EncodingDetector {
    const FIRST_NORMAL: usize = 3;
    const UTF_8_INDEX: usize = 0;
    const ISO_2022_JP_INDEX: usize = 1;
    const VISUAL_INDEX: usize = 2;
    const LOGICAL_INDEX: usize = 16;

    pub fn new() -> EncodingDetector {
        use InnerCandidate as I;
        let sb = |i: usize| &SINGLE_BYTE_DATA[i];
        EncodingDetector {
            candidates: [
                Candidate::new(I::Utf8(Utf8Stream::default())),                        // 0
                Candidate::new(I::Iso2022(Iso2022Candidate::default())),               // 1
                Candidate::new(I::Visual(VisualCandidate::new(sb(ISO_8859_8_INDEX)))), // 2
                Candidate::new(I::Gbk(GbkCandidate::new())),                           // 3
                Candidate::new(I::EucJp(EucJpCandidate::new())),                       // 4
                Candidate::new(I::EucKr(EucKrCandidate::new())),                       // 5
                Candidate::new(I::Shift(ShiftJisCandidate::new())),                    // 6
                Candidate::new(I::Big5(Big5Candidate::new())),                         // 7
                Candidate::new(I::Latin(LatinCandidate::new(sb(WINDOWS_1252_INDEX)))), // 8
                Candidate::new(I::NonLatinCased(NonLatinCasedCandidate::new(sb(WINDOWS_1251_INDEX)))), // 9
                Candidate::new(I::Latin(LatinCandidate::new(sb(WINDOWS_1250_INDEX)))), // 10
                Candidate::new(I::Latin(LatinCandidate::new(sb(ISO_8859_2_INDEX)))),   // 11
                Candidate::new(I::ArabicFrench(ArabicFrenchCandidate::new(sb(WINDOWS_1256_INDEX)))), // 12
                Candidate::new(I::Latin(LatinCandidate::new(sb(WINDOWS_1252_ICELANDIC_INDEX)))), // 13
                Candidate::new(I::Latin(LatinCandidate::new(sb(WINDOWS_1254_INDEX)))), // 14
                Candidate::new(I::Caseless(CaselessCandidate::new(sb(WINDOWS_874_INDEX)))), // 15
                Candidate::new(I::Logical(LogicalCandidate::new(sb(WINDOWS_1255_INDEX)))), // 16
                Candidate::new(I::NonLatinCased(NonLatinCasedCandidate::new(sb(WINDOWS_1253_INDEX)))), // 17
                Candidate::new(I::NonLatinCased(NonLatinCasedCandidate::new(sb(ISO_8859_7_INDEX)))), // 18
                Candidate::new(I::Latin(LatinCandidate::new(sb(WINDOWS_1257_INDEX)))), // 19
                Candidate::new(I::Latin(LatinCandidate::new(sb(ISO_8859_13_INDEX)))),  // 20
                Candidate::new(I::NonLatinCased(NonLatinCasedCandidate::new(sb(KOI8_U_INDEX)))), // 21
                Candidate::new(I::NonLatinCased(NonLatinCasedCandidate::new(sb(IBM866_INDEX)))), // 22
                Candidate::new(I::Caseless(CaselessCandidate::new(sb(ISO_8859_6_INDEX)))), // 23
                Candidate::new(I::Latin(LatinCandidate::new(sb(WINDOWS_1258_INDEX)))), // 24
                Candidate::new(I::Latin(LatinCandidate::new(sb(ISO_8859_4_INDEX)))),   // 25
                Candidate::new(I::NonLatinCased(NonLatinCasedCandidate::new(sb(ISO_8859_5_INDEX)))), // 26
            ],
            non_ascii_seen: 0,
            last_before_non_ascii: BeforeNonAscii::default(),
            esc_seen: false,
            closed: false,
        }
    }

    fn feed_all(&mut self, buffer: &[u8], last: bool) {
        for candidate in self.candidates.iter_mut() {
            candidate.feed(buffer, last);
        }
        self.non_ascii_seen += buffer.iter().filter(|b| !b.is_ascii()).count() as u64;
    }

    /// Takes the next part of the input. `last` marks the end of it; no
    /// call may follow one with `last` set. Returns whether any non-ASCII
    /// byte has been seen.
    ///
    /// # Panics
    ///
    /// If called again after a call with `last` set.
    pub fn feed(&mut self, buffer: &[u8], last: bool) -> bool {
        assert!(!self.closed, "the detector was already fed its last bytes");
        self.closed = last;
        // Candidates see nothing until the first non-ASCII byte or escape,
        // and then the two bytes before it.
        let start = if self.non_ascii_seen == 0 && !self.esc_seen {
            let up_to = buffer.iter().position(|b| !b.is_ascii()).unwrap_or(buffer.len());
            let start = match buffer[..up_to].iter().position(|&b| b == 0x1B) {
                Some(escape) => {
                    self.esc_seen = true;
                    escape
                }
                None => up_to,
            };
            if start == buffer.len() {
                self.last_before_non_ascii.push(buffer);
                return self.non_ascii_seen != 0;
            }
            if start < 2 {
                let before = std::mem::take(&mut self.last_before_non_ascii);
                self.feed_all(before.as_slice(), false);
                0
            } else {
                start - 2
            }
        } else {
            0
        };
        self.feed_all(&buffer[start..], last);
        self.non_ascii_seen != 0
    }

    /// The guess. With `allow_utf8` false, valid UTF-8 guesses
    /// windows-1252, as browsers must for unlabelled pages.
    pub fn guess(&self, allow_utf8: bool) -> Encoding {
        self.rank(allow_utf8).0
    }

    /// The guess with a confidence.
    pub fn detection(&self, allow_utf8: bool) -> Detection {
        let non_ascii = self.non_ascii_seen != 0;
        let (encoding, top, runner_up) = self.rank(allow_utf8);
        let confidence = match top {
            // Decided by validity, not by score.
            None => 1.0,
            Some(top) => {
                let margin = (top - runner_up.unwrap_or(0).max(0)).max(0) as f32;
                margin / (margin + HALF_CONFIDENCE_MARGIN)
            }
        };
        Detection { encoding, confidence, non_ascii }
    }

    /// The competing score of the first candidate for `encoding` (the
    /// windows-1252 one for windows-1252, which has a second candidate
    /// with Icelandic statistics), None if it dropped out or never had a
    /// word long enough to count. Some(0) for an encoding with no
    /// candidate.
    pub fn score(&self, encoding: Encoding) -> Option<i64> {
        self.candidates.iter().find(|c| c.encoding() == encoding).map_or(Some(0), Candidate::score)
    }

    /// The guess, its score (None when a validity rule decided) and the
    /// best score among candidates for other encodings.
    fn rank(&self, allow_utf8: bool) -> (Encoding, Option<i64>, Option<i64>) {
        if self.non_ascii_seen == 0 && self.esc_seen && self.candidates[Self::ISO_2022_JP_INDEX].score.is_some() {
            return (Encoding::Iso2022Jp, None, None);
        }
        if self.candidates[Self::UTF_8_INDEX].score.is_some() {
            // Without UTF-8 allowed, windows-1252 is a policy, not a
            // finding: it gets no confidence.
            return if allow_utf8 { (Encoding::Utf8, None, None) } else { (Encoding::Windows1252, Some(0), None) };
        }
        let mut encoding = Encoding::Windows1252;
        let mut max = 0i64;
        let mut best = None;
        for (i, candidate) in self.candidates.iter().enumerate().skip(Self::FIRST_NORMAL) {
            if let Some(score) = candidate.score()
                && score > max
            {
                max = score;
                encoding = candidate.encoding();
                best = Some(i);
            }
        }
        let mut top = max;
        let visual = &self.candidates[Self::VISUAL_INDEX];
        if let Some(visual_score) = visual.score()
            && (visual_score > max || encoding == Encoding::Windows1255)
            && visual.plausible_punctuation() > self.candidates[Self::LOGICAL_INDEX].plausible_punctuation()
        {
            encoding = Encoding::Iso8859_8;
            best = Some(Self::VISUAL_INDEX);
            top = visual_score;
        }
        let runner_up = self
            .candidates
            .iter()
            .enumerate()
            .skip(Self::VISUAL_INDEX)
            .filter(|&(i, c)| Some(i) != best && c.encoding() != encoding)
            .filter_map(|(_, c)| c.score())
            .max();
        (encoding, Some(top), runner_up)
    }
}

/// Detects the encoding of `bytes` in one call, UTF-8 allowed.
pub fn detect(bytes: &[u8]) -> Detection {
    let mut detector = EncodingDetector::new();
    detector.feed(bytes, true);
    detector.detection(true)
}
