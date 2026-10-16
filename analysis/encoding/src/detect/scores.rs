//! Calibration score constants for the encoding detector, ported from
//! chardetng 0.1.17 (`src/lib.rs`, Copyright Mozilla Foundation, licensed
//! under Apache-2.0 or MIT; see NOTICE).

pub(super) const LATIN_ADJACENCY_PENALTY: i64 = -50;

pub(super) const IMPLAUSIBILITY_PENALTY: i64 = -220;

pub(super) const ORDINAL_BONUS: i64 = 300;

/// Must match the ISO-8859-2 score for " Š ". Note: There
/// are four Slovenian Wikipedia list page titles where the
/// list is split by letter so that Š stands alone for the
/// list part for Š. Let's assume that's a special case not
/// worth detecting even though the copyright sign detection
/// makes Slovenian title detection round to one percentage
/// point worse.
pub(super) const COPYRIGHT_BONUS: i64 = 222;

pub(super) const IMPLAUSIBLE_LATIN_CASE_TRANSITION_PENALTY: i64 = -180;

pub(super) const NON_LATIN_CAPITALIZATION_BONUS: i64 = 40;

pub(super) const NON_LATIN_ALL_CAPS_PENALTY: i64 = -40;

pub(super) const NON_LATIN_MIXED_CASE_PENALTY: i64 = -20;

// Manually calibrated relative to windows-1256 Arabic
pub(super) const CJK_BASE_SCORE: i64 = 41;

pub(super) const CJK_SECONDARY_BASE_SCORE: i64 = 20;

pub(super) const SHIFT_JIS_SCORE_PER_KANA: i64 = 20;

pub(super) const SHIFT_JIS_SCORE_PER_LEVEL_1_KANJI: i64 = CJK_BASE_SCORE;

pub(super) const SHIFT_JIS_SCORE_PER_LEVEL_2_KANJI: i64 = CJK_SECONDARY_BASE_SCORE;

// Manually calibrated relative to windows-1256 Persian and Urdu
pub(super) const SHIFT_JIS_INITIAL_HALF_WIDTH_KATAKANA_PENALTY: i64 = -75;

pub(super) const HALF_WIDTH_KATAKANA_SCORE: i64 = 1;

// Unclear if this is a good idea; seems not harmful, but can't be sure.
pub(super) const HALF_WIDTH_KATAKANA_VOICING_SCORE: i64 = 10;

pub(super) const SHIFT_JIS_PUA_PENALTY: i64 = -(CJK_BASE_SCORE * 10);

pub(super) const SHIFT_JIS_EXTENSION_PENALTY: i64 = SHIFT_JIS_PUA_PENALTY * 2;

pub(super) const SHIFT_JIS_SINGLE_BYTE_EXTENSION_PENALTY: i64 = SHIFT_JIS_EXTENSION_PENALTY;

pub(super) const EUC_JP_SCORE_PER_KANA: i64 = CJK_BASE_SCORE + (CJK_BASE_SCORE / 3); // Relative to Big5

pub(super) const EUC_JP_SCORE_PER_NEAR_OBSOLETE_KANA: i64 = CJK_BASE_SCORE - 1;

pub(super) const EUC_JP_SCORE_PER_LEVEL_1_KANJI: i64 = CJK_BASE_SCORE;

pub(super) const EUC_JP_SCORE_PER_LEVEL_2_KANJI: i64 = CJK_SECONDARY_BASE_SCORE;

pub(super) const EUC_JP_SCORE_PER_OTHER_KANJI: i64 = CJK_SECONDARY_BASE_SCORE / 4;

pub(super) const EUC_JP_INITIAL_KANA_PENALTY: i64 = -((CJK_BASE_SCORE / 3) + 1);

// Needs to be more severe than for Shift_JIS to avoid misdetecting EUC-KR!
pub(super) const EUC_JP_EXTENSION_PENALTY: i64 = -(CJK_BASE_SCORE * 50);

pub(super) const BIG5_SCORE_PER_LEVEL_1_HANZI: i64 = CJK_BASE_SCORE;

pub(super) const BIG5_SCORE_PER_OTHER_HANZI: i64 = CJK_SECONDARY_BASE_SCORE;

// More severe than other PUA penalties to avoid misdetecting EUC-KR! (25 as
// the multiplier is too little)
pub(super) const BIG5_PUA_PENALTY: i64 = -(CJK_BASE_SCORE * 30);

pub(super) const BIG5_SINGLE_BYTE_EXTENSION_PENALTY: i64 = -(CJK_BASE_SCORE * 40);

pub(super) const EUC_KR_SCORE_PER_EUC_HANGUL: i64 = CJK_BASE_SCORE + 1;

pub(super) const EUC_KR_SCORE_PER_NON_EUC_HANGUL: i64 = CJK_SECONDARY_BASE_SCORE / 5;

pub(super) const EUC_KR_SCORE_PER_HANJA: i64 = CJK_SECONDARY_BASE_SCORE / 2;

pub(super) const EUC_KR_HANJA_AFTER_HANGUL_PENALTY: i64 = -(CJK_BASE_SCORE * 10);

pub(super) const EUC_KR_LONG_WORD_PENALTY: i64 = -6;

pub(super) const EUC_KR_PUA_PENALTY: i64 = GBK_PUA_PENALTY - 1; // Break tie in favor of GBK

pub(super) const EUC_KR_MAC_KOREAN_PENALTY: i64 = EUC_KR_PUA_PENALTY * 2;

pub(super) const EUC_KR_SINGLE_BYTE_EXTENSION_PENALTY: i64 = EUC_KR_MAC_KOREAN_PENALTY;

pub(super) const GBK_SCORE_PER_LEVEL_1: i64 = CJK_BASE_SCORE;

pub(super) const GBK_SCORE_PER_LEVEL_2: i64 = CJK_SECONDARY_BASE_SCORE;

pub(super) const GBK_SCORE_PER_NON_EUC: i64 = CJK_SECONDARY_BASE_SCORE / 4;

pub(super) const GBK_PUA_PENALTY: i64 = -(CJK_BASE_SCORE * 10);

pub(super) const GBK_SINGLE_BYTE_EXTENSION_PENALTY: i64 = GBK_PUA_PENALTY * 4;

pub(super) const CJK_LATIN_ADJACENCY_PENALTY: i64 = -CJK_BASE_SCORE; // smaller penalty than LATIN_ADJACENCY_PENALTY

pub(super) const CJ_PUNCTUATION: i64 = CJK_BASE_SCORE / 2;

pub(super) const CJK_OTHER: i64 = CJK_SECONDARY_BASE_SCORE / 4;

/// Latin letter caseless class
pub(super) const LATIN_LETTER: u8 = 1;
