//! The per-code-point artifact classes and the two contextual classes,
//! zero-width characters and Unicode tags: a port of the class scan of
//! OpenNLP's `CursorArtifactDetector`, over the UTF-8 text.

use turbo_normalize_layers::ArtifactKind;
use turbo_unicode::category::general_category;
use turbo_unicode::{java, props};

use super::{Artifact, KindSet, char_at, char_before};

const REPLACEMENT: char = '\u{FFFD}';
const ZERO_WIDTH_JOINER: char = '\u{200D}';
const VARIATION_SELECTOR_16: char = '\u{FE0F}';
const WAVING_BLACK_FLAG: char = '\u{1F3F4}';
const TAG_BASE: u32 = 0xE0000;
const CANCEL_TAG: u32 = 0xE007F;
/// An alphabetic region code has two letters, a numeric one three
/// digits; a subdivision adds one to four letters or digits (UTS #35,
/// `unicode_subdivision_id`).
const ALPHA_REGION_LENGTH: usize = 2;
const NUMERIC_REGION_LENGTH: usize = 3;
const MAX_SUBDIVISION_SUFFIX: usize = 4;

/// One pass that reports maximal runs of each per-code-point class and
/// the zero-width and tag runs its context rules leave reported. Kinds
/// not in `kinds` are treated as ordinary text.
pub(crate) fn scan(text: &str, kinds: KindSet, out: &mut Vec<Artifact>) {
    let bytes = text.as_bytes();
    let zero_width = kinds.contains(ArtifactKind::ZeroWidth);
    let tags = kinds.contains(ArtifactKind::UnicodeTag);
    let mut run: Option<(ArtifactKind, usize)> = None;
    let mut i = 0;
    while i < bytes.len() {
        // Printable ASCII is never in a class: close any run and move on.
        if (0x20..0x7F).contains(&bytes[i]) {
            close(&mut run, i, out);
            i += 1;
            continue;
        }
        let c = char_at(text, i);
        let kind = classify(c).filter(|&k| kinds.contains(k));
        if run.is_some_and(|(k, _)| Some(k) != kind) {
            close(&mut run, i, out);
        }
        if let (Some(k), None) = (kind, run) {
            run = Some((k, i));
        }
        if kind.is_none() && zero_width && is_zero_width(c) {
            i = zero_width_run(text, i, out);
            continue;
        }
        if kind.is_none() && tags && is_tag(c) {
            i = tag_run(text, i, out);
            continue;
        }
        i += c.len_utf8();
    }
    close(&mut run, bytes.len(), out);
}

fn close(run: &mut Option<(ArtifactKind, usize)>, end: usize, out: &mut Vec<Artifact>) {
    if let Some((kind, start)) = run.take() {
        out.push(Artifact::new(start, end, kind));
    }
}

/// The class of one code point, ignoring context. Zero-width characters
/// and tags have none here: their runs are resolved by context.
pub(crate) fn classify(c: char) -> Option<ArtifactKind> {
    let cp = c as u32;
    if c == REPLACEMENT {
        return Some(ArtifactKind::ReplacementChar);
    }
    if (cp < 0x20 || (0x7F..=0x9F).contains(&cp)) && !props::white_space(c) {
        return Some(ArtifactKind::Control);
    }
    if (0xFDD0..=0xFDEF).contains(&cp) || cp & 0xFFFE == 0xFFFE {
        return Some(ArtifactKind::Noncharacter);
    }
    if (0xE000..=0xF8FF).contains(&cp) || (0xF0000..=0xFFFFD).contains(&cp) || (0x100000..=0x10FFFD).contains(&cp) {
        return Some(ArtifactKind::PrivateUse);
    }
    if (0x202A..=0x202E).contains(&cp) || (0x2066..=0x2069).contains(&cp) || matches!(cp, 0x200E | 0x200F | 0x061C) {
        return Some(ArtifactKind::BidiControl);
    }
    None
}

/// The invisible zero-width characters whose reporting depends on context.
pub(crate) fn is_zero_width(c: char) -> bool {
    matches!(c, '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}')
}

/// The Tags block, U+E0000 to U+E007F.
pub(crate) fn is_tag(c: char) -> bool {
    (TAG_BASE..=CANCEL_TAG).contains(&(c as u32))
}

/// Resolves the maximal zero-width run starting at `start` and returns
/// the offset after it. Two or more are reported. One alone is left in
/// two contexts: a joiner next to a pictograph or U+FE0F (emoji ZWJ
/// sequences), and any of them between letters, combining marks passed
/// over (ZWNJ in Persian, ZWJ in Sinhala, ZWSP in Thai).
fn zero_width_run(text: &str, start: usize, out: &mut Vec<Artifact>) -> usize {
    let mut end = start;
    let mut count = 0;
    let mut only = '\0';
    while end < text.len() {
        let c = char_at(text, end);
        if !is_zero_width(c) {
            break;
        }
        only = c;
        count += 1;
        end += c.len_utf8();
    }
    let reported = count >= 2 || !orthographic(text, start, end, only);
    if reported {
        out.push(Artifact::new(start, end, ArtifactKind::ZeroWidth));
    }
    end
}

/// Whether a single zero-width character `c` at `start..end` is in one of
/// the contexts where it belongs to the text.
fn orthographic(text: &str, start: usize, end: usize, c: char) -> bool {
    let emoji = |n: Option<char>| n.is_some_and(|n| props::extended_pictographic(n) || n == VARIATION_SELECTOR_16);
    if c == ZERO_WIDTH_JOINER && (emoji(char_before(text, start)) || emoji(text[end..].chars().next())) {
        return true;
    }
    letter_before(text, start) && letter_after(text, end)
}

/// Whether the first code point before `at` that is not a combining mark
/// is a letter.
fn letter_before(text: &str, at: usize) -> bool {
    text[..at].chars().rev().find(|&c| !general_category(c).is_mark()).is_some_and(java::is_letter)
}

/// Whether the first code point from `at` that is not a combining mark is
/// a letter.
fn letter_after(text: &str, at: usize) -> bool {
    text[at..].chars().find(|&c| !general_category(c).is_mark()).is_some_and(java::is_letter)
}

/// Resolves the maximal tag run starting at `start`, reporting it unless
/// it is the tag sequence of a flag, and returns the offset after it.
fn tag_run(text: &str, start: usize, out: &mut Vec<Artifact>) -> usize {
    let end = start + text[start..].chars().take_while(|&c| is_tag(c)).map(char::len_utf8).sum::<usize>();
    if !is_flag_tags(text, start, end) {
        out.push(Artifact::new(start, end, ArtifactKind::UnicodeTag));
    }
    end
}

/// Whether the tag run `start..end` follows U+1F3F4 WAVING BLACK FLAG and
/// spells a region or subdivision code ended by CANCEL TAG, by the syntax
/// of UTS #51 (emoji tag sequences) and UTS #35 (`unicode_subdivision_id`):
/// two letters or three digits, then at most four letters or digits in
/// all past the region, then U+E007F. Whether CLDR assigns the code is
/// not checked.
fn is_flag_tags(text: &str, start: usize, end: usize) -> bool {
    if char_before(text, start) != Some(WAVING_BLACK_FLAG) {
        return false;
    }
    let mut count = 0;
    let mut alpha_region = true;
    let mut numeric_region = true;
    let mut chars = text[start..end].chars();
    while let Some(c) = chars.next() {
        if c as u32 == CANCEL_TAG {
            // A region alone is not enough for the alphabetic form: a
            // subdivision needs at least one more character.
            let alpha = alpha_region
                && (ALPHA_REGION_LENGTH + 1..=ALPHA_REGION_LENGTH + MAX_SUBDIVISION_SUFFIX).contains(&count);
            let numeric = numeric_region
                && (NUMERIC_REGION_LENGTH..=NUMERIC_REGION_LENGTH + MAX_SUBDIVISION_SUFFIX).contains(&count);
            return chars.as_str().is_empty() && (alpha || numeric);
        }
        let ascii = c as u32 - TAG_BASE;
        let letter = (u32::from(b'a')..=u32::from(b'z')).contains(&ascii);
        let digit = (u32::from(b'0')..=u32::from(b'9')).contains(&ascii);
        if !letter && !digit {
            return false;
        }
        if count < ALPHA_REGION_LENGTH {
            alpha_region &= letter;
        }
        if count < NUMERIC_REGION_LENGTH {
            numeric_region &= digit;
        }
        count += 1;
        if count > NUMERIC_REGION_LENGTH + MAX_SUBDIVISION_SUFFIX {
            return false;
        }
    }
    false
}
