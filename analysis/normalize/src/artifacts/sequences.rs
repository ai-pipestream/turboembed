//! Mojibake found one UTF-8 sequence at a time, the way the character
//! classes of ftfy's `badness.py` find it: a character that is the image
//! of a UTF-8 lead byte (Ã Â Ð Ñ Î â in Windows-1252, √ ‚ in MacRoman)
//! followed by the images of the continuation bytes that lead needs.
//!
//! The byte round trip (`mojibake.rs`) needs a whole run of non-ASCII
//! characters to decode. It misses text that was damaged further: a
//! genuine character next to the image ("cafÃ©…", where the ellipsis was
//! typed), a byte Windows-1252 leaves undefined replaced by U+FFFD
//! ("Ã\u{FFFD}" for "Á"), and UTF-8 read as MacRoman ("caf√©"). This scan
//! looks at each sequence on its own, so those are found too.
//!
//! A two-character image can occur by chance in genuine text (an accented
//! capital before a typographic quote), so a sequence counts only when it
//! decodes to a character mojibake is made of: for Windows-1252, a
//! two-byte sequence must decode to U+00A0..U+017F (Latin-1 Supplement
//! and Latin Extended-A) or U+0370..U+04FF (Greek and Cyrillic), and a
//! longer one to any scalar; for MacRoman, whose high half holds the
//! accented lowercase letters that follow quotes and dashes in genuine
//! text, a two-byte sequence must decode to U+00A0..U+00FF and a
//! three-byte one must start with ‚ (byte 0xE2) and decode to
//! U+2000..U+214F (punctuation, currency and letterlike symbols, the
//! "‚Äô" family). These are the ranges ftfy's rules single out.

use turbo_analysis::Span;

use super::codepage::CodePage;
use super::mojibake::{is_continuation, lead};

const REPLACEMENT: char = '\u{FFFD}';

/// Reports each maximal stretch of adjacent sequence images, leaving out
/// sequences that overlap a span of `skip` (the round-trip findings,
/// which take precedence). `skip` is in text order.
pub(crate) fn scan(text: &str, skip: &[Span], out: &mut Vec<Span>) {
    let mut skip = skip.iter().peekable();
    let mut i = 0;
    while i < text.len() {
        let c = super::char_at(text, i);
        let Some(len) = sequence_at(text, i, c) else {
            i += c.len_utf8();
            continue;
        };
        let found = Span::new(i as u32, (i + len) as u32);
        i += len;
        while skip.next_if(|s| s.end <= found.start).is_some() {}
        if skip.peek().is_some_and(|s| s.start < found.end) {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.end == found.start => last.end = found.end,
            _ => out.push(found),
        }
    }
}

/// The byte length of the sequence image starting with `c` at `at`, in
/// the first code page that has one there.
fn sequence_at(text: &str, at: usize, c: char) -> Option<usize> {
    if c.is_ascii() {
        return None;
    }
    [CodePage::Windows1252, CodePage::MacRoman].into_iter().find_map(|page| image_len(page, text, at, c))
}

/// One continuation slot of a sequence image.
#[derive(Copy, Clone)]
enum Slot {
    /// A character that is the image of this continuation byte.
    Byte(u8),
    /// U+FFFD where a decoder dropped a byte it could not map.
    Lost,
}

fn image_len(page: CodePage, text: &str, at: usize, c: char) -> Option<usize> {
    let lead_byte = page.byte(c)?;
    let (continuations, min, max) = lead(lead_byte)?;
    let mut slots = [Slot::Lost; 3];
    let mut end = at + c.len_utf8();
    let mut rest = text[end..].chars();
    for slot in slots.iter_mut().take(continuations) {
        let next = rest.next()?;
        *slot = match page.byte(next) {
            Some(b) if is_continuation(b) => Slot::Byte(b),
            // Only Windows-1252 has undefined bytes in the continuation
            // range (0x81, 0x8D, 0x8F, 0x90, 0x9D).
            _ if next == REPLACEMENT && page == CodePage::Windows1252 => Slot::Lost,
            _ => return None,
        };
        end += next.len_utf8();
    }
    let slots = &slots[..continuations];
    if let Slot::Byte(second) = slots[0]
        && !second_byte_allowed(lead_byte, second)
    {
        return None;
    }
    let accepted = match decode(lead_byte, continuations, slots) {
        Some(scalar) => {
            scalar >= min
                && scalar <= max
                && !(0xD800..=0xDFFF).contains(&scalar)
                && target(page, lead_byte, continuations, scalar)
        }
        None => lost_target(page, lead_byte, continuations),
    };
    accepted.then_some(end - at)
}

/// The second-byte ranges of RFC 3629 that rule out overlong forms,
/// surrogates and values past U+10FFFF before the sequence is complete.
fn second_byte_allowed(lead: u8, second: u8) -> bool {
    match lead {
        0xE0 => (0xA0..=0xBF).contains(&second),
        0xED => (0x80..=0x9F).contains(&second),
        0xF0 => (0x90..=0xBF).contains(&second),
        0xF4 => (0x80..=0x8F).contains(&second),
        _ => true,
    }
}

/// The scalar the sequence encodes, or `None` when a slot was lost.
fn decode(lead: u8, continuations: usize, slots: &[Slot]) -> Option<u32> {
    let mut scalar = u32::from(lead) & (0x3F >> continuations);
    for slot in slots {
        let Slot::Byte(b) = *slot else { return None };
        scalar = (scalar << 6) | u32::from(b & 0x3F);
    }
    Some(scalar)
}

/// Whether a complete sequence decodes to a character mojibake is made of
/// (see the module comment).
fn target(page: CodePage, lead: u8, continuations: usize, scalar: u32) -> bool {
    match (page, continuations) {
        (CodePage::Windows1252, 1) => (0xA0..=0x17F).contains(&scalar) || (0x370..=0x4FF).contains(&scalar),
        (CodePage::Windows1252, _) => true,
        (CodePage::MacRoman, 1) => (0xA0..=0xFF).contains(&scalar),
        (CodePage::MacRoman, 2) => lead == 0xE2 && (0x2000..=0x214F).contains(&scalar),
        (CodePage::MacRoman, _) => false,
    }
}

/// Whether a sequence with a lost byte could have decoded to a target: a
/// two-byte lead must have some completion inside the two-byte target
/// ranges; longer sequences decode to any scalar under Windows-1252, the
/// only code page with lost bytes.
fn lost_target(page: CodePage, lead: u8, continuations: usize) -> bool {
    if continuations > 1 {
        return page == CodePage::Windows1252;
    }
    let low = (u32::from(lead) & 0x1F) << 6;
    (low..=low + 0x3F).any(|scalar| target(page, lead, 1, scalar))
}
