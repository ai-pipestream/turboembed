//! Decodes HTML character references (named, decimal,
//! hexadecimal) in one pass, recording one edit per decoded
//! reference so decoded characters map back to the exact
//! reference bytes.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;
use crate::entities_table;

/// Decodes every character reference the WHATWG table
/// defines: all 2,231 named references (the 106 legacy names
/// also match without their semicolon), decimal `&#160;` and
/// hexadecimal `&#xA0;` references with the spec's
/// replacements (NUL, surrogates and out-of-range values
/// become U+FFFD; C1 controls remap through the Windows-1252
/// table), with or without the trailing semicolon.
///
/// Decoding is a single pass and never rescans its own
/// output, so `&amp;lt;` decodes to the literal text `&lt;`,
/// not to `<`. An `&` that starts no valid reference is kept
/// as it is.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct HtmlEntityDecodeNormalizer;

impl HtmlEntityDecodeNormalizer {
    pub fn new() -> HtmlEntityDecodeNormalizer {
        HtmlEntityDecodeNormalizer
    }
}

impl Normalizer for HtmlEntityDecodeNormalizer {
    const ID: &'static str = "turbo:normalize:html-entities";
    const VERSION: &'static str = "1";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'&'
                && let Some((end, decoded)) = reference_at(text, i)
            {
                out.keep(i);
                out.replace(end, &decoded);
                i = end;
                continue;
            }
            i += 1;
        }
    }
}

/// The reference at byte `at` (which holds `&`): the byte
/// just past it and its decoded text.
fn reference_at(text: &str, at: usize) -> Option<(usize, String)> {
    let rest = &text[at + 1..];
    if let Some(numeric) = rest.strip_prefix('#') {
        let (radix, digits) = match numeric.strip_prefix(['x', 'X']) {
            Some(hex) => (16u32, hex),
            None => (10u32, numeric),
        };
        let taken = digits.bytes().take_while(|b| (*b as char).is_digit(radix)).count();
        if taken == 0 {
            return None;
        }
        let mut value: u32 = 0;
        for b in digits.bytes().take(taken) {
            value = value.saturating_mul(radix).saturating_add((b as char).to_digit(radix).unwrap());
            if value > 0x10FFFF {
                value = 0x110000;
            }
        }
        // Total length: '&', the '#' or '#x' prefix, the
        // digits, and the semicolon when present.
        let prefix = rest.len() - digits.len();
        let semicolon = usize::from(digits.as_bytes().get(taken) == Some(&b';'));
        let end = at + 1 + prefix + taken + semicolon;
        return Some((end, map_code_point(value).to_string()));
    }
    // Named: the longest table key that prefixes the input.
    // Keys are ASCII and at most MAX_KEY bytes, so slicing to
    // that window cannot split a character in a key position
    // that could match.
    let window = &rest[..rest.len().min(entities_table::MAX_KEY)];
    for len in (1..=window.len()).rev() {
        let Some(candidate) = window.get(..len) else { continue };
        if let Some(decoded) = entities_table::lookup(candidate) {
            return Some((at + 1 + len, decoded.to_string()));
        }
    }
    None
}

/// The WHATWG numeric-reference code point mapping.
fn map_code_point(value: u32) -> char {
    match value {
        0x00 | 0xD800..=0xDFFF | 0x110000.. => '\u{FFFD}',
        // The C1 remap table (Windows-1252): entries not
        // listed (81, 8D, 8F, 90, 9D) keep their value.
        0x80 => '\u{20AC}',
        0x82 => '\u{201A}',
        0x83 => '\u{0192}',
        0x84 => '\u{201E}',
        0x85 => '\u{2026}',
        0x86 => '\u{2020}',
        0x87 => '\u{2021}',
        0x88 => '\u{02C6}',
        0x89 => '\u{2030}',
        0x8A => '\u{0160}',
        0x8B => '\u{2039}',
        0x8C => '\u{0152}',
        0x8E => '\u{017D}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201C}',
        0x94 => '\u{201D}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x98 => '\u{02DC}',
        0x99 => '\u{2122}',
        0x9A => '\u{0161}',
        0x9B => '\u{203A}',
        0x9C => '\u{0153}',
        0x9E => '\u{017E}',
        0x9F => '\u{0178}',
        v => char::from_u32(v).unwrap_or('\u{FFFD}'),
    }
}
