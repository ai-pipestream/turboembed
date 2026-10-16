//! The single-byte encodings and x-user-defined. ASCII runs are copied
//! with no edit; every other byte is one edit, to its character or to
//! U+FFFD where the index has none.

use turbo_analysis::DecodeBuilder;

use super::REPLACEMENT;

pub(crate) fn decode(src: &[u8], table: &[u16; 128], out: &mut DecodeBuilder<'_>) -> u32 {
    decode_with(src, out, |b| char::from_u32(u32::from(table[usize::from(b - 0x80)])).filter(|&c| c != '\0'))
}

/// x-user-defined: byte 0x80 + n is U+F780 + n.
pub(crate) fn x_user_defined(src: &[u8], out: &mut DecodeBuilder<'_>) -> u32 {
    decode_with(src, out, |b| char::from_u32(0xF780 + u32::from(b) - 0x80))
}

fn decode_with(src: &[u8], out: &mut DecodeBuilder<'_>, high: impl Fn(u8) -> Option<char>) -> u32 {
    let mut errors = 0;
    let mut i = out.position();
    while i < src.len() {
        let run = src[i..].iter().position(|b| !b.is_ascii()).unwrap_or(src.len() - i);
        if run > 0 {
            i += run;
            out.keep(i);
            continue;
        }
        i += 1;
        match high(src[i - 1]) {
            Some(c) => out.push(i, c),
            None => {
                out.replace(i, REPLACEMENT);
                errors += 1;
            }
        }
    }
    errors
}
