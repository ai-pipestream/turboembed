//! UTF-8. Valid stretches are copied with no edit. Each maximal subpart of
//! an ill-formed sequence (the standard's UTF-8 decoder, and Unicode's
//! "substitution of maximal subparts") becomes one U+FFFD mapped to
//! exactly those bytes; `str::utf8_chunks` splits the input the same way.

use turbo_analysis::DecodeBuilder;

use super::REPLACEMENT;

pub(crate) fn decode(src: &[u8], out: &mut DecodeBuilder<'_>) -> u32 {
    let mut errors = 0;
    for chunk in src[out.position()..].utf8_chunks() {
        out.keep_str(chunk.valid());
        let bad = chunk.invalid();
        if !bad.is_empty() {
            out.replace(out.position() + bad.len(), REPLACEMENT);
            errors += 1;
        }
    }
    errors
}
