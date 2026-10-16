//! Loading a dictionary from the bytes of its two files: OpenNLP's
//! `loadStreams`.
//!
//! The steps, and so the order in which errors are reported, are Java's:
//! the affix file's size, its `SET` encoding, the lines it does not parse
//! (an unsupported directive fails a strict load here), its raw flag bytes,
//! decoding and reading it; then the word list's size, its raw flag bytes,
//! decoding and reading it.

pub(crate) mod affix_file;
pub(crate) mod bytes;
pub(crate) mod directive;
pub(crate) mod table;
pub(crate) mod word_list;

use std::hash::Hasher;

use crate::LoadMode;
use crate::charset::Charset;
use crate::dictionary::Dictionary;
use crate::error::{LoadError, Stream};
use crate::hash::FoldHasher as Fingerprint;

/// What the affix format writes for no text: an empty strip or affix, an
/// empty replacement, or a `CHECKCOMPOUNDPATTERN` end that asks for an
/// unaffixed part.
pub(crate) const NO_MATERIAL: &str = "0";

/// The largest file the reader takes, 64 MiB, as OpenNLP's
/// `MAX_STREAM_BYTES`.
pub const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

/// Fails when `bytes` is larger than [`MAX_FILE_BYTES`].
pub(crate) fn check_size(bytes: usize, stream: Stream) -> Result<(), LoadError> {
    if bytes > MAX_FILE_BYTES { Err(LoadError::TooLarge { stream, limit: MAX_FILE_BYTES }) } else { Ok(()) }
}

/// `text` without a leading U+FEFF, which decoding a UTF-8 byte order mark
/// leaves.
fn without_byte_order_mark(text: &str) -> &str {
    text.strip_prefix('\u{FEFF}').unwrap_or(text)
}

/// Loads a dictionary. `words` gives the word list once the affix file has
/// been read; `source` names the affix file in a diagnostic about an
/// unsupported directive.
pub(crate) fn load(
    mut affix: Vec<u8>,
    words: impl FnOnce() -> Result<Vec<u8>, LoadError>,
    mode: LoadMode,
    source: &str,
) -> Result<Dictionary, LoadError> {
    check_size(affix.len(), Stream::Affix)?;
    let mut fingerprint = Fingerprint::default();
    fingerprint.write(&affix);
    let charset = bytes::declared_charset(&affix)?;
    let unsupported = bytes::mask_unparsed_lines(&mut affix, mode, source)?;
    let raw_flags = charset == Charset::Utf8 && !bytes::uses_unicode_or_numeric_flags(&affix);
    let affix = bytes::normalize_affix_flags(affix, charset);
    let text = charset.decode(&affix).ok_or(LoadError::Malformed { stream: Stream::Affix, charset: charset.name() })?;
    let parsed = affix_file::parse(without_byte_order_mark(&text))?;
    drop(text);
    drop(affix);
    let mut words = words()?;
    check_size(words.len(), Stream::Dictionary)?;
    fingerprint.write(&words);
    fingerprint.write_u8(mode as u8);
    if raw_flags {
        words = bytes::normalize_word_flags(&words);
    }
    let text =
        charset.decode(&words).ok_or(LoadError::Malformed { stream: Stream::Dictionary, charset: charset.name() })?;
    let list = word_list::parse(without_byte_order_mark(&text), &parsed)?;
    Ok(Dictionary::build(parsed, list, unsupported, fingerprint.finish()))
}

/// Reads an open file of at most [`MAX_FILE_BYTES`].
pub(crate) fn read_bounded(file: std::fs::File, stream: Stream) -> Result<Vec<u8>, LoadError> {
    use std::io::Read;
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES as u64 + 1).read_to_end(&mut bytes)?;
    check_size(bytes.len(), stream)?;
    Ok(bytes)
}
