//! `Urls`: OpenNLP's `UrlCharSequenceNormalizer`.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};

use super::mail;
use crate::Normalizer;
use crate::placeholder::{self, Sink, span};

/// Replaces each `http://` or `https://` URL, and then each e-mail
/// address, with one U+0020.
///
/// A URL is the lowercase scheme `http` or `https`, `://`, and a greedy
/// run of at least one character of `[-_.?&~;+=/#0-9A-Za-z]`. The run
/// ends at the first character outside that set, so a port, a percent
/// escape, userinfo or a non-ASCII host cuts the URL short
/// (`http://f:21/` becomes `" :21/"`). The scheme needs nothing before
/// it: `xhttp://y` becomes `"x "`. E-mail addresses follow the rules in
/// the `mail` module.
///
/// OpenNLP counterpart: `UrlCharSequenceNormalizer` (main branch), which
/// removes URLs in one pass over the text and e-mail addresses in a second
/// pass over the result. This scan does both in one pass and produces the
/// same text: see [`Urls::scan`].
///
/// As an `Annotator` it reads a `View` and writes the normalized `View`
/// and a `Placeholders` layer with one `Url` or `Email` row per
/// replacement, spanning the replaced text in the view it read.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Urls;

/// The URL body set, `[-_.?&~;+=/#0-9A-Za-z]`.
fn is_url_body(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'?' | b'&' | b'~' | b';' | b'+' | b'=' | b'/' | b'#')
}

/// The end of the URL that starts at byte `start`, if one does.
fn url_end(text: &[u8], start: usize) -> Option<usize> {
    let rest = &text[start..];
    let body = if rest.starts_with(b"https://") {
        start + 8
    } else if rest.starts_with(b"http://") {
        start + 7
    } else {
        return None;
    };
    if !is_url_body(*text.get(body)?) {
        return None;
    }
    let mut end = body + 1;
    while end < text.len() && is_url_body(text[end]) {
        end += 1;
    }
    Some(end)
}

/// The end of the e-mail address that starts at byte `start`, as OpenNLP's
/// second pass sees it after the first has replaced every URL with a
/// space. A URL can start inside the domain run (`a@bchttp://x`): the
/// first pass turns it into a space, which ends the domain there, so the
/// domain is cut at that URL and must still be two characters long. A URL
/// cannot start inside the local part, because the `:` of its scheme is
/// not a local-part character and the local part must end at `@`.
fn mail_end(text: &[u8], start: usize) -> Option<usize> {
    let address = mail::match_at(text, start)?;
    let cut = (address.domain_start..address.end).find(|&q| text[q] == b'h' && url_end(text, q).is_some());
    match cut {
        None => Some(address.end),
        Some(q) => (q - address.domain_start >= mail::MIN_DOMAIN_LEN).then_some(q),
    }
}

impl Urls {
    /// Writes `text` with every URL and e-mail address replaced by one
    /// space, and reports each replacement to `sink`.
    ///
    /// OpenNLP runs two passes; this is one, trying a URL and then an
    /// address at each position. The results are the same:
    ///
    /// - The URL pass does not look at what precedes a URL, so its matches
    ///   do not depend on the address pass. Addresses never contain a URL
    ///   start (see `mail_end`), so the URLs are found at the same places.
    /// - The address pass refuses an address whose preceding character is
    ///   a local-part character. After a URL, OpenNLP's second pass sees
    ///   the space the URL became, where this scan sees the URL's last
    ///   character. The difference cannot matter: an address starts with a
    ///   local-part character, every local-part character is in the URL
    ///   body set, and so the character right after a URL never starts an
    ///   address in either scan.
    ///
    /// A randomized test compares this scan with a two-pass reference.
    pub fn scan(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        let bytes = text.as_bytes();
        // Every URL contains ':' and every address '@'.
        if !bytes.iter().any(|&b| b == b':' || b == b'@') {
            return;
        }
        let mut i = 0;
        while i < bytes.len() {
            let found = match bytes[i] {
                b'h' => url_end(bytes, i).map(|end| (end, PlaceholderKind::Url)),
                _ => None,
            }
            .or_else(|| mail_end(bytes, i).map(|end| (end, PlaceholderKind::Email)));
            match found {
                Some((end, kind)) => {
                    out.keep(i);
                    out.replace(end, " ");
                    sink(span(i, end), kind);
                    i = end;
                }
                None => i += 1,
            }
        }
    }
}

impl Normalizer for Urls {
    const ID: &'static str = "turbo:normalize:urls";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        Urls::scan(text, out, &mut placeholder::discard);
    }
}

impl Annotator for Urls {
    const ID: &'static str = <Urls as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, Urls::scan))
    }
}
