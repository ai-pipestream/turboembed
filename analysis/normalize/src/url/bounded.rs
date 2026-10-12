//! `BoundedUrls`: OpenNLP's `BoundedUrlCharSequenceNormalizer`.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};
use turbo_unicode::category::{GeneralCategory, general_category};

use super::mail;
use crate::Normalizer;
use crate::placeholder::{self, Sink, span};
use crate::tables;

/// Replaces each `http` or `https` URL, taken whole as it is bounded in
/// running text, and each e-mail address with one U+0020. URLs with other
/// schemes are kept as they are.
///
/// A URL is a scheme (an ASCII letter, then ASCII letters, digits, `+`,
/// `-` or `.`), `://`, and a body:
///
/// - The scheme must start the text or follow a character that is not a
///   letter or digit (`java.lang.Character.isLetterOrDigit`) and not one
///   of `_ + - . :`, so the `https` inside `git+https://` or
///   `blob:https://` does not start a URL of its own.
/// - The body runs to the end of the text or to the first whitespace
///   character (the 25 characters of OpenNLP's whitespace table), control
///   character (Cc), initial or final quotation mark (Pi, Pf), `<`, `>` or
///   `"`. It therefore includes ports, userinfo, percent escapes,
///   bracketed IPv6 hosts and non-ASCII hosts and paths.
/// - Trailing `. , ; : ! ?` are given back to the text, and so is a
///   trailing `)`, `]` or `}` that has no opening bracket in the body.
/// - An empty body (`http://`, `http://.`) is no URL.
///
/// The scheme is compared ASCII case-insensitively: `HTTP://X` is
/// replaced. A URL with another scheme is copied unchanged, and no e-mail
/// address is matched inside it.
///
/// E-mail addresses follow the rules of [`Urls`](super::Urls), with every
/// non-ASCII letter, digit and combining mark added to the local part and
/// the domain, so internationalized addresses (RFC 6531, with U-label
/// domains) such as `José@example.com` are replaced as a whole.
///
/// OpenNLP counterpart: `BoundedUrlCharSequenceNormalizer`, which is a
/// single pass like this one. OpenNLP's address scan is ASCII only: it
/// leaves `José@example.com` unchanged, because the local part before `@`
/// must be a run of ASCII characters that starts after a non-local one.
///
/// As an `Annotator` it reads a `View` and writes the normalized `View`
/// and a `Placeholders` layer with one `Url` or `Email` row per
/// replacement.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BoundedUrls;

/// `[A-Za-z0-9+.-]`: the characters of a scheme after its first letter.
fn is_scheme_body(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')
}

/// Whether the character before a scheme makes it part of a larger token.
fn joins_scheme(c: char) -> bool {
    turbo_unicode::java::is_letter_or_digit(c) || matches!(c, '_' | '+' | '-' | '.' | ':')
}

/// The scheme that starts at byte `start`: where it ends (before `://`)
/// and where the body starts (after it).
fn scheme_at(text: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    if !bytes[start].is_ascii_alphabetic() {
        return None;
    }
    if text[..start].chars().next_back().is_some_and(joins_scheme) {
        return None;
    }
    let mut at = start + 1;
    while at < bytes.len() && is_scheme_body(bytes[at]) {
        at += 1;
    }
    bytes[at..].starts_with(b"://").then_some((at, at + 3))
}

/// Whether `c` ends a URL body.
fn ends_body(c: char) -> bool {
    if c.is_ascii() {
        return matches!(c, '<' | '>' | '"') || c.is_ascii_control() || tables::is_whitespace(c);
    }
    tables::is_whitespace(c)
        || matches!(general_category(c), GeneralCategory::Cc | GeneralCategory::Pi | GeneralCategory::Pf)
}

/// The end of the body that starts at byte `body`, after the trailing
/// characters that belong to the text are given back. Equal to `body`
/// when nothing is left.
fn body_end(text: &str, body: usize) -> usize {
    let end = text[body..].char_indices().find(|&(_, c)| ends_body(c)).map_or(text.len(), |(k, _)| body + k);
    trim_trailing(text.as_bytes(), body, end)
}

/// Gives back trailing sentence punctuation and closing brackets that have
/// no opening bracket in the body. The brackets are counted over the whole
/// body first, so a closer is kept when the body opened one anywhere.
fn trim_trailing(bytes: &[u8], body: usize, mut end: usize) -> usize {
    let (mut round, mut square, mut curly) = (0i64, 0i64, 0i64);
    for &b in &bytes[body..end] {
        match b {
            b'(' => round += 1,
            b')' => round -= 1,
            b'[' => square += 1,
            b']' => square -= 1,
            b'{' => curly += 1,
            b'}' => curly -= 1,
            _ => {}
        }
    }
    while end > body {
        let balance = match bytes[end - 1] {
            b'.' | b',' | b';' | b':' | b'!' | b'?' => None,
            b')' if round < 0 => Some(&mut round),
            b']' if square < 0 => Some(&mut square),
            b'}' if curly < 0 => Some(&mut curly),
            _ => break,
        };
        if let Some(count) = balance {
            *count += 1;
        }
        end -= 1;
    }
    end
}

/// Whether the scheme `text[start..end]` is `http` or `https`, ASCII
/// case-insensitively.
fn is_http(text: &str, start: usize, end: usize) -> bool {
    let scheme = &text[start..end];
    scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
}

impl BoundedUrls {
    /// Writes `text` with every http(s) URL and e-mail address replaced by
    /// one space, and reports each replacement to `sink`.
    pub fn scan(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        let bytes = text.as_bytes();
        if !bytes.iter().any(|&b| b == b':' || b == b'@') {
            return;
        }
        let mut i = 0;
        while i < bytes.len() {
            if let Some((scheme_end, body)) = scheme_at(text, i) {
                let end = body_end(text, body);
                if end > body {
                    if is_http(text, i, scheme_end) {
                        out.keep(i);
                        out.replace(end, " ");
                        sink(span(i, end), PlaceholderKind::Url);
                    }
                    i = end;
                    continue;
                }
            }
            if let Some(address) = mail::match_international_at(text, i) {
                out.keep(i);
                out.replace(address.end, " ");
                sink(span(i, address.end), PlaceholderKind::Email);
                i = address.end;
                continue;
            }
            i += utf8_len(bytes[i]);
        }
    }
}

/// The length of the UTF-8 sequence that starts with byte `b`.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

impl Normalizer for BoundedUrls {
    const ID: &'static str = "turbo:normalize:bounded-urls";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        BoundedUrls::scan(text, out, &mut placeholder::discard);
    }
}

impl Annotator for BoundedUrls {
    const ID: &'static str = <BoundedUrls as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, BoundedUrls::scan))
    }
}
