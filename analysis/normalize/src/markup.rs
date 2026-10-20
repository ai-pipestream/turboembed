//! Strips HTML/XML markup with a cursor state machine (no
//! regular expressions), keeping the text and recording one
//! edit per removed construct so every kept byte maps back
//! to its exact source span.

use turbo_analysis::ViewBuilder;

use crate::{Normalizer, hash_config};

/// Strips element tags (quoted attribute values may hold
/// `>`), comments, doctypes, processing instructions, CDATA
/// wrappers (their inner text stays), and the bodies of
/// `<script>` and `<style>` elements. `<title>` and
/// `<textarea>` bodies are kept as text. A `<` that starts
/// no valid markup is literal text.
///
/// With `space_at_block_boundaries` (the default), a removed
/// block-level tag or raw-text element leaves one space
/// behind (never a second one in a row), so `</p><p>` cannot
/// fuse the words it separated; inline tags leave nothing,
/// so `un<b>break</b>able` stays one word. An unterminated
/// construct at end of input is dropped like a complete one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct HtmlTagStripNormalizer {
    pub space_at_block_boundaries: bool,
}

impl Default for HtmlTagStripNormalizer {
    fn default() -> Self {
        HtmlTagStripNormalizer { space_at_block_boundaries: true }
    }
}

impl HtmlTagStripNormalizer {
    pub fn new() -> HtmlTagStripNormalizer {
        Self::default()
    }

    /// Strips markup without leaving spaces at block
    /// boundaries.
    pub fn without_boundary_spaces() -> HtmlTagStripNormalizer {
        HtmlTagStripNormalizer { space_at_block_boundaries: false }
    }
}

/// What one markup construct does to the text around it.
#[derive(PartialEq)]
enum Emitted {
    /// A space stands where the construct was.
    Space,
    /// Nothing stands where the construct was.
    Nothing,
}

struct Scanner<'a> {
    text: &'a str,
    bytes: &'a [u8],
    /// The last byte written to the output, for boundary
    /// spaces that should not repeat.
    last: Option<u8>,
    space_at_block_boundaries: bool,
}

impl<'a> Scanner<'a> {
    /// Copies `text[from..to]` unchanged.
    fn keep(&mut self, out: &mut ViewBuilder<'_>, from: usize, to: usize) {
        if from < to {
            out.keep(to);
            self.last = Some(self.bytes[to - 1]);
        }
    }

    /// Drops the construct ending at `end`, leaving a space
    /// when `emitted` asks for one and the output does not
    /// already end in whitespace (or is empty).
    fn drop_to(&mut self, out: &mut ViewBuilder<'_>, end: usize, emitted: Emitted) {
        let space = emitted == Emitted::Space
            && self.space_at_block_boundaries
            && self.last.is_some_and(|b| !b.is_ascii_whitespace());
        out.replace(end, if space { " " } else { "" });
        if space {
            self.last = Some(b' ');
        }
    }
}

impl Normalizer for HtmlTagStripNormalizer {
    const ID: &'static str = "turbo:normalize:html-strip";
    const VERSION: &'static str = "1";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let mut scan = Scanner {
            text,
            bytes: text.as_bytes(),
            last: None,
            space_at_block_boundaries: self.space_at_block_boundaries,
        };
        let mut i = 0usize;
        while i < scan.bytes.len() {
            if scan.bytes[i] != b'<' || !starts_markup(scan.bytes, i) {
                i += 1;
                continue;
            }
            scan.keep(out, out.position(), i);
            i = construct(&mut scan, out, i);
        }
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.space_at_block_boundaries)
    }
}

/// Whether the `<` at `i` opens markup: a tag name, an end
/// tag, a declaration, or a processing instruction follows.
fn starts_markup(bytes: &[u8], i: usize) -> bool {
    match bytes.get(i + 1) {
        Some(b) if b.is_ascii_alphabetic() => true,
        Some(b'/') => bytes.get(i + 2).is_some_and(|b| b.is_ascii_alphabetic() || *b == b'>'),
        Some(b'!') | Some(b'?') => true,
        _ => false,
    }
}

/// Consumes the markup construct at `at` (a `<` that starts
/// markup) and returns the offset just past it.
fn construct(scan: &mut Scanner<'_>, out: &mut ViewBuilder<'_>, at: usize) -> usize {
    let bytes = scan.bytes;
    let rest = &scan.text[at..];
    if let Some(inner) = rest.strip_prefix("<!--") {
        let body = at + 4;
        let end = if inner.starts_with('>') || inner.starts_with("->") {
            // Abruptly closed empty comments: <!--> and <!--->.
            body + if inner.starts_with('>') { 1 } else { 2 }
        } else {
            match find_comment_close(scan.text, body) {
                Some((at, len)) => at + len,
                None => bytes.len(),
            }
        };
        scan.drop_to(out, end, Emitted::Nothing);
        return end;
    }
    if rest.starts_with("<![CDATA[") {
        let inner_start = at + 9;
        match scan.text[inner_start..].find("]]>") {
            Some(off) => {
                let inner_end = inner_start + off;
                scan.drop_to(out, inner_start, Emitted::Nothing);
                scan.keep(out, inner_start, inner_end);
                scan.drop_to(out, inner_end + 3, Emitted::Nothing);
                return inner_end + 3;
            }
            None => {
                scan.drop_to(out, inner_start, Emitted::Nothing);
                scan.keep(out, inner_start, bytes.len());
                return bytes.len();
            }
        }
    }
    if rest.starts_with("<!") {
        // A doctype or another declaration; a doctype's
        // internal subset may hold its own `>` inside `[...]`.
        let end = if rest.len() >= 9 && rest.as_bytes()[..9].eq_ignore_ascii_case(b"<!doctype") {
            scan_doctype(scan.text, at)
        } else {
            scan_tag_end(scan.text, at)
        };
        scan.drop_to(out, end, Emitted::Nothing);
        return end;
    }
    if rest.starts_with("<?") {
        let end = match scan.text[at + 2..].find("?>") {
            Some(off) => at + 2 + off + 2,
            None => bytes.len(),
        };
        scan.drop_to(out, end, Emitted::Nothing);
        return end;
    }

    // An element tag, start or end.
    let is_end = bytes[at + 1] == b'/';
    let name_start = at + 1 + usize::from(is_end);
    if bytes.get(name_start) == Some(&b'>') {
        // `</>` is dropped whole.
        scan.drop_to(out, name_start + 1, Emitted::Nothing);
        return name_start + 1;
    }
    let mut name_end = name_start;
    while name_end < bytes.len()
        && (bytes[name_end].is_ascii_alphanumeric() || matches!(bytes[name_end], b'-' | b':' | b'_'))
    {
        name_end += 1;
    }
    let name = &scan.text[name_start..name_end];
    let end = scan_tag_end(scan.text, at);
    let boundary = is_block(name);
    scan.drop_to(out, end, if boundary { Emitted::Space } else { Emitted::Nothing });
    if is_end || end >= bytes.len() {
        return end;
    }
    // Raw-text and RCDATA elements: their bodies are not
    // markup. A tag closed with `/>` has no body.
    let self_closed = bytes[end - 2] == b'/';
    let lower = name.to_ascii_lowercase();
    let raw = match lower.as_str() {
        "script" | "style" => Some(false),
        "title" | "textarea" => Some(true),
        _ => None,
    };
    let Some(keep_body) = raw.filter(|_| !self_closed) else { return end };
    let close = format!("</{lower}");
    match find_raw_close(scan.text, end, &close) {
        Some(close_at) => {
            if keep_body {
                scan.keep(out, out.position(), close_at);
            } else {
                scan.drop_to(out, close_at, Emitted::Space);
            }
            construct(scan, out, close_at)
        }
        None => {
            if keep_body {
                scan.keep(out, out.position(), bytes.len());
            } else {
                scan.drop_to(out, bytes.len(), Emitted::Space);
            }
            bytes.len()
        }
    }
}

/// The offset just past the `>` ending the tag at `at`,
/// honoring quoted attribute values; end of input for an
/// unterminated tag.
fn scan_tag_end(text: &str, at: usize) -> usize {
    // The WHATWG tag states, reduced to what finding the tag
    // end needs. A quote starts a quoted value only in
    // BeforeValue; in a name or an unquoted value, quotes
    // and `=` are ordinary characters and never hide `>`.
    #[derive(PartialEq)]
    enum State {
        TagName,
        BeforeName,
        Name,
        AfterName,
        BeforeValue,
        UnquotedValue,
        QuotedValue(u8),
    }
    let bytes = text.as_bytes();
    let mut state = State::TagName;
    let mut i = at + 1;
    while i < bytes.len() {
        let b = bytes[i];
        match state {
            State::QuotedValue(q) => {
                if b == q {
                    state = State::BeforeName;
                }
            }
            State::TagName => match b {
                b'>' => return i + 1,
                _ if b.is_ascii_whitespace() => state = State::BeforeName,
                b'/' => state = State::BeforeName,
                _ => {}
            },
            State::BeforeName => match b {
                b'>' => return i + 1,
                _ if b.is_ascii_whitespace() || b == b'/' => {}
                _ => state = State::Name,
            },
            State::Name => match b {
                b'>' => return i + 1,
                b'=' => state = State::BeforeValue,
                _ if b.is_ascii_whitespace() => state = State::AfterName,
                b'/' => state = State::BeforeName,
                _ => {}
            },
            State::AfterName => match b {
                b'>' => return i + 1,
                b'=' => state = State::BeforeValue,
                _ if b.is_ascii_whitespace() => {}
                b'/' => state = State::BeforeName,
                _ => state = State::Name,
            },
            State::BeforeValue => match b {
                b'>' => return i + 1,
                _ if b.is_ascii_whitespace() => {}
                b'"' | b'\'' => state = State::QuotedValue(b),
                _ => state = State::UnquotedValue,
            },
            State::UnquotedValue => match b {
                b'>' => return i + 1,
                _ if b.is_ascii_whitespace() => state = State::BeforeName,
                _ => {}
            },
        }
        i += 1;
    }
    bytes.len()
}

/// The offset just past a doctype at `at`: like a tag, but a
/// `>` inside the `[...]` internal subset does not end it.
fn scan_doctype(text: &str, at: usize) -> usize {
    let bytes = text.as_bytes();
    let (mut quote, mut depth) = (0u8, 0usize);
    let mut i = at + 1;
    while i < bytes.len() {
        let b = bytes[i];
        if quote != 0 {
            if b == quote {
                quote = 0;
            }
        } else {
            match b {
                b'"' | b'\'' => quote = b,
                b'[' => depth += 1,
                b']' => depth = depth.saturating_sub(1),
                b'>' if depth == 0 => return i + 1,
                _ => {}
            }
        }
        i += 1;
    }
    bytes.len()
}

/// The comment closer (`-->` or `--!>`) at or after `from`:
/// its offset and length. One forward scan over `--`
/// candidates, so a document full of ordinary comments is
/// linear work, never a rescan to end of input per comment
/// searching for the rarer closer.
fn find_comment_close(text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut at = from;
    while let Some(off) = text[at..].find("--") {
        let dashes = at + off;
        match bytes.get(dashes + 2) {
            Some(b'>') => return Some((dashes, 3)),
            Some(b'!') if bytes.get(dashes + 3) == Some(&b'>') => return Some((dashes, 4)),
            _ => at = dashes + 1,
        }
    }
    None
}

/// The offset of the `</name` closing a raw-text element:
/// the name must be followed by whitespace, `/`, or `>` (or
/// end of input), matched case-insensitively.
fn find_raw_close(text: &str, from: usize, close: &str) -> Option<usize> {
    let hay = &text.as_bytes()[from..];
    let needle = close.as_bytes();
    if hay.len() < needle.len() {
        return None;
    }
    for i in 0..=hay.len() - needle.len() {
        if hay[i..i + needle.len()].eq_ignore_ascii_case(needle) {
            match hay.get(i + needle.len()) {
                None => return Some(from + i),
                Some(b) if b.is_ascii_whitespace() || *b == b'/' || *b == b'>' => return Some(from + i),
                _ => {}
            }
        }
    }
    None
}

/// Whether `name` is a block-level element whose removal
/// separates the words around it.
fn is_block(name: &str) -> bool {
    const BLOCK: &[&str] = &[
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    ];
    BLOCK.iter().any(|b| name.eq_ignore_ascii_case(b))
}
