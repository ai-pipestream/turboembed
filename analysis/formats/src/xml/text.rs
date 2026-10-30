use super::*;
/// XML 1.0's `Char` production. Rust strings cannot hold surrogates, so
/// only control characters and U+FFFE and U+FFFF are left to refuse.
pub(super) fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{FFFD}' | '\u{10000}'..)
}

pub(super) fn is_name_start(c: char) -> bool {
    matches!(c,
        ':' | 'A'..='Z' | '_' | 'a'..='z' | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}' | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}' | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

pub(super) fn is_name_char(c: char) -> bool {
    is_name_start(c) || matches!(c, '-' | '.' | '0'..='9' | '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
}

pub(super) fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// The local part of a qualified name.
pub(super) fn local(name: &str) -> &str {
    name.split_once(':').map_or(name, |(_, local)| local)
}
/// Passes text to the handler with `\r\n` and lone `\r` turned into `\n`.
pub(super) fn emit_normalized(mut text: &str, handler: &mut impl Handler) {
    while let Some(cr) = text.find('\r') {
        if cr > 0 {
            handler.characters(&text[..cr]);
        }
        handler.characters("\n");
        let skip = if text[cr + 1..].starts_with('\n') { 2 } else { 1 };
        text = &text[cr + skip..];
    }
    if !text.is_empty() {
        handler.characters(text);
    }
}

/// Resolves the entity or character reference at the start of `text`,
/// giving the character and the length of the reference.
pub(super) fn reference(text: &str, offset: usize) -> Result<(char, usize)> {
    let Some(semi) = text.find(';') else {
        return Err(Error::Xml { offset, message: "a reference is not closed by ';'" });
    };
    let body = &text[1..semi];
    let c = match body {
        "lt" => '<',
        "gt" => '>',
        "amp" => '&',
        "apos" => '\'',
        "quot" => '"',
        _ => {
            let value = if let Some(hex) = body.strip_prefix("#x") {
                (!hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()))
                    .then(|| u32::from_str_radix(hex, 16).ok())
                    .flatten()
            } else if let Some(dec) = body.strip_prefix('#') {
                (!dec.is_empty() && dec.bytes().all(|b| b.is_ascii_digit())).then(|| dec.parse::<u32>().ok()).flatten()
            } else {
                return Err(Error::Xml { offset, message: "an entity other than the five XML defines" });
            };
            match value.and_then(char::from_u32).filter(|&c| is_xml_char(c)) {
                Some(c) => c,
                None => {
                    return Err(Error::Xml {
                        offset,
                        message: "a character reference to a character XML does not allow",
                    });
                }
            }
        }
    };
    Ok((c, semi + 1))
}

/// An attribute value with references resolved and each tab, line feed
/// and carriage return (a `\r\n` pair counting as one) replaced by a
/// space. A character reference to whitespace is kept as it is.
pub(super) fn attribute_value(raw: &str, offset: usize) -> Result<Cow<'_, str>> {
    if !raw.bytes().any(|b| matches!(b, b'&' | b'\t' | b'\n' | b'\r')) {
        return Ok(Cow::Borrowed(raw));
    }
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    let bytes = raw.as_bytes();
    while i < raw.len() {
        match bytes[i] {
            b'&' => {
                let (c, used) = reference(&raw[i..], offset + i)?;
                out.push(c);
                i += used;
            }
            b'\r' => {
                out.push(' ');
                i += if bytes.get(i + 1) == Some(&b'\n') { 2 } else { 1 };
            }
            b'\t' | b'\n' => {
                out.push(' ');
                i += 1;
            }
            _ => {
                let next = raw[i..].find(['&', '\t', '\n', '\r']).map_or(raw.len(), |n| i + n);
                out.push_str(&raw[i..next]);
                i = next;
            }
        }
    }
    Ok(Cow::Owned(out))
}
