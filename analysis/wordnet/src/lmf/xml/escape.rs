use super::super::malformed;
use crate::error::WordNetError;

/// Decodes the body of a numeric character reference (`#65` or `#x41`, without `&` and `;`).
///
/// XML 1.0 section 4.1 allows only ASCII digits (no sign, no blanks) after `#` or a lowercase `#x`,
/// and the result must match the `Char` production of section 2.2.
fn character_reference(entity: &str, resource_name: &str, line: usize) -> Result<char, WordNetError> {
    let (digits, radix) = match entity.strip_prefix("#x") {
        Some(hex) => (hex, 16),
        None => (&entity[1..], 10),
    };
    let well_formed = !digits.is_empty() && digits.chars().all(|c| c.is_digit(radix));
    let code = u32::from_str_radix(digits, radix)
        .ok()
        .filter(|_| well_formed)
        .ok_or_else(|| malformed(resource_name, Some(line), format!("Invalid character reference '&{entity};'")))?;
    char::from_u32(code).filter(|&ch| is_xml_char(ch)).ok_or_else(|| {
        malformed(resource_name, Some(line), format!("Character reference '&{entity};' is not a legal XML character"))
    })
}

/// XML 1.0 section 2.2 `Char`: tab, newline, carriage return, and every scalar value from
/// U+0020 except the surrogates (already excluded by [`char`]) and U+FFFE and U+FFFF.
pub(super) fn is_xml_char(ch: char) -> bool {
    matches!(ch, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
}

/// XML 1.0 (Fifth Edition) + Namespaces in XML 1.0 `NCNameStartChar` (NameStartChar minus `:`).
fn is_ncname_start_char(ch: char) -> bool {
    matches!(
        ch,
        'A'..='Z'
            | '_'
            | 'a'..='z'
            | '\u{C0}'..='\u{D6}'
            | '\u{D8}'..='\u{F6}'
            | '\u{F8}'..='\u{2FF}'
            | '\u{370}'..='\u{37D}'
            | '\u{37F}'..='\u{1FFF}'
            | '\u{200C}'..='\u{200D}'
            | '\u{2070}'..='\u{218F}'
            | '\u{2C00}'..='\u{2FEF}'
            | '\u{3001}'..='\u{D7FF}'
            | '\u{F900}'..='\u{FDCF}'
            | '\u{FDF0}'..='\u{FFFD}'
            | '\u{10000}'..='\u{EFFFF}'
    )
}

/// XML 1.0 (Fifth Edition) + Namespaces in XML 1.0 `NCNameChar` (NameChar minus `:`).
fn is_ncname_char(ch: char) -> bool {
    is_ncname_start_char(ch)
        || matches!(ch, '-' | '.' | '0'..='9' | '\u{B7}' | '\u{0300}'..='\u{036F}' | '\u{203F}'..='\u{2040}')
}

/// XML Namespaces 1.0 `NCName` production (`Name` without `:`).
pub(super) fn is_ncname(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else { return false };
    is_ncname_start_char(first) && chars.all(is_ncname_char)
}

pub(super) fn split_qname<'a>(
    qname: &'a str,
    resource_name: &str,
    line: usize,
) -> Result<(&'a str, &'a str), WordNetError> {
    if let Some((prefix, local)) = qname.split_once(':') {
        if prefix.is_empty() || local.is_empty() || local.contains(':') {
            return Err(malformed(resource_name, Some(line), format!("Invalid XML QName '{qname}'")));
        }
        Ok((prefix, local))
    } else {
        Ok(("", qname))
    }
}

/// Unescapes XML character/entity references.
/// When `normalize_attr_whitespace` is `true` (per W3C XML 1.0 section 3.3.3), literal
/// `\r\n`, `\r`, `\n`, and `\t` outside character references are normalized to `' '`,
/// whereas numeric character references like `&#9;` expand to their exact character (`\t`).
pub(in crate::lmf) fn unescape_xml(
    raw: &str,
    resource_name: &str,
    line: usize,
    normalize_attr_whitespace: bool,
) -> Result<String, WordNetError> {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if normalize_attr_whitespace && matches!(b, b'\r' | b'\n' | b'\t') {
            if b == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            out.push(' ');
            continue;
        }
        if b == b'&' {
            let after_amp = &raw[i + 1..];
            let semi = after_amp.find(';').ok_or_else(|| {
                malformed(resource_name, Some(line), format!("Unescaped '&' or unclosed entity in '{raw}'"))
            })?;
            let entity = &after_amp[..semi];
            match entity {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "apos" => out.push('\''),
                "quot" => out.push('"'),
                _ if entity.starts_with('#') => out.push(character_reference(entity, resource_name, line)?),
                _ => {
                    return Err(malformed(
                        resource_name,
                        Some(line),
                        format!("Unknown XML entity reference '&{entity};'"),
                    ));
                }
            }
            i += 1 + semi + 1;
            continue;
        }
        let ch = raw[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}
