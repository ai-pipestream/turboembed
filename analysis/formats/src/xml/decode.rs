use super::*;
/// Decodes the document's bytes to text: UTF-8 (with or without a byte
/// order mark) unless the XML declaration names US-ASCII or ISO-8859-1.
pub fn decode(bytes: &[u8]) -> Result<Cow<'_, str>> {
    if bytes.starts_with(&[0xFE, 0xFF]) || bytes.starts_with(&[0xFF, 0xFE]) {
        return Err(Error::XmlEncoding("UTF-16".into()));
    }
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let encoding = declared_encoding(bytes);
    let utf8 = |bytes| {
        std::str::from_utf8(bytes)
            .map_err(|e| Error::Xml { offset: e.valid_up_to(), message: "the document is not valid UTF-8" })
    };
    match encoding.map(|e| e.to_ascii_uppercase()) {
        None => Ok(Cow::Borrowed(utf8(bytes)?)),
        Some(e) if e == b"UTF-8" || e == b"UTF8" => Ok(Cow::Borrowed(utf8(bytes)?)),
        Some(e) if e == b"US-ASCII" || e == b"ASCII" => match bytes.iter().position(|b| !b.is_ascii()) {
            Some(offset) => Err(Error::Xml { offset, message: "a byte above 0x7F in a US-ASCII document" }),
            None => Ok(Cow::Borrowed(utf8(bytes)?)),
        },
        Some(e) if e == b"ISO-8859-1" || e == b"LATIN1" || e == b"ISO8859-1" || e == b"ISO_8859-1" => {
            Ok(Cow::Owned(bytes.iter().map(|&b| char::from(b)).collect()))
        }
        Some(_) => {
            let name = encoding.expect("matched Some");
            Err(Error::XmlEncoding(String::from_utf8_lossy(name).into()))
        }
    }
}

/// The `encoding` pseudo-attribute of the XML declaration, if the
/// document starts with one that has it.
fn declared_encoding(bytes: &[u8]) -> Option<&[u8]> {
    let rest = bytes.strip_prefix(b"<?xml")?;
    if !rest.first().is_some_and(|b| b.is_ascii_whitespace()) {
        return None;
    }
    let end = rest.windows(2).position(|w| w == b"?>")?;
    let decl = &rest[..end];
    let at = decl.windows(8).position(|w| w == b"encoding")?;
    let mut i = at + 8;
    while decl.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }
    if decl.get(i) != Some(&b'=') {
        return None;
    }
    i += 1;
    while decl.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }
    let quote = *decl.get(i)?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    let value = &decl[i + 1..];
    let close = value.iter().position(|&b| b == quote)?;
    Some(&value[..close])
}

/// XMLDecl: version first, then optional encoding and standalone.
pub(super) fn validate_declaration(declaration: &str) -> Result<()> {
    let bad = || Error::Xml { offset: 0, message: "invalid XML declaration" };
    let mut rest = &declaration[5..];
    let mut fields = Vec::new();
    while !rest.is_empty() {
        let trimmed = rest.trim_start_matches(|c: char| c.is_ascii() && is_space(c as u8));
        if trimmed.is_empty() {
            break;
        }
        if trimmed.len() == rest.len() {
            return Err(bad());
        }
        let name_end = trimmed.find(|c: char| !c.is_ascii_alphabetic()).ok_or_else(bad)?;
        let name = &trimmed[..name_end];
        rest = trimmed[name_end..].trim_start_matches(|c: char| c.is_ascii() && is_space(c as u8));
        rest = rest.strip_prefix('=').ok_or_else(bad)?;
        rest = rest.trim_start_matches(|c: char| c.is_ascii() && is_space(c as u8));
        let quote = rest.chars().next().filter(|c| matches!(c, '\'' | '"')).ok_or_else(bad)?;
        rest = &rest[1..];
        let end = rest.find(quote).ok_or_else(bad)?;
        fields.push((name, &rest[..end]));
        if fields.len() > 3 {
            return Err(bad());
        }
        rest = &rest[end + 1..];
    }
    if fields.first() != Some(&("version", "1.0")) {
        return Err(bad());
    }
    let mut at = 1;
    if fields.get(at).is_some_and(|(name, _)| *name == "encoding") {
        let value = fields[at].1.as_bytes();
        if !value.first().is_some_and(u8::is_ascii_alphabetic)
            || !value.iter().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(b))
        {
            return Err(bad());
        }
        at += 1;
    }
    if fields.get(at).is_some_and(|(name, value)| *name == "standalone" && matches!(*value, "yes" | "no")) {
        at += 1;
    }
    if at != fields.len() {
        return Err(bad());
    }
    Ok(())
}
