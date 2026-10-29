use turbo_analysis::Span;

use super::LegalError;
use super::text::Decoded;
use crate::sample::DocumentSample;

/// Entries of Bouvier's 1856 dictionary in the Constitution Society's HTML.
#[derive(Debug)]
pub struct BouvierReader {
    entries: std::vec::IntoIter<DocumentSample<'static>>,
}

impl BouvierReader {
    pub fn new(source: &str) -> Result<BouvierReader, LegalError> {
        if source.len() > u32::MAX as usize {
            return Err(LegalError::Markup { offset: source.len(), expected: "a document within the offset range" });
        }
        let mut entries: Vec<DocumentSample<'static>> = Vec::new();
        let mut pos = 0;
        while let Some(open) = find_ci(source, "<p", pos) {
            let after_open = source[open..]
                .find('>')
                .map(|i| open + i + 1)
                .ok_or(LegalError::Markup { offset: open, expected: "the end of a paragraph tag" })?;
            let close = find_ci(source, "</p>", after_open)
                .ok_or(LegalError::Markup { offset: after_open, expected: "a closing paragraph tag" })?;
            let end = close + 4;
            if let Some(bold) = find_ci(source, "<b>", after_open).filter(|&i| i < close) {
                let bold_close = find_ci(source, "</b>", bold + 3)
                    .filter(|&i| i < close)
                    .ok_or(LegalError::Markup { offset: bold, expected: "a closing bold tag" })?;
                let term = decode(source, bold + 3, bold_close)?;
                if term.is_empty() {
                    return Err(LegalError::Markup { offset: bold, expected: "a nonempty term" });
                }
                let definition = decode(source, bold_close + 4, close)?;
                entries.push(DocumentSample {
                    category: term.field(),
                    tokens: definition.tokens(),
                    source: Span::new(open as u32, end as u32),
                });
            } else if continuation(&source[after_open..close]) {
                let definition = decode(source, after_open, close)?;
                if let Some(entry) = entries.last_mut() {
                    entry.tokens.extend(definition.tokens());
                    entry.source.end = end as u32;
                }
            }
            pos = end;
        }
        Ok(BouvierReader { entries: entries.into_iter() })
    }
}

impl Iterator for BouvierReader {
    type Item = DocumentSample<'static>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next()
    }
}

fn continuation(text: &str) -> bool {
    let mut chars = text.trim_start().chars();
    chars.next().is_some_and(|c| c.is_ascii_digit()) && chars.next() == Some('.')
}

fn find_ci(source: &str, needle: &str, start: usize) -> Option<usize> {
    source[start..].to_ascii_lowercase().find(needle).map(|i| start + i)
}

fn decode(source: &str, start: usize, end: usize) -> Result<Decoded, LegalError> {
    let mut out = Decoded::new(start);
    let mut pos = start;
    while pos < end {
        if source.as_bytes()[pos] == b'<' {
            let close = source[pos..end]
                .find('>')
                .map(|i| pos + i)
                .ok_or(LegalError::Markup { offset: pos, expected: "the end of a tag" })?;
            pos = close + 1;
            continue;
        }
        if source.as_bytes()[pos] == b'&' {
            let close = source[pos..end]
                .find(';')
                .map(|i| pos + i)
                .ok_or(LegalError::Markup { offset: pos, expected: "the end of an entity" })?;
            let entity = &source[pos + 1..close];
            let c = entity_char(entity).ok_or(LegalError::Markup { offset: pos, expected: "a known entity" })?;
            out.push(c, pos, close + 1);
            pos = close + 1;
            continue;
        }
        let c = source[pos..].chars().next().expect("pos is within the source");
        out.push(c, pos, pos + c.len_utf8());
        pos += c.len_utf8();
    }
    Ok(out)
}

fn entity_char(entity: &str) -> Option<char> {
    let named = match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "sect" => Some('\u{a7}'),
        "nbsp" => Some('\u{a0}'),
        "laquo" => Some('\u{ab}'),
        "raquo" => Some('\u{bb}'),
        "ldquo" | "lsquo" => Some('\u{201c}'),
        "rdquo" | "rsquo" => Some('\u{201d}'),
        "hellip" => Some('\u{2026}'),
        "middot" => Some('\u{b7}'),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    let value = entity.strip_prefix('#')?;
    let (radix, digits) = value.strip_prefix(['x', 'X']).map_or((10, value), |digits| (16, digits));
    char::from_u32(u32::from_str_radix(digits, radix).ok()?)
}
