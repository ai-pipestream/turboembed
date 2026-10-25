//! Bounded line-oriented source key/value fields.
use crate::table_grid::{code_lines, lines, trim};
use turbo_analysis::{Annotator, Error, Layer, Refs, Span, View};
use turbo_analysis_layers::{KeyValuePair, KeyValuePairs};
use turbo_unicode::{WhitespaceMode, java};

#[derive(Clone, Copy, Debug)]
pub struct KeyValueAnnotator {
    pub max_key_tokens: u16,
}

impl Default for KeyValueAnnotator {
    fn default() -> Self {
        Self { max_key_tokens: 6 }
    }
}

impl Annotator for KeyValueAnnotator {
    const ID: &'static str = "turbo-structure:key-values";
    type Reads = (View,);
    type Writes = (KeyValuePairs,);
    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(KeyValuePairs,), Error> {
        let fail = |detail: &str| Error::Annotator { annotator: Self::ID, detail: detail.into() };
        if self.max_key_tokens == 0 {
            return Err(fail("max_key_tokens must be positive"));
        }
        let text = view.text();
        if text.len() > u32::MAX as usize {
            return Err(fail("text exceeds the u32 coordinate range"));
        }
        let lines = lines(text);
        let blocked = code_lines(text, view.id(), &lines);
        let mut out = KeyValuePairs::new(view.id());
        for (line, blocked) in lines.iter().zip(blocked) {
            if blocked {
                continue;
            }
            let Some(relative) = text[line.start..line.end].find([':', '=']) else { continue };
            let at = line.start + relative;
            let key = trim(text, line.start, at);
            let value = trim(text, at + 1, line.end);
            let name = &text[key.range()];
            let count =
                name.split(|c: char| WhitespaceMode::Unicode.is_whitespace(c as u32)).filter(|s| !s.is_empty()).count();
            if count == 0
                || count > usize::from(self.max_key_tokens)
                || !name.chars().any(java::is_letter)
                || !name.chars().all(|c| {
                    java::is_letter_or_digit(c)
                        || WhitespaceMode::Unicode.is_whitespace(c as u32)
                        || matches!(c, '_' | '-' | '.' | '(' | ')' | '[' | ']' | '%' | '/')
                })
                || text[value.range()].starts_with("//")
            {
                continue;
            }
            out.spans.push(Span::new(key.start, value.end));
            out.pairs.push(KeyValuePair { key, value, separator: Span::new(at as u32, at as u32 + 1) });
        }
        out.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((out,))
    }
    fn config_hash(&self) -> u64 {
        (0x6b76_0000_0000_0001 ^ u64::from(self.max_key_tokens)).wrapping_mul(0x0100_0000_01b3)
    }
}
