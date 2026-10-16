//! Morfologik `.info` metadata properties and `fsa.dict.input-conversion` substitutions.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use turbo_unicode::java::to_lowercase;

use crate::charset::DictionaryCharset;
use crate::encoding::{BaseFormEncoding, trim_java_string};
use crate::error::MorfologikError;

/// Property key declaring the field separator character (`fsa.dict.separator`).
pub const KEY_SEPARATOR: &str = "fsa.dict.separator";

/// Property key declaring the dictionary character encoding (`fsa.dict.encoding`).
pub const KEY_ENCODING: &str = "fsa.dict.encoding";

/// Property key declaring the base-form encoder (`fsa.dict.encoder`).
pub const KEY_ENCODER: &str = "fsa.dict.encoder";

/// Property key declaring ordered input substitutions (`fsa.dict.input-conversion`).
pub const KEY_INPUT_CONVERSION: &str = "fsa.dict.input-conversion";

/// Ordered string substitutions from Morfologik `fsa.dict.input-conversion` metadata
/// (`opennlp.tools.formats.MorfologikInputConversion`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MorfologikInputConversion {
    replacements: Vec<(String, String)>,
}

impl MorfologikInputConversion {
    /// Parses comma-separated pairs containing a single space between strings.
    pub fn parse(value: &str) -> Result<Self, MorfologikError> {
        let mut pairs: Vec<&str> = Vec::new();
        let bytes = value.as_bytes();
        let mut start = 0usize;
        let mut separated = false;
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b',' {
                pairs.push(&value[start..i]);
                separated = true;
                start = i + 1;
                while start < bytes.len() && is_separator_whitespace(bytes[start]) {
                    start += 1;
                }
                i = start;
                continue;
            }
            i += 1;
        }
        pairs.push(&value[start..]);
        if separated {
            while pairs.last().is_some_and(|p| p.is_empty()) {
                pairs.pop();
            }
        }

        let mut replacements = Vec::new();
        let mut inputs = HashSet::new();
        for pair in pairs {
            let trimmed = trim_java_string(pair);
            let Some(space) = trimmed.find(' ') else {
                return Err(MorfologikError::invalid_argument(format!(
                    "{KEY_INPUT_CONVERSION} requires space-separated string pairs"
                )));
            };
            if space == 0 || space == trimmed.len() - 1 || trimmed[space + 1..].contains(' ') {
                return Err(MorfologikError::invalid_argument(format!(
                    "{KEY_INPUT_CONVERSION} requires space-separated string pairs"
                )));
            }
            let from = &trimmed[..space];
            let to = &trimmed[space + 1..];
            if !inputs.insert(from) {
                return Err(MorfologikError::invalid_argument(format!(
                    "{KEY_INPUT_CONVERSION} repeats input string: {from}"
                )));
            }
            if from != to {
                replacements.push((from.to_owned(), to.to_owned()));
            }
        }
        Ok(Self { replacements })
    }

    /// Whether the rule list has no substitutions.
    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty()
    }

    /// Replaces non-overlapping matches for each rule in metadata order.
    pub fn apply(&self, token: &str) -> String {
        if self.replacements.is_empty() {
            return token.to_owned();
        }
        let mut current = token.to_owned();
        for (from, to) in &self.replacements {
            if !current.contains(from.as_str()) {
                continue;
            }
            let mut next = String::with_capacity(current.len());
            let mut rest = current.as_str();
            while let Some(pos) = rest.find(from.as_str()) {
                next.push_str(&rest[..pos]);
                next.push_str(to);
                rest = &rest[pos + from.len()..];
            }
            next.push_str(rest);
            current = next;
        }
        current
    }

    /// Lower-cases `token` with `Locale.ROOT` (`turbo_unicode::java::to_lowercase`),
    /// applies the ordered input conversions, and folds the result to lower case as
    /// `DictionaryLemmatizer.lemmatize` does on lookup.
    pub fn convert(&self, token: &str) -> String {
        let mut lower = String::with_capacity(token.len());
        to_lowercase(token, &mut lower);
        if self.replacements.is_empty() {
            lower
        } else {
            let applied = self.apply(&lower);
            let mut out = String::with_capacity(applied.len());
            to_lowercase(&applied, &mut out);
            out
        }
    }
}

#[inline]
fn is_separator_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0c | 0x0b)
}

/// Parsed `.info` dictionary metadata (`morfologik.stemming.DictionaryMetadata`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorfologikMetadata {
    /// Single separator byte in `charset`.
    pub separator: u8,
    /// Character encoding of the dictionary.
    pub charset: DictionaryCharset,
    /// Base-form encoder (`NONE`, `SUFFIX`, `PREFIX`, `INFIX`).
    pub encoder: BaseFormEncoding,
    /// Optional ordered input substitutions (`fsa.dict.input-conversion`).
    pub input_conversion: MorfologikInputConversion,
}

impl MorfologikMetadata {
    /// Parses UTF-8 `.info` metadata bytes.
    pub fn parse(info_bytes: &[u8]) -> Result<Self, MorfologikError> {
        let text =
            std::str::from_utf8(info_bytes).map_err(|_| MorfologikError::format("invalid UTF-8 in .info metadata"))?;
        let props = parse_java_properties(text)?;

        let separator_str = props.get(KEY_SEPARATOR).ok_or_else(|| {
            MorfologikError::invalid_argument(format!("missing required metadata key: {KEY_SEPARATOR}"))
        })?;
        let mut sep_chars = separator_str.chars();
        let Some(sep_char) = sep_chars.next() else {
            return Err(MorfologikError::invalid_argument(format!("{KEY_SEPARATOR} must be a single character")));
        };
        if sep_chars.next().is_some() {
            return Err(MorfologikError::invalid_argument(format!("{KEY_SEPARATOR} must be a single character")));
        }

        let encoding_str = props.get(KEY_ENCODING).ok_or_else(|| {
            MorfologikError::invalid_argument(format!("missing required metadata key: {KEY_ENCODING}"))
        })?;
        let charset = DictionaryCharset::for_name(encoding_str)?;

        let encoder_str = props.get(KEY_ENCODER).ok_or_else(|| {
            MorfologikError::invalid_argument(format!("missing required metadata key: {KEY_ENCODER}"))
        })?;
        let encoder = BaseFormEncoding::parse_name(encoder_str)?;
        let separator = charset.encode_single_byte(sep_char)?;

        let input_conversion = match props.get(KEY_INPUT_CONVERSION) {
            Some(rules) => MorfologikInputConversion::parse(rules)?,
            None => MorfologikInputConversion::default(),
        };

        Ok(Self { separator, charset, encoder, input_conversion })
    }

    /// Derives the expected `.info` metadata file path for a `.dict` or `.txt` file path
    /// (`MorfologikUtil.getExpectedPropertiesFile` / `DictionaryMetadata.getExpectedMetadataLocation`).
    pub fn expected_metadata_path(dict_path: impl AsRef<Path>) -> PathBuf {
        let path = dict_path.as_ref();
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            return path.with_extension("info");
        };
        if let Some(stem) = file_name
            .strip_suffix(".dict")
            .or_else(|| file_name.strip_suffix(".txt"))
            .or_else(|| file_name.rfind('.').filter(|&i| i > 0).map(|i| &file_name[..i]))
        {
            path.with_file_name(format!("{stem}.info"))
        } else {
            path.with_file_name(format!("{file_name}.info"))
        }
    }
}

/// Parses Java `.properties` text (`java.util.Properties.load(Reader)`).
fn parse_java_properties(text: &str) -> Result<HashMap<String, String>, MorfologikError> {
    let mut map = HashMap::new();
    for line in logical_property_lines(text) {
        let (key_raw, val_raw) = split_property_line(&line);
        let key = unescape_java_property(key_raw)?;
        let val = unescape_java_property(val_raw)?;
        map.insert(key, val);
    }
    Ok(map)
}

/// Joins natural lines (ended by `\n`, `\r`, or `\r\n`) into logical lines as
/// `Properties.load` does: comment and blank lines are dropped, leading whitespace is
/// stripped, and a line ending in an odd number of backslashes continues on the next line.
fn logical_property_lines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current: Option<String> = None;
    let mut rest = text;
    while !rest.is_empty() {
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let natural = &rest[..end];
        rest = &rest[end..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        } else if !rest.is_empty() {
            rest = &rest[1..];
        }

        let line = natural.trim_start_matches([' ', '\t', '\u{000c}']);
        if current.is_none() && (line.is_empty() || line.starts_with('#') || line.starts_with('!')) {
            continue;
        }
        let trailing_backslashes = line.bytes().rev().take_while(|&b| b == b'\\').count();
        let continues = trailing_backslashes % 2 == 1;
        let body = if continues { &line[..line.len() - 1] } else { line };
        current.get_or_insert_with(String::new).push_str(body);
        if !continues {
            lines.extend(current.take());
        }
    }
    lines.extend(current);
    lines
}

fn split_property_line(line: &str) -> (&str, &str) {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    let mut escaped = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else if b == b'=' || b == b':' || b == b' ' || b == b'\t' || b == 0x0c {
            break;
        }
        i += 1;
    }
    let key = &line[..i];
    let mut j = i;
    while j < bytes.len() && matches!(bytes[j], b' ' | b'\t' | 0x0c) {
        j += 1;
    }
    if j < bytes.len() && (bytes[j] == b'=' || bytes[j] == b':') {
        j += 1;
        while j < bytes.len() && matches!(bytes[j], b' ' | b'\t' | 0x0c) {
            j += 1;
        }
    }
    (key, &line[j..])
}

fn unescape_java_property(s: &str) -> Result<String, MorfologikError> {
    // `\uXXXX` escapes produce UTF-16 code units, so a supplementary character may be
    // written as an escaped surrogate pair; only unpaired surrogates are rejected.
    let mut units: Vec<u16> = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(ch.encode_utf16(&mut buf));
            continue;
        }
        let Some(esc) = chars.next() else {
            units.push(u16::from(b'\\'));
            break;
        };
        let decoded = match esc {
            't' => '\t',
            'n' => '\n',
            'r' => '\r',
            'f' => '\u{000c}',
            'u' => {
                let mut code: u16 = 0;
                for _ in 0..4 {
                    let Some(hex) = chars.next().and_then(|c| c.to_digit(16)) else {
                        return Err(MorfologikError::invalid_argument("malformed \\uXXXX escape in .info metadata"));
                    };
                    code = (code << 4) | hex as u16;
                }
                units.push(code);
                continue;
            }
            other => other,
        };
        let mut buf = [0u16; 2];
        units.extend_from_slice(decoded.encode_utf16(&mut buf));
    }
    String::from_utf16(&units).map_err(|_| MorfologikError::invalid_argument("lone surrogate in .info metadata"))
}
