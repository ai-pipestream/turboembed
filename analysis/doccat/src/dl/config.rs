//! Configuration parser for OpenNLP `DocumentCategorizerDL`
//! (`opennlp.dl.doccat.DocumentCategorizerConfig`).

use std::collections::BTreeMap;

use super::DlDoccatError;

/// JSON configuration key naming the id-to-label map in Hugging Face
/// `config.json` files (`"id2label"`).
pub const ID2LABEL_KEY: &str = "id2label";

/// Configuration for `DocumentCategorizerDL`, ported from
/// `opennlp.dl.doccat.DocumentCategorizerConfig`.
///
/// Parses the `"id2label"` object from a Hugging Face `config.json` payload.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocumentCategorizerConfig {
    id2label: BTreeMap<String, String>,
}

impl DocumentCategorizerConfig {
    /// Creates a `DocumentCategorizerConfig` from a map of string IDs to
    /// category labels.
    pub fn new(id2label: BTreeMap<String, String>) -> Self {
        Self { id2label }
    }

    /// Parses `"id2label"` from a `config.json` string, matching
    /// `DocumentCategorizerConfig.fromJson(String)`.
    ///
    /// Returns `None` if the JSON does not contain a valid `"id2label"` object
    /// with string keys and string values.
    pub fn from_json(json: &str) -> Option<Self> {
        let map = parse_id2label_object(json)?;
        Some(Self { id2label: map })
    }

    /// Parses integer-indexed categories (`"0"`..`"N-1"`) from a `config.json`
    /// string, matching `DocumentCategorizerDL.readCategoriesFromFile`.
    pub fn read_categories_from_json(json: &str) -> Result<BTreeMap<usize, String>, DlDoccatError> {
        let cfg = Self::from_json(json).ok_or_else(|| {
            DlDoccatError::InvalidConfig("Failed to parse DocumentCategorizerConfig from config.json".into())
        })?;
        cfg.indexed_categories()
    }

    /// Returns the raw `"id2label"` map.
    pub fn id2label(&self) -> &BTreeMap<String, String> {
        &self.id2label
    }

    /// Converts the `"id2label"` entries into an integer-keyed map
    /// (`0..num_categories`), rejecting non-integer keys, empty maps, or
    /// non-contiguous category indices.
    pub fn indexed_categories(&self) -> Result<BTreeMap<usize, String>, DlDoccatError> {
        if self.id2label.is_empty() {
            return Err(DlDoccatError::InvalidConfig("No categories found in config.json id2label".into()));
        }
        let mut categories = BTreeMap::new();
        for (k, v) in &self.id2label {
            let idx = k.trim().parse::<usize>().map_err(|_| {
                DlDoccatError::InvalidConfig(format!("Category index key {k:?} in id2label is not a valid usize"))
            })?;
            if v.trim().is_empty() {
                return Err(DlDoccatError::InvalidConfig(format!("Category label at index {idx} must not be blank")));
            }
            if categories.insert(idx, v.clone()).is_some() {
                return Err(DlDoccatError::InvalidConfig(format!("Duplicate category index {idx} in id2label")));
            }
        }
        for expected in 0..categories.len() {
            if !categories.contains_key(&expected) {
                return Err(DlDoccatError::InvalidConfig(format!(
                    "id2label indices must be contiguous 0..{}, missing index {expected}",
                    categories.len()
                )));
            }
        }
        Ok(categories)
    }

    /// Returns the category labels ordered by index `0..num_categories`.
    pub fn ordered_categories(&self) -> Result<Vec<String>, DlDoccatError> {
        let indexed = self.indexed_categories()?;
        Ok(indexed.into_values().collect())
    }
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

fn parse_json_string(src: &str, start: usize) -> Option<(String, usize)> {
    let bytes = src.as_bytes();
    if bytes.get(start).copied() != Some(b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        let ch = src[i..].chars().next()?;
        i += ch.len_utf8();
        match ch {
            '"' => return Some((out, i)),
            '\\' => {
                let esc = src.get(i..)?.chars().next()?;
                i += esc.len_utf8();
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000C}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let hex = src.get(i..i + 4)?;
                        let code = u16::from_str_radix(hex, 16).ok()?;
                        i += 4;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if src.get(i..i + 2) != Some("\\u") {
                                return None;
                            }
                            let low_hex = src.get(i + 2..i + 6)?;
                            let low = u16::from_str_radix(low_hex, 16).ok()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return None;
                            }
                            i += 6;
                            let combined = 0x10000 + (((u32::from(code) - 0xD800) << 10) | (u32::from(low) - 0xDC00));
                            out.push(char::from_u32(combined)?);
                        } else {
                            out.push(char::from_u32(u32::from(code))?);
                        }
                    }
                    _ => return None,
                }
            }
            c if c.is_control() => return None,
            c => out.push(c),
        }
    }
    None
}

const MAX_JSON_DEPTH: usize = 64;

fn parse_string_map(json: &str, start: usize) -> Option<(BTreeMap<String, String>, usize)> {
    let bytes = json.as_bytes();
    if bytes.get(start).copied() != Some(b'{') {
        return None;
    }
    let mut i = skip_ws(bytes, start + 1);
    let mut map = BTreeMap::new();
    if bytes.get(i).copied() == Some(b'}') {
        return Some((map, i + 1));
    }
    loop {
        let (entry_key, after_key) = parse_json_string(json, i)?;
        i = skip_ws(bytes, after_key);
        if bytes.get(i).copied() != Some(b':') {
            return None;
        }
        i = skip_ws(bytes, i + 1);
        let (entry_val, after_val) = parse_json_string(json, i)?;
        if map.insert(entry_key, entry_val).is_some() {
            return None;
        }
        i = skip_ws(bytes, after_val);
        match bytes.get(i).copied() {
            Some(b',') => {
                i = skip_ws(bytes, i + 1);
            }
            Some(b'}') => return Some((map, i + 1)),
            _ => return None,
        }
    }
}

fn skip_json_number(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    if bytes.get(i).copied() == Some(b'-') {
        i += 1;
    }
    match bytes.get(i).copied()? {
        b'0' => {
            i += 1;
            if matches!(bytes.get(i).copied(), Some(b'0'..=b'9')) {
                return None;
            }
        }
        b'1'..=b'9' => {
            i += 1;
            while matches!(bytes.get(i).copied(), Some(b'0'..=b'9')) {
                i += 1;
            }
        }
        _ => return None,
    }
    if bytes.get(i).copied() == Some(b'.') {
        i += 1;
        if !matches!(bytes.get(i).copied(), Some(b'0'..=b'9')) {
            return None;
        }
        while matches!(bytes.get(i).copied(), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    if matches!(bytes.get(i).copied(), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i).copied(), Some(b'+' | b'-')) {
            i += 1;
        }
        if !matches!(bytes.get(i).copied(), Some(b'0'..=b'9')) {
            return None;
        }
        while matches!(bytes.get(i).copied(), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    Some(i)
}

fn skip_json_value(json: &str, start: usize, depth: usize) -> Option<usize> {
    if depth > MAX_JSON_DEPTH {
        return None;
    }
    let bytes = json.as_bytes();
    let i = skip_ws(bytes, start);
    match bytes.get(i).copied()? {
        b'"' => parse_json_string(json, i).map(|(_, next)| next),
        b'{' => {
            let mut cur = skip_ws(bytes, i + 1);
            if bytes.get(cur).copied() == Some(b'}') {
                return Some(cur + 1);
            }
            loop {
                let (_, after_key) = parse_json_string(json, cur)?;
                cur = skip_ws(bytes, after_key);
                if bytes.get(cur).copied() != Some(b':') {
                    return None;
                }
                cur = skip_json_value(json, cur + 1, depth + 1)?;
                cur = skip_ws(bytes, cur);
                match bytes.get(cur).copied() {
                    Some(b',') => cur = skip_ws(bytes, cur + 1),
                    Some(b'}') => return Some(cur + 1),
                    _ => return None,
                }
            }
        }
        b'[' => {
            let mut cur = skip_ws(bytes, i + 1);
            if bytes.get(cur).copied() == Some(b']') {
                return Some(cur + 1);
            }
            loop {
                cur = skip_json_value(json, cur, depth + 1)?;
                cur = skip_ws(bytes, cur);
                match bytes.get(cur).copied() {
                    Some(b',') => cur = skip_ws(bytes, cur + 1),
                    Some(b']') => return Some(cur + 1),
                    _ => return None,
                }
            }
        }
        b't' => bytes.get(i..i + 4).filter(|&s| s == b"true").map(|_| i + 4),
        b'f' => bytes.get(i..i + 5).filter(|&s| s == b"false").map(|_| i + 5),
        b'n' => bytes.get(i..i + 4).filter(|&s| s == b"null").map(|_| i + 4),
        b'-' | b'0'..=b'9' => skip_json_number(bytes, i),
        _ => None,
    }
}

fn parse_id2label_object(json: &str) -> Option<BTreeMap<String, String>> {
    let bytes = json.as_bytes();
    let mut i = skip_ws(bytes, 0);
    if bytes.get(i).copied() != Some(b'{') {
        return None;
    }
    i = skip_ws(bytes, i + 1);
    if bytes.get(i).copied() == Some(b'}') {
        return None;
    }
    let mut found: Option<BTreeMap<String, String>> = None;
    loop {
        let (key, after_key) = parse_json_string(json, i)?;
        i = skip_ws(bytes, after_key);
        if bytes.get(i).copied() != Some(b':') {
            return None;
        }
        i = skip_ws(bytes, i + 1);
        if key == ID2LABEL_KEY {
            if found.is_some() {
                return None;
            }
            let (map, after_map) = parse_string_map(json, i)?;
            found = Some(map);
            i = skip_ws(bytes, after_map);
        } else {
            let after_val = skip_json_value(json, i, 1)?;
            i = skip_ws(bytes, after_val);
        }
        match bytes.get(i).copied() {
            Some(b',') => {
                i = skip_ws(bytes, i + 1);
            }
            Some(b'}') => {
                i = skip_ws(bytes, i + 1);
                if i != bytes.len() {
                    return None;
                }
                return found;
            }
            _ => return None,
        }
    }
}
