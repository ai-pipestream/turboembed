//! Reading the UCD files in tests, independently of the generator.

#![allow(dead_code)]

use std::ops::RangeInclusive;
use std::path::PathBuf;

/// The directory of the UCD files.
pub fn ucd_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ucd")
}

/// The text of the UCD file `name`.
pub fn read(name: &str) -> String {
    let path = ucd_dir().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The data lines of a UCD file split into trimmed fields, with comments
/// and blank lines dropped.
pub fn records(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .filter(|data| !data.trim().is_empty())
        .map(|data| data.split(';').map(|field| field.trim().to_owned()).collect())
        .collect()
}

/// A hexadecimal code point.
pub fn cp(hex: &str) -> u32 {
    u32::from_str_radix(hex.trim(), 16).unwrap_or_else(|_| panic!("not a code point: {hex}"))
}

/// A code point or a `XXXX..YYYY` range.
pub fn range(field: &str) -> RangeInclusive<u32> {
    match field.split_once("..") {
        Some((a, b)) => cp(a)..=cp(b),
        None => cp(field)..=cp(field),
    }
}

/// A space-separated code point sequence as a string.
pub fn string(field: &str) -> String {
    field.split_whitespace().map(|hex| char::from_u32(cp(hex)).expect("not a surrogate")).collect()
}

/// Every `char` (every code point but the surrogates).
pub fn all_chars() -> impl Iterator<Item = char> {
    (0..=0x10FFFFu32).filter_map(char::from_u32)
}

/// The code points a binary property of `file` covers, as a membership table.
pub fn binary_property(file: &str, property: &str) -> Vec<bool> {
    let mut set = vec![false; 0x110000];
    let mut found = false;
    for fields in records(&read(file)) {
        if fields.len() == 2 && fields[1] == property {
            found = true;
            for c in range(&fields[0]) {
                set[c as usize] = true;
            }
        }
    }
    assert!(found, "{file} has no {property}");
    set
}

/// One line of UnicodeData.txt, expanded over First/Last ranges.
pub struct UnicodeDataRow {
    pub cp: u32,
    pub fields: Vec<String>,
}

/// Every assigned code point of UnicodeData.txt with its fields.
pub fn unicode_data() -> Vec<UnicodeDataRow> {
    let text = read("UnicodeData.txt");
    let mut rows = Vec::new();
    let mut first: Option<u32> = None;
    for line in text.lines() {
        let fields: Vec<String> = line.split(';').map(str::to_owned).collect();
        let code = cp(&fields[0]);
        if fields[1].ends_with(", First>") {
            first = Some(code);
            continue;
        }
        let start = if fields[1].ends_with(", Last>") { first.take().expect("Last after First") } else { code };
        for c in start..=code {
            rows.push(UnicodeDataRow { cp: c, fields: fields.clone() });
        }
    }
    rows
}
