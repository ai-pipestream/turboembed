//! Parsing `UnicodeData.txt`.
//!
//! Each line describes one code point in 15 fields; a pair of lines whose
//! names end in `, First>` and `, Last>` describes every code point of a
//! range (CJK ideographs, Hangul syllables, private use planes). Code
//! points the file does not list are unassigned.

use std::path::Path;

use crate::ucd::{CODE_POINTS, Record, UcdFile};

/// The general category values in the order of their numeric encoding in
/// the generated tables. The library's `GeneralCategory` enum uses the
/// same order.
pub const GENERAL_CATEGORIES: [&str; 30] = [
    "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm",
    "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
];

/// The encoding of `Cn`, the category of unassigned code points.
pub const UNASSIGNED: u8 = 29;

/// The encoding of `Nd`, decimal digits.
pub const DECIMAL_NUMBER: u8 = 8;

/// A decomposition mapping from field 5.
#[derive(Clone)]
pub struct Decomposition {
    /// True for a mapping with a `<tag>`: a compatibility decomposition.
    pub compatibility: bool,
    /// The code points the character decomposes to, one level deep.
    pub mapping: Vec<u32>,
}

/// What `UnicodeData.txt` says about one assigned code point.
#[derive(Clone)]
pub struct CharData {
    /// The general category, encoded as its index in [`GENERAL_CATEGORIES`].
    pub category: u8,
    /// The canonical combining class.
    pub combining_class: u8,
    /// The decomposition mapping, when the character has one.
    pub decomposition: Option<Decomposition>,
    /// The decimal digit value (field 6).
    pub decimal: Option<u8>,
    /// The simple uppercase mapping (field 12), when not the character itself.
    pub upper: Option<u32>,
    /// The simple lowercase mapping (field 13), when not the character itself.
    pub lower: Option<u32>,
}

/// `UnicodeData.txt`, one entry per code point.
pub struct UnicodeData {
    /// Indexed by code point; `None` for unassigned code points.
    pub chars: Vec<Option<CharData>>,
}

impl UnicodeData {
    /// Reads and parses `UnicodeData.txt` from `dir`.
    pub fn read(dir: &Path) -> UnicodeData {
        let file = UcdFile::read(dir, "UnicodeData.txt");
        let mut chars: Vec<Option<CharData>> = vec![None; CODE_POINTS];
        let mut range_start: Option<(u32, String)> = None;
        for record in file.records() {
            if record.field_count() != 15 {
                record.fail("expected 15 fields");
            }
            let cp = record.code_point(0);
            let name = record.field(1);
            let data = parse_char(&record);
            if let Some(prefix) = name.strip_suffix(", First>") {
                range_start = Some((cp, prefix.to_owned()));
                chars[cp as usize] = Some(data);
            } else if let Some(prefix) = name.strip_suffix(", Last>") {
                let Some((first, first_prefix)) = range_start.take() else { record.fail("range end without a start") };
                if first_prefix != prefix {
                    record.fail("range end does not match its start");
                }
                for c in first..=cp {
                    chars[c as usize] = Some(data.clone());
                }
            } else {
                if range_start.is_some() {
                    record.fail("range start without an end");
                }
                if chars[cp as usize].is_some() {
                    record.fail("code point listed twice");
                }
                chars[cp as usize] = Some(data);
            }
        }
        UnicodeData { chars }
    }

    /// The data of `cp`, `None` when it is unassigned.
    pub fn get(&self, cp: u32) -> Option<&CharData> {
        self.chars[cp as usize].as_ref()
    }

    /// The general category encoding of `cp`; unassigned code points are `Cn`.
    pub fn category(&self, cp: u32) -> u8 {
        self.get(cp).map_or(UNASSIGNED, |data| data.category)
    }

    /// The canonical combining class of `cp`; 0 for unassigned code points.
    pub fn combining_class(&self, cp: u32) -> u8 {
        self.get(cp).map_or(0, |data| data.combining_class)
    }

    /// The decomposition mapping of `cp`, when it has one.
    pub fn decomposition(&self, cp: u32) -> Option<&Decomposition> {
        self.get(cp).and_then(|data| data.decomposition.as_ref())
    }
}

fn parse_char(record: &Record<'_>) -> CharData {
    let category = record.field(2);
    let category = GENERAL_CATEGORIES
        .iter()
        .position(|&gc| gc == category)
        .unwrap_or_else(|| record.fail(&format!("unknown general category {category}"))) as u8;
    let combining_class =
        record.field(3).parse().unwrap_or_else(|_| record.fail("canonical combining class is not a number"));
    let decimal = match record.field(6) {
        "" => None,
        digit => Some(digit.parse().unwrap_or_else(|_| record.fail("decimal digit value is not a number"))),
    };
    let cp = record.code_point(0);
    let mapping = |i: usize| match record.field(i) {
        "" => None,
        _ => Some(record.code_point(i)).filter(|&mapped| mapped != cp),
    };
    CharData {
        category,
        combining_class,
        decomposition: parse_decomposition(record),
        decimal,
        upper: mapping(12),
        lower: mapping(13),
    }
}

fn parse_decomposition(record: &Record<'_>) -> Option<Decomposition> {
    let field = record.field(5);
    if field.is_empty() {
        return None;
    }
    let (compatibility, rest) = match field.strip_prefix('<') {
        Some(tagged) => match tagged.split_once('>') {
            Some((_, rest)) => (true, rest),
            None => record.fail("unterminated decomposition tag"),
        },
        None => (false, field),
    };
    let cps = rest.split_ascii_whitespace();
    let mapping: Vec<u32> = cps
        .map(|cp| match u32::from_str_radix(cp, 16) {
            Ok(value) if (4..=6).contains(&cp.len()) => value,
            _ => record.fail(&format!("`{cp}` is not a code point")),
        })
        .collect();
    if mapping.is_empty() {
        record.fail("empty decomposition");
    }
    Some(Decomposition { compatibility, mapping })
}
