//! Holds the character tables of `turbo_normalize::tables` against the
//! Unicode Character Database files in `analysis/unicode/ucd/`, so a new
//! Unicode version that changes one of the properties a table follows
//! makes a test fail instead of going unnoticed.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbo_normalize::tables::{
    BULLETS, CORNER_BRACKETS, DASHES, DOUBLE_QUOTES, ELLIPSES, INVISIBLE, LINE_BREAKS, MATH_MINUS, SINGLE_QUOTES,
    WHITESPACE,
};

fn ucd(file: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "unicode", "ucd", file].iter().collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn hex(s: &str) -> u32 {
    u32::from_str_radix(s.trim(), 16).unwrap_or_else(|e| panic!("{s:?}: {e}"))
}

/// The code points of `property` in a file of `range ; property` lines.
fn property(file: &str, property: &str) -> BTreeSet<char> {
    let mut set = BTreeSet::new();
    for line in ucd(file).lines() {
        let line = line.split('#').next().unwrap_or("");
        let mut fields = line.split(';');
        let (Some(range), Some(name)) = (fields.next(), fields.next()) else { continue };
        if name.trim() != property {
            continue;
        }
        let (lo, hi) = match range.trim().split_once("..") {
            Some((lo, hi)) => (hex(lo), hex(hi)),
            None => (hex(range), hex(range)),
        };
        set.extend((lo..=hi).filter_map(char::from_u32));
    }
    set
}

/// One UnicodeData.txt row: name, general category, decomposition field.
fn unicode_data(c: char) -> Option<(String, String, String)> {
    let key = format!("{:04X};", c as u32);
    let data = ucd("UnicodeData.txt");
    let line = data.lines().find(|l| l.starts_with(&key))?;
    let f: Vec<&str> = line.split(';').collect();
    Some((f[1].to_string(), f[2].to_string(), f[5].to_string()))
}

fn set(table: &[char]) -> BTreeSet<char> {
    table.iter().copied().collect()
}

#[test]
fn whitespace_is_the_white_space_property() {
    assert_eq!(set(WHITESPACE), property("PropList.txt", "White_Space"));
    // The line breaks are the members in Cc with a vertical effect and the
    // line and paragraph separators (UAX #14 classes BK, CR, LF and NL).
    for &c in LINE_BREAKS {
        let (_, category, _) = unicode_data(c).unwrap();
        assert!(matches!(category.as_str(), "Cc" | "Zl" | "Zp"), "{c:?} {category}");
    }
}

#[test]
fn dashes_are_the_dash_property() {
    assert_eq!(set(DASHES), property("PropList.txt", "Dash"));
    for &c in DASHES {
        let (_, category, _) = unicode_data(c).unwrap();
        assert_eq!(category == "Sm", MATH_MINUS.contains(&c), "{c:?}");
    }
}

#[test]
fn quotes_are_the_quotation_mark_property_without_ascii_and_corner_brackets() {
    let mut expected = property("PropList.txt", "Quotation_Mark");
    expected.remove(&'"');
    expected.remove(&'\'');
    for c in CORNER_BRACKETS {
        assert!(expected.remove(c), "{c:?} is not a Quotation_Mark");
    }
    expected.insert('\u{02BC}');
    let folded: BTreeSet<char> = SINGLE_QUOTES.iter().chain(DOUBLE_QUOTES).copied().collect();
    assert_eq!(folded, expected);
    // Each mark goes to the ASCII mark its name says.
    for &c in SINGLE_QUOTES.iter().filter(|&&c| c != '\u{02BC}') {
        let (name, _, _) = unicode_data(c).unwrap();
        assert!(name.contains("SINGLE") || name.contains("APOSTROPHE"), "{c:?} {name}");
    }
    for &c in DOUBLE_QUOTES {
        let (name, _, _) = unicode_data(c).unwrap();
        assert!(name.contains("DOUBLE") || name == "FULLWIDTH QUOTATION MARK", "{c:?} {name}");
    }
}

#[test]
fn bullets_are_named_bullets() {
    for &c in BULLETS {
        let (name, _, _) = unicode_data(c).unwrap();
        assert!(name.contains("BULLET"), "{c:?} {name}");
    }
}

#[test]
fn ellipses_expand_to_their_compatibility_decomposition() {
    // Follows <compat> and <vertical> decompositions down to full stops.
    fn dots(c: char) -> String {
        if c == '.' {
            return ".".to_string();
        }
        let (_, _, decomposition) = unicode_data(c).unwrap();
        let parts: Vec<&str> = decomposition.split(' ').filter(|p| !p.starts_with('<')).collect();
        assert!(decomposition.starts_with('<'), "{c:?} has no compatibility decomposition");
        parts.iter().map(|p| dots(char::from_u32(hex(p)).unwrap())).collect()
    }
    for &(c, expansion) in ELLIPSES {
        assert_eq!(dots(c), expansion, "{c:?}");
    }
}

#[test]
fn invisible_characters_are_default_ignorable_format_characters() {
    let ignorable = property("DerivedCoreProperties.txt", "Default_Ignorable_Code_Point");
    for &c in INVISIBLE {
        assert!(ignorable.contains(&c), "{c:?} is not Default_Ignorable_Code_Point");
        let (_, category, _) = unicode_data(c).unwrap();
        assert_eq!(category, "Cf", "{c:?}");
    }
}
