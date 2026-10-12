//! Writes `tables/word_break.rs`: the UAX #29 `Word_Break` property from
//! `WordBreakProperty.txt` and `Extended_Pictographic` from
//! `emoji-data.txt`.

use std::fmt::Write;
use std::path::Path;

use crate::rust_enum::{self, Variant, variant as v};
use crate::trie::{BitTrie, ByteTrie};
use crate::ucd::{self, CODE_POINTS};

/// `Word_Break` values in the order of OpenNLP's `WordBreak` enum; the
/// label is the `WordBreakProperty.txt` value name.
const WORD_BREAKS: &[Variant] = &[
    v("Other", 0, "XX", "Other", "No `Word_Break` value: every code point the property file does not list."),
    v("Cr", 1, "CR", "CR", "Carriage return (U+000D)."),
    v("Lf", 2, "LF", "LF", "Line feed (U+000A)."),
    v("Newline", 3, "NL", "Newline", "The other mandatory breaks: U+000B, U+000C, U+0085, U+2028, U+2029."),
    v("Extend", 4, "EXT", "Extend", "Combining marks and other characters that extend the one before."),
    v("Zwj", 5, "ZWJ", "ZWJ", "Zero width joiner (U+200D)."),
    v("RegionalIndicator", 6, "RI", "Regional_Indicator", "Regional indicator symbols, used in pairs for flags."),
    v("Format", 7, "FO", "Format", "Format characters."),
    v("Katakana", 8, "KA", "Katakana", "Katakana."),
    v("HebrewLetter", 9, "HL", "Hebrew_Letter", "Hebrew letters."),
    v("ALetter", 10, "LE", "ALetter", "Alphabetic letters."),
    v("SingleQuote", 11, "SQ", "Single_Quote", "The apostrophe (U+0027)."),
    v("DoubleQuote", 12, "DQ", "Double_Quote", "The quotation mark (U+0022)."),
    v("MidNumLet", 13, "MB", "MidNumLet", "Characters that join letters or numbers, such as the full stop."),
    v("MidLetter", 14, "ML", "MidLetter", "Characters that join letters, such as the middle dot."),
    v("MidNum", 15, "MN", "MidNum", "Characters that join numbers, such as the comma."),
    v("Numeric", 16, "NU", "Numeric", "Digits."),
    v("ExtendNumLet", 17, "EX", "ExtendNumLet", "Characters that extend letters or numbers, such as the low line."),
    v("WSegSpace", 18, "WS", "WSegSpace", "Spaces that separate words."),
];

/// Reads the UCD files and writes `<out>/word_break.rs`.
pub fn generate(ucd_dir: &Path, out_dir: &Path) -> Result<(), String> {
    let word_break = ucd::read(ucd_dir, "WordBreakProperty.txt")?;
    let emoji = ucd::read(ucd_dir, "emoji-data.txt")?;

    let mut values = vec![0u8; CODE_POINTS];
    for range in ucd::ranges("WordBreakProperty.txt", &word_break)? {
        let value = rust_enum::by_label(WORD_BREAKS, range.value)
            .ok_or_else(|| format!("unknown Word_Break value {}", range.value))?;
        for cp in range.start..=range.end {
            if values[cp as usize] != 0 {
                return Err(format!("WordBreakProperty.txt gives U+{cp:04X} two values"));
            }
            values[cp as usize] = value;
        }
    }
    let mut pictographic = vec![0u8; CODE_POINTS];
    ucd::set_flag("emoji-data.txt", &emoji, "Extended_Pictographic", &mut pictographic, 1)?;
    let pictographic: Vec<bool> = pictographic.iter().map(|&bit| bit != 0).collect();

    let versions = [
        ("WordBreakProperty.txt", ucd::version("WordBreakProperty.txt", &word_break)?),
        ("emoji-data.txt", ucd::version("emoji-data.txt", &emoji)?),
    ];
    let source = write(&values, &pictographic, &versions);
    let path = out_dir.join("word_break.rs");
    std::fs::write(&path, source).map_err(|error| format!("{}: {error}", path.display()))
}

fn write(values: &[u8], pictographic: &[bool], versions: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(1 << 19);
    writeln!(out, "// UAX #29 property tables for turbo-unicode.\n//").unwrap();
    writeln!(out, "// Written by turbo-unicode-gen (analysis/unicode-gen) from these files in").unwrap();
    writeln!(out, "// analysis/unicode/ucd; not edited by hand:\n//").unwrap();
    for (name, version) in versions {
        writeln!(out, "// - {name}, Unicode {version}").unwrap();
    }
    writeln!(out).unwrap();

    rust_enum::write_enum(
        &mut out,
        "A UAX #29 `Word_Break` property value, in the order of OpenNLP's\n`WordBreak` enum.",
        "WordBreak",
        WORD_BREAKS,
    );
    writeln!(out, "impl WordBreak {{").unwrap();
    rust_enum::write_all(&mut out, "WordBreak", WORD_BREAKS);
    rust_enum::write_to_str(
        &mut out,
        "The value name as `WordBreakProperty.txt` writes it, such as `ALetter`.",
        "property_name",
        "WordBreak",
        WORD_BREAKS,
        &|v| v.label.to_string(),
    );
    rust_enum::write_from_label(
        &mut out,
        "The value with this `WordBreakProperty.txt` name.",
        "from_property_name",
        "WordBreak",
        WORD_BREAKS,
    );
    writeln!(out, "}}\n").unwrap();

    writeln!(
        out,
        "\
/// The `Word_Break` value of a code point.
#[inline]
pub(crate) fn word_break(cp: u32) -> WordBreak {{
    crate::trie::lookup(&word_break_table::INDEX, &word_break_table::BLOCKS, cp)
}}

/// Whether a code point has the `Extended_Pictographic` property.
#[inline]
pub(crate) fn is_extended_pictographic(cp: u32) -> bool {{
    crate::trie::lookup_bit(&pictographic_table::INDEX, &pictographic_table::BLOCKS, cp)
}}
"
    )
    .unwrap();

    let table = ByteTrie::build(values);
    writeln!(out, "/// The `Word_Break` table: {} stored blocks.", table.stored_blocks()).unwrap();
    writeln!(out, "mod word_break_table {{\n    use super::WordBreak;\n").unwrap();
    rust_enum::write_aliases(&mut out, "WordBreak", WORD_BREAKS, values);
    table.write(&mut out, "WordBreak", &|value| rust_enum::by_value(WORD_BREAKS, value).alias.to_string());
    writeln!(out, "}}\n").unwrap();

    let bits = BitTrie::build(pictographic);
    writeln!(out, "/// The `Extended_Pictographic` bits: {} stored blocks of 256 bits.", bits.stored_blocks()).unwrap();
    writeln!(out, "mod pictographic_table {{").unwrap();
    bits.write(&mut out);
    writeln!(out, "}}").unwrap();
    out
}
