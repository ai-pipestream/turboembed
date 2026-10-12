//! Writes `tables/java.rs`: the data behind `java.lang.Character`'s
//! classes, derived from the UCD.
//!
//! - general category (`Character.getType`): `DerivedGeneralCategory.txt`,
//!   numbered as Java numbers it;
//! - `isLetter`: general category `L*`; `isDigit`: `Nd`;
//! - `isUpperCase`: `Lu` or `Other_Uppercase` (`PropList.txt`);
//! - `isAlphabetic`: `Alphabetic` (`DerivedCoreProperties.txt`);
//! - `isWhitespace`: `Zs`, `Zl` or `Zp` except U+00A0, U+2007 and U+202F,
//!   or U+0009..U+000D, or U+001C..U+001F;
//! - `toLowerCase(int)`: the simple lowercase mapping, field 13 of
//!   `UnicodeData.txt`;
//! - `Character.UnicodeScript`, for the scripts the UAX #29 word type
//!   needs: `Scripts.txt`.

use std::fmt::Write;
use std::path::Path;

use crate::rust_enum::{self, Variant, variant as v};
use crate::trie::{ByteTrie, write_rows};
use crate::ucd::{self, CODE_POINTS};

/// Java's general categories with `Character.getType`'s numbering as the
/// discriminant; 17 is unused in Java. The label is the UCD abbreviation.
const CATEGORIES: &[Variant] = &[
    v("Unassigned", 0, "CN", "Cn", "`Cn`: not assigned to a character (Java `UNASSIGNED`)."),
    v("UppercaseLetter", 1, "LU", "Lu", "`Lu`: upper case letter (Java `UPPERCASE_LETTER`)."),
    v("LowercaseLetter", 2, "LL", "Ll", "`Ll`: lower case letter (Java `LOWERCASE_LETTER`)."),
    v("TitlecaseLetter", 3, "LT", "Lt", "`Lt`: title case letter (Java `TITLECASE_LETTER`)."),
    v("ModifierLetter", 4, "LM", "Lm", "`Lm`: modifier letter (Java `MODIFIER_LETTER`)."),
    v("OtherLetter", 5, "LO", "Lo", "`Lo`: other letter (Java `OTHER_LETTER`)."),
    v("NonSpacingMark", 6, "MN", "Mn", "`Mn`: non-spacing mark (Java `NON_SPACING_MARK`)."),
    v("EnclosingMark", 7, "ME", "Me", "`Me`: enclosing mark (Java `ENCLOSING_MARK`)."),
    v("CombiningSpacingMark", 8, "MC", "Mc", "`Mc`: spacing mark (Java `COMBINING_SPACING_MARK`)."),
    v("DecimalDigitNumber", 9, "ND", "Nd", "`Nd`: decimal digit (Java `DECIMAL_DIGIT_NUMBER`)."),
    v("LetterNumber", 10, "NL", "Nl", "`Nl`: letter number (Java `LETTER_NUMBER`)."),
    v("OtherNumber", 11, "NO", "No", "`No`: other number (Java `OTHER_NUMBER`)."),
    v("SpaceSeparator", 12, "ZS", "Zs", "`Zs`: space separator (Java `SPACE_SEPARATOR`)."),
    v("LineSeparator", 13, "ZL", "Zl", "`Zl`: line separator (Java `LINE_SEPARATOR`)."),
    v("ParagraphSeparator", 14, "ZP", "Zp", "`Zp`: paragraph separator (Java `PARAGRAPH_SEPARATOR`)."),
    v("Control", 15, "CC", "Cc", "`Cc`: control character (Java `CONTROL`)."),
    v("Format", 16, "CF", "Cf", "`Cf`: format character (Java `FORMAT`)."),
    v("PrivateUse", 18, "CO", "Co", "`Co`: private use (Java `PRIVATE_USE`)."),
    v("Surrogate", 19, "CS", "Cs", "`Cs`: surrogate code unit (Java `SURROGATE`)."),
    v("DashPunctuation", 20, "PD", "Pd", "`Pd`: dash punctuation (Java `DASH_PUNCTUATION`)."),
    v("StartPunctuation", 21, "PS", "Ps", "`Ps`: opening punctuation (Java `START_PUNCTUATION`)."),
    v("EndPunctuation", 22, "PE", "Pe", "`Pe`: closing punctuation (Java `END_PUNCTUATION`)."),
    v("ConnectorPunctuation", 23, "PC", "Pc", "`Pc`: connector punctuation (Java `CONNECTOR_PUNCTUATION`)."),
    v("OtherPunctuation", 24, "PO", "Po", "`Po`: other punctuation (Java `OTHER_PUNCTUATION`)."),
    v("MathSymbol", 25, "SM", "Sm", "`Sm`: mathematical symbol (Java `MATH_SYMBOL`)."),
    v("CurrencySymbol", 26, "SC", "Sc", "`Sc`: currency symbol (Java `CURRENCY_SYMBOL`)."),
    v("ModifierSymbol", 27, "SK", "Sk", "`Sk`: modifier symbol (Java `MODIFIER_SYMBOL`)."),
    v("OtherSymbol", 28, "SO", "So", "`So`: other symbol (Java `OTHER_SYMBOL`)."),
    v("InitialQuotePunctuation", 29, "PI", "Pi", "`Pi`: initial quotation mark (Java `INITIAL_QUOTE_PUNCTUATION`)."),
    v("FinalQuotePunctuation", 30, "PF", "Pf", "`Pf`: final quotation mark (Java `FINAL_QUOTE_PUNCTUATION`)."),
];

/// The scripts the word type distinguishes; the label is the `Scripts.txt`
/// value. Every other script is `Other`.
const SCRIPTS: &[Variant] = &[
    v("Other", 0, "X", "Other", "Any other script, `Common`, `Inherited` and `Unknown` included."),
    v("Han", 1, "HANI", "Han", "Han ideographs."),
    v("Hiragana", 2, "HIRA", "Hiragana", "Hiragana."),
    v("Katakana", 3, "KANA", "Katakana", "Katakana."),
    v("Hangul", 4, "HANG", "Hangul", "Hangul."),
    v("Thai", 5, "THAI", "Thai", "Thai."),
    v("Lao", 6, "LAOO", "Lao", "Lao."),
    v("Myanmar", 7, "MYMR", "Myanmar", "Myanmar."),
    v("Khmer", 8, "KHMR", "Khmer", "Khmer."),
    v("TaiLe", 9, "TALE", "Tai_Le", "Tai Le."),
    v("NewTaiLue", 10, "TALU", "New_Tai_Lue", "New Tai Lue."),
    v("TaiViet", 11, "TAVT", "Tai_Viet", "Tai Viet."),
];

/// The flag bits; the generated `flag` module repeats them with docs.
const WHITESPACE: u8 = 0x01;
const ALPHABETIC: u8 = 0x02;
const UPPER_CASE: u8 = 0x04;
const LETTER: u8 = 0x08;
const DIGIT: u8 = 0x10;
const HAS_LOWER_CASE: u8 = 0x20;

/// Code points below this have a direct lower case table.
const LOWER_DIRECT_LEN: usize = 0x800;

/// The derived columns, one entry per code point.
struct Columns {
    category: Vec<u8>,
    flags: Vec<u8>,
    script: Vec<u8>,
    lower: Vec<u32>,
}

/// Reads the UCD files and writes `<out>/java.rs`.
pub fn generate(ucd_dir: &Path, out_dir: &Path) -> Result<(), String> {
    let general = ucd::read(ucd_dir, "DerivedGeneralCategory.txt")?;
    let core = ucd::read(ucd_dir, "DerivedCoreProperties.txt")?;
    let props = ucd::read(ucd_dir, "PropList.txt")?;
    let scripts = ucd::read(ucd_dir, "Scripts.txt")?;
    let data = ucd::read(ucd_dir, "UnicodeData.txt")?;
    let columns = derive(&general, &core, &props, &scripts, &data)?;
    let versions = [
        ("DerivedGeneralCategory.txt", ucd::version("DerivedGeneralCategory.txt", &general)?),
        ("DerivedCoreProperties.txt", ucd::version("DerivedCoreProperties.txt", &core)?),
        ("PropList.txt", ucd::version("PropList.txt", &props)?),
        ("Scripts.txt", ucd::version("Scripts.txt", &scripts)?),
    ];
    let source = write(&columns, &versions);
    let path = out_dir.join("java.rs");
    std::fs::write(&path, source).map_err(|error| format!("{}: {error}", path.display()))
}

fn derive(general: &str, core: &str, props: &str, scripts: &str, data: &str) -> Result<Columns, String> {
    let mut category = vec![0u8; CODE_POINTS];
    for range in ucd::ranges("DerivedGeneralCategory.txt", general)? {
        let value = rust_enum::by_label(CATEGORIES, range.value)
            .ok_or_else(|| format!("unknown general category {}", range.value))?;
        category[range.start as usize..=range.end as usize].fill(value);
    }

    let mut flags = vec![0u8; CODE_POINTS];
    ucd::set_flag("DerivedCoreProperties.txt", core, "Alphabetic", &mut flags, ALPHABETIC)?;
    ucd::set_flag("PropList.txt", props, "Other_Uppercase", &mut flags, UPPER_CASE)?;
    for (cp, flag) in flags.iter_mut().enumerate() {
        let value = category[cp];
        // Lu, Ll, Lt, Lm, Lo are 1..=5 in Java's numbering.
        if (1..=5).contains(&value) {
            *flag |= LETTER;
        }
        match value {
            1 => *flag |= UPPER_CASE,
            9 => *flag |= DIGIT,
            _ => {}
        }
        // Zs, Zl, Zp are 12, 13, 14; Java leaves out the no-break spaces.
        let separator = (12..=14).contains(&value) && !matches!(cp, 0x00A0 | 0x2007 | 0x202F);
        if separator || (0x09..=0x0D).contains(&cp) || (0x1C..=0x1F).contains(&cp) {
            *flag |= WHITESPACE;
        }
    }

    let mut script = vec![0u8; CODE_POINTS];
    for range in ucd::ranges("Scripts.txt", scripts)? {
        if let Some(value) = rust_enum::by_label(SCRIPTS, range.value) {
            script[range.start as usize..=range.end as usize].fill(value);
        }
    }

    let mut lower: Vec<u32> = (0..CODE_POINTS as u32).collect();
    for (cp, fields) in ucd::unicode_data(data)? {
        let mapping = fields[13];
        if !mapping.is_empty() {
            lower[cp as usize] = u32::from_str_radix(mapping, 16)
                .map_err(|_| format!("UnicodeData.txt: bad lower case mapping for U+{cp:04X}"))?;
        }
    }
    for (cp, flag) in flags.iter_mut().enumerate() {
        if lower[cp] != cp as u32 {
            *flag |= HAS_LOWER_CASE;
        }
    }
    if lower[..LOWER_DIRECT_LEN].iter().any(|&to| to > 0xFFFF) {
        return Err("a lower case mapping below U+0800 does not fit the direct table".to_string());
    }
    Ok(Columns { category, flags, script, lower })
}

fn write(columns: &Columns, versions: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(1 << 20);
    writeln!(out, "// Java character class tables for turbo-unicode.\n//").unwrap();
    writeln!(out, "// Written by turbo-unicode-gen (analysis/unicode-gen) from these files in").unwrap();
    writeln!(out, "// analysis/unicode/ucd; not edited by hand:\n//").unwrap();
    for (name, version) in versions {
        writeln!(out, "// - {name}, Unicode {version}").unwrap();
    }
    writeln!(out, "// - UnicodeData.txt (field 13, the simple lowercase mapping)\n//").unwrap();
    writeln!(out, "// The derivation of each Java class is documented in analysis/unicode/src/java.rs.\n").unwrap();

    write_category_enum(&mut out);
    write_script_enum(&mut out);
    write_flags_and_lookups(&mut out);

    let category = ByteTrie::build(&columns.category);
    writeln!(out, "/// The general category table: {} stored blocks.", category.stored_blocks()).unwrap();
    writeln!(out, "mod category {{\n    use super::GeneralCategory;\n").unwrap();
    rust_enum::write_aliases(&mut out, "GeneralCategory", CATEGORIES, &columns.category);
    category.write(&mut out, "GeneralCategory", &|value| rust_enum::by_value(CATEGORIES, value).alias.to_string());
    writeln!(out, "}}\n").unwrap();

    let flags = ByteTrie::build(&columns.flags);
    writeln!(out, "/// The flag table: {} stored blocks.", flags.stored_blocks()).unwrap();
    writeln!(out, "mod flag_table {{").unwrap();
    flags.write(&mut out, "u8", &|value| format!("0x{value:02X}"));
    writeln!(out, "}}\n").unwrap();

    let script = ByteTrie::build(&columns.script);
    writeln!(out, "/// The script table: {} stored blocks.", script.stored_blocks()).unwrap();
    writeln!(out, "mod script_table {{\n    use super::Script;\n").unwrap();
    rust_enum::write_aliases(&mut out, "Script", SCRIPTS, &columns.script);
    script.write(&mut out, "Script", &|value| rust_enum::by_value(SCRIPTS, value).alias.to_string());
    writeln!(out, "}}\n").unwrap();

    write_lower(&mut out, &columns.lower);
    out
}

fn write_category_enum(out: &mut String) {
    rust_enum::write_enum(
        out,
        "A general category as `Character.getType` returns it, with Java's\n\
         numbering as the discriminant (17 is unused in Java).",
        "GeneralCategory",
        CATEGORIES,
    );
    writeln!(out, "impl GeneralCategory {{").unwrap();
    rust_enum::write_all(out, "GeneralCategory", CATEGORIES);
    writeln!(out, "    /// The `Character.getType` value.\n    pub const fn java_type(self) -> u8 {{\n        self as u8\n    }}\n").unwrap();
    rust_enum::write_to_str(
        out,
        "The two-letter Unicode abbreviation, such as `Lu`.",
        "abbreviation",
        "GeneralCategory",
        CATEGORIES,
        &|v| v.label.to_string(),
    );
    writeln!(out, "    /// The category with this `Character.getType` value.").unwrap();
    writeln!(out, "    pub const fn from_java_type(value: u8) -> Option<GeneralCategory> {{\n        match value {{")
        .unwrap();
    for variant in CATEGORIES {
        writeln!(out, "            {} => Some(GeneralCategory::{}),", variant.value, variant.name).unwrap();
    }
    writeln!(out, "            _ => None,\n        }}\n    }}\n").unwrap();
    rust_enum::write_from_label(
        out,
        "The category with this two-letter Unicode abbreviation.",
        "from_abbreviation",
        "GeneralCategory",
        CATEGORIES,
    );
    writeln!(out, "}}\n").unwrap();
}

fn write_script_enum(out: &mut String) {
    rust_enum::write_enum(
        out,
        "The script of a code point, for the scripts the UAX #29 word type\n\
         distinguishes (Java's `Character.UnicodeScript` for these).",
        "Script",
        SCRIPTS,
    );
    writeln!(out, "impl Script {{").unwrap();
    rust_enum::write_all(out, "Script", SCRIPTS);
    rust_enum::write_to_str(
        out,
        "The `Character.UnicodeScript` constant name, such as `TAI_LE`; `OTHER` for `Other`.",
        "java_name",
        "Script",
        SCRIPTS,
        &|v| v.label.to_ascii_uppercase(),
    );
    rust_enum::write_from_label(
        out,
        "The script with this `Scripts.txt` value name, such as `Tai_Le`.",
        "from_property_name",
        "Script",
        SCRIPTS,
    );
    writeln!(out, "}}\n").unwrap();
}

fn write_flags_and_lookups(out: &mut String) {
    writeln!(
        out,
        "\
/// The bits of the flag table, one byte per code point.
pub(crate) mod flag {{
    /// `Character.isWhitespace(int)`.
    pub const WHITESPACE: u8 = 0x{WHITESPACE:02X};
    /// `Character.isAlphabetic(int)`.
    pub const ALPHABETIC: u8 = 0x{ALPHABETIC:02X};
    /// `Character.isUpperCase(int)`.
    pub const UPPER_CASE: u8 = 0x{UPPER_CASE:02X};
    /// `Character.isLetter(int)`.
    pub const LETTER: u8 = 0x{LETTER:02X};
    /// `Character.isDigit(int)`.
    pub const DIGIT: u8 = 0x{DIGIT:02X};
    /// `Character.toLowerCase(int)` returns a different code point.
    pub const HAS_LOWER_CASE: u8 = 0x{HAS_LOWER_CASE:02X};
}}

/// The general category of a code point.
#[inline]
pub(crate) fn general_category(cp: u32) -> GeneralCategory {{
    crate::trie::lookup(&category::INDEX, &category::BLOCKS, cp)
}}

/// The flag byte of a code point; see [`flag`].
#[inline]
pub(crate) fn flags(cp: u32) -> u8 {{
    crate::trie::lookup(&flag_table::INDEX, &flag_table::BLOCKS, cp)
}}

/// The script of a code point, as far as [`Script`] distinguishes them.
#[inline]
pub(crate) fn script(cp: u32) -> Script {{
    crate::trie::lookup(&script_table::INDEX, &script_table::BLOCKS, cp)
}}

/// The simple lowercase mapping: a direct table below U+{LOWER_DIRECT_LEN:04X}; above
/// it the flag table says whether the code point changes at all, and only
/// those that do are looked up in the sorted pairs.
#[inline]
pub(crate) fn to_lower_case(cp: u32) -> u32 {{
    if cp < {LOWER_DIRECT_LEN:#X} {{
        return u32::from(lower::DIRECT[cp as usize]);
    }}
    if flags(cp) & flag::HAS_LOWER_CASE == 0 {{
        return cp;
    }}
    match lower::FROM.binary_search(&cp) {{
        Ok(i) => lower::TO[i],
        Err(_) => cp,
    }}
}}
"
    )
    .unwrap();
}

fn write_lower(out: &mut String, lower: &[u32]) {
    let pairs: Vec<(u32, u32)> =
        (LOWER_DIRECT_LEN..CODE_POINTS).map(|cp| (cp as u32, lower[cp])).filter(|&(cp, to)| cp != to).collect();
    writeln!(out, "/// The simple lowercase mapping: a direct table, then the sorted pairs").unwrap();
    writeln!(out, "/// of the code points above it that change.\nmod lower {{").unwrap();
    writeln!(out, "    pub(super) static DIRECT: [u16; {LOWER_DIRECT_LEN:#X}] = [").unwrap();
    write_rows(out, "        ", lower[..LOWER_DIRECT_LEN].iter().map(|to| format!("0x{to:04X}")));
    writeln!(out, "    ];\n").unwrap();
    writeln!(out, "    pub(super) static FROM: [u32; {}] = [", pairs.len()).unwrap();
    write_rows(out, "        ", pairs.iter().map(|(from, _)| format!("0x{from:05X}")));
    writeln!(out, "    ];\n").unwrap();
    writeln!(out, "    pub(super) static TO: [u32; {}] = [", pairs.len()).unwrap();
    write_rows(out, "        ", pairs.iter().map(|(_, to)| format!("0x{to:05X}")));
    writeln!(out, "    ];\n}}").unwrap();
}
