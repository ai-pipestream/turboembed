//! Java's casing of a correction: `String.toUpperCase(Locale.ROOT)` for an
//! all-uppercase token, `Character.toUpperCase` on the first code point for
//! a capitalized one, on this crate's Unicode 18.0 data.

use turbo_unicode::{java, props};

/// The code points whose uppercase is more than one character, with that
/// uppercase: the unconditional entries of SpecialCasing.txt (Unicode
/// 18.0.0) whose upper field has two or more code points, in code point
/// order. `String.toUpperCase(Locale.ROOT)` applies exactly these (the
/// conditional entries are for Lithuanian, Turkish and Azeri); every other
/// character takes its simple mapping. The rows were extracted from
/// `analysis/unicode/ucd/SpecialCasing.txt`, and the unit test
/// `special_uppercase_matches_special_casing_txt` compares them with it.
static SPECIAL_UPPER: [(u32, &str); 103] = [
    (0x00DF, "\u{0053}\u{0053}"),
    (0x0149, "\u{02BC}\u{004E}"),
    (0x01F0, "\u{004A}\u{030C}"),
    (0x0390, "\u{0399}\u{0308}\u{0301}"),
    (0x03B0, "\u{03A5}\u{0308}\u{0301}"),
    (0x0587, "\u{0535}\u{0552}"),
    (0x1E96, "\u{0048}\u{0331}"),
    (0x1E97, "\u{0054}\u{0308}"),
    (0x1E98, "\u{0057}\u{030A}"),
    (0x1E99, "\u{0059}\u{030A}"),
    (0x1E9A, "\u{0041}\u{02BE}"),
    (0x1F50, "\u{03A5}\u{0313}"),
    (0x1F52, "\u{03A5}\u{0313}\u{0300}"),
    (0x1F54, "\u{03A5}\u{0313}\u{0301}"),
    (0x1F56, "\u{03A5}\u{0313}\u{0342}"),
    (0x1F80, "\u{1F08}\u{0399}"),
    (0x1F81, "\u{1F09}\u{0399}"),
    (0x1F82, "\u{1F0A}\u{0399}"),
    (0x1F83, "\u{1F0B}\u{0399}"),
    (0x1F84, "\u{1F0C}\u{0399}"),
    (0x1F85, "\u{1F0D}\u{0399}"),
    (0x1F86, "\u{1F0E}\u{0399}"),
    (0x1F87, "\u{1F0F}\u{0399}"),
    (0x1F88, "\u{1F08}\u{0399}"),
    (0x1F89, "\u{1F09}\u{0399}"),
    (0x1F8A, "\u{1F0A}\u{0399}"),
    (0x1F8B, "\u{1F0B}\u{0399}"),
    (0x1F8C, "\u{1F0C}\u{0399}"),
    (0x1F8D, "\u{1F0D}\u{0399}"),
    (0x1F8E, "\u{1F0E}\u{0399}"),
    (0x1F8F, "\u{1F0F}\u{0399}"),
    (0x1F90, "\u{1F28}\u{0399}"),
    (0x1F91, "\u{1F29}\u{0399}"),
    (0x1F92, "\u{1F2A}\u{0399}"),
    (0x1F93, "\u{1F2B}\u{0399}"),
    (0x1F94, "\u{1F2C}\u{0399}"),
    (0x1F95, "\u{1F2D}\u{0399}"),
    (0x1F96, "\u{1F2E}\u{0399}"),
    (0x1F97, "\u{1F2F}\u{0399}"),
    (0x1F98, "\u{1F28}\u{0399}"),
    (0x1F99, "\u{1F29}\u{0399}"),
    (0x1F9A, "\u{1F2A}\u{0399}"),
    (0x1F9B, "\u{1F2B}\u{0399}"),
    (0x1F9C, "\u{1F2C}\u{0399}"),
    (0x1F9D, "\u{1F2D}\u{0399}"),
    (0x1F9E, "\u{1F2E}\u{0399}"),
    (0x1F9F, "\u{1F2F}\u{0399}"),
    (0x1FA0, "\u{1F68}\u{0399}"),
    (0x1FA1, "\u{1F69}\u{0399}"),
    (0x1FA2, "\u{1F6A}\u{0399}"),
    (0x1FA3, "\u{1F6B}\u{0399}"),
    (0x1FA4, "\u{1F6C}\u{0399}"),
    (0x1FA5, "\u{1F6D}\u{0399}"),
    (0x1FA6, "\u{1F6E}\u{0399}"),
    (0x1FA7, "\u{1F6F}\u{0399}"),
    (0x1FA8, "\u{1F68}\u{0399}"),
    (0x1FA9, "\u{1F69}\u{0399}"),
    (0x1FAA, "\u{1F6A}\u{0399}"),
    (0x1FAB, "\u{1F6B}\u{0399}"),
    (0x1FAC, "\u{1F6C}\u{0399}"),
    (0x1FAD, "\u{1F6D}\u{0399}"),
    (0x1FAE, "\u{1F6E}\u{0399}"),
    (0x1FAF, "\u{1F6F}\u{0399}"),
    (0x1FB2, "\u{1FBA}\u{0399}"),
    (0x1FB3, "\u{0391}\u{0399}"),
    (0x1FB4, "\u{0386}\u{0399}"),
    (0x1FB6, "\u{0391}\u{0342}"),
    (0x1FB7, "\u{0391}\u{0342}\u{0399}"),
    (0x1FBC, "\u{0391}\u{0399}"),
    (0x1FC2, "\u{1FCA}\u{0399}"),
    (0x1FC3, "\u{0397}\u{0399}"),
    (0x1FC4, "\u{0389}\u{0399}"),
    (0x1FC6, "\u{0397}\u{0342}"),
    (0x1FC7, "\u{0397}\u{0342}\u{0399}"),
    (0x1FCC, "\u{0397}\u{0399}"),
    (0x1FD2, "\u{0399}\u{0308}\u{0300}"),
    (0x1FD3, "\u{0399}\u{0308}\u{0301}"),
    (0x1FD6, "\u{0399}\u{0342}"),
    (0x1FD7, "\u{0399}\u{0308}\u{0342}"),
    (0x1FE2, "\u{03A5}\u{0308}\u{0300}"),
    (0x1FE3, "\u{03A5}\u{0308}\u{0301}"),
    (0x1FE4, "\u{03A1}\u{0313}"),
    (0x1FE6, "\u{03A5}\u{0342}"),
    (0x1FE7, "\u{03A5}\u{0308}\u{0342}"),
    (0x1FF2, "\u{1FFA}\u{0399}"),
    (0x1FF3, "\u{03A9}\u{0399}"),
    (0x1FF4, "\u{038F}\u{0399}"),
    (0x1FF6, "\u{03A9}\u{0342}"),
    (0x1FF7, "\u{03A9}\u{0342}\u{0399}"),
    (0x1FFC, "\u{03A9}\u{0399}"),
    (0xFB00, "\u{0046}\u{0046}"),
    (0xFB01, "\u{0046}\u{0049}"),
    (0xFB02, "\u{0046}\u{004C}"),
    (0xFB03, "\u{0046}\u{0046}\u{0049}"),
    (0xFB04, "\u{0046}\u{0046}\u{004C}"),
    (0xFB05, "\u{0053}\u{0054}"),
    (0xFB06, "\u{0053}\u{0054}"),
    (0xFB13, "\u{0544}\u{0546}"),
    (0xFB14, "\u{0544}\u{0535}"),
    (0xFB15, "\u{0544}\u{053B}"),
    (0xFB16, "\u{054E}\u{0546}"),
    (0xFB17, "\u{0544}\u{053D}"),
    (0x1DF95, "\u{0053}\u{0053}"),
];

/// Appends `String.toUpperCase(Locale.ROOT)` of `s` to `out`.
pub(crate) fn to_uppercase(s: &str, out: &mut String) {
    for c in s.chars() {
        match SPECIAL_UPPER.binary_search_by_key(&(c as u32), |&(k, _)| k) {
            Ok(i) => out.push_str(SPECIAL_UPPER[i].1),
            Err(_) => out.push(java::to_upper_char(c)),
        }
    }
}

/// `Character.isUpperCase`: general category Lu or the Other_Uppercase
/// property, which is the Uppercase property.
pub(crate) fn is_upper(c: char) -> bool {
    props::uppercase(c)
}

/// Whether `s` has more than one code point, a letter, and no letter that
/// is not uppercase: OpenNLP's test for an all-uppercase token.
fn is_all_upper(s: &str) -> bool {
    let mut letter = false;
    for c in s.chars() {
        if java::is_letter(c) {
            letter = true;
            if !is_upper(c) {
                return false;
            }
        }
    }
    letter && s.chars().nth(1).is_some()
}

/// Writes `corrected` to `out` with the capitalization of `original`, as
/// OpenNLP's `applyCasing` does: all uppercase for an all-uppercase
/// original, the first code point uppercased for an original that starts
/// with an uppercase code point, the suggestion as it is otherwise.
pub(crate) fn apply(original: &str, corrected: &str, out: &mut String) {
    let Some(initial) = corrected.chars().next() else { return };
    if is_all_upper(original) {
        to_uppercase(corrected, out);
    } else if original.chars().next().is_some_and(is_upper) {
        out.push(java::to_upper_char(initial));
        out.push_str(&corrected[initial.len_utf8()..]);
    } else {
        out.push_str(corrected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upper(s: &str) -> String {
        let mut out = String::new();
        to_uppercase(s, &mut out);
        out
    }

    #[test]
    fn special_uppercase_matches_special_casing_txt() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../unicode/ucd/SpecialCasing.txt");
        let text = std::fs::read_to_string(path).expect("SpecialCasing.txt");
        let mut rows = Vec::new();
        for line in text.lines() {
            let data = line.split('#').next().unwrap_or("");
            let fields: Vec<&str> = data.split(';').map(str::trim).collect();
            // code; lower; title; upper; (condition;) comment
            if fields.len() < 5 || !fields[4].is_empty() {
                continue;
            }
            let upper: Vec<char> = fields[3]
                .split_whitespace()
                .map(|h| char::from_u32(u32::from_str_radix(h, 16).expect("hex")).expect("scalar"))
                .collect();
            if upper.len() > 1 {
                rows.push((u32::from_str_radix(fields[0], 16).expect("hex"), upper.into_iter().collect::<String>()));
            }
        }
        rows.sort();
        let table: Vec<(u32, String)> = SPECIAL_UPPER.iter().map(|&(k, v)| (k, v.to_owned())).collect();
        assert_eq!(table, rows);
    }

    #[test]
    fn uppercase_applies_special_casing_and_simple_mappings() {
        assert_eq!(upper("straße"), "STRASSE");
        assert_eq!(upper("ﬁx"), "FIX");
        assert_eq!(upper("ŉ"), "\u{2BC}N");
        assert_eq!(upper("\u{10428}word"), "\u{10400}WORD");
        // Final sigma and dotless i take their simple mappings.
        assert_eq!(upper("ςıi"), "ΣII");
    }

    #[test]
    fn casing_follows_the_original() {
        let cased = |o: &str, c: &str| {
            let mut out = String::new();
            apply(o, c, &mut out);
            out
        };
        assert_eq!(cased("WROLD", "world"), "WORLD");
        assert_eq!(cased("Wrold", "world"), "World");
        assert_eq!(cased("wrold", "world"), "world");
        // One code point is never all uppercase, but starts with one.
        assert_eq!(cased("A", "ab"), "Ab");
        // Digits do not count as letters either way.
        assert_eq!(cased("W0RLD", "world"), "WORLD");
        assert_eq!(cased("12", "ab"), "ab");
        assert_eq!(cased("Wrold", ""), "");
    }
}
