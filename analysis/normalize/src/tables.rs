//! The fixed character tables the punctuation, whitespace and invisible
//! character normalizers work from.
//!
//! The tables start from their Apache OpenNLP sources (`UnicodeWhitespace`,
//! `UnicodeDash`, `QuoteCharSequenceNormalizer`,
//! `BulletCharSequenceNormalizer`, `EllipsisCharSequenceNormalizer`,
//! `InvisibleCharSequenceNormalizer`) and are checked against the Unicode
//! Character Database in `analysis/unicode/ucd/`:
//!
//! - [`WHITESPACE`] is exactly the `White_Space` property and
//!   [`LINE_BREAKS`] exactly the characters UAX #14 gives the mandatory
//!   break classes BK, CR, LF and NL. OpenNLP's lists are the same.
//! - [`DASHES`] is exactly the `Dash` property, as OpenNLP's list is.
//! - The quote tables are the `Quotation_Mark` property without the ASCII
//!   marks and without the CJK corner brackets, plus U+02BC MODIFIER LETTER
//!   APOSTROPHE. That adds U+2E42 to OpenNLP's list.
//! - [`BULLETS`] adds the other list bullets of the General Punctuation,
//!   Geometric Shapes and Miscellaneous Mathematical Symbols-B blocks to
//!   OpenNLP's five.
//! - [`ELLIPSES`] adds the one dot leader and the two vertical presentation
//!   forms to OpenNLP's two, each with its compatibility decomposition.
//! - [`INVISIBLE`] adds the six deprecated format characters U+206A to
//!   U+206F to OpenNLP's twenty.
//!
//! The tests in `tests/unicode_tables.rs` hold the statements about
//! Unicode properties and decompositions against the data files, and docs/normalizers.md lists what each addition
//! changes. Every list is sorted by code point, so it can be searched with
//! `binary_search` as well as read in order.

/// The 25 whitespace code points of OpenNLP's `UnicodeWhitespace` table:
/// the characters with the Unicode `White_Space` property, including the
/// no-break spaces U+00A0, U+2007 and U+202F (which `Character.isWhitespace`
/// excludes).
pub const WHITESPACE: &[char] = &whitespace_chars();

const fn whitespace_chars() -> [char; turbo_unicode::whitespace::UNICODE_WHITESPACE.len()] {
    let points = turbo_unicode::whitespace::UNICODE_WHITESPACE;
    let mut chars = ['\0'; turbo_unicode::whitespace::UNICODE_WHITESPACE.len()];
    let mut i = 0;
    while i < points.len() {
        chars[i] = match char::from_u32(points[i]) {
            Some(c) => c,
            None => panic!("whitespace must be a scalar value"),
        };
        i += 1;
    }
    chars
}

/// The members of [`WHITESPACE`] that force a line or paragraph break: the
/// entries OpenNLP marks `LINE_BREAK`.
pub const LINE_BREAKS: &[char] = &[
    '\u{000A}', // line feed (LF)
    '\u{000B}', // line tabulation (VT)
    '\u{000C}', // form feed (FF)
    '\u{000D}', // carriage return (CR)
    '\u{0085}', // next line (NEL)
    '\u{2028}', // line separator
    '\u{2029}', // paragraph separator
];

/// Whether `c` is in [`WHITESPACE`], using the shared Unicode mode.
#[inline]
pub fn is_whitespace(c: char) -> bool {
    turbo_unicode::whitespace::WhitespaceMode::Unicode.is_whitespace(c as u32)
}

/// Whether `c` is in [`LINE_BREAKS`].
#[inline]
pub fn is_line_break(c: char) -> bool {
    matches!(c, '\u{000A}'..='\u{000D}' | '\u{0085}' | '\u{2028}' | '\u{2029}')
}

/// Whether `c` is whitespace that does not break a line: a member of
/// [`WHITESPACE`] that is not in [`LINE_BREAKS`]. This is the indentation
/// the dehyphenator skips at the start of a continuation line.
#[inline]
pub fn is_horizontal_whitespace(c: char) -> bool {
    is_whitespace(c) && !is_line_break(c)
}

/// OpenNLP's `UnicodeDash` table: 31 dash and hyphen code points, three of
/// them mathematical minus signs (general category Sm), listed separately
/// in [`MATH_MINUS`].
pub const DASHES: &[char] = &[
    '\u{002D}',  // hyphen-minus
    '\u{058A}',  // armenian hyphen
    '\u{05BE}',  // hebrew punctuation maqaf
    '\u{1400}',  // canadian syllabics hyphen
    '\u{1806}',  // mongolian todo soft hyphen
    '\u{2010}',  // hyphen
    '\u{2011}',  // non-breaking hyphen
    '\u{2012}',  // figure dash
    '\u{2013}',  // en dash
    '\u{2014}',  // em dash
    '\u{2015}',  // horizontal bar
    '\u{2053}',  // swung dash (Po)
    '\u{207B}',  // superscript minus (Sm)
    '\u{208B}',  // subscript minus (Sm)
    '\u{2212}',  // minus sign (Sm)
    '\u{2E17}',  // double oblique hyphen
    '\u{2E1A}',  // hyphen with diaeresis
    '\u{2E3A}',  // two-em dash
    '\u{2E3B}',  // three-em dash
    '\u{2E40}',  // double hyphen
    '\u{2E5D}',  // oblique hyphen
    '\u{301C}',  // wave dash
    '\u{3030}',  // wavy dash
    '\u{30A0}',  // katakana-hiragana double hyphen
    '\u{FE31}',  // presentation form for vertical em dash
    '\u{FE32}',  // presentation form for vertical en dash
    '\u{FE58}',  // small em dash
    '\u{FE63}',  // small hyphen-minus
    '\u{FF0D}',  // fullwidth hyphen-minus
    '\u{10D6E}', // garay hyphen
    '\u{10EAD}', // yezidi hyphenation mark
];

/// The three entries of [`DASHES`] in general category Sm. OpenNLP's
/// default dash fold leaves them alone, so "5\u{2212}3" keeps its minus sign.
pub const MATH_MINUS: &[char] = &[
    '\u{207B}', // superscript minus
    '\u{208B}', // subscript minus
    '\u{2212}', // minus sign
];

/// Whether `c` is in OpenNLP's default dash set: [`DASHES`] without
/// [`MATH_MINUS`].
#[inline]
pub fn is_default_dash(c: char) -> bool {
    matches!(
        c,
        '\u{002D}' | '\u{058A}' | '\u{05BE}' | '\u{1400}' | '\u{1806}' | '\u{2010}'
            ..='\u{2015}'
                | '\u{2053}'
                | '\u{2E17}'
                | '\u{2E1A}'
                | '\u{2E3A}'
                | '\u{2E3B}'
                | '\u{2E40}'
                | '\u{2E5D}'
                | '\u{301C}'
                | '\u{3030}'
                | '\u{30A0}'
                | '\u{FE31}'
                | '\u{FE32}'
                | '\u{FE58}'
                | '\u{FE63}'
                | '\u{FF0D}'
                | '\u{10D6E}'
                | '\u{10EAD}'
    )
}

/// Whether `c` is in [`MATH_MINUS`].
#[inline]
pub fn is_math_minus(c: char) -> bool {
    matches!(c, '\u{207B}' | '\u{208B}' | '\u{2212}')
}

/// The single quotation marks and apostrophes the quote fold maps to
/// U+0027 APOSTROPHE: OpenNLP's eight. U+02BC is a letter, not a
/// `Quotation_Mark`, but is written as an apostrophe often enough that
/// OpenNLP folds it, and so does this table.
pub const SINGLE_QUOTES: &[char] = &[
    '\u{02BC}', // modifier letter apostrophe
    '\u{2018}', // left single quotation mark
    '\u{2019}', // right single quotation mark
    '\u{201A}', // single low-9 quotation mark
    '\u{201B}', // single high-reversed-9 quotation mark
    '\u{2039}', // single left-pointing angle quotation mark
    '\u{203A}', // single right-pointing angle quotation mark
    '\u{FF07}', // fullwidth apostrophe
];

/// The double quotation marks the quote fold maps to U+0022 QUOTATION
/// MARK: OpenNLP's ten and U+2E42, which has the `Quotation_Mark` property
/// and is missing from OpenNLP's list.
pub const DOUBLE_QUOTES: &[char] = &[
    '\u{00AB}', // left-pointing double angle quotation mark
    '\u{00BB}', // right-pointing double angle quotation mark
    '\u{201C}', // left double quotation mark
    '\u{201D}', // right double quotation mark
    '\u{201E}', // double low-9 quotation mark
    '\u{201F}', // double high-reversed-9 quotation mark
    '\u{2E42}', // double low-reversed-9 quotation mark (not in OpenNLP)
    '\u{301D}', // reversed double prime quotation mark
    '\u{301E}', // double prime quotation mark
    '\u{301F}', // low double prime quotation mark
    '\u{FF02}', // fullwidth quotation mark
];

/// The `Quotation_Mark` characters the quote fold leaves alone besides
/// the ASCII marks: the CJK corner brackets and their vertical and
/// halfwidth forms. They come in opening and closing pairs that also mark
/// titles and emphasis, and folding them to a symmetric ASCII mark would
/// lose which side of the pair a character is.
pub const CORNER_BRACKETS: &[char] = &[
    '\u{300C}', // left corner bracket
    '\u{300D}', // right corner bracket
    '\u{300E}', // left white corner bracket
    '\u{300F}', // right white corner bracket
    '\u{FE41}', // presentation form for vertical left corner bracket
    '\u{FE42}', // presentation form for vertical right corner bracket
    '\u{FE43}', // presentation form for vertical left white corner bracket
    '\u{FE44}', // presentation form for vertical right white corner bracket
    '\u{FF62}', // halfwidth left corner bracket
    '\u{FF63}', // halfwidth right corner bracket
];

/// The ASCII form of a quotation mark, or `None` when `c` is not in
/// [`SINGLE_QUOTES`] or [`DOUBLE_QUOTES`].
#[inline]
pub fn quote_fold(c: char) -> Option<&'static str> {
    match c {
        '\u{02BC}' | '\u{2018}'..='\u{201B}' | '\u{2039}' | '\u{203A}' | '\u{FF07}' => Some("'"),
        '\u{00AB}' | '\u{00BB}' | '\u{201C}'..='\u{201F}' | '\u{2E42}' | '\u{301D}'..='\u{301F}' | '\u{FF02}' => {
            Some("\"")
        }
        _ => None,
    }
}

/// The bullets the bullet fold maps to U+0020 SPACE: OpenNLP's five
/// (U+2022, U+2023, U+2043, U+2219, U+25E6) and the five other characters
/// whose Unicode name calls them a bullet and which serve as list
/// markers. The heart and floral bullets of the Dingbats and
/// Miscellaneous Symbols blocks are ornaments and are not included.
/// U+00B7 MIDDLE DOT is not a bullet here: Catalan writes it inside words.
pub const BULLETS: &[char] = &[
    '\u{2022}', // bullet
    '\u{2023}', // triangular bullet
    '\u{2043}', // hyphen bullet
    '\u{204C}', // black leftwards bullet (not in OpenNLP)
    '\u{204D}', // black rightwards bullet (not in OpenNLP)
    '\u{2219}', // bullet operator
    '\u{25D8}', // inverse bullet (not in OpenNLP)
    '\u{25E6}', // white bullet
    '\u{29BE}', // circled white bullet (not in OpenNLP)
    '\u{29BF}', // circled bullet (not in OpenNLP)
];

/// Whether `c` is in [`BULLETS`].
#[inline]
pub fn is_bullet(c: char) -> bool {
    matches!(
        c,
        '\u{2022}'
            | '\u{2023}'
            | '\u{2043}'
            | '\u{204C}'
            | '\u{204D}'
            | '\u{2219}'
            | '\u{25D8}'
            | '\u{25E6}'
            | '\u{29BE}'
            | '\u{29BF}'
    )
}

/// The leaders and ellipses the ellipsis fold expands, with their ASCII
/// forms: each character's compatibility decomposition (NFKC) written in
/// full stops. OpenNLP expands U+2025 and U+2026 only.
pub const ELLIPSES: &[(char, &str)] = &[
    ('\u{2024}', "."),   // one dot leader (not in OpenNLP)
    ('\u{2025}', ".."),  // two dot leader
    ('\u{2026}', "..."), // horizontal ellipsis
    ('\u{FE19}', "..."), // presentation form for vertical horizontal ellipsis (not in OpenNLP)
    ('\u{FE30}', ".."),  // presentation form for vertical two dot leader (not in OpenNLP)
];

/// The ASCII form of a leader or ellipsis in [`ELLIPSES`].
#[inline]
pub fn ellipsis_fold(c: char) -> Option<&'static str> {
    match c {
        '\u{2024}' => Some("."),
        '\u{2025}' | '\u{FE30}' => Some(".."),
        '\u{2026}' | '\u{FE19}' => Some("..."),
        _ => None,
    }
}

/// The code points the invisible character fold deletes: OpenNLP's 20
/// (the soft hyphen, zero width space, word joiner and invisible
/// operators, the byte order mark, and the bidirectional marks,
/// embeddings, overrides and isolates) and the six deprecated format
/// characters U+206A to U+206F, which have had no effect on display since
/// Unicode 3.0. U+200C ZERO WIDTH NON-JOINER, U+200D ZERO WIDTH JOINER,
/// U+180E MONGOLIAN VOWEL SEPARATOR and the variation selectors are not
/// here: they change how text is shaped or rendered, and ZWJ holds emoji
/// sequences together.
pub const INVISIBLE: &[char] = &[
    '\u{00AD}', // soft hyphen
    '\u{061C}', // arabic letter mark
    '\u{200B}', // zero width space
    '\u{200E}', // left-to-right mark
    '\u{200F}', // right-to-left mark
    '\u{202A}', // left-to-right embedding
    '\u{202B}', // right-to-left embedding
    '\u{202C}', // pop directional formatting
    '\u{202D}', // left-to-right override
    '\u{202E}', // right-to-left override
    '\u{2060}', // word joiner
    '\u{2061}', // function application
    '\u{2062}', // invisible times
    '\u{2063}', // invisible separator
    '\u{2064}', // invisible plus
    '\u{2066}', // left-to-right isolate
    '\u{2067}', // right-to-left isolate
    '\u{2068}', // first strong isolate
    '\u{2069}', // pop directional isolate
    '\u{206A}', // inhibit symmetric swapping (deprecated, not in OpenNLP)
    '\u{206B}', // activate symmetric swapping (deprecated, not in OpenNLP)
    '\u{206C}', // inhibit arabic form shaping (deprecated, not in OpenNLP)
    '\u{206D}', // activate arabic form shaping (deprecated, not in OpenNLP)
    '\u{206E}', // national digit shapes (deprecated, not in OpenNLP)
    '\u{206F}', // nominal digit shapes (deprecated, not in OpenNLP)
    '\u{FEFF}', // zero width no-break space (byte order mark)
];

/// Whether `c` is in [`INVISIBLE`].
#[inline]
pub fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{061C}'
            | '\u{200B}'
            | '\u{200E}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_char() -> impl Iterator<Item = char> {
        (0..=0x10FFFFu32).filter_map(char::from_u32)
    }

    fn assert_sorted(table: &[char]) {
        assert!(table.windows(2).all(|w| w[0] < w[1]), "{table:?} is not sorted and free of duplicates");
    }

    #[test]
    fn tables_are_sorted_and_have_the_opennlp_sizes() {
        for table in [
            WHITESPACE,
            LINE_BREAKS,
            DASHES,
            MATH_MINUS,
            SINGLE_QUOTES,
            DOUBLE_QUOTES,
            CORNER_BRACKETS,
            BULLETS,
            INVISIBLE,
        ] {
            assert_sorted(table);
        }
        assert_eq!(WHITESPACE.len(), 25);
        assert_eq!(LINE_BREAKS.len(), 7);
        assert_eq!(DASHES.len(), 31);
        assert_eq!(MATH_MINUS.len(), 3);
        assert_eq!(SINGLE_QUOTES.len(), 8);
        assert_eq!(DOUBLE_QUOTES.len(), 11);
        assert_eq!(BULLETS.len(), 10);
        assert_eq!(INVISIBLE.len(), 26);
        assert!(ELLIPSES.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(LINE_BREAKS.iter().all(|c| WHITESPACE.contains(c)));
        assert!(MATH_MINUS.iter().all(|c| DASHES.contains(c)));
    }

    #[test]
    fn predicates_agree_with_the_lists_on_every_code_point() {
        for c in every_char() {
            assert_eq!(is_whitespace(c), WHITESPACE.binary_search(&c).is_ok(), "{c:?}");
            assert_eq!(is_line_break(c), LINE_BREAKS.binary_search(&c).is_ok(), "{c:?}");
            assert_eq!(is_horizontal_whitespace(c), is_whitespace(c) && !LINE_BREAKS.contains(&c), "{c:?}");
            let dash = DASHES.binary_search(&c).is_ok();
            let minus = MATH_MINUS.binary_search(&c).is_ok();
            assert_eq!(is_math_minus(c), minus, "{c:?}");
            assert_eq!(is_default_dash(c), dash && !minus, "{c:?}");
            assert_eq!(is_invisible(c), INVISIBLE.binary_search(&c).is_ok(), "{c:?}");
            assert_eq!(is_bullet(c), BULLETS.binary_search(&c).is_ok(), "{c:?}");
            let quote = if SINGLE_QUOTES.contains(&c) {
                Some("'")
            } else if DOUBLE_QUOTES.contains(&c) {
                Some("\"")
            } else {
                None
            };
            assert_eq!(quote_fold(c), quote, "{c:?}");
            assert_eq!(ellipsis_fold(c), ELLIPSES.iter().find(|e| e.0 == c).map(|e| e.1), "{c:?}");
        }
    }

    /// Times the two ways of testing membership in the 25-entry whitespace
    /// table over 1 MB of text that mixes ASCII, Latin-1, CJK and the
    /// whitespace members. Run with
    /// `cargo test --release -p turbo-normalize whitespace_membership_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn whitespace_membership_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        let pieces =
            ["The quick brown fox ", "\u{00E9}t\u{00E9} ", "\u{4E2D}\u{6587}\u{3000}", "a\u{00A0}b\t", "line\r\n"];
        let mut text = String::new();
        let mut k = 0;
        while text.len() < 1 << 20 {
            text.push_str(pieces[k % pieces.len()]);
            k += 1;
        }
        fn run(text: &str, name: &str, f: impl Fn(char) -> bool) {
            let mut best = u128::MAX;
            let mut hits = 0usize;
            for _ in 0..20 {
                let start = Instant::now();
                hits = black_box(text).chars().filter(|&c| f(black_box(c))).count();
                best = best.min(start.elapsed().as_micros());
            }
            println!("{name}: {best} us per MB ({hits} members)");
        }
        run(&text, "match", is_whitespace);
        run(&text, "binary_search", |c| WHITESPACE.binary_search(&c).is_ok());
    }
}
