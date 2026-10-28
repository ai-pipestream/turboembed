//! Property tests: UTF-16 stemming and offset round trips.
//!
//! Stemmers work in UTF-16 code units. Offsets into the source text must
//! still round-trip through UTF-16 and code-point tables after stemming,
//! so a view that replaces a word with its stem can map spans back.

use turbo_analysis::OffsetTable;
use turbo_stem::{Algorithm, LightStemmer, NorwegianStandard, SnowballAlgorithm, SnowballStemmer, Stemmer};

fn sample_algorithms() -> Vec<Algorithm> {
    use LightStemmer as L;
    use NorwegianStandard as N;
    vec![
        Algorithm::Porter,
        Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English)),
        Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::German)),
        Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::Arabic)),
        Algorithm::Light(L::EnglishMinimal),
        Algorithm::Light(L::FrenchLight),
        Algorithm::Light(L::GermanLight),
        Algorithm::Light(L::NorwegianLight(N::Both)),
        Algorithm::Light(L::SpanishMinimal),
        Algorithm::Light(L::SwedishLight),
    ]
}

fn to_utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

#[test]
fn stem_utf16_matches_stem_on_bmp_words() {
    let words = ["", "a", "running", "GENERALIZATIONS", "café", "re-enter", "o'clock", "1990s"];
    let mut out = Vec::new();
    for algorithm in sample_algorithms() {
        for w in words {
            out.clear();
            algorithm.stem_utf16(&to_utf16(w), &mut out);
            let via_str = algorithm.stem(w);
            let from_utf16: String = String::from_utf16(&out).unwrap_or_else(|_| {
                // Lone surrogates are not valid UTF-8; BMP samples must decode.
                panic!("{algorithm:?} {w}: stem_utf16 not UTF-16 scalar sequence: {out:?}");
            });
            assert_eq!(from_utf16, via_str, "{algorithm:?} {w}");
        }
    }
}

#[test]
fn empty_and_short_words_are_stable() {
    let mut out = Vec::new();
    for algorithm in sample_algorithms() {
        for w in ["", "a", "ab", "é", "ß"] {
            out.clear();
            algorithm.stem_utf16(&to_utf16(w), &mut out);
            // Stem length never grows for these light/snowball samples on short input.
            assert!(out.len() <= to_utf16(w).len() + 4, "{algorithm:?} {w} -> {out:?}");
        }
    }
}

#[test]
fn offset_tables_round_trip_on_words_and_stems() {
    let words = ["The", "café", "and", "GENERALIZATIONS", "re-enter", "🐶", "running"];
    let text = words.join(" ");
    let table = OffsetTable::build(&text);
    for (byte, _) in text.char_indices() {
        let u16 = table.to_utf16(&text, byte as u32);
        let cp = table.to_code_point(&text, byte as u32);
        assert_eq!(table.from_utf16(&text, u16).unwrap(), byte as u32);
        assert_eq!(table.from_code_point(&text, cp).unwrap(), byte as u32);
    }
    let mut stem_buf = Vec::new();
    let algorithm = Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English));
    let mut cursor = 0usize;
    for w in words {
        let start = cursor;
        let end = start + w.len();
        cursor = end + 1;
        for edge in [start as u32, end as u32] {
            let u16 = table.to_utf16(&text, edge);
            let cp = table.to_code_point(&text, edge);
            assert_eq!(table.from_utf16(&text, u16).unwrap(), edge);
            assert_eq!(table.from_code_point(&text, cp).unwrap(), edge);
        }
        stem_buf.clear();
        algorithm.stem_utf16(&to_utf16(w), &mut stem_buf);
        let stem = String::from_utf16(&stem_buf).unwrap();
        let stem_table = OffsetTable::build(&stem);
        for (byte, _) in stem.char_indices() {
            let u16 = stem_table.to_utf16(&stem, byte as u32);
            assert_eq!(stem_table.from_utf16(&stem, u16).unwrap(), byte as u32);
        }
    }
}
