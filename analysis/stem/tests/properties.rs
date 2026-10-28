//! Pure-Rust property tests for stemmers (no JVM goldens).
//!
//! Stemmers work in UTF-16 code units. Offsets into the source text must
//! still round-trip through UTF-16 and code-point tables after stemming,
//! so a view that replaces a word with its stem can map spans back.

use turbo_analysis::{Analyzer, Bound, Document, Key, Layer, OffsetTable, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_stem::{
    Algorithm, LightStemmer, NorwegianStandard, SnowballAlgorithm, SnowballStemmer, StemAnnotator, Stemmer, Stems,
};

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

fn space_tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0usize;
    for piece in text.split(' ') {
        let end = start + piece.len();
        spans.push(Span::new(start as u32, end as u32));
        start = end + 1;
    }
    Tokens { spans, sentence: None }
}

#[test]
fn every_stemmer_returns_valid_utf8_and_matches_borrow_equivalence() {
    let words = ["", "a", "running", "GENERALIZATIONS", "café", "re-enter", "o'clock", "1990s"];
    let mut out = Vec::new();
    for algorithm in sample_algorithms() {
        for w in words {
            out.clear();
            algorithm.stem_utf16(&to_utf16(w), &mut out);
            let via_str = algorithm.stem(w);
            let from_utf16 = String::from_utf16(&out).unwrap_or_else(|_| {
                panic!("{algorithm:?} {w}: stem_utf16 not UTF-16 scalar sequence: {out:?}");
            });
            assert_eq!(from_utf16, via_str, "{algorithm:?} {w}");
            assert!(from_utf16.is_char_boundary(from_utf16.len()));
        }
    }
}

#[test]
fn porter_never_grows_ascii_words_and_is_idempotent_after_two_passes() {
    let porter = Algorithm::Porter;
    let words = [
        "a",
        "run",
        "running",
        "generalizations",
        "ponies",
        "caresses",
        "relational",
        "allowed",
        "misc",
        "overhead",
        "xyz",
    ];
    let mut once = Vec::new();
    let mut twice = Vec::new();
    for w in words {
        let units = to_utf16(w);
        once.clear();
        porter.stem_utf16(&units, &mut once);
        assert!(once.len() <= units.len(), "porter grew {w}: {} -> {}", units.len(), once.len());
        twice.clear();
        porter.stem_utf16(&once, &mut twice);
        assert_eq!(once, twice, "porter not idempotent on {w}: {once:?} vs {twice:?}");
    }
}

#[test]
fn snowball_stemmers_never_panic_on_arbitrary_unicode_or_long_runs() {
    let probes = [
        "",
        "a",
        "\u{0}",
        "\u{FFFF}",
        "🐶🐱",
        "a\u{0301}",
        &"x".repeat(10_000),
        &"ß".repeat(1_000),
        "مرحبا",
        "Ελληνικά",
    ];
    let mut out = Vec::new();
    for algorithm in SnowballAlgorithm::ALL {
        let stemmer = SnowballStemmer::new(algorithm);
        for p in probes {
            out.clear();
            stemmer.stem_utf16(&to_utf16(p), &mut out);
            let _ = String::from_utf16_lossy(&out);
        }
        // Lone surrogate units must not panic.
        out.clear();
        stemmer.stem_utf16(&[0xD800, 0x0041, 0xDC00], &mut out);
    }
}

#[test]
fn light_stemmers_never_grow_short_ascii_and_leave_empty_intact() {
    let lights = [
        Algorithm::Light(LightStemmer::EnglishMinimal),
        Algorithm::Light(LightStemmer::FrenchLight),
        Algorithm::Light(LightStemmer::GermanMinimal),
        Algorithm::Light(LightStemmer::SpanishMinimal),
    ];
    let mut out = Vec::new();
    for algorithm in lights {
        for w in ["", "a", "ab", "the", "ing"] {
            let units = to_utf16(w);
            out.clear();
            algorithm.stem_utf16(&units, &mut out);
            if w.is_empty() {
                assert!(out.is_empty(), "{algorithm:?} empty -> {out:?}");
            }
            assert!(out.len() <= units.len(), "{algorithm:?} grew {w}: {} -> {}", units.len(), out.len());
        }
    }
}

#[test]
fn offset_tables_round_trip_on_source_and_stem_strings() {
    // OffsetTable round-trip on source words and their stems; not a stemmer property.
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

#[test]
fn stems_layer_check_aligns_to_tokens() {
    // Stems lives in analysis-layers (src/); assert from tests/ that each stem
    // equals the stemmer output for its token and Layer::check passes.
    let text = "the cats were running";
    let algorithm = Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English));
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(StemAnnotator::new(algorithm)))
        .build()
        .unwrap();
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, space_tokens(text), Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let doc = &docs[0];
    let tokens = doc.get::<Tokens>().unwrap();
    let stems = doc.get::<Stems>().unwrap();
    assert_eq!(stems.len(), tokens.spans.len());
    assert_eq!(stems.stem.len(), tokens.spans.len());
    stems.check().unwrap_or_else(|e| panic!("Stems::check: {e}"));
    for (i, span) in tokens.spans.iter() {
        let piece = &doc.source()[span.range()];
        let want = algorithm.stem(piece);
        let got = stems.stem.get(i.get());
        assert_eq!(got, want.as_str(), "token {piece:?}");
    }
}
