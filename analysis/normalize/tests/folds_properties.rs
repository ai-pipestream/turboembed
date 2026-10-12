//! Generated inputs for the whitespace and single code point folds: each
//! output is compared with a slow reference written here independently of
//! the crate (character vectors and string building, following OpenNLP's
//! two-pass collapse then trim), and every alignment is checked.

mod folds_support;

use folds_support::{Rng, check_alignment, per_char_edit};
use turbo_normalize::tables::{
    BULLETS, DASHES, DOUBLE_QUOTES, ELLIPSES, INVISIBLE, LINE_BREAKS, MATH_MINUS, SINGLE_QUOTES, WHITESPACE,
};
use turbo_normalize::{
    Bullets, Dashes, Ellipsis, Invisible, Normalizer, Quotes, Whitespace, WhitespaceMode, normalize,
};

const ROUNDS: u64 = 3000;

/// Pieces the generated texts are built from: every whitespace member,
/// CR LF, letters of one to four bytes, hyphens, dashes, quotes, bullets,
/// ellipses, invisible characters and digits.
fn pool() -> Vec<String> {
    let mut pool: Vec<String> = WHITESPACE.iter().map(|c| c.to_string()).collect();
    for s in [
        "\r\n",
        "a",
        "Z",
        "word",
        "\u{FC}",
        "\u{4E2D}",
        "\u{10400}",
        "-",
        "\u{AD}",
        "\u{2010}",
        "\u{2011}",
        "\u{2212}",
        "\u{2014}",
        "\u{10EAD}",
        "'",
        "\"",
        "\u{2019}",
        "\u{201C}",
        "\u{AB}",
        "\u{2022}",
        "\u{B7}",
        "\u{2026}",
        "\u{2025}",
        "\u{2024}",
        "\u{FE19}",
        "\u{2E42}",
        "\u{300C}",
        "\u{204C}",
        "\u{206A}",
        "\u{200B}",
        "\u{200D}",
        "\u{FEFF}",
        "\u{202E}",
        "\u{663}",
        "\u{FF15}",
        "\u{1D7CE}",
        "7",
        "\u{1F600}",
    ] {
        pool.push(s.to_string());
    }
    pool
}

fn texts(seed: u64) -> impl Iterator<Item = String> {
    let pool = pool();
    let mut rng = Rng::new(seed);
    (0..ROUNDS).map(move |_| {
        let refs: Vec<&str> = pool.iter().map(String::as_str).collect();
        rng.text(&refs, 24)
    })
}

/// OpenNLP's whitespace normalizers as written there: collapse every run
/// to one character, then trim members from both ends of the result.
fn reference_whitespace(src: &str, mode: WhitespaceMode) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut collapsed: Vec<char> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !WHITESPACE.contains(&chars[i]) {
            collapsed.push(chars[i]);
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && WHITESPACE.contains(&chars[j]) {
            j += 1;
        }
        let mut breaks = 0;
        let mut k = i;
        while k < j {
            if LINE_BREAKS.contains(&chars[k]) {
                breaks += 1;
                if chars[k] == '\r' && k + 1 < j && chars[k + 1] == '\n' {
                    k += 1;
                }
            }
            k += 1;
        }
        let newline = match mode {
            WhitespaceMode::Collapse => false,
            WhitespaceMode::KeepLineBreaks => breaks >= 1,
            WhitespaceMode::KeepParagraphs => breaks >= 2,
        };
        collapsed.push(if newline { '\n' } else { ' ' });
        i = j;
    }
    let start = collapsed.iter().position(|c| !WHITESPACE.contains(c)).unwrap_or(collapsed.len());
    let end = collapsed.iter().rposition(|c| !WHITESPACE.contains(c)).map_or(start, |p| p + 1);
    collapsed[start..end].iter().collect()
}

/// A per character fold applied to every character.
fn reference_fold(src: &str, fold: impl Fn(char) -> Option<&'static str>) -> String {
    let mut out = String::new();
    for c in src.chars() {
        match fold(c) {
            Some(s) => out.push_str(s),
            None => out.push(c),
        }
    }
    out
}

fn check_fold<N: Normalizer>(n: &N, seed: u64, fold: impl Fn(char) -> Option<&'static str> + Copy) {
    for src in texts(seed) {
        let (out, a) = normalize(n, &src);
        assert_eq!(out, reference_fold(&src, fold), "{} on {src:?}", N::ID);
        check_alignment(&src, &out, &a, per_char_edit(fold));
    }
}

#[test]
fn whitespace_matches_the_two_pass_reference() {
    for (seed, mode) in
        [(1, WhitespaceMode::Collapse), (2, WhitespaceMode::KeepLineBreaks), (3, WhitespaceMode::KeepParagraphs)]
    {
        let n = Whitespace::new(mode);
        for src in texts(seed) {
            let (out, a) = normalize(&n, &src);
            assert_eq!(out, reference_whitespace(&src, mode), "{mode:?} on {src:?}");
            check_alignment(&src, &out, &a, |p, c| {
                p.chars().all(|ch| WHITESPACE.contains(&ch)) && matches!(c, "" | " " | "\n")
            });
            // Normalizing again changes nothing, except in KeepParagraphs:
            // its paragraph boundary is a single "\n", which a second run
            // reads as one line break and turns into a space.
            if mode != WhitespaceMode::KeepParagraphs {
                let (again, b) = normalize(&n, &out);
                assert_eq!(again, out);
                assert!(b.edits().is_empty(), "{mode:?} is not idempotent on {src:?}");
            }
        }
    }
}

#[test]
fn dashes_match_the_reference() {
    check_fold(&Dashes::new(), 4, |c| (c != '-' && DASHES.contains(&c) && !MATH_MINUS.contains(&c)).then_some("-"));
    check_fold(&Dashes::with_math_minus(), 5, |c| (c != '-' && DASHES.contains(&c)).then_some("-"));
}

#[test]
fn quotes_match_the_reference() {
    check_fold(&Quotes, 6, |c| {
        if SINGLE_QUOTES.contains(&c) {
            Some("'")
        } else if DOUBLE_QUOTES.contains(&c) {
            Some("\"")
        } else {
            None
        }
    });
}

#[test]
fn bullets_match_the_reference() {
    check_fold(&Bullets, 7, |c| BULLETS.contains(&c).then_some(" "));
}

#[test]
fn ellipsis_matches_the_reference() {
    check_fold(&Ellipsis, 8, |c| ELLIPSES.iter().find(|e| e.0 == c).map(|e| e.1));
}

#[test]
fn invisible_matches_the_reference() {
    for src in texts(9) {
        let (out, a) = normalize(&Invisible, &src);
        let expected: String = src.chars().filter(|c| !INVISIBLE.contains(c)).collect();
        assert_eq!(out, expected, "{src:?}");
        check_alignment(&src, &out, &a, |p, c| c.is_empty() && p.chars().all(|ch| INVISIBLE.contains(&ch)));
    }
}
