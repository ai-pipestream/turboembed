//! The single code point folds: dashes, quotes, bullets, ellipsis and
//! invisible characters. The expectations are those of OpenNLP's
//! SetBasedNormalizerTest, UnicodeCharSequenceNormalizerTest and
//! AlignedNormalizerPipelineTest, with UTF-16 offsets written as UTF-8
//! byte offsets.

mod folds_support;

use folds_support::{covered, forward, per_char_edit, run_checked};
use turbo_normalize::tables::{
    BULLETS, CORNER_BRACKETS, DASHES, DOUBLE_QUOTES, ELLIPSES, INVISIBLE, MATH_MINUS, SINGLE_QUOTES, is_invisible,
    quote_fold,
};
use turbo_normalize::{Bullets, Chain, Dashes, Ellipsis, Invisible, Normalizer, Quotes, Whitespace, apply, normalize};

fn dash_fold(c: char) -> Option<&'static str> {
    (c != '-' && DASHES.contains(&c) && !MATH_MINUS.contains(&c)).then_some("-")
}

fn dash_fold_with_minus(c: char) -> Option<&'static str> {
    (c != '-' && DASHES.contains(&c)).then_some("-")
}

fn bullet_fold(c: char) -> Option<&'static str> {
    BULLETS.contains(&c).then_some(" ")
}

fn ellipsis_fold(c: char) -> Option<&'static str> {
    ELLIPSES.iter().find(|e| e.0 == c).map(|e| e.1)
}

fn invisible_edit(parent: &str, child: &str) -> bool {
    child.is_empty() && parent.chars().all(|c| INVISIBLE.contains(&c))
}

fn dashes(s: &str) -> String {
    run_checked(&Dashes::new(), s, per_char_edit(dash_fold))
}

fn quotes(s: &str) -> String {
    run_checked(&Quotes, s, per_char_edit(quote_fold))
}

fn bullets(s: &str) -> String {
    run_checked(&Bullets, s, per_char_edit(bullet_fold))
}

fn ellipsis(s: &str) -> String {
    run_checked(&Ellipsis, s, per_char_edit(ellipsis_fold))
}

fn invisible(s: &str) -> String {
    run_checked(&Invisible, s, invisible_edit)
}

#[test]
fn ids() {
    assert_eq!(Dashes::ID, "turbo:normalize:dashes");
    assert_eq!(Quotes::ID, "turbo:normalize:quotes");
    assert_eq!(Bullets::ID, "turbo:normalize:bullets");
    assert_eq!(Ellipsis::ID, "turbo:normalize:ellipsis");
    assert_eq!(Invisible::ID, "turbo:normalize:invisible");
    assert_ne!(Dashes::new().config_hash(), Dashes::with_math_minus().config_hash());
    assert!(Dashes::with_math_minus().folds_math_minus());
    assert!(!Dashes::default().folds_math_minus());
}

#[test]
fn empty_text_passes_through() {
    assert_eq!(dashes(""), "");
    assert_eq!(quotes(""), "");
    assert_eq!(bullets(""), "");
    assert_eq!(ellipsis(""), "");
    assert_eq!(invisible(""), "");
}

// UnicodeCharSequenceNormalizerTest.testDashFoldsUnicodeDashesButNotMathMinus
#[test]
fn dashes_fold_but_the_minus_sign_stays() {
    assert_eq!(dashes("a\u{2014}b"), "a-b");
    assert_eq!(dashes("5\u{2212}3"), "5\u{2212}3");
}

#[test]
fn every_default_dash_folds_and_the_minus_signs_fold_only_on_request() {
    for &d in DASHES {
        let src = format!("x{d}y");
        let expected = if MATH_MINUS.contains(&d) { src.clone() } else { "x-y".to_string() };
        assert_eq!(dashes(&src), expected, "{d:?}");
        assert_eq!(run_checked(&Dashes::with_math_minus(), &src, per_char_edit(dash_fold_with_minus)), "x-y");
    }
    // Hyphen-minus is a member but is never an edit.
    let (_, a) = normalize(&Dashes::new(), "well-known - yes");
    assert!(a.edits().is_empty());
    // Dash-like characters outside the table are left alone.
    assert_eq!(dashes("\u{2043}\u{2E3C}\u{1806}"), "\u{2043}\u{2E3C}-");
}

// AlignedNormalizerPipelineTest.dashFoldOfSupplementaryDashMapsSpanBackToOriginal
#[test]
fn a_supplementary_dash_maps_back_to_its_four_bytes() {
    let src = "a\u{10EAD}b";
    let (out, a) = normalize(&Dashes::new(), src);
    assert_eq!(out, "a-b");
    assert_eq!(covered(src, &a, 2, 3), "b");
    assert_eq!(covered(src, &a, 1, 2), "\u{10EAD}");
    assert_eq!(forward(&a, 5, 6), (2, 3));
}

// SetBasedNormalizerTest.testQuotesFoldSingleAndDouble
#[test]
fn quotes_fold_single_and_double() {
    assert_eq!(quotes("don\u{2019}t"), "don't");
    assert_eq!(quotes("\u{201C}hi\u{201D}"), "\"hi\"");
    assert_eq!(quotes("\u{AB}x\u{BB}"), "\"x\"");
    assert_eq!(quotes("\u{2039}y\u{203A}"), "'y'");
    assert_eq!(quotes("\u{FF07}"), "'");
    assert_eq!(quotes("\u{FF02}"), "\"");
    assert_eq!(quotes("\u{02BC}"), "'");
}

// SetBasedNormalizerTest.testQuotesLeaveAsciiAndNonQuotesAlone
#[test]
fn quotes_leave_ascii_and_other_characters_alone() {
    assert_eq!(quotes("'a' \"b\""), "'a' \"b\"");
    assert_eq!(quotes("abc"), "abc");
    assert_eq!(quotes("\u{2014}"), "\u{2014}");
    // Prime marks and the grave accent are not quotation marks here.
    assert_eq!(quotes("\u{2032}\u{2033}`\u{B4}"), "\u{2032}\u{2033}`\u{B4}");
}

// Addition to OpenNLP: U+2E42 has the Quotation_Mark property. The corner
// brackets have it too but are left alone.
#[test]
fn the_low_reversed_double_quote_folds_and_corner_brackets_do_not() {
    assert_eq!(quotes("\u{2E42}Wort\u{201C}"), "\"Wort\"");
    for &b in CORNER_BRACKETS {
        let s = format!("{b}x");
        assert_eq!(quotes(&s), s, "{b:?}");
    }
}

#[test]
fn every_quote_in_the_tables_folds() {
    for &q in SINGLE_QUOTES {
        assert_eq!(quotes(&format!("{q}s{q}")), "'s'", "{q:?}");
    }
    for &q in DOUBLE_QUOTES {
        assert_eq!(quotes(&format!("{q}s{q}")), "\"s\"", "{q:?}");
    }
}

// AlignedNormalizerPipelineTest.quoteFoldMapsSpanBackToOriginal
#[test]
fn a_folded_quote_maps_back_to_the_curly_quote() {
    let src = "\u{201C}hi\u{201D}";
    let (out, a) = normalize(&Quotes, src);
    assert_eq!(out, "\"hi\"");
    assert_eq!(covered(src, &a, 1, 3), "hi");
    assert_eq!(covered(src, &a, 0, 1), "\u{201C}");
}

#[test]
fn neighbouring_folds_share_one_edit() {
    // OpenNLP maps each output character to its own source character; the
    // builder here merges touching edits, so "''" from two curly quotes is
    // one edit and either output quote maps back to both.
    let src = "\u{2018}\u{2019}";
    let (out, a) = normalize(&Quotes, src);
    assert_eq!(out, "''");
    assert_eq!(a.edits().len(), 1);
    assert_eq!(covered(src, &a, 0, 1), src);
}

// SetBasedNormalizerTest.testBulletsBecomeSeparatorSpaces and testBulletsLeaveMiddleDotAndTextAlone
#[test]
fn bullets_become_spaces() {
    assert_eq!(bullets("\u{2022}item"), " item");
    assert_eq!(bullets("\u{25E6}item"), " item");
    assert_eq!(bullets("a\u{2043}b"), "a b");
    assert_eq!(bullets("\u{B7}"), "\u{B7}");
    assert_eq!(bullets("plain"), "plain");
    for &b in BULLETS {
        assert_eq!(bullets(&format!("{b}x")), " x", "{b:?}");
    }
    // Additions to OpenNLP; the heart bullets are ornaments and stay.
    assert_eq!(bullets("\u{204C}a\u{25D8}b\u{29BF}"), " a b ");
    assert_eq!(bullets("\u{2619}\u{2765}"), "\u{2619}\u{2765}");
}

// SetBasedNormalizerTest.testEllipsisExpandsToAsciiDots
#[test]
fn ellipsis_expands_to_dots() {
    assert_eq!(ellipsis("\u{2026}"), "...");
    assert_eq!(ellipsis("wait\u{2026}"), "wait...");
    assert_eq!(ellipsis("\u{2025}"), "..");
    assert_eq!(ellipsis("..."), "...");
    // U+22EF MIDLINE HORIZONTAL ELLIPSIS is a mathematical symbol with no
    // decomposition and is left alone.
    assert_eq!(ellipsis("\u{22EF}"), "\u{22EF}");
}

// Additions to OpenNLP: the one dot leader and the vertical presentation
// forms expand like their compatibility decompositions.
#[test]
fn leaders_and_vertical_forms_expand() {
    assert_eq!(ellipsis("a\u{2024}b"), "a.b");
    assert_eq!(ellipsis("\u{FE19}"), "...");
    assert_eq!(ellipsis("\u{FE30}"), "..");
}

// AlignedNormalizerPipelineTest.ellipsisExpansionMapsSpanBackToOriginal
#[test]
fn every_dot_of_an_expansion_maps_back_to_the_one_ellipsis() {
    let src = "a\u{2026}b";
    let (out, a) = normalize(&Ellipsis, src);
    assert_eq!(out, "a...b");
    assert_eq!(covered(src, &a, 4, 5), "b");
    assert_eq!(covered(src, &a, 1, 4), "\u{2026}");
    assert_eq!(covered(src, &a, 2, 3), "\u{2026}");
}

// SetBasedNormalizerTest.testInvisibleRemovesFormatAndBidiControls
#[test]
fn invisible_controls_are_deleted() {
    assert_eq!(invisible("a\u{FEFF}b"), "ab");
    assert_eq!(invisible("a\u{200B}b"), "ab");
    assert_eq!(invisible("a\u{2060}b"), "ab");
    assert_eq!(invisible("soft\u{AD}hyphen"), "softhyphen");
    assert_eq!(invisible("\u{202E}evil\u{202C}"), "evil");
    for &c in INVISIBLE {
        assert_eq!(invisible(&format!("x{c}y{c}")), "xy", "{c:?}");
    }
    // Addition to OpenNLP: the deprecated format characters.
    assert_eq!(invisible("a\u{206A}b\u{206F}"), "ab");
}

// SetBasedNormalizerTest.testInvisibleKeepsJoinersVariationSelectorsAndText
#[test]
fn joiners_and_variation_selectors_are_kept() {
    assert_eq!(invisible("a\u{200D}b"), "a\u{200D}b");
    assert_eq!(invisible("a\u{200C}b"), "a\u{200C}b");
    assert_eq!(invisible("\u{1F468}\u{200D}\u{1F469}"), "\u{1F468}\u{200D}\u{1F469}");
    assert_eq!(invisible("\u{2764}\u{FE0F}"), "\u{2764}\u{FE0F}");
    assert_eq!(invisible("hello"), "hello");
    assert!(!is_invisible('\u{200D}') && !is_invisible('\u{FE0F}') && !is_invisible('\u{180E}'));
    assert_eq!(invisible("\u{1820}\u{180E}\u{1821}"), "\u{1820}\u{180E}\u{1821}");
}

// AlignedNormalizerPipelineTest.invisibleStripMapsSpanBackAcrossDeletion
#[test]
fn a_deleted_character_is_not_covered_by_its_neighbour() {
    let src = "a\u{200B}b";
    let (out, a) = normalize(&Invisible, src);
    assert_eq!(out, "ab");
    assert_eq!(covered(src, &a, 1, 2), "b");
    // Neighbouring deletions are one edit.
    let (_, a) = normalize(&Invisible, "a\u{200E}\u{200F}\u{FEFF}b");
    assert_eq!(a.edits().len(), 1);
}

#[test]
fn unchanged_text_is_borrowed() {
    let src = "plain ASCII text - with 'quotes' and \"more\"...";
    assert!(matches!(apply(&Dashes::new(), src), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(apply(&Quotes, src), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(apply(&Bullets, src), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(apply(&Ellipsis, src), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(apply(&Invisible, src), std::borrow::Cow::Borrowed(_)));
}

// AlignedNormalizerPipelineTest.pipelineComposesStripInvisibleWhitespaceAndDashesBackToOriginal
// and pipelineMapsAnOriginalSpanForwardToTheNormalizedText
#[test]
fn invisible_whitespace_and_dashes_compose_in_a_chain() {
    let src = "a\u{200B}  b\u{2014}c";
    let chain = Chain::new().then(Invisible).then(Whitespace::collapse()).then(Dashes::new());
    let (out, a) = normalize(&chain, src);
    assert_eq!(out, "a b-c");
    assert_eq!(covered(src, &a, 2, 5), "b\u{2014}c");
    // 'b' is at UTF-16 index 4 in OpenNLP's test and byte 6 here.
    assert_eq!(forward(&a, 6, 7), (2, 3));
}

// AlignedNormalizerPipelineTest.lineBreakPreservingComposesInAnAlignedPipeline
// and paragraphPreservingComposesInAnAlignedPipeline
#[test]
fn break_preserving_whitespace_composes_in_a_chain() {
    let src = "a\u{200B}  b\n\nc\u{2014}d";
    for ws in [Whitespace::keep_line_breaks(), Whitespace::keep_paragraphs()] {
        let chain = Chain::new().then(Invisible).then(ws).then(Dashes::new());
        let (out, a) = normalize(&chain, src);
        assert_eq!(out, "a b\nc-d");
        assert_eq!(covered(src, &a, 4, 7), "c\u{2014}d");
    }
}

// AlignedNormalizerPipelineTest.substitutionFoldsComposeInAnAlignedPipeline
#[test]
fn quotes_and_ellipsis_compose_in_a_chain() {
    let src = "say \u{201C}hi\u{201D}\u{2026}";
    let chain = Chain::new().then(Quotes).then(Ellipsis);
    let (out, a) = normalize(&chain, src);
    assert_eq!(out, "say \"hi\"...");
    assert_eq!(covered(src, &a, 5, 7), "hi");
    // OpenNLP maps the "..." back to the ellipsis alone. The chain here
    // merges changed stretches that touch, so the closing quote and the
    // ellipsis next to it are one edit and the dots map back to both.
    assert_eq!(covered(src, &a, 8, 11), "\u{201D}\u{2026}");
    assert_eq!(covered(src, &a, 7, 8), "\u{201D}\u{2026}");
}
