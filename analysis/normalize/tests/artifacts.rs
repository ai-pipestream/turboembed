//! The artifact detector. The expectations of OpenNLP's text-artifacts
//! tests (CursorArtifactDetectorTest, ArtifactBoundaryTest,
//! ArtifactEncodingTest, ArtifactFalsePositiveTest,
//! ArtifactUnicodeContextTest, ArtifactsManualExampleTest,
//! ArtifactAnnotatorTest) come first, with OpenNLP's UTF-16 offsets
//! written as UTF-8 byte offsets. Their unpaired-surrogate cases cannot
//! occur: a Rust `str` holds no unpaired surrogate. Then the two kinds
//! added here, and property loops over generated text.

#[path = "artifacts/additions.rs"]
mod additions;
#[path = "artifacts/annotator.rs"]
mod annotator;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Layer, Provenance, Span, View};
use turbo_normalize::{Artifact, ArtifactDetector, KindSet, RepeatedLines};
use turbo_normalize_layers::ArtifactKind::{self, *};
use turbo_normalize_layers::Artifacts;

/// A string from code points.
fn cp(points: &[u32]) -> String {
    points.iter().map(|&p| char::from_u32(p).expect("a scalar value")).collect()
}

/// ASCII written as Unicode tag characters, optionally with CANCEL TAG.
fn tags(ascii: &str, cancel: bool) -> String {
    let mut s: String = ascii.chars().map(|c| char::from_u32(0xE0000 + c as u32).unwrap()).collect();
    if cancel {
        s.push('\u{E007F}');
    }
    s
}

fn detect(text: &str) -> Vec<Artifact> {
    ArtifactDetector::default().detect(text)
}

fn only(kinds: &[ArtifactKind], text: &str) -> Vec<Artifact> {
    ArtifactDetector::builder().only(kinds).build().detect(text)
}

fn a(start: usize, end: usize, kind: ArtifactKind) -> Artifact {
    Artifact { span: Span::new(start as u32, end as u32), kind }
}

/// The findings as (covered text, kind).
fn covered<'t>(text: &'t str, found: &[Artifact]) -> Vec<(&'t str, ArtifactKind)> {
    found.iter().map(|f| (&text[f.span.range()], f.kind)).collect()
}

/// UTF-8 bytes read as ISO-8859-1.
fn latin1(s: &str) -> String {
    s.bytes().map(char::from).collect()
}

/// Windows-1252 as the JDK decodes it: the five undefined bytes become
/// U+FFFD. Written out here independently of the detector's table.
fn windows_1252_decode(bytes: &[u8]) -> String {
    const HIGH: [u32; 32] = [
        0x20AC, 0xFFFD, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0xFFFD,
        0x017D, 0xFFFD, 0xFFFD, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A,
        0x0153, 0xFFFD, 0x017E, 0x0178,
    ];
    bytes
        .iter()
        .map(|&b| {
            if (0x80..0xA0).contains(&b) { char::from_u32(HIGH[usize::from(b - 0x80)]).unwrap() } else { char::from(b) }
        })
        .collect()
}

/// Adds the layer to a document over `text`, which runs the layer's own
/// check and the document's bounds and boundary checks.
fn check_layer(text: &str, detector: &ArtifactDetector) -> Artifacts {
    let mut doc = Document::new(text);
    let view = doc.get::<View>().unwrap().clone();
    let (layer,) = detector.annotate((&view,)).unwrap();
    layer.check().unwrap();
    let copy = Artifacts { spans: layer.spans.clone(), kind: layer.kind.clone() };
    doc.add(&Key::<Artifacts>::DEFAULT, copy, Provenance::by("test")).unwrap();
    layer
}

// ---------------------------------------------------------------------------
// CursorArtifactDetectorTest

// testFlagsTheClassWithExactSpan
#[test]
fn flags_each_class_with_its_exact_span() {
    let cases: [(String, ArtifactKind, String); 13] = [
        (format!("bad {} decode", cp(&[0xFFFD, 0xFFFD])), ReplacementChar, cp(&[0xFFFD, 0xFFFD])),
        (format!("nul{}byte", cp(&[0])), Control, cp(&[0])),
        (format!("esc{}[31m", cp(&[0x1B])), Control, cp(&[0x1B])),
        (format!("internal {} sentinel", cp(&[0xFDD0])), Noncharacter, cp(&[0xFDD0])),
        (format!("plane end {}", cp(&[0xFFFE])), Noncharacter, cp(&[0xFFFE])),
        (format!("icon font {} glyphs", cp(&[0xE000, 0xF8FF])), PrivateUse, cp(&[0xE000, 0xF8FF])),
        (format!("plane 16 {} glyph", cp(&[0x100000])), PrivateUse, cp(&[0x100000])),
        (format!("override {}txet here", cp(&[0x202E])), BidiControl, cp(&[0x202E])),
        (format!("isolate {}x", cp(&[0x2068])), BidiControl, cp(&[0x2068])),
        (format!("stuffed{}text", cp(&[0x200B; 3])), ZeroWidth, cp(&[0x200B; 3])),
        (format!("{}leading bom", cp(&[0xFEFF])), ZeroWidth, cp(&[0xFEFF])),
        (format!("don{}t", cp(&[0xE2, 0x20AC, 0x2122])), Mojibake, cp(&[0xE2, 0x20AC, 0x2122])),
        (format!("caf{}", cp(&[0xC3, 0xA9])), Mojibake, cp(&[0xC3, 0xA9])),
    ];
    for (text, kind, expected) in &cases {
        let found = detect(text);
        assert_eq!(covered(text, &found), vec![(expected.as_str(), *kind)], "{text:?}");
    }
}

// testCleanTextYieldsNothing
#[test]
fn clean_text_yields_nothing() {
    let cases = [
        "plain ASCII text, with punctuation.".to_string(),
        format!("d{}j{} vu {} caf{} na{}ve", cp(&[0xE9]), cp(&[0xE0]), cp(&[0x2014]), cp(&[0xE9]), cp(&[0xEF])),
        format!("Jo{}o n{}o {} s{}", cp(&[0xE3]), cp(&[0xE3]), cp(&[0xE9]), cp(&[0xF3])),
        "tabs\tand\nnewlines\r\nare whitespace, not controls".to_string(),
        cp(&[0x0628, 0x200C, 0x0628]),
        format!("family: {}", cp(&[0x1F469, 0x200D, 0x1F469, 0x200D, 0x1F466])),
        format!("pict {}", cp(&[0x2764, 0xFE0F, 0x200D, 0x1F525])),
        format!("han {} text", cp(&[0x23BB4])),
    ];
    for text in &cases {
        assert_eq!(detect(text), vec![], "{text:?}");
    }
}

// testUnpairedSurrogates and testUnpairedHighSurrogateAtEndOfText cannot
// occur in a `str`; the kind exists for layers read from OpenNLP output.
#[test]
fn unpaired_surrogate_kind_is_named_but_never_produced() {
    assert_eq!(UnpairedSurrogate.name(), "unpaired-surrogate");
    assert!(KindSet::OPENNLP.contains(UnpairedSurrogate));
    assert_eq!(only(&[UnpairedSurrogate], "x\u{D7FF}\u{E000}y"), vec![]);
}

// testSingleZeroWidthSpaceBetweenLettersIsOrthographic
#[test]
fn single_zero_width_space_between_thai_letters_is_kept() {
    let thai = cp(&[0x0E01, 0x0E23, 0x0E38, 0x0E07, 0x200B, 0x0E40, 0x0E17, 0x0E1E]);
    assert_eq!(detect(&thai), vec![]);
}

// testZeroWidthOutsideLetterContextIsFlagged
#[test]
fn zero_width_after_punctuation_is_flagged() {
    let text = format!("end.{} next", cp(&[0x200B]));
    assert_eq!(detect(&text), vec![a(4, 7, ZeroWidth)]);
}

// testJoinerBetweenLettersIsOrthographic
#[test]
fn joiner_between_persian_letters_is_kept() {
    assert_eq!(detect(&cp(&[0x0645, 0x06CC, 0x200C, 0x062E, 0x0648])), vec![]);
}

// testMultipleFindingsInOrder
#[test]
fn multiple_findings_in_order() {
    let text = format!("a{}b caf{} d{}e", cp(&[7]), cp(&[0xC3, 0xA9]), cp(&[0xFFFD]));
    assert_eq!(detect(&text), vec![a(1, 2, Control), a(7, 11, Mojibake), a(13, 16, ReplacementChar)]);
}

// testMojibakeRunsAreSeparatedByAscii
#[test]
fn ascii_separates_mojibake_runs() {
    let damaged = cp(&[0xC3, 0xA9]);
    let text = format!("{damaged}t{damaged} chez papa");
    assert_eq!(detect(&text), vec![a(0, 4, Mojibake), a(5, 9, Mojibake)]);
}

// testLatin1C1Mojibake
#[test]
fn latin1_c1_mojibake() {
    for damaged in [cp(&[0xC2, 0x80]), cp(&[0xE2, 0x82, 0xAC])] {
        let text = format!("value {damaged} end");
        assert_eq!(covered(&text, &detect(&text)), vec![(damaged.as_str(), Mojibake)]);
    }
}

// testUnicodeTagRunIsFlagged
#[test]
fn unicode_tag_run_is_flagged() {
    let hidden = tags("secret", true);
    let text = format!("visible {hidden} text");
    assert_eq!(covered(&text, &detect(&text)), vec![(hidden.as_str(), UnicodeTag)]);
}

// testWholeUnicodeTagsBlockIsClassified
#[test]
fn whole_tags_block_is_classified() {
    let hidden = cp(&[0xE0000, 0xE0001, 0xE0010, 0xE007F]);
    assert_eq!(detect(&hidden), vec![a(0, hidden.len(), UnicodeTag)]);
}

// testEmojiTagFlagIsOrthographic
#[test]
fn england_flag_is_kept() {
    assert_eq!(detect(&format!("\u{1F3F4}{}", tags("gbeng", true))), vec![]);
}

// testMalformedEmojiTagFlagIsFlagged
#[test]
fn unterminated_subdivision_flag_is_flagged() {
    let hidden = tags("gbeng", false);
    let text = format!("\u{1F3F4}{hidden}");
    assert_eq!(covered(&text, &detect(&text)), vec![(hidden.as_str(), UnicodeTag)]);
}

// testInvalidUtf8ImagesAreNotMojibake
#[test]
fn invalid_utf8_images_are_not_mojibake() {
    for text in
        [format!("x {} y", cp(&[0xC0, 0x20AC])), format!("copyright {} sign", cp(&[0xA9])), cp(&[0xED, 0xA0, 0x20AC])]
    {
        assert_eq!(detect(&text), vec![], "{text:?}");
    }
}

// testAdjacentDifferentTypeRunsAbut
#[test]
fn adjacent_runs_of_different_kinds_abut() {
    let text = format!("x{}y", cp(&[7, 0xFFFD]));
    assert_eq!(detect(&text), vec![a(1, 2, Control), a(2, 5, ReplacementChar)]);
}

// testRunEndingExactlyAtTextLength
#[test]
fn run_ending_at_the_end_of_the_text() {
    let text = format!("ok{}", cp(&[0xFFFD, 0xFFFD]));
    assert_eq!(detect(&text), vec![a(2, text.len(), ReplacementChar)]);
}

// testSpansAreSortedAndNeverOverlap
#[test]
fn mixed_findings_are_sorted_and_disjoint() {
    let text =
        format!("a.{}b{} caf{} d{}", cp(&[0x200B, 0x200B]), cp(&[7, 0x1B]), cp(&[0xC3, 0xA9]), cp(&[0xFFFD, 0xFDD0]));
    let found = detect(&text);
    assert_eq!(found.len(), 5, "{found:?}");
    assert!(found.windows(2).all(|w| w[0].span.end <= w[1].span.start));
}

// testCharSequenceInputMatchesStringInput: a `String` and a `&str` are one
// type here; testRejectsNull has no counterpart.

// testEmptyTextYieldsNothing
#[test]
fn empty_text_yields_nothing() {
    assert_eq!(detect(""), vec![]);
}

// testTypeLimitedDetector
#[test]
fn type_limited_detector_reports_only_its_kinds() {
    let text = format!("control {} caf{} {}", cp(&[7]), cp(&[0xC3, 0xA9]), tags("secret", true));
    let found = only(&[Mojibake], &text);
    assert_eq!(found.iter().map(|f| f.kind).collect::<Vec<_>>(), vec![Mojibake]);
}

// testTypeLimitedDetectorRejectsInvalidSets: an unknown kind cannot be
// written, and an empty set is a detector that reports nothing.
#[test]
fn empty_kind_set_reports_nothing() {
    assert_eq!(only(&[], &format!("{}{}", cp(&[0, 0xFFFD, 0xC3, 0xA9]), tags("x", true))), vec![]);
}

// testTypeLimitedDetectorDefensivelyCopiesSet: the builder takes the kinds
// by value; changing the slice afterwards cannot reach the detector.
#[test]
fn builder_copies_the_kinds() {
    let mut kinds = vec![Control];
    let detector = ArtifactDetector::builder().only(&kinds).build();
    kinds.push(Mojibake);
    let text = format!("{} {}", cp(&[7]), cp(&[0xC3, 0xA9]));
    assert_eq!(detector.detect(&text), vec![a(0, 1, Control)]);
}

// ---------------------------------------------------------------------------
// ArtifactBoundaryTest

// testMojibakeDoesNotDropFollowingControls
#[test]
fn mojibake_keeps_the_controls_after_it() {
    for damage in [cp(&[0xC2, 0x80]), cp(&[0xE2, 0x80, 0x99]), cp(&[0xF0, 0x9F, 0x98, 0x80])] {
        for controls in [cp(&[0]), cp(&[1, 2]), cp(&[0x7F])] {
            let prefix = format!("{} ", cp(&[0x1F642]));
            let text = format!("{prefix}{damage}{controls} end");
            let start = prefix.len();
            let end = start + damage.len();
            assert_eq!(detect(&text), vec![a(start, end, Mojibake), a(end, end + controls.len(), Control)], "{text:?}");
        }
    }
}

// testControlsBetweenMojibakeRuns
#[test]
fn controls_between_mojibake_runs() {
    let damage = cp(&[0xC2, 0x80]);
    let text = format!("{damage}{}{damage}{}{}", cp(&[0]), cp(&[1]), cp(&[0xFFFD]));
    assert_eq!(
        detect(&text),
        vec![a(0, 4, Mojibake), a(4, 5, Control), a(5, 9, Mojibake), a(9, 10, Control), a(10, 13, ReplacementChar)]
    );
}

// testSelectedTypesKeepTheirOwnSpans
#[test]
fn selected_kinds_keep_their_own_spans() {
    let text = cp(&[0xC2, 0x80, 0]);
    assert_eq!(only(&[Mojibake], &text), vec![a(0, 4, Mojibake)]);
    assert_eq!(only(&[Control], &text), vec![a(2, 5, Control)]);
}

// ---------------------------------------------------------------------------
// ArtifactEncodingTest

// testValidUtf8Boundaries
#[test]
fn valid_utf8_at_every_length_boundary_is_mojibake() {
    for p in [0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF] {
        let damage = latin1(&cp(&[p]));
        assert_eq!(only(&[Mojibake], &damage), vec![a(0, damage.len(), Mojibake)], "U+{p:04X}");
    }
}

// testWindows1252Mappings
#[test]
fn windows_1252_images_of_two_byte_sequences() {
    for continuation in 0x80..=0xBFu8 {
        let damage = windows_1252_decode(&[0xC2, continuation]);
        let expected = if damage.contains('\u{FFFD}') { vec![] } else { vec![a(0, damage.len(), Mojibake)] };
        assert_eq!(only(&[Mojibake], &damage), expected, "C2 {continuation:02X}");
    }
}

// testInvalidUtf8Boundaries
#[test]
fn invalid_utf8_is_not_mojibake() {
    let cases: [(&str, &[u32]); 13] = [
        ("overlong two-byte form", &[0xC0, 0x80]),
        ("invalid two-byte lead", &[0xC1, 0xBF]),
        ("overlong three-byte form", &[0xE0, 0x9F, 0xBF]),
        ("high surrogate", &[0xED, 0xA0, 0x80]),
        ("low surrogate", &[0xED, 0xBF, 0xBF]),
        ("overlong four-byte form", &[0xF0, 0x8F, 0xBF, 0xBF]),
        ("beyond Unicode maximum", &[0xF4, 0x90, 0x80, 0x80]),
        ("invalid four-byte lead", &[0xF5, 0x80, 0x80, 0x80]),
        ("invalid leading byte", &[0xFF, 0x80]),
        ("bare continuation", &[0x80]),
        ("truncated two-byte form", &[0xC2]),
        ("truncated three-byte form", &[0xE2, 0x82]),
        ("truncated four-byte form", &[0xF0, 0x9F, 0x98]),
    ];
    for (reason, points) in cases {
        assert_eq!(only(&[Mojibake], &cp(points)), vec![], "{reason}");
    }
}

// testConcurrentDetection
#[test]
fn one_detector_shared_between_threads() {
    let detector = ArtifactDetector::default();
    let text = cp(&[0xC2, 0x80, 0, 0xFFFD]);
    let expected = vec![a(0, 4, Mojibake), a(4, 5, Control), a(5, 8, ReplacementChar)];
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..32).map(|_| s.spawn(|| detector.detect(&text))).collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), expected);
        }
    });
}

// ---------------------------------------------------------------------------
// ArtifactFalsePositiveTest

const CLEAN_TEXT: [&str; 10] = [
    "OpenNLP 3.0 processes ordinary ASCII and version 3.0.0.",
    "Déjà vu à Paris, café, naïve, and João.",
    "Typography uses ‘quotes’, an ellipsis…, an euro sign €, and an en dash.",
    "Καλημέρα κόσμε",
    "Привет, мир",
    "東京で自然言語処理を試す",
    "مرحبا بالعالم",
    "हैलो दुनिया",
    "emoji 😀 and a family 👩‍👩‍👦 are ordinary text",
    "HTML tags like <script> and template braces {{value}} are visible text",
];

// testCleanTextYieldsNothing, testJoinedCorpusYieldsNothing
#[test]
fn multilingual_text_yields_nothing() {
    for text in CLEAN_TEXT {
        assert_eq!(detect(text), vec![], "{text}");
    }
    assert_eq!(detect(&CLEAN_TEXT.join(" | ")), vec![]);
}

// testControlTextStillFindsDamageAndHiddenTags
#[test]
fn positive_control_finds_damage_and_hidden_tags() {
    let hidden = cp(&[0xE0073, 0xE0065, 0xE0063, 0xE007F]);
    let damaged = cp(&[0xE2, 0x82, 0xAC]);
    assert_eq!(detect(&format!("price {damaged} {hidden}")).len(), 2);
}
