//! The artifact detector. The expectations of OpenNLP's text-artifacts
//! tests (CursorArtifactDetectorTest, ArtifactBoundaryTest,
//! ArtifactEncodingTest, ArtifactFalsePositiveTest,
//! ArtifactUnicodeContextTest, ArtifactsManualExampleTest,
//! ArtifactAnnotatorTest) come first, with OpenNLP's UTF-16 offsets
//! written as UTF-8 byte offsets. Their unpaired-surrogate cases cannot
//! occur: a Rust `str` holds no unpaired surrogate. Then the two kinds
//! added here, and property loops over generated text.

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

// ---------------------------------------------------------------------------
// ArtifactUnicodeContextTest

// testInvalidFlagTagsRemainVisibleToTheDetector
#[test]
fn invalid_flag_tags_are_reported() {
    let long1 = format!("gb{}", "a".repeat(29));
    let long2 = format!("gbeng{}", "secret".repeat(1000));
    for spec in ["gb", "g1eng", "1ab", "12a", "GBENG", "gb-eng", "gb12345", "00112345", &long1, &long2] {
        let text = format!("\u{1F3F4}{}", tags(spec, true));
        assert_eq!(detect(&text), vec![a(4, text.len(), UnicodeTag)], "{spec}");
    }
}

// testFlagTagSyntaxIsPreserved
#[test]
fn flag_tag_syntax_is_kept() {
    for spec in ["gbeng", "gbsct", "gbwls", "usca", "001", "419", "gb1234", "001abcd"] {
        assert_eq!(detect(&format!("\u{1F3F4}{}", tags(spec, true))), vec![], "{spec}");
    }
}

// testMarksDoNotBreakLetterContext
#[test]
fn marks_do_not_break_letter_context() {
    let cases: [(&str, &[u32]); 5] = [
        ("Malayalam ZWNJ", &[0x0D26, 0x0D43, 0x0D15, 0x0D4D, 0x200C, 0x0D38, 0x0D3E, 0x0D15, 0x0D4D, 0x0D37, 0x0D3F]),
        ("Sinhala ZWJ", &[0x0DC1, 0x0DCA, 0x200D, 0x0DBB, 0x0DD3, 0x20, 0x0DBD, 0x0D82, 0x0D9A, 0x0DCF]),
        ("marks before and after", &[0x0628, 0x064E, 0x200C, 0x0650, 0x0628]),
        ("decomposed accent", &['e' as u32, 0x0301, 0x200B, 'x' as u32]),
        ("supplementary letters", &[0x10400, 0x0301, 0x200C, 0x10401]),
    ];
    for (label, points) in cases {
        assert_eq!(only(&[ZeroWidth], &cp(points)), vec![], "{label}");
    }
}

// testMarksDoNotHideIsolatedZeroWidthCharacters
#[test]
fn marks_do_not_hide_isolated_zero_width_characters() {
    let (a_, b_, x, dot) = ('a' as u32, 'b' as u32, 'x' as u32, '.' as u32);
    let cases: [&[u32]; 6] = [
        &[dot, 0x0301, 0x200C, x],
        &[x, 0x200C, 0x0301, dot],
        &[0x0301, 0x200C, x],
        &[x, 0x200C, 0x0301],
        &[a_, dot, 0x0301, 0x200B, b_],
        &[a_, 0x0301, 0x200B, 0x200B, b_],
    ];
    for points in cases {
        let found = only(&[ZeroWidth], &cp(points));
        assert_eq!(found.iter().map(|f| f.kind).collect::<Vec<_>>(), vec![ZeroWidth], "{points:X?}");
    }
}

// ---------------------------------------------------------------------------
// ArtifactsManualExampleTest

fn manual_text() -> String {
    format!("caf{} costs {}8", cp(&[0xC3, 0xA9]), cp(&[0xFFFD]))
}

// testDetectionExamplePrintsTheStatedLines (mojibake [3..5),
// replacement [12..13) in UTF-16)
#[test]
fn manual_detection_example() {
    let found = detect(&manual_text());
    let lines: Vec<String> =
        found.iter().map(|f| format!("{} [{}..{})", f.kind.name(), f.span.start, f.span.end)).collect();
    assert_eq!(lines, ["mojibake [3..7)", "replacement [14..17)"]);
}

// testAccentedTextYieldsNothing
#[test]
fn manual_accented_text_yields_nothing() {
    assert_eq!(detect("déjà vu"), vec![]);
}

// testSelectiveDetectorExample
#[test]
fn manual_selective_detector() {
    let found = only(&[Mojibake, UnicodeTag], &manual_text());
    assert_eq!(found.iter().map(|f| f.kind).collect::<Vec<_>>(), vec![Mojibake]);
}

// testMaskingExampleProducesTheStatedString
#[test]
fn manual_masking_example() {
    let text = manual_text();
    let found = detect(&text);
    let masked: String = text
        .char_indices()
        .map(|(i, c)| if found.iter().any(|f| f.span.range().contains(&i)) { '#' } else { c })
        .collect();
    assert_eq!(masked, "caf## costs #8");
}

// testLayerExampleCarriesTheStatedAnnotations, ArtifactAnnotatorTest
// testAnnotatesFindingsWithExactSpans
#[test]
fn manual_layer_example() {
    let text = manual_text();
    let layer = check_layer(&text, &ArtifactDetector::default());
    assert_eq!(layer.len(), 2);
    let first = layer.spans.iter().next().unwrap().1;
    assert_eq!(&text[first.range()], cp(&[0xC3, 0xA9]));
    assert_eq!(layer.kind[0], Mojibake);
}

// testAdjacentControlExample
#[test]
fn manual_adjacent_control_example() {
    let text = format!("value {}", cp(&[0xC2, 0x80, 0]));
    let layer = check_layer(&text, &ArtifactDetector::default());
    let spans: Vec<Span> = layer.spans.iter().map(|(_, s)| s).collect();
    assert_eq!(spans, [Span::new(6, 10), Span::new(10, 11)]);
    assert_eq!(layer.kind, [Mojibake, Control]);
}

// ---------------------------------------------------------------------------
// ArtifactAnnotatorTest and TextArtifactTest

// testProvidesTheArtifactsLayer
#[test]
fn annotator_id_and_layer() {
    assert_eq!(<ArtifactDetector as Annotator>::ID, "turbo:artifacts");
    assert_eq!(Artifacts::ID, "turbo:artifacts");
    assert_ne!(
        ArtifactDetector::default().config_hash(),
        ArtifactDetector::builder().with(MojibakeSequence).build().config_hash()
    );
}

// testCleanDocumentGetsAnEmptyLayer, run through an analyzer.
#[test]
fn analyzer_adds_the_layer_even_when_empty() {
    let analyzer = Analyzer::builder().add(ArtifactDetector::default()).build().unwrap();
    let mut docs = vec![Document::new("all clean"), Document::new(manual_text())];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    assert!(docs[0].get::<Artifacts>().unwrap().is_empty());
    assert_eq!(docs[1].get::<Artifacts>().unwrap().len(), 2);
    assert_eq!(docs[1].source(), manual_text());
}

#[test]
fn kind_names_are_opennlp_strings() {
    let names: Vec<&str> = ArtifactKind::OPENNLP.iter().map(|k| k.name()).collect();
    assert_eq!(
        names,
        [
            "replacement",
            "control",
            "noncharacter",
            "unpaired-surrogate",
            "private-use",
            "bidi-control",
            "zero-width",
            "unicode-tag",
            "mojibake"
        ]
    );
    for k in ArtifactKind::ALL {
        assert_eq!(ArtifactKind::from_name(k.name()), Some(k));
    }
    assert_eq!(ArtifactKind::from_name("custom-detector-type"), None);
}

#[test]
fn layer_check_refuses_overlaps_and_missing_kinds() {
    let mut layer = Artifacts::new(turbo_analysis::ViewId::SOURCE);
    layer.spans.push(Span::new(0, 3));
    layer.spans.push(Span::new(2, 4));
    layer.kind.push(Control);
    assert!(layer.check().is_err());
    layer.kind.push(Control);
    assert!(layer.check().is_err());
}

// ---------------------------------------------------------------------------
// MojibakeSequence (off by default)

fn sequences(text: &str) -> Vec<Artifact> {
    ArtifactDetector::builder().with(MojibakeSequence).build().detect(text)
}

#[test]
fn sequences_are_off_by_default() {
    assert!(!KindSet::OPENNLP.contains(MojibakeSequence));
    let text = format!("caf{}\u{2026}", cp(&[0xC3, 0xA9]));
    assert_eq!(detect(&text), vec![]);
}

#[test]
fn sequence_next_to_a_genuine_character() {
    // The typed ellipsis is byte 0x85 in Windows-1252, a bare continuation
    // after the image of "é", so the whole run does not round-trip.
    let text = format!("caf{}\u{2026} and na{}\u{2014}ve", cp(&[0xC3, 0xA9]), cp(&[0xC3, 0xAF]));
    assert_eq!(
        covered(&text, &sequences(&text)),
        vec![("\u{C3}\u{A9}", MojibakeSequence), ("\u{C3}\u{AF}", MojibakeSequence)]
    );
}

#[test]
fn sequence_with_a_byte_lost_to_a_replacement_character() {
    // "Á" is C3 81; Windows-1252 leaves 0x81 undefined.
    let text = format!("{}LVAREZ", cp(&[0xC3, 0xFFFD]));
    assert_eq!(detect(&text), vec![a(2, 5, ReplacementChar)]);
    assert_eq!(sequences(&text), vec![a(0, 5, MojibakeSequence)]);
}

#[test]
fn damaged_japanese_is_one_sequence_run() {
    // こんにちは as UTF-8 read as Windows-1252, the undefined 0x81 bytes
    // replaced: the round trip fails at every U+FFFD.
    let text = windows_1252_decode("こんにちは".as_bytes());
    assert!(text.contains('\u{FFFD}'));
    assert!(detect(&text).iter().all(|f| f.kind == ReplacementChar));
    assert_eq!(sequences(&text), vec![a(0, text.len(), MojibakeSequence)]);
}

#[test]
fn mac_roman_sequences() {
    // é is C3 A9: √© in MacRoman. ’ is E2 80 99: ‚Äô.
    for (text, image) in [("caf\u{221A}\u{A9}", "\u{221A}\u{A9}"), ("don\u{201A}\u{C4}\u{F4}t", "\u{201A}\u{C4}\u{F4}")]
    {
        assert_eq!(detect(text), vec![], "{text}");
        assert_eq!(covered(text, &sequences(text)), vec![(image, MojibakeSequence)], "{text}");
    }
}

#[test]
fn round_trip_mojibake_takes_precedence_over_sequences() {
    let text = format!("caf{} {}", cp(&[0xC3, 0xA9]), cp(&[0xE2, 0x20AC, 0x2122]));
    assert_eq!(sequences(&text), detect(&text));
    assert!(sequences(&text).iter().all(|f| f.kind == Mojibake));
}

#[test]
fn genuine_text_has_no_sequences() {
    let texts = [
        "\u{201C}\u{E9}lan\u{201D} and \u{AB}\u{E9}t\u{E9}\u{BB}",
        "CAF\u{C9} \u{2013} BAR, \u{2014}\u{E0} bient\u{F4}t\u{2026}",
        "\u{2018}\u{E0}\u{2019} \u{2026}\u{E0}",
        "S\u{C3}O PAULO, \u{C5}NGSTR\u{D6}M, \u{C9}COLE",
        "Stra\u{DF}e \u{2022} Gr\u{FC}\u{DF}e \u{2122} \u{A9} 2024",
    ];
    for text in texts.iter().copied().chain(CLEAN_TEXT) {
        assert_eq!(sequences(text), vec![], "{text}");
    }
}

// The round trip of OpenNLP reports an accented capital before a dash or
// quote: "É–" is C9 96, the UTF-8 of U+0256. The default keeps that for
// parity; the sequence rule alone would not report it (U+0256 is outside
// its two-byte target ranges).
#[test]
fn round_trip_reports_an_accented_capital_before_a_dash() {
    let text = "CAF\u{C9}\u{2013}BAR";
    assert_eq!(detect(text), vec![a(3, 8, Mojibake)]);
    assert_eq!(ArtifactDetector::builder().only(&[MojibakeSequence]).build().detect(text), vec![]);
}

#[test]
fn sequences_take_precedence_over_classes() {
    // U+0081 is a control and the continuation of "Á" after Ã; with the
    // genuine ellipsis after it the run does not round-trip.
    let text = format!("{}\u{2026}", cp(&[0xC3, 0x81]));
    assert_eq!(detect(&text), vec![a(2, 4, Control)]);
    assert_eq!(sequences(&text), vec![a(0, 4, MojibakeSequence)]);
}

// ---------------------------------------------------------------------------
// RepeatedLine (off by default)

fn report(pages: usize) -> String {
    let mut text = String::new();
    for page in 1..=pages {
        text.push_str("ACME Corporation  Annual Report 2024\n");
        text.push_str(&format!("Body text of page {page}, which differs: {}.\n", "x".repeat(page)));
        text.push_str(&format!("{page}\n"));
        text.push_str(&format!("   Page {page} of {pages}\r\n\n"));
    }
    text
}

fn repeated(text: &str, min: u32) -> Vec<Artifact> {
    ArtifactDetector::builder().repeated_lines(RepeatedLines { min_occurrences: min }).build().detect(text)
}

#[test]
fn repeated_lines_are_off_by_default() {
    assert_eq!(detect(&report(4)), vec![]);
}

#[test]
fn headers_and_page_labels_repeat() {
    let text = report(3);
    let found = repeated(&text, 3);
    let lines: Vec<&str> = found.iter().map(|f| &text[f.span.range()]).collect();
    assert_eq!(
        lines,
        [
            "ACME Corporation  Annual Report 2024",
            "Page 1 of 3",
            "ACME Corporation  Annual Report 2024",
            "Page 2 of 3",
            "ACME Corporation  Annual Report 2024",
            "Page 3 of 3",
        ]
    );
    assert!(found.iter().all(|f| f.kind == RepeatedLine));
    check_layer(&text, &ArtifactDetector::builder().repeated_lines(RepeatedLines::default()).build());
}

#[test]
fn two_occurrences_are_not_enough_by_default() {
    assert_eq!(repeated(&report(2), 3), vec![]);
    assert_eq!(repeated(&report(2), 2).len(), 4);
}

#[test]
fn inner_whitespace_and_digit_runs_are_folded() {
    let text = "Page 1 of 9\nPage  22 of 9\nPage\t333 of 10\nPage one of nine\n";
    let found = repeated(text, 3);
    assert_eq!(found.len(), 3);
    assert_eq!(&text[found[2].span.range()], "Page\t333 of 10");
}

#[test]
fn lines_without_letters_are_not_compared() {
    let text = "12.5\n13.7\n8.2\n- 4 -\n- 5 -\n- 6 -\n";
    assert_eq!(repeated(text, 3), vec![]);
}

#[test]
fn character_findings_inside_a_repeated_line_keep_their_spans() {
    let text = "Confidential\u{0}Draft\nConfidential\u{0}Draft\nConfidential\u{0}Draft\n";
    let found = repeated(text, 3);
    assert_eq!(found[..3], [a(0, 12, RepeatedLine), a(12, 13, Control), a(13, 18, RepeatedLine)]);
    assert_eq!(found.len(), 9);
}

// ---------------------------------------------------------------------------
// Property loops

/// A deterministic generator (xorshift64*), so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Characters of every class and of every context the rules look at.
const POOL: &[char] = &[
    'a',
    'b',
    'z',
    'Q',
    ' ',
    '.',
    '\n',
    '\t',
    '\r',
    '1', // ordinary
    '\u{FFFD}',
    '\u{0}',
    '\u{1B}',
    '\u{7F}',
    '\u{85}',
    '\u{9F}',
    '\u{1C}', // replacement, controls, NEL
    '\u{FDD0}',
    '\u{FFFE}',
    '\u{1FFFF}',
    '\u{10FFFF}', // noncharacters
    '\u{E000}',
    '\u{F8FF}',
    '\u{F0000}',
    '\u{100000}', // private use
    '\u{202E}',
    '\u{2068}',
    '\u{200E}',
    '\u{061C}', // bidi
    '\u{200B}',
    '\u{200C}',
    '\u{200D}',
    '\u{2060}',
    '\u{FEFF}', // zero width
    '\u{E0067}',
    '\u{E0062}',
    '\u{E0065}',
    '\u{E006E}',
    '\u{E007F}',
    '\u{E0031}',
    '\u{1F3F4}', // tags
    '\u{1F600}',
    '\u{FE0F}',
    '\u{2764}', // emoji context
    '\u{301}',
    '\u{E01}',
    '\u{628}',
    '\u{10400}', // marks and letters
    '\u{C3}',
    '\u{C2}',
    '\u{E2}',
    '\u{F0}',
    '\u{A9}',
    '\u{AF}',
    '\u{80}',
    '\u{20AC}',
    '\u{2122}',
    '\u{9F}',
    '\u{98}',
    '\u{2026}',
    '\u{221A}',
    '\u{201A}',
    '\u{C4}',
    '\u{F4}',
    '\u{E9}', // mojibake images
];

fn generate(rng: &mut Rng) -> String {
    let n = rng.below(40);
    (0..n).map(|_| POOL[rng.below(POOL.len())]).collect()
}

/// The class of a code point, written out apart from the detector.
fn reference_class(c: char) -> Option<ArtifactKind> {
    let p = c as u32;
    match p {
        0xFFFD => Some(ReplacementChar),
        0x09..=0x0D | 0x85 => None,
        0..=0x1F | 0x7F..=0x9F => Some(Control),
        0xFDD0..=0xFDEF => Some(Noncharacter),
        _ if p & 0xFFFE == 0xFFFE => Some(Noncharacter),
        0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD => Some(PrivateUse),
        0x202A..=0x202E | 0x2066..=0x2069 | 0x200E | 0x200F | 0x061C => Some(BidiControl),
        _ => None,
    }
}

/// Whether the characters' Windows-1252 bytes are valid multi-byte UTF-8.
fn round_trips(s: &str) -> bool {
    let bytes: Vec<u8> = s
        .chars()
        .map(|c| {
            let p = c as u32;
            if p <= 0xFF {
                p as u8
            } else {
                let decoded = (0x80..0xA0u8).find(|&b| windows_1252_decode(&[b]).starts_with(c));
                decoded.expect("a Windows-1252 character")
            }
        })
        .collect();
    std::str::from_utf8(&bytes).is_ok_and(|t| !t.is_ascii())
}

const CLASS_KINDS: [ArtifactKind; 5] = [ReplacementChar, Control, Noncharacter, PrivateUse, BidiControl];

fn check_structure(text: &str, found: &[Artifact]) {
    for f in found {
        assert!(!f.span.is_empty() && f.span.end as usize <= text.len(), "{text:?} {f:?}");
        assert!(text.is_char_boundary(f.span.start as usize) && text.is_char_boundary(f.span.end as usize));
    }
    assert!(found.windows(2).all(|w| w[0].span.end <= w[1].span.start), "{text:?} {found:?}");
}

fn covering(found: &[Artifact], at: usize) -> Option<ArtifactKind> {
    found.iter().find(|f| f.span.range().contains(&at)).map(|f| f.kind)
}

#[test]
fn generated_text_default_detector() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let detector = ArtifactDetector::default();
    for _ in 0..4000 {
        let text = generate(&mut rng);
        let found = detector.detect(&text);
        check_structure(&text, &found);
        check_layer(&text, &detector);
        for f in &found {
            let s = &text[f.span.range()];
            match f.kind {
                k if CLASS_KINDS.contains(&k) => assert!(s.chars().all(|c| reference_class(c) == Some(k)), "{s:?}"),
                ZeroWidth => assert!(s.chars().all(|c| "\u{200B}\u{200C}\u{200D}\u{2060}\u{FEFF}".contains(c))),
                UnicodeTag => assert!(s.chars().all(|c| ('\u{E0000}'..='\u{E007F}').contains(&c))),
                Mojibake => assert!(round_trips(s), "{s:?}"),
                other => panic!("{other:?} is not a default kind"),
            }
        }
        // Every classified character is reported, as its class or inside
        // mojibake, and a class run ends where its class ends.
        for (i, c) in text.char_indices() {
            if let Some(k) = reference_class(c) {
                assert!(matches!(covering(&found, i), Some(x) if x == k || x == Mojibake), "{text:?} at {i}");
            }
        }
        for f in found.iter().filter(|f| CLASS_KINDS.contains(&f.kind)) {
            let after = text[f.span.end as usize..].chars().next();
            if after.and_then(reference_class) == Some(f.kind) {
                assert_eq!(covering(&found, f.span.end as usize), Some(Mojibake), "{text:?}");
            }
        }
    }
}

#[test]
fn generated_text_each_kind_alone() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let without_mojibake = ArtifactDetector::builder().without(Mojibake).build();
    let mojibake_only = ArtifactDetector::builder().only(&[Mojibake]).build();
    let default = ArtifactDetector::default();
    for _ in 0..3000 {
        let text = generate(&mut rng);
        let all = without_mojibake.detect(&text);
        for kind in [ReplacementChar, Control, Noncharacter, PrivateUse, BidiControl, ZeroWidth, UnicodeTag] {
            let alone = only(&[kind], &text);
            let filtered: Vec<Artifact> = all.iter().copied().filter(|f| f.kind == kind).collect();
            assert_eq!(alone, filtered, "{kind:?} in {text:?}");
        }
        let full: Vec<Artifact> = default.detect(&text).into_iter().filter(|f| f.kind == Mojibake).collect();
        assert_eq!(mojibake_only.detect(&text), full, "{text:?}");
    }
}

#[test]
fn generated_text_with_the_added_kinds() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let default = ArtifactDetector::default();
    let extended = ArtifactDetector::builder().with(MojibakeSequence).repeated_lines(RepeatedLines::default()).build();
    for _ in 0..3000 {
        let mut text = generate(&mut rng);
        // Repeat a generated line now and then so repeated lines occur.
        if rng.below(4) == 0 {
            let line = format!("Header {}\n", generate(&mut rng).replace(['\n', '\r'], " "));
            text = format!("{line}{text}\n{line}{line}");
        }
        let base = default.detect(&text);
        let found = extended.detect(&text);
        check_structure(&text, &found);
        check_layer(&text, &extended);
        // Round-trip findings are unchanged; every other default finding is
        // still covered, by its own kind or by a mojibake sequence.
        let moji = |v: &[Artifact]| v.iter().copied().filter(|f| f.kind == Mojibake).collect::<Vec<_>>();
        assert_eq!(moji(&found), moji(&base), "{text:?}");
        for f in &base {
            for (i, _) in text[f.span.range()].char_indices() {
                let at = f.span.start as usize + i;
                assert!(matches!(covering(&found, at), Some(k) if k == f.kind || k == MojibakeSequence), "{text:?}");
            }
        }
        for f in found.iter().filter(|f| f.kind == RepeatedLine) {
            assert!(!text[f.span.range()].contains('\n'), "{text:?}");
        }
    }
}

#[test]
fn every_kind_has_a_set_bit() {
    let mut set = KindSet::default();
    for k in ArtifactKind::ALL {
        assert!(!set.contains(k));
        set = set.with(k);
        assert!(set.contains(k));
    }
    assert_eq!(set, KindSet::ALL);
    assert!(!set.without(Mojibake).contains(Mojibake));
}
