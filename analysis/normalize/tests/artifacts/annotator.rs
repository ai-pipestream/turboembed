//! Artifacts beside Unicode text, the manual's examples, and the annotator
//! and its record: OpenNLP's `ArtifactUnicodeContextTest`,
//! `ArtifactsManualExampleTest`, `ArtifactAnnotatorTest` and
//! `TextArtifactTest`.

use super::*;

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
