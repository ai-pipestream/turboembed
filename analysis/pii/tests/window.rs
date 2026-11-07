//! Context-word window (`turbo_pii::window`), `PiiConfidence::Labelled`,
//! `LabelPolicy`, and caller overrides on `PiiDetector`.

mod support;

use std::time::Instant;

use support::*;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Annotator, Document, Provenance, Span};
use turbo_analysis_layers::export::codecs;
use turbo_pii::text::{on_end, on_number_start};
use turbo_pii::window::{
    DEFAULT_WINDOW_WORDS, LabelPolicy, default_labels, default_policy, default_window, has_kind_label, has_label,
    has_strict_prefix_label,
};
use turbo_pii::{Extractor, Pack, Pii, PiiConfidence, PiiDetector, PiiKind, PiiSink, Scanner};

const PASSPORT_KIND: PiiKind = PiiKind::custom("passport");

/// Toy 9-digit candidate scanner (`Format` confidence) used to exercise
/// `label_required`, `label_boosts`, and window width overrides on a custom
/// kind without check-digit coupling.
fn scan_nine_with(text: &str, sink: &mut PiiSink<'_>, conf: PiiConfidence) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 9 <= b.len() {
        if on_number_start(text, i) && b[i..i + 9].iter().all(u8::is_ascii_digit) && on_end(text, i + 9) {
            sink.add_text(Span::new(i as u32, (i + 9) as u32), PASSPORT_KIND, conf);
            i += 9;
        } else {
            i += 1;
        }
    }
}

fn scan_nine_digits(text: &str, sink: &mut PiiSink<'_>) {
    scan_nine_with(text, sink, PiiConfidence::Format);
}

#[test]
fn window_helper_matches_labels_before_and_after_and_across_newlines() {
    let labels = &["ssn", "social security number", "social security no"];

    // Label before the span.
    let text = "SSN: 123-45-6789";
    assert!(has_label(text, Span::new(5, 16), labels, 4));
    assert!(has_kind_label(text, Span::new(5, 16), PiiKind::UsSsn));

    // Label after the span, inside parentheses or punctuation.
    let text = "123-45-6789 (SSN)";
    assert!(has_label(text, Span::new(0, 11), labels, 4));
    assert!(has_kind_label(text, Span::new(0, 11), PiiKind::UsSsn));

    let text = "046-454-286 (SIN)";
    assert!(has_kind_label(text, Span::new(0, 11), PiiKind::CaSin));

    let text = "490154203237518 - IMEI";
    assert!(has_kind_label(text, Span::new(0, 15), PiiKind::Imei));

    // Multi-word label phrases before and after the span.
    let text = "Social Security Number: 123-45-6789";
    assert!(has_label(text, Span::new(24, 35), labels, 1));

    let text = "123-45-6789 is a Social Security No.";
    assert!(has_label(text, Span::new(0, 11), labels, 4));

    // Labels across LF and CRLF line breaks.
    let text = "SSN:\n123-45-6789";
    assert!(has_label(text, Span::new(5, 16), labels, 4));

    let text = "Social Security Number\r\n123-45-6789";
    assert!(has_label(text, Span::new(24, 35), labels, 1));

    let text = "123-45-6789\r\n(SSN)";
    assert!(has_label(text, Span::new(0, 11), labels, 1));
}

#[test]
fn window_helper_enforces_word_distance_bounds_and_invalid_spans() {
    let labels = &["ssn", "social security number"];

    // 4 words before the span: "ssn" is at distance 4 ("ssn", "on", "file", "is").
    let d4_before = "SSN on file is 123-45-6789";
    let span = Span::new(15, 26);
    assert!(has_label(d4_before, span, labels, 4));
    assert!(!has_label(d4_before, span, labels, 3));

    // 5 words before the span: "ssn" is at distance 5 ("ssn", "kept", "on", "file", "is").
    let d5_before = "SSN kept on file is 123-45-6789";
    let span = Span::new(20, 31);
    assert!(!has_label(d5_before, span, labels, 4));
    assert!(has_label(d5_before, span, labels, 5));

    // 4 words after the span: "ssn" is at distance 4 ("is", "the", "applicant", "ssn").
    let d4_after = "123-45-6789 is the applicant SSN";
    let span = Span::new(0, 11);
    assert!(has_label(d4_after, span, labels, 4));
    assert!(!has_label(d4_after, span, labels, 3));

    // 5 words after the span: "ssn" is at distance 5 ("is", "the", "primary", "applicant", "ssn").
    let d5_after = "123-45-6789 is the primary applicant SSN";
    let span = Span::new(0, 11);
    assert!(!has_label(d5_after, span, labels, 4));
    assert!(has_label(d5_after, span, labels, 5));

    // Multi-word phrase distance is measured from the near boundary of the phrase.
    let multi_before = "Social Security Number on file is 123-45-6789";
    let span = Span::new(34, 45);
    assert!(has_label(multi_before, span, labels, 4));
    assert!(!has_label(multi_before, span, labels, 3));

    let multi_after = "123-45-6789 is the applicant Social Security Number";
    let span = Span::new(0, 11);
    assert!(has_label(multi_after, span, labels, 4));
    assert!(!has_label(multi_after, span, labels, 3));

    // Degenerate inputs return false cleanly without panicking.
    assert!(!has_label("SSN: 123-45-6789", Span::new(5, 16), labels, 0));
    assert!(!has_label("SSN: 123-45-6789", Span::new(5, 16), &[], 4));
    assert!(!has_label("SSN: 123-45-6789", Span::new(5, 99), labels, 4));
    assert!(!has_label("éSSN: 123-45-6789", Span::new(1, 5), labels, 4));
}

#[test]
fn kind_defaults_match_specification() {
    for kind in PiiKind::ALL {
        assert!(!default_labels(kind).is_empty(), "{kind} should have default labels");
        if matches!(kind, PiiKind::CaSin | PiiKind::Imei) {
            assert_eq!(default_window(kind), 1, "{kind}");
            assert_eq!(default_policy(kind), LabelPolicy::Required, "{kind}");
        } else {
            assert_eq!(default_window(kind), DEFAULT_WINDOW_WORDS, "{kind}");
            assert_eq!(default_policy(kind), LabelPolicy::None, "{kind}");
        }
    }
    assert!(default_labels(PASSPORT_KIND).is_empty());
    assert_eq!(default_window(PASSPORT_KIND), DEFAULT_WINDOW_WORDS);
    assert_eq!(default_policy(PASSPORT_KIND), LabelPolicy::None);
}

#[test]
fn sin_and_imei_preserve_default_strict_prefix_parity_and_support_opt_in_window() {
    let sin_default = PiiDetector::pack(Pack::CaIdentity);
    let imei_default = PiiDetector::pack(Pack::Device);

    // Default CaSin and Imei preserve OpenNLP `LabeledPiiExtractor` parity:
    // trailing labels, disallowed separators (`.` for IMEI, `/`, `(`, `,`, `-`),
    // and split multi-word phrases are rejected by default.
    for imei_probe in [
        "IMEI. 490154203237518",
        "490154203237518 IMEI",
        "IMEI/490154203237518",
        "IMEI (490154203237518)",
        "IMEI, 490154203237518",
        "imei-490154203237518",
    ] {
        assert_none(&imei_default, imei_probe);
    }
    for sin_probe in [
        "046-454-286 SIN",
        "SIN (046-454-286)",
        "SIN/046-454-286",
        "Social  insurance\nnumber 046 454 286",
        "the sin, 046 454 286",
    ] {
        assert_none(&sin_default, sin_probe);
    }
    assert!(has_strict_prefix_label("SIN. 046-454-286", 5, PiiKind::CaSin));
    assert!(!has_strict_prefix_label("IMEI. 490154203237518", 6, PiiKind::Imei));

    // Opting into the general UAX #29 window via `with_label_window` accepts
    // trailing labels, punctuation-separated labels, and multi-line phrases.
    let sin_window = PiiDetector::pack(Pack::CaIdentity).with_label_window(PiiKind::CaSin, 1);
    for text in [
        "046-454-286 SIN",
        "046-454-286 (SIN)",
        "SIN (046-454-286)",
        "SIN/046-454-286",
        "Social  insurance\nnumber 046 454 286",
        "the sin, 046 454 286",
        "046 454 286 - Social Insurance Number",
        "046454286 (social insurance no.)",
        "Social Insurance Number\r\n046 454 286",
    ] {
        let pii = layer(&sin_window, text);
        let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
        assert_eq!(items, [(PiiKind::CaSin, PiiConfidence::LabelledChecksum, "046454286")], "{text}");
    }

    let imei_window = PiiDetector::pack(Pack::Device).with_label_window(PiiKind::Imei, 1);
    for text in [
        "IMEI. 490154203237518",
        "490154203237518 IMEI",
        "IMEI/490154203237518",
        "IMEI (490154203237518)",
        "IMEI, 490154203237518",
        "imei-490154203237518",
        "490154203237518\r\n(imei)",
    ] {
        let pii = layer(&imei_window, text);
        let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
        assert_eq!(items, [(PiiKind::Imei, PiiConfidence::LabelledChecksum, "490154203237518")], "{text}");
    }
}

#[test]
fn regional_indicator_runs_scan_in_linear_time() {
    let flags = "\u{1F1E6}".repeat(4_000);
    let text = format!("{flags} 490154203237518 {flags}");
    let start = flags.len() as u32 + 1;
    let span = Span::new(start, start + 15);

    let t0 = Instant::now();
    assert!(!has_label(&text, span, &["imei"], 4));
    let imei_window = PiiDetector::pack(Pack::Device).with_label_window(PiiKind::Imei, 4);
    assert_none(&imei_window, &text);
    assert!(t0.elapsed().as_millis() < 500, "4,000 regional indicators took {} ms", t0.elapsed().as_millis());
}

#[test]
fn label_required_drops_unlabelled_and_promotes_labelled_candidates() {
    // Format-backed kind (`UsSsn`): unlabelled is dropped, labelled becomes `PiiConfidence::Labelled`.
    let ssn_req = PiiDetector::pack(Pack::UsIdentity).label_required(PiiKind::UsSsn);
    assert_eq!(ssn_req.label_policy(PiiKind::UsSsn), LabelPolicy::Required);
    assert_eq!(ssn_req.label_policy(PiiKind::UsItin), LabelPolicy::None);

    let text = "Bare 123-45-6789 stands alone near the start of this paragraph, \
                whereas SSN 123-45-6789 and 234-56-7890 (social security) are kept, \
                and unlabelled ITIN 912-70-1234 still matches.";
    let pii = layer(&ssn_req, text);
    let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
    assert_eq!(
        items,
        [
            (PiiKind::UsSsn, PiiConfidence::Labelled, "123-45-6789"),
            (PiiKind::UsSsn, PiiConfidence::Labelled, "234-56-7890"),
            (PiiKind::UsItin, PiiConfidence::Format, "912-70-1234"),
        ]
    );

    // Checksum-backed kind (`UkNhs`): unlabelled is dropped, labelled becomes `PiiConfidence::LabelledChecksum`.
    let nhs_req = PiiDetector::pack(Pack::EuIdentity).label_required(PiiKind::UkNhs);
    let text = "Unlabelled 943 476 5919 stands alone near the start of this paragraph; \
                NHS number: 943 476 5919 is kept.";
    let pii = layer(&nhs_req, text);
    let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
    assert_eq!(items, [(PiiKind::UkNhs, PiiConfidence::LabelledChecksum, "9434765919")]);

    // Pre-labelled custom extractor candidates (`Labelled` and `LabelledChecksum`) must also pass `label_required`.
    let scan_lbl: fn(&str, &mut PiiSink<'_>) = |t, s| scan_nine_with(t, s, PiiConfidence::Labelled);
    let scan_lbl_chk: fn(&str, &mut PiiSink<'_>) = |t, s| scan_nine_with(t, s, PiiConfidence::LabelledChecksum);
    for (pre_conf, scan_fn) in [(PiiConfidence::Labelled, scan_lbl), (PiiConfidence::LabelledChecksum, scan_lbl_chk)] {
        let det = PiiDetector::new(vec![Scanner::custom_fn("PreLabelled", PASSPORT_KIND, scan_fn)])
            .label_required(PASSPORT_KIND)
            .with_extra_label(PASSPORT_KIND, "passport");
        assert_none(&det, "Unlabelled 123456789 has no context label");
        let pii = layer(&det, "Unlabelled 111222333 sits far away from the start; passport: 123456789.");
        let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
        assert_eq!(items, [(PASSPORT_KIND, pre_conf, "123456789")]);
    }
}

#[test]
fn label_boosts_and_binary_wire_tags_preserve_backwards_compatibility() {
    let det = PiiDetector::new(vec![
        Scanner::all(Extractor::UsIdentity),
        Scanner::all(Extractor::EuIdentity),
        Scanner::all(Extractor::Device),
    ])
    .label_boosts(PiiKind::UsSsn);

    let text = "Unlabelled 123-45-6789 and 943 476 5919 stand alone near the start of this paragraph; \
                IMEI: 490154203237518 and labelled SSN: 123-45-6789.";
    let pii = layer(&det, text);
    let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
    assert_eq!(
        items,
        [
            (PiiKind::UsSsn, PiiConfidence::Format, "123-45-6789"),
            (PiiKind::UkNhs, PiiConfidence::Checksum, "9434765919"),
            (PiiKind::Imei, PiiConfidence::LabelledChecksum, "490154203237518"),
            (PiiKind::UsSsn, PiiConfidence::Labelled, "123-45-6789"),
        ]
    );

    // Binary graph export/import preserves existing wire tags:
    // Format = 0, Checksum = 1, LabelledChecksum = 2, Labelled = 3.
    let mut doc = Document::new(text);
    doc.add(&turbo_analysis::Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    assert!(
        bytes.windows(8).any(|w| w == [4, 0, 0, 0, 0, 1, 2, 3]),
        "expected wire tags [0, 1, 2, 3] for [Format, Checksum, LabelledChecksum, Labelled]"
    );
    let round_tripped = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let back_pii = round_tripped.get::<Pii>().unwrap();
    assert_eq!(
        back_pii.confidence,
        [PiiConfidence::Format, PiiConfidence::Checksum, PiiConfidence::LabelledChecksum, PiiConfidence::Labelled,]
    );
}

#[test]
fn caller_overrides_extra_labels_window_width_and_sin_policy() {
    // Custom kind with caller-supplied labels and `label_required`.
    let passport_det = PiiDetector::new(vec![Scanner::custom_fn("PassportScanner", PASSPORT_KIND, scan_nine_digits)])
        .label_required(PASSPORT_KIND)
        .with_extra_labels(PASSPORT_KIND, ["passport", "pass no", "travel document"]);

    assert_eq!(passport_det.extra_labels(PASSPORT_KIND), ["passport", "pass no", "travel document"]);
    assert_eq!(passport_det.label_window(PASSPORT_KIND), DEFAULT_WINDOW_WORDS);

    let text = "Order 111222333 was shipped early this morning; \
                Passport # 123456789; travel document: 987654321; 555666777 (Pass No.).";
    let pii = layer(&passport_det, text);
    let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect();
    assert_eq!(
        items,
        [
            (PASSPORT_KIND, PiiConfidence::Labelled, "123456789"),
            (PASSPORT_KIND, PiiConfidence::Labelled, "987654321"),
            (PASSPORT_KIND, PiiConfidence::Labelled, "555666777"),
        ]
    );
    let pass_start = text.find("123456789").unwrap() as u32;
    assert!(passport_det.has_label(text, Span::new(pass_start, pass_start + 9), PASSPORT_KIND));
    assert!(!passport_det.has_label(text, Span::new(6, 15), PASSPORT_KIND));

    // Widening SIN's window from 1 to 4 admits intervening words, and adding a French label matches "NAS".
    let wide_sin = PiiDetector::pack(Pack::CaIdentity)
        .with_label_window(PiiKind::CaSin, 4)
        .with_extra_label(PiiKind::CaSin, "NAS");
    let text = "SIN was recorded as 046 454 286, and NAS: 046-454-286.";
    let pii = layer(&wide_sin, text);
    assert_eq!(pii.mentions().count(), 2);

    // Relaxing SIN's policy to `Boosts` admits unlabelled SINs with `Checksum` and labelled with `LabelledChecksum`.
    let boost_sin = PiiDetector::pack(Pack::CaIdentity).with_label_policy(PiiKind::CaSin, LabelPolicy::Boosts);
    let text = "Unlabelled 046 454 286 and SIN: 046-454-286.";
    let pii = layer(&boost_sin, text);
    let items: Vec<_> = pii.mentions().map(|m| (m.confidence, m.normalized)).collect();
    assert_eq!(items, [(PiiConfidence::Checksum, "046454286"), (PiiConfidence::LabelledChecksum, "046454286")]);
}

#[test]
fn known_limitation_nearby_label_for_another_number_and_window_narrowing() {
    let det = PiiDetector::new(vec![Scanner::custom_fn("PassportScanner", PASSPORT_KIND, scan_nine_digits)])
        .label_required(PASSPORT_KIND)
        .with_extra_label(PASSPORT_KIND, "passport");

    // Known limitation: with the default 4-word window, "Passport" is 3 words
    // before "987654321" ("Passport", "123456789", "order"), so both numbers
    // see the label in their window.
    let text = "Passport 123456789, order 987654321";
    let pii = layer(&det, text);
    let found: Vec<_> = pii.mentions().map(|m| m.normalized).collect();
    assert_eq!(found, ["123456789", "987654321"]);

    // Narrowing the window to 1 word restricts the match to the number
    // immediately adjacent to "Passport".
    let narrow = det.with_label_window(PASSPORT_KIND, 1);
    let pii = layer(&narrow, text);
    let found: Vec<_> = pii.mentions().map(|m| m.normalized).collect();
    assert_eq!(found, ["123456789"]);
}

#[test]
fn config_hash_tracks_non_default_window_overrides_deterministically() {
    let base = PiiDetector::pack(Pack::UsIdentity);
    let base_hash = base.config_hash();

    // Setting a policy or window to its default value leaves `config_hash` unchanged.
    assert_eq!(base.clone().with_label_policy(PiiKind::UsSsn, LabelPolicy::None).config_hash(), base_hash);
    assert_eq!(base.clone().with_label_window(PiiKind::UsSsn, DEFAULT_WINDOW_WORDS).config_hash(), base_hash);
    assert_eq!(
        base.clone().label_required(PiiKind::UsSsn).with_label_policy(PiiKind::UsSsn, LabelPolicy::None).config_hash(),
        base_hash
    );

    // Non-default policy, window width, and extra labels each produce a distinct `config_hash`.
    let h_req = base.clone().label_required(PiiKind::UsSsn).config_hash();
    let h_boost = base.clone().label_boosts(PiiKind::UsSsn).config_hash();
    let h_win = base.clone().with_label_window(PiiKind::UsSsn, 2).config_hash();
    let h_extra = base.clone().with_extra_label(PiiKind::UsSsn, "soc-sec").config_hash();
    assert_ne!(h_req, base_hash);
    assert_ne!(h_boost, base_hash);
    assert_ne!(h_req, h_boost);
    assert_ne!(h_win, base_hash);
    assert_ne!(h_extra, base_hash);

    // Configuring kinds in different orders yields equal detectors and identical hashes.
    let a = base.clone().label_required(PiiKind::UsSsn).label_boosts(PiiKind::UsItin);
    let b = base.label_boosts(PiiKind::UsItin).label_required(PiiKind::UsSsn);
    assert_eq!(a, b);
    assert_eq!(a.config_hash(), b.config_hash());
}
