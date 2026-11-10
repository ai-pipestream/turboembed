use turbo_analysis::{Document, Ix, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::{
    AllowRule, AllowScope, ChecksumAlgorithm, CustomKinds, Pii, PiiCandidate, PiiCandidates, PiiDetectors,
    PiiDisposition, PiiEvidence, PiiKind, PiiMention, PiiScore, PiiScoreKind, RejectReason,
};
use turbo_formats::doc_export::{
    DocumentFormatter, ExportOffsetMode, FormatterRegistry, JSONLD_CONTEXT, JsonLdFormatter, MarkdownFormatter,
};

#[test]
fn jsonld_context_document_lists_all_pii_evidence_score_and_disposition_terms() {
    for term in [
        "\"detector\": \"turbo:detector\"",
        "\"id\": \"turbo:id\"",
        "\"version\": \"turbo:version\"",
        "\"evidence\": \"turbo:evidence\"",
        "\"shape\": \"turbo:shape\"",
        "\"checksum\": \"turbo:checksum\"",
        "\"algorithm\": \"turbo:algorithm\"",
        "\"passed\": \"turbo:passed\"",
        "\"label\": \"turbo:label\"",
        "\"span\": \"turbo:span\"",
        "\"phrase\": \"turbo:phrase\"",
        "\"denyHit\": \"turbo:denyHit\"",
        "\"list\": \"turbo:list\"",
        "\"term\": \"turbo:term\"",
        "\"rule\": \"turbo:rule\"",
        "\"score\": \"turbo:score\"",
        "\"value\": \"turbo:value\"",
        "\"kind\": \"turbo:kind\"",
        "\"modelRef\": \"turbo:modelRef\"",
        "\"calibrated\": \"turbo:calibrated\"",
        "\"disposition\": \"turbo:disposition\"",
        "\"kept\": \"turbo:kept\"",
        "\"allowedByRule\": \"turbo:allowedByRule\"",
        "\"scope\": \"turbo:scope\"",
        "\"pattern\": \"turbo:pattern\"",
        "\"nestedIn\": \"turbo:nestedIn\"",
        "\"rejectedByRule\": \"turbo:rejectedByRule\"",
        "\"repeats\": \"turbo:repeats\"",
    ] {
        assert!(JSONLD_CONTEXT.contains(term), "missing term {term} in JSONLD_CONTEXT");
    }
}

#[test]
fn jsonld_and_markdown_export_luhn_card_labelled_custom_kind_and_suppressed_candidate() {
    // Text layout (UTF-8 byte offsets):
    // 0..4:   "Pay "
    // 4..23:  "4532-0151-1283-0366" (Luhn card; normalized digits "4532015112830366" are not in text)
    // 23..25: ", "
    // 25..30: "badge" (context label span 25..30, phrase 1)
    // 30..31: " "
    // 31..39: "EMP-9042" (custom kind "employee-badge"; normalized "emp9042" is not in text)
    // 39..44: " ref "
    // 44..60: "support@acme.example" (suppressed candidate at 44..64)
    let text = "Pay 4532-0151-1283-0366, badge EMP-9042 ref support@acme.example.";
    let planted_card_norm = "4532015112830366";
    let planted_badge_norm = "emp9042";

    let mut doc = Document::new(text);

    let mut custom_kinds = CustomKinds::new();
    let badge_kind = custom_kinds.add("employee-badge").unwrap();

    let mut detectors = PiiDetectors::new();
    let card_det = detectors.add("turbo-pii:cursor", "0.1.0").unwrap();
    let badge_det = detectors.add("acme:badge-scanner", "2.3").unwrap();
    let badge_model = detectors.add("acme:badge-Ranker", "1.0").unwrap();
    let email_det = detectors.add("turbo-pii:email", "0.1.0").unwrap();

    let mut pii = Pii::with_tables(ViewId::SOURCE, custom_kinds.clone(), detectors.clone());
    let card_ev = [PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true }];
    pii.push(PiiMention {
        span: Span::new(4, 23),
        kind: PiiKind::Card,
        detector: card_det,
        evidence: &card_ev,
        score: None,
        normalized: planted_card_norm,
    });

    let badge_ev = [PiiEvidence::Rule { rule: 7 }, PiiEvidence::Label { span: Span::new(25, 30), phrase: 1 }];
    let badge_score =
        Some(PiiScore { value: 0.925, kind: PiiScoreKind::Probability, model: badge_model, calibrated: true });
    pii.push(PiiMention {
        span: Span::new(31, 39),
        kind: badge_kind,
        detector: badge_det,
        evidence: &badge_ev,
        score: badge_score,
        normalized: planted_badge_norm,
    });
    doc.add(&Key::DEFAULT, pii, Provenance::by("turbo-pii:pii")).unwrap();

    let mut candidates = PiiCandidates::with_tables(ViewId::SOURCE, custom_kinds, detectors);
    candidates.push(PiiCandidate {
        span: Span::new(4, 23),
        kind: PiiKind::Card,
        detector: card_det,
        evidence: &card_ev,
        score: None,
        disposition: PiiDisposition::Kept { mention: Ix::new(0) },
    });
    candidates.push(PiiCandidate {
        span: Span::new(31, 39),
        kind: badge_kind,
        detector: badge_det,
        evidence: &badge_ev,
        score: badge_score,
        disposition: PiiDisposition::Kept { mention: Ix::new(1) },
    });
    candidates.push(PiiCandidate {
        span: Span::new(35, 39),
        kind: PiiKind::PhoneExtension,
        detector: card_det,
        evidence: &[PiiEvidence::Shape],
        score: None,
        disposition: PiiDisposition::RejectedByRule { reason: RejectReason::LabelRequired },
    });
    candidates.push(PiiCandidate {
        span: Span::new(44, 64),
        kind: PiiKind::Email,
        detector: email_det,
        evidence: &[PiiEvidence::Shape, PiiEvidence::DenyHit { list: 0, term: 3 }],
        score: None,
        disposition: PiiDisposition::AllowedByRule { scope: AllowScope::Global, rule: AllowRule::Term(2) },
    });
    let mut cand_prov = Provenance::by("turbo-pii:pii");
    cand_prov.on = vec![Key::<Pii>::DEFAULT.id()];
    doc.add(&Key::DEFAULT, candidates, cand_prov).unwrap();

    let reg = FormatterRegistry::standard();

    // 1. Default export (`detail` off): mentions carry detector, evidence and score;
    // candidates and normalized values are absent from every byte of both formats.
    let (_, jsonld_default_bytes) = reg.format_to_vec("jsonld", "doc-pii", &doc, ExportOffsetMode::Utf8).unwrap();
    let jsonld_default = String::from_utf8(jsonld_default_bytes).unwrap();
    assert!(!jsonld_default.contains(planted_card_norm), "{jsonld_default}");
    assert!(!jsonld_default.contains(planted_badge_norm), "{jsonld_default}");
    assert!(!jsonld_default.contains("turbo:pii-candidates"), "{jsonld_default}");
    assert!(!jsonld_default.contains("turbo:PiiCandidate"), "{jsonld_default}");
    assert!(!jsonld_default.contains("allowedByRule"), "{jsonld_default}");

    assert!(
        jsonld_default.contains(
            "{\"@type\": \"turbo:PiiMention\", \"kind\": \"card\", \"confidence\": \"checksum\", \"start\": 4, \"end\": 23, \
             \"detector\": {\"id\": \"turbo-pii:cursor\", \"version\": \"0.1.0\"}, \
             \"evidence\": [{\"checksum\": {\"algorithm\": \"luhn\", \"passed\": true}}]}"
        ),
        "{jsonld_default}"
    );
    assert!(
        jsonld_default.contains(
            "{\"@type\": \"turbo:PiiMention\", \"kind\": \"employee-badge\", \"confidence\": \"labelled\", \"start\": 31, \"end\": 39, \
             \"detector\": {\"id\": \"acme:badge-scanner\", \"version\": \"2.3\"}, \
             \"evidence\": [{\"rule\": {\"rule\": 7}}, {\"label\": {\"span\": {\"start\": 25, \"end\": 30}, \"phrase\": 1}}], \
             \"score\": {\"value\": 0.925000, \"kind\": \"probability\", \
             \"modelRef\": {\"id\": \"acme:badge-Ranker\", \"version\": \"1.0\"}, \"calibrated\": true}}"
        ),
        "{jsonld_default}"
    );

    let (_, md_default_bytes) = reg.format_to_vec("markdown", "doc-pii", &doc, ExportOffsetMode::Utf8).unwrap();
    let md_default = String::from_utf8(md_default_bytes).unwrap();
    assert!(!md_default.contains(planted_card_norm), "{md_default}");
    assert!(!md_default.contains(planted_badge_norm), "{md_default}");
    assert!(!md_default.contains("## PII Candidates"), "{md_default}");
    assert!(!md_default.contains("allowed-by-rule"), "{md_default}");

    let md_summary = md_default.split("## PII Summary\n\n").nth(1).expect("PII Summary present");
    assert!(
        md_summary.contains(
            "- **card** [4..23] (confidence: `checksum`, detector: `turbo-pii:cursor@0.1.0`, evidence: luhn ok)\n"
        ),
        "{md_summary}"
    );
    assert!(
        md_summary.contains(
            "- **employee-badge** [31..39] (confidence: `labelled`, detector: `acme:badge-scanner@2.3`, \
             evidence: rule 7, label 25..30, score: 0.9250 (probability, model: `acme:badge-Ranker@1.0`, calibrated: `true`))\n"
        ),
        "{md_summary}"
    );
    // Label text ("badge") is never written into the PII summary.
    assert!(!md_summary.contains("badge "), "{md_summary}");

    // 2. Export with `detail = true`: candidates section/array is included with dispositions and no values.
    let jsonld_detail =
        JsonLdFormatter.format_to_string_with_detail("doc-pii", &doc, ExportOffsetMode::Utf8, true).unwrap();
    assert!(!jsonld_detail.contains(planted_card_norm), "{jsonld_detail}");
    assert!(!jsonld_detail.contains(planted_badge_norm), "{jsonld_detail}");
    assert!(jsonld_detail.contains("\"turbo:pii-candidates\": ["), "{jsonld_detail}");
    assert!(jsonld_detail.contains("\"disposition\": {\"kept\": 0}"), "{jsonld_detail}");
    assert!(jsonld_detail.contains("\"disposition\": {\"kept\": 1}"), "{jsonld_detail}");
    assert!(
        jsonld_detail.contains(
            "{\"@type\": \"turbo:PiiCandidate\", \"kind\": \"email\", \"confidence\": \"format\", \"start\": 44, \"end\": 64, \
             \"detector\": {\"id\": \"turbo-pii:email\", \"version\": \"0.1.0\"}, \
             \"evidence\": [{\"shape\": {}}, {\"denyHit\": {\"list\": 0, \"term\": 3}}], \
             \"disposition\": {\"allowedByRule\": {\"scope\": \"global\", \"rule\": {\"term\": 2}}}}"
        ),
        "{jsonld_detail}"
    );
    assert!(jsonld_detail.contains("\"disposition\": {\"rejectedByRule\": \"label-required\"}"), "{jsonld_detail}");

    let md_detail =
        MarkdownFormatter.format_to_string_with_detail("doc-pii", &doc, ExportOffsetMode::Utf8, true).unwrap();
    assert!(!md_detail.contains(planted_card_norm), "{md_detail}");
    assert!(!md_detail.contains(planted_badge_norm), "{md_detail}");
    let md_cands = md_detail.split("## PII Candidates\n\n").nth(1).expect("PII Candidates section present");
    assert!(
        md_cands.contains(
            "- **card** [4..23] (disposition: `kept 0`, detector: `turbo-pii:cursor@0.1.0`, evidence: luhn ok)\n"
        ),
        "{md_cands}"
    );
    assert!(
        md_cands.contains(
            "- **employee-badge** [31..39] (disposition: `kept 1`, detector: `acme:badge-scanner@2.3`, \
             evidence: rule 7, label 25..30, score: 0.9250 (probability, model: `acme:badge-Ranker@1.0`, calibrated: `true`))\n"
        ),
        "{md_cands}"
    );
    assert!(
        md_cands.contains(
            "- **email** [44..64] (disposition: `allowed-by-rule (global term 2)`, \
             detector: `turbo-pii:email@0.1.0`, evidence: shape, deny-hit 0:3)\n"
        ),
        "{md_cands}"
    );
    assert!(
        md_cands.contains(
            "- **phone-extension** [35..39] (disposition: `rejected-by-rule (label-required)`, \
             detector: `turbo-pii:cursor@0.1.0`, evidence: shape)\n"
        ),
        "{md_cands}"
    );
    assert!(!md_cands.contains("support@acme.example"), "{md_cands}");
}
