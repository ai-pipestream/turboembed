//! Every reject reason of a `turbo:pii-candidates` row reaches the C API
//! under its own `TURBO_PII_REJECT_*` id (#2286).

use turbo::{
    TURBO_ANALYZER_PII_DETAIL, TURBO_ANALYZER_PII_KIND_IDS, TURBO_PII_DISPOSITION_KEPT,
    TURBO_PII_DISPOSITION_REJECTED_BY_RULE, TURBO_PII_REJECT_LABEL_REQUIRED, TURBO_PII_REJECT_RULE_PRECEDENCE,
    TURBO_PII_REJECT_THRESHOLD, turbo_analysis_result, turbo_analysis_result_pii_candidates,
    turbo_analysis_result_release, turbo_analyzer, turbo_analyzer_create, turbo_analyzer_desc,
    turbo_analyzer_inspect_graph, turbo_analyzer_release, turbo_error, turbo_pii_candidate,
};
use turbo_analysis::{Codecs, Document, Ix, Key, Provenance, Span, ViewId, export};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{
    Pii, PiiCandidate, PiiCandidates, PiiDisposition, PiiEvidence, PiiKind, PiiMention, RejectReason,
};

/// One kept mention and one candidate rejected for each reason, as a graph.
fn graph() -> Vec<u8> {
    let mut doc = Document::new("a@b.co c@d.co");
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test:det", "1").unwrap();
    let (kind, score, evidence) = (PiiKind::Email, None, [PiiEvidence::Shape]);
    pii.push(PiiMention { span: Span::new(0, 6), kind, detector, evidence: &evidence, score, normalized: "a@b.co" });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    let mut candidates = PiiCandidates::new(ViewId::SOURCE);
    let detector = candidates.detectors.add("test:det", "1").unwrap();
    let rejected = |reason| PiiDisposition::RejectedByRule { reason };
    for (span, disposition) in [
        (Span::new(0, 6), PiiDisposition::Kept { mention: Ix::new(0) }),
        (Span::new(7, 13), rejected(RejectReason::LabelRequired)),
        (Span::new(7, 13), rejected(RejectReason::Threshold)),
        (Span::new(7, 13), rejected(RejectReason::RulePrecedence)),
    ] {
        candidates.push(PiiCandidate { span, kind, detector, evidence: &evidence, score, disposition });
    }
    let mut by = Provenance::by("test:pii");
    by.on.push(Key::<Pii>::DEFAULT.id());
    doc.add(&Key::<PiiCandidates>::DEFAULT, candidates, by).unwrap();
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    export(&doc, &codecs).unwrap().to_bytes().unwrap()
}

#[test]
fn each_reject_reason_has_its_own_id_in_the_c_api() {
    let wire = graph();
    unsafe {
        let mut err: turbo_error = std::mem::zeroed();
        err.struct_size = size_of::<turbo_error>() as u32;
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.flags = TURBO_ANALYZER_PII_KIND_IDS | TURBO_ANALYZER_PII_DETAIL;
        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), turbo::status::OK);
        let mut result: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_inspect_graph(analyzer, wire.as_ptr(), wire.len() as u64, &mut result, &mut err),
            turbo::status::OK
        );
        let (mut rows, mut count) = (std::ptr::null::<turbo_pii_candidate>(), 0u32);
        assert_eq!(turbo_analysis_result_pii_candidates(result, &mut rows, &mut count, &mut err), turbo::status::OK);
        let rows = std::slice::from_raw_parts(rows, count as usize);
        let decided: Vec<(u32, u32)> = rows.iter().map(|c| (c.disposition, c.disposition_a)).collect();
        assert_eq!(
            decided,
            [
                (TURBO_PII_DISPOSITION_KEPT, 0),
                (TURBO_PII_DISPOSITION_REJECTED_BY_RULE, TURBO_PII_REJECT_LABEL_REQUIRED),
                (TURBO_PII_DISPOSITION_REJECTED_BY_RULE, TURBO_PII_REJECT_THRESHOLD),
                (TURBO_PII_DISPOSITION_REJECTED_BY_RULE, TURBO_PII_REJECT_RULE_PRECEDENCE),
            ]
        );
        turbo_analysis_result_release(result);
        turbo_analyzer_release(analyzer);
    }
}
