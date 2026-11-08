//! Unit tests of the in-process analyzer engine.

use turbo_analysis_layers::PiiKind;

use super::*;

#[test]
fn detect_pii_maps_labelled_to_4_and_labelled_checksum_to_3() {
    let mut engine = AnalyzerEngine::from_desc(AnalyzerConfig::default()).unwrap();
    engine.pii_detector = PiiDetector::pack(Pack::AllStructured).label_boosts(PiiKind::UsSsn);
    let result = engine.detect_pii("SSN: 123-45-6789, SIN: 046 454 286.").unwrap();
    let classes: Vec<(&str, u32)> = result
        .spans
        .iter()
        .map(|row| {
            let start = row.label_offset as usize;
            let end = start + row.label_len as usize;
            (std::str::from_utf8(&result.pool[start..end]).unwrap(), row.classification)
        })
        .collect();
    assert_eq!(
        classes,
        [("us-ssn", TURBO_PII_CONFIDENCE_LABELLED), ("ca-sin", TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM)]
    );
    assert_eq!(TURBO_PII_CONFIDENCE_LABELLED, 4);
    assert_eq!(TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM, 3);
}
