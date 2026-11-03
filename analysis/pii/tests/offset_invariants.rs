//! Pack-level offset invariants kept after the fuzz differential PII runs
//! were removed: `check_document` (source projections, UTF-16/code-point
//! conversions) plus `OffsetTable` round-trips on every mention edge.
//! Not blessable expected-output.

use turbo_analysis::{Analyzer, Document, OffsetTable, ViewId};
use turbo_goldens::invariants::check_document;
use turbo_pii::{Pack, Pii, PiiDetector};

fn challenging_texts() -> [&'static str; 7] {
    [
        "email me at user@example.com please",
        "card 4111-1111-1111-1111 expires",
        "call +1 (415) 555-2671 now",
        "ssn 078-05-1120 filed",
        // Non-BMP emoji before a hit.
        "😀 email user@example.com",
        // Precomposed accent.
        "café user@example.com",
        // Combining acute on 'a' before a hit.
        "a\u{0301} user@example.com",
    ]
}

#[test]
fn every_pack_keeps_offset_invariants_on_challenging_texts() {
    let mut layers = 0;
    let mut mentions = 0;
    for pack in Pack::ALL {
        let analyzer = Analyzer::builder().add(PiiDetector::pack(pack).keeping_nested()).build().expect("analyzer");
        for text in challenging_texts() {
            let table = OffsetTable::build(text);
            let mut docs = vec![Document::new(text)];
            for result in analyzer.run(&mut docs) {
                result.unwrap_or_else(|e| panic!("{}/{text:?}: {e}", pack.name()));
            }
            let (checked, violations) = check_document(&docs[0]);
            assert!(violations.is_empty(), "{} on {text:?}: {:?}", pack.name(), violations);
            layers += checked;
            let empty = Pii::new(ViewId::SOURCE);
            let layer = docs[0].get::<Pii>().unwrap_or(&empty);
            for m in layer.mentions() {
                mentions += 1;
                for edge in [m.span.start, m.span.end] {
                    let u16 = table.to_utf16(text, edge);
                    let cp = table.to_code_point(text, edge);
                    assert_eq!(table.from_utf16(text, u16).unwrap(), edge);
                    assert_eq!(table.from_code_point(text, cp).unwrap(), edge);
                }
            }
        }
    }
    assert!(layers > 0, "no PII layers checked");
    assert!(mentions > 0, "no mentions to round-trip");
}
