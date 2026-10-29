//! Native behavior contracts independent of historical JVM expected outputs.
//! The full seeded input walk and layer checks live in `invariants.rs`; the
//! normalizer boundary contracts live with the normalizers in
//! `analysis/normalize/tests/native_projection.rs`.

use turbo_analysis::{Document, Ix, Span, View, ViewId};
use turbo_analysis_layers::Sentences;
use turbo_fuzz::{Resources, runs};
use turbo_goldens::invariants::{check_document, check_spans};

fn scalar_contract(doc: &Document, view: &View) {
    let starts: Vec<_> = view.text().char_indices().map(|(offset, _)| offset as u32).collect();
    let ends: Vec<_> = starts.iter().skip(1).copied().chain([view.text().len() as u32]).take(starts.len()).collect();
    let violations = check_spans(doc, "native scalar boundaries", view.id(), &starts, &ends);
    assert!(violations.is_empty(), "{violations:?}");
    let offsets = doc.offsets(view.id());
    for byte in starts.into_iter().chain([view.text().len() as u32]) {
        let prefix = &view.text()[..byte as usize];
        assert_eq!(offsets.to_utf16(view.text(), byte).0 as usize, prefix.encode_utf16().count());
        assert_eq!(offsets.to_code_point(view.text(), byte).0 as usize, prefix.chars().count());
    }
}

#[test]
fn newline_spans_follow_native_crlf_and_unicode_trim_contract() {
    let text = " 😀\r\nZoë\n\n東京 ";
    let run = runs::open("sentdetect", "newline", &Resources::new("unused-model-directory")).unwrap();
    let doc = run.annotate(text, None).unwrap();
    let sentences = doc.get::<Sentences>().unwrap();
    assert_eq!(sentences.spans.len(), 3);
    for (index, (start, end)) in [(1, 5), (7, 11), (13, 19)].into_iter().enumerate() {
        assert_eq!(sentences.spans.get(Ix::new(index as u32)), Span::new(start, end));
    }
    let (layers, violations) = check_document(&doc);
    assert!(layers > 0);
    assert!(violations.is_empty(), "{violations:?}");
    scalar_contract(&doc, doc.get::<View>().unwrap());
}

#[test]
fn native_checker_rejects_corrupt_span_columns_without_skipping() {
    let doc = Document::new("😀é");
    assert!(check_spans(&doc, "valid", ViewId::SOURCE, &[0, 4], &[4, 6]).is_empty());
    for (starts, ends, reason) in [
        (vec![1], vec![4], "cuts a character"),
        (vec![0], vec![5], "cuts a character"),
        (vec![0], vec![7], "not inside"),
        (vec![4], vec![0], "not inside"),
        (vec![0, 4], vec![4], "starts and"),
    ] {
        let violations = check_spans(&doc, "corrupt", ViewId::SOURCE, &starts, &ends);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].what.contains(reason), "{violations:?}");
    }
}
