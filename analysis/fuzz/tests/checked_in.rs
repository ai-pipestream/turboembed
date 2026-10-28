//! Native behavior contracts independent of historical JVM expected outputs.
//! The full seeded input walk and layer checks live in `invariants.rs`.

use turbo_analysis::{Document, Ix, Key, Span, View, ViewId};
use turbo_analysis_layers::Sentences;
use turbo_fuzz::{Resources, runs};
use turbo_goldens::invariants::{check_document, check_spans};

fn normalize(name: &str, text: &str) -> Document {
    let run = runs::open("normalize", name, &Resources::new("unused-model-directory")).unwrap();
    run.annotate(text, None).unwrap()
}

fn normalized(doc: &Document) -> &View {
    doc.get_key::<View>(&Key::named("norm")).unwrap()
}

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
fn turkish_lowercase_has_exact_native_text_and_source_boundaries() {
    for (input, expected) in [
        ("Iİiı", "ıiiı"),
        ("I\u{307}", "i"),
        ("İSTANBUL IĞDIR", "istanbul ığdır"),
        ("😀 I İ café 東京", "😀 ı i café 東京"),
        ("", ""),
    ] {
        let doc = normalize("case-fold-tr", input);
        let view = normalized(&doc);
        assert_eq!(view.text(), expected, "{input:?}");
        scalar_contract(&doc, view);
        // The trailing combining dot is deleted and contributes no output bytes.
        let contributing_end = if input == "I\u{307}" { 1 } else { input.len() as u32 };
        assert_eq!(doc.to_source(view.id(), Span::new(0, expected.len() as u32)), Span::new(0, contributing_end));
    }
    let doc = normalize("case-fold-tr", "Iİiı");
    let view = normalized(&doc);
    for (child, source) in [((0, 2), (0, 1)), ((2, 3), (1, 3)), ((3, 4), (3, 4)), ((4, 6), (4, 6))] {
        assert_eq!(doc.to_source(view.id(), Span::new(child.0, child.1)), Span::new(source.0, source.1));
    }
}

#[test]
fn native_expansion_and_deletion_keep_source_projection() {
    for (name, input, expected) in [
        ("ellipsis", "…\u{fe19}", "......"),
        ("invisible", "a\u{200b}b", "ab"),
        ("full-case-fold", "Straße", "strasse"),
        ("whitespace", "😀  beta", "😀 beta"),
    ] {
        let doc = normalize(name, input);
        let view = normalized(&doc);
        assert_eq!(view.text(), expected, "{name}");
        scalar_contract(&doc, view);
        assert_eq!(doc.to_source(view.id(), Span::new(0, expected.len() as u32)), Span::new(0, input.len() as u32));
    }
    let doc = normalize("whitespace", "😀  beta");
    assert_eq!(doc.to_source(normalized(&doc).id(), Span::new(5, 9)), Span::new(6, 10));
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
