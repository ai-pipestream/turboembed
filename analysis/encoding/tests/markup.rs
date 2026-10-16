//! A windows-1252 HTML file, decoded, then stripped of its markup the way
//! an extractor would report it: a word in the extracted text maps through
//! the markup view and view 0 to the file's bytes.

use turbo_analysis::text::Side;
use turbo_analysis::{Key, MarkupText, Provenance, Span, TagKind, View};
use turbo_encoding::{Encoding, open};

#[test]
fn extracted_words_map_to_the_bytes_of_the_file() {
    let bytes = b"<p>Caf\xE9 &amp; cr\xE8me</p>".to_vec();
    let mut doc = open(bytes.clone(), Some(Encoding::Windows1252)).unwrap();
    let html = doc.source().to_string();
    assert_eq!(html, "<p>Café &amp; crème</p>");
    // The extractor works on view 0 and reports offsets in it.
    let at = |s: &str| html.find(s).unwrap() as u32;
    let (amp, cr) = (at("&amp;"), at("crème"));
    let mut m = MarkupText::new();
    m.push_tag(Span::new(0, 3), TagKind::Start);
    m.push_copied(&html, Span::new(3, amp));
    m.push_decoded(Span::new(amp, amp + 5), "&");
    m.push_copied(&html, Span::new(amp + 5, cr + 6));
    m.push_tag(Span::new(cr + 6, cr + 10), TagKind::End);
    let view = m.view(doc.get::<View>().unwrap()).unwrap();
    doc.add(&Key::<View>::named("markup"), view, Provenance::by("test")).unwrap();
    let v = doc.get_key(&Key::<View>::named("markup")).unwrap().id();
    assert_eq!(doc.text(v), "Café & crème");

    // "Café" is bytes 3..7 of the file, "crème" bytes 14..19, "&" the
    // five bytes of the entity.
    assert_eq!(doc.to_source_bytes(v, Span::new(0, 5)), Span::new(3, 7));
    assert_eq!(doc.to_source_bytes(v, Span::new(8, 14)), Span::new(14, 19));
    assert_eq!(doc.to_source_bytes(v, Span::new(6, 7)), Span::new(8, 13));
    assert_eq!(&bytes[14..19], b"cr\xE8me");
    // And back: the bytes of "crème" are the word in the extracted text,
    // the bytes of a tag are no text.
    assert_eq!(doc.from_source_bytes(v, Span::new(14, 19)), Span::new(8, 14));
    assert_eq!(doc.from_source_bytes(v, Span::new(0, 3)), Span::new(0, 0));
    assert_eq!(doc.offset_to_source_bytes(v, 14, Side::End), 19);
    // An offset inside the é (one byte in the file, two in the text)
    // widens to that byte.
    assert_eq!(doc.to_source_bytes(v, Span::new(3, 5)), Span::new(6, 7));
}
