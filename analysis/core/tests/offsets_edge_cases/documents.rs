//! The tables of a document's views, and two hundred pseudo-random texts.

use super::*;

#[test]
fn document_views_each_have_a_table_for_their_own_text() {
    // A source with a ligature-able "fi" and a long tail, so the derived
    // views can be shorter and longer than it.
    let src = format!("fine {} caf\u{E9} {}", W4, "word ".repeat(120));
    let mut doc = Document::new(src.as_str());
    check_all(&src);

    // Shorter: two ASCII bytes become the 3 byte ligature (longer in bytes,
    // shorter in characters), and the long tail is dropped.
    let (shorter, alignment) = {
        let mut b = ViewBuilder::new(&src);
        b.replace(2, "\u{FB01}");
        b.keep(src.find("caf").unwrap());
        b.replace(src.len(), "");
        b.finish()
    };
    assert!(shorter.len() < src.len());
    let view = View::derived(doc.get::<View>().unwrap(), shorter.clone(), alignment);
    doc.add(&Key::named("shorter"), view, Provenance::by("t")).unwrap();

    // Longer: every ASCII space becomes a 4 byte character.
    let (longer, alignment) = {
        let mut b = ViewBuilder::new(&src);
        let mut at = 0;
        while let Some(i) = src[at..].find(' ') {
            let i = at + i;
            b.keep(i);
            b.replace(i + 1, "\u{1F600}");
            at = i + 1;
        }
        b.finish()
    };
    assert!(longer.len() > src.len() + 300);
    let view = View::derived(doc.get::<View>().unwrap(), longer.clone(), alignment);
    doc.add(&Key::named("longer"), view, Provenance::by("t")).unwrap();

    for (id, text) in [(ViewId::SOURCE, src.as_str()), (ViewId(1), shorter.as_str()), (ViewId(2), longer.as_str())] {
        assert_eq!(doc.text(id), text);
        let t = doc.offsets(id);
        assert_eq!(t.len() as usize, text.len());
        assert_eq!(t.utf16_len(), Utf16Offset(text.encode_utf16().count() as u32));
        assert_eq!(t.code_point_len(), CodePointOffset(text.chars().count() as u32));
        for (b, u, c) in boundaries(text) {
            assert_eq!(t.to_utf16(text, b), Utf16Offset(u));
            assert_eq!(t.to_code_point(text, b), CodePointOffset(c));
            assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(b));
            assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(b));
        }
        assert!(std::ptr::eq(t, doc.offsets(id)), "built once and kept");
    }
    // The three tables differ from each other.
    assert_ne!(doc.offsets(ViewId::SOURCE), doc.offsets(ViewId(1)));
    assert_ne!(doc.offsets(ViewId::SOURCE), doc.offsets(ViewId(2)));
    assert_ne!(doc.offsets(ViewId(1)), doc.offsets(ViewId(2)));
}

#[test]
fn document_source_view_matches_its_own_text() {
    let text = format!("{}{}", W3.to_string().repeat(100), "a".repeat(400));
    let doc = Document::new(text.as_str());
    let t = doc.offsets(ViewId::SOURCE);
    assert!(!t.is_ascii());
    assert_eq!(t.len() as usize, text.len());
    check_all(&text);
    assert_eq!(t, &OffsetTable::build(&text));
}

/// A small deterministic generator, so the test needs no crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

#[test]
fn two_hundred_pseudo_random_texts() {
    const ALPHABET: [char; 8] = ['a', 'Z', '\u{E9}', '\u{3B1}', '\u{20AC}', '\u{4E2D}', '\u{1D11E}', '\u{1F600}'];
    let mut r = Lcg(0x5EED);
    for _ in 0..200 {
        let limit = (r.next() % 3001) as usize;
        let mut text = String::new();
        while text.len() < limit {
            // Mostly mixed, with occasional ASCII-only runs so some
            // strides hold no wide characters.
            let c = if r.next().is_multiple_of(4) { 'q' } else { ALPHABET[(r.next() % 8) as usize] };
            if text.len() + c.len_utf8() > limit {
                break;
            }
            text.push(c);
        }
        assert!(text.len() <= 3000);
        check_all(&text);
    }
}
