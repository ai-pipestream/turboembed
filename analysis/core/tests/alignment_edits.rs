//! Alignments built by `ViewBuilder` and used through `Alignment` and
//! `Document`, with edits that touch each other, are separated by a single
//! byte, sit at the ends of the text, or are stacked across two views.
//!
//! If `ViewBuilder::replace` merged an edit into the one before it when they
//! touch, five of these tests would fail (the per-character ones: fifty
//! characters, every offset, the chain of two views, the zero-length keep and
//! the edit count), because a merged edit maps each character to the whole
//! run it was merged into.

use turbo_analysis::text::{Edit, Side};
use turbo_analysis::*;

/// Text with characters of every UTF-8 width, 1 to 4 bytes, repeated
/// until it has `n` characters.
fn mixed_width_text(n: usize) -> String {
    const CYCLE: [char; 8] = ['a', '\u{e9}', '\u{20ac}', '\u{1f600}', 'z', '\u{3b1}', '\u{4e2d}', '\u{10348}'];
    (0..n).map(|i| CYCLE[i % CYCLE.len()]).collect()
}

/// A character of a different width than `c`, chosen by `c`'s width.
fn other_width(c: char) -> char {
    match c.len_utf8() {
        1 => '\u{1f600}',
        2 => 'b',
        3 => '\u{e9}',
        _ => '\u{20ac}',
    }
}

/// Replaces every character of `text` with `f(c)`, one `replace` call each.
fn map_chars(text: &str, f: impl Fn(char) -> char) -> (String, Alignment) {
    let mut b = ViewBuilder::new(text);
    for (i, c) in text.char_indices() {
        b.replace(i + c.len_utf8(), &f(c).to_string());
    }
    b.finish()
}

/// The byte span of every character of `text`.
fn char_spans(text: &str) -> Vec<Span> {
    text.char_indices().map(|(i, c)| Span::new(i as u32, (i + c.len_utf8()) as u32)).collect()
}

/// Adds a view made from `parent` to the document and returns it with its id set.
fn add_view(doc: &mut Document, parent: &View, name: &'static str, text: String, alignment: Alignment) -> View {
    let key = Key::<View>::named(name);
    doc.add(&key, View::derived(parent, text, alignment), Provenance::by("test:view")).unwrap();
    doc.get_key(&key).unwrap().clone()
}

fn source_view(doc: &Document) -> View {
    doc.get::<View>().unwrap().clone()
}

// With merged touching edits, the 50 characters replaced one call each
// would become a single edit and every character would map to the whole run
// instead of its own bytes, so this test would fail.
#[test]
fn fifty_touching_replacements_each_map_to_their_own_bytes() {
    let src = mixed_width_text(50);
    let (out, a) = map_chars(&src, other_width);
    assert_eq!(a.edits().len(), 50);

    let parents = char_spans(&src);
    let children = char_spans(&out);
    assert_eq!(parents.len(), 50);
    assert_eq!(children.len(), 50);
    for (i, (p, c)) in parents.iter().zip(&children).enumerate() {
        assert_eq!(a.edits()[i], Edit { parent: *p, child: *c }, "edit {i}");
        assert_eq!(a.span_to_parent(*c), *p, "character {i} back to the source");
        assert_eq!(a.span_to_child(*p), *c, "character {i} forward to the view");
        assert_eq!(src[a.span_to_parent(*c).range()].chars().count(), 1);
        assert_eq!(out[a.span_to_child(*p).range()].chars().count(), 1);
    }
    // The edits touch: each one starts where the one before it ends.
    for pair in a.edits().windows(2) {
        assert_eq!(pair[0].parent.end, pair[1].parent.start);
        assert_eq!(pair[0].child.end, pair[1].child.start);
    }
    // A span over several characters covers exactly those characters.
    assert_eq!(
        a.span_to_parent(Span::new(children[10].start, children[19].end)),
        Span::new(parents[10].start, parents[19].end)
    );
    assert_eq!(
        a.span_to_child(Span::new(parents[10].start, parents[19].end)),
        Span::new(children[10].start, children[19].end)
    );
    // The whole text maps to the whole text.
    assert_eq!(a.span_to_parent(Span::new(0, out.len() as u32)), Span::new(0, src.len() as u32));
    assert_eq!(a.span_to_child(Span::new(0, src.len() as u32)), Span::new(0, out.len() as u32));
}

#[test]
fn every_offset_between_touching_replacements_maps_to_the_boundary() {
    let src = mixed_width_text(24);
    let (out, a) = map_chars(&src, other_width);
    let parents = char_spans(&src);
    let children = char_spans(&out);
    for (p, c) in parents.iter().zip(&children) {
        // The start of a character is the start of its output, whichever
        // side asks, and the end of a character is the end of its output.
        for side in [Side::Start, Side::End] {
            if p.start != 0 {
                assert_eq!(a.to_child(p.start, side), c.start, "parent {} {side:?}", p.start);
            }
            if c.start != 0 {
                assert_eq!(a.to_parent(c.start, side), p.start, "child {} {side:?}", c.start);
            }
        }
        assert_eq!(a.to_child(p.end, Side::End), c.end);
        assert_eq!(a.to_parent(c.end, Side::End), p.end);
        assert_eq!(a.to_child(p.start, Side::Start), c.start);
        assert_eq!(a.to_parent(c.start, Side::Start), p.start);
    }
}

#[test]
fn two_derived_views_with_touching_edits_map_each_character_through_the_chain() {
    let src = mixed_width_text(40);
    let mut doc = Document::new(src.clone());

    // View A: every character replaced by one of another width.
    let (a_text, a_align) = map_chars(&src, other_width);
    let source = source_view(&doc);
    let a_view = add_view(&mut doc, &source, "a", a_text.clone(), a_align);
    let a = a_view.id();

    // View B, made from A: every character replaced again, so no character
    // of B has the width of its source character in general.
    let (b_text, b_align) = map_chars(&a_text, |c| if c.len_utf8() == 4 { 'x' } else { '\u{1f600}' });
    let b = add_view(&mut doc, &a_view, "b", b_text.clone(), b_align).id();

    assert_eq!(doc.text(a), a_text);
    assert_eq!(doc.text(b), b_text);
    let source_chars = char_spans(&src);
    let a_chars = char_spans(&a_text);
    let b_chars = char_spans(&b_text);
    assert_eq!(b_chars.len(), 40);
    for i in 0..40 {
        assert_eq!(doc.to_source(b, b_chars[i]), source_chars[i], "B character {i} to the source");
        assert_eq!(doc.from_source(b, source_chars[i]), b_chars[i], "source character {i} to B");
        assert_eq!(doc.to_source(a, a_chars[i]), source_chars[i], "A character {i} to the source");
        assert_eq!(doc.from_source(a, source_chars[i]), a_chars[i], "source character {i} to A");
    }
    // A run of characters crosses both alignments as a run.
    let run = Span::new(source_chars[5].start, source_chars[14].end);
    let in_b = doc.from_source(b, run);
    assert_eq!(in_b, Span::new(b_chars[5].start, b_chars[14].end));
    assert_eq!(doc.to_source(b, in_b), run);
}

#[test]
fn a_chain_where_the_second_view_edits_between_the_first_views_edits() {
    // View A replaces the odd characters only; view B (made from A)
    // replaces the even ones, which A kept. B's edits touch each other and
    // sit on bytes A did not change.
    let src = "abcdefgh";
    let mut doc = Document::new(src);
    let mut b1 = ViewBuilder::new(src);
    for (i, c) in src.char_indices() {
        if i % 2 == 1 {
            b1.replace(i + 1, &c.to_uppercase().to_string().repeat(2));
        } else {
            b1.keep(i + 1);
        }
    }
    let (a_text, a_align) = b1.finish();
    assert_eq!(a_text, "aBBcDDeFFgHH");
    assert_eq!(a_align.edits().len(), 4);
    let source = source_view(&doc);
    let a = add_view(&mut doc, &source, "a", a_text.clone(), a_align);

    let mut b2 = ViewBuilder::new(&a_text);
    b2.replace(1, "\u{e9}");
    b2.keep(3);
    b2.replace(4, "\u{20ac}");
    b2.keep(6);
    b2.replace(7, "\u{1f600}");
    b2.keep(9);
    b2.replace(10, "!");
    b2.keep(12);
    let (b_text, b_align) = b2.finish();
    assert_eq!(b_text, "\u{e9}BB\u{20ac}DD\u{1f600}FF!HH");
    let b = add_view(&mut doc, &a, "b", b_text, b_align).id();

    // Source bytes 0, 2, 4, 6 are the characters B replaced.
    for (i, src_span) in [(0, Span::new(0, 1)), (1, Span::new(2, 3)), (2, Span::new(4, 5)), (3, Span::new(6, 7))] {
        let in_b = doc.from_source(b, src_span);
        assert_eq!(doc.to_source(b, in_b), src_span, "source byte {} round trip", src_span.start);
        let width = ["\u{e9}", "\u{20ac}", "\u{1f600}", "!"][i].len() as u32;
        assert_eq!(in_b.len(), width, "source byte {} width in B", src_span.start);
    }
    // The doubled letters A made stay whole through B.
    let bb = doc.from_source(b, Span::new(1, 2));
    assert_eq!(&doc.text(b)[bb.range()], "BB");
    assert_eq!(doc.to_source(b, bb), Span::new(1, 2));
}

#[test]
fn a_replace_after_a_zero_length_keep_is_an_ordinary_edit() {
    let src = "ab\u{20ac}d";
    let mut b = ViewBuilder::new(src);
    b.keep(0);
    b.replace(1, "A");
    b.keep(1);
    b.replace(2, "B");
    b.keep(2);
    b.keep(2);
    b.replace(5, "E");
    b.keep(5);
    let (out, a) = b.finish();
    assert_eq!(out, "ABEd");
    assert_eq!(
        a.edits(),
        [
            Edit { parent: Span::new(0, 1), child: Span::new(0, 1) },
            Edit { parent: Span::new(1, 2), child: Span::new(1, 2) },
            Edit { parent: Span::new(2, 5), child: Span::new(2, 3) },
        ]
    );
    assert_eq!(a.span_to_parent(Span::new(2, 3)), Span::new(2, 5));
    assert_eq!(a.span_to_child(Span::new(2, 5)), Span::new(2, 3));
    assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(5, 6));
}

#[test]
fn edits_separated_by_one_kept_byte_keep_that_byte_exact() {
    // Every other byte is replaced, so every edit is one kept byte from the
    // next, and the replacements change the width.
    let src = "a-b-c-d-e";
    let mut b = ViewBuilder::new(src);
    for i in (0..src.len()).step_by(2) {
        b.replace(i + 1, "\u{20ac}");
        if i + 1 < src.len() {
            b.keep(i + 2);
        }
    }
    let (out, a) = b.finish();
    assert_eq!(out, "\u{20ac}-\u{20ac}-\u{20ac}-\u{20ac}-\u{20ac}");
    assert_eq!(a.edits().len(), 5);
    for (n, i) in (0..src.len() as u32).step_by(2).enumerate() {
        let parent = Span::new(i, i + 1);
        let child = Span::new(4 * n as u32, 4 * n as u32 + 3);
        assert_eq!(a.edits()[n], Edit { parent, child });
        // The kept dash after this edit.
        if i + 1 < src.len() as u32 {
            let dash_in_src = Span::new(i + 1, i + 2);
            let dash_in_out = Span::new(child.end, child.end + 1);
            assert_eq!(a.span_to_child(dash_in_src), dash_in_out);
            assert_eq!(a.span_to_parent(dash_in_out), dash_in_src);
            assert_eq!(&out[dash_in_out.range()], "-");
        }
    }
    // A span across a kept byte and the edits on both sides of it.
    assert_eq!(a.span_to_parent(Span::new(0, 7)), Span::new(0, 3));
    assert_eq!(a.span_to_child(Span::new(0, 3)), Span::new(0, 7));
}

#[test]
fn an_insertion_at_the_start_leaves_the_text_after_it_in_place() {
    let src = "h\u{e9}llo";
    let mut b = ViewBuilder::new(src);
    b.replace(0, ">> ");
    let (out, a) = b.finish();
    assert_eq!(out, ">> h\u{e9}llo");
    assert_eq!(a.edits(), [Edit { parent: Span::new(0, 0), child: Span::new(0, 3) }]);
    // The inserted bytes belong to no source bytes: a start offset lands
    // after them, an end offset before them.
    assert_eq!(a.to_child(0, Side::Start), 3);
    assert_eq!(a.to_child(0, Side::End), 0);
    assert_eq!(a.to_parent(3, Side::Start), 0);
    assert_eq!(a.to_parent(3, Side::End), 0);
    // The rest of the text keeps its bytes, shifted by the insertion.
    assert_eq!(a.span_to_child(Span::new(0, src.len() as u32)), Span::new(3, out.len() as u32));
    assert_eq!(a.span_to_parent(Span::new(3, out.len() as u32)), Span::new(0, src.len() as u32));
    assert_eq!(a.span_to_parent(Span::new(4, 6)), Span::new(1, 3));
    assert_eq!(a.to_child(src.len() as u32, Side::End), out.len() as u32);
    assert_eq!(a.to_parent(out.len() as u32, Side::End), src.len() as u32);
}

#[test]
fn an_insertion_at_the_end_leaves_the_text_before_it_in_place() {
    let src = "h\u{e9}llo";
    let mut b = ViewBuilder::new(src);
    b.keep(src.len());
    b.replace(src.len(), " <<");
    let (out, a) = b.finish();
    assert_eq!(out, "h\u{e9}llo <<");
    let n = src.len() as u32;
    assert_eq!(a.edits(), [Edit { parent: Span::new(n, n), child: Span::new(n, n + 3) }]);
    assert_eq!(a.span_to_child(Span::new(0, n)), Span::new(0, n));
    assert_eq!(a.span_to_parent(Span::new(0, n)), Span::new(0, n));
    assert_eq!(a.to_child(n, Side::Start), n + 3);
    assert_eq!(a.to_child(n, Side::End), n);
    assert_eq!(a.to_parent(n, Side::End), n);
    assert_eq!(a.to_parent(n + 3, Side::Start), n);
    assert_eq!(a.to_parent(n + 3, Side::End), n);
}

#[test]
fn deleting_the_whole_text_is_one_edit_and_empties_the_view() {
    let src = "gone \u{1f600}";
    let n = src.len() as u32;
    let mut b = ViewBuilder::new(src);
    b.replace(src.len(), "");
    let (out, a) = b.finish();
    assert_eq!(out, "");
    assert_eq!(a.edits(), [Edit { parent: Span::new(0, n), child: Span::new(0, 0) }]);
    // Every source offset lands at 0 of the empty view.
    for x in 0..=n {
        for side in [Side::Start, Side::End] {
            assert_eq!(a.to_child(x, side), 0, "parent {x} {side:?}");
        }
    }
    assert_eq!(a.span_to_child(Span::new(0, n)), Span::new(0, 0));
    assert_eq!(a.span_to_child(Span::new(2, 4)), Span::new(0, 0));
    // The one offset of the empty view is the start or the end of the
    // deleted text, depending on the side.
    assert_eq!(a.to_parent(0, Side::Start), n);
    assert_eq!(a.to_parent(0, Side::End), 0);
}

#[test]
fn a_document_maps_through_a_view_that_inserted_at_both_ends() {
    let src = "mid";
    let mut doc = Document::new(src);
    let mut b = ViewBuilder::new(src);
    b.replace(0, "[");
    b.keep(3);
    b.replace(3, "]");
    let (text, a) = b.finish();
    assert_eq!(text, "[mid]");
    assert_eq!(a.edits().len(), 2);
    let source = source_view(&doc);
    let v = add_view(&mut doc, &source, "brackets", text, a).id();
    assert_eq!(doc.from_source(v, Span::new(0, 3)), Span::new(1, 4));
    assert_eq!(doc.to_source(v, Span::new(1, 4)), Span::new(0, 3));
    // A span that includes a bracket widens to the neighbouring source
    // bytes only, never past the text.
    assert_eq!(doc.to_source(v, Span::new(0, 5)), Span::new(0, 3));
    assert_eq!(doc.to_source(v, Span::new(0, 2)), Span::new(0, 1));
    assert_eq!(doc.to_source(v, Span::new(3, 5)), Span::new(2, 3));
}

#[test]
fn the_edit_count_is_the_number_of_calls_that_changed_something() {
    let src = "abcdef\u{20ac}gh";
    let mut b = ViewBuilder::new(src);
    let mut changed = 0;
    // Consumes a byte, writes a byte: an edit, even though the text is the same.
    b.replace(1, "a");
    changed += 1;
    // Consumes and writes nothing: no edit.
    b.replace(1, "");
    // Inserts: an edit.
    b.replace(1, "+");
    changed += 1;
    // Deletes: an edit.
    b.replace(2, "");
    changed += 1;
    // Keeps do not count, and a zero-length keep does nothing.
    b.keep(2);
    b.keep(2);
    // Two replaces in a row that touch: two edits.
    b.replace(3, "C");
    b.replace(4, "D");
    changed += 2;
    // An empty call between them again records nothing.
    b.replace(4, "");
    b.replace(5, "E");
    changed += 1;
    b.replace(5, "");
    // Replaces for the f and the three-byte euro sign, then an insertion that touches them.
    b.replace(6, "F");
    b.replace(9, "|");
    b.replace(9, "#");
    changed += 3;
    let (out, a) = b.finish();
    assert_eq!(out, "a+CDEF|#gh");
    assert_eq!(a.edits().len(), changed);
    // Edits are in text order on both sides and do not overlap.
    for pair in a.edits().windows(2) {
        assert!(pair[0].parent.end <= pair[1].parent.start);
        assert!(pair[0].child.end <= pair[1].child.start);
    }
    // The g after the last edit keeps its own byte, shifted.
    assert_eq!(a.span_to_child(Span::new(9, 10)), Span::new(8, 9));
    assert_eq!(a.span_to_parent(Span::new(8, 9)), Span::new(9, 10));
}
