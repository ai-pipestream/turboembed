use super::*;

#[test]
fn spans_map_both_ways_through_edits() {
    // "The ﬁnal  report" -> "The final report": a ligature expanded and
    // a run of spaces folded.
    let src = "The \u{FB01}nal  report";
    let mut b = ViewBuilder::new(src);
    b.keep(4);
    b.replace(7, "fi"); // U+FB01 is 3 bytes
    b.keep(10);
    b.replace(12, " ");
    let (out, a) = b.finish();
    assert_eq!(out, "The final report");
    // "final" in the view is the ligature and "nal" in the source.
    let final_view = Span::new(4, 9);
    assert_eq!(&src[a.span_to_parent(final_view).range()], "\u{FB01}nal");
    // "report" moves back by one byte.
    let report = Span::new(10, 16);
    assert_eq!(&out[report.range()], "report");
    assert_eq!(&src[a.span_to_parent(report).range()], "report");
    // And forward again.
    assert_eq!(a.span_to_child(Span::new(12, 18)), report);
    // An offset inside the ligature widens to the whole of it.
    assert_eq!(a.span_to_parent(Span::new(5, 6)), Span::new(4, 7));
}

#[test]
fn deleted_bytes_belong_to_neither_neighbour() {
    // A soft hyphen (2 bytes) removed: "co\u{AD}operate" -> "cooperate".
    let src = "co\u{AD}operate";
    let mut b = ViewBuilder::new(src);
    b.keep(2);
    b.replace(4, "");
    let (out, a) = b.finish();
    assert_eq!(out, "cooperate");
    assert_eq!(&src[a.span_to_parent(Span::new(0, 2)).range()], "co");
    assert_eq!(&src[a.span_to_parent(Span::new(2, 9)).range()], "operate");
    assert_eq!(&src[a.span_to_parent(Span::new(0, 9)).range()], src);
}

#[test]
fn characters_replaced_one_by_one_keep_their_own_source_bytes() {
    // Three Arabic-Indic digits (2 bytes each) folded to ASCII, one
    // replace call per digit.
    let src = "n=\u{661}\u{662}\u{663};";
    let mut b = ViewBuilder::new(src);
    b.keep(2);
    b.replace(4, "1");
    b.replace(6, "2");
    b.replace(8, "3");
    let (out, a) = b.finish();
    assert_eq!(out, "n=123;");
    assert_eq!(
        a.edits(),
        [
            Edit { parent: Span::new(2, 4), child: Span::new(2, 3) },
            Edit { parent: Span::new(4, 6), child: Span::new(3, 4) },
            Edit { parent: Span::new(6, 8), child: Span::new(4, 5) },
        ]
    );
    // The middle digit maps to its own source digit and back.
    assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(4, 6));
    assert_eq!(&src[a.span_to_parent(Span::new(3, 4)).range()], "\u{662}");
    assert_eq!(a.span_to_child(Span::new(4, 6)), Span::new(3, 4));
    // The whole number still maps to the whole run.
    assert_eq!(a.span_to_parent(Span::new(2, 5)), Span::new(2, 8));
    assert_eq!(a.span_to_child(Span::new(2, 8)), Span::new(2, 5));
}

#[test]
fn a_run_replaced_in_one_call_is_one_edit() {
    let src = "n=\u{661}\u{662}\u{663};";
    let mut b = ViewBuilder::new(src);
    b.keep(2);
    b.replace(8, "123");
    let (out, a) = b.finish();
    assert_eq!(out, "n=123;");
    assert_eq!(a.edits(), [Edit { parent: Span::new(2, 8), child: Span::new(2, 5) }]);
    // One edit is one unit: the middle digit widens to the whole run.
    assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(2, 8));
}

#[test]
fn a_call_that_changes_nothing_records_nothing() {
    let mut b = ViewBuilder::new("abc");
    b.keep(1);
    b.replace(1, "");
    let (out, a) = b.finish();
    assert_eq!(out, "abc");
    assert!(a.edits().is_empty());
}

/// Checks every offset of both texts against a table written out by
/// hand: `to_child[x]` is where parent offset x lands as (a start, an
/// end), and `to_parent` the same the other way.
fn assert_maps(a: &Alignment, to_child: &[(u32, u32)], to_parent: &[(u32, u32)]) {
    for (x, &want) in to_child.iter().enumerate() {
        let got = (a.to_child(x as u32, Side::Start), a.to_child(x as u32, Side::End));
        assert_eq!(got, want, "parent offset {x} to child");
    }
    for (x, &want) in to_parent.iter().enumerate() {
        let got = (a.to_parent(x as u32, Side::Start), a.to_parent(x as u32, Side::End));
        assert_eq!(got, want, "child offset {x} to parent");
    }
}

// The tables below follow these rules. Outside every edit an offset
// shifts by the edits before it. Strictly inside an edit, a start goes
// to the start of the edit's output and an end to its end. At the
// boundary between two touching edits, a start goes with the edit that
// begins there and an end with the edit that ends there. Inserted or
// deleted bytes (an edit with one empty side) belong to neither
// neighbour: a start at them goes after them, an end before them.

#[test]
fn touching_replacements_map_at_every_boundary() {
    // "aXYb" -> "apqrb": X became "pq", then Y became "r".
    let mut b = ViewBuilder::new("aXYb");
    b.keep(1);
    b.replace(2, "pq");
    b.replace(3, "r");
    let (out, a) = b.finish();
    assert_eq!(out, "apqrb");
    assert_eq!(a.edits().len(), 2);
    assert_maps(
        &a,
        &[(0, 0), (1, 1), (3, 3), (4, 4), (5, 5)],
        // Child 2 is inside "pq", so it widens to X alone; child 3 is
        // the boundary between "pq" and "r", which is parent 2.
        &[(0, 0), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4)],
    );
}

#[test]
fn an_insertion_touching_a_replacement_maps_at_every_boundary() {
    // "aXb" -> "a+pqb": "+" inserted before X, then X became "pq".
    let mut b = ViewBuilder::new("aXb");
    b.keep(1);
    b.replace(1, "+");
    b.replace(2, "pq");
    let (out, a) = b.finish();
    assert_eq!(out, "a+pqb");
    assert_eq!(a.edits().len(), 2);
    assert_maps(
        &a,
        // Parent 1: a start goes after "+", to the start of "pq"; an
        // end goes before "+".
        &[(0, 0), (2, 1), (4, 4), (5, 5)],
        // Child 1 and 2 are both parent 1 (before and after "+"); child
        // 3 is inside "pq", which widens to X.
        &[(0, 0), (1, 1), (1, 1), (1, 2), (2, 2), (3, 3)],
    );
}

#[test]
fn a_replacement_touching_a_deletion_maps_at_every_boundary() {
    // "aXYb" -> "apqb": X became "pq", then Y was deleted.
    let mut b = ViewBuilder::new("aXYb");
    b.keep(1);
    b.replace(2, "pq");
    b.replace(3, "");
    let (out, a) = b.finish();
    assert_eq!(out, "apqb");
    assert_eq!(a.edits().len(), 2);
    assert_maps(
        &a,
        // Parent 2 and 3 are both child 3 (before and after Y).
        &[(0, 0), (1, 1), (3, 3), (3, 3), (4, 4)],
        // Child 3: a start goes after Y, an end before it.
        &[(0, 0), (1, 1), (1, 2), (3, 2), (4, 4)],
    );
    // "pq" maps back to X without Y, "b" to "b" without Y.
    assert_eq!(a.span_to_parent(Span::new(1, 3)), Span::new(1, 2));
    assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(3, 4));
}

#[test]
fn a_deletion_touching_an_insertion_maps_at_every_boundary() {
    // "aXb" -> "a+b": X deleted, then "+" inserted after it. Two
    // calls, so two edits, not one replacement of X by "+".
    let mut b = ViewBuilder::new("aXb");
    b.keep(1);
    b.replace(2, "");
    b.replace(2, "+");
    let (out, a) = b.finish();
    assert_eq!(out, "a+b");
    assert_eq!(a.edits().len(), 2);
    assert_maps(
        &a,
        // Parent 2: a start goes after "+", an end before it (and
        // after X, which became nothing).
        &[(0, 0), (1, 1), (2, 1), (3, 3)],
        // Child 1: a start goes after X, an end before it.
        &[(0, 0), (2, 1), (2, 2), (3, 3)],
    );
    // "+" maps to the empty stretch after X, X to the empty stretch
    // before "+".
    assert_eq!(a.span_to_parent(Span::new(1, 2)), Span::new(2, 2));
    assert_eq!(a.span_to_child(Span::new(1, 2)), Span::new(1, 1));
}

#[test]
fn invert_swaps_directions_and_round_trips() {
    let mut b = ViewBuilder::new("a\u{FB01}b--c");
    b.keep(1);
    b.replace(4, "fi");
    b.keep(5);
    b.replace(7, "");
    b.replace(7, "+++");
    let (out, a) = b.finish();
    assert_eq!(out, "afib+++c");

    let inv = a.invert();
    assert_eq!(inv.invert(), a);
    assert!(Alignment::from_edits(inv.edits().to_vec()).is_ok());

    for x in 0..=10 {
        for side in [Side::Start, Side::End] {
            assert_eq!(inv.to_child(x, side), a.to_parent(x, side));
            assert_eq!(inv.to_parent(x, side), a.to_child(x, side));
        }
    }
    for s in 0..=9 {
        for e in s..=9 {
            let span = Span::new(s, e);
            assert_eq!(inv.span_to_child(span), a.span_to_parent(span));
            assert_eq!(inv.span_to_parent(span), a.span_to_child(span));
        }
    }
}

#[test]
fn compose_with_identity_is_identity() {
    let mut b = ViewBuilder::new("hello--world");
    b.keep(5);
    b.replace(7, " ");
    let (_, a) = b.finish();
    assert_eq!(a.compose(&Alignment::identity()), a);
    assert_eq!(Alignment::identity().compose(&a), a);
    assert_eq!(Alignment::identity().compose(&Alignment::identity()), Alignment::identity());
}

#[test]
fn compose_disjoint_and_touching_edits_matches_two_step_mapping() {
    // View A: "aXYb--cd"
    // View B (replace X->"pq" at 1..2, delete "--" at 4..6): "apqYbcd"
    let mut b1 = ViewBuilder::new("aXYb--cd");
    b1.keep(1);
    b1.replace(2, "pq");
    b1.keep(4);
    b1.replace(6, "");
    let (text_b, ab) = b1.finish();
    assert_eq!(text_b, "apqYbcd");

    // View C from B (replace Y->"r" at 3..4 touching "pq", insert "+" at 6..6 before "d"): "apqrbc+d"
    let mut b2 = ViewBuilder::new(&text_b);
    b2.keep(3);
    b2.replace(4, "r");
    b2.keep(6);
    b2.replace(6, "+");
    let (text_c, bc) = b2.finish();
    assert_eq!(text_c, "apqrbc+d");

    let ac = ab.compose(&bc);
    assert!(Alignment::from_edits(ac.edits().to_vec()).is_ok());

    for x in 0..="aXYb--cd".len() as u32 {
        for side in [Side::Start, Side::End] {
            let step = bc.to_child(ab.to_child(x, side), side);
            assert_eq!(ac.to_child(x, side), step, "A->C at {x} ({side:?})");
        }
    }
    for z in 0..=text_c.len() as u32 {
        for side in [Side::Start, Side::End] {
            let step = ab.to_parent(bc.to_parent(z, side), side);
            assert_eq!(ac.to_parent(z, side), step, "C->A at {z} ({side:?})");
        }
    }
    for s in 0..="aXYb--cd".len() as u32 {
        for e in s..="aXYb--cd".len() as u32 {
            let span = Span::new(s, e);
            assert_eq!(ac.span_to_child(span), bc.span_to_child(ab.span_to_child(span)));
        }
    }

    // When a deletion in A->B ("aXb" -> "ab") and an insertion in B->C ("ab" -> "a+b")
    // happen at the same offset in B, compose produces the exact two-edit sequence
    // from `a_deletion_touching_an_insertion_maps_at_every_boundary` and preserves
    // two-step boundary offsets in both directions.
    let mut del_b = ViewBuilder::new("aXb");
    del_b.keep(1);
    del_b.replace(2, "");
    let (_, del_ab) = del_b.finish();
    let mut ins_c = ViewBuilder::new("ab");
    ins_c.keep(1);
    ins_c.replace(1, "+");
    let (_, ins_bc) = ins_c.finish();
    let del_ins = del_ab.compose(&ins_bc);
    assert_eq!(
        del_ins.edits(),
        [
            Edit { parent: Span::new(1, 2), child: Span::new(1, 1) },
            Edit { parent: Span::new(2, 2), child: Span::new(1, 2) }
        ]
    );
    for x in 0..=3 {
        for side in [Side::Start, Side::End] {
            assert_eq!(
                del_ins.to_child(x, side),
                ins_bc.to_child(del_ab.to_child(x, side), side),
                "del_ins to_child at {x} ({side:?})"
            );
            assert_eq!(
                del_ins.to_parent(x, side),
                del_ab.to_parent(ins_bc.to_parent(x, side), side),
                "del_ins to_parent at {x} ({side:?})"
            );
        }
    }
    assert_eq!(del_ins.span_to_child(Span::new(1, 2)), ins_bc.span_to_child(del_ab.span_to_child(Span::new(1, 2))));
    assert_eq!(del_ins.span_to_parent(Span::new(1, 2)), del_ab.span_to_parent(ins_bc.span_to_parent(Span::new(1, 2))));
}

#[test]
fn compose_splits_pure_insertions_and_deletions_across_adjacent_edits() {
    // Reproduction from review: "abcd" -> "xxabcd" (insert "xx" at 0) -> "xbcd"
    // (keep first 'x', delete second 'x' + original 'a' at 1..3 in middle).
    let mut b = ViewBuilder::new("abcd");
    b.replace(0, "xx");
    let (middle, ab) = b.finish();
    assert_eq!(middle, "xxabcd");

    let mut b = ViewBuilder::new(&middle);
    b.keep(1);
    b.replace(3, "");
    let (final_text, bc) = b.finish();
    assert_eq!(final_text, "xbcd");

    let ac = ab.compose(&bc);
    assert_eq!(
        ac.edits(),
        [
            Edit { parent: Span::new(0, 0), child: Span::new(0, 1) },
            Edit { parent: Span::new(0, 1), child: Span::new(1, 1) },
        ]
    );
    let original_a = Span::new(0, 1);
    assert_eq!(bc.span_to_child(ab.span_to_child(original_a)), Span::new(1, 1));
    assert_eq!(ac.span_to_child(original_a), Span::new(1, 1));
    // And the surviving inserted 'x' (0..1 in final) maps back to the empty insertion point 0..0.
    assert_eq!(ab.span_to_parent(bc.span_to_parent(Span::new(0, 1))), Span::new(0, 0));
    assert_eq!(ac.span_to_parent(Span::new(0, 1)), Span::new(0, 0));

    for x in 0..=4 {
        for side in [Side::Start, Side::End] {
            assert_eq!(ac.to_child(x, side), bc.to_child(ab.to_child(x, side), side));
            assert_eq!(ac.to_parent(x, side), ab.to_parent(bc.to_parent(x, side), side));
        }
    }
}

#[test]
fn compose_overlapping_and_cancelling_edits_coalesces_cleanly() {
    // View A: "abcde"
    // View B: insert "--" at 1, replace "cd" (2..4) with "XYZ": "a--bXYZ" + "e" = "a--bXYZe"
    let mut b1 = ViewBuilder::new("abcde");
    b1.keep(1);
    b1.replace(1, "--");
    b1.keep(2);
    b1.replace(4, "XYZ");
    let (text_b, ab) = b1.finish();
    assert_eq!(text_b, "a--bXYZe");

    // View C: delete the inserted "--" (1..3 in B), and replace "XY" (4..6 in B) with "w": "abwZe"
    let mut b2 = ViewBuilder::new(&text_b);
    b2.keep(1);
    b2.replace(3, "");
    b2.keep(4);
    b2.replace(6, "w");
    let (text_c, bc) = b2.finish();
    assert_eq!(text_c, "abwZe");

    let ac = ab.compose(&bc);
    assert!(Alignment::from_edits(ac.edits().to_vec()).is_ok());
    // The "--" insertion followed by deletion cancels completely; "cd" (2..4) -> "XYZ" -> "wZ" (2..4).
    assert_eq!(ac.edits(), [Edit { parent: Span::new(2, 4), child: Span::new(2, 4) }]);
    assert_eq!(ac.span_to_child(Span::new(0, 2)), Span::new(0, 2));
    assert_eq!(ac.span_to_child(Span::new(2, 4)), Span::new(2, 4));
    assert_eq!(ac.span_to_child(Span::new(4, 5)), Span::new(4, 5));
}
