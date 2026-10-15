use turbo_analysis::{
    Alternatives, Document, Error, Ix, Key, Layer, ModeSet, NBest, Provenance, Span, SplitMode, Tag, TagSet, ViewId,
};

struct Tok;
impl Layer for Tok {
    const ID: &'static str = "opennlp:tokens";
    fn len(&self) -> usize {
        5
    }
}

#[test]
fn sudachi_modes_a_b_c_round_trip_from_one_lattice() {
    // Sentence: "東京都選挙管理委員会に行った" (42 UTF-8 bytes, 14 kanji/kana chars, 3 bytes each)
    // Finest positions (Mode A, 8 positions):
    //   0: "東京" (0..6)      [A, B]
    //   1: "都"   (6..9)      [A, B]
    //   2: "選挙" (9..15)     [A, B]
    //   3: "管理" (15..21)    [A, B]
    //   4: "委員" (21..27)    [A]
    //   5: "会"   (27..30)    [A]
    //   6: "に"   (30..33)    [A, B, C, Best]
    //   7: "行った" (33..42)  [A, B, C, Best]
    // Mode B compounds:
    //   4..6 (pos=4, len=2): "委員会" (21..30) [B, Best]
    //   0..2 (pos=0, len=2): "東京都" (0..9)   [C, Best]
    //   2..4 (pos=2, len=2): "選挙管理" (9..21) [Best]
    // Mode C compounds:
    //   2..6 (pos=2, len=4): "選挙管理委員会" (9..30) [C]
    let text = "東京都選挙管理委員会に行った";
    assert_eq!(text.len(), 42);
    let mut doc = Document::new(text);
    let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);

    let a = SplitMode::A;
    let b = SplitMode::B;
    let c = SplitMode::C;
    let best = SplitMode::Best;

    // Position 0
    lat.push(Span::new(0, 6), 0, 1, a | b);
    lat.push(Span::new(0, 9), 0, 2, c | best);
    // Position 1
    lat.push(Span::new(6, 9), 1, 1, a | b);
    // Position 2
    lat.push(Span::new(9, 15), 2, 1, a | b);
    lat.push(Span::new(9, 21), 2, 2, ModeSet::from_mode(best));
    lat.push(Span::new(9, 30), 2, 4, ModeSet::from_mode(c));
    // Position 3
    lat.push(Span::new(15, 21), 3, 1, a | b);
    // Position 4
    lat.push(Span::new(21, 27), 4, 1, ModeSet::from_mode(a));
    lat.push(Span::new(21, 30), 4, 2, b | best);
    // Position 5
    lat.push(Span::new(27, 30), 5, 1, ModeSet::from_mode(a));
    // Position 6
    lat.push(Span::new(30, 33), 6, 1, ModeSet::ALL);
    // Position 7
    lat.push(Span::new(33, 42), 7, 1, ModeSet::ALL);

    assert_eq!(lat.check(), Ok(()));

    let surfaces = |path: &[Ix<Alternatives<Tok>>]| -> Vec<&str> {
        path.iter()
            .map(|&ix| {
                let s = lat.spans.get(ix);
                &text[s.start as usize..s.end as usize]
            })
            .collect()
    };

    let path_a = lat.path_for_mode(SplitMode::A).unwrap();
    assert_eq!(surfaces(&path_a), ["東京", "都", "選挙", "管理", "委員", "会", "に", "行った"]);

    let path_b = lat.path_for_mode(SplitMode::B).unwrap();
    assert_eq!(surfaces(&path_b), ["東京", "都", "選挙", "管理", "委員会", "に", "行った"]);

    let path_c = lat.path_for_mode(SplitMode::C).unwrap();
    assert_eq!(surfaces(&path_c), ["東京都", "選挙管理委員会", "に", "行った"]);

    let path_best = lat.best_path();
    assert_eq!(surfaces(&path_best), ["東京都", "選挙管理", "委員会", "に", "行った"]);

    // Every path in the lattice tiles the exact bytes 0..42 (2 ways across 0..2 * 5 ways across 2..6 = 10 paths).
    let all = lat.all_paths();
    assert_eq!(all.len(), 10);
    let best_bytes = lat.covered_bytes(&path_best);
    assert_eq!(best_bytes, [Span::new(0, 42)]);
    for p in &all {
        assert_eq!(lat.covered_bytes(p), best_bytes);
        let spans = lat.path_spans(p);
        assert_eq!(spans.first().unwrap().start, 0);
        assert_eq!(spans.last().unwrap().end, 42);
        assert!(spans.windows(2).all(|w| w[0].end == w[1].start));
    }

    doc.add(&Key::DEFAULT, lat, Provenance::by("test")).unwrap();
    assert_eq!(doc.get::<Alternatives<Tok>>().unwrap().num_positions, 8);
}

#[test]
fn synonym_graph_ny_and_new_york_spans_two_positions() {
    // "in new york today"
    // pos 0: "in"    (0..2, len 1)
    // pos 1: "new"   (3..6, len 1)
    // pos 1: "ny"    (3..11, len 2, synonym form "ny")
    // pos 2: "york"  (7..11, len 1)
    // pos 3: "today" (12..17, len 1)
    let text = "in new york today";
    let mut doc = Document::new(text);
    let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
    lat.push_full(Span::new(0, 2), 0, 1, ModeSet::BEST | SplitMode::A | SplitMode::C, Some(1.0), None);
    lat.push_full(Span::new(3, 6), 1, 1, ModeSet::BEST | SplitMode::A, Some(1.0), None);
    lat.push_full(Span::new(3, 11), 1, 2, ModeSet::from_mode(SplitMode::C), Some(0.9), Some("ny"));
    lat.push_full(Span::new(7, 11), 2, 1, ModeSet::BEST | SplitMode::A, Some(1.0), None);
    lat.push_full(Span::new(12, 17), 3, 1, ModeSet::BEST | SplitMode::A | SplitMode::C, Some(1.0), None);

    assert_eq!(lat.check(), Ok(()));
    let token_forms = |path: &[Ix<Alternatives<Tok>>]| -> Vec<&str> {
        path.iter()
            .map(|&ix| {
                let alt = lat.get(ix);
                alt.form.unwrap_or(&text[alt.span.start as usize..alt.span.end as usize])
            })
            .collect()
    };
    let expanded = lat.path_for_mode(SplitMode::A).unwrap();
    assert_eq!(token_forms(&expanded), ["in", "new", "york", "today"]);

    let contracted = lat.path_for_mode(SplitMode::C).unwrap();
    assert_eq!(token_forms(&contracted), ["in", "ny", "today"]);
    assert_eq!(lat.get(contracted[1]).position_length, 2);
    assert_eq!(lat.get(contracted[1]).span, Span::new(3, 11));

    // Both paths cover the exact same token bytes.
    let best = lat.best_path();
    assert_eq!(lat.covered_bytes(&expanded), lat.covered_bytes(&best));
    assert_eq!(lat.covered_bytes(&contracted), lat.covered_bytes(&best));

    doc.add(&Key::DEFAULT, lat, Provenance::by("test")).unwrap();
}

#[test]
fn property_every_path_in_random_lattices_covers_the_same_bytes_as_best_path() {
    // Deterministic LCG generating 60 random multi-branch lattices with
    // both contiguous and whitespace-gapped boundaries and multi-position
    // arcs of length 1..=4.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_u32 = || -> u32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };

    for case in 0..60 {
        let num_pos = 3 + (next_u32() % 8) as usize;
        let contiguous = case % 2 == 0;
        let mut pos_start = Vec::with_capacity(num_pos);
        let mut pos_end = Vec::with_capacity(num_pos);
        let mut cursor = 0u32;
        for _ in 0..num_pos {
            let len = 1 + (next_u32() % 5);
            pos_start.push(cursor);
            pos_end.push(cursor + len);
            cursor += len + if contiguous { 0 } else { next_u32() % 2 };
        }

        let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
        for p in 0..num_pos {
            // Always include the unit edge (p, p + 1) so every boundary is anchored.
            let score = f64::from(next_u32() % 100) * 0.01;
            lat.push_full(Span::new(pos_start[p], pos_end[p]), p as u32, 1, ModeSet::A, Some(score), None);
            // Randomly add multi-position edges of length 2..=4.
            for span_len in 2..=4 {
                if p + span_len <= num_pos && next_u32().is_multiple_of(3) {
                    let score = f64::from(next_u32() % 250) * 0.01;
                    let mode = if span_len == 2 { ModeSet::B } else { ModeSet::C };
                    lat.push_full(
                        Span::new(pos_start[p], pos_end[p + span_len - 1]),
                        p as u32,
                        span_len as u32,
                        mode,
                        Some(score),
                        None,
                    );
                }
            }
        }

        assert_eq!(lat.check(), Ok(()), "case {case} failed check");
        let best = lat.best_path();
        assert!(!best.is_empty());
        let want_bytes = lat.covered_bytes(&best);
        let paths = lat.all_paths();
        assert!(!paths.is_empty());
        // The unit spans every path must cover, from the generator's
        // own bookkeeping, not from the lattice.
        let units: Vec<Span> = (0..num_pos).map(|p| Span::new(pos_start[p], pos_end[p])).collect();
        let sum = |path: &[Ix<Alternatives<Tok>>]| -> f64 { path.iter().map(|&ix| lat.get(ix).score.unwrap()).sum() };
        for path in &paths {
            assert_eq!(lat.covered_bytes(path), want_bytes, "case {case} path {path:?}");
            let spans = lat.path_spans(path);
            assert_eq!(spans.first().unwrap().start, 0, "case {case}");
            assert_eq!(spans.last().unwrap().end, pos_end[num_pos - 1], "case {case}");
            assert!(spans.windows(2).all(|w| w[0].end <= w[1].start), "case {case}: {spans:?}");
            // Each arc's bytes are the unit spans it covers, gaps included.
            for &ix in path {
                let a = lat.get(ix);
                let (u, v) = (a.position as usize, (a.position + a.position_length) as usize);
                assert_eq!(a.span, Span::new(units[u].start, units[v - 1].end), "case {case} row {ix:?}");
            }
            if contiguous {
                assert_eq!(lat.covered_bytes(path), [Span::new(0, cursor)]);
            }
            assert!(sum(path) <= sum(&best) + 1e-9, "case {case}: {path:?} outscores the best path");
        }
        // Mode A has only unit arcs, so its path is the unit tiling.
        assert_eq!(lat.path_spans(&lat.path_for_mode(SplitMode::A).unwrap()), units);

        // A damaged copy is refused: a shifted start byte, a shifted end
        // byte, a dropped unit arc, and a position no arc reaches.
        let last = lat.spans.len() - 1;
        let victim = lat.at_position((num_pos - 1) as u32).start.get();
        let damaged = |skip: Option<usize>, edit: &dyn Fn(usize, Span) -> Span| {
            let mut out = Alternatives::<Tok>::new(ViewId::SOURCE);
            for r in 0..lat.spans.len() {
                if Some(r) != skip {
                    let a = lat.get(Ix::new(r as u32));
                    out.push_full(edit(r, a.span), a.position, a.position_length, a.mode, a.score, a.form);
                }
            }
            out
        };
        // The first unit arc running into the next position's bytes,
        // and the last one starting before the position before it ends.
        let shifted = damaged(None, &|r, s| if r == 0 { Span::new(s.start, cursor) } else { s });
        assert!(shifted.check().is_err(), "case {case}: an end byte past the next position passed");
        let shifted = damaged(None, &|r, s| if r == last { Span::new(0, s.end) } else { s });
        assert!(shifted.check().is_err(), "case {case}: a start byte before the previous position passed");
        let mut gapped = lat.clone();
        gapped.num_positions += 1;
        assert!(gapped.check().is_err(), "case {case}: a position with no arc passed");
        let mut dropped = damaged(Some(victim), &|_, s| s);
        dropped.num_positions = lat.num_positions;
        assert!(dropped.check().is_err(), "case {case}: a dropped last unit arc passed");
    }
}

#[test]
fn the_path_of_a_mode_is_the_highest_sum_not_the_greedy_pick() {
    // Three readings of "abc", all in Best: "a" then "bc" scores 1.0
    // locally at position 0, but the whole "abc" scores more in sum.
    let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
    lat.push_full(Span::new(0, 1), 0, 1, ModeSet::BEST, Some(1.0), None);
    lat.push_full(Span::new(0, 3), 0, 2, ModeSet::BEST, Some(0.9), None);
    lat.push_full(Span::new(1, 3), 1, 1, ModeSet::BEST, Some(-10.0), None);
    assert_eq!(lat.check(), Ok(()));
    assert_eq!(lat.path_for_mode(SplitMode::Best).unwrap(), [Ix::new(1)]);
    assert_eq!(lat.best_path(), [Ix::new(1)]);

    // Without scores, the best path takes the fewest steps, and a mode
    // path is decided by granularity: A keeps the units, C the compound.
    let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
    lat.push(Span::new(0, 1), 0, 1, ModeSet::A | SplitMode::C);
    lat.push(Span::new(0, 3), 0, 2, ModeSet::A | SplitMode::C);
    lat.push(Span::new(1, 3), 1, 1, ModeSet::A | SplitMode::C);
    assert_eq!(lat.check(), Ok(()));
    assert_eq!(lat.best_path(), [Ix::new(1)]);
    assert_eq!(lat.path_for_mode(SplitMode::A).unwrap(), [Ix::new(0), Ix::new(2)]);
    assert_eq!(lat.path_for_mode(SplitMode::C).unwrap(), [Ix::new(1)]);

    // An infinite score is refused, and so are more positions than
    // alternatives.
    let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
    lat.push_full(Span::new(0, 1), 0, 1, ModeSet::A, Some(f64::NEG_INFINITY), None);
    assert!(lat.check().unwrap_err().contains("finite"));
    let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
    lat.push(Span::new(0, 1), 0, 1, ModeSet::A);
    lat.num_positions = u32::MAX;
    assert!(lat.check().unwrap_err().contains("positions for 1 alternatives"));
}

#[test]
fn a_refused_candidate_leaves_the_n_best_as_it_was() {
    let tagset = TagSet::new("ud", &["NOUN", "VERB"]);
    let mut nbest = NBest::<Tok>::new(tagset, 2);
    nbest.push(-1.0, &[Tag(0), Tag(1)], Some(&[0.5, 0.5])).unwrap();
    assert!(nbest.push(-2.0, &[Tag(1), Tag(0)], None).is_err());
    assert_eq!(nbest.candidates(), 1);
    assert_eq!(nbest.tags.len(), 2);
    assert_eq!(nbest.check(), Ok(()));
}

#[test]
fn check_refuses_broken_tiling_dead_ends_and_unanchored_positions() {
    // 1. Mismatched start byte at position 0.
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 3), 0, 1, ModeSet::A);
    bad.push(Span::new(1, 3), 0, 1, ModeSet::B);
    assert!(bad.check().unwrap_err().contains("starts at byte 1"));

    // 2. Mismatched end byte at position 1.
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 3), 0, 1, ModeSet::A);
    bad.push(Span::new(0, 4), 0, 1, ModeSet::B);
    assert!(bad.check().unwrap_err().contains("ends at byte 4"));

    // 3. Overlapping byte boundary at position 1 (end_at[1] = 4 > start_at[1] = 3).
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 4), 0, 1, ModeSet::A);
    bad.push(Span::new(3, 6), 1, 1, ModeSet::A);
    assert!(bad.check().unwrap_err().contains("overlaps in bytes"));

    // 4. Unanchored internal boundary: two len=2 edges over num_positions = 2.
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 6), 0, 2, ModeSet::C);
    bad.push(Span::new(0, 6), 0, 2, ModeSet::B);
    assert!(bad.check().unwrap_err().contains("position boundary 1 has no path"));
    // With one edge, there are more positions than alternatives.
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 6), 0, 2, ModeSet::C);
    assert!(bad.check().unwrap_err().contains("2 positions for 1 alternatives"));

    // 5. Dead-end edge that cannot reach num_positions.
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 6), 0, 2, ModeSet::C);
    bad.push(Span::new(0, 3), 0, 1, ModeSet::A);
    assert!(bad.check().unwrap_err().contains("not on any complete path"));

    // 6. Zero position_length.
    let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
    bad.push(Span::new(0, 3), 0, 0, ModeSet::A);
    bad.num_positions = 1;
    assert!(bad.check().unwrap_err().contains("position length of zero"));
}

#[test]
fn nbest_sequences_align_to_anchor_and_validate_scores_and_probs() {
    let mut doc = Document::new("dogs bark loud today .");
    doc.add(&Key::DEFAULT, Tok, Provenance::by("test")).unwrap();

    let tagset = TagSet::new("ud", &["NOUN", "VERB", "ADV", "ADJ", "PUNCT"]);
    let mut nbest = NBest::<Tok>::new(tagset.clone(), 5);
    nbest.push(-1.2, &[Tag(0), Tag(1), Tag(2), Tag(0), Tag(4)], Some(&[0.9, 0.8, 0.7, 0.9, 0.99])).unwrap();
    nbest.push(-1.8, &[Tag(0), Tag(1), Tag(3), Tag(0), Tag(4)], Some(&[0.9, 0.8, 0.3, 0.9, 0.99])).unwrap();

    assert_eq!(nbest.check(), Ok(()));
    assert_eq!(nbest.candidates(), 2);
    assert_eq!(nbest.tags_of(0), &[Tag(0), Tag(1), Tag(2), Tag(0), Tag(4)]);
    assert_eq!(nbest.at(1, Ix::new(2)), (Tag(3), Some(0.3)));

    doc.add(&Key::DEFAULT, nbest, Provenance::by("test")).unwrap();
    assert_eq!(doc.get::<NBest<Tok>>().unwrap().candidates(), 2);

    // Misaligned NBest (4 items instead of 5) is refused by Document::add.
    let mut wrong_len = NBest::<Tok>::new(tagset.clone(), 4);
    wrong_len.push(-1.0, &[Tag(0), Tag(1), Tag(2), Tag(4)], None).unwrap();
    let err = doc.add(&Key::named("wrong"), wrong_len, Provenance::by("test")).unwrap_err();
    assert!(matches!(err, Error::Misaligned { .. }), "{err:?}");

    // Out-of-order candidate scores are refused by Layer::check.
    let mut bad_order = NBest::<Tok>::new(tagset, 5);
    bad_order.push(-2.0, &[Tag(0); 5], None).unwrap();
    bad_order.push(-1.0, &[Tag(0); 5], None).unwrap();
    assert!(bad_order.check().unwrap_err().contains("non-increasing score order"));
}
