use turbo_analysis::*;
use turbo_analysis_layers::Entities;

#[test]
fn eof_window_page_is_inside_its_advertised_page_count() {
    for (text, len) in [("abcd", 2), ("a😀b", 3), ("", 2)] {
        let doc = Document::new(text);
        let window = doc.window(ViewId::SOURCE, text.len() as u32, len).unwrap();
        assert!(window.span.is_empty());
        assert_eq!(window.page, window.pages - 1, "EOF of {text:?}");
    }
}

#[test]
fn every_utf8_viewport_matches_fragment_coverage_and_clips_cleanly() {
    let doc = Document::new("α😀b文c");
    let mut a = Spans::<Entities>::new(ViewId::SOURCE);
    a.try_push_fragments(&[Span::new(0, 2), Span::new(7, 10)]).unwrap();
    a.push(Span::new(2, 7));
    a.push(Span::new(6, 6));
    a.push(Span::new(11, 11));
    let mut b = Spans::<Entities>::new(ViewId::SOURCE);
    b.push(Span::new(6, 11));
    b.push(Span::new(0, 6));
    b.push(Span::new(2, 2));
    let key = Key::<Entities>::DEFAULT.id();
    let prepared = doc.prepare_segments(ViewId::SOURCE, &[(key, a.text_spans()), (key, b.text_spans())]).unwrap();
    for from in 0..=doc.source().len() as u32 {
        for len in 1..=13 {
            let w = doc.window(ViewId::SOURCE, from, len).unwrap();
            assert!(doc.source().is_char_boundary(w.span.start as usize));
            assert!(doc.source().is_char_boundary(w.span.end as usize));
            assert_eq!(w.text(&doc), &doc.source()[w.span.range()]);
            let shape = doc.combined_segments(&w, &[(key, a.text_spans()), (key, b.text_spans())]).unwrap();
            assert_eq!(prepared.combined_segments(&w).unwrap(), shape);
            let mut at = w.span.start;
            for segment in &shape.segments {
                if segment.span.is_empty() {
                    continue;
                }
                assert_eq!(segment.span.start, at);
                at = segment.span.end;
                assert!(at <= w.span.end);
                for byte in segment.span.start..segment.span.end {
                    let want: Vec<_> = [&a, &b]
                        .into_iter()
                        .enumerate()
                        .flat_map(|(layer, spans)| {
                            spans
                                .iter()
                                .filter(move |(i, _)| spans.fragments(*i).any(|f| f.start <= byte && byte < f.end))
                                .map(move |(i, _)| ShapeMember { layer, item: i.get() as u32 })
                        })
                        .collect();
                    assert_eq!(shape.members_of(segment), want);
                }
            }
            assert_eq!(at, w.span.end);
        }
    }
}

#[test]
fn prepared_ordered_fragments_and_empty_runs_match_raw_at_every_window_boundary() {
    let doc = Document::new("0123456789");
    let mut spans = Spans::<Entities>::new(ViewId::SOURCE);
    spans.try_push_fragments(&[Span::new(0, 2), Span::new(5, 7)]).unwrap();
    spans.push(Span::new(7, 7));
    spans.push(Span::new(7, 7));
    spans.push(Span::new(7, 10));
    spans.push(Span::new(10, 10));
    assert!(spans.is_ordered());
    let layers = [(Key::<Entities>::DEFAULT.id(), spans.text_spans())];
    let prepared = doc.prepare_segments(ViewId::SOURCE, &layers).unwrap();
    for start in 0..=10 {
        for end in start..=10 {
            let window =
                DocumentWindow { view: ViewId::SOURCE, span: Span::new(start, end), page: 0, pages: 1, clamped: false };
            assert_eq!(
                prepared.combined_segments(&window).unwrap(),
                doc.combined_segments(&window, &layers).unwrap(),
                "window {start}..{end}"
            );
        }
    }
}

#[test]
fn preparing_checks_outside_window_data_and_prepared_calls_check_windows() {
    let doc = Document::new("a😀b");
    let key = Key::<Entities>::DEFAULT.id();
    let mut invalid = Spans::<Entities>::new(ViewId::SOURCE);
    invalid.push(Span::new(0, 1));
    invalid.push(Span::new(2, 3));
    assert!(matches!(doc.prepare_segments(ViewId::SOURCE, &[(key, invalid.text_spans())]), Err(Error::Bounds { .. })));
    let starts = [0];
    let ends = [];
    let malformed = TextSpans {
        view: ViewId::SOURCE,
        start: &starts,
        end: &ends,
        frag_start: &[],
        frag_end: &[],
        frag_item_end: &[],
    };
    assert!(doc.prepare_segments(ViewId::SOURCE, &[(key, malformed)]).is_err());
    assert!(doc.prepare_segments(ViewId(99), &[]).is_err());
    let mut valid = Spans::<Entities>::new(ViewId::SOURCE);
    valid.push(Span::new(0, 1));
    let prepared = doc.prepare_segments(ViewId::SOURCE, &[(key, valid.text_spans())]).unwrap();
    let mut window = doc.window(ViewId::SOURCE, 0, 1).unwrap();
    window.view = ViewId(99);
    assert!(prepared.combined_segments(&window).is_err());
    window.view = ViewId::SOURCE;
    window.span = Span::new(0, 3);
    assert!(prepared.combined_segments(&window).is_err());
}
