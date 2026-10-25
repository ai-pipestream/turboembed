//! `Tokens::in_sentence`, `Tokens::sentence_ranges` and
//! `Subwords::in_token` (#1083), with and without parent columns.

use turbo_analysis::{Ix, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Subwords, Tokens};

fn spans<L>(list: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in list {
        s.push(Span::new(a, b));
    }
    s
}

fn model() -> ModelRef {
    ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() }
}

#[test]
fn in_sentence_agrees_with_and_without_the_column() {
    let sentences = Sentences { spans: spans(&[(0, 9), (10, 21)]) };
    let col = vec![Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(1)];
    let with = Tokens { spans: spans(&[(0, 3), (4, 9), (10, 13), (14, 17), (18, 21)]), sentence: Some(col) };
    let without = Tokens { spans: spans(&[(0, 3), (4, 9), (10, 13), (14, 17), (18, 21)]), sentence: None };
    for s in [Ix::new(0), Ix::new(1)] {
        assert_eq!(with.in_sentence(&sentences, s), without.in_sentence(&sentences, s));
        assert_eq!(with.in_sentence(&sentences, s), with.of(s));
    }
    assert_eq!(without.in_sentence(&sentences, Ix::new(0)).len(), 2);
    assert_eq!(without.in_sentence(&sentences, Ix::new(1)).len(), 3);
}

#[test]
fn sentence_ranges_split_runs_and_cover_the_stream() {
    let col = vec![Ix::new(0), Ix::new(0), Ix::new(2), Ix::new(2), Ix::new(2)];
    let tokens = Tokens { spans: spans(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]), sentence: Some(col) };
    let runs: Vec<_> =
        tokens.sentence_ranges().map(|(s, r)| (s.map(|i| i.get()), r.start.get(), r.end.get())).collect();
    assert_eq!(runs, vec![(Some(0), 0, 2), (Some(2), 2, 5)]);
    let bare = Tokens { spans: spans(&[(0, 1), (1, 2)]), sentence: None };
    let runs: Vec<_> = bare.sentence_ranges().map(|(s, r)| (s, r.start.get(), r.end.get())).collect();
    assert_eq!(runs, vec![(None, 0, 2)]);
    let empty = Tokens { spans: spans(&[]), sentence: None };
    assert_eq!(empty.sentence_ranges().count(), 0);
}

#[test]
fn in_token_agrees_with_and_without_the_column() {
    let tokens = Tokens { spans: spans(&[(0, 5), (6, 11)]), sentence: None };
    let pieces = spans(&[(0, 2), (2, 5), (6, 8), (8, 11)]);
    let with = Subwords {
        spans: spans(&[(0, 2), (2, 5), (6, 8), (8, 11)]),
        ids: vec![0; 4],
        model: model(),
        token: Some(vec![Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1)]),
    };
    let without = Subwords { spans: pieces, ids: vec![0; 4], model: model(), token: None };
    for t in [Ix::new(0), Ix::new(1)] {
        assert_eq!(with.in_token(&tokens, t), without.in_token(&tokens, t));
        assert_eq!(with.in_token(&tokens, t), with.of(t));
        assert_eq!(without.in_token(&tokens, t).len(), 2);
    }
}
