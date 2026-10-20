use turbo::qa::{Answer, DecodeOptions, WindowLogits, decode};

fn window(offsets: &[(usize, usize)], start: &[f32], end: &[f32], null: f32) -> WindowLogits {
    WindowLogits {
        start: std::iter::once(null).chain(start.iter().copied()).collect(),
        end: std::iter::once(null).chain(end.iter().copied()).collect(),
        context: std::iter::once(None).chain(offsets.iter().map(|&(a, b)| Some(a..b))).collect(),
        max_context: std::iter::once(false).chain(std::iter::repeat_n(true, offsets.len())).collect(),
        cls: 0,
    }
}
fn spans(result: Answer) -> (Vec<turbo::qa::SpanAnswer>, f64) {
    match result {
        Answer::Spans { answers, null_probability, .. } => (answers, null_probability),
        other => panic!("expected spans: {other:?}"),
    }
}

#[test]
fn second_window_answer_keeps_exact_unicode_bytes() {
    let text = "Paris 東京 café";
    let w1 = window(&[(0, 5)], &[1.0], &[1.0], 10.0);
    let w2 = window(&[(6, 12), (13, 18)], &[5.0, 0.0], &[5.0, 0.0], 0.0);
    let (answers, null) = spans(decode(text, &[w1, w2], DecodeOptions::default()).unwrap());
    assert_eq!(answers[0].bytes, 6..12);
    assert_eq!(&text[answers[0].bytes.clone()], "東京");
    assert!((answers.iter().map(|a| a.probability).sum::<f64>() + null - 1.0).abs() < 1e-12);
}

#[test]
fn null_wins_at_threshold_equality_and_empty_context() {
    let w = window(&[(0, 1)], &[1.0], &[1.0], 1.0);
    assert_eq!(
        decode("x", std::slice::from_ref(&w), DecodeOptions::default()).unwrap(),
        Answer::None { null_score: 2.0 }
    );
    let options = DecodeOptions { null_threshold: -0.1, ..Default::default() };
    assert!(matches!(decode("x", &[w], options).unwrap(), Answer::Spans { .. }));
    assert_eq!(decode("", &[window(&[], &[], &[], 3.0)], options).unwrap(), Answer::None { null_score: 6.0 });
}

#[test]
fn ownership_filters_starts_without_losing_answer_endpoints() {
    let mut w = window(&[(0, 1), (2, 3), (4, 5)], &[10.0, 30.0, 0.0], &[0.0, 10.0, 0.0], 0.0);
    w.max_context[2] = false;
    let (a, _) =
        spans(decode("a b c", std::slice::from_ref(&w), DecodeOptions { top_k: 1, ..Default::default() }).unwrap());
    assert_eq!(a[0].bytes, 0..3);
    let (a, _) =
        spans(decode("a b c", &[w], DecodeOptions { max_answer_tokens: 1, top_k: 1, ..Default::default() }).unwrap());
    assert_eq!(a[0].bytes, 0..1);
}

#[test]
fn duplicate_spans_use_best_score_and_extreme_logits_remain_finite() {
    let first = window(&[(0, 1)], &[f32::MAX], &[f32::MAX], -f32::MAX);
    let second = window(&[(0, 1)], &[1.0], &[1.0], 0.0);
    let (a, null) = spans(decode("x", &[first, second], DecodeOptions::default()).unwrap());
    assert_eq!(a.len(), 1);
    assert!(a[0].score.is_finite());
    assert_eq!(a[0].probability, 1.0);
    assert_eq!(null, 0.0);
}

#[test]
fn malformed_columns_offsets_logits_and_options_return_errors() {
    let w = window(&[(0, 2)], &[1.0], &[1.0], 0.0);
    assert!(decode("é", std::slice::from_ref(&w), DecodeOptions::default()).is_ok());
    for bad in [0..1, 1..2, 0..3, 2..2] {
        let mut b = w.clone();
        b.context[1] = Some(bad);
        assert!(decode("é", &[b], DecodeOptions::default()).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut b = w.clone();
        b.start[0] = value;
        assert!(decode("é", &[b], DecodeOptions::default()).is_err());
    }
    let mut b = w.clone();
    b.end.clear();
    assert!(decode("é", &[b], DecodeOptions::default()).is_err());
    let mut b = w.clone();
    b.cls = usize::MAX;
    assert!(decode("é", &[b], DecodeOptions::default()).is_err());
    let mut b = w.clone();
    b.max_context[0] = true;
    assert!(decode("é", &[b], DecodeOptions::default()).is_err());
    for options in [
        DecodeOptions { top_k: 0, ..Default::default() },
        DecodeOptions { max_answer_tokens: 0, ..Default::default() },
        DecodeOptions { null_threshold: f64::NAN, ..Default::default() },
    ] {
        assert!(decode("é", std::slice::from_ref(&w), options).is_err());
    }
    assert!(decode("é", &[], DecodeOptions::default()).is_err());
    assert!(
        decode("é", &[w], DecodeOptions { top_k: usize::MAX, max_answer_tokens: usize::MAX, ..Default::default() })
            .is_ok()
    );
}
