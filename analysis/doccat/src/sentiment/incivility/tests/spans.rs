//! Unit tests for toxic evidence span projection and evaluation (`#890`),
//! fusion gate / severe toxicity rule (`#963`), per-label thresholds (`#888`),
//! and zoo loading (`#888`).

use super::*;
use turbo_analysis::Span;

#[test]
fn char_span_to_utf8_and_token_range_handles_accented_cjk_and_emoji_text() {
    // Mix of accented Latin (2-byte UTF-8), CJK (3-byte UTF-8), and emoji (4-byte UTF-8).
    // "Café 日本語 🔥😡🤬 idiot!"
    // Code points:
    //  0: 'C' (byte 0..1)
    //  1: 'a' (byte 1..2)
    //  2: 'f' (byte 2..3)
    //  3: 'é' (byte 3..5)
    //  4: ' ' (byte 5..6)
    //  5: '日' (byte 6..9)
    //  6: '本' (byte 9..12)
    //  7: '語' (byte 12..15)
    //  8: ' ' (byte 15..16)
    //  9: '🔥' (byte 16..20)
    // 10: '😡' (byte 20..24)
    // 11: '🤬' (byte 24..28)
    // 12: ' ' (byte 28..29)
    // 13..18: "idiot" (byte 29..34)
    // 18: '!' (byte 34..35)
    let text = "Café 日本語 🔥😡🤬 idiot!";
    let doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let view = doc.get::<View>().unwrap().clone();
    let tokens = doc.get::<Tokens>().unwrap();

    // 1. Accented word "Café" at codepoints 0..4 -> bytes 0..5 -> token 0..1.
    let cafe_bytes = char_span_to_utf8_span(&view, 0, 4).unwrap();
    assert_eq!(cafe_bytes, Span::new(0, 5));
    assert_eq!(&text[cafe_bytes.start as usize..cafe_bytes.end as usize], "Café");
    assert_eq!(char_span_to_token_range(&view, tokens, 0, 4).unwrap(), Some(IxRange::new(0, 1)));

    // 2. CJK word "日本語" at codepoints 5..8 -> bytes 6..15 -> token 1..2.
    let cjk_bytes = char_span_to_utf8_span(&view, 5, 8).unwrap();
    assert_eq!(cjk_bytes, Span::new(6, 15));
    assert_eq!(&text[cjk_bytes.start as usize..cjk_bytes.end as usize], "日本語");
    assert_eq!(char_span_to_token_range(&view, tokens, 5, 8).unwrap(), Some(IxRange::new(1, 2)));

    // 3. Emoji run "🔥😡🤬" at codepoints 9..12 -> bytes 16..28 -> token 2..3.
    let emoji_bytes = char_span_to_utf8_span(&view, 9, 12).unwrap();
    assert_eq!(emoji_bytes, Span::new(16, 28));
    assert_eq!(&text[emoji_bytes.start as usize..emoji_bytes.end as usize], "🔥😡🤬");
    assert_eq!(char_span_to_token_range(&view, tokens, 9, 12).unwrap(), Some(IxRange::new(2, 3)));

    // 4. ASCII word "idiot" after multi-byte characters at codepoints 13..18 -> bytes 29..34 -> token 3..4.
    let idiot_bytes = char_span_to_utf8_span(&view, 13, 18).unwrap();
    assert_eq!(idiot_bytes, Span::new(29, 34));
    assert_eq!(&text[idiot_bytes.start as usize..idiot_bytes.end as usize], "idiot");
    assert_eq!(char_span_to_token_range(&view, tokens, 13, 18).unwrap(), Some(IxRange::new(3, 4)));

    // 5. Span covering emoji + "idiot" (codepoints 9..18) -> tokens 2..4.
    assert_eq!(char_span_to_token_range(&view, tokens, 9, 18).unwrap(), Some(IxRange::new(2, 4)));

    // 6. Whitespace-only span (codepoints 4..5) -> None.
    assert_eq!(char_span_to_token_range(&view, tokens, 4, 5).unwrap(), None);

    // 7. Out-of-bounds or reversed codepoint spans are errors.
    assert!(char_span_to_utf8_span(&view, 5, 4).is_err());
    assert!(char_span_to_utf8_span(&view, 0, 100).is_err());
}

#[test]
fn evaluate_toxic_spans_computes_token_and_codepoint_prf1() {
    let text1 = "Café idiot!";
    let d1 = make_doc_with_sentences(text1, &[(0, text1.len() as u32)]);
    let v1 = d1.get::<View>().unwrap();
    let t1 = d1.get::<Tokens>().unwrap();
    // "idiot" is codepoints 5..10, token index 1..2.
    let g1_utf8 = char_span_to_utf8_span(v1, 5, 10).unwrap();
    let g1_tok = char_span_to_token_range(v1, t1, 5, 10).unwrap().unwrap();
    assert_eq!(g1_tok, IxRange::new(1, 2));

    let text2 = "Bonjour le monde.";
    let d2 = make_doc_with_sentences(text2, &[(0, text2.len() as u32)]);
    let v2 = d2.get::<View>().unwrap();
    let t2 = d2.get::<Tokens>().unwrap();

    let gold = vec![
        GoldToxicSpanUnit { utf8_spans: vec![g1_utf8], token_ranges: vec![g1_tok] },
        GoldToxicSpanUnit { utf8_spans: Vec::new(), token_ranges: Vec::new() },
    ];

    let mut pred = Incivility::new();
    pred.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(0)),
        scores: IncivilityScores { insult: 0.9, ..Default::default() },
        evidence: &[(IncivilityLabel::Insult, IxRange::new(1, 2))],
    });
    pred.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(1)),
        scores: IncivilityScores::default(),
        evidence: &[],
    });

    let metrics = evaluate_toxic_spans(&gold, &[v1, v2], &[t1, t2], &pred, &[0.5; 6]).unwrap();
    assert_eq!(metrics.units, 2);
    assert_eq!(metrics.units_with_gold_spans, 1);
    assert_eq!(metrics.units_with_predicted_spans, 1);
    assert!((metrics.token_f1 - 1.0).abs() < 1e-12);
    assert!((metrics.micro_token_f1 - 1.0).abs() < 1e-12);
    assert!((metrics.gold_positive_char_f1 - 1.0).abs() < 1e-12);
    assert!((metrics.predicted_positive_gold_overlap_rate - 1.0).abs() < 1e-12);
}

#[test]
fn fusion_gate_and_severe_toxicity_rule_and_per_label_thresholds() {
    struct FixedHead(IncivilityScores);
    impl IncivilityVectorScorer for FixedHead {
        fn score_incivility(&self, _vec: &[f32]) -> Result<IncivilityScores, String> {
            Ok(self.0)
        }
    }

    // "idiot" has lexicon insult = 0.85; head insult = 0.30.
    // At default fusion_gate = 0.25, noisy-or gives 1 - (1 - 0.85)*(1 - 0.30) = 0.895.
    // At fusion_gate = 0.50 (or 1.0), max gives 0.85.
    let head = Arc::new(FixedHead(IncivilityScores { insult: 0.30, threat: 0.60, ..Default::default() }));
    let tokens = vec!["idiot".to_owned()];
    let default_ann = IncivilityAnnotator::new().with_vector_scorer(head.clone());
    let (s_def, _) = default_ann.score_unit(&tokens, 0, 1, None, Some(&[1.0])).unwrap();
    assert!((s_def.insult - (1.0 - (1.0 - 0.85) * (1.0 - 0.30))).abs() < 1e-12);
    // Threat (0.60 >= 0.5) + Insult (0.895 >= 0.5) triggers SevereToxicity co-occurrence rule by default.
    assert!(s_def.severe_toxicity >= 0.5);

    let max_no_sev = default_ann.clone().with_fusion_gate(0.50).with_severe_toxicity_rule(false);
    let (s_max, _) = max_no_sev.score_unit(&tokens, 0, 1, None, Some(&[1.0])).unwrap();
    assert!((s_max.insult - 0.85).abs() < 1e-12);
    assert_eq!(s_max.severe_toxicity, 0.0);
    assert_ne!(Annotator::config_hash(&default_ann), Annotator::config_hash(&max_no_sev));

    // Per-label thresholds control both evidence fallback and corpus evaluation.
    let custom_ts = [0.90, 0.50, 0.40, 0.50, 0.50, 0.50];
    let ts_ann = max_no_sev.clone().with_thresholds(custom_ts);
    let (_, ev) = ts_ann.score_unit(&tokens, 0, 1, None, Some(&[1.0])).unwrap();
    // Threat (0.60 >= 0.40) gets fallback evidence; Insult (0.82 < 0.90) is below its threshold,
    // though its lexicon match is still recorded by `scan_lexicon`.
    assert!(ev.iter().any(|&(l, _)| l == IncivilityLabel::Threat));
    assert_ne!(Annotator::config_hash(&max_no_sev), Annotator::config_hash(&ts_ann));

    // Invalid fusion gate is rejected.
    assert!(default_ann.clone().with_fusion_gate(1.5).score_unit(&tokens, 0, 1, None, Some(&[1.0])).is_err());
    assert!(default_ann.with_fusion_gate(f64::NAN).score_unit(&tokens, 0, 1, None, Some(&[1.0])).is_err());
}
