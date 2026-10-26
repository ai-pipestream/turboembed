use super::*;
use turbo_analysis::{Analyzer, Document, Key, Provenance};

// GoEmotions CC-BY-4.0 test comments eecwqtt and ed5f85d, pinned in the zoo
// at google-research/google-research 7de3624754e2b0f2f505ae496db3ba194bb40be9.
const COMMENTS: [&str; 2] = [
    "I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!",
    "It's wonderful because it's awful. At not with.",
];

fn rows(texts: &[&str]) -> Vec<Row> {
    texts.iter().map(|s| Row { text: (*s).to_owned(), labels: Vec::new(), targets: EmotionScores::default() }).collect()
}

#[test]
fn official_comment_offsets_keep_utf8_and_project_to_all_units() {
    let rows = rows(&COMMENTS);
    let p = prepare(&rows).unwrap();
    assert_eq!(p.tokens.spans.columns().0[1], 1);
    assert_eq!(p.tokens.spans.columns().1[1], 4);
    let mut token = 0;
    let mut base = 0;
    for (i, row) in rows.iter().enumerate() {
        let doc = Document::new(row.text.as_str());
        let mut expected = Vec::new();
        turbo_tokenize::simple_spans(&row.text, turbo_tokenize::TokenizerOptions::default(), &mut expected);
        assert_eq!(p.sentences.spans.columns().0[i], base);
        assert_eq!(p.sentences.spans.columns().1[i], base + row.text.len() as u32);
        for original in expected {
            let range = IxRange::new(token, token + 1);
            let actual = p.source_span(i, range).unwrap();
            assert_eq!(actual, Span::new(original.start, original.end));
            let utf16 = doc.view_span_to_utf16(ViewId::SOURCE, actual).unwrap();
            let points = doc.view_span_to_code_point(ViewId::SOURCE, actual).unwrap();
            assert_eq!(doc.view_span_from_utf16(ViewId::SOURCE, utf16).unwrap(), actual);
            assert_eq!(doc.view_span_from_code_point(ViewId::SOURCE, points).unwrap(), actual);
            assert_eq!(p.tokens.spans.columns().0[token as usize], base + actual.start);
            assert_eq!(p.tokens.spans.columns().1[token as usize], base + actual.end);
            token += 1;
        }
        base += row.text.len() as u32;
    }
}

#[test]
fn empty_comments_add_no_synthetic_tokens_and_keep_following_offsets() {
    let rows = rows(&["", " \t\n", "😀", COMMENTS[0]]);
    let p = prepare(&rows).unwrap();
    assert!(p.keys[0].is_empty() && p.keys[1].is_empty());
    assert_eq!(p.tokens.spans.columns().0[0], 3);
    assert_eq!(p.tokens.spans.columns().1[0], 7);
    let scored = score_split(&EmotionAnnotator::new(), &p, &[], 0).unwrap();
    assert!(scored.evidence_of(Ix::new(0)).next().is_none());
    assert!(scored.evidence_of(Ix::new(1)).next().is_none());
    assert!(p.source_span(0, IxRange::new(0, 1)).is_err());
    assert!(p.source_span(3, IxRange::new(0, 1)).is_err());
    let doc = Document::new("😀");
    let emoji = p.source_span(2, IxRange::new(0, 1)).unwrap();
    let utf16 = doc.view_span_to_utf16(ViewId::SOURCE, emoji).unwrap();
    assert_eq!(doc.view_span_from_utf16(ViewId::SOURCE, utf16).unwrap(), emoji);
}

#[test]
fn corpus_evidence_matches_production_annotator_on_original_comments() {
    let rows = rows(&COMMENTS);
    let p = prepare(&rows).unwrap();
    let a = EmotionAnnotator::new();
    let corpus = score_split(&a, &p, &[], 0).unwrap();
    let mut token_offset = 0;
    for (i, row) in rows.iter().enumerate() {
        let mut doc = Document::new(row.text.as_str());
        let single = prepare(std::slice::from_ref(row)).unwrap();
        doc.add(&Key::DEFAULT, single.sentences, Provenance::by("original-comment")).unwrap();
        doc.add(&Key::DEFAULT, single.tokens, Provenance::by("original-comment")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .add(a.clone())
            .build()
            .unwrap();
        for result in analyzer.run(std::slice::from_mut(&mut doc)) {
            result.unwrap();
        }
        let actual = doc.get::<EmotionLayer>().unwrap();
        assert_eq!(corpus.scores_of(Ix::new(i as u32)), actual.scores_of(Ix::new(0)));
        let expected: Vec<_> = actual
            .evidence_of(Ix::new(0))
            .map(|(d, r)| (d, IxRange::new(r.start.get() as u32 + token_offset, r.end.get() as u32 + token_offset)))
            .collect();
        assert_eq!(corpus.evidence_of(Ix::new(i as u32)).collect::<Vec<_>>(), expected);
        for (_, range) in expected {
            assert!(!row.text[p.source_span(i, range).unwrap().range()].is_empty());
        }
        token_offset += p.keys[i].len() as u32;
    }
}
