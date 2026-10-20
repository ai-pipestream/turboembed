//! Regression probes from the PR #743 reviews (codex and argon), one
//! test per finding, run against the public extractors.

use turbo_analysis::{Ix, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Terms, Tokens};
use turbo_keywords::{TextRankExtractor, YakeExtractor};
use turbo_stopword::StopwordLanguage;

/// Two content tokens, one sentence.
fn two_tokens() -> (Sentences, Tokens, Terms) {
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 9));
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 4));
    t.push(Span::new(5, 9));
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    for token in ["cats", "cats"] {
        surface.push(token);
        folded.push(token);
    }
    (Sentences { spans: s }, Tokens { spans: t, sentence: None }, terms(&["cats", "cats"]))
}

fn terms(tokens: &[&str]) -> Terms {
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    for token in tokens {
        surface.push(token);
        folded.push(&token.to_lowercase());
    }
    Terms { surface, folded, stem: None, lemma: None }
}

fn pos(tags: &[Tag], labels: &[&str]) -> PosTags {
    PosTags { tagset: TagSet::new("universal", labels), tags: tags.to_vec(), prob: None }
}

/// [P1] Repeated adjacent content terms: "cats cats" has one graph
/// vertex of centrality 1; the phrase must not sum it past 1.0.
#[test]
fn repeated_adjacent_terms_stay_in_range() {
    let (s, t, terms) = two_tokens();
    let p = pos(&[Tag(0), Tag(0)], &["NOUN"]);
    let ranked = TextRankExtractor::new().extract(&s, &t, &p, &terms).unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].phrase, "cats cats");
    assert_eq!(ranked[0].terms, 2, "both occurrences are in the phrase");
    assert!(ranked[0].salience <= 1.0 && ranked[0].salience >= 0.0, "salience {} outside [0, 1]", ranked[0].salience);
    // And the layer the annotator writes passes its own check, so the
    // document takes it.
    use turbo_analysis::{Analyzer, Bound, Document, Key, Layer, Provenance};
    let mut doc = Document::new("cats cats");
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, s, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, t, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, p, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, terms, by).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Terms>::DEFAULT)
        .step(Bound::new(turbo_keywords::TextRankAnnotator::new()))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs)[0].is_ok(), "the document accepts the layer");
    let layer = docs[0].get::<turbo_keywords::Keywords>().unwrap();
    layer.check().unwrap();
    assert!(layer.iter().all(|(_, item)| (0.0..=1.0).contains(&item.score)));
}

/// [P2] A window of usize::MAX saturates instead of overflowing.
#[test]
fn max_window_does_not_overflow() {
    let (s, t, terms) = two_tokens();
    let p = pos(&[Tag(0), Tag(0)], &["NOUN"]);
    let ranked = TextRankExtractor::new().with_window(usize::MAX).extract(&s, &t, &p, &terms);
    assert!(ranked.is_ok(), "textrank with usize::MAX window: {ranked:?}");

    let (s, t, terms) = two_tokens();
    let yake = YakeExtractor::new(turbo_keywords::Stopwords::bundled(StopwordLanguage::English))
        .with_window_size(usize::MAX)
        .extract(&s, &t, &terms);
    assert!(yake.is_ok(), "yake with usize::MAX window: {yake:?}");
}

/// [P2] Malformed layers are refused, not panicked on: a POS tag
/// outside its tag set, and terms columns of different lengths.
#[test]
fn malformed_layers_are_refused_not_panics() {
    let (s, t, terms) = two_tokens();
    let bad = pos(&[Tag(99), Tag(0)], &["NOUN"]);
    let result = TextRankExtractor::new().extract(&s, &t, &bad, &terms);
    assert!(matches!(result, Err(turbo_analysis::Error::Invalid { .. })), "{result:?}");

    // Terms whose folded column is shorter than the surface column.
    let mut surface = Strs::default();
    surface.push("cats");
    surface.push("cats");
    let mut folded = Strs::default();
    folded.push("cats");
    let short = Terms { surface, folded, stem: None, lemma: None };
    let (s, t, _) = two_tokens();
    let result = YakeExtractor::new(turbo_keywords::Stopwords::custom(["qwerty"]).unwrap()).extract(&s, &t, &short);
    assert!(matches!(result, Err(turbo_analysis::Error::Invalid { .. })), "{result:?}");
    let p = pos(&[Tag(0), Tag(0)], &["NOUN"]);
    let result = TextRankExtractor::new().extract(&s, &t, &p, &short);
    assert!(matches!(result, Err(turbo_analysis::Error::Invalid { .. })), "{result:?}");
}

/// [P2] Sentence assignment validates full containment, matching views,
/// and the tokens' own sentence column when present.
#[test]
fn crossing_views_and_lying_sentence_columns_are_refused() {
    // The second token crosses the sentence end.
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 6));
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 4));
    t.push(Span::new(5, 9));
    let terms = terms(&["cats", "cats"]);
    let p = pos(&[Tag(0), Tag(0)], &["NOUN"]);
    let sentences = Sentences { spans: s };
    let tokens = Tokens { spans: t, sentence: None };
    assert!(TextRankExtractor::new().extract(&sentences, &tokens, &p, &terms).is_err());
    assert!(
        YakeExtractor::new(turbo_keywords::Stopwords::custom(["qwerty"]).unwrap())
            .extract(&sentences, &tokens, &terms)
            .is_err()
    );

    // Sentences in another view than the tokens.
    let mut s = Spans::new(ViewId(7));
    s.push(Span::new(0, 9));
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 4));
    t.push(Span::new(5, 9));
    let sentences = Sentences { spans: s };
    let tokens = Tokens { spans: t, sentence: None };
    assert!(TextRankExtractor::new().extract(&sentences, &tokens, &p, &terms).is_err());

    // The tokens' own sentence column claims a different sentence than
    // the spans do.
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 4));
    s.push(Span::new(5, 9));
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 4));
    t.push(Span::new(5, 9));
    let sentences = Sentences { spans: s };
    let tokens = Tokens { spans: t, sentence: Some(vec![Ix::new(1), Ix::new(1)]) };
    assert!(TextRankExtractor::new().extract(&sentences, &tokens, &p, &terms).is_err());
    assert!(
        YakeExtractor::new(turbo_keywords::Stopwords::custom(["qwerty"]).unwrap())
            .extract(&sentences, &tokens, &terms)
            .is_err()
    );
}
