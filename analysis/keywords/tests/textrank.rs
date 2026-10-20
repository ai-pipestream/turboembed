//! The TextRank extractor and annotator end to end: a hand-verified
//! ranking against an independent reference, multi-word collapsing,
//! the POS filter, and the boundary cases.

use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Error, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Terms, Tokens};
use turbo_keywords::{Keywords, TEXTRANK, TextRankAnnotator};

/// One document from `sentences`: tokens at the single spaces, `pos`
/// tags per token, folded forms lowercased.
fn document(sentences: &[Vec<(&str, &str)>]) -> Document {
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut cursor = 0usize;
    for sent in sentences {
        let start = cursor;
        for (token, _) in sent {
            if !text.is_empty() {
                text.push(' ');
            }
            let s = text.len() as u32;
            text.push_str(token);
            spans.push(Span::new(s, text.len() as u32));
            surface.push(token);
            folded.push(&token.to_lowercase());
            cursor = text.len();
        }
        s_spans.push(Span::new(start as u32, cursor as u32));
    }
    let mut doc = Document::new(text);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, by.clone()).unwrap();
    let labels = ["ADJ", "NOUN", "PROPN", "VERB"];
    let tags: Vec<Tag> = sentences
        .iter()
        .flat_map(|s| s.iter())
        .map(|(_, pos)| Tag(labels.iter().position(|l| l == pos).unwrap() as u16))
        .collect();
    doc.add(&Key::DEFAULT, PosTags { tagset: TagSet::new("universal", &labels), tags, prob: None }, by.clone())
        .unwrap();
    doc.add(&Key::DEFAULT, Terms { surface, folded, stem: None, lemma: None }, by).unwrap();
    doc
}

fn run(doc: &mut Document, annotator: TextRankAnnotator) -> Keywords {
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Terms>::DEFAULT)
        .step(Bound::new(annotator))
        .build()
        .unwrap();
    let mut docs = vec![std::mem::replace(doc, Document::new(""))];
    let results = analyzer.run(&mut docs);
    results[0].as_ref().unwrap();
    *doc = docs.pop().unwrap();
    doc.get::<Keywords>().expect("the keywords layer").check().unwrap();
    let layer = doc.get::<Keywords>().unwrap();
    let mut out = Keywords::new(layer.spans.view());
    let tag = turbo_keywords::algorithm_tagset().tag(TEXTRANK).unwrap();
    for (span, item) in layer.iter() {
        out.push(span, item.phrase, item.score, tag, item.terms);
    }
    out
}

/// "cats chase furry mice. furry mice hunt.": the content tokens are
/// cats, furry, mice (twice, second sentence), with chase and hunt
/// filtered out.
fn cats_doc() -> Document {
    document(&[
        vec![("cats", "NOUN"), ("chase", "VERB"), ("furry", "ADJ"), ("mice", "NOUN")],
        vec![("furry", "ADJ"), ("mice", "NOUN"), ("hunt", "VERB")],
    ])
}

#[test]
fn ranks_the_fixture_hand_verified() {
    let mut doc = cats_doc();
    let kws = run(&mut doc, TextRankAnnotator::new());
    let text = doc.source();
    let items: Vec<_> = kws.iter().map(|(s, i)| (s, i.phrase.to_owned(), i.score, i.terms)).collect();
    // Three items: the verb breaks the token adjacency, so sentence 1
    // collapses to "cats" and "furry mice", never "cats furry mice".
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].0, Span::new(0, 4));
    assert_eq!(items[0].1, "cats");
    assert_eq!(items[0].3, 1);
    assert!((items[0].2 - 0.259743316197).abs() < 1e-9, "the cats centrality");
    assert_eq!(&text[0..4], "cats");
    for item in &items[1..] {
        assert_eq!(item.1, "furry mice");
        assert!((item.2 - 0.740256683803).abs() < 1e-9, "furry + mice centralities");
        assert_eq!(item.3, 2);
    }
    assert_eq!(items[1].0, Span::new(11, 21), "sentence 1's furry mice");
    assert_eq!(items[2].0, Span::new(22, 32), "sentence 2's");
    // The verbs never reach a phrase: no content tag, no vertex.
    assert!(!items.iter().any(|(_, p, _, _)| p.contains("chase") || p.contains("hunt")));
}

#[test]
fn top_n_marks_only_the_best_vertices() {
    // Two vertices marked: cats drops out, furry mice collapses twice.
    let mut doc = cats_doc();
    let kws = run(&mut doc, TextRankAnnotator::new().with_top_n(2));
    let items: Vec<_> = kws.iter().map(|(s, i)| (s, i.phrase.to_owned(), i.score)).collect();
    assert_eq!(items.len(), 2, "one phrase, two occurrences");
    assert!(items.iter().all(|(_, p, _)| p == "furry mice"), "{items:?}");
    assert_eq!(items[0].0, Span::new(11, 21), "sentence 1's furry mice");
    assert_eq!(items[1].0, Span::new(22, 32), "sentence 2's");
    for (_, _, score) in &items {
        assert!((score - 0.740256683803).abs() < 1e-9);
    }
}

#[test]
fn the_window_shapes_the_graph() {
    // Window 1 links only neighbours: different centralities, different
    // salience than the hand-verified window-2 fixture.
    let mut doc = cats_doc();
    let kws = run(&mut doc, TextRankAnnotator::new().with_window(1));
    let fuzzy: f64 = kws.iter().filter(|(_, i)| i.phrase == "furry mice").map(|(_, i)| i.score).next().unwrap();
    assert!((fuzzy - 0.740256683803).abs() > 1e-6, "window 1 is a different graph: {fuzzy}");
}

#[test]
fn adjacency_is_in_the_token_stream() {
    // "cats like cats": the verb between the nouns breaks adjacency, so
    // the phrase is two occurrences of "cats", never "cats cats".
    let mut doc = document(&[vec![("cats", "NOUN"), ("like", "VERB"), ("cats", "NOUN")]]);
    let text = doc.source().to_owned();
    let kws = run(&mut doc, TextRankAnnotator::new());
    let items: Vec<_> = kws.iter().map(|(s, i)| (s, i.phrase.to_owned())).collect();
    assert_eq!(items.len(), 2);
    assert!(items.iter().all(|(_, p)| p == "cats"), "{items:?}");
    assert_eq!(&text[items[0].0.start as usize..items[0].0.end as usize], "cats");
    assert_eq!(&text[items[1].0.start as usize..items[1].0.end as usize], "cats");
}

#[test]
fn empty_and_contentless_documents() {
    let mut doc = document(&[]);
    let kws = run(&mut doc, TextRankAnnotator::new());
    assert_eq!(kws.len(), 0, "the layer is present and empty");

    // No content tags at all.
    let mut doc = document(&[vec![("chase", "VERB"), ("hunt", "VERB")]]);
    let kws = run(&mut doc, TextRankAnnotator::new());
    assert_eq!(kws.len(), 0);

    // A single content token is one single-word keyphrase.
    let mut doc = document(&[vec![("solo", "NOUN")]]);
    let kws = run(&mut doc, TextRankAnnotator::new());
    assert_eq!(kws.len(), 1);
    let (span, item) = kws.iter().next().unwrap();
    assert_eq!((item.phrase, item.score, item.terms, item.algorithm), ("solo", 1.0, 1, TEXTRANK));
    assert_eq!(span, Span::new(0, 4));
}

#[test]
fn invalid_configurations_fail_the_document() {
    use turbo_analysis::View;
    let doc = cats_doc();
    let view = doc.get::<View>().unwrap();
    let s = doc.get::<Sentences>().unwrap();
    let t = doc.get::<Tokens>().unwrap();
    let p = doc.get::<PosTags>().unwrap();
    let terms = doc.get::<Terms>().unwrap();
    for annotator in [
        TextRankAnnotator::new().with_window(0),
        TextRankAnnotator::new().with_top_n(0),
        TextRankAnnotator::new().with_damping(1.0),
        TextRankAnnotator::new().with_damping(0.0),
        TextRankAnnotator::new().with_epsilon(0.0),
    ] {
        assert!(
            matches!(
                annotator.annotate((view, s, t, p, terms)),
                Err(Error::Annotator { annotator: TextRankAnnotator::ID, .. })
            ),
            "{annotator:?}"
        );
    }
}

#[test]
fn misaligned_layers_fail_the_document() {
    use turbo_analysis::View;
    let doc = cats_doc();
    let view = doc.get::<View>().unwrap();
    let s = doc.get::<Sentences>().unwrap();
    let t = doc.get::<Tokens>().unwrap();
    let p = doc.get::<PosTags>().unwrap();
    let short_terms = Terms { surface: Strs::default(), folded: Strs::default(), stem: None, lemma: None };
    let annotator = TextRankAnnotator::new();
    assert!(matches!(
        annotator.annotate((view, s, t, p, &short_terms)),
        Err(Error::Annotator { annotator: TextRankAnnotator::ID, .. })
    ));
    let short_tags = PosTags { tagset: p.tagset.clone(), tags: vec![Tag(0)], prob: None };
    assert!(matches!(
        annotator.annotate((view, s, t, &short_tags, doc.get::<Terms>().unwrap())),
        Err(Error::Annotator { annotator: TextRankAnnotator::ID, .. })
    ));
}

#[test]
fn configurations_hash_apart() {
    let base = TextRankAnnotator::new();
    let variants = [
        base.clone().with_window(3),
        base.clone().with_top_n(5),
        base.clone().with_damping(0.9),
        base.clone().with_epsilon(1e-3),
    ];
    let base_hash = turbo_analysis::Annotator::config_hash(&base);
    for v in &variants {
        assert_ne!(turbo_analysis::Annotator::config_hash(v), base_hash);
    }
    for (i, a) in variants.iter().enumerate() {
        assert!(
            !variants[i + 1..]
                .iter()
                .any(|b| turbo_analysis::Annotator::config_hash(a) == turbo_analysis::Annotator::config_hash(b))
        );
    }
}
