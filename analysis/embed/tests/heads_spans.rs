//! `VectorHeadAnnotator` and the span heads (#321): sentence, chunk,
//! opinion and aspect spans, each span pooled on its own, so several
//! spans in one sentence get distinct embeddings and polarities.

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Categories, Chunks, Cut, Incivility, IncivilityLabel, Opinion, Opinions, Polarity,
    PolarityScores, Pragmatics, Sentences, Subwords, Tokens, Vectors,
};
use turbo_tokenize::SimpleTokenizer;

use heads_support::*;

#[test]
fn vector_head_and_span_heads_write_expected_layers() {
    let (_sub, vec_ann) = open_cpu(64);

    // Mixed-polarity sentence containing two distinct opinion spans and two aspects.
    // Tokens from SimpleTokenizer on "I love the camera, though I hate the battery.":
    // 0: "I" (0..1)
    // 1: "love" (2..6)
    // 2: "the" (7..10)
    // 3: "camera" (11..17)
    // 4: "," (17..18)
    // 5: "though" (19..25)
    // 6: "I" (26..27)
    // 7: "hate" (28..32)
    // 8: "the" (33..36)
    // 9: "battery" (37..44)
    // 10: "." (44..45)
    let text = "I love the camera, though I hate the battery.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let mut c = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::Sentence],
        cost: vec![0],
        model_tokens: vec![0],
        over_budget: vec![false],
        prompt_overhead: 0,
    };
    c.spans.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, c, Provenance::by("test")).unwrap();

    let prep = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .add(SentenceChunkVectors(Arc::clone(&vec_ann)))
        .build()
        .unwrap();
    let mut docs = [doc];
    prep.run(&mut docs).remove(0).unwrap();
    let mut doc = docs.into_iter().next().unwrap();

    let (opinions, aspects) = {
        let tokens = doc.get::<Tokens>().unwrap();
        assert_eq!(tokens.len(), 11);
        let sentences = doc.get::<Sentences>().unwrap();

        let opinions = Opinions::from_items(
            tokens,
            [
                Opinion {
                    tokens: IxRange::new(0, 4), // "I love the camera"
                    polarity: 0.0,
                    intensity: 1.0,
                    negated: false,
                },
                Opinion {
                    tokens: IxRange::new(6, 10), // "I hate the battery"
                    polarity: 0.0,
                    intensity: 1.0,
                    negated: false,
                },
            ],
        )
        .unwrap();
        let aspect_cats = TagSet::new("aspect-cat", &["hardware"]);
        let op0 = [Ix::<Opinions>::new(0)];
        let op1 = [Ix::<Opinions>::new(1)];
        let aspects = Aspects::from_items(
            tokens,
            Some(sentences),
            Some(&opinions),
            Arc::clone(&aspect_cats),
            [
                Aspect {
                    target: AspectTarget::Term {
                        tokens: IxRange::new(3, 4), // "camera"
                        sentence: Some(Ix::new(0)),
                    },
                    category: Some(Tag(0)),
                    polarity: PolarityScores::new(0.0, 0.0, 1.0),
                    opinions: &op0,
                },
                Aspect {
                    target: AspectTarget::Term {
                        tokens: IxRange::new(9, 10), // "battery"
                        sentence: Some(Ix::new(0)),
                    },
                    category: Some(Tag(0)),
                    polarity: PolarityScores::new(0.0, 0.0, 1.0),
                    opinions: &op1,
                },
            ],
        )
        .unwrap();
        (opinions, aspects)
    };
    doc.add(&Key::DEFAULT, opinions.clone(), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, aspects.clone(), Provenance::by("test")).unwrap();

    // Embed the two opinion spans and the two aspect spans via `embed_spans`
    // to verify that two spans inside the same sentence receive distinct
    // span-pooled embeddings.
    let (opinion_vecs, aspect_vecs) = {
        let tokens = doc.get::<Tokens>().unwrap();
        let sentences = doc.get::<Sentences>().unwrap();
        let subwords = doc.get::<Subwords>().unwrap();
        let op_spans = [opinions.span(Ix::new(0), tokens), opinions.span(Ix::new(1), tokens)];
        let opinion_vecs: Vectors<Opinions> = vec_ann
            .embed_spans("test:opinions", subwords, Some(&sentences.spans), tokens.spans.view(), &op_spans)
            .unwrap();
        assert_eq!(opinion_vecs.len(), 2);
        assert_ne!(opinion_vecs.row(Ix::new(0)), opinion_vecs.row(Ix::new(1)));

        let asp_spans = [
            aspects.span(Ix::new(0), tokens, Some(sentences)).unwrap(),
            aspects.span(Ix::new(1), tokens, Some(sentences)).unwrap(),
        ];
        let aspect_vecs: Vectors<Aspects> = vec_ann
            .embed_spans("test:aspects", subwords, Some(&sentences.spans), tokens.spans.view(), &asp_spans)
            .unwrap();
        assert_eq!(aspect_vecs.len(), 2);
        assert_ne!(aspect_vecs.row(Ix::new(0)), aspect_vecs.row(Ix::new(1)));
        (opinion_vecs, aspect_vecs)
    };

    // Fit an opinion polarity head on the positive and negative span vectors
    // and an aspect polarity head on the camera and battery span vectors.
    let pol_tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
    let op_head = ClassifierHead::fit_single_label(
        vec_ann.model().clone(),
        32,
        Arc::clone(&pol_tags),
        &[(opinion_vecs.row(Ix::new(0)), Tag(0)), (opinion_vecs.row(Ix::new(1)), Tag(1))],
        1e-4,
        100,
        2.0,
    )
    .unwrap();
    let asp_head = ClassifierHead::fit_single_label(
        vec_ann.model().clone(),
        32,
        Arc::clone(&pol_tags),
        &[(aspect_vecs.row(Ix::new(0)), Tag(0)), (aspect_vecs.row(Ix::new(1)), Tag(1))],
        1e-4,
        100,
        2.0,
    )
    .unwrap();

    // Build multi-label Incivility and Pragmatics heads.
    let inciv_names = IncivilityLabel::ALL.map(|l| l.name());
    let inciv_tags = TagSet::new("incivility", &inciv_names);
    let inciv_head = ClassifierHead::linear(
        vec_ann.model().clone(),
        32,
        inciv_tags,
        HeadMode::MultiLabel,
        vec![0.0; 6 * 32],
        vec![-1.0, -2.0, -3.0, -2.5, -2.0, -3.5],
    )
    .unwrap();
    let prag_tags = TagSet::new("pragmatics", &["sarcasm", "irony"]);
    let prag_head = ClassifierHead::linear(
        vec_ann.model().clone(),
        32,
        prag_tags,
        HeadMode::MultiLabel,
        vec![0.0; 2 * 32],
        vec![0.8, -0.8],
    )
    .unwrap();

    let sent_head = SentenceSpanHead::new(Arc::clone(&vec_ann), op_head.clone());
    let chunk_head = ChunkSpanHead::new(Arc::clone(&vec_ann), op_head.clone());
    let opinion_span_head = OpinionSpanHead::new(Arc::clone(&vec_ann), op_head);
    let aspect_span_head = AspectSpanHead::new(Arc::clone(&vec_ann), asp_head);
    let inciv_ann = IncivilityHeadAnnotator(Arc::new(inciv_head));
    let prag_ann = PragmaticsHeadAnnotator(Arc::new(prag_head));

    let heads_pipeline = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .given(Key::<Vectors<Chunks>>::DEFAULT)
        .given(Key::<Opinions>::DEFAULT)
        .given(Key::<Aspects>::DEFAULT)
        .add(sent_head)
        .add(chunk_head)
        .add(opinion_span_head)
        .add(aspect_span_head.clone())
        .add(inciv_ann)
        .add(prag_ann)
        .build()
        .unwrap();

    let mut docs = [doc];
    heads_pipeline.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    // Verify opinion spans received opposite polarity predictions inside the same sentence.
    let op_cats = doc.get::<Categories<Opinions>>().unwrap();
    assert_eq!(op_cats.len(), 2);
    assert_eq!(op_cats.best_category(0), "positive");
    assert_eq!(op_cats.best_category(1), "negative");

    // Verify aspect spans received opposite polarity predictions and populate `Aspects`.
    let asp_cats = doc.get::<Categories<Aspects>>().unwrap();
    assert_eq!(asp_cats.len(), 2);
    assert_eq!(asp_cats.best_category(0), "positive");
    assert_eq!(asp_cats.best_category(1), "negative");

    let scored_aspects = aspect_span_head
        .score_aspects_polarity(
            doc.get::<Subwords>().unwrap(),
            doc.get::<Tokens>().unwrap(),
            Some(doc.get::<Sentences>().unwrap()),
            Some(&opinions),
            &aspects,
        )
        .unwrap();
    assert_eq!(scored_aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);
    assert_eq!(scored_aspects.get(Ix::new(1)).polarity.label(), Polarity::Negative);

    // A head of the same width for another encoder is refused on this path
    // too, as categorize_vectors refuses it.
    let other_head = ClassifierHead::fit_single_label(
        ModelRef { id: "different-encoder".into(), revision: "0".into(), sha256: "0".into() },
        32,
        Arc::clone(&pol_tags),
        &[(aspect_vecs.row(Ix::new(0)), Tag(0)), (aspect_vecs.row(Ix::new(1)), Tag(1))],
        1e-4,
        10,
        2.0,
    )
    .unwrap();
    let err = AspectSpanHead::new(Arc::clone(&vec_ann), other_head)
        .score_aspects_polarity(
            doc.get::<Subwords>().unwrap(),
            doc.get::<Tokens>().unwrap(),
            Some(doc.get::<Sentences>().unwrap()),
            Some(&opinions),
            &aspects,
        )
        .unwrap_err();
    assert!(err.to_string().contains("different-encoder"), "{err}");

    // Verify sentence and chunk categories and Incivility/Pragmatics layers were populated.
    assert_eq!(doc.get::<Categories<Sentences>>().unwrap().len(), 1);
    assert_eq!(doc.get::<Categories<Chunks>>().unwrap().len(), 1);
    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    assert!(inciv.scores(Ix::new(0)).insult > 0.2);
    let prag = doc.get::<Pragmatics>().unwrap();
    assert_eq!(prag.len(), 1);
    assert!(prag.get(Ix::new(0)).score > 0.6);
}
