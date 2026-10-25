//! Named sources and mixed target views survive the shared-library adapter.
#[path = "inspect_safety/common.rs"]
#[allow(dead_code)]
mod common;
use common::*;
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{
    Block, Categories, Chunks, Cut, DocumentHeatmap, EmotionItem, EmotionLayer, EmotionScores, EmotionTarget, Entities,
    Incivility, IncivilityItem, IncivilityScores, IncivilityTarget, Pii, PiiConfidence, PiiKind, PiiMention, Sentences,
    Structure, Tokens,
};
use turbo_kserve::proto::result::OffsetMode;
fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut spans = Spans::new(view);
    for &(s, e) in pairs {
        spans.push(Span::new(s, e));
    }
    spans
}
fn chunks(view: ViewId, end: u32) -> Chunks {
    Chunks {
        spans: spans(view, &[(0, end)]),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::DocumentStart],
        cost: vec![0],
        model_tokens: vec![17],
        over_budget: vec![false],
        prompt_overhead: 0,
    }
}
fn mixed() -> Document {
    let mut d = Document::new("é 😀");
    let source = d.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.replace(2, "e");
    let (text, align) = b.finish();
    d.add(&Key::named("plain"), View::derived(&source, text, align), Provenance::by("test")).unwrap();
    d.add(
        &Key::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 2), (3, 7)]), sentence: None },
        Provenance::by("test"),
    )
    .unwrap();
    d.add(
        &Key::named("gold"),
        Tokens { spans: spans(ViewId(1), &[(0, 1), (2, 6)]), sentence: None },
        Provenance::by("test"),
    )
    .unwrap();
    d.add(&Key::DEFAULT, Sentences { spans: spans(ViewId::SOURCE, &[(0, 7)]) }, Provenance::by("test")).unwrap();
    let gold = Key::<Sentences>::named("gold");
    d.add(&gold, Sentences { spans: spans(ViewId(1), &[(0, 6)]) }, Provenance::by("test")).unwrap();
    d.add(&Key::DEFAULT, chunks(ViewId::SOURCE, 7), Provenance::by("test")).unwrap();
    let chunk = Key::<Chunks>::named("gold");
    d.add(&chunk, chunks(ViewId(1), 6), Provenance::by("test")).unwrap();
    let mut cats = Categories::<Sentences>::new(TagSet::new("sentiment", &["positive", "negative"]));
    cats.push(Tag(0), &[0.9, 0.1]);
    d.add(&Key::named("gold"), cats, Provenance { on: vec![gold.id()], ..Provenance::by("test") }).unwrap();
    let mut cats = Categories::<Chunks>::new(TagSet::new("quality", &["useful", "other"]));
    cats.push(Tag(0), &[0.6, 0.4]);
    d.add(&Key::named("gold"), cats, Provenance { on: vec![chunk.id()], ..Provenance::by("test") }).unwrap();
    d.add(
        &Key::DEFAULT,
        Structure {
            spans: spans(ViewId::SOURCE, &[(0, 7)]),
            kind: vec![Block::Paragraph],
            parent: vec![None],
            depth: vec![0],
        },
        Provenance::by("test"),
    )
    .unwrap();
    d.add(
        &Key::named("gold"),
        Entities {
            spans: spans(ViewId(1), &[(2, 6)]),
            tokens: vec![IxRange::new(1, 2)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("entity", &["EMOJI"]),
            score: vec![0.2],
        },
        Provenance { on: vec![Key::<Tokens>::named("gold").id()], ..Provenance::by("test") },
    )
    .unwrap();
    let mut pii = Pii::new(ViewId(1));
    pii.push(PiiMention {
        span: Span::new(0, 1),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "x",
    });
    d.add(&Key::named("gold"), pii, Provenance::by("test")).unwrap();
    let mut inc = Incivility::on(Key::DEFAULT, Key::DEFAULT, chunk, Key::DEFAULT);
    for (target, score) in [
        (IncivilityTarget::Sentence(Ix::new(0)), 0.2),
        (IncivilityTarget::Chunk(Ix::new(0)), 0.8),
        (IncivilityTarget::Block(Ix::new(0)), 0.4),
    ] {
        inc.push(IncivilityItem {
            target,
            evidence: &[],
            scores: IncivilityScores {
                insult: score,
                profanity: 0.0,
                threat: 0.0,
                identity_attack: 0.0,
                intolerance: 0.0,
                severe_toxicity: 0.0,
            },
        });
    }
    d.add(
        &Key::named("mixed"),
        inc,
        Provenance {
            reads: vec![
                Key::<Tokens>::DEFAULT.id(),
                Key::<Sentences>::DEFAULT.id(),
                chunk.id(),
                Key::<Structure>::DEFAULT.id(),
            ],
            ..Provenance::by("test")
        },
    )
    .unwrap();
    let mut emotion = EmotionLayer::on(Key::DEFAULT, Key::DEFAULT, chunk);
    for (target, value) in [(EmotionTarget::Sentence(Ix::new(0)), 0.3), (EmotionTarget::Chunk(Ix::new(0)), 0.7)] {
        let mut scores = EmotionScores::default();
        scores.0[0] = value;
        emotion.push(EmotionItem { target, scores, evidence: &[] });
    }
    d.add(
        &Key::named("mixed"),
        emotion,
        Provenance {
            reads: vec![Key::<Tokens>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id(), chunk.id()],
            ..Provenance::by("test")
        },
    )
    .unwrap();
    d
}
#[tokio::test]
async fn named_sources_use_the_requested_view_and_keep_original_target_indices() {
    let d = mixed();
    let bytes = graph(&d);
    let (srv, mut client) = start().await;
    assert!(
        DocumentHeatmap::build(&d, ViewId(1), None).is_err(),
        "existing strict API still refuses default tokens on the other view"
    );
    assert!(DocumentHeatmap::build_instances(&Document::new(""), ViewId(1), None).is_err());
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            for bins in [0, 1, 9] {
                let mut req = window(bytes.clone(), mode);
                req.view = view;
                req.heatmap_buckets = bins;
                let out = client.inspect_document_window(req).await.unwrap().into_inner();
                let ids: Vec<_> = out.heatmap_lanes.iter().map(|l| l.id.as_str()).collect();
                let expected = if view == 0 {
                    vec!["tokens", "incivility@mixed", "emotion@mixed"]
                } else {
                    vec![
                        "tokens@gold",
                        "entities@gold",
                        "pii@gold",
                        "sentiment@gold",
                        "incivility@mixed",
                        "emotion@mixed",
                        "chunks@gold",
                    ]
                };
                assert_eq!(ids, expected);
                for (id, score, primary, count) in if view == 0 {
                    [("incivility@mixed", 0.4, 2, 2), ("emotion@mixed", 0.3, 0, 1)]
                } else {
                    [("incivility@mixed", 0.8, 1, 1), ("emotion@mixed", 0.7, 1, 1)]
                } {
                    let lane = out.heatmap_lanes.iter().find(|l| l.id == id).unwrap();
                    for b in lane.buckets.iter().filter(|b| b.count > 0) {
                        assert!((b.score - score).abs() < 1e-6);
                        assert_eq!(b.primary_item, Some(primary));
                        assert_eq!(b.count, count);
                    }
                }
            }
        }
    }
    drop(client);
    srv.stop().await;
}
