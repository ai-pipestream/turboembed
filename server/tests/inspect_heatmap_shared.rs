//! Real-graph regressions for the shared heatmap adapter (#1220).
#[path = "inspect_safety/common.rs"]
#[allow(dead_code)]
mod common;
use common::*;
use turbo_analysis::{
    Analyzer, Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    Block, Categories, Chunks, Cut, EmotionItem, EmotionLayer, EmotionScores, EmotionTarget, Incivility,
    IncivilityItem, IncivilityScores, IncivilityTarget, Pii, PiiConfidence, PiiKind, PiiMention, Sentences,
    SparseTerms, Structure, Tokens,
};
use turbo_chunker::terms::{TermCounter, TermIndexer};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::inspect::{inspect_document_window, project_chunks};
use turbo_kserve::proto::result::OffsetMode;

fn document() -> Document {
    let mut d = Document::new("é 😀");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 7));
    d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    d
}
fn add_chunks(d: &mut Document) {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 7));
    d.add(
        &Key::DEFAULT,
        Chunks {
            spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![17],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        Provenance::by("test"),
    )
    .unwrap();
}
fn score(d: &Document, kind: &str, expected: f32) {
    let bytes = graph(d);
    let mut failures = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        for n in [0, 1, 3] {
            let mut req = window(bytes.clone(), mode);
            req.heatmap_buckets = n;
            let out = inspect_document_window(&req).unwrap();
            let lane = out.heatmap_lanes.iter().find(|lane| lane.kind == kind);
            let Some(lane) = lane else {
                failures.push(format!("{mode:?}, bins={n}: omitted"));
                continue;
            };
            if !lane.buckets.iter().any(|b| b.count > 0) {
                failures.push(format!("{mode:?}, bins={n}: empty"));
            }
            for b in lane.buckets.iter().filter(|b| b.count > 0) {
                if (b.score - expected).abs() >= 1e-6 {
                    failures.push(format!("{mode:?}, bins={n}: expected {expected}, actual {}", b.score));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{kind}: {}", failures.join("; "));
}
#[test]
fn heatmap_binary_sentiment_is_positive_minus_negative() {
    let mut d = document();
    let mut cats = Categories::<Sentences>::new(TagSet::new("sentiment", &["positive", "negative"]));
    cats.push(Tag(0), &[0.7, 0.3]);
    d.add(&Key::DEFAULT, cats, Provenance::by("test")).unwrap();
    score(&d, "SentimentPolarity", 0.4);
}
#[test]
fn heatmap_generic_topic_is_not_sentiment() {
    let mut d = document();
    let mut cats = Categories::<Sentences>::new(TagSet::new("topic", &["negotiation", "sport"]));
    cats.push(Tag(0), &[0.9, 0.1]);
    d.add(&Key::DEFAULT, cats, Provenance::by("test")).unwrap();
    let bytes = graph(&d);
    let mut failures = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        for n in [0, 1, 3] {
            let mut req = window(bytes.clone(), mode);
            req.heatmap_buckets = n;
            let out = inspect_document_window(&req).unwrap();
            if out
                .heatmap_lanes
                .iter()
                .filter(|l| l.kind == "SentimentPolarity")
                .flat_map(|l| &l.buckets)
                .any(|b| b.count > 0 || b.score != 0.0)
            {
                failures.push(format!("{mode:?}, bins={n}: {:?}", out.heatmap_lanes));
            }
        }
    }
    assert!(failures.is_empty(), "generic negotiation topic emitted as sentiment: {}", failures.join("; "));
}
#[test]
fn heatmap_chunk_score_is_category_score() {
    let mut d = document();
    add_chunks(&mut d);
    let mut cats = Categories::<Chunks>::new(TagSet::new("relevance", &["useful", "other"]));
    cats.push(Tag(0), &[0.9, 0.1]);
    d.add(&Key::DEFAULT, cats, Provenance::by("test")).unwrap();
    score(&d, "ChunkScore", 0.9);
}
#[test]
fn heatmap_emotion_is_emitted() {
    let mut d = document();
    let mut emotion = EmotionLayer::new();
    let mut scores = EmotionScores::default();
    scores.0[0] = 0.75;
    emotion.push(EmotionItem { target: EmotionTarget::Sentence(Ix::new(0)), scores, evidence: &[] });
    d.add(&Key::DEFAULT, emotion, Provenance::by("test")).unwrap();
    score(&d, "EmotionIntensity", 0.75);
}
#[test]
fn heatmap_block_incivility_is_emitted() {
    let mut d = document();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 7));
    d.add(
        &Key::DEFAULT,
        Structure { spans, kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] },
        Provenance::by("test"),
    )
    .unwrap();
    let mut inc = Incivility::new();
    inc.push(IncivilityItem {
        target: IncivilityTarget::Block(Ix::new(0)),
        evidence: &[],
        scores: IncivilityScores {
            insult: 0.75,
            profanity: 0.0,
            threat: 0.0,
            identity_attack: 0.0,
            intolerance: 0.0,
            severe_toxicity: 0.0,
        },
    });
    d.add(&Key::DEFAULT, inc, Provenance::by("test")).unwrap();
    score(&d, "IncivilityScore", 0.75);
}
#[test]
fn heatmap_entity_density_counts_items() {
    score(&base(), "EntityDensity", 1.0);
}
#[test]
fn heatmap_pii_density_counts_items() {
    let mut d = document();
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span::new(0, 7),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "x",
    });
    d.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    score(&d, "PiiDensity", 1.0);
}

#[test]
fn sparse_same_view_spans_and_occurrences_refuse_bad_character_boundaries() {
    for span in [Span::new(1, 2), Span::new(0, 1), Span::new(0, 8), Span::new(8, 8)] {
        for occurrence in [false, true] {
            let mut d = document();
            let mut terms = SparseTerms::<View>::new(model());
            terms.push(7, 1.5, Some(Span::new(3, 7)));
            terms.unit();
            terms.push_occurrences(&[Span::new(3, 7)]);
            d.add(&Key::DEFAULT, terms, Provenance::by("test")).unwrap();
            let mut g = turbo_analysis::export::export(&d, turbo_kserve::results::document::codecs()).unwrap();
            let body =
                g.layers.iter_mut().find(|l| l.layer.starts_with("turbo:sparse-terms")).unwrap().body.as_mut().unwrap();
            // Existing codec: row, term, weight arrays; optional span; occurrence arrays.
            let at = if occurrence { 53 } else { 33 };
            assert_eq!(&body[at..at + 4], &3u32.to_le_bytes());
            assert_eq!(&body[at + 4..at + 8], &7u32.to_le_bytes());
            body[at..at + 4].copy_from_slice(&span.start.to_le_bytes());
            body[at + 4..at + 8].copy_from_slice(&span.end.to_le_bytes());
            let bytes = g.to_bytes().unwrap();
            for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
                let mut req = project(bytes.clone());
                req.offsets = mode as i32;
                let out = std::panic::catch_unwind(|| project_chunks(&req)).expect("no panic");
                assert_eq!(out.expect_err("bad span accepted").code, TURBO_E_INVALID_ARGUMENT);
            }
        }
    }
}
#[test]
fn sparse_matching_view_projects_unicode_occurrences() {
    let mut d = document();
    let mut terms = SparseTerms::<View>::new(model());
    terms.push(7, 1.5, Some(Span::new(3, 7)));
    terms.unit();
    d.add(&Key::DEFAULT, terms, Provenance::by("test")).unwrap();
    let bytes = graph(&d);
    for (mode, expected) in [(OffsetMode::Utf8, (3, 7)), (OffsetMode::Utf16, (2, 4)), (OffsetMode::CodePoint, (2, 3))] {
        let mut req = project(bytes.clone());
        req.offsets = mode as i32;
        let stack = project_chunks(&req).unwrap().document_term_stack.unwrap();
        assert_eq!((stack.segments[0].occ_start[0], stack.segments[0].occ_end[0]), expected);
    }
}
#[test]
fn sparse_other_view_is_refused_even_when_its_spans_fit_source() {
    let mut d = document();
    let source = d.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.replace(2, "e with acute");
    b.keep(source.text().len());
    let (text, alignment) = b.finish();
    let plain = Key::<View>::named("plain");
    d.add(&plain, View::derived(&source, text, alignment), Provenance::by("test")).unwrap();
    let mut terms = SparseTerms::<View>::new(model());
    terms.push(7, 1.5, Some(Span::new(3, 7)));
    terms.unit();
    let mut p = Provenance::by("test");
    p.on.push(plain.id());
    d.add(&Key::DEFAULT, terms, p).unwrap();
    let bytes = graph(&d);
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let mut req = project(bytes.clone());
        req.offsets = mode as i32;
        assert_eq!(project_chunks(&req).expect_err("wrong view accepted").code, TURBO_E_INVALID_ARGUMENT);
    }
}
#[test]
fn indexed_token_fallback_and_chunk_cuts_match_containment_oracle() {
    for n in [4, 17, 64] {
        let text: String = (0..n).map(|i| format!("t{i} ")).collect();
        let mut d = Document::new(text.clone());
        let spans: Vec<_> = text
            .split_whitespace()
            .scan(0u32, |at, t| {
                let s = Span::new(*at, *at + t.len() as u32);
                *at = s.end + 1;
                Some(s)
            })
            .collect();
        let mut tokens = Spans::new(ViewId::SOURCE);
        let mut sentences = Spans::new(ViewId::SOURCE);
        for &s in &spans {
            tokens.push(s);
            sentences.push(s);
        }
        d.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test")).unwrap();
        d.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
        let selected: Vec<_> = spans.chunks(3).map(|ss| Span::new(ss[0].start, ss.last().unwrap().end)).collect();
        let m = selected.len();
        let mut chunks = Spans::new(ViewId::SOURCE);
        for &s in &selected {
            chunks.push(s);
        }
        d.add(
            &Key::DEFAULT,
            Chunks {
                spans: chunks,
                level: vec![0; m],
                parent: vec![None; m],
                sentences: (0..m).map(|i| IxRange::new((i * 3) as u32, ((i * 3 + 3).min(n)) as u32)).collect(),
                cut: vec![Cut::Sentence; m],
                cost: vec![0; m],
                model_tokens: vec![3; m],
                over_budget: vec![false; m],
                prompt_overhead: 0,
            },
            Provenance::by("test"),
        )
        .unwrap();
        let mut req = project(graph(&d));
        req.include_drift_curve = true;
        let out = project_chunks(&req).unwrap();
        assert_eq!(out.chunk_term_stacks.len(), m);
        for (stack, &chunk) in out.chunk_term_stacks.iter().zip(&selected) {
            let mut expected: Vec<_> = spans
                .iter()
                .filter(|s| s.start >= chunk.start && s.end <= chunk.end)
                .map(|s| text[s.range()].to_lowercase())
                .collect();
            expected.sort();
            assert_eq!(stack.segments.iter().map(|s| s.term.clone()).collect::<Vec<_>>(), expected);
            assert_eq!(stack.total_frequency, expected.len() as u64);
        }
        for (gap, sentence) in out.drift_curve.unwrap().gaps.iter().zip(&spans) {
            assert_eq!(gap.is_chunk_cut, selected.iter().any(|c| c.end == sentence.end));
        }
        let pipeline = Analyzer::builder()
            .given(Key::<View>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .add(TermIndexer)
            .add(TermCounter::<Chunks>::new())
            .add(TermCounter::<View>::new())
            .build()
            .unwrap();
        assert_eq!(pipeline.run(std::slice::from_mut(&mut d))[0], Ok(()));
        req.graph = graph(&d);
        let stats = project_chunks(&req).unwrap();
        for (a, b) in out.chunk_term_stacks.iter().zip(&stats.chunk_term_stacks) {
            let values = |s: &turbo_kserve::proto::result::TermVectorStackProto| {
                s.segments
                    .iter()
                    .map(|t| (t.term.clone(), t.frequency, t.occ_start.clone(), t.occ_end.clone()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(values(a), values(b));
            assert_eq!(a.total_frequency, b.total_frequency);
        }
        assert_eq!(stats.document_term_stack.unwrap().total_frequency, n as u64);
    }
}
