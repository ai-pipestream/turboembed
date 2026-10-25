#[path = "inspect_safety/common.rs"]
mod common;
use common::*;
use tonic::Code;
use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Boundaries, Categories, Chunks, Cut, PosTags, Sentences, Tokens, Vectors};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::proto::result::OffsetMode;
#[tokio::test]
async fn projection_respects_the_vectors_actual_named_sentence_binding() {
    let (srv, mut c) = start().await;
    let mut d = base();
    let gold = Key::<Sentences>::named("gold");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 3));
    s.push(Span::new(3, 7));
    d.add(&gold, Sentences { spans: s }, Provenance::by("review-fixture")).unwrap();
    let mut provenance = Provenance::by("review-fixture");
    provenance.on.push(gold.id());
    let vectors = Key::<Vectors<Sentences>>::DEFAULT;
    d.add(&vectors, Vectors::<Sentences>::new(2, vec![1.0, 0.0, 0.0, 1.0], Vec::new(), model()), provenance).unwrap();
    assert_eq!(d.made_on(vectors.id()), &[gold.id()]);
    let result = c.project_chunks(project(graph(&d))).await;
    drop(c);
    srv.stop().await;
    match result {
        Err(s) => error(s, Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT),
        Ok(out) => {
            let p = out.into_inner().projection.unwrap();
            assert_eq!(p.points[0].end, 3, "projection used default sentence span instead of its gold binding");
        }
    }
}

#[tokio::test]
async fn inspect_includes_named_positional_instances() {
    let (srv, mut c) = start().await;
    let mut d = Document::new("é 😀");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 7));
    d.add(&Key::<Tokens>::named("gold"), Tokens { spans, sentence: None }, Provenance::by("review-fixture")).unwrap();
    let out = c.inspect_document_window(window(graph(&d), OffsetMode::Utf8)).await.unwrap().into_inner();
    drop(c);
    srv.stop().await;
    assert!(out.summary.unwrap().layers.iter().any(|l| l.layer == "opennlp:tokens" && l.instance == "gold"));
    assert!(out.heatmap_lanes.iter().any(|lane| lane.id == "tokens@gold"));
    assert!(
        out.segments.iter().flat_map(|s| &s.annotations).any(|a| a.layer == "opennlp:tokens" && a.instance == "gold"),
        "empty filter promises all positional layers; named tokens vanished"
    );
}

#[tokio::test]
async fn named_instances_keep_their_views_filters_and_pos_bindings() {
    let (srv, mut c) = start().await;
    let mut d = base();
    let gold = Key::<Tokens>::named("gold");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 7));
    d.add(&gold, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut provenance = Provenance::by("test");
    provenance.reads.push(gold.id());
    d.add(
        &Key::DEFAULT,
        PosTags { tagset: TagSet::new("pos", &["GOLD"]), tags: vec![Tag(0); 2], prob: None },
        provenance,
    )
    .unwrap();
    let sentences = Key::<Sentences>::named("gold");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 7));
    d.add(&sentences, Sentences { spans }, Provenance::by("test")).unwrap();
    let mut categories = Categories::<Sentences>::new(TagSet::new("sentiment", &["positive"]));
    categories.push(Tag(0), &[0.9]);
    let mut provenance = Provenance::by("test");
    provenance.on.push(sentences.id());
    d.add(&Key::named("gold-sentiment"), categories, provenance).unwrap();
    let source = d.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(2, "e");
    let (text, alignment) = builder.finish();
    let normalized = View::derived(&source, text, alignment);
    d.add(&Key::named("normalized"), normalized, Provenance::by("test")).unwrap();
    let view = d.get_key(&Key::<View>::named("normalized")).unwrap().id();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 1));
    spans.push(Span::new(2, 6));
    d.add(&Key::<Tokens>::named("normalized"), Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let bytes = graph(&d);
    let out = c.inspect_document_window(window(bytes.clone(), OffsetMode::Utf8)).await.unwrap().into_inner();
    assert!(out.heatmap_lanes.iter().any(|lane| lane.id == "tokens@gold"));
    assert!(out.heatmap_lanes.iter().any(|lane| lane.id == "tokens"));
    assert!(!out.heatmap_lanes.iter().any(|lane| lane.id == "tokens@normalized"));
    assert_eq!(
        out.heatmap_lanes.iter().map(|lane| lane.id.as_str()).collect::<Vec<_>>(),
        ["tokens", "tokens@gold", "entities", "sentiment@gold-sentiment"]
    );
    let sentiment = out.heatmap_lanes.iter().find(|lane| lane.id == "sentiment@gold-sentiment").unwrap();
    assert_eq!(sentiment.buckets.len(), 1);
    assert_eq!((sentiment.buckets[0].start, sentiment.buckets[0].end), (0, 7));
    let annotations: Vec<_> =
        out.segments.iter().flat_map(|s| &s.annotations).filter(|a| a.layer == "opennlp:tokens").collect();
    assert!(annotations.iter().any(|a| a.instance == "gold" && a.label == "GOLD"));
    assert!(annotations.iter().any(|a| a.instance.is_empty() && a.label == "token"));
    assert!(!annotations.iter().any(|a| a.instance == "normalized"));
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let mut req = window(bytes.clone(), mode);
        req.view = u32::from(view.0);
        req.filter_layers = vec!["opennlp:tokens".into()];
        let out = c.inspect_document_window(req).await.unwrap().into_inner();
        assert!(out.heatmap_lanes.iter().any(|lane| lane.id == "tokens@normalized"));
        assert!(!out.heatmap_lanes.iter().any(|lane| lane.id == "tokens" || lane.id == "tokens@gold"));
        assert!(
            out.segments
                .iter()
                .flat_map(|s| &s.annotations)
                .all(|a| a.layer == "opennlp:tokens" && a.instance == "normalized")
        );
        assert!(out.segments.iter().flat_map(|s| &s.annotations).any(|a| a.instance == "normalized"));
    }
    drop(c);
    srv.stop().await;
}

fn two_chunks(first_end: u32) -> Chunks {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, first_end));
    spans.push(Span::new(3, 7));
    Chunks {
        spans,
        level: vec![0; 2],
        parent: vec![None; 2],
        sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
        cut: vec![Cut::DocumentStart, Cut::Sentence],
        cost: vec![0; 2],
        model_tokens: vec![1; 2],
        over_budget: vec![false; 2],
        prompt_overhead: 0,
    }
}

#[tokio::test]
async fn named_chunk_and_boundary_joins_refuse_but_matching_chunks_project() {
    let (srv, mut c) = start().await;
    for named in [true, false] {
        let mut d = base();
        d.add(&Key::DEFAULT, two_chunks(2), Provenance::by("test")).unwrap();
        let gold = Key::<Chunks>::named("gold");
        let mut provenance = Provenance::by("test");
        if named {
            d.add(&gold, two_chunks(3), Provenance::by("test")).unwrap();
            provenance.on.push(gold.id());
        }
        d.add(&Key::DEFAULT, Vectors::<Chunks>::new(2, vec![1.0, 0.0, 0.0, 1.0], Vec::new(), model()), provenance)
            .unwrap();
        let out = c.project_chunks(project(graph(&d))).await;
        if named {
            let status = out.unwrap_err();
            assert!(status.message().contains("gold"));
            error(status, Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
        } else {
            assert_eq!(out.unwrap().into_inner().projection.unwrap().points.len(), 2);
        }
    }
    let mut d = base();
    let gold = Key::<Sentences>::named("gold");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 7));
    d.add(&gold, Sentences { spans }, Provenance::by("test")).unwrap();
    let mut provenance = Provenance::by("test");
    provenance.reads.push(gold.id());
    d.add(&Key::DEFAULT, Boundaries { cost: vec![1, 2], reason: vec![Cut::Sentence; 2] }, provenance).unwrap();
    let mut req = project(graph(&d));
    req.include_drift_curve = true;
    let status = c.project_chunks(req).await.unwrap_err();
    assert!(status.message().contains("gold"));
    error(status, Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    drop(c);
    srv.stop().await;
}

#[tokio::test]
async fn finite_inputs_must_not_emit_infinite_projected_coordinates() {
    let (srv, mut c) = start().await;
    let mut d = base();
    d.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            4,
            vec![f32::MAX, f32::MAX, f32::MAX, f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX],
            Vec::new(),
            model(),
        ),
        Provenance::by("review-fixture"),
    )
    .unwrap();
    let result = c.project_chunks(project(graph(&d))).await;
    drop(c);
    srv.stop().await;
    match result {
        Err(s) => error(s, Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT),
        Ok(out) => {
            let p = out.into_inner().projection.unwrap();
            assert!(
                p.points.iter().all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()),
                "finite source vectors yielded non-finite coordinates: {:?}",
                p.points
            );
        }
    }
}

#[tokio::test]
async fn overlapping_heatmap_density_counts_are_preserved() {
    let (srv, mut c) = start().await;
    for n in [64u32, 128] {
        let mut d = Document::new("x".repeat((2 * n) as usize));
        let mut spans = Spans::new(ViewId::SOURCE);
        for i in 0..n {
            spans.push(Span::new(i, 2 * n - i));
        }
        d.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("review-fixture")).unwrap();
        let mut req = window(graph(&d), OffsetMode::Utf8);
        req.filter_layers = vec!["no-segmentation".into()];
        let out = c.inspect_document_window(req).await.unwrap().into_inner();
        let lane = out.heatmap_lanes.iter().find(|l| l.id == "tokens").unwrap();
        let density_count_sum: u64 = lane.buckets.iter().map(|b| u64::from(b.count)).sum();
        assert_eq!(lane.buckets.len(), (2 * n - 1) as usize);
        assert_eq!(density_count_sum, u64::from(n) * u64::from(n));
        eprintln!(
            "{n} spans: {} endpoints, {} output buckets, {density_count_sum} returned density counts",
            2 * n,
            lane.buckets.len()
        );
    }
    drop(c);
    srv.stop().await;
}
