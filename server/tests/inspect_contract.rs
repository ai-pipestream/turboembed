#[path = "inspect_safety/common.rs"]
mod common;
use common::*;
use tonic::Code;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis::{Tag, TagSet};
use turbo_analysis_layers::{Sentences, Tokens, Vectors};
use turbo_kserve::api::{TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT};
use turbo_kserve::proto::result::{FormatDocumentRequest, OffsetMode};
#[tokio::test]
async fn real_rpc_goldens_offsets_views_and_refusals() {
    let (srv, mut c) = start().await;
    srv.load().await.unwrap();
    let mut d = base();
    d.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(2, vec![1.0, 0.0, 0.0, 1.0], Vec::new(), model()),
        Provenance::by("review-fixture"),
    )
    .unwrap();
    let source = d.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.replace(2, "e");
    let (text, alignment) = b.finish();
    d.add(&Key::named("folded"), View::derived(&source, text, alignment), Provenance::by("review-fixture")).unwrap();
    let parent = d.get_key(&Key::<View>::named("folded")).unwrap().clone();
    let mut b = ViewBuilder::new(parent.text());
    b.replace(0, "!");
    let (text, alignment) = b.finish();
    d.add(&Key::named("prefixed"), View::derived(&parent, text, alignment), Provenance::by("review-fixture")).unwrap();
    let bytes = graph(&d);
    for (mode, a, b) in [(OffsetMode::Utf8, 3, 7), (OffsetMode::Utf16, 2, 4), (OffsetMode::CodePoint, 2, 3)] {
        let out = c.inspect_document_window(window(bytes.clone(), mode)).await.unwrap().into_inner();
        assert!(out.segments.iter().any(|s| s.text == "😀" && s.start == a && s.end == b));
        let mut req = window(bytes.clone(), mode);
        req.view = 2;
        req.xray_target_view = Some(0);
        let out = c.inspect_document_window(req).await.unwrap().into_inner();
        let x = out.xray.unwrap();
        assert_eq!(x.runs.iter().map(|r| r.parent_text.as_str()).collect::<String>(), "!e 😀");
        assert_eq!(x.runs.iter().map(|r| r.child_text.as_str()).collect::<String>(), "é 😀");
        for r in &x.runs {
            if r.kind == 0 {
                assert_eq!(r.parent_text, r.child_text);
            }
        }
        for id in ["brat", "conllu", "csv", "jsonld", "markdown", "tsv", "warc"] {
            let out = c
                .format_document(FormatDocumentRequest {
                    graph: bytes.clone(),
                    format_id: id.into(),
                    doc_id: "unicode".into(),
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            assert!(!out.content.is_empty(), "{id}");
        }
    }
    let a = c.project_chunks(project(bytes.clone())).await.unwrap().into_inner();
    let b = c.project_chunks(project(bytes.clone())).await.unwrap().into_inner();
    assert_eq!(a, b);
    let p = a.projection.unwrap();
    assert_eq!(p.points.len(), 2);
    assert!((p.explained_variance[0] - 1.0).abs() < 1e-5);
    assert!((p.points[0].x.abs() - (0.5f32).sqrt()).abs() < 1e-5);
    assert!((p.points[0].x + p.points[1].x).abs() < 1e-5);
    for bad in [Vec::new(), b"not a graph".to_vec()] {
        error(
            c.inspect_document_window(window(bad.clone(), OffsetMode::Utf8)).await.unwrap_err(),
            Code::InvalidArgument,
            TURBO_E_INVALID_ARGUMENT,
        );
        error(
            c.project_chunks(project(bad.clone())).await.unwrap_err(),
            Code::InvalidArgument,
            TURBO_E_INVALID_ARGUMENT,
        );
        error(
            c.format_document(FormatDocumentRequest {
                graph: bad,
                format_id: "jsonld".into(),
                doc_id: "bad".into(),
                offsets: 0,
            })
            .await
            .unwrap_err(),
            Code::InvalidArgument,
            TURBO_E_INVALID_ARGUMENT,
        );
    }
    for (v, p, t) in [(u32::MAX, 0, None), (0, u32::MAX, None), (0, 0, Some(u32::MAX))] {
        let mut r = window(bytes.clone(), OffsetMode::Utf8);
        r.view = v;
        r.page = p;
        r.xray_target_view = t;
        error(c.inspect_document_window(r).await.unwrap_err(), Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    }
    let mut r = window(bytes.clone(), OffsetMode::Utf8);
    r.offsets = 99;
    error(c.inspect_document_window(r).await.unwrap_err(), Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    let mut r = window(bytes.clone(), OffsetMode::Utf8);
    r.page_size = 65_537;
    error(c.inspect_document_window(r).await.unwrap_err(), Code::OutOfRange, TURBO_E_CAPACITY);
    error(
        c.format_document(FormatDocumentRequest {
            graph: bytes.clone(),
            format_id: "unknown".into(),
            doc_id: "bad".into(),
            offsets: 0,
        })
        .await
        .unwrap_err(),
        Code::InvalidArgument,
        TURBO_E_INVALID_ARGUMENT,
    );
    let mut invalid_utf8 = bytes.clone();
    let at = invalid_utf8.windows("é 😀".len()).position(|w| w == "é 😀".as_bytes()).unwrap();
    invalid_utf8[at] = 0xff;
    error(
        c.inspect_document_window(window(invalid_utf8, OffsetMode::Utf8)).await.unwrap_err(),
        Code::InvalidArgument,
        TURBO_E_INVALID_ARGUMENT,
    );
    let mut malformed = turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap();
    malformed.source.text = "😀 é".into();
    malformed.source.bytes = "😀 é".as_bytes().to_vec();
    malformed.views[0].text = Some("😀 é".into());
    error(
        c.inspect_document_window(window(malformed.to_bytes().unwrap(), OffsetMode::Utf8)).await.unwrap_err(),
        Code::InvalidArgument,
        TURBO_E_INVALID_ARGUMENT,
    );
    let mut bound = base();
    let gold = Key::<Tokens>::named("gold");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 7));
    bound.add(&gold, Tokens { spans, sentence: None }, Provenance::by("review-fixture")).unwrap();
    let mut prov = Provenance::by("review-fixture");
    prov.reads.push(gold.id());
    bound
        .add(
            &Key::DEFAULT,
            turbo_analysis_layers::PosTags { tagset: TagSet::new("pos", &["NOUN"]), tags: vec![Tag(0)], prob: None },
            prov,
        )
        .unwrap();
    for id in ["conllu", "tsv", "csv", "jsonld", "warc"] {
        error(
            c.format_document(FormatDocumentRequest {
                graph: graph(&bound),
                format_id: id.into(),
                doc_id: "bound".into(),
                offsets: 0,
            })
            .await
            .unwrap_err(),
            Code::InvalidArgument,
            TURBO_E_INVALID_ARGUMENT,
        );
    }
    for (n, dim) in [(513usize, 1usize), (2, 2049)] {
        let mut capacity = Document::new("x".repeat(2 * n));
        let mut spans = Spans::new(ViewId::SOURCE);
        for i in 0..n {
            spans.push(Span::new((2 * i) as u32, (2 * i + 1) as u32));
        }
        capacity.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("review-fixture")).unwrap();
        capacity
            .add(
                &Key::DEFAULT,
                Vectors::<Sentences>::new(dim as u32, vec![1.0; n * dim], Vec::new(), model()),
                Provenance::by("review-fixture"),
            )
            .unwrap();
        error(c.project_chunks(project(graph(&capacity))).await.unwrap_err(), Code::OutOfRange, TURBO_E_CAPACITY);
    }
    drop(c);
    srv.stop().await;
}
