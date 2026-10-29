//! Fixed public output for the model-free default profile over real gRPC.
//! Expected spans are reviewed source offsets, independent of the annotators.

use tokio_stream::StreamExt;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Span, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{AnalyzePipelineRequest, LayerWireStatus, OffsetMode};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

const TEXT: &str = "Zoë met Alice.\nCafé 😀 is open! 東京 works.";
const STEPS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-abbrev:schwartz-hearst",
    "turbo-stem:terms",
    "turbo-stem:stemmer",
    "turbo-stopword:bundled",
    "turbo-keywords:yake",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];
// UTF-8 byte span, UTF-16 span, code-point span, source text.
const WORDS: &[(u32, u32, u32, u32, u32, u32, &str)] = &[
    (0, 4, 0, 3, 0, 3, "Zoë"),
    (5, 8, 4, 7, 4, 7, "met"),
    (9, 14, 8, 13, 8, 13, "Alice"),
    (16, 21, 15, 19, 15, 19, "Café"),
    (22, 26, 20, 22, 20, 21, "😀"),
    (27, 29, 23, 25, 22, 24, "is"),
    (30, 34, 26, 30, 25, 29, "open"),
    (36, 39, 32, 33, 31, 32, "東"),
    (39, 42, 33, 34, 32, 33, "京"),
    (43, 48, 35, 40, 34, 39, "works"),
];

#[tokio::test]
async fn default_profile_preserves_unicode_text_tokens_offsets_and_steps_without_models() {
    let server = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        Vec::new(),
        16 * 1024 * 1024,
        Serve { admin: true, ..Serve::default() },
    )
    .await
    .unwrap();
    // No load call and no configured model: only the actual rule pipeline runs.
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    let mut reference = None;
    for profile in ["", "default", "standard"] {
        let req = AnalyzePipelineRequest {
            text: TEXT.into(),
            profile: profile.into(),
            offsets: OffsetMode::Utf8 as i32,
            ..Default::default()
        };
        let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
        let mut completed = Vec::new();
        let mut terminal = None;
        let mut sequence = 0;
        while let Some(event) = stream.next().await {
            let event = event.unwrap();
            sequence += 1;
            assert_eq!(event.sequence, sequence);
            match event.event.unwrap() {
                Event::Started(started) => {
                    let mut actual = started.step_ids;
                    actual.sort();
                    let mut expected: Vec<_> = STEPS.iter().copied().map(str::to_owned).collect();
                    expected.sort();
                    assert_eq!(actual, expected);
                }
                Event::LayersReady(batch) => {
                    assert!(batch.layers.iter().all(|l| l.model.is_none()));
                    completed.push(batch.completed_step);
                }
                Event::Complete(complete) => terminal = Some(complete),
                other => panic!("default pipeline failed: {other:?}"),
            }
        }
        assert_eq!(sequence, 12);
        assert_eq!(completed.len(), STEPS.len());
        let complete = terminal.unwrap();
        assert_eq!(complete.source_text, TEXT);
        assert_eq!(complete.layers.len(), 11);
        assert!(complete.layers.iter().all(|l| l.status == LayerWireStatus::Ok as i32 && l.model.is_none()));
        eprintln!("default_pipeline_graph_sha256 {}", turbo::bundle::sha256_hex(&complete.graph));
        let graph = DocumentGraph::from_bytes(&complete.graph).unwrap();
        let doc = import(&graph, codecs()).unwrap();
        assert_eq!(doc.source(), TEXT);
        let sentences = doc.get::<Sentences>().unwrap();
        let spans: Vec<_> = sentences.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(spans, [(0, 15), (16, 35), (36, 49)]);
        let tokens = doc.get::<Tokens>().unwrap();
        assert_eq!(tokens.spans.len(), WORDS.len());
        assert_eq!(
            tokens.sentence.as_ref().unwrap().iter().map(|i| i.get()).collect::<Vec<_>>(),
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        );
        let source = doc.get::<View>().unwrap();
        for ((_, span), &(a, b, u, v, c, d, word)) in tokens.spans.iter().zip(WORDS) {
            assert_eq!(span, Span::new(a, b));
            assert_eq!(&source.text()[span.range()], word);
            assert_eq!(doc.map_span(tokens.spans.view(), ViewId::SOURCE, span).unwrap(), span);
            let utf16 = source.span_to_utf16(span);
            assert_eq!((utf16.start.0, utf16.end.0), (u, v));
            assert_eq!(source.span_from_utf16(utf16).unwrap(), span);
            let cp = source.span_to_code_point(span);
            assert_eq!((cp.start.0, cp.end.0), (c, d));
            assert_eq!(source.span_from_code_point(cp).unwrap(), span);
        }
        if let Some(expected) = &reference {
            assert_eq!(&complete.graph, expected);
        } else {
            reference = Some(complete.graph);
        }
    }
    server.stop().await;
}
