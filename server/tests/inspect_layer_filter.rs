#[allow(dead_code)]
#[path = "inspect_safety/common.rs"]
mod common;
use common::*;
use turbo_analysis::{Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_kserve::proto::result::{LayerInstanceFilter, OffsetMode};

fn selector(layer: &str, instance: &str) -> LayerInstanceFilter {
    LayerInstanceFilter { layer: layer.into(), instance: instance.into() }
}

#[tokio::test]
async fn exact_layer_instances_preserve_text_offsets_and_legacy_filters() {
    let (srv, mut client) = start().await;
    let mut doc = base();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 7));
    doc.add(&Key::named("gold"), Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let bytes = graph(&doc);
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let unfiltered = client.inspect_document_window(window(bytes.clone(), mode)).await.unwrap().into_inner();
        for instances in [vec![""], vec!["gold"], vec!["", "gold"], vec!["missing"]] {
            let mut req = window(bytes.clone(), mode);
            req.filter_layer_instances = instances.iter().map(|instance| selector("turbo:tokens", instance)).collect();
            let out = client.inspect_document_window(req).await.unwrap().into_inner();
            assert_eq!(out.window, unfiltered.window);
            assert_eq!(out.summary, unfiltered.summary);
            assert_eq!(out.heatmap_lanes, unfiltered.heatmap_lanes);
            assert_eq!(out.segments.iter().map(|s| s.text.as_str()).collect::<String>(), "é 😀");
            let annotations: Vec<_> = out.segments.iter().flat_map(|s| &s.annotations).collect();
            assert!(annotations.iter().all(|a| a.layer == "turbo:tokens" && instances.contains(&a.instance.as_str())));
            for instance in instances.iter().filter(|&&instance| instance != "missing") {
                assert!(annotations.iter().any(|a| a.instance == *instance));
            }
            if instances == ["missing"] {
                assert!(annotations.is_empty());
            }
            for segment in &out.segments {
                let expected: Vec<_> = unfiltered
                    .segments
                    .iter()
                    .filter(|s| s.start <= segment.start && s.end >= segment.end)
                    .flat_map(|s| &s.annotations)
                    .filter(|a| a.layer == "turbo:tokens" && instances.contains(&a.instance.as_str()))
                    .collect();
                assert_eq!(segment.annotations.iter().collect::<Vec<_>>(), expected);
            }
        }
        let mut req = window(bytes.clone(), mode);
        req.filter_layer_instances = vec![selector("turbo:tokens", ""), selector("turbo:entities", "gold")];
        let pairs = client.inspect_document_window(req.clone()).await.unwrap().into_inner();
        let annotations: Vec<_> = pairs.segments.iter().flat_map(|s| &s.annotations).collect();
        assert!(!annotations.is_empty());
        assert!(
            annotations.iter().all(|a| a.layer == "turbo:tokens" && a.instance.is_empty()),
            "selectors match pairs, not independent lists"
        );
        req.filter_layer_instances.clear();
        req.filter_layers = vec![" turbo:tokens ".into()];
        let legacy = client.inspect_document_window(req.clone()).await.unwrap().into_inner();
        assert!(legacy.segments.iter().flat_map(|s| &s.annotations).any(|a| a.instance == "gold"));
        assert!(legacy.segments.iter().flat_map(|s| &s.annotations).any(|a| a.instance.is_empty()));
        req.filter_layer_instances = vec![selector("turbo:entities", "gold")];
        let out = client.inspect_document_window(req).await.unwrap().into_inner();
        assert!(out.segments.iter().all(|s| s.annotations.is_empty()), "legacy and exact constraints intersect");
    }
    drop(client);
    srv.stop().await;
}
