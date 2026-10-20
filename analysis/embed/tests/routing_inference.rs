#[path = "routing/common.rs"]
mod common;
#[path = "../../../core/tests/rerank/common.rs"]
mod rerank_fixture;
use std::{
    path::Path,
    time::{Duration, Instant},
};
use turbo::bundle::Bundle;
use turbo::rerank::{RerankOptions, Reranker};
use turbo_analysis_embed::routing::*;
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};
use turbo_vector_index::{IndexBuilder, Metric, Normalization};

fn request(task: Task) -> Request<'static> {
    Request {
        task,
        budget: LatencyBudget::MaxLatency(Duration::from_secs(60)),
        items: 1,
        tokens: 8,
        width: None,
        normalization: None,
        space: None,
    }
}
#[test]
fn two_loaded_same_width_bundles_cannot_score_each_others_vectors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let variant = common::Variant::new();
    let paths = [root.join("testdata/tiny-bert-bundle"), variant.dir.clone()];
    let mut rows = Rows::new(64, 0);
    rows.push(&[&[2, 10, 11, 3]]);
    let mut handles = Vec::new();
    let mut outputs = Vec::new();
    let mut encoders = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let b = Bundle::open(path).unwrap();
        let mut encoder = Encoder::open(path, &EncoderOptions { cpu_threads: 1, ..Default::default() }).unwrap();
        let mut values = Vec::new();
        let start = Instant::now();
        encoder.run_rows(&rows, true, &mut values).unwrap();
        let cost = start.elapsed().max(Duration::from_nanos(1));
        let caps = [Role::DenseDocument, Role::DenseQuery]
            .map(|role| RoleCapability {
                role,
                width: 32,
                normalization: Normalization::Unit,
                context_limit: 64,
                cost_per_item: cost,
            })
            .to_vec();
        let p = ModelProfile::from_bundle(&b, caps).unwrap();
        handles.push(LoadedModel::new(i, p));
        outputs.push(values);
        encoders.push(encoder);
    }
    let r = ModelRouter::new(handles, RoutingPolicy::Fastest).unwrap();
    let spaces: Vec<_> = r
        .models()
        .iter()
        .map(|m| turbo_vector_index::Space::new(m.profile().model().clone(), 32, Normalization::Unit))
        .collect();
    assert_ne!(spaces[0].model, spaces[1].model);
    let q0 = r.route(Request { space: Some(&spaces[0]), ..request(Task::Query) }).unwrap();
    let q1 = r.route(Request { space: Some(&spaces[1]), ..request(Task::Query) }).unwrap();
    let mut builder = IndexBuilder::new(spaces[0].clone(), Metric::Dot).unwrap();
    builder.push(&outputs[*q0.handle()], "doc").unwrap();
    let index = builder.build();
    assert_eq!(
        q0.top_k(&index, QueryVector { space: q0.space(), values: &outputs[*q0.handle()] }, 1).unwrap().len(),
        1
    );
    assert!(matches!(
        q1.top_k(&index, QueryVector { space: q1.space(), values: &outputs[*q1.handle()] }, 1),
        Err(RoutingError::CrossSpace { .. })
    ));
}
#[test]
fn advertised_matryoshka_widths_and_reranker_are_routed() {
    let variant = common::Variant::new();
    let b = Bundle::open(&variant.dir).unwrap();
    let caps = [16, 32]
        .map(|width| RoleCapability {
            role: Role::DenseQuery,
            width,
            normalization: Normalization::Unit,
            context_limit: 64,
            cost_per_item: Duration::from_micros(u64::from(width)),
        })
        .to_vec();
    let p = ModelProfile::from_bundle(&b, caps).unwrap();
    let router = ModelRouter::new(vec![LoadedModel::new(0, p)], RoutingPolicy::Widest).unwrap();
    assert_eq!(router.route(request(Task::Query)).unwrap().space().dim, 32);
    assert_eq!(
        router
            .route(Request { budget: LatencyBudget::MaxLatency(Duration::from_micros(20)), ..request(Task::Query) })
            .unwrap()
            .space()
            .dim,
        16
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = rerank_fixture::Fixture::new(&root);
    let b = Bundle::open(&fixture.dir).unwrap();
    let mut reranker = Reranker::open(&fixture.dir, 1).unwrap();
    let start = Instant::now();
    let result = reranker.score(&[("France", "Paris")], RerankOptions::default()).unwrap();
    let cost = start.elapsed().max(Duration::from_nanos(1));
    let p = ModelProfile::from_bundle(
        &b,
        vec![RoleCapability {
            role: Role::Rerank,
            width: 1,
            normalization: Normalization::None,
            context_limit: 32,
            cost_per_item: cost,
        }],
    )
    .unwrap();
    let router = ModelRouter::new(vec![LoadedModel::new(0, p)], RoutingPolicy::Fastest).unwrap();
    let route = router.route(request(Task::Rerank)).unwrap();
    assert_eq!(route.model().sha256, result.model.manifest_sha256);
    assert!(router.route(request(Task::Query)).is_err());
}
