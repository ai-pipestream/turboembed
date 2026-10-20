#[path = "routing/common.rs"]
mod common;
use std::{path::Path, time::Duration};
use turbo::bundle::Bundle;
use turbo_analysis_embed::routing::*;
use turbo_vector_index::{IndexBuilder, Metric, Normalization};

fn bundle(name: &str) -> Bundle {
    Bundle::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name)).unwrap()
}
fn cap(role: Role, width: u32, micros: u64) -> RoleCapability {
    RoleCapability {
        role,
        width,
        normalization: Normalization::Unit,
        context_limit: 64,
        cost_per_item: Duration::from_micros(micros),
    }
}
fn request() -> Request<'static> {
    Request {
        task: Task::Query,
        budget: LatencyBudget::MaxLatency(Duration::from_millis(1)),
        items: 1,
        tokens: 16,
        width: None,
        normalization: None,
        space: None,
    }
}
fn profile(b: &Bundle, cost: u64) -> ModelProfile {
    ModelProfile::from_bundle(b, vec![cap(Role::DenseQuery, b.manifest.embed().dim, cost)]).unwrap()
}
#[test]
fn real_bundles_choose_deterministically_and_respect_hard_budget() {
    let a = bundle("tiny-bert-bundle");
    let b = bundle("tiny-static-bundle");
    for reverse in [false, true] {
        let mut entries = vec![LoadedModel::new(0, profile(&a, 100)), LoadedModel::new(1, profile(&b, 50))];
        if reverse {
            entries.reverse();
        }
        let router = ModelRouter::new(entries, RoutingPolicy::Fastest).unwrap();
        for _ in 0..100 {
            let r = router.route(request()).unwrap();
            assert_eq!(*r.handle(), 1);
            assert_eq!(r.rationale(), Rationale::FastestWithinBudget);
            assert_eq!(r.space().dim, 16);
        }
        let q = Request { budget: LatencyBudget::MaxLatency(Duration::from_micros(49)), ..request() };
        match router.route(q) {
            Err(RoutingError::Unavailable { costs, .. }) => {
                assert_eq!(costs.len(), 2);
                assert!(costs.iter().any(|c| c.cost_per_item == Duration::from_micros(50)));
                assert!(costs.iter().any(|c| c.cost_per_item == Duration::from_micros(100)));
            }
            _ => panic!("over-budget request must fail with timings"),
        }
        assert!(router.route(Request { tokens: 65, ..request() }).is_err());
        assert!(router.route(Request { items: 21, ..request() }).is_err());
        assert!(router.route(Request { items: 20, ..request() }).is_ok());
        assert!(router.route(Request { budget: LatencyBudget::MinItemsPerSecond(20_000.0), ..request() }).is_ok());
        assert!(router.route(Request { budget: LatencyBudget::MinItemsPerSecond(20_000.1), ..request() }).is_err());
    }
}
#[test]
fn width_preference_and_ties_do_not_depend_on_registration_order() {
    let a = bundle("tiny-bert-bundle");
    let b = bundle("tiny-static-bundle");
    let mut expected = None;
    for reverse in [false, true] {
        let mut entries = vec![LoadedModel::new(0, profile(&a, 50)), LoadedModel::new(1, profile(&b, 50))];
        if reverse {
            entries.reverse();
        }
        let router = ModelRouter::new(entries, RoutingPolicy::Widest).unwrap();
        let r = router.route(request()).unwrap();
        assert_eq!(r.space().dim, 32);
        assert_eq!(r.rationale(), Rationale::WidestWithinBudget);
        if let Some(e) = &expected {
            assert_eq!(r.model(), e);
        }
        expected = Some(r.model().clone());
    }
    // Two real bundles with the same output width and equal cost.
    let variant = common::Variant::new();
    let b = Bundle::open(&variant.dir).unwrap();
    let p = profile(&b, 50);
    let mut chosen = None;
    for reverse in [false, true] {
        let mut entries = vec![LoadedModel::new(0, profile(&a, 50)), LoadedModel::new(1, p.clone())];
        if reverse {
            entries.reverse();
        }
        let r = ModelRouter::new(entries, RoutingPolicy::Fastest).unwrap();
        let model = r.route(request()).unwrap().model().clone();
        if let Some(c) = &chosen {
            assert_eq!(&model, c);
        }
        chosen = Some(model);
    }
}
#[test]
fn profiles_reject_unadvertised_widths_roles_and_duplicate_timings() {
    let b = bundle("tiny-bert-bundle");
    for c in [
        cap(Role::DenseQuery, 16, 1),
        cap(Role::StaticVectors, 32, 1),
        cap(Role::Sparse, 32, 1),
        cap(Role::Rerank, 1, 1),
    ] {
        assert!(ModelProfile::from_bundle(&b, vec![c]).is_err());
    }
    let c = cap(Role::DenseQuery, 32, 1);
    assert!(ModelProfile::from_bundle(&b, vec![c.clone(), c]).is_err());
    let p = profile(&b, 10);
    assert!(
        ModelRouter::new(vec![LoadedModel::new(0, p.clone()), LoadedModel::new(1, p)], RoutingPolicy::Fastest).is_err()
    );
    let s = bundle("tiny-static-bundle");
    assert!(ModelProfile::from_bundle(&s, vec![cap(Role::StaticVectors, 16, 1)]).is_ok());
    assert!(ModelProfile::from_bundle(&s, vec![cap(Role::TokenVectors, 16, 1)]).is_err());
    assert!(ModelProfile::from_bundle(&b, vec![cap(Role::TokenVectors, 32, 1)]).is_ok());
}
#[test]
fn numeric_boundaries_do_not_relax_budgets() {
    let b = bundle("tiny-bert-bundle");
    let p = profile(&b, 1);
    let router = ModelRouter::new(vec![LoadedModel::new(0, p.clone())], RoutingPolicy::Fastest).unwrap();
    for budget in [
        LatencyBudget::MaxLatency(Duration::ZERO),
        LatencyBudget::MinItemsPerSecond(0.0),
        LatencyBudget::MinItemsPerSecond(-1.0),
        LatencyBudget::MinItemsPerSecond(f64::NAN),
        LatencyBudget::MinItemsPerSecond(f64::INFINITY),
    ] {
        assert!(matches!(router.route(Request { budget, ..request() }), Err(RoutingError::Invalid(_))));
    }
    for q in
        [Request { items: 0, ..request() }, Request { tokens: 0, ..request() }, Request { width: Some(0), ..request() }]
    {
        assert!(matches!(router.route(q), Err(RoutingError::Invalid(_))));
    }
    let mut c = cap(Role::DenseQuery, 32, 1);
    c.cost_per_item = Duration::MAX;
    let huge = ModelProfile::from_bundle(&b, vec![c]).unwrap();
    let router = ModelRouter::new(vec![LoadedModel::new(0, huge)], RoutingPolicy::Fastest).unwrap();
    assert!(router.route(Request { items: 2, budget: LatencyBudget::MaxLatency(Duration::MAX), ..request() }).is_err());
    assert!(router.route(Request { items: 1, budget: LatencyBudget::MaxLatency(Duration::MAX), ..request() }).is_ok());
    let mut c = cap(Role::DenseQuery, 32, 0);
    assert!(ModelProfile::from_bundle(&b, vec![c.clone()]).is_err());
    c.cost_per_item = Duration::from_nanos(1);
    c.context_limit = 0;
    assert!(ModelProfile::from_bundle(&b, vec![c]).is_err());
}
#[test]
fn scoring_checks_all_three_spaces_even_for_empty_results() {
    let b = bundle("tiny-bert-bundle");
    let router = ModelRouter::new(vec![LoadedModel::new(0, profile(&b, 1))], RoutingPolicy::Fastest).unwrap();
    let route = router.route(request()).unwrap();
    let mut v = vec![0.0; 32];
    v[0] = 1.0;
    let mut builder = IndexBuilder::new(route.space().clone(), Metric::Dot).unwrap();
    builder.push(&v, "doc").unwrap();
    let index = builder.build();
    let q = QueryVector { space: route.space(), values: &v };
    assert_eq!(route.top_k(&index, q, 1).unwrap()[0].score, 1.0);
    for change in 0..4 {
        let mut other = route.space().clone();
        match change {
            0 => other.model.revision.push('x'),
            1 => other.model.sha256.push('x'),
            2 => other.dim = 16,
            _ => other.normalization = Normalization::None,
        }
        assert!(matches!(
            route.top_k(&index, QueryVector { space: &other, values: &v }, 0),
            Err(RoutingError::CrossSpace { .. })
        ));
        let wrong = IndexBuilder::<()>::new(other.clone(), Metric::Dot).unwrap().build();
        assert!(matches!(route.top_k(&wrong, q, 0), Err(RoutingError::CrossSpace { .. })));
        assert!(matches!(
            router.route(Request { space: Some(&other), ..request() }),
            Err(RoutingError::SpaceUnavailable { .. })
        ));
    }
    assert!(router.route(Request { space: Some(index.space()), width: Some(16), ..request() }).is_err());
    assert!(route.top_k(&index, QueryVector { space: route.space(), values: &[f32::NAN; 32] }, 1).is_err());
}
