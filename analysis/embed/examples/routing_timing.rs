//! Release ABBA comparison against direct calls to the same loaded encoder.
//! Run pinned to one CPU; output contains individual rounds for inspection.
use std::{
    hint::black_box,
    path::Path,
    time::{Duration, Instant},
};
use turbo::bundle::Bundle;
use turbo_analysis_embed::routing::*;
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};
use turbo_vector_index::Normalization;

fn elapsed(
    encoders: &mut [Encoder],
    rows: &Rows,
    router: &ModelRouter<usize>,
    q: Request<'_>,
    routed: bool,
    iterations: u32,
    out: &mut Vec<f32>,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let handle = if routed { *router.route(black_box(q)).unwrap().handle() } else { 0 };
        encoders[black_box(handle)].run_rows(black_box(rows), true, out).unwrap();
        black_box(&*out);
    }
    start.elapsed().as_secs_f64() * 1e9 / f64::from(iterations)
}
fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("testdata/tiny-bert-bundle");
    let b = Bundle::open(&path).unwrap();
    let mut encoder = Encoder::open(&path, &EncoderOptions { cpu_threads: 1, ..Default::default() }).unwrap();
    let mut rows = Rows::new(64, 0);
    rows.push(&[&[2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 3]]);
    let mut out = Vec::new();
    for _ in 0..100 {
        encoder.run_rows(&rows, true, &mut out).unwrap();
    }
    let start = Instant::now();
    for _ in 0..200 {
        encoder.run_rows(&rows, true, &mut out).unwrap();
    }
    let cost = start.elapsed() / 200;
    let profile = ModelProfile::from_bundle(
        &b,
        vec![RoleCapability {
            role: Role::DenseQuery,
            width: 32,
            normalization: Normalization::Unit,
            context_limit: 16,
            cost_per_item: cost,
        }],
    )
    .unwrap();
    let router = ModelRouter::new(vec![LoadedModel::new(0, profile)], RoutingPolicy::Fastest).unwrap();
    let q = Request {
        task: Task::Query,
        budget: LatencyBudget::MaxLatency(Duration::from_secs(1)),
        items: 1,
        tokens: 16,
        width: Some(32),
        normalization: Some(Normalization::Unit),
        space: None,
    };
    println!("model={} cost_ns={}", router.models()[0].profile().model().sha256, cost.as_nanos());
    let mut encoders = [encoder];
    let mut ratios = Vec::new();
    for round in 0..12 {
        let mut direct = 0.0;
        let mut routed = 0.0;
        for route in [false, true, true, false] {
            let ns = elapsed(&mut encoders, &rows, &router, q, route, 3000, &mut out);
            if route {
                routed += ns;
            } else {
                direct += ns;
            }
        }
        let delta = (routed / direct - 1.0) * 100.0;
        ratios.push(delta);
        println!("round={round} direct_ns={:.1} routed_ns={:.1} delta_pct={delta:.4}", direct / 2.0, routed / 2.0);
    }
    ratios.sort_by(f64::total_cmp);
    println!("median_delta_pct={:.4}", (ratios[5] + ratios[6]) / 2.0);
}
