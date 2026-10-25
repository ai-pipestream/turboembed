#![cfg(feature = "cpu")]
//! Encoder latency at real DeBERTa-v3 shapes, on synthetic weights (speed
//! does not depend on their values), on one thread and on every processor,
//! one sequence a call and a batch of them. Ignored by default; run in
//! release on an otherwise idle machine:
//! `cargo test --release --locked -p turbo --test deberta_speed -- --ignored --nocapture`
#[path = "deberta_support/mod.rs"]
mod deberta_support;

use std::time::Instant;

use deberta_support::weights;
use turbo::deberta::{Config, Encoder};

fn config(hidden: usize, heads: usize) -> Config {
    Config {
        hidden,
        intermediate: 4 * hidden,
        heads,
        layers: 12,
        vocabulary: 1000,
        max_sequence: 512,
        position_buckets: 256,
        max_relative_positions: 512,
        layer_norm_eps: 1e-7,
    }
}

/// The median of timed calls of `f`: at least 3, then until a second has
/// passed or 50 were made, after one untimed call.
fn median_ms(mut f: impl FnMut()) -> (f64, usize) {
    f();
    let (mut times, start) = (Vec::new(), Instant::now());
    while times.len() < 3 || (start.elapsed().as_secs_f64() < 1.0 && times.len() < 50) {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times.len())
}

#[test]
#[ignore = "latency measurement; run in release on an idle machine"]
fn encoder_latency_at_deberta_v3_shapes() {
    let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    for (model, hidden, heads) in [("deberta-v3-xsmall", 384, 6), ("gliner2-base", 768, 12)] {
        let c = config(hidden, heads);
        let bytes = weights(&c);
        for threads in [1, most] {
            let encoder = Encoder::from_safetensors(c.clone(), &bytes).unwrap().with_threads(threads).unwrap();
            for active in [16, 64, 128] {
                for padded in [false, true] {
                    let n = if padded { 512 } else { active };
                    let ids: Vec<u32> = (0..n).map(|i| (i * 37 % 1000) as u32).collect();
                    let mask: Vec<bool> = (0..n).map(|i| i < active).collect();
                    let (ms, runs) = median_ms(|| {
                        std::hint::black_box(encoder.encode(&ids, &mask).unwrap());
                    });
                    println!(
                        "bench model={model} threads={threads} active={active} len={n} median_ms={ms:.3} runs={runs}"
                    );
                }
                // A batch of 16 sequences of this length, one call.
                let rows: Vec<(Vec<u32>, Vec<bool>)> = (0..16)
                    .map(|r| ((0..active).map(|i| ((i * 37 + r) % 1000) as u32).collect(), vec![true; active]))
                    .collect();
                let views: Vec<(&[u32], &[bool])> = rows.iter().map(|(i, m)| (i.as_slice(), m.as_slice())).collect();
                let (ms, runs) = median_ms(|| {
                    for r in encoder.encode_batch(&views) {
                        std::hint::black_box(r.unwrap());
                    }
                });
                println!(
                    "bench model={model} threads={threads} batch=16 active={active} median_ms={ms:.3} runs={runs}"
                );
            }
        }
    }
}
