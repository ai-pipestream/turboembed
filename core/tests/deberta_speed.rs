#![cfg(feature = "cpu")]
//! Encoder latency at real DeBERTa-v3 shapes, on synthetic weights (speed
//! does not depend on their values). Ignored by default; run in release,
//! pinned to one core:
//! `taskset -c 2 cargo test --release --locked -p turbo --test deberta_speed -- --ignored --nocapture`
use std::time::Instant;
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

/// Canonical safetensors with every tensor the configuration names:
/// LayerNorm weights 1, every other value small and spread.
fn weights(c: &Config) -> Vec<u8> {
    let shapes = c.tensor_shapes().unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for (name, shape) in &shapes {
        let n: u64 = shape.iter().product();
        let start = data.len();
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let v =
                if name.ends_with("LayerNorm.weight") { 1.0 } else { ((seed >> 40) as f32 / 16_777_216.0 - 0.5) * 0.1 };
            data.extend_from_slice(&f32::to_le_bytes(v));
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}

#[test]
#[ignore = "latency measurement; run in release on a pinned core"]
fn encoder_latency_at_deberta_v3_shapes() {
    for (model, hidden, heads) in [("deberta-v3-xsmall", 384, 6), ("gliner2-base", 768, 12)] {
        let c = config(hidden, heads);
        let encoder = Encoder::from_safetensors(c.clone(), &weights(&c)).unwrap();
        for active in [16, 64, 128] {
            for padded in [false, true] {
                let n = if padded { 512 } else { active };
                let ids: Vec<u32> = (0..n).map(|i| (i * 37 % 1000) as u32).collect();
                let mask: Vec<bool> = (0..n).map(|i| i < active).collect();
                encoder.encode(&ids, &mask).unwrap();
                let (mut times, start) = (Vec::new(), Instant::now());
                while times.len() < 3 || (start.elapsed().as_secs_f64() < 1.0 && times.len() < 50) {
                    let t = Instant::now();
                    std::hint::black_box(encoder.encode(&ids, &mask).unwrap());
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "bench model={model} active={active} len={n} median_ms={:.3} runs={}",
                    times[times.len() / 2],
                    times.len()
                );
            }
        }
    }
}
