//! Hailo MiniLM latency/throughput bench — Raspberry Pi AI HAT+ only.
//!
//! Run on a provisioned Pi (docs/hailo-embed.md):
//!   cargo run -p turboembed --features hailo --release --example hailo_bench
//!
//! Prints a JSON receipt to stdout. No files are written.

#[cfg(feature = "hailo")]
mod imp {
    use std::time::Instant;

    use turboembed::{Device, EmbedOptions, Engine};

    const WARMUP: usize = 20;
    const ITERS: usize = 200;

    fn percentile(sorted: &[u128], pct: f64) -> u128 {
        let idx = ((sorted.len() as f64) * pct / 100.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn run() {
        let engine = Engine::create(Device::Hailo).expect("create hailo engine");
        engine
            .load_model("minilm")
            .expect("load minilm (run scripts/hailo-select-hef.sh + the exporter)");

        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(8);
        let texts = [
            "hello world",
            "Berlin is the capital of Germany and has 3.5 million inhabitants.",
            long_text.as_str(),
            "🦀 unicode emoji café 日本語",
        ];
        let opts = EmbedOptions::default();

        let t0 = Instant::now();
        let first = engine
            .embed_one("minilm", texts[0], &opts)
            .expect("first embed");
        let first_us = t0.elapsed().as_micros();
        assert_eq!(first.dim(), 384);
        for _ in 0..WARMUP {
            engine.embed_one("minilm", texts[0], &opts).unwrap();
        }

        let mut report = Vec::new();
        for text in texts {
            let mut lat: Vec<u128> = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t = Instant::now();
                engine.embed_one("minilm", text, &opts).unwrap();
                lat.push(t.elapsed().as_micros());
            }
            lat.sort_unstable();
            let p50 = percentile(&lat, 50.0);
            let p95 = percentile(&lat, 95.0);
            let p99 = percentile(&lat, 99.0);
            report.push(format!(
                "    {{\"text_len\": {}, \"p50_us\": {p50}, \"p95_us\": {p95}, \"p99_us\": {p99}, \"emb_per_s\": {:.1}}}",
                text.len(),
                1e6 / p50 as f64
            ));
        }

        let batch: Vec<&str> = std::iter::repeat_n(
            "Berlin is the capital of Germany and has 3.5 million inhabitants.",
            32,
        )
        .collect();
        let mut batch_lat: Vec<u128> = Vec::with_capacity(50);
        for _ in 0..50 {
            let t = Instant::now();
            engine.embed("minilm", &batch, &opts).unwrap();
            batch_lat.push(t.elapsed().as_micros());
        }
        batch_lat.sort_unstable();
        let b50 = percentile(&batch_lat, 50.0);

        println!("{{");
        println!("  \"device\": \"hailo\",");
        println!("  \"model\": \"minilm (encoder HEF, INT8)\",");
        println!("  \"first_call_us\": {first_us},");
        println!("  \"warmup\": {WARMUP}, \"iters\": {ITERS},");
        println!("  \"single_stream\": [");
        println!("{}", report.join(",\n"));
        println!("  ],");
        println!(
            "  \"batch32_p50_us\": {b50}, \"batch32_rows_per_s\": {:.1}",
            32.0 * 1e6 / b50 as f64
        );
        println!("}}");
    }
}

fn main() {
    #[cfg(feature = "hailo")]
    imp::run();
    #[cfg(not(feature = "hailo"))]
    eprintln!("hailo_bench needs --features hailo (Raspberry Pi AI HAT+)");
}
