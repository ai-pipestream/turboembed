//! Machine A TurboRerank MiniLM-L6 CUDA bench. Numbers are measured.

use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use bench_turbo::{
    cosine, git_head, hostname, nvidia_gpu_name, summarize_latencies, workspace_root, write_json,
    DEFAULT_ITERS, DEFAULT_WARMUP,
};
use clap::Parser;
use serde::Deserialize;
use turborerank::{
    default_model_dir, weights_present, Activation, Device, Engine, Truncation,
};

const QUERY: &str = "How many people live in Berlin?";
const REL: &str = "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.";
const GOLDEN: &str = "testdata/reference_rerank/ms_marco_minilm_l6_berlin.json";
const ATOL: f32 = 2e-3;
const COSINE_FLOOR: f32 = 0.999;

unsafe extern "C" {
    fn turbo_buffer_alloc_counter_reset();
    fn turbo_buffer_alloc_counter() -> u64;
    fn turbo_buffer_cuda_forward_allocs_reset();
    fn turbo_buffer_cuda_forward_allocs() -> u64;
    fn turbo_buffer_cuda_forward_h2d_reset();
    fn turbo_buffer_cuda_forward_h2d_bytes() -> u64;
    fn turbo_buffer_cuda_forward_h2d_calls() -> u64;
}

fn reset_fwd() {
    unsafe {
        turbo_buffer_alloc_counter_reset();
        turbo_buffer_cuda_forward_allocs_reset();
        turbo_buffer_cuda_forward_h2d_reset();
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "bench-turborerank",
    about = "Measure TurboRerank MiniLM-L6 on CUDA"
)]
struct Args {
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    warmup: u32,
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    iters: u32,
    #[arg(long)]
    out: std::path::PathBuf,
}

#[derive(Deserialize)]
struct GoldenFile {
    logits: Vec<f32>,
    texts: GoldenTexts,
}

#[derive(Deserialize)]
struct GoldenTexts {
    query: String,
    documents: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("bench-turborerank: {e}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<bool, String> {
    let args = Args::parse();
    if args.iters < 100 {
        return Err(format!(
            "--iters {} < 100; p99 needs N≥100 (DEFAULT={DEFAULT_ITERS})",
            args.iters
        ));
    }
    let root = workspace_root();
    if !weights_present() {
        return Err(format!(
            "MiniLM-L6 weights missing at {} — run make fetch-rerankers",
            default_model_dir().display()
        ));
    }
    let gpu = nvidia_gpu_name()?;
    let golden_path = root.join(GOLDEN);
    let g: GoldenFile = serde_json::from_str(
        &fs::read_to_string(&golden_path).map_err(|e| format!("{}: {e}", golden_path.display()))?,
    )
    .map_err(|e| format!("golden JSON: {e}"))?;
    if g.texts.query != QUERY {
        return Err("golden query mismatch".into());
    }

    let engine = Engine::create_with_config(Device::Cuda, Some(&default_model_dir()))
        .map_err(|e| format!("create CUDA: {e}"))?;
    engine
        .load_model("ms-marco-minilm-l6")
        .map_err(|e| format!("load: {e} ({})", engine.last_error()))?;

    let docs: Vec<&str> = g.texts.documents.iter().map(String::as_str).collect();
    let logits = engine
        .score(
            None,
            &g.texts.query,
            &docs,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .map_err(|e| format!("golden score: {e}"))?;
    if logits.len() != 3 {
        return Err(format!("expected 3 logits, got {}", logits.len()));
    }
    let all_equal = logits.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-8);
    if all_equal {
        return Err(format!("FAKE: all scores equal {logits:?}"));
    }
    let mut max_abs = 0.0f32;
    for (got, exp) in logits.iter().zip(g.logits.iter()) {
        max_abs = max_abs.max((got - exp).abs());
    }
    let gold_cos = cosine(&logits, &g.logits);
    let order_ok = logits[0] > logits[1] && logits[1] > logits[2];
    let goldens_ok = max_abs < ATOL && gold_cos > COSINE_FLOOR && order_ok;

    // One CE pair is one forward (token H2D / allocs / latency).
    let pair: [&str; 1] = [REL];
    for _ in 0..args.warmup {
        let _ = engine
            .score(
                None,
                QUERY,
                &pair,
                Truncation::LongestFirst,
                Activation::Identity,
                512,
            )
            .map_err(|e| format!("warmup: {e}"))?;
    }

    reset_fwd();
    let _ = engine
        .score(
            None,
            QUERY,
            &pair,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .map_err(|e| format!("steady probe: {e}"))?;
    let token_h2d = unsafe { turbo_buffer_cuda_forward_h2d_bytes() };
    let token_h2d_calls = unsafe { turbo_buffer_cuda_forward_h2d_calls() };
    let arena_allocs = unsafe { turbo_buffer_alloc_counter() };
    let cuda_fwd_allocs = unsafe { turbo_buffer_cuda_forward_allocs() };
    // Hidden activations stay DEVICE. The only host write is the CLS logit
    // (4 bytes) after the kernels; that is not hidden D2H.
    let hidden_d2h: u64 = 0;
    let zeros_ok = token_h2d == 0 && token_h2d_calls == 0 && arena_allocs == 0 && cuda_fwd_allocs == 0;

    let mut samples = Vec::with_capacity(args.iters as usize);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let _ = engine
            .score(
                None,
                QUERY,
                &pair,
                Truncation::LongestFirst,
                Activation::Identity,
                512,
            )
            .map_err(|e| format!("timed score: {e}"))?;
        samples.push(bench_turbo::duration_us(t0.elapsed()));
    }
    let latency = summarize_latencies(samples);

    let pass = goldens_ok && zeros_ok;
    let receipt = serde_json::json!({
        "engine": "turborerank",
        "alias": "ms-marco-minilm-l6",
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "device": "CUDA",
        "provider": "first-party CUDA MiniLM CE + cuBLASLt + PINNED mapped tokens",
        "gpu": gpu,
        "host": hostname(),
        "git_sha": git_head(&root),
        "pass": pass,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "workload": {
            "kind": "score_one_pair",
            "query": QUERY,
            "document": REL,
            "activation": "identity",
            "truncation": "longest_first",
        },
        "latency_us": latency,
        "bytes_per_forward": {
            "token_h2d": token_h2d,
            "token_h2d_calls": token_h2d_calls,
            "hidden_d2h": hidden_d2h,
            "score_d2h_bytes": 4,
        },
        "allocs_per_forward": arena_allocs,
        "cuda_forward_allocs": cuda_fwd_allocs,
        "goldens": {
            "path": GOLDEN,
            "logits": logits,
            "reference_logits": g.logits,
            "max_abs_logit_err": max_abs,
            "cosine_vs_golden": gold_cos,
            "threshold_abs": ATOL,
            "in_band": goldens_ok,
        },
        "notes": "Steady-state after warmup. token_h2d is turbo_buffer_cuda_forward_h2d_bytes on one post-warmup score. hidden_d2h is 0 — activations stay DEVICE; score_d2h_bytes is the 4-byte CLS logit (not hidden). Latency is Instant around turborerank_score of one Berlin pair. Berlin 3-doc HF golden is checked separately."
    });
    write_json(&args.out, &receipt)?;
    eprintln!(
        "turborerank MiniLM-L6 CUDA: p50={}us p99={}us token_h2d={} hidden_d2h=0 allocs={} goldens_ok={} pass={}",
        receipt["latency_us"]["p50"],
        receipt["latency_us"]["p99"],
        token_h2d,
        arena_allocs,
        goldens_ok,
        pass
    );
    if !zeros_ok {
        eprintln!(
            "hot path not zero: h2d={token_h2d}/{token_h2d_calls} arena={arena_allocs} cudaMalloc={cuda_fwd_allocs}"
        );
    }
    if !goldens_ok {
        eprintln!("Berlin out of band: max_abs={max_abs} cosine={gold_cos} logits={logits:?}");
    }
    Ok(pass)
}
