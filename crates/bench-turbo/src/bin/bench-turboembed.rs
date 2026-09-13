//! Machine A TurboEmbed MiniLM CUDA bench. Numbers are measured.

use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use bench_turbo::{
    cosine, git_head, hostname, nvidia_gpu_name, summarize_latencies, workspace_root, write_json,
    DEFAULT_ITERS, DEFAULT_WARMUP,
};
use clap::Parser;
use serde_json::Value;
use turboembed::{Device, EmbedOptions, Engine, Pooling};

const ALIAS: &str = "minilm";
const GOLDEN: &str = "testdata/e2e/goldens/nvidia/minilm.json";
const COSINE_FLOOR: f32 = 0.99;
const LATENCY_TEXT: &str = "hello world";
const SUBSET_PREFIX: &str = "parity:";

unsafe extern "C" {
    fn turboembed_ort_hot_path_reset();
    fn turboembed_ort_arena_allocs() -> u64;
    fn turboembed_ort_external_allocs() -> u64;
    fn turboembed_ort_d2h_bytes() -> u64;
    fn turboembed_ort_d2h_calls() -> u64;
    fn turboembed_ort_d2h_result_bytes() -> u64;
    fn turboembed_ort_cuda_forward_allocs() -> u64;
    fn turboembed_ort_cuda_forward_h2d_bytes() -> u64;
    fn turboembed_ort_cuda_forward_h2d_calls() -> u64;
}

fn reset_hot_path() {
    unsafe { turboembed_ort_hot_path_reset() };
}

#[derive(Parser, Debug)]
#[command(name = "bench-turboembed", about = "Measure TurboEmbed MiniLM on CUDA")]
struct Args {
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    warmup: u32,
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    iters: u32,
    #[arg(long)]
    out: std::path::PathBuf,
}

fn load_subset(path: &Path) -> Result<(usize, Vec<(String, String, Vec<f32>)>, Vec<f32>), String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let golden: Value = serde_json::from_str(&raw).map_err(|e| format!("golden JSON: {e}"))?;
    if golden["alias"] != "minilm" {
        return Err("golden alias is not minilm".into());
    }
    let dim = golden["dim"].as_u64().ok_or("golden dim")? as usize;
    let hello: Vec<f32> = golden["vector"]
        .as_array()
        .ok_or("top-level vector")?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let mut subset = Vec::new();
    for item in golden["items"].as_array().ok_or("items")? {
        let id = item["id"].as_str().unwrap_or_default();
        if !id.starts_with(SUBSET_PREFIX) {
            continue;
        }
        let text = item["text"].as_str().ok_or("item.text")?.to_string();
        let vector: Vec<f32> = item["vector"]
            .as_array()
            .ok_or("item.vector")?
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        subset.push((id.to_string(), text, vector));
    }
    if subset.is_empty() {
        return Err(format!("no {SUBSET_PREFIX}* items"));
    }
    Ok((dim, subset, hello))
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("bench-turboembed: {e}");
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
    let gpu = nvidia_gpu_name()?;
    let (dim, subset, hello) = load_subset(&root.join(GOLDEN))?;

    let engine = Engine::create(Device::Cuda).map_err(|e| format!("create CUDA: {e}"))?;
    engine
        .load_model(ALIAS)
        .map_err(|e| format!("load_model({ALIAS}): {e}"))?;
    let info = engine
        .list_models()
        .map_err(|e| e.to_string())?
        .get(0)
        .ok_or_else(|| "list_models empty".to_string())?;
    if info.device != Device::Cuda {
        return Err(format!(
            "FAKE: list_models device is {:?}, not CUDA",
            info.device
        ));
    }
    if info.dim != dim as u32 {
        return Err(format!("dim {} != golden {dim}", info.dim));
    }

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };

    // Goldens first so a broken engine never writes a latency-only receipt.
    reset_hot_path();
    let hello_live = engine
        .embed_one(ALIAS, LATENCY_TEXT, &opts)
        .map_err(|e| format!("embed_one hello world: {e}"))?;
    if hello_live.dim() == 8 {
        return Err("FAKE: dim=8 FNV mock".into());
    }
    let hello_cos = cosine(hello_live.values(), &hello);
    drop(hello_live);

    let mut worst = 1.0_f32;
    let mut sum = 0.0_f32;
    for (id, text, expected) in &subset {
        let got = engine
            .embed_one(ALIAS, text, &opts)
            .map_err(|e| format!("embed_one {id}: {e}"))?;
        let sim = cosine(got.values(), expected);
        if sim < COSINE_FLOOR {
            return Err(format!("{id}: cosine {sim} < {COSINE_FLOOR}"));
        }
        worst = worst.min(sim);
        sum += sim;
        drop(got);
    }
    let mean_cos = sum / subset.len() as f32;
    let goldens_ok = hello_cos >= COSINE_FLOOR && worst >= COSINE_FLOOR;

    for _ in 0..args.warmup {
        let one = engine
            .embed_one(ALIAS, LATENCY_TEXT, &opts)
            .map_err(|e| format!("warmup: {e}"))?;
        drop(one);
    }

    reset_hot_path();
    let probe = engine
        .embed_one(ALIAS, LATENCY_TEXT, &opts)
        .map_err(|e| format!("steady probe: {e}"))?;
    drop(probe);
    let token_h2d = unsafe { turboembed_ort_cuda_forward_h2d_bytes() };
    let token_h2d_calls = unsafe { turboembed_ort_cuda_forward_h2d_calls() };
    let hidden_d2h = unsafe { turboembed_ort_d2h_bytes() };
    let hidden_d2h_calls = unsafe { turboembed_ort_d2h_calls() };
    let d2h_result = unsafe { turboembed_ort_d2h_result_bytes() };
    let arena_allocs = unsafe { turboembed_ort_arena_allocs() };
    let cuda_fwd_allocs = unsafe { turboembed_ort_cuda_forward_allocs() };
    let ort_ext = unsafe { turboembed_ort_external_allocs() };
    let zeros_ok = token_h2d == 0
        && token_h2d_calls == 0
        && hidden_d2h == 0
        && hidden_d2h_calls == 0
        && d2h_result == 0
        && arena_allocs == 0
        && cuda_fwd_allocs == 0
        && ort_ext == 0;

    let mut samples = Vec::with_capacity(args.iters as usize);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let one = engine
            .embed_one(ALIAS, LATENCY_TEXT, &opts)
            .map_err(|e| format!("timed embed: {e}"))?;
        let us = bench_turbo::duration_us(t0.elapsed());
        drop(one);
        samples.push(us);
    }
    let latency = summarize_latencies(samples);

    let pass = goldens_ok && zeros_ok;
    let receipt = serde_json::json!({
        "engine": "turboembed",
        "alias": ALIAS,
        "device": "CUDA",
        "provider": "ORT CUDA EP + IoBinding + DEVICE mean+L2",
        "gpu": gpu,
        "host": hostname(),
        "git_sha": git_head(&root),
        "pass": pass,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "workload": {
            "kind": "embed_one",
            "text": LATENCY_TEXT,
            "pooling": "mean",
            "normalize": true,
        },
        "latency_us": latency,
        "bytes_per_forward": {
            "token_h2d": token_h2d,
            "token_h2d_calls": token_h2d_calls,
            "hidden_d2h": hidden_d2h,
            "hidden_d2h_calls": hidden_d2h_calls,
            "d2h_result_bytes": d2h_result,
        },
        "allocs_per_forward": arena_allocs,
        "cuda_forward_allocs": cuda_fwd_allocs,
        "ort_gpu_external_allocs": ort_ext,
        "goldens": {
            "path": GOLDEN,
            "hello_world_cosine": hello_cos,
            "worst_cosine": worst,
            "mean_cosine": mean_cos,
            "n_texts": subset.len(),
            "subset": "parity:*",
            "threshold": COSINE_FLOOR,
            "in_band": goldens_ok,
        },
        "notes": "Steady-state after warmup. token_h2d / hidden_d2h are intercepted counters on one post-warmup embed_one. Latency is Instant around embed_one (tokenize + ORT Run + DEVICE mean+L2)."
    });
    write_json(&args.out, &receipt)?;
    eprintln!(
        "turboembed MiniLM CUDA: p50={}us p99={}us token_h2d={} hidden_d2h={} allocs={} goldens_ok={} pass={}",
        receipt["latency_us"]["p50"],
        receipt["latency_us"]["p99"],
        token_h2d,
        hidden_d2h,
        arena_allocs,
        goldens_ok,
        pass
    );
    if !zeros_ok {
        eprintln!(
            "hot path not zero: h2d={token_h2d}/{token_h2d_calls} hidden_d2h={hidden_d2h} \
             result_d2h={d2h_result} arena={arena_allocs} cudaMalloc={cuda_fwd_allocs} ext={ort_ext}"
        );
    }
    Ok(pass)
}
