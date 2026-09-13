//! Merge measured TurboEmbed + TurboRerank phase receipts into one file.

use std::process::ExitCode;

use bench_turbo::{git_head, hostname, nvidia_gpu_name, read_json, workspace_root, write_json};
use clap::Parser;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(
    name = "bench-turbo",
    about = "Merge Machine A/B/C turbo bench phase JSON into testdata/receipts/bench/"
)]
struct Args {
    /// Machine letter (`A` on NVIDIA CUDA).
    #[arg(long, env = "MACHINE")]
    machine: String,
    #[arg(long)]
    embed: std::path::PathBuf,
    #[arg(long)]
    rerank: std::path::PathBuf,
    #[arg(long)]
    out: std::path::PathBuf,
}

fn require_pass(phase: &str, v: &Value) -> Result<bool, String> {
    v.get("pass")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{phase} JSON missing pass"))
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("bench-turbo: {e}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<bool, String> {
    let args = Args::parse();
    let letter = args.machine.trim().to_ascii_uppercase();
    if letter != "A" {
        return Err(format!(
            "this merger writes Machine A CUDA receipts; got MACHINE={letter}"
        ));
    }
    let embed = read_json(&args.embed)?;
    let rerank = read_json(&args.rerank)?;
    if embed["engine"] != "turboembed" {
        return Err("embed phase is not turboembed".into());
    }
    if rerank["engine"] != "turborerank" {
        return Err("rerank phase is not turborerank".into());
    }
    if embed["device"] != "CUDA" || rerank["device"] != "CUDA" {
        return Err("both phases must be CUDA on Machine A".into());
    }
    let embed_ok = require_pass("embed", &embed)?;
    let rerank_ok = require_pass("rerank", &rerank)?;
    let root = workspace_root();
    let gpu = nvidia_gpu_name().unwrap_or_else(|_| {
        embed["gpu"]
            .as_str()
            .or_else(|| rerank["gpu"].as_str())
            .unwrap_or("unknown")
            .to_string()
    });
    let pass = embed_ok && rerank_ok;
    let receipt = serde_json::json!({
        "schema_version": 1,
        "kind": "turbo-bench",
        "machine": "Machine A",
        "device": "CUDA",
        "gpu": gpu,
        "host": hostname(),
        "git_sha": git_head(&root),
        "command": "make bench-machine-a",
        "commands": [
            "make bench-machine-a",
            "make bench-turbo MACHINE=A"
        ],
        "pass": pass,
        "turboembed": embed,
        "turborerank": rerank,
        "notes": "SOLIDIFY (7) Machine A CUDA bench. p50/p99 are nearest-rank over N≥100 timed forwards after warmup. token_h2d and embed hidden_d2h are intercepted counters (must be 0 on the fixed paths). allocs/forward must be 0. Berlin HF CE and nvidia MiniLM embed goldens must stay in band. Numbers are measured on this GPU — do not hand-edit."
    });
    write_json(&args.out, &receipt)?;
    eprintln!(
        "Machine A CUDA bench pass={pass} embed_p50={}us embed_p99={}us rerank_p50={}us rerank_p99={}us",
        embed["latency_us"]["p50"],
        embed["latency_us"]["p99"],
        rerank["latency_us"]["p50"],
        rerank["latency_us"]["p99"]
    );
    Ok(pass)
}
