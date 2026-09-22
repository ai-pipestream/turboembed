//! The direct-native half of the ggml provider's matched benchmark
//! (PLAN.md section 11): llama.cpp through the `llama-cpp-2` crate alone,
//! with no libturbo in the timed path.
//!
//! `generate` runs `turbo-bench generate`'s workload (the same prompt
//! through the GGUF's own chat template, `new_tokens` tokens, greedy, the
//! end token suppressed so every iteration produces the same count) and
//! `embed` runs the texts of a `turbo-bench embed --dump-tokens` dump
//! through llama.cpp's own tokenizer and pooling. Both write a receipt of
//! kind `native` that `turbo-bench compare` reads.
//!
//! ```text
//! reference-llama-cpp generate --gguf model.gguf --device 0 --new-tokens 128 --out native.json
//! reference-llama-cpp embed --tokens tokens.json --device 0 --out native.json
//! ```

#![deny(missing_docs)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use turbo_bench::receipt::{
    commit, machine, run_cmd, summarize, today, BundleId, Device, EmbedCell, GenerateCell, PerRun, ProviderId, Receipt,
    TokenDump,
};

#[derive(Parser)]
#[command(name = "reference-llama-cpp", version, about = "llama.cpp alone on turbo-bench's workloads")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Args)]
struct Common {
    /// ggml device index for the GPU backend; `--cpu` runs on the CPU.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Run on the CPU (no layers offloaded).
    #[arg(long)]
    cpu: bool,
    /// Timed iterations.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
    iters: u32,
    /// Untimed warm-up iterations.
    #[arg(long, default_value_t = 5)]
    warmup: u32,
    /// Receipt file to write (JSON).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Commit to record when built from a tree without git.
    #[arg(long)]
    commit: Option<String>,
    /// Device name to record; default: the ggml backend's name for the device.
    #[arg(long)]
    device_name: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// `new_tokens` tokens from a fixed prompt.
    Generate {
        #[command(flatten)]
        common: Common,
        /// The GGUF file.
        #[arg(long)]
        gguf: PathBuf,
        /// The token dump of a `turbo-bench generate` run is not needed; the
        /// bundle identity comes from this libturbo receipt.
        #[arg(long)]
        turbo_receipt: PathBuf,
        /// New tokens per generation.
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..))]
        new_tokens: u32,
        /// Prompt text (user role).
        #[arg(long, default_value = "Write a short paragraph about the history of the bicycle.")]
        prompt: String,
    },
    /// Embeddings of the texts of a token dump.
    Embed {
        #[command(flatten)]
        common: Common,
        /// The dump `turbo-bench embed --dump-tokens` wrote (texts; ids are
        /// not used, llama.cpp tokenizes them itself as the provider does).
        #[arg(long)]
        tokens: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let (common, result) = match &cli.command {
        Cmd::Generate { common, gguf, turbo_receipt, new_tokens, prompt } => {
            (common, generate(common, gguf, turbo_receipt, *new_tokens, prompt))
        }
        Cmd::Embed { common, tokens } => (common, embed(common, tokens)),
    };
    match result {
        Ok(receipt) => {
            let json = match serde_json::to_string_pretty(&receipt) {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("error: the receipt does not serialize: {e}");
                    return ExitCode::from(2);
                }
            };
            match &common.out {
                Some(path) => {
                    if let Err(e) = std::fs::write(path, format!("{json}\n")) {
                        eprintln!("error: write {}: {e}", path.display());
                        return ExitCode::from(2);
                    }
                    eprintln!("receipt written to {}", path.display());
                }
                None => println!("{json}"),
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

struct Loaded {
    backend: LlamaBackend,
    model: LlamaModel,
    device: Device,
    driver: String,
}

fn load(common: &Common, gguf: &std::path::Path) -> Result<Loaded, String> {
    let backend = LlamaBackend::init().map_err(|e| format!("llama.cpp backend: {e}"))?;
    let mut params = LlamaModelParams::default();
    let (device, driver) = if common.cpu {
        params = params.with_n_gpu_layers(0);
        (Device { name: "CPU".into(), kind: "Cpu".into(), ordinal: 0, caps: "0x0".into(), memory_total: 0 }, String::new())
    } else {
        params = params
            .with_n_gpu_layers(u32::MAX)
            .with_devices(&[common.device])
            .map_err(|e| format!("ggml device {}: {e}", common.device))?;
        // The GPU as the driver names it, when nvidia-smi is present; the
        // libturbo receipt's name carries the same prefix.
        let smi = run_cmd(
            "nvidia-smi",
            &["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits", &format!("--id={}", common.device)],
        )
        .unwrap_or_default();
        let mut f = smi.split(',').map(str::trim);
        let name = f.next().filter(|s| !s.is_empty()).map(str::to_string);
        let driver = f.next().unwrap_or("").to_string();
        let mem: u64 = f.next().and_then(|m| m.parse().ok()).unwrap_or(0);
        let name = common.device_name.clone().or(name).ok_or("no nvidia-smi on this machine; pass --device-name")?;
        (Device { name, kind: "Gpu".into(), ordinal: common.device as u32, caps: "0x0".into(), memory_total: mem * 1024 * 1024 }, driver)
    };
    let model = LlamaModel::load_from_file(&backend, gguf, &params).map_err(|e| format!("load {}: {e}", gguf.display()))?;
    Ok(Loaded { backend, model, device, driver })
}

fn provider_id(driver: String) -> ProviderId {
    ProviderId {
        id: "llama.cpp".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        runtime_version: format!("llama.cpp through llama-cpp-2 0.1.156{}", if cfg!(feature = "cuda") { ", CUDA backend" } else { "" }),
        driver_version: driver,
    }
}

fn generate(common: &Common, gguf: &std::path::Path, turbo_receipt: &std::path::Path, new_tokens: u32, prompt: &str) -> Result<Receipt, String> {
    let turbo: Receipt = serde_json::from_str(&std::fs::read_to_string(turbo_receipt).map_err(|e| format!("{}: {e}", turbo_receipt.display()))?)
        .map_err(|e| format!("{}: {e}", turbo_receipt.display()))?;
    let bundle: BundleId = turbo.bundle.clone();
    let l = load(common, gguf)?;
    let template = l.model.chat_template(None).map_err(|e| format!("the GGUF carries no chat template: {e}"))?;
    let chat = vec![LlamaChatMessage::new("user".into(), prompt.to_string()).map_err(|e| e.to_string())?];
    let text = l.model.apply_chat_template(&template, &chat, true).map_err(|e| format!("chat template: {e}"))?;
    let prompt_tokens = l.model.str_to_token(&text, AddBos::Never).map_err(|e| format!("tokenize: {e}"))?;
    let n_ctx = (prompt_tokens.len() as u32 + new_tokens + 8).max(512);
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx.min(2048))
        .with_n_seq_max(1);
    let mut ctx = l.model.new_context(&l.backend, params).map_err(|e| format!("context: {e}"))?;
    let mut batch = LlamaBatch::new(n_ctx.min(2048) as usize, 1);
    // Greedy decoding with the end token suppressed: the same workload the
    // libturbo run measured (temperature 0, min_new_tokens = max_new_tokens).
    // Every end-of-generation token the vocabulary marks, not only EOS
    // (Qwen ends turns with <|im_end|>), the same set the provider's
    // min_new_tokens path suppresses.
    let biases: Vec<LlamaLogitBias> = (0..l.model.n_vocab())
        .map(llama_cpp_2::token::LlamaToken)
        .filter(|&t| l.model.is_eog_token(t))
        .map(|t| LlamaLogitBias::new(t, f32::NEG_INFINITY))
        .collect();
    if biases.is_empty() {
        return Err("the vocabulary marks no end-of-generation token".to_string());
    }
    let mut ttft = Vec::new();
    let mut decode_rate = Vec::new();
    let mut totals = Vec::new();
    let mut generated = Vec::new();
    for i in 0..(common.warmup as u64 + common.iters as u64) {
        let mut sampler = LlamaSampler::chain(vec![LlamaSampler::logit_bias(l.model.n_vocab(), &biases), LlamaSampler::greedy()], false);
        let t0 = Instant::now();
        ctx.clear_kv_cache();
        batch.clear();
        for (pos, &t) in prompt_tokens.iter().enumerate() {
            batch.add(t, pos as i32, &[0], pos + 1 == prompt_tokens.len()).map_err(|e| format!("batch add: {e}"))?;
        }
        ctx.decode(&mut batch).map_err(|e| format!("prefill decode: {e}"))?;
        let mut n_past = prompt_tokens.len() as i32;
        let mut first: Option<Duration> = None;
        let mut n = 0u64;
        while n < new_tokens as u64 {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);
            n += 1;
            if first.is_none() {
                first = Some(t0.elapsed());
            }
            if n == new_tokens as u64 {
                break;
            }
            batch.clear();
            batch.add(token, n_past, &[0], true).map_err(|e| format!("batch add: {e}"))?;
            n_past += 1;
            ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
        }
        let total = t0.elapsed();
        if i < common.warmup as u64 {
            continue;
        }
        let f = first.ok_or("no token was generated")?;
        ttft.push(f);
        let decode = total.saturating_sub(f).as_secs_f64();
        if n > 1 && decode > 0.0 {
            decode_rate.push((n - 1) as f64 / decode);
        }
        totals.push(total);
        generated.push(n as f64);
    }
    let p50 = |v: &mut Vec<Duration>| {
        v.sort();
        v[v.len() / 2].as_secs_f64() * 1e3
    };
    decode_rate.sort_by(f64::total_cmp);
    let cell = GenerateCell {
        new_tokens_requested: new_tokens,
        generated_tokens_mean: generated.iter().sum::<f64>() / generated.len() as f64,
        prompt_tokens: prompt_tokens.len() as u32,
        time_to_first_token_ms_p50: p50(&mut ttft),
        decode_tokens_per_s_p50: if decode_rate.is_empty() { None } else { Some(decode_rate[decode_rate.len() / 2]) },
        total_ms_p50: p50(&mut totals),
        finish_reasons: vec!["Length".to_string(); common.iters as usize],
        iters: common.iters,
    };
    eprintln!(
        "generate {new_tokens} tokens: ttft p50 {:.1} ms, decode {} tok/s p50, total p50 {:.0} ms",
        cell.time_to_first_token_ms_p50,
        cell.decode_tokens_per_s_p50.map(|r| format!("{r:.1}")).unwrap_or_else(|| "n/a".into()),
        cell.total_ms_p50
    );
    Ok(Receipt {
        receipt_version: 1,
        kind: "native".to_string(),
        date: today()?,
        machine: machine()?,
        commit: commit(common.commit.as_deref())?,
        provider: provider_id(l.driver.clone()),
        device: l.device.clone(),
        bundle,
        embed: Vec::new(),
        rerank: None,
        generate: Some(cell),
        budget_check: None,
        native_reference: format!(
            "this is the native side: llama.cpp through llama-cpp-2 with the GGUF's chat template, greedy decoding with the end token suppressed, on the prompt of {}; bundle identity copied from that receipt",
            turbo_receipt.display()
        ),
    })
}

fn embed(common: &Common, tokens: &std::path::Path) -> Result<Receipt, String> {
    let dump: TokenDump = serde_json::from_str(&std::fs::read_to_string(tokens).map_err(|e| format!("{}: {e}", tokens.display()))?)
        .map_err(|e| format!("{}: {e}", tokens.display()))?;
    let gguf = dump.artifacts.get("gguf").ok_or("the token dump names no `gguf` artifact")?;
    let pooling = match dump.pooling.as_str() {
        "mean" => LlamaPoolingType::Mean,
        "cls" => LlamaPoolingType::Cls,
        "last" => LlamaPoolingType::Last,
        other => return Err(format!("pooling `{other}` is not mean, cls or last")),
    };
    let normalize = match dump.normalize.as_str() {
        "l2" => true,
        "none" | "" => false,
        other => return Err(format!("normalize `{other}` is not l2 or none")),
    };
    let l = load(common, std::path::Path::new(gguf))?;
    let mut cells = Vec::new();
    for cell in &dump.cells {
        let (b, s) = (cell.batch as usize, cell.seq as usize);
        let rows: Vec<Vec<llama_cpp_2::token::LlamaToken>> = cell
            .texts
            .iter()
            .map(|t| l.model.str_to_token(t, AddBos::Always).map_err(|e| format!("tokenize: {e}")))
            .collect::<Result<_, _>>()?;
        let rows: Vec<Vec<_>> = rows.into_iter().map(|mut r| {
            r.truncate(s);
            r
        }).collect();
        let n_tokens: usize = rows.iter().map(Vec::len).sum();
        let n_ctx = (b * s) as u32;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_ubatch(n_ctx)
            .with_n_seq_max(b as u32)
            .with_embeddings(true)
            .with_pooling_type(pooling);
        let mut ctx = l.model.new_context(&l.backend, params).map_err(|e| format!("context {b}x{s}: {e}"))?;
        let mut batch = LlamaBatch::new(n_tokens.max(1), b as i32);
        let mut out = vec![0f32; b * l.model.n_embd() as usize];
        let mut run_once = |ctx: &mut llama_cpp_2::context::LlamaContext<'_>| -> Result<(), String> {
            ctx.clear_kv_cache();
            batch.clear();
            for (r, row) in rows.iter().enumerate() {
                for (pos, &t) in row.iter().enumerate() {
                    batch.add(t, pos as i32, &[r as i32], true).map_err(|e| format!("batch add: {e}"))?;
                }
            }
            ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
            let h = l.model.n_embd() as usize;
            for r in 0..b {
                let v = ctx.embeddings_seq_ith(r as i32).map_err(|e| format!("embeddings row {r}: {e}"))?;
                let row = &mut out[r * h..(r + 1) * h];
                row.copy_from_slice(v);
                if normalize {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for o in row.iter_mut() {
                            *o /= norm;
                        }
                    }
                }
            }
            Ok(())
        };
        for _ in 0..common.warmup {
            run_once(&mut ctx)?;
        }
        let mut samples = Vec::with_capacity(common.iters as usize);
        for _ in 0..common.iters {
            let t0 = Instant::now();
            run_once(&mut ctx)?;
            samples.push(t0.elapsed());
        }
        let lat = summarize(&mut samples, b as u64, Some(n_tokens as u64))?;
        eprintln!("embed batch {b:>2} seq {s:>3}: text p50 {:.3} ms ({:.0} rows/s)", lat.p50_ms, lat.rows_per_s);
        cells.push(EmbedCell {
            batch: cell.batch,
            seq: cell.seq,
            live_tokens_per_row: n_tokens as f64 / b as f64,
            token_count_source: "tokenizer".to_string(),
            text_path: lat,
            prepared_tokens_path: None,
            prepared_tokens_note: "llama.cpp tokenizes inside the timed path, as the ggml provider does".to_string(),
            per_run: PerRun { h2d_bytes: None, d2h_bytes: None, host_allocs: None, provider_allocs: None },
        });
    }
    Ok(Receipt {
        receipt_version: 1,
        kind: "native".to_string(),
        date: today()?,
        machine: machine()?,
        commit: commit(common.commit.as_deref())?,
        provider: provider_id(l.driver.clone()),
        device: l.device.clone(),
        bundle: dump.bundle_id.clone(),
        embed: cells,
        rerank: None,
        generate: None,
        budget_check: None,
        native_reference: format!(
            "this is the native side: llama.cpp through llama-cpp-2 tokenizing and pooling the texts of {}; bundle identity copied from that dump",
            tokens.display()
        ),
    })
}
