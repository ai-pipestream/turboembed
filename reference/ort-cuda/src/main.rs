//! The direct-native half of the cuda provider's matched benchmark
//! (PLAN.md section 11): ONNX Runtime's CUDA execution provider called
//! through the `ort` crate alone, with no libturbo in the timed path.
//!
//! It runs the token rows `turbo-bench embed --dump-tokens` wrote (the same
//! ids, masks and shapes libturbo's prepared-token path ran), pools and
//! normalizes on the host as a plain ONNX Runtime user would, and writes a
//! receipt of kind `native` that `turbo-bench compare` reads.
//!
//! ```text
//! LD_LIBRARY_PATH=.libs/nvidia/lib reference-ort-cuda --tokens tokens.json \
//!     --device 0 --iters 30 --warmup 5 --out native.json [--commit <sha>]
//! ```

#![deny(missing_docs)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::Parser;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use turbo_bench::receipt::{
    commit, machine, run_cmd, summarize, today, warm_up, Device, EmbedCell, PerRun, ProviderId, Receipt, TokenDump,
};

#[derive(Parser)]
#[command(name = "reference-ort-cuda", version, about = "ONNX Runtime CUDA alone on turbo-bench's token rows")]
struct Cli {
    /// The token dump `turbo-bench embed --dump-tokens` wrote.
    #[arg(long)]
    tokens: PathBuf,
    /// CUDA device id.
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// Timed iterations per cell.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
    iters: u32,
    /// Untimed warm-up iterations per cell.
    #[arg(long, default_value_t = 5)]
    warmup: u32,
    /// Receipt file to write (JSON).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Commit to record when built from a tree without git.
    #[arg(long)]
    commit: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(receipt) => {
            let json = match serde_json::to_string_pretty(&receipt) {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("error: the receipt does not serialize: {e}");
                    return ExitCode::from(2);
                }
            };
            match &cli.out {
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

fn run(cli: &Cli) -> Result<Receipt, String> {
    let text = std::fs::read_to_string(&cli.tokens).map_err(|e| format!("{}: {e}", cli.tokens.display()))?;
    let dump: TokenDump = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", cli.tokens.display()))?;
    let onnx = dump.artifacts.get("onnx").ok_or_else(|| {
        format!("the token dump names no `onnx` artifact; artifacts: {:?}", dump.artifacts.keys().collect::<Vec<_>>())
    })?;
    let pooling = dump.pooling.as_str();
    if pooling != "mean" && pooling != "cls" {
        return Err(format!("pooling `{pooling}` is not implemented by this reference (mean, cls)"));
    }
    let normalize = match dump.normalize.as_str() {
        "l2" => true,
        "none" | "" => false,
        other => return Err(format!("normalize `{other}` is not l2 or none")),
    };

    // The GPU as the driver names it, so the receipt's device matches the
    // libturbo receipt's before its "(sm_xy)" suffix.
    let smi = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
            &format!("--id={}", cli.device),
        ],
    )?;
    let mut fields = smi.split(',').map(str::trim);
    let gpu_name = fields.next().unwrap_or("").to_string();
    let driver = fields.next().unwrap_or("").to_string();
    // An integrated GPU (a Jetson, where the GPU shares system memory)
    // reports `[N/A]` for memory.total. Record 0 for "the driver does not
    // report it" rather than failing or inventing a number.
    let memory_field = fields.next().unwrap_or("0");
    let memory_mib: u64 = match memory_field {
        "[N/A]" | "N/A" | "" => 0,
        v => v.parse().map_err(|e| format!("nvidia-smi memory.total `{smi}`: {e}"))?,
    };

    let ep = ort::ep::CUDA::default().with_device_id(cli.device);
    let mut session = Session::builder()
        .map_err(|e| format!("session builder: {e}"))?
        .with_execution_providers([ep.build().error_on_failure()])
        .map_err(|e| format!("CUDA execution provider: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("optimization level: {e}"))?
        .commit_from_file(onnx)
        .map_err(|e| format!("load {onnx}: {e}"))?;
    let input_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    for needed in ["input_ids", "attention_mask"] {
        if !input_names.iter().any(|n| n == needed) {
            return Err(format!("model has no `{needed}` input; inputs: {input_names:?}"));
        }
    }
    let wants_types = input_names.iter().any(|n| n == "token_type_ids");
    let output_name = session.outputs().first().map(|o| o.name().to_string()).ok_or("model has no outputs")?;

    let mut cells = Vec::new();
    for cell in &dump.cells {
        let (b, s) = (cell.batch as usize, cell.seq as usize);
        if cell.ids.len() != b * s || cell.mask.len() != b * s {
            return Err(format!("cell {b}x{s}: ids/mask hold {} and {} elements", cell.ids.len(), cell.mask.len()));
        }
        let ids: Vec<i64> = cell.ids.iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = cell.mask.iter().map(|&m| m as i64).collect();
        let types: Vec<i64> = vec![0; b * s];
        let live: u64 = cell.lengths.iter().map(|&l| l as u64).sum();
        let mut out = Vec::new();
        let mut run_once = |session: &mut Session| -> Result<(), String> {
            // A plain ONNX Runtime user's loop: build the input tensors, run,
            // read the hidden state back, pool and normalize on the host.
            let shape = vec![b as i64, s as i64];
            let t_ids = Tensor::<i64>::from_array((shape.clone(), ids.clone())).map_err(|e| e.to_string())?;
            let t_mask = Tensor::<i64>::from_array((shape.clone(), mask.clone())).map_err(|e| e.to_string())?;
            let outputs = if wants_types {
                let t_types = Tensor::<i64>::from_array((shape.clone(), types.clone())).map_err(|e| e.to_string())?;
                session
                    .run(ort::inputs!["input_ids" => t_ids, "attention_mask" => t_mask, "token_type_ids" => t_types])
                    .map_err(|e| format!("run: {e}"))?
            } else {
                session
                    .run(ort::inputs!["input_ids" => t_ids, "attention_mask" => t_mask])
                    .map_err(|e| format!("run: {e}"))?
            };
            let (oshape, hidden) =
                outputs[output_name.as_str()].try_extract_tensor::<f32>().map_err(|e| format!("output: {e}"))?;
            let dims: Vec<i64> = oshape.iter().copied().collect();
            if dims.len() != 3 || dims[0] as usize != b || dims[1] as usize != s {
                return Err(format!("output shape {dims:?} is not [{b}, {s}, hidden]"));
            }
            let h = dims[2] as usize;
            out.clear();
            out.resize(b * h, 0.0);
            for r in 0..b {
                let row = &mut out[r * h..(r + 1) * h];
                if pooling == "cls" {
                    row.copy_from_slice(&hidden[r * s * h..r * s * h + h]);
                } else {
                    let mut n = 0.0f32;
                    for t in 0..s {
                        if cell.mask[r * s + t] != 0 {
                            n += 1.0;
                            let src = &hidden[(r * s + t) * h..(r * s + t + 1) * h];
                            for (o, v) in row.iter_mut().zip(src) {
                                *o += v;
                            }
                        }
                    }
                    if n > 0.0 {
                        for o in row.iter_mut() {
                            *o /= n;
                        }
                    }
                }
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
        warm_up(cli.warmup, || run_once(&mut session))?;
        let mut samples: Vec<Duration> = Vec::with_capacity(cli.iters as usize);
        for _ in 0..cli.iters {
            let t0 = Instant::now();
            run_once(&mut session)?;
            samples.push(t0.elapsed());
        }
        let prepared = summarize(&mut samples, b as u64, Some(live))?;
        eprintln!(
            "embed batch {b:>2} seq {s:>3}: tokens p50 {:.3} ms ({:.0} rows/s, {:.0} tok/s)",
            prepared.p50_ms,
            prepared.rows_per_s,
            prepared.tokens_per_s.unwrap_or(0.0)
        );
        cells.push(EmbedCell {
            batch: cell.batch,
            seq: cell.seq,
            live_tokens_per_row: live as f64 / b as f64,
            token_count_source: "tokenizer".to_string(),
            // The reference has no tokenizer in its timed path; the text
            // path is the prepared-token path plus nothing, so the same
            // figure stands for both and compare uses the prepared one.
            text_path: prepared.clone(),
            prepared_tokens_path: Some(prepared),
            prepared_tokens_note: String::new(),
            per_run: PerRun { h2d_bytes: None, d2h_bytes: None, host_allocs: None, provider_allocs: None },
        });
    }
    Ok(Receipt {
        receipt_version: 1,
        kind: "native".to_string(),
        date: today()?,
        machine: machine()?,
        commit: commit(cli.commit.as_deref())?,
        provider: ProviderId {
            id: "onnxruntime-cuda".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_version: format!("ONNX Runtime {} (ort {}), CUDA execution provider", ort::info(), "2.0.0-rc.13"),
            driver_version: driver,
        },
        device: Device { name: gpu_name, kind: "Gpu".to_string(), ordinal: cli.device as u32, caps: "0x0".to_string(), memory_total: memory_mib * 1024 * 1024 },
        bundle: dump.bundle_id.clone(),
        embed: cells,
        rerank: None,
        generate: None,
        budget_check: None,
        native_reference: format!(
            "this is the native side: ONNX Runtime CUDA through the ort crate on the token rows of {}; bundle identity copied from that dump; host-side {} pooling and {} normalization; no session counters",
            cli.tokens.display(),
            pooling,
            if normalize { "l2" } else { "no" }
        ),
    })
}
