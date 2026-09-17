//! NVIDIA matched native-overhead pilot: direct ONNX Runtime CUDA vs the
//! TurboEmbed C ABI (through the safe Rust wrapper), following the
//! methodology of `docs/library-design.md#native-performance-acceptance`
//! and the Intel pilot (`docs/intel-prepared-performance-2026-09-14.md`).
//!
//! Both paths run the identical MiniLM ONNX checkpoint, tokenizer revision,
//! f32 precision, mean+L2 postprocessing, fixed `[batch, max_seq]` execution
//! shape, and synchronous completion on CUDA device 0. The direct reference
//! is a raw `ort` consumer: reusable host token buffers, IoBinding, a
//! preallocated host hidden-state output, and host mean+L2 pooling. The
//! stock checkpoint exposes only `last_hidden_state`, so a direct consumer
//! reads the full hidden state back; TurboEmbed instead pools on device and
//! reads back `[batch, dim]`. That difference is a library capability and is
//! recorded, not hidden.
//!
//! Gates per repeat (predeclared, same as the Intel pilot): ABI p50 within
//! 5% of the direct baseline and ABI throughput at least 95% of it. Parity
//! per case: max abs error <= 5e-4 and RMSE <= 1e-4 between the two paths.
//! p99 is descriptive only below 1000 samples.

use std::ffi::c_void;
use std::process::ExitCode;
use std::ptr::{self, NonNull};
use std::time::{Duration, Instant};

use bench_turbo::{
    git_head, hostname, nearest_rank_us, nvidia_gpu_name, workspace_root, write_json,
};
use clap::Parser;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{PrimitiveTensorElementType, Shape, Tensor};
use ort::{ortsys, AsPointer};
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use turboembed::{Device, EmbedOptions, Engine, Pooling};

const ALIAS: &str = "minilm";
const MAX_SEQ: usize = 256;
const PARITY_MAX_ABS: f32 = 5e-4;
const PARITY_MAX_RMSE: f64 = 1e-4;
const P50_OVERHEAD_LIMIT: f64 = 1.05;
const THROUGHPUT_FLOOR: f64 = 0.95;
const P99_MIN_SAMPLES: usize = 1000;

unsafe extern "C" {
    fn turboembed_ort_hot_path_reset();
    fn turboembed_ort_arena_allocs() -> u64;
    fn turboembed_ort_external_allocs() -> u64;
    fn turboembed_ort_d2h_bytes() -> u64;
    fn turboembed_ort_cuda_forward_allocs() -> u64;
    fn turboembed_ort_cuda_forward_h2d_bytes() -> u64;
}

#[derive(Parser, Debug)]
#[command(
    name = "bench-nvidia-overhead",
    about = "Matched direct-ORT-CUDA vs TurboEmbed-ABI MiniLM overhead pilot"
)]
struct Args {
    /// Untimed per-path executions before each case's parity and timing.
    #[arg(long, default_value_t = 20)]
    warmup: u32,
    /// Timed repeats per case per path (order alternates).
    #[arg(long, default_value_t = 3)]
    repeats: u32,
    /// Per-repeat wall-clock cap in seconds.
    #[arg(long, default_value_t = 10)]
    max_seconds: u64,
    /// Per-repeat completed-request cap.
    #[arg(long, default_value_t = 10_000)]
    max_requests: usize,
    /// Reduced case grid for smoke runs (never a receipt).
    #[arg(long, default_value_t = false)]
    quick: bool,
    /// Skip the two-engine concurrent observation.
    #[arg(long, default_value_t = false)]
    skip_concurrent: bool,
    #[arg(long)]
    out: std::path::PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("bench-nvidia-overhead: {e}");
            ExitCode::from(1)
        }
    }
}

fn nvidia_driver_version() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Minimal reader for `[models.minilm.nvidia]`; the ABI engine resolves the
/// same entry through the built-in catalog, so both paths share one pin.
struct CatalogNvidiaMinilm {
    model_path: String,
    tokenizer_json: String,
}

fn read_catalog(root: &std::path::Path) -> Result<CatalogNvidiaMinilm, String> {
    let raw = std::fs::read_to_string(root.join("config/catalog.toml"))
        .map_err(|e| format!("config/catalog.toml: {e}"))?;
    let value: toml::Value = toml::from_str(&raw).map_err(|e| format!("catalog parse: {e}"))?;
    let entry = value
        .get("models")
        .and_then(|m| m.get(ALIAS))
        .and_then(|m| m.get("nvidia"))
        .ok_or("catalog has no [models.minilm.nvidia]")?;
    let model_path = entry
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or("catalog nvidia entry missing path")?
        .to_string();
    let tokenizer_dir = entry
        .get("tokenizer_dir")
        .and_then(|v| v.as_str())
        .ok_or("catalog nvidia entry missing tokenizer_dir")?;
    let max_seq = entry
        .get("max_seq_len")
        .and_then(|v| v.as_integer())
        .ok_or("catalog nvidia entry missing max_seq_len")?;
    if max_seq as usize != MAX_SEQ {
        return Err(format!(
            "catalog max_seq_len {max_seq} != pilot MAX_SEQ {MAX_SEQ}; keep the shapes matched"
        ));
    }
    let tokenizer_json = std::path::Path::new(tokenizer_dir).join("tokenizer.json");
    if !tokenizer_json.is_file() {
        return Err(format!("{} is not a file", tokenizer_json.display()));
    }
    Ok(CatalogNvidiaMinilm {
        model_path,
        tokenizer_json: tokenizer_json.to_string_lossy().into_owned(),
    })
}

fn ort_ok<T>(r: ort::Result<T>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

/// Same rationale as the turboembed provider: `TensorRefMut::from_raw`
/// rebuilds a CPU MemoryInfo in ort 2.0.0-rc.13, so create the OrtValue
/// directly over caller-owned memory.
fn tensor_from_data<T: PrimitiveTensorElementType + std::fmt::Debug>(
    info: &MemoryInfo<'_>,
    data: *mut c_void,
    shape: Shape,
) -> Result<Tensor<T>, String> {
    let mut value_ptr: *mut ort::sys::OrtValue = ptr::null_mut();
    let nbytes = shape.num_elements() * std::mem::size_of::<T>();
    ort_ok((|| {
        ortsys![
            unsafe CreateTensorWithDataAsOrtValue(
                info.ptr(),
                data,
                nbytes,
                shape.as_ptr(),
                shape.len(),
                T::into_tensor_element_type().into(),
                &mut value_ptr
            )?;
            nonNull(value_ptr)
        ];
        Ok(())
    })())
    .map_err(|e| format!("CreateTensorWithDataAsOrtValue: {e}"))?;
    let nn = NonNull::new(value_ptr)
        .ok_or_else(|| "CreateTensorWithDataAsOrtValue returned null".to_string())?;
    Ok(unsafe { Tensor::<T>::from_ptr(nn, None) })
}

fn mean_pool_l2(
    hidden: &[f32],
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    for b in 0..batch {
        let mut count = 0f32;
        for s in 0..seq {
            if mask[b * seq + s] == 0 {
                continue;
            }
            count += 1.0;
            let row = &hidden[(b * seq + s) * dim..(b * seq + s + 1) * dim];
            let acc = &mut out[b * dim..(b + 1) * dim];
            for (a, v) in acc.iter_mut().zip(row) {
                *a += v;
            }
        }
        let acc = &mut out[b * dim..(b + 1) * dim];
        if count > 0.0 {
            for a in acc.iter_mut() {
                *a /= count;
            }
        }
        let norm: f64 = acc.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let norm = norm.sqrt() as f32;
        if norm > 0.0 {
            for a in acc.iter_mut() {
                *a /= norm;
            }
        }
    }
}

/// Direct ONNX Runtime CUDA reference: stock `ort` consumer with reusable
/// buffers and host mean+L2 pooling. No TurboEmbed API on this path.
struct DirectOrt {
    session: Session,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    output_name: String,
    hidden_dim: usize,
    ids: Vec<i64>,
    mask: Vec<i64>,
    types: Vec<i64>,
    hidden: Vec<f32>,
    pooled: Vec<f32>,
    max_batch: usize,
}

impl DirectOrt {
    fn load(catalog: &CatalogNvidiaMinilm, max_batch: usize) -> Result<Self, String> {
        let mut tokenizer =
            Tokenizer::from_file(&catalog.tokenizer_json).map_err(|e| format!("tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ,
                ..Default::default()
            }))
            .map_err(|e| format!("truncation: {e}"))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(MAX_SEQ),
            ..Default::default()
        }));

        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort opt level: {e}"))?
            .with_memory_pattern(true)
            .map_err(|e| format!("ort memory pattern: {e}"))?
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| format!("CUDA execution provider unavailable (fail loud): {e}"))?
            .commit_from_file(&catalog.model_path)
            .map_err(|e| format!("direct ORT load on CUDA EP: {e}"))?;

        const KNOWN: [&str; 3] = ["input_ids", "attention_mask", "token_type_ids"];
        let mut input_names = Vec::new();
        for input in session.inputs() {
            if KNOWN.contains(&input.name()) {
                input_names.push(input.name().to_string());
            } else {
                return Err(format!("unsupported model input {:?}", input.name()));
            }
        }
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or("model has no outputs")?;
        let hidden_dim = 384usize;

        let n = max_batch * MAX_SEQ;
        Ok(Self {
            session,
            tokenizer,
            input_names,
            output_name,
            hidden_dim,
            ids: vec![0i64; n],
            mask: vec![0i64; n],
            types: vec![0i64; n],
            hidden: vec![0f32; n * hidden_dim],
            pooled: vec![0f32; max_batch * hidden_dim],
            max_batch,
        })
    }

    fn tokenize(&mut self, texts: &[String]) -> Result<(), String> {
        let batch = texts.len();
        if batch > self.max_batch {
            return Err(format!("batch {batch} > max_batch {}", self.max_batch));
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let n = batch * MAX_SEQ;
        self.ids[..n].fill(0);
        self.mask[..n].fill(0);
        self.types[..n].fill(0);
        for (b, encoding) in encodings.iter().enumerate() {
            if encoding.len() > MAX_SEQ {
                return Err(format!("row {b} tokenized past MAX_SEQ"));
            }
            let row = b * MAX_SEQ;
            for (i, v) in encoding.get_ids().iter().enumerate() {
                self.ids[row + i] = i64::from(*v);
            }
            for (i, v) in encoding.get_attention_mask().iter().enumerate() {
                self.mask[row + i] = i64::from(*v);
            }
            for (i, v) in encoding.get_type_ids().iter().enumerate() {
                self.types[row + i] = i64::from(*v);
            }
        }
        Ok(())
    }

    /// One synchronous text-to-pooled-result request: tokenize, bind the
    /// reusable buffers, run on the CUDA EP, read back `[batch, seq, dim]`,
    /// mean+L2 pool on host. Returns `[batch, dim]` in `self.pooled`.
    fn embed(&mut self, texts: &[String]) -> Result<(), String> {
        self.tokenize(texts)?;
        let batch = texts.len();
        let shape = [batch as i64, MAX_SEQ as i64];

        let cpu = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| format!("CPU MemoryInfo: {e}"))?;

        let mut binding = self
            .session
            .create_binding()
            .map_err(|e| format!("IoBinding create: {e}"))?;
        let mut held: Vec<Tensor<i64>> = Vec::new();
        for name in &self.input_names {
            let host: *mut c_void = match name.as_str() {
                "input_ids" => self.ids.as_mut_ptr().cast(),
                "attention_mask" => self.mask.as_mut_ptr().cast(),
                "token_type_ids" => self.types.as_mut_ptr().cast(),
                _ => unreachable!("validated at load"),
            };
            let tensor = tensor_from_data::<i64>(&cpu, host, Shape::new(shape))?;
            binding
                .bind_input(name, &tensor)
                .map_err(|e| format!("bind_input {name}: {e}"))?;
            held.push(tensor);
        }
        let hidden_shape = Shape::new([batch as i64, MAX_SEQ as i64, self.hidden_dim as i64]);
        let hidden_tensor =
            tensor_from_data::<f32>(&cpu, self.hidden.as_mut_ptr().cast(), hidden_shape)?;
        binding
            .bind_output(self.output_name.as_str(), hidden_tensor)
            .map_err(|e| format!("bind_output: {e}"))?;

        ort_ok((|| {
            ortsys![unsafe RunWithBinding(self.session.ptr_mut(), ptr::null(), binding.ptr())?];
            Ok(())
        })())
        .map_err(|e| format!("direct CUDA run: {e}"))?;
        binding
            .synchronize_outputs()
            .map_err(|e| format!("synchronize_outputs: {e}"))?;
        drop(held);

        let n = batch * MAX_SEQ;
        mean_pool_l2(
            &self.hidden[..n * self.hidden_dim],
            &self.mask[..n],
            batch,
            MAX_SEQ,
            self.hidden_dim,
            &mut self.pooled[..batch * self.hidden_dim],
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    target_tokens: usize,
    mixed: bool,
}

impl Case {
    fn name(&self) -> String {
        format!(
            "b{}_t{}_{}",
            self.batch,
            self.target_tokens,
            if self.mixed { "mixed" } else { "full" }
        )
    }
}

fn case_grid(quick: bool) -> Vec<Case> {
    let batches: &[usize] = if quick { &[1, 8] } else { &[1, 8, 32] };
    let targets: &[usize] = if quick { &[32] } else { &[32, 128, 256] };
    let mut cases = Vec::new();
    for &batch in batches {
        for &target in targets {
            for mixed in [false, true] {
                cases.push(Case {
                    batch,
                    target_tokens: target,
                    mixed,
                });
            }
        }
    }
    cases
}

/// Deterministic texts. Every "the" is one WordPiece token, so a row
/// targeting `t` tokens is `t - 2` words plus [CLS]/[SEP].
fn case_texts(case: &Case) -> Vec<String> {
    let row_tokens = |target: usize, index: usize, batch: usize, mixed: bool| -> usize {
        if !mixed {
            return target;
        }
        let lo = 16usize.min(target);
        if batch <= 1 {
            return (lo + target) / 2;
        }
        lo + ((target - lo) * index) / (batch - 1)
    };
    (0..case.batch)
        .map(|i| {
            let tokens = row_tokens(case.target_tokens, i, case.batch, case.mixed).max(3);
            let words = tokens - 2;
            let mut text = String::with_capacity(words * 4);
            for w in 0..words {
                if w > 0 {
                    text.push(' ');
                }
                text.push_str("the");
            }
            text
        })
        .collect()
}

fn verify_row_tokens(tokenizer_json: &str, texts: &[String]) -> Result<Vec<usize>, String> {
    let mut probe =
        Tokenizer::from_file(tokenizer_json).map_err(|e| format!("probe tokenizer: {e}"))?;
    probe
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ,
            ..Default::default()
        }))
        .map_err(|e| format!("probe truncation: {e}"))?;
    // tokenizer.json ships an embedded padding config; the probe measures
    // real row lengths, so disable it.
    probe.with_padding(None);
    let mut lens = Vec::new();
    for text in texts {
        let enc = probe
            .encode(text.as_str(), true)
            .map_err(|e| format!("probe encode: {e}"))?;
        lens.push(enc.len());
    }
    Ok(lens)
}

struct RepeatStats {
    n: usize,
    seconds: f64,
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    min_us: u64,
    max_us: u64,
    rps: f64,
}

fn timed_repeat(
    mut call: impl FnMut() -> Result<(), String>,
    max_seconds: u64,
    max_requests: usize,
) -> Result<RepeatStats, String> {
    let mut samples: Vec<u64> = Vec::with_capacity(max_requests.min(16_384));
    let start = Instant::now();
    let deadline = start + Duration::from_secs(max_seconds);
    while samples.len() < max_requests && Instant::now() < deadline {
        let t0 = Instant::now();
        call()?;
        samples.push(bench_turbo::duration_us(t0.elapsed()));
    }
    let seconds = start.elapsed().as_secs_f64();
    let n = samples.len();
    if n == 0 {
        return Err("timed repeat produced no samples".into());
    }
    samples.sort_unstable();
    Ok(RepeatStats {
        n,
        seconds,
        p50_us: nearest_rank_us(&samples, 50.0),
        p90_us: nearest_rank_us(&samples, 90.0),
        p99_us: nearest_rank_us(&samples, 99.0),
        min_us: samples[0],
        max_us: samples[n - 1],
        rps: n as f64 / seconds,
    })
}

fn stats_json(s: &RepeatStats) -> Value {
    json!({
        "n": s.n,
        "seconds": s.seconds,
        "p50_us": s.p50_us,
        "p90_us": s.p90_us,
        "p99_us": s.p99_us,
        "p99_sufficient": s.n >= P99_MIN_SAMPLES,
        "min_us": s.min_us,
        "max_us": s.max_us,
        "requests_per_second": s.rps,
    })
}

fn abi_counters() -> Value {
    unsafe {
        json!({
            "arena_allocs": turboembed_ort_arena_allocs(),
            "ort_gpu_external_allocs": turboembed_ort_external_allocs(),
            "cuda_forward_allocs": turboembed_ort_cuda_forward_allocs(),
            "token_h2d_bytes": turboembed_ort_cuda_forward_h2d_bytes(),
            "hidden_d2h_bytes": turboembed_ort_d2h_bytes(),
        })
    }
}

fn abi_counters_zero() -> bool {
    unsafe {
        turboembed_ort_arena_allocs() == 0
            && turboembed_ort_external_allocs() == 0
            && turboembed_ort_cuda_forward_allocs() == 0
            && turboembed_ort_cuda_forward_h2d_bytes() == 0
            && turboembed_ort_d2h_bytes() == 0
    }
}

fn run() -> Result<bool, String> {
    let args = Args::parse();
    let root = workspace_root();
    let gpu = nvidia_gpu_name()?;
    let driver = nvidia_driver_version();
    let catalog = read_catalog(&root)?;
    let cases = case_grid(args.quick);
    let max_batch = cases.iter().map(|c| c.batch).max().unwrap_or(1);

    eprintln!("direct ORT load: {}", catalog.model_path);
    let mut direct = DirectOrt::load(&catalog, max_batch)?;

    eprintln!("ABI engine load: alias {ALIAS} on CUDA");
    let engine = Engine::create(Device::Cuda).map_err(|e| format!("engine create: {e}"))?;
    engine
        .load_model(ALIAS)
        .map_err(|e| format!("load_model({ALIAS}): {e}"))?;
    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    let dim = {
        let probe = engine
            .embed_one(ALIAS, "the", &opts)
            .map_err(|e| format!("ABI probe: {e}"))?;
        probe.dim()
    };
    if dim != direct.hidden_dim {
        return Err(format!(
            "ABI dim {dim} != direct hidden dim {}",
            direct.hidden_dim
        ));
    }

    let mut case_reports = Vec::new();
    let mut all_pass = true;

    for (case_index, case) in cases.iter().enumerate() {
        let texts = case_texts(case);
        let views: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
        let row_tokens = verify_row_tokens(&catalog.tokenizer_json, &texts)?;
        if !case.mixed && row_tokens.iter().any(|&l| l != case.target_tokens) {
            return Err(format!(
                "{}: constructed rows are {row_tokens:?}, expected {}",
                case.name(),
                case.target_tokens
            ));
        }

        // Warmup both paths, then check parity before any timing.
        for _ in 0..args.warmup {
            direct.embed(&texts)?;
            let out = engine
                .embed(ALIAS, &views, &opts)
                .map_err(|e| format!("{} ABI warmup: {e}", case.name()))?;
            drop(out);
        }
        direct.embed(&texts)?;
        let abi_out = engine
            .embed(ALIAS, &views, &opts)
            .map_err(|e| format!("{} ABI parity embed: {e}", case.name()))?;
        let abi_values = abi_out.values();
        let need = case.batch * dim;
        if abi_values.len() != need {
            return Err(format!(
                "{}: ABI returned {} values, expected {need}",
                case.name(),
                abi_values.len()
            ));
        }
        let mut max_abs = 0f32;
        let mut sq_sum = 0f64;
        for (a, b) in abi_values.iter().zip(&direct.pooled[..need]) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sq_sum += f64::from(d) * f64::from(d);
        }
        let rmse = (sq_sum / need as f64).sqrt();
        drop(abi_out);
        let parity_ok = max_abs <= PARITY_MAX_ABS && rmse <= PARITY_MAX_RMSE;
        if !parity_ok {
            all_pass = false;
        }

        // Steady-state ABI counters on one post-warmup request.
        unsafe { turboembed_ort_hot_path_reset() };
        let probe = engine
            .embed(ALIAS, &views, &opts)
            .map_err(|e| format!("{} ABI counter probe: {e}", case.name()))?;
        drop(probe);
        let counters = abi_counters();
        let counters_zero = abi_counters_zero();
        if !counters_zero {
            all_pass = false;
        }

        let mut repeats = Vec::new();
        for repeat in 0..args.repeats {
            let abi_first = (case_index + repeat as usize) % 2 == 1;
            let mut direct_stats: Option<RepeatStats> = None;
            let mut abi_stats: Option<RepeatStats> = None;
            for leg in 0..2 {
                let run_abi = (leg == 0) == abi_first;
                if run_abi {
                    abi_stats = Some(timed_repeat(
                        || {
                            let out = engine
                                .embed(ALIAS, &views, &opts)
                                .map_err(|e| format!("{} ABI timed: {e}", case.name()))?;
                            drop(out);
                            Ok(())
                        },
                        args.max_seconds,
                        args.max_requests,
                    )?);
                } else {
                    direct_stats = Some(timed_repeat(
                        || direct.embed(&texts),
                        args.max_seconds,
                        args.max_requests,
                    )?);
                }
            }
            let direct_stats = direct_stats.expect("direct leg ran");
            let abi_stats = abi_stats.expect("abi leg ran");
            let p50_ratio = abi_stats.p50_us as f64 / direct_stats.p50_us as f64;
            let throughput_ratio = abi_stats.rps / direct_stats.rps;
            let pass = p50_ratio <= P50_OVERHEAD_LIMIT && throughput_ratio >= THROUGHPUT_FLOOR;
            if !pass {
                all_pass = false;
            }
            repeats.push(json!({
                "order": if abi_first { "abi_first" } else { "direct_first" },
                "direct": stats_json(&direct_stats),
                "abi": stats_json(&abi_stats),
                "abi_p50_over_direct_p50": p50_ratio,
                "abi_throughput_over_direct": throughput_ratio,
                "pass": pass,
            }));
            eprintln!(
                "{} repeat {}: direct p50={}us abi p50={}us ratio={:.4} rps {:.1}/{:.1} pass={}",
                case.name(),
                repeat,
                direct_stats.p50_us,
                abi_stats.p50_us,
                p50_ratio,
                direct_stats.rps,
                abi_stats.rps,
                pass
            );
        }

        case_reports.push(json!({
            "case": case.name(),
            "batch": case.batch,
            "target_tokens": case.target_tokens,
            "mixed": case.mixed,
            "row_tokens": row_tokens,
            "execution_shape": [case.batch, MAX_SEQ],
            "parity": {
                "max_abs": max_abs,
                "rmse": rmse,
                "max_abs_gate": PARITY_MAX_ABS,
                "rmse_gate": PARITY_MAX_RMSE,
                "pass": parity_ok,
            },
            "abi_steady_state_counters": counters,
            "abi_counters_zero": counters_zero,
            "repeats": repeats,
        }));
    }

    // Two-engine concurrent observation (allocator isolation under load).
    let concurrent = if args.skip_concurrent {
        Value::Null
    } else {
        let case = Case {
            batch: 8,
            target_tokens: 128,
            mixed: true,
        };
        let texts = case_texts(&case);
        let engine_b = Engine::create(Device::Cuda).map_err(|e| format!("engine B: {e}"))?;
        engine_b
            .load_model(ALIAS)
            .map_err(|e| format!("engine B load: {e}"))?;
        // Engine is deliberately !Sync (single-engine serialization contract),
        // so each thread owns its engine, matching the allocator-isolation test.
        let run_one = |eng: Engine| -> Result<RepeatStats, String> {
            let views: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
            for _ in 0..args.warmup {
                let out = eng
                    .embed(ALIAS, &views, &opts)
                    .map_err(|e| format!("concurrent warmup: {e}"))?;
                drop(out);
            }
            timed_repeat(
                || {
                    let out = eng
                        .embed(ALIAS, &views, &opts)
                        .map_err(|e| format!("concurrent embed: {e}"))?;
                    drop(out);
                    Ok(())
                },
                args.max_seconds,
                args.max_requests,
            )
        };
        let (a, b) = std::thread::scope(|scope| {
            let ja = scope.spawn(|| run_one(engine));
            let jb = scope.spawn(|| run_one(engine_b));
            (ja.join(), jb.join())
        });
        let a = a.map_err(|_| "engine A thread panicked".to_string())??;
        let b = b.map_err(|_| "engine B thread panicked".to_string())??;
        json!({
            "case": case.name(),
            "engines": 2,
            "engine_a": stats_json(&a),
            "engine_b": stats_json(&b),
            "note": "Recorded two-engine observation on one GPU, not a scalability acceptance result.",
        })
    };

    let receipt = json!({
        "experiment": "nvidia-native-overhead-pilot",
        "provider": "NVIDIA ORT CUDA",
        "alias": ALIAS,
        "model": catalog.model_path,
        "tokenizer": catalog.tokenizer_json,
        "direct_baseline": "raw ort 2.0.0-rc.13 consumer: CUDA EP, IoBinding, reusable host token/hidden buffers, host mean+L2 pooling of last_hidden_state",
        "abi_path": "turboembed.h text ABI via safe Rust Engine (CUDA EP + IoBinding + DEVICE mean+L2 kernel)",
        "execution_shape_note": "Both paths execute fixed [batch, 256]; the stock checkpoint has no pooled output, so the direct consumer reads back [batch, seq, dim] while the ABI pools on device and reads back [batch, dim].",
        "ort_build": ort::info(),
        "gpu": gpu,
        "driver_version": driver,
        "host": hostname(),
        "git_sha": git_head(&root),
        "quick": args.quick,
        "gates": {
            "abi_p50_over_direct_p50_max": P50_OVERHEAD_LIMIT,
            "abi_throughput_over_direct_min": THROUGHPUT_FLOOR,
            "parity_max_abs": PARITY_MAX_ABS,
            "parity_rmse": PARITY_MAX_RMSE,
            "p99_min_samples": P99_MIN_SAMPLES,
        },
        "config": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_seconds": args.max_seconds,
            "max_requests": args.max_requests,
        },
        "cases": case_reports,
        "concurrent_two_engines": concurrent,
        "pass": all_pass,
    });
    write_json(&args.out, &receipt)?;
    eprintln!("nvidia overhead pilot pass={all_pass}");
    Ok(all_pass)
}
