//! The library's side of a record: the bundle loaded on the device, the
//! token rows, the timed runs and the conformance check, all through the
//! C interface.

mod rows;
#[cfg(test)]
mod tests;

use std::collections::btree_map::Entry;
use std::path::PathBuf;
use std::time::Instant;

pub use rows::{Reference, RowKind, Rows};
use turbo::bundle::Bundle;
use turbo::manifest::{Artifact, GraphInput, Manifest, Normalize, Pooling};
use turbo::record::{Conformance, Timing};
use turbo::{TURBO_DEVICE_CPU, TURBO_TASK_EMBED, turbo_device_info, turbo_model_info, turbo_session_info};

use crate::Result;
use crate::api::{self, Runtime, field};

/// What to measure, as the command line gave it.
#[derive(Debug, Clone)]
pub struct Plan {
    pub bundle: PathBuf,
    /// A runtime device index or a backend name.
    pub device: String,
    /// TURBO_PRECISION_*.
    pub precision: u32,
    /// Rows per run; None: the smaller of 32 and the model's max_batch.
    pub batch: Option<u32>,
    /// Tokens per row; None: the longest reference case that fits the model.
    pub seq: Option<u32>,
    pub rows: RowKind,
    pub warmup: u32,
    pub iterations: u32,
}

/// Everything the library's side of a record holds.
pub struct Measurement {
    pub device: turbo_device_info,
    /// The CPU device's name, for context; empty when none is listed.
    pub host_cpu: String,
    /// turbo_version().
    pub build: String,
    pub model: turbo_model_info,
    pub manifest: Manifest,
    pub bundle_dir: PathBuf,
    pub precision: u32,
    pub compute_dtype: u32,
    pub rows: Rows,
    pub reference: Reference,
    pub timing: Timing,
    pub conformance: Conformance,
    /// The vectors of the last timed run, row by row.
    pub vectors: Vec<Vec<f32>>,
    /// What each row's vector should be: the reference vector for a
    /// whole case, and for a row cut to seq, the library's vector of that
    /// row run alone, which is all there is to compare it with.
    pub expected: Vec<Vec<f32>>,
    /// Each of record::LIBRARY_VARS for the backend that was set, as
    /// `NAME=value`: what the library read besides the session's options.
    pub settings: Vec<String>,
}

impl Measurement {
    pub fn is_cpu(&self) -> bool {
        self.device.kind == TURBO_DEVICE_CPU
    }

    pub fn backend(&self) -> String {
        field(&self.device.backend)
    }

    pub fn pooling(&self) -> Pooling {
        self.manifest.embed().pooling
    }

    pub fn normalize(&self) -> bool {
        self.manifest.embed().normalize == Normalize::L2
    }
}

/// The nearest-rank percentile of sorted samples.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

/// The lowest cosine and largest absolute difference of `vectors`, row
/// by row, against the reference vectors of `cases`.
pub fn compare(vectors: &[Vec<f32>], cases: &[u32], reference: &Reference) -> Result<Conformance> {
    let mut c = Conformance { rows: 0, min_cosine: 1.0, max_abs_diff: 0.0 };
    for (v, &case) in vectors.iter().zip(cases) {
        let want = &reference.vectors[case as usize];
        if v.len() != want.len() {
            return Err(format!("a vector has {} values, the reference {}", v.len(), want.len()));
        }
        c.min_cosine = c.min_cosine.min(cosine(v, want));
        c.max_abs_diff = c.max_abs_diff.max(max_abs_diff(v, want));
        c.rows += 1;
    }
    Ok(c)
}

/// Each of record::LIBRARY_VARS for `backend`, as `NAME=value`, in that
/// order: the variables set in this process, and what the session reports
/// of its kernel choices, whatever the environment said:
/// TURBO_<BACKEND>_TUNED, where they came from, and
/// TURBO_<BACKEND>_CHOICES, the choices themselves, for a backend that
/// reports any.
pub fn library_settings(backend: &str, session: &turbo_session_info) -> Vec<String> {
    let choices = field(&session.choices);
    let prefix = format!("TURBO_{}_", backend.to_uppercase());
    turbo::record::LIBRARY_VARS
        .iter()
        .filter(|(b, _)| *b == backend)
        .filter_map(|(_, v)| match v.strip_prefix(prefix.as_str()) {
            Some("TUNED") => (!choices.is_empty())
                .then(|| turbo::tuning::tuned_name(session.tuned).map(|t| format!("{v}={t}")))
                .flatten(),
            Some("CHOICES") => (!choices.is_empty()).then(|| format!("{v}={choices}")),
            // The device selected this. The environment cannot rename it.
            Some("GRAPH_FORMAT") => (!choices.is_empty()).then(|| format!("{v}={choices}")),
            // The bench appends this from the loaded artifact.
            Some("GRAPH_INPUT") => None,
            _ => std::env::var(v).ok().map(|x| format!("{v}={x}")),
        })
        .collect()
}

/// The artifact the model loaded, by the hash `turbo_model_info` reports.
pub fn loaded_artifact<'a>(manifest: &'a Manifest, artifact_sha256: &str) -> Result<&'a Artifact> {
    manifest
        .artifacts
        .iter()
        .find(|a| turbo::model::artifact_sha256(manifest, a) == artifact_sha256)
        .ok_or_else(|| format!("the loaded artifact {artifact_sha256} is not in the manifest"))
}

/// The published npu cell is one compiled frame. `--batch` and `--seq`
/// are the loaded artifact's `fixed_batch` and `fixed_seq`, both above
/// 0. One library frame is one `benchmark_app` request of
/// `[fixed_batch, fixed_seq]`. The token-id MiniLM seal
/// (`openvino-f16`) is `--batch 1 --seq 128`. Omitting either flag is
/// the tool's default shape, which is not that cell.
pub fn require_compiled_frame(batch: Option<u32>, seq: Option<u32>, fixed_batch: u32, fixed_seq: u32) -> Result<()> {
    if fixed_batch == 0 || fixed_seq == 0 {
        return Err(
            "npu: the loaded artifact has no fixed_batch and fixed_seq, so there is no one-frame cell to publish"
                .into(),
        );
    }
    if batch == Some(fixed_batch) && seq == Some(fixed_seq) {
        return Ok(());
    }
    Err(format!(
        "npu: the published cell is one compiled frame, --batch {fixed_batch} --seq {fixed_seq}. One library frame \
         is one benchmark_app request of [{fixed_batch}, {fixed_seq}]. The token-id MiniLM seal (openvino-f16) is \
         --batch 1 --seq 128"
    ))
}

/// Token positions one library run computed. The npu graph executes
/// each compiled frame whole, including positions the mask marks as
/// padding. One frame is `batch` x `seq`. A mixed npu pass whose frame
/// batch is 1 runs every fitting case as its own frame, so the count is
/// that many frames times `seq`. Other backends pack through each row's
/// last live token.
pub fn library_computed_tokens(backend: &str, rows: &Rows) -> u64 {
    if matches!(backend, "npu" | "hailo") { rows.padded_tokens() } else { rows.packed_tokens() }
}

/// As a record names a graph input.
pub fn graph_input_name(input: GraphInput) -> &'static str {
    match input {
        GraphInput::TokenIds => "INPUT_TOKEN_IDS",
        GraphInput::Embeddings => "INPUT_EMBEDDINGS",
    }
}

/// Why an npu `INPUT_EMBEDDINGS` run must not be given a speed_ratio.
/// The host gather is not `benchmark_app`'s full ONNX encoder. A
/// library-only record (both references disabled) is allowed.
pub fn npu_embeddings_speed<'a>(backend: &str, settings: &[String], tei: bool, openvino: bool) -> Option<&'a str> {
    if backend == "npu" && settings.iter().any(|s| s == "TURBO_NPU_GRAPH_INPUT=INPUT_EMBEDDINGS") && (tei || openvino) {
        Some(
            "INPUT_EMBEDDINGS has no speed_ratio: the host gather is not benchmark_app's token-id graph. Pass \
             --no-tei and --no-openvino to write a library-only record",
        )
    } else {
        None
    }
}

/// Load the bundle on the device, time the runs, and check the vectors.
pub fn measure(plan: &Plan) -> Result<Measurement> {
    if plan.iterations == 0 {
        return Err("--iterations must be above 0".into());
    }
    let bundle_dir = std::fs::canonicalize(&plan.bundle).map_err(|e| format!("{}: {e}", plan.bundle.display()))?;
    let bundle = Bundle::open(&bundle_dir).map_err(|e| e.message)?;
    let reference = Reference::read(&bundle)?;

    let rt = Runtime::create()?;
    let index = rt.find(&plan.device)?;
    let device = rt.device_info(index)?;
    let host_cpu = rt.cpu()?.map(|c| field(&c.name)).unwrap_or_default();
    let cap = rt.capability(index, TURBO_TASK_EMBED, plan.precision)?;
    if cap.status == turbo::backend::TURBO_CAP_UNSUPPORTED {
        return Err(format!("device {index}: embed at this precision is unsupported: {}", field(&cap.reason)));
    }
    let tokenizer = rt.tokenizer(&bundle_dir)?;
    let pad = tokenizer.info()?.pad_id.max(0);
    let ctx = rt.context(index)?;
    let model = ctx.load(&bundle_dir)?;
    let mi = model.info()?;
    let backend_name = field(&device.backend);
    let loaded = loaded_artifact(&bundle.manifest, &field(&mi.artifact_sha256))?;
    if backend_name == "npu" {
        require_compiled_frame(plan.batch, plan.seq, loaded.fixed_batch, loaded.fixed_seq)?;
    }
    if backend_name == "hailo" && plan.seq != Some(loaded.fixed_seq) {
        return Err(format!("hailo: --seq must equal the HEF fixed_seq {}", loaded.fixed_seq));
    }
    let graph_input = graph_input_name(loaded.graph_input);

    let seq = match plan.seq {
        Some(s) => s,
        None => reference.ids.iter().map(|r| r.len() as u32).filter(|&n| n <= mi.max_seq).max().unwrap_or(0),
    };
    let batch = plan.batch.unwrap_or(mi.max_batch.min(32));
    // A compiled frame of one row holds case 0 only if the batch is
    // filled and stopped. On npu that frame is fixed, so mixed rows
    // cycle every case that fits, each in its own [1, seq] frame.
    // Raising batch would name a shape the graph was not built for.
    let rows = match plan.rows {
        RowKind::Mixed if backend_name == "npu" && batch == 1 => Rows::cycle(&reference, pad, seq)?,
        RowKind::Mixed => Rows::build(&reference, pad, batch, seq)?,
        RowKind::Dense => Rows::dense(&reference, &tokenizer, batch, seq)?,
    };
    drop(tokenizer);
    let session = model.session(batch, seq, plan.precision)?;
    let si = session.info()?;
    let dim = mi.dim as usize;

    // Conformance: each reference case that fits the rows alone, then
    // each whole row of the timed batch below.
    let mut vectors = Vec::new();
    let mut cases = Vec::new();
    let mut one = vec![0f32; dim];
    for case in 0..reference.ids.len() as u32 {
        if reference.ids[case as usize].len() <= seq as usize {
            session.embed_into(&Rows::case(&reference, case).as_batch(), &mut one)?;
            vectors.push(one.clone());
            cases.push(case);
        }
    }
    // A row cut to seq has no reference vector: it should give what it
    // gives alone.
    let mut expected = Vec::with_capacity(rows.row_count());
    let mut alone = std::collections::BTreeMap::new();
    for r in 0..rows.row_count() {
        if rows.whole(r, &reference) {
            expected.push(reference.vectors[rows.cases[r] as usize].clone());
        } else {
            let v = match alone.entry(rows.cases[r]) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    session.embed_into(&rows.single(r).as_batch(), &mut one)?;
                    e.insert(one.clone())
                }
            };
            expected.push(v.clone());
        }
    }

    // One pass is every frame, in order. A single frame is one embed of
    // the batch, as before. A batch-1 cycle embeds each case on its own.
    let frames = rows.frames();
    let mut out = vec![0f32; rows.batch as usize * dim];
    let mut last = Vec::with_capacity(rows.row_count());
    for _ in 0..plan.warmup {
        for f in 0..frames {
            session.embed_into(&rows.frame_batch(f), &mut out)?;
        }
    }
    let mut ms = Vec::with_capacity(plan.iterations as usize);
    let started = Instant::now();
    for i in 0..plan.iterations {
        let t = Instant::now();
        let keep = i + 1 == plan.iterations;
        if keep {
            last.clear();
        }
        for f in 0..frames {
            session.embed_into(&rows.frame_batch(f), &mut out)?;
            if keep {
                last.extend(out.chunks(dim).map(<[f32]>::to_vec));
            }
        }
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let total = started.elapsed().as_secs_f64();
    let timed = last;
    for (r, v) in timed.iter().enumerate() {
        if rows.whole(r, &reference) {
            vectors.push(v.clone());
            cases.push(rows.cases[r]);
        } else {
            let Some(tol) = turbo::record::tolerance(si.compute_dtype) else {
                let name = turbo::record::dtype_name(si.compute_dtype)
                    .map_or_else(|| si.compute_dtype.to_string(), str::to_owned);
                return Err(format!("no tolerance for dtype {name}; a dense run can't check its cut rows"));
            };
            let (c, d) = (cosine(v, &expected[r]), max_abs_diff(v, &expected[r]));
            if c < tol.min_cosine || tol.max_abs_diff.is_some_and(|most| d > most) {
                return Err(format!(
                    "row {r}, reference case {} cut to {seq} tokens, differs in the batch from alone: \
                     cosine {c}, max abs diff {d:e}",
                    rows.cases[r]
                ));
            }
        }
    }
    let conformance = compare(&vectors, &cases, &reference)?;

    let mut sorted = ms.clone();
    sorted.sort_by(f64::total_cmp);
    let timing = Timing {
        warmup: plan.warmup,
        iterations: plan.iterations,
        p50_ms: percentile(&sorted, 50.0),
        p99_ms: percentile(&sorted, 99.0),
        mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
        rows_per_second: (rows.row_count() as f64 * plan.iterations as f64) / total,
        computed_tokens: Some(library_computed_tokens(&backend_name, &rows)),
    };
    let mut settings = library_settings(&backend_name, &si);
    if backend_name == "npu" {
        settings.push(format!("TURBO_NPU_GRAPH_INPUT={graph_input}"));
    }
    Ok(Measurement {
        device,
        host_cpu,
        build: api::version(),
        model: mi,
        manifest: bundle.manifest,
        bundle_dir,
        precision: plan.precision,
        compute_dtype: si.compute_dtype,
        rows,
        reference,
        timing,
        conformance,
        vectors: timed,
        expected,
        settings,
    })
}
