//! The library's side of a record: the bundle loaded on the device, the
//! token rows, the timed runs and the conformance check, all through the
//! C interface.

use std::path::PathBuf;
use std::time::Instant;

use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{Manifest, Normalize, Pooling};
use turbo::record::{Conformance, Timing};
use turbo::safetensors::{self, Dtype};
use turbo::{TURBO_DEVICE_CPU, TURBO_TASK_EMBED, turbo_device_info, turbo_model_info};

use crate::Result;
use crate::api::{self, Batch, Runtime, field};

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
    pub warmup: u32,
    pub iterations: u32,
}

/// The bundle's reference: each case's ids and fp32 vector.
pub struct Reference {
    pub ids: Vec<Vec<i32>>,
    pub vectors: Vec<Vec<f32>>,
}

impl Reference {
    /// Read through the core's bundle reader, so every byte is checked
    /// against the manifest's hash first.
    pub fn read(bundle: &Bundle) -> Result<Reference> {
        let file = &bundle.manifest.reference.file;
        let bytes = bundle.read_verified(file).map_err(|e| e.message)?;
        let st = safetensors::File::parse(file, &bytes).map_err(|e| e.message)?;
        let ids = st.get("ids", Dtype::I32, 2).map_err(|e| e.message)?;
        let lengths = st.get("lengths", Dtype::I32, 1).map_err(|e| e.message)?.i32s();
        let emb = st.get("embeddings", Dtype::F32, 2).map_err(|e| e.message)?;
        let (width, dim) = (ids.shape[1] as usize, emb.shape[1] as usize);
        let flat = ids.i32s();
        let mut out = Reference { ids: Vec::new(), vectors: emb.f32s().chunks(dim).map(<[f32]>::to_vec).collect() };
        for (i, &n) in lengths.iter().enumerate() {
            let n = usize::try_from(n).ok().filter(|&n| n >= 1 && n <= width);
            let n = n.ok_or_else(|| format!("{file}: case {i} has length {}", lengths[i]))?;
            out.ids.push(flat[i * width..i * width + n].to_vec());
        }
        if out.ids.len() != out.vectors.len() {
            return Err(format!("{file}: {} id rows and {} vectors", out.ids.len(), out.vectors.len()));
        }
        Ok(out)
    }
}

/// The token rows every program is measured on.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    pub batch: u32,
    pub seq: u32,
    pub ids: Vec<i32>,
    pub mask: Vec<i32>,
    pub types: Vec<i32>,
    /// The reference case each row is.
    pub cases: Vec<u32>,
}

impl Rows {
    /// `batch` rows of `seq` tokens: the reference cases no longer than
    /// `seq`, in order, repeated until the batch is full, each padded with
    /// `pad` and mask 0; types all 0.
    pub fn build(reference: &Reference, pad: i32, batch: u32, seq: u32) -> Result<Rows> {
        let fit: Vec<u32> =
            (0..reference.ids.len() as u32).filter(|&i| reference.ids[i as usize].len() <= seq as usize).collect();
        if fit.is_empty() || batch == 0 {
            return Err(format!("no reference case fits {seq} tokens, or the batch is empty"));
        }
        let n = batch as usize * seq as usize;
        let mut rows = Rows { batch, seq, ids: vec![pad; n], mask: vec![0; n], types: vec![0; n], cases: Vec::new() };
        for r in 0..batch as usize {
            let case = fit[r % fit.len()];
            let ids = &reference.ids[case as usize];
            rows.ids[r * seq as usize..][..ids.len()].copy_from_slice(ids);
            rows.mask[r * seq as usize..][..ids.len()].fill(1);
            rows.cases.push(case);
        }
        Ok(rows)
    }

    /// Row `r`'s live ids, without padding.
    pub fn live(&self, r: usize) -> &[i32] {
        let row = r * self.seq as usize..(r + 1) * self.seq as usize;
        let n = self.mask[row.clone()].iter().filter(|&&m| m == 1).count();
        &self.ids[row][..n]
    }

    pub fn live_tokens(&self) -> u64 {
        self.mask.iter().filter(|&&m| m == 1).count() as u64
    }

    /// docs/benchmarks.md, "Token rows": SHA-256 of `turbo-bench rows 1`,
    /// a NUL, batch and seq as little-endian u32, then ids, mask and types
    /// as little-endian i32, row-major.
    pub fn sha256(&self) -> String {
        let mut b = b"turbo-bench rows 1\0".to_vec();
        b.extend(self.batch.to_le_bytes());
        b.extend(self.seq.to_le_bytes());
        for v in self.ids.iter().chain(&self.mask).chain(&self.types) {
            b.extend(v.to_le_bytes());
        }
        sha256_hex(&b)
    }

    pub fn as_batch(&self) -> Batch<'_> {
        Batch { batch: self.batch, seq: self.seq, ids: &self.ids, mask: &self.mask, types: &self.types }
    }

    /// One row as a batch of one, at its own length.
    pub fn single(&self, r: usize) -> Rows {
        let ids = self.live(r).to_vec();
        let n = ids.len();
        Rows { batch: 1, seq: n as u32, ids, mask: vec![1; n], types: vec![0; n], cases: vec![self.cases[r]] }
    }
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
    let pad = rt.tokenizer(&bundle_dir)?.info()?.pad_id.max(0);
    let ctx = rt.context(index)?;
    let model = ctx.load(&bundle_dir)?;
    let mi = model.info()?;

    let seq = match plan.seq {
        Some(s) => s,
        None => reference.ids.iter().map(|r| r.len() as u32).filter(|&n| n <= mi.max_seq).max().unwrap_or(0),
    };
    let batch = plan.batch.unwrap_or(mi.max_batch.min(32));
    let rows = Rows::build(&reference, pad, batch, seq)?;
    let session = model.session(batch, seq, plan.precision)?;
    let si = session.info()?;
    let dim = mi.dim as usize;

    // Conformance: each row's case alone, then the timed batch below.
    let mut vectors = Vec::new();
    let mut cases = Vec::new();
    let mut one = vec![0f32; dim];
    let mut seen = std::collections::BTreeSet::new();
    for r in 0..rows.batch as usize {
        if seen.insert(rows.cases[r]) {
            let single = rows.single(r);
            session.embed_into(&single.as_batch(), &mut one)?;
            vectors.push(one.clone());
            cases.push(rows.cases[r]);
        }
    }

    let batch_rows = rows.as_batch();
    let mut out = vec![0f32; batch as usize * dim];
    for _ in 0..plan.warmup {
        session.embed_into(&batch_rows, &mut out)?;
    }
    let mut ms = Vec::with_capacity(plan.iterations as usize);
    let started = Instant::now();
    for _ in 0..plan.iterations {
        let t = Instant::now();
        session.embed_into(&batch_rows, &mut out)?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let total = started.elapsed().as_secs_f64();
    let timed: Vec<Vec<f32>> = out.chunks(dim).map(<[f32]>::to_vec).collect();
    vectors.extend(timed.iter().cloned());
    cases.extend(&rows.cases);
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
        rows_per_second: (batch as f64 * plan.iterations as f64) / total,
    };
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
    })
}
