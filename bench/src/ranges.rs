//! `turbo-bench ranges`: what one vector per chunk costs on a device, three
//! ways, on the same dense token rows (docs/benchmarks.md, "Ranges"):
//!
//! - rows: one vector per row, `turbo_embed_write_tokens`, the cost of
//!   reading the text at all;
//! - ranges: the same rows with one range per chunk,
//!   `turbo_embed_write_ranges`, a vector per chunk from one encoder pass;
//! - chunk rows: each chunk embedded as a row of its own,
//!   `[CLS] chunk [SEP]`, in frames of as many rows as a session holds.
//!
//! A chunk is `chunk` consecutive text tokens of a row (the last of a row
//! may be shorter); the row's first and last positions, its special
//! tokens, are in no chunk. Rows and ranges share one session. The record
//! also checks that a range over a whole row gives that row's vector,
//! and reports how close each range's vector is to its chunk row's,
//! which reads the chunk without the text around it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use turbo::bundle::Bundle;
use turbo::record::{self, BundleId, Device, Library, Machine};
use turbo::turbo_token_range;

use crate::Result;
use crate::api::{Batch, Runtime, field};
use crate::git::Provenance;
use crate::measure::{Reference, Rows, cosine, max_abs_diff, percentile};

/// The version of `RangeRecord`.
pub const RANGE_RECORD_VERSION: u32 = 1;

/// Where range records go, under the repository.
pub const RANGES_DIR: &str = "benchmarks/ranges/";

/// What to measure, as the command line gave it.
#[derive(Debug, Clone)]
pub struct RangePlan {
    pub bundle: PathBuf,
    pub device: String,
    pub precision: u32,
    /// Rows per run; None: the smaller of 32 and the model's max_batch.
    pub batch: Option<u32>,
    /// Tokens per row; None: the smaller of 256 and the model's max_seq.
    pub seq: Option<u32>,
    /// Text tokens per chunk.
    pub chunk: u32,
    pub warmup: u32,
    pub iterations: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    pub precision: String,
    pub compute_dtype: String,
    pub bundle: BundleId,
    pub shape: Shape,
    pub rows: Mode,
    pub ranges: Mode,
    pub chunk_rows: Mode,
    /// The largest difference of a range over a whole row from that
    /// row's vector, at any value; 0 when the two are the same bits.
    pub whole_row_max_abs_diff: f64,
    /// The lowest cosine of a chunk's range vector to its chunk row's.
    pub chunk_min_cosine: f64,
    /// ranges.p50_ms over rows.p50_ms: what a vector per chunk adds to
    /// reading the rows.
    pub ranges_over_rows: f64,
    /// chunk_rows.p50_ms over ranges.p50_ms: how much longer one row per
    /// chunk takes than the ranges.
    pub chunk_rows_over_ranges: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    /// Dense rows: every position live.
    pub batch: u32,
    pub seq: u32,
    /// Text tokens per chunk, and chunks per row.
    pub chunk: u32,
    pub chunks_per_row: u32,
    /// Rows per frame of the chunk rows, and frames per run.
    pub chunk_batch: u32,
    pub chunk_frames: u32,
    /// Which reference case each row is.
    pub cases: Vec<u32>,
    /// docs/benchmarks.md, "Token rows": the hash of the rows written.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mode {
    /// Vectors one run gives.
    pub vectors: u32,
    pub warmup: u32,
    pub iterations: u32,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub vectors_per_second: f64,
}

/// The ranges of `batch` rows of `seq` tokens: chunks of `chunk` text
/// tokens, positions 1 to `seq - 1`.
pub fn chunk_ranges(batch: u32, seq: u32, chunk: u32) -> Result<Vec<turbo_token_range>> {
    if chunk == 0 || seq < 3 {
        return Err(format!("chunks of {chunk} tokens in rows of {seq}: a chunk needs a token, a row a text token"));
    }
    let mut out = Vec::new();
    for row in 0..batch {
        let mut start = 1;
        while start < seq - 1 {
            let end = (start + chunk).min(seq - 1);
            out.push(turbo_token_range { row, start, end });
            start = end;
        }
    }
    Ok(out)
}

/// Each range as a row of its own, `[CLS] tokens [SEP]` from `rows`' own
/// first and last tokens, padded with `pad` to the longest.
pub fn chunk_rows(rows: &Rows, ranges: &[turbo_token_range], pad: i32) -> Rows {
    let width = ranges.iter().map(|r| r.end - r.start).max().unwrap_or(0) + 2;
    let (mut ids, mut mask) = (Vec::new(), Vec::new());
    for r in ranges {
        let row = &rows.ids[(r.row * rows.seq) as usize..((r.row + 1) * rows.seq) as usize];
        let len = (r.end - r.start) as usize;
        ids.push(row[0]);
        ids.extend_from_slice(&row[r.start as usize..r.end as usize]);
        ids.push(row[rows.seq as usize - 1]);
        ids.extend(std::iter::repeat_n(pad, width as usize - len - 2));
        mask.extend(std::iter::repeat_n(1, len + 2));
        mask.extend(std::iter::repeat_n(0, width as usize - len - 2));
    }
    let n = ids.len();
    Rows {
        kind: rows.kind,
        batch: ranges.len() as u32,
        seq: width,
        ids,
        mask,
        types: vec![0; n],
        cases: ranges.iter().map(|r| rows.cases[r.row as usize]).collect(),
    }
}

/// `warmup` untimed runs of `f`, then `iterations` timed ones.
fn time(warmup: u32, iterations: u32, vectors: u32, mut f: impl FnMut() -> Result<()>) -> Result<Mode> {
    for _ in 0..warmup {
        f()?;
    }
    let mut ms = Vec::with_capacity(iterations as usize);
    let started = Instant::now();
    for _ in 0..iterations {
        let t = Instant::now();
        f()?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let total = started.elapsed().as_secs_f64();
    let mut sorted = ms.clone();
    sorted.sort_by(f64::total_cmp);
    Ok(Mode {
        vectors,
        warmup,
        iterations,
        p50_ms: percentile(&sorted, 50.0),
        p99_ms: percentile(&sorted, 99.0),
        mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
        vectors_per_second: f64::from(vectors) * f64::from(iterations) / total,
    })
}

/// Measure the three ways and make the record.
pub fn measure(plan: &RangePlan, p: &Provenance, recorded_at: String) -> Result<RangeRecord> {
    if plan.iterations == 0 {
        return Err("--iterations must be above 0".into());
    }
    let dir = std::fs::canonicalize(&plan.bundle).map_err(|e| format!("{}: {e}", plan.bundle.display()))?;
    let bundle = Bundle::open(&dir).map_err(|e| e.message)?;
    let reference = Reference::read(&bundle)?;
    let rt = Runtime::create()?;
    let index = rt.find(&plan.device)?;
    let d = rt.device_info(index)?;
    let host_cpu = rt.cpu()?.map(|c| field(&c.name)).unwrap_or_default();
    let tokenizer = rt.tokenizer(&dir)?;
    let pad = tokenizer.info()?.pad_id.max(0);
    let ctx = rt.context(index)?;
    let model = ctx.load(&dir)?;
    let mi = model.info()?;
    let batch = plan.batch.unwrap_or(mi.max_batch.min(32));
    let seq = plan.seq.unwrap_or(mi.max_seq.min(256));
    let rows = Rows::dense(&reference, &tokenizer, batch, seq)?;
    drop(tokenizer);
    let ranges = chunk_ranges(batch, seq, plan.chunk)?;
    let per_row = (ranges.len() / batch as usize) as u32;
    let dim = mi.dim as usize;
    let session = model.session_with_ranges(batch, seq, plan.precision, ranges.len() as u32)?;
    let si = session.info()?;

    // A range over each whole row against the row's vector.
    let whole: Vec<turbo_token_range> = (0..batch).map(|row| turbo_token_range { row, start: 0, end: seq }).collect();
    let mut row_vectors = vec![0f32; batch as usize * dim];
    let mut whole_vectors = row_vectors.clone();
    session.embed_into(&rows.as_batch(), &mut row_vectors)?;
    session.embed_ranges_into(&rows.as_batch(), &whole, &mut whole_vectors)?;
    let whole_row_max_abs_diff = max_abs_diff(&row_vectors, &whole_vectors);

    let chunked = chunk_rows(&rows, &ranges, pad);
    let chunk_batch = (ranges.len() as u32).min(mi.max_batch);
    let frames = (ranges.len() as u32).div_ceil(chunk_batch);
    let chunk_session = model.session(chunk_batch, chunked.seq, plan.precision)?;
    let frame = |f: u32| {
        let a = (f * chunk_batch) as usize;
        let b = ((f + 1) * chunk_batch).min(chunked.batch) as usize;
        let w = chunked.seq as usize;
        Batch {
            batch: (b - a) as u32,
            seq: chunked.seq,
            ids: &chunked.ids[a * w..b * w],
            mask: &chunked.mask[a * w..b * w],
            types: &chunked.types[a * w..b * w],
        }
    };
    let mut range_vectors = vec![0f32; ranges.len() * dim];
    let mut chunk_vectors = vec![0f32; ranges.len() * dim];
    session.embed_ranges_into(&rows.as_batch(), &ranges, &mut range_vectors)?;
    for f in 0..frames {
        let a = (f * chunk_batch) as usize * dim;
        let b = frame(f);
        let n = b.batch as usize * dim;
        chunk_session.embed_into(&b, &mut chunk_vectors[a..a + n])?;
    }
    let chunk_min_cosine =
        range_vectors.chunks(dim).zip(chunk_vectors.chunks(dim)).map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);

    let (w, it) = (plan.warmup, plan.iterations);
    let rows_mode = time(w, it, batch, || session.embed_into(&rows.as_batch(), &mut row_vectors).map(drop))?;
    let ranges_mode = time(w, it, ranges.len() as u32, || {
        session.embed_ranges_into(&rows.as_batch(), &ranges, &mut range_vectors).map(drop)
    })?;
    let chunk_mode = time(w, it, ranges.len() as u32, || {
        for f in 0..frames {
            let a = (f * chunk_batch) as usize * dim;
            let b = frame(f);
            let n = b.batch as usize * dim;
            chunk_session.embed_into(&b, &mut chunk_vectors[a..a + n])?;
        }
        Ok(())
    })?;

    let backend = field(&d.backend);
    let r = RangeRecord {
        record_version: RANGE_RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu, os: std::env::consts::OS.into() },
        device: Device {
            backend: backend.clone(),
            kind: crate::kind_name(d.kind).into(),
            name: field(&d.name),
            vendor: field(&d.vendor),
            driver_version: field(&d.driver_version),
            runtime_version: field(&d.runtime_version),
            memory_total: d.memory_total,
        },
        library: Library {
            version: crate::api::version().split(' ').next().unwrap_or("").into(),
            build: crate::api::version(),
            commit: p.commit.clone(),
            pushed_to: p.pushed_to.clone(),
            settings: crate::measure::library_settings(&backend, &si),
        },
        precision: record::precision_name(plan.precision).ok_or("precision is not a TURBO_PRECISION_* value")?.into(),
        compute_dtype: record::dtype_name(si.compute_dtype)
            .ok_or_else(|| format!("compute dtype {} has no name in a record", si.compute_dtype))?
            .into(),
        bundle: BundleId {
            model_id: field(&mi.model_id),
            revision: field(&mi.revision),
            manifest_sha256: field(&mi.manifest_sha256),
            artifact_sha256: field(&mi.artifact_sha256),
            tokenizer_sha256: field(&mi.tokenizer_sha256),
        },
        shape: Shape {
            batch,
            seq,
            chunk: plan.chunk,
            chunks_per_row: per_row,
            chunk_batch,
            chunk_frames: frames,
            cases: rows.cases.clone(),
            sha256: rows.sha256(),
        },
        ranges_over_rows: ranges_mode.p50_ms / rows_mode.p50_ms,
        chunk_rows_over_ranges: chunk_mode.p50_ms / ranges_mode.p50_ms,
        rows: rows_mode,
        ranges: ranges_mode,
        chunk_rows: chunk_mode,
        whole_row_max_abs_diff,
        chunk_min_cosine,
    };
    Ok(r)
}

/// The record's file name: machine, backend, precision, model and commit.
pub fn file_name(r: &RangeRecord) -> String {
    let slug = |s: &str| {
        let mut out = String::new();
        for c in s.chars() {
            if c.is_ascii_alphanumeric() {
                out.push(c.to_ascii_lowercase());
            } else if !out.is_empty() && !out.ends_with('-') {
                out.push('-');
            }
        }
        out.trim_end_matches('-').to_owned()
    };
    let mut model = slug(r.bundle.model_id.rsplit('/').next().unwrap_or(""));
    model.truncate(32);
    format!(
        "{}.{}.ranges.{}.{}-{}.{}.json",
        slug(&r.machine.arch),
        slug(&r.device.backend),
        slug(r.precision.strip_prefix("PRECISION_").unwrap_or(&r.precision)),
        model.trim_end_matches('-'),
        r.bundle.manifest_sha256.get(..8).unwrap_or(""),
        r.library.commit.get(..12).unwrap_or("")
    )
}

/// Write the record under `dir`; a record is never replaced.
pub fn write(r: &RangeRecord, dir: &Path) -> Result<PathBuf> {
    let mut bytes = serde_json::to_vec_pretty(r).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(file_name(r));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}; a record is never replaced", path.display()))?;
    std::io::Write::write_all(&mut f, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
#[path = "ranges_tests.rs"]
mod tests;
