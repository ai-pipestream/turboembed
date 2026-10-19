use super::*;
/// The digest of a container image pinned as `name@sha256:<64 hex>`,
/// where the name is lower-case letters, digits and `._/:-`, starting
/// with a letter or digit (so it can never be read as an option).
pub fn pinned(image: &str) -> Option<&str> {
    let (name, digest) = image.split_once("@sha256:")?;
    let mut b = name.bytes();
    let first = b.next()?;
    let ok = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    (ok(first) && b.all(|c| ok(c) || b"._/:-".contains(&c)) && is_hex(digest, 64)).then_some(digest)
}

/// One capability cell as the core asks about it.
#[derive(Debug, Clone, Copy)]
pub struct Cell<'a> {
    /// turbo_device_info.arch.
    pub arch: &'a str,
    /// turbo_device_info.name: part of the key for a CPU.
    pub name: &'a str,
    pub cpu: bool,
    pub backend: &'a str,
    pub task: u32,
    pub precision: u32,
    /// The compute dtype the backend says the precision resolves to.
    pub dtype: u32,
    /// This build's version number.
    pub version: &'a str,
    /// The operating system this build is for, std::env::consts::OS.
    pub os: &'a str,
    /// The graph format an npu cell loads (`NGRAPH_LITE` or `NATIVE`).
    /// None for every other backend. An npu cell with None matches no
    /// record. A named format matches only when library.settings names it.
    pub graph_format: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Supported {
        benchmark: String,
        cosine_floor: f64,
        speed_ratio: f64,
    },
    /// Why not: no record, or what the best one lacks.
    Not(String),
}

impl Record {
    /// Whether the record is for the cell: measured on mixed rows, with
    /// the same arch label, operating system, backend, task and precision,
    /// and for a CPU the same processor. Mixed rows are what a server sees,
    /// so only they back a capability and its speed_ratio; a dense record
    /// is kept and parsed as reference evidence and backs nothing.
    pub fn is_for(&self, cell: &Cell) -> bool {
        self.rows.kind == ROWS_MIXED
            && self.machine.arch == cell.arch
            && self.machine.os == cell.os
            && self.device.backend == cell.backend
            && task_name(cell.task) == Some(self.task.as_str())
            && precision_name(cell.precision) == Some(self.precision.as_str())
            && (!cell.cpu || (self.device.kind == "DEVICE_CPU" && self.device.name == cell.name))
            && match (cell.backend, cell.graph_format) {
                // A listed npu device always names NGRAPH_LITE or NATIVE.
                // None is an unlisted ordinal, and it must not inherit a
                // record of either format.
                ("npu", None) => false,
                (_, None) => true,
                (_, Some(fmt)) => setting(self, "TURBO_NPU_GRAPH_FORMAT") == Some(fmt),
            }
    }

    /// Why a record for the cell does not back SUPPORTED, or None when it
    /// does: it is for this library version, computed in the dtype the
    /// backend resolves the precision to, reached that dtype's tolerance,
    /// measured at least one reference program, and its mixed rows list
    /// every reference case that fits `seq`. `is_for` can match a thinner
    /// file. Conformance counts each fitting case run alone and then every
    /// measured row. That second count is `rows.cases.len()`: one row of
    /// a single batch, or one `[1, seq]` frame per case when the frame
    /// batch is 1. `conformance.rows` minus that count is how many cases
    /// fit, and the rows have to list that many distinct cases.
    pub fn falls_short(&self, cell: &Cell) -> Option<String> {
        if !self.references.iter().any(|r| r.measured.is_some() && REFERENCES.iter().any(|&(n, _)| n == r.name)) {
            return Some("no reference program measured".into());
        }
        // Records can be constructed or mutated without going through parse.
        // A capability verdict must not trust unchecked floating point fields.
        if let Err(error) = self.check() {
            return Some(format!("invalid record: {error}"));
        }
        if self.library.version != cell.version {
            return Some(format!("recorded with library {}, this is {}", self.library.version, cell.version));
        }
        let Some(tol) = dtype_value(&self.compute_dtype).and_then(tolerance) else {
            return Some(format!("{} has no conformance floor", self.compute_dtype));
        };
        if dtype_value(&self.compute_dtype) != Some(cell.dtype) {
            return Some(format!("computed in {}, the cell computes in dtype {}", self.compute_dtype, cell.dtype));
        }
        let c = &self.conformance;
        if c.min_cosine < tol.min_cosine {
            return Some(format!("min cosine {} is under {}", c.min_cosine, tol.min_cosine));
        }
        if let Some(most) = tol.max_abs_diff
            && c.max_abs_diff > most
        {
            return Some(format!("max abs diff {:e} is over {most:e}", c.max_abs_diff));
        }
        // Mixed rows are the cases that fit seq. A file that times one of
        // them still parses and still matches `is_for`. The measured rows
        // are `cases.len()`: one per row of a single batch (so the count
        // equals `batch`), or one [1, seq] frame per case when batch is 1.
        if self.rows.kind == ROWS_MIXED {
            let timed = self.rows.cases.len() as u32;
            let compared = self.conformance.rows;
            if compared < timed {
                return Some(format!("conformance compared {compared} vectors, fewer than the {timed} measured rows"));
            }
            let fit = compared - timed;
            let mut cases = self.rows.cases.clone();
            cases.sort_unstable();
            cases.dedup();
            let n = cases.len() as u32;
            if n != fit {
                return Some(format!("rows cover {n} of {fit} reference cases that fit seq"));
            }
        }
        None
    }
}

/// The verdict of `records` on a cell. Of the records for it, those that
/// back SUPPORTED are ranked by recorded_at, then name, and the newest
/// wins. When none backs it, the reason is the newest record's.
pub fn decide<'a>(records: impl IntoIterator<Item = (&'a str, &'a Record)>, cell: &Cell) -> Verdict {
    let mut matching: Vec<(&str, &Record)> = records.into_iter().filter(|(_, r)| r.is_for(cell)).collect();
    matching.sort_by(|a, b| (&a.1.recorded_at, a.0).cmp(&(&b.1.recorded_at, b.0)));
    if let Some((name, r)) = matching.iter().rev().find(|(_, r)| r.falls_short(cell).is_none()) {
        return Verdict::Supported {
            benchmark: (*name).to_owned(),
            cosine_floor: r.conformance.min_cosine,
            speed_ratio: r.speed_ratio.unwrap_or(0.0),
        };
    }
    match matching.last() {
        Some((name, r)) => Verdict::Not(format!("{name}: {}", r.falls_short(cell).unwrap_or_default())),
        None => Verdict::Not(NO_RECORD.into()),
    }
}

// EMBEDDED: every file name and its text, from benchmarks/records/.
include!(concat!(env!("OUT_DIR"), "/records.rs"));

/// The records compiled into this build, each parsed, or why it does not.
pub fn embedded() -> &'static [(&'static str, Result<Record, String>)] {
    static PARSED: OnceLock<Vec<(&'static str, Result<Record, String>)>> = OnceLock::new();
    PARSED.get_or_init(|| EMBEDDED.iter().map(|(name, text)| (*name, Record::parse(name, text.as_bytes()))).collect())
}

/// The verdict of the compiled-in records on a cell. A record that does
/// not parse backs nothing; a test keeps the committed ones parsing.
/// When `model_id` is set, only records of that bundle are considered.
pub fn decide_embedded_for(cell: &Cell, model_id: Option<&str>) -> Verdict {
    decide(
        embedded().iter().filter_map(|(n, r)| {
            r.as_ref()
                .ok()
                .filter(|r| model_id.is_none() || model_id == Some(r.bundle.model_id.as_str()))
                .map(|r| (*n, r))
        }),
        cell,
    )
}

/// The verdict of the compiled-in records on a cell. A record that does
/// not parse backs nothing; a test keeps the committed ones parsing.
pub fn decide_embedded(cell: &Cell) -> Verdict {
    decide_embedded_for(cell, None)
}
