//! Score-task records use pass timings and upstream score/logit bounds,
//! unlike embedding records, which measure vector cosine.
use super::*;
use crate::{TURBO_TASK_EXTRACTIVE_QA, TURBO_TASK_RERANK};

mod rerank {
    include!(concat!(env!("OUT_DIR"), "/rerank.rs"));
}
mod qa {
    include!(concat!(env!("OUT_DIR"), "/qa.rs"));
}

// The v1 task timings have no field for upstream conformance. Bind the two
// independently verified EXACT/F32 runs in docs/reranking.md (Parity and
// Throughput) to their entire timing receipts: device, model manifest, code
// and request set. The bench at this commit used default RerankOptions.
// A new or altered timing needs its own parity evidence. TEI's score
// difference is not the error against the pinned Transformers reference.
const EXACT_RERANK_PARITY: &[(&str, f64)] = &[
    ("61e9f3812f356b430a27e14c169affa5a64cc91264a7c0655000853587dbda63", 0.0000024), // MiniLM
    ("d2830cd7173f66c6989cd0c090e920dcba8a5f3a2d7f58234b8eec4cd9666130", 0.0000075), // BGE
];

// Only the fields used by the capability decision. The bench owns and
// validates the full v1 file formats, including bin and throughput data.
#[derive(Deserialize)]
struct TaskRecord {
    record_version: u32,
    recorded_at: String,
    machine: Machine,
    device: Device,
    library: Library,
    model_id: String,
    precision: String,
    turbo: Option<Pass>,
    tei: Option<Tei>,
}

#[derive(Deserialize)]
struct Pass {
    p50_ms: f64,
}

#[derive(Deserialize)]
struct Tei {
    side: Pass,
}

impl TaskRecord {
    fn reference_ratio(&self, task: u32, text: &str) -> Result<f64, &'static str> {
        let (TURBO_TASK_RERANK, Some(ours), Some(tei)) = (task, &self.turbo, &self.tei) else {
            return Err("no reference program measured");
        };
        let digest = sha256_hex(text.as_bytes());
        if !EXACT_RERANK_PARITY.iter().any(|&(hash, error)| digest == hash && error <= 1e-4) {
            return Err("no matching upstream score conformance");
        }
        Ok(ours.p50_ms / tei.side.p50_ms)
    }
}

struct Measurement {
    task: u32,
    name: &'static str,
    record: TaskRecord,
    ratio: Result<f64, &'static str>,
}

fn measurements() -> &'static [Measurement] {
    static MEASURED: OnceLock<Vec<Measurement>> = OnceLock::new();
    MEASURED.get_or_init(|| {
        let mut reports = Vec::new();
        for (task, files) in [(TURBO_TASK_RERANK, rerank::EMBEDDED), (TURBO_TASK_EXTRACTIVE_QA, qa::EMBEDDED)] {
            for &(name, text) in files {
                let Ok(record) = serde_json::from_str::<TaskRecord>(text) else { continue };
                if record.record_version != 1 || name.len() > NAME_MAX {
                    continue;
                }
                let ratio = record.reference_ratio(task, text);
                reports.push(Measurement { task, name, record, ratio });
            }
        }
        reports.sort_by(|a, b| (&a.record.recorded_at, a.name).cmp(&(&b.record.recorded_at, b.name)));
        reports
    })
}

pub(super) fn decide(cell: &Cell, model_id: Option<&str>) -> Verdict {
    let mut reason = None;
    for m in measurements().iter().rev() {
        let r = &m.record;
        if cell.task != m.task
            || cell.arch != r.machine.arch
            || cell.os != r.machine.os
            || cell.backend != r.device.backend
            || cell.name != r.device.name
            || cell.cpu != (r.device.kind == "DEVICE_CPU")
            || precision_name(cell.precision) != Some(r.precision.as_str())
            || model_id.is_some_and(|id| id != r.model_id)
        {
            continue;
        }
        let why = match m.ratio {
            _ if r.library.version != cell.version => "recorded with a different library version",
            Err(why) => why,
            Ok(_) if cell.precision != TURBO_PRECISION_EXACT || cell.dtype != TURBO_DTYPE_F32 => {
                "score conformance measured at EXACT in F32"
            }
            Ok(speed_ratio) => return Verdict::Supported { benchmark: m.name.into(), cosine_floor: 0.0, speed_ratio },
        };
        reason.get_or_insert_with(|| format!("{}: {why}", m.name));
    }
    Verdict::Not(reason.unwrap_or_else(|| NO_RECORD.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_changed_workload_cannot_inherit_upstream_score_parity() {
        let (_, text) = rerank::EMBEDDED.iter().find(|(name, _)| name.starts_with("rtx4080super.")).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(text).unwrap();
        value["requests"] = 1.into();
        let changed = value.to_string();
        let record: TaskRecord = serde_json::from_str(&changed).unwrap();
        assert_eq!(record.reference_ratio(TURBO_TASK_RERANK, &changed), Err("no matching upstream score conformance"));
    }
}
