//! The evaluation record: what one pipeline scored on one set, what it
//! cost, and where it ran. One JSON file per set and pipeline.
//!
//! Records are written and read through these types alone. Every field is
//! required (a field that may be absent is written as `null`, never left
//! out) and a field the types do not know is an error, at every level.
//! They are written with `serde_json`'s pretty printer and a final
//! newline, as turbo-bench writes its records.
//!
//! The provenance block is turbo-bench's: the commit and the origin
//! branches that contain it as `library.commit` and `library.pushed_to`
//! name them, and the core's own `Machine`, `Device` and `BundleId` types
//! from `turbo::record`, reused rather than copied.

use std::fs;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo::record::{BundleId, Device, HOST_PATHS, Machine};

use crate::Result;
use crate::cost::Cost;
use crate::metrics::{DocScores, RecallAtK, mean};
use crate::spans::{SpanAtK, SpanEval, SpanScores};

/// The schema version this crate writes and reads.
pub const EVAL_RECORD_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalRecord {
    pub record_version: u32,
    /// When the run finished, UTC, `YYYY-MM-DDTHH:MM:SSZ`.
    pub recorded_at: String,
    pub provenance: Provenance,
    pub set: SetId,
    pub pipeline: Pipeline,
    pub metrics: Metrics,
    pub cost: Cost,
}

/// Where the run was made, as turbo-bench records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// The commit the pipeline was built from, 40 hex.
    pub commit: String,
    /// The branches of origin that contain the commit, `origin/<name>`.
    pub pushed_to: Vec<String>,
    pub machine: Machine,
    /// The device the embedder ran on.
    pub device: Device,
    /// The embedder's bundle. The pipeline description does not repeat
    /// it.
    pub bundle: BundleId,
}

/// Which evaluation set, and exactly which bytes of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetId {
    pub name: String,
    /// SHA-256 of the set as read, 64 lower case hex.
    pub sha256: String,
}

/// What the pipeline did before retrieval. The embedder is the
/// provenance block's bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pipeline {
    /// The analyzer's annotators, by name, in the order they ran.
    pub analyzer: Vec<String>,
    pub chunker: Chunker,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunker {
    pub name: String,
    /// Every setting the chunker ran with, defaults included, in the
    /// chunker's own order.
    pub config: Vec<Setting>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Setting {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    /// Queries scored at document level: those with a relevant document.
    pub queries: u32,
    /// Mean nDCG@10 over those queries, from 0 to 1.
    pub ndcg_at_10: f64,
    pub recall: RecallAtK,
    /// The span measures; null when the set has no span judgements. The
    /// field must still be present.
    #[serde(deserialize_with = "Option::deserialize")]
    pub span: Option<SpanMetrics>,
    /// nDCG@10 of each scored query, in query byte order. The gate pairs
    /// these between two records for its significance tests.
    pub per_query: Vec<QueryNdcg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanMetrics {
    /// Queries scored at span level: those with a relevant span.
    pub queries: u32,
    pub mean: SpanAtK,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryNdcg {
    pub query: String,
    pub ndcg_at_10: f64,
}

impl Metrics {
    /// The record's metrics from the document and span evaluations.
    pub fn new(docs: &DocScores, spans: Option<&SpanEval>) -> Result<Metrics> {
        let count = |n: usize| u32::try_from(n).map_err(|_| format!("{n} queries is too many for a record"));
        Ok(Metrics {
            queries: count(docs.per_query.len())?,
            ndcg_at_10: docs.ndcg_at_10,
            recall: docs.recall,
            span: match spans {
                Some(s) => Some(SpanMetrics { queries: count(s.per_query.len())?, mean: s.mean }),
                None => None,
            },
            per_query: docs
                .per_query
                .iter()
                .map(|q| QueryNdcg { query: q.query.clone(), ndcg_at_10: q.ndcg_at_10 })
                .collect(),
        })
    }
}

pub(crate) fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn unit(what: &str, v: f64) -> Result<()> {
    if (0.0..=1.0).contains(&v) { Ok(()) } else { Err(format!("{what} is {v}, outside [0, 1]")) }
}

fn recall_in_unit(what: &str, r: &RecallAtK) -> Result<()> {
    unit(&format!("{what}.at_1"), r.at_1)?;
    unit(&format!("{what}.at_5"), r.at_5)?;
    unit(&format!("{what}.at_10"), r.at_10)?;
    unit(&format!("{what}.at_100"), r.at_100)
}

fn span_in_unit(what: &str, s: &SpanScores) -> Result<()> {
    unit(&format!("{what}.precision"), s.precision)?;
    unit(&format!("{what}.recall"), s.recall)?;
    unit(&format!("{what}.contained"), s.contained)
}

impl EvalRecord {
    /// Checks what serde cannot: the version, the shape of each hash and
    /// time, measures within their range, per query scores in order and
    /// agreeing with the mean, and no host path anywhere.
    pub fn check(&self) -> Result<()> {
        if self.record_version != EVAL_RECORD_VERSION {
            return Err(format!("record_version {} is not {EVAL_RECORD_VERSION}", self.record_version));
        }
        let shape = b"dddd-dd-ddTdd:dd:ddZ";
        let t = self.recorded_at.as_bytes();
        if t.len() != shape.len()
            || !t.iter().zip(shape).all(|(&c, &s)| if s == b'd' { c.is_ascii_digit() } else { c == s })
        {
            return Err(format!("recorded_at {:?} is not YYYY-MM-DDTHH:MM:SSZ", self.recorded_at));
        }
        let p = &self.provenance;
        if !is_hex(&p.commit, 40) {
            return Err(format!("provenance.commit {:?} is not 40 lower case hex", p.commit));
        }
        if p.pushed_to.is_empty() {
            return Err("provenance.pushed_to is empty: the commit is on no branch of origin".into());
        }
        if self.set.name.is_empty() {
            return Err("set.name is empty".into());
        }
        if !is_hex(&self.set.sha256, 64) {
            return Err(format!("set.sha256 {:?} is not 64 lower case hex", self.set.sha256));
        }
        if self.pipeline.chunker.name.is_empty() {
            return Err("pipeline.chunker.name is empty".into());
        }

        let m = &self.metrics;
        if m.per_query.is_empty() {
            return Err("metrics.per_query is empty".into());
        }
        if m.queries as usize != m.per_query.len() {
            return Err(format!("metrics.queries is {} and per_query has {}", m.queries, m.per_query.len()));
        }
        for w in m.per_query.windows(2) {
            if w[0].query >= w[1].query {
                return Err(format!("metrics.per_query: {} is not before {}", w[0].query, w[1].query));
            }
        }
        for q in &m.per_query {
            unit(&format!("metrics.per_query {}", q.query), q.ndcg_at_10)?;
        }
        unit("metrics.ndcg_at_10", m.ndcg_at_10)?;
        let from_queries = mean(m.per_query.iter().map(|q| q.ndcg_at_10));
        if (from_queries - m.ndcg_at_10).abs() > 1e-12 {
            return Err(format!("metrics.ndcg_at_10 is {} and the per query mean is {from_queries}", m.ndcg_at_10));
        }
        recall_in_unit("metrics.recall", &m.recall)?;
        if let Some(s) = &m.span {
            if s.queries == 0 {
                return Err("metrics.span.queries is 0; a set without span judgements has span null".into());
            }
            span_in_unit("metrics.span.mean.at_1", &s.mean.at_1)?;
            span_in_unit("metrics.span.mean.at_5", &s.mean.at_5)?;
            span_in_unit("metrics.span.mean.at_10", &s.mean.at_10)?;
            span_in_unit("metrics.span.mean.at_100", &s.mean.at_100)?;
        }

        let c = &self.cost;
        for (what, ms) in [
            ("analysis_ms", c.wall.analysis_ms),
            ("chunking_ms", c.wall.chunking_ms),
            ("embedding_ms", c.wall.embedding_ms),
            ("retrieval_ms", c.wall.retrieval_ms),
        ] {
            if !(ms.is_finite() && ms >= 0.0) {
                return Err(format!("cost.wall.{what} is {ms}"));
            }
        }
        if !c.chunk_bytes.is_ordered() || !c.chunk_tokens.is_ordered() {
            return Err("cost: a chunk length distribution is not min <= p50 <= p90 <= max".into());
        }

        let text = serde_json::to_string(self).map_err(|e| e.to_string())?;
        if let Some(h) = HOST_PATHS.iter().find(|h| text.contains(*h)) {
            return Err(format!("the record holds a host path under {h}"));
        }
        Ok(())
    }

    /// The record as written to disk: checked, pretty printed, and ended
    /// with a newline.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.check()?;
        let mut bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// A record read from bytes, and checked.
    pub fn parse(bytes: &[u8]) -> Result<EvalRecord> {
        let r: EvalRecord = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        r.check()?;
        Ok(r)
    }

    pub fn read(path: &Path) -> Result<EvalRecord> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        EvalRecord::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Writes the record to `path`. An existing file is never replaced,
    /// as with turbo-bench's records.
    pub fn write(&self, path: &Path) -> Result<()> {
        let bytes = self.to_bytes()?;
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("{}: {e}; a record is never replaced", path.display()))?;
        f.write_all(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cost::{Distribution, StageTimes};

    /// A well formed record whose per query scores are `scores`, named
    /// q00, q01, ... in order.
    pub(crate) fn sample(set: &str, scores: &[f64]) -> EvalRecord {
        let per_query: Vec<QueryNdcg> =
            scores.iter().enumerate().map(|(i, &s)| QueryNdcg { query: format!("q{i:02}"), ndcg_at_10: s }).collect();
        let spans = SpanScores { precision: 0.25, recall: 0.5, contained: 0.75 };
        EvalRecord {
            record_version: EVAL_RECORD_VERSION,
            recorded_at: "2026-10-12T08:30:00Z".into(),
            provenance: Provenance {
                commit: "0123456789abcdef0123456789abcdef01234567".into(),
                pushed_to: vec!["origin/main".into()],
                machine: Machine { arch: "x86_64".into(), host_cpu: "Example CPU".into(), os: "linux".into() },
                device: Device {
                    backend: "cpu".into(),
                    kind: "DEVICE_CPU".into(),
                    name: "Example CPU".into(),
                    vendor: "Example".into(),
                    driver_version: String::new(),
                    runtime_version: String::new(),
                    memory_total: 1 << 34,
                },
                bundle: BundleId {
                    model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
                    revision: "f".repeat(40),
                    manifest_sha256: "a".repeat(64),
                    artifact_sha256: "b".repeat(64),
                    tokenizer_sha256: "c".repeat(64),
                },
            },
            set: SetId { name: set.into(), sha256: "d".repeat(64) },
            pipeline: Pipeline {
                analyzer: vec!["sentences".into(), "tokens".into()],
                chunker: Chunker {
                    name: "turbo-chunker".into(),
                    config: vec![Setting { name: "max_tokens".into(), value: "256".into() }],
                },
            },
            metrics: Metrics {
                queries: scores.len() as u32,
                ndcg_at_10: mean(scores.iter().copied()),
                recall: RecallAtK { at_1: 0.1, at_5: 0.4, at_10: 0.6, at_100: 0.9 },
                span: Some(SpanMetrics {
                    queries: 2,
                    mean: SpanAtK { at_1: spans, at_5: spans, at_10: spans, at_100: spans },
                }),
                per_query,
            },
            cost: Cost {
                wall: StageTimes { analysis_ms: 1.5, chunking_ms: 2.0, embedding_ms: 300.25, retrieval_ms: 4.0 },
                tokens_embedded: 123_456,
                chunks: 789,
                chunk_bytes: Distribution { min: 12, p50: 800, p90: 1400, max: 2000 },
                chunk_tokens: Distribution { min: 3, p50: 180, p90: 250, max: 256 },
            },
        }
    }

    fn text(r: &EvalRecord) -> String {
        String::from_utf8(r.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn round_trip() {
        let r = sample("scifact", &[0.5, 0.25, 1.0 / 3.0]);
        let bytes = r.to_bytes().unwrap();
        assert!(bytes.ends_with(b"}\n"));
        assert!(bytes.starts_with(b"{\n  \"record_version\": 1,\n"));
        let back = EvalRecord::parse(&bytes).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.to_bytes().unwrap(), bytes);

        let mut none = r.clone();
        none.metrics.span = None;
        assert!(text(&none).contains("\"span\": null"));
        assert_eq!(EvalRecord::parse(none.to_bytes().unwrap().as_slice()).unwrap(), none);
    }

    #[test]
    fn write_and_read_a_file() {
        let dir = std::env::temp_dir().join(format!("turbo-eval-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("record.json");
        let _ = fs::remove_file(&path);
        let r = sample("scifact", &[0.5]);
        r.write(&path).unwrap();
        assert_eq!(EvalRecord::read(&path).unwrap(), r);
        assert!(r.write(&path).is_err(), "a record is never replaced");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_fields_are_refused() {
        let t = text(&sample("scifact", &[0.5]));
        let top = t.replacen("\"record_version\": 1,", "\"record_version\": 1,\n  \"extra\": 0,", 1);
        assert!(EvalRecord::parse(top.as_bytes()).unwrap_err().contains("unknown field `extra`"));
        let nested = t.replacen("\"arch\": \"x86_64\",", "\"arch\": \"x86_64\", \"cores\": 8,", 1);
        assert!(EvalRecord::parse(nested.as_bytes()).unwrap_err().contains("unknown field `cores`"));
        let deep = t.replacen("\"contained\": 0.75", "\"contained\": 0.75, \"f1\": 0.1", 1);
        assert!(EvalRecord::parse(deep.as_bytes()).unwrap_err().contains("unknown field `f1`"));
    }

    #[test]
    fn missing_fields_are_refused() {
        let t = text(&sample("scifact", &[0.5]));
        let cut = |field: &str| -> String {
            t.lines().filter(|l| !l.trim_start().starts_with(&format!("\"{field}\":"))).collect::<Vec<_>>().join("\n")
        };
        for field in ["recorded_at", "tokens_embedded", "commit", "embedding_ms"] {
            let err = EvalRecord::parse(cut(field).as_bytes()).unwrap_err();
            assert!(err.contains(&format!("missing field `{field}`")), "{field}: {err}");
        }
        // span may be null, but not absent.
        let mut none = sample("scifact", &[0.5]);
        none.metrics.span = None;
        let t = text(&none).replace("\"span\": null,\n", "");
        assert!(EvalRecord::parse(t.as_bytes()).unwrap_err().contains("missing field `span`"));
    }

    #[test]
    fn inconsistent_records_are_refused() {
        let good = sample("scifact", &[0.5, 0.75]);
        let mut r = good.clone();
        r.metrics.ndcg_at_10 = 0.7;
        assert!(r.check().unwrap_err().contains("per query mean"));
        let mut r = good.clone();
        r.metrics.per_query.reverse();
        assert!(r.check().is_err());
        let mut r = good.clone();
        r.metrics.queries = 3;
        assert!(r.check().is_err());
        let mut r = good.clone();
        r.set.sha256 = "D".repeat(64);
        assert!(r.check().is_err());
        let mut r = good.clone();
        r.recorded_at = "2026-10-12 08:30:00".into();
        assert!(r.check().is_err());
        let mut r = good.clone();
        r.cost.chunk_bytes.p90 = 5000;
        assert!(r.check().is_err());
        let mut r = good.clone();
        r.pipeline.chunker.config.push(Setting { name: "vocab".into(), value: "/home/someone/vocab.txt".into() });
        assert!(r.check().unwrap_err().contains("host path"));
        let mut r = good;
        r.metrics.recall.at_5 = 1.5;
        assert!(r.check().is_err());
    }
}
