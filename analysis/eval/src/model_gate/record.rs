use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::record::is_hex;

use super::error::Result;
use super::{
    HeldOutSet, MARGIN_POINTS, MODEL_GATE_RECORD_VERSION, Measure, ModelGate, ModelGateError, ModelId, PairedUnit,
    RESAMPLES, SEED, Task, gate_models,
};

/// A record's version, read before the rest of it; its other fields are
/// ignored.
#[derive(Deserialize)]
struct RecordVersion {
    record_version: u32,
}

/// The record of one gate run, kept beside the model card.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGateRecord {
    pub record_version: u32,
    pub task: Task,
    pub measure: Measure,
    pub set: HeldOutSet,
    pub candidate: ModelId,
    pub reference: ModelId,
    pub resamples: u32,
    pub seed: i64,
    pub result: ModelGate,
    /// Both models' counts on each held-out unit, in the order of the set.
    pub units: Vec<PairedUnit>,
}

impl ModelGateRecord {
    /// Runs the gate on `units` with [`RESAMPLES`] and [`SEED`] and
    /// records it.
    pub fn new(
        task: Task,
        set: HeldOutSet,
        candidate: ModelId,
        reference: ModelId,
        units: Vec<PairedUnit>,
    ) -> Result<ModelGateRecord> {
        let measure = task.measure();
        let result = gate_models(measure, &units, RESAMPLES, SEED)?;
        Ok(ModelGateRecord {
            record_version: MODEL_GATE_RECORD_VERSION,
            task,
            measure,
            set,
            candidate,
            reference,
            resamples: RESAMPLES,
            seed: SEED,
            result,
            units,
        })
    }

    /// Checks the version; that the names are not empty and the digests
    /// are 64 lower case hex; that the measure is the task's; that the
    /// record was drawn with [`RESAMPLES`] and [`SEED`]; that the scoring
    /// is the task's ([`Task::scoring`]); for tags, that each model's
    /// units count the held-out tokens less those left out, that each tag
    /// mapping maps a non-empty held-out tag once to another non-empty tag,
    /// and that a lemmatizer's record has no mapping; for typed spans, that
    /// a type is scored and none twice, that each renaming renames a type
    /// not scored as written, once, to a type scored, that each reference
    /// input maps a held-out tag, once, to another tag, and that each
    /// model's units count the held-out spans scored as references; for
    /// labels, that no sentence was skipped, that there is one unit per
    /// held-out sample and each scores one sample, and that each mapping
    /// has two labels that differ and is listed once; and that the result
    /// is what the gate gives on the recorded units, to the bit.
    pub fn check(&self) -> Result<()> {
        if self.record_version != MODEL_GATE_RECORD_VERSION {
            return Err(ModelGateError::Version { found: self.record_version });
        }
        let names =
            [("set.name", &self.set.name), ("candidate.id", &self.candidate.id), ("reference.id", &self.reference.id)];
        if let Some((field, value)) = names.into_iter().find(|(_, v)| v.is_empty()) {
            return Err(ModelGateError::Field { field, value: value.clone() });
        }
        let digests = [
            ("set.corpus_sha256", &self.set.corpus_sha256),
            ("set.samples_sha256", &self.set.samples_sha256),
            ("candidate.sha256", &self.candidate.sha256),
            ("reference.sha256", &self.reference.sha256),
        ];
        if let Some((field, value)) = digests.into_iter().find(|(_, v)| !is_hex(v, 64)) {
            return Err(ModelGateError::Field { field, value: value.clone() });
        }
        if self.resamples != RESAMPLES || self.seed != SEED {
            return Err(ModelGateError::Parameters { resamples: self.resamples, seed: self.seed });
        }
        if self.measure != self.task.measure() {
            return Err(ModelGateError::MeasureMismatch { task: self.task, measure: self.measure });
        }
        self.set.check(self.task, &self.units)?;
        let again = gate_models(self.measure, &self.units, self.resamples, self.seed)?;
        if again != self.result {
            return Err(ModelGateError::ResultMismatch);
        }
        Ok(())
    }

    /// The record as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a gate record serializes");
        s.push('\n');
        s
    }

    /// Reads and checks a record. The version is read first, alone, so a
    /// record of another version is refused with
    /// [`ModelGateError::Version`] whatever its other fields hold.
    pub fn read(path: &Path) -> Result<ModelGateRecord> {
        let json =
            |e: serde_json::Error| ModelGateError::Json { path: path.display().to_string(), message: e.to_string() };
        let bytes = fs::read(path)
            .map_err(|e| ModelGateError::Io { path: path.display().to_string(), message: e.to_string() })?;
        let version: RecordVersion = serde_json::from_slice(&bytes).map_err(json)?;
        if version.record_version != MODEL_GATE_RECORD_VERSION {
            return Err(ModelGateError::Version { found: version.record_version });
        }
        let record: ModelGateRecord = serde_json::from_slice(&bytes).map_err(json)?;
        record.check()?;
        Ok(record)
    }

    /// Checks and writes the record to `path`, which must not exist yet.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.check()?;
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| ModelGateError::Io { path: path.display().to_string(), message: e.to_string() })?;
        f.write_all(self.to_json().as_bytes())
            .map_err(|e| ModelGateError::Io { path: path.display().to_string(), message: e.to_string() })
    }
}

impl fmt::Display for ModelGateRecord {
    /// The two scores with their intervals, the difference with its
    /// interval, and the verdict.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.result;
        let measure = match self.measure {
            Measure::FMeasure => "F-measure",
            Measure::Accuracy => "accuracy",
        };
        writeln!(f, "{:?} on {}, {} units, {measure} in points:", self.task, self.set.name, self.units.len())?;
        for (who, id, s) in [("candidate", &self.candidate, r.candidate), ("reference", &self.reference, r.reference)] {
            writeln!(f, "  {who} {}: {:.2} (95% interval {:.2} to {:.2})", id.id, s.score, s.low, s.high)?;
        }
        writeln!(
            f,
            "  difference: {:+.2} (95% interval {:+.2} to {:+.2})",
            r.difference.mean, r.difference.low, r.difference.high
        )?;
        if r.pass {
            write!(
                f,
                "PASS: the difference's interval starts at {:+.2}, not below -{MARGIN_POINTS}: a loss of more than {MARGIN_POINTS} points is ruled out",
                r.difference.low
            )
        } else {
            write!(
                f,
                "FAIL: the difference's interval starts at {:+.2}, below -{MARGIN_POINTS}: a loss of more than {MARGIN_POINTS} points is not ruled out",
                r.difference.low
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_gate::tests::{labels, labels_of, language_record, record};

    #[test]
    fn a_record_of_another_version_is_refused_for_its_version_before_its_fields_are_read() {
        assert_eq!(MODEL_GATE_RECORD_VERSION, 3);
        let units = labels(40, 20);
        let r = language_record(units.clone(), labels_of(&units));
        let json = r.to_json();
        assert!(json.starts_with("{\n  \"record_version\": 3,\n"), "{json}");
        let dir = std::env::temp_dir().join(format!("turbo-eval-model-gate-version-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gate.json");
        let read = |text: &str| {
            fs::write(&path, text).unwrap();
            ModelGateRecord::read(&path)
        };
        assert_eq!(read(&json).unwrap(), r);
        // A record of version 2 holds no language detector; a later one
        // may hold a task this build does not know. Both are refused for
        // their version, not as JSON this build cannot parse.
        let two = json.replacen("\"record_version\": 3", "\"record_version\": 2", 1);
        assert_eq!(read(&two), Err(ModelGateError::Version { found: 2 }));
        let later = json
            .replacen("\"record_version\": 3", "\"record_version\": 4", 1)
            .replacen("\"language_detector\"", "\"a_later_task\"", 1)
            .replacen("\"labels\": {", "\"a_later_scoring\": {", 1);
        assert_eq!(read(&later), Err(ModelGateError::Version { found: 4 }));
        // At this version, the same values are refused as JSON.
        let unknown = later.replacen("\"record_version\": 4", "\"record_version\": 3", 1);
        let Err(ModelGateError::Json { message, .. }) = read(&unknown) else { panic!("{unknown}") };
        assert!(message.contains("unknown variant `a_later_task`"), "{message}");
        // No version at all is not a record.
        assert!(matches!(read("{\"task\": \"tokenizer\"}"), Err(ModelGateError::Json { .. })));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_record_round_trips_and_is_checked_against_its_units() {
        let r = record();
        assert_eq!(r.measure, Measure::FMeasure);
        r.check().unwrap();
        let dir = std::env::temp_dir().join(format!("turbo-eval-model-gate-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gate.json");
        let _ = fs::remove_file(&path);

        // An invalid record is refused before creating a file on disk.
        let mut invalid_write = r.clone();
        invalid_write.candidate.id.clear();
        assert!(matches!(invalid_write.write(&path), Err(ModelGateError::Field { field: "candidate.id", .. })));
        assert!(!path.exists());

        r.write(&path).unwrap();
        assert!(matches!(r.write(&path), Err(ModelGateError::Io { .. })), "a record is never replaced");
        assert_eq!(ModelGateRecord::read(&path).unwrap(), r);
        fs::write(&path, "{}").unwrap();
        assert!(matches!(ModelGateRecord::read(&path), Err(ModelGateError::Json { .. })));
        fs::remove_dir_all(&dir).unwrap();

        let mut changed = r.clone();
        changed.units[0].candidate.true_positives += 1;
        assert_eq!(changed.check(), Err(ModelGateError::ResultMismatch));
        let mut wrong = r.clone();
        wrong.measure = Measure::Accuracy;
        assert!(matches!(wrong.check(), Err(ModelGateError::MeasureMismatch { .. })));
        let mut old = r.clone();
        old.record_version = 0;
        assert_eq!(old.check(), Err(ModelGateError::Version { found: 0 }));
        // A record drawn with fewer resamples or another seed is refused,
        // even though its result is what those parameters give.
        for (resamples, seed) in [(1, SEED), (RESAMPLES, SEED + 1)] {
            let mut other = r.clone();
            other.resamples = resamples;
            other.seed = seed;
            other.result = gate_models(other.measure, &other.units, resamples, seed).unwrap();
            assert_eq!(other.check(), Err(ModelGateError::Parameters { resamples, seed }));
        }
        let mut unnamed = r.clone();
        unnamed.candidate.id.clear();
        assert!(matches!(unnamed.check(), Err(ModelGateError::Field { field: "candidate.id", .. })));
        let mut upper = r.clone();
        upper.set.samples_sha256 = "B".repeat(64);
        assert!(matches!(upper.check(), Err(ModelGateError::Field { field: "set.samples_sha256", .. })));
        let mut short = r.clone();
        short.reference.sha256 = "c".repeat(63);
        assert!(matches!(short.check(), Err(ModelGateError::Field { field: "reference.sha256", .. })));
        let json = r.to_json().replace("\"seed\": 154,", "\"seed\": 154,\n  \"extra\": 1,");
        assert!(serde_json::from_str::<ModelGateRecord>(&json).is_err(), "unknown fields are refused");
        let text = r.to_string();
        assert!(text.contains("candidate trained:"), "{text}");
        assert_eq!(text.contains("PASS: "), r.result.pass, "{text}");
        assert_eq!(text.contains("FAIL: "), !r.result.pass, "{text}");
    }
}
