//! The receipt a run writes beside its outputs (`receipt.json`): what was
//! run, on what data and encoder, with what settings, what it scored
//! against the majority baseline, and how long each stage took. Two runs
//! of one command at one commit on one device give the same receipt but
//! for `timing_s`, `host` and `unix_time`.

use serde::{Deserialize, Serialize};
use turbo::manifest::HeadSpec;

use crate::corpus::{CorpusRecord, FileRecord};
use crate::eval::{Comparison, Metrics};

pub const SCHEMA: &str = "turboembed.trainer.receipt/1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema: String,
    /// The command line, as run.
    pub command: Vec<String>,
    /// The commit the binary was built from, as given by `--commit` or
    /// `git rev-parse HEAD`, or "unknown".
    pub commit: String,
    pub host: String,
    pub unix_time: u64,
    pub corpus: CorpusRecord,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<EncoderRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<HeadSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<ConfigRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingRecord>,
    pub eval: EvalRecord,
    pub timing_s: Timing,
    /// The files written, with their digests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<FileRecord>,
}

/// The frozen encoder the head was trained on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderRecord {
    /// The bundle's model id and revision, from its manifest.
    pub model: String,
    pub revision: String,
    /// The SHA-256 of the bundle's `manifest.json`, which pins every file.
    pub manifest_sha256: String,
    pub hidden: usize,
    /// The encoder layers run (`architecture.layers`): a bundle cut at a
    /// middle layer differs from the full one only here and in its
    /// manifest digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<u32>,
    /// Pieces a row holds beside its special tokens.
    pub width: usize,
    pub backend: String,
    pub device: String,
    pub precision: String,
}

/// The training settings (`turbo_trainer::Config`) and the rows dropped
/// before training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigRecord {
    pub seed: u64,
    pub epochs: u32,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub clip: f64,
    pub warmup: f64,
    pub patience: u32,
    /// How far under the best a dev score may fall and hold level.
    #[serde(default)]
    pub tolerance: f64,
    /// Threads a batch ran on, 0 for every core; the weights do not
    /// depend on it.
    #[serde(default)]
    pub threads: usize,
    /// The first this many training sentences, or every one when 0.
    pub max_train: usize,
    /// Dev sentences left out of early stopping because training never
    /// gave one of their labels.
    pub dev_left_out: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRecord {
    pub train_loss: Vec<f64>,
    pub dev: Vec<f64>,
    pub best_epoch: usize,
    /// Wall seconds each epoch took, dev scoring included.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub epoch_seconds: Vec<f64>,
}

/// Scores on the test split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalRecord {
    pub split: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<Metrics>,
    pub baseline: Metrics,
    /// The model against the baseline, by paired bootstrap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<Comparison>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub read: f64,
    /// Reading the train and dev words through the encoder: exactly
    /// `feature_words` words.
    pub features: f64,
    pub train: f64,
    pub eval: f64,
    #[serde(default)]
    pub feature_words: u64,
}

impl Timing {
    /// Words read through the encoder a second, or None when none were.
    pub fn words_per_s(&self) -> Option<f64> {
        (self.feature_words > 0 && self.features > 0.0).then(|| self.feature_words as f64 / self.features)
    }
}

/// This machine's name, from `HOSTNAME` or `/etc/hostname`, or "unknown".
pub fn host() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok().map(|h| h.trim().to_owned()))
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Seconds since the Unix epoch.
pub fn unix_time() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}
