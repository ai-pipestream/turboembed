//! The receipt a run writes beside its outputs (`receipt.json`): what was
//! run, on what data and encoder, with what settings, what it scored
//! against the majority baseline, and how long each stage took. Two runs
//! of one command at one commit on one device give the same receipt but
//! for `timing_s` and `unix_time`.

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
    pub unix_time: u64,
    pub corpus: CorpusRecord,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<EncoderRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<HeadSpec>,
    /// The peer system scored in place of a head, for a `peer` receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<crate::peer::PeerRecord>,
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

/// The encoder the head was trained on, frozen or tuned by the recipe.
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
    /// The gain over the last counted dev score that resets patience.
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
    /// Temperature used to resample a multi-treebank training split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRecord {
    pub train_loss: Vec<f64>,
    pub dev: Vec<f64>,
    /// The epoch (counted from 1, as the training log counts) whose
    /// weights were kept.
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
    /// Case-insensitive label-name dictionary matcher (`train-open-span`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<Metrics>,
    /// Seen and zero-shot held-out evaluation slices (`train-open-span`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_span: Option<OpenSpanEvalRecord>,
    /// The model against the baseline, by paired bootstrap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<Comparison>,
    /// A parser's unlabelled attachment, model and baseline; its labelled
    /// attachment is the tag accuracy above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uas: Option<Uas>,
    /// For `train-sentences`, the rule-based sentence splitter baseline
    /// (`analysis/sentdetect`) and its paired bootstrap comparison with
    /// the model, beside the newline baseline in `baseline`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rules: Option<BaselineEval>,
    /// A previous head of the same task on the same test split (named by
    /// its directory and decode), and the model's paired bootstrap against
    /// it, for a run given `--against`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub against: Option<BaselineEval>,
    /// Per-language scores for a multi-treebank run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub by_language: Vec<LanguageEval>,
}

/// One language's model, baseline and paired bootstrap evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageEval {
    pub language: String,
    pub model: Option<Metrics>,
    pub baseline: Metrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<Comparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uas: Option<Uas>,
}

impl EvalRecord {
    /// The test split's record of a baseline alone (the `baseline`
    /// command): no model and no comparison.
    pub fn baseline_only(baseline: Metrics) -> Self {
        EvalRecord {
            split: "test".into(),
            model: None,
            baseline,
            dictionary: None,
            open_span: None,
            comparison: None,
            uas: None,
            rules: None,
            against: None,
            by_language: Vec::new(),
        }
    }
}

/// A named deterministic baseline on the test split and its paired
/// bootstrap comparison with the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineEval {
    pub name: String,
    pub baseline: Metrics,
    pub comparison: Comparison,
    /// For a baseline read from files (`--against`), each file and its
    /// digest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileRecord>,
}

/// Seen-type and zero-shot held-out-type evaluation slices for an open-span run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenSpanEvalRecord {
    pub seen_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub held_out_types: Vec<String>,
    pub seen_model: Metrics,
    pub seen_baseline: Metrics,
    pub seen_dictionary: Metrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_model: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_baseline: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_dictionary: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary_comparison: Option<Comparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_comparison: Option<Comparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_dictionary_comparison: Option<Comparison>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<OpenSpanDatasetEval>,
}

/// One corpus's independent metrics within a multi-corpus open-span run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenSpanDatasetEval {
    pub corpus_id: String,
    pub model: Metrics,
    pub baseline: Metrics,
    pub dictionary: Metrics,
    pub comparison: Comparison,
    pub dictionary_comparison: Comparison,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_model: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_baseline: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_dictionary: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_comparison: Option<Comparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_dictionary_comparison: Option<Comparison>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Uas {
    pub model: f64,
    pub baseline: f64,
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
        (self.feature_words > 0 && self.features > 0.0).then_some(self.feature_words as f64 / self.features)
    }
}

/// Seconds since the Unix epoch.
pub fn unix_time() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}
