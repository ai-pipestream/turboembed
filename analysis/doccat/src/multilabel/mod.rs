//! Multi-label document categorization: a document or span scores every
//! known category label independently, and every label clearing its own
//! decision threshold is emitted, so an item gets zero, one, or many
//! labels. Zero labels is the categorizer abstaining (the layer item's
//! best category is [`turbo_analysis::Tag::NONE`]), never a forced argmax.
//!
//! A [`MultiLabelDoccatModel`] holds one linear head per label, trained
//! one-vs-rest ([`train`]): per-label weights and a bias, sigmoided into
//! a score in `0.0..=1.0` (the scores do not sum to 1.0). Each label
//! carries its own threshold in the validated open interval `(0.0, 1.0)`
//! (default 0.5); [`MultiLabelCategorizer`] emits every label whose score
//! reaches it, optionally capped by `max_labels` and floored by
//! `min_score`.
//!
//! [`ThresholdCalibrator`] fits the per-label thresholds on a held-out
//! validation set by sweeping candidate cut points and maximizing each
//! label's F_beta (beta configurable, default 1.0), optionally under a
//! minimum-precision floor. The calibrated thresholds serialize inside
//! the model entry and round-trip with the rest of the model.

pub mod bytes;
pub mod calibrate;
pub mod categorizer;
pub mod error;
pub mod head;
pub mod metrics;
pub mod model;
pub mod train;
pub mod validate;

/// The `Component-Name` of a multi-label model entry.
pub const COMPONENT_NAME: &str = "MultiLabelDoccat";

pub use bytes::crc32c;
pub use calibrate::{ThresholdCalibrator, f_beta, macro_f1, precision_recall};
pub use categorizer::{MultiLabelCategorizer, MultiLabelSpanCategorizer};
pub use error::MultiLabelError;
pub use head::LinearHead;
pub use metrics::{MultiLabelReport, Prf, evaluate};
pub use model::{FORMAT_VERSION, FORMAT_VERSION_V1, MODEL_ENTRY, MODEL_TYPE, MultiLabelDoccatModel, TAGSET_NAME};
pub use train::{MultiLabelSample, MultiLabelTrainingParameters, train_multilabel};
