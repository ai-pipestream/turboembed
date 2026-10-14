//! Training OpenNLP 3.x maxent models: the event side of
//! `opennlp.tools.ml`, ported so that what the JVM's trainers consume,
//! this crate consumes identically.
//!
//! - [`event`][]: `Event`, the `FileEventStream` and `RealValueFileEventStream`
//!   formats, and the CRC-32C event checksum a trained model's manifest
//!   records (`Training-Eventhash`).
//! - [`java_string`][]: `JavaString`, a predicate as Java holds it, which may
//!   be half a surrogate pair (the tokenizer's features are cut at UTF-16
//!   units), kept in the modified UTF-8 the model files use.
//! - [`index`][]: `OnePassDataIndexer`, `TwoPassDataIndexer` and
//!   `OnePassRealValueDataIndexer`, which number outcomes and predicates,
//!   apply the cutoff, and sort and merge events, quirks included.
//! - [`params`][]: `TrainingParameters` with Java's case-insensitive keys and
//!   typed values, the parameters file format, `AlgorithmType` and
//!   `TrainerFactory`'s trainer types.
//! - [`monitor`][]: training progress monitors and stop criteria.
//! - [`context`][]: `BasicContextGenerator`.
//!
//! Sample types (sentences, tokens, tags) and the event streams that make
//! events from them belong to `analysis/formats`; this crate starts at
//! events. Numbers are formatted and parsed as Java does (`turbo-jvm`),
//! predicate labels are sorted by Java's string order, and `f32` is used
//! where OpenNLP uses `float`.
//!
//! OpenNLP behaviours a reader may not expect, all kept and tested:
//!
//! - `FileEventStream` throws on an empty line ([`Error::EmptyEventLine`]).
//! - The indexers number the outcome of an event before dropping it for
//!   having no predicate above the cutoff, so outcome labels can name an
//!   outcome no indexed event has.
//! - The indexers sort events by the values of their full context, read at
//!   the positions of the predicates that survived the cutoff, and
//!   `OnePassRealValue` keeps those full arrays.
//! - `TwoPass` cannot index a string over 65535 bytes of modified UTF-8
//!   ([`Error::StringTooLong`]).
//! - `TrainingParameters.serialize` fails on a parameter put as a number
//!   or a boolean ([`Error::SerializeType`]).

pub mod context;
pub mod crc32c;
pub mod error;
pub mod event;
pub mod index;
pub mod java_string;
pub mod monitor;
pub mod params;

pub use context::{BasicContextGenerator, Separator};
pub use crc32c::{Crc32c, crc32c};
pub use error::{Error, Result};
pub use event::{Event, EventChecksum, FileEventStream, RealValueFileEventStream};
pub use index::{DataIndex, IndexParams, IndexerKind};
pub use java_string::JavaString;
pub use monitor::{
    DefaultProgressMonitor, IterDeltaAccuracyUnderTolerance, LogLikelihoodThresholdBreached, ProgressMonitor,
    StopCriteria, TrainingConfiguration, TrainingMeasure,
};
pub use params::{Algorithm, Key, TrainerType, TrainingParameters, Value};
