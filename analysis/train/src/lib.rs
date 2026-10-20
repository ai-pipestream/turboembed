//! Training OpenNLP 3.x maxent models: the event side of
//! `opennlp.tools.ml`, ported so that what the JVM's trainers consume,
//! this crate consumes identically.
//!
//! - [`event`][]: `Event`, the `FileEventStream` and `RealValueFileEventStream`
//!   formats, and the CRC-32C event checksum a trained model's manifest
//!   records (`Training-Eventhash`).
//! - [`java_string`][]: `JavaString` (from `turbo-maxent`), a predicate or
//!   an outcome as Java holds it, which may hold half a surrogate pair (the
//!   tokenizer's features are cut at UTF-16 units, and a lemma class edits
//!   the units of a word), kept in the modified UTF-8 the model files use.
//! - [`index`][]: `OnePassDataIndexer`, `TwoPassDataIndexer` and
//!   `OnePassRealValueDataIndexer`, which number outcomes and predicates,
//!   apply the cutoff, and sort and merge events, quirks included.
//! - [`params`][]: `TrainingParameters` with Java's case-insensitive keys and
//!   typed values, the parameters file format, `AlgorithmType` and
//!   `TrainerFactory`'s trainer types.
//! - [`monitor`][]: training progress monitors and stop criteria.
//! - [`gis`][]: `GISTrainer`, the model as `GISModel` holds it, and the
//!   manifest entries `AbstractEventTrainer` reports.
//! - [`qn`][]: `QNTrainer`, the L-BFGS trainer: `QNMinimizer`, its line
//!   searches, the L1 and L2 penalties and the threaded negative log
//!   likelihood, and the model as `QNModel` holds it.
//! - [`perceptron`][]: `PerceptronTrainer` and
//!   `SimplePerceptronSequenceTrainer`, with averaging and the tolerance
//!   stop, and the model as `PerceptronModel` holds it.
//! - [`naive_bayes`][]: `NaiveBayesTrainer` and `NaiveBayesModel`, with
//!   Lidstone smoothing (`delta = 0.05`).
//! - [`liblinear`][]: the liblinear trainer of the OpenNLP addons, with
//!   liblinear-java's logistic regression solvers, the model it returns
//!   and its serialized form.
//! - [`trainer`][]: `TrainerFactory.getEventTrainer` and
//!   `getEventModelSequenceTrainer`, the trainer the `Algorithm` parameter
//!   names, and the model it trained.
//! - [`writer`][]: `GenericModelWriter` for a GIS, QN, perceptron or naive
//!   Bayes model, the bytes of a `.model` entry, chunked strings included.
//! - [`package`][]: the `.bin` file, `BaseModel.serialize`: a zip archive
//!   with the `manifest.properties` entry, typed.
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
//! - `GISTrainer` with `Threads` above one sums the log likelihood and the
//!   expectations per chunk of events and then across chunks, so its
//!   results differ in the last bits from one thread's; the chunking is
//!   ported and the sums are in the JVM's order.
//! - `ModelParameterChunker` measures the chunks of a string over 65535
//!   bytes in standard UTF-8 and writes them in modified UTF-8, and stops
//!   at a lone surrogate, dropping the rest of the string ([`writer`][]).
//! - `QNTrainer` reports nothing to the progress monitor and reads no
//!   stop criterion, and `QNMinimizer` aliases its stored corrections once
//!   it starts discarding the oldest ([`qn`][]).
//! - `PerceptronTrainer`'s `UseSkippedAveraging` only turns averaging on,
//!   its monitor hears of iterations 1 to 9 and every tenth, and a
//!   perceptron model with every parameter zero cannot be written
//!   ([`perceptron`][], [`writer`][]).

pub mod context;
pub mod crc32c;
pub mod error;
pub mod event;
pub mod gis;
pub mod index;
pub mod liblinear;
pub mod monitor;
pub mod naive_bayes;
pub mod package;
pub mod params;
pub mod perceptron;
pub mod qn;
pub mod trainer;
pub mod writer;

pub use turbo_maxent::java_string;

pub use context::{BasicContextGenerator, Separator};
pub use crc32c::{Crc32c, crc32c};
pub use error::{Error, Result};
pub use event::{Event, EventChecksum, FileEventStream, RealValueFileEventStream};
pub use gis::{GisModel, GisParams, Smoothing, TrainingReport};
pub use index::{DataIndex, IndexParams, IndexerKind};
pub use java_string::JavaString;
pub use monitor::{
    DefaultProgressMonitor, IterDeltaAccuracyUnderTolerance, LogLikelihoodThresholdBreached, ProgressMonitor,
    StopCriteria, TrainingConfiguration, TrainingMeasure,
};
pub use naive_bayes::NaiveBayesModel;
pub use package::{Manifest, ModelPackage, TimeZone, sentence_model_package};
pub use params::{Algorithm, Key, TrainerType, TrainingParameters, Value};
pub use perceptron::{PerceptronIteration, PerceptronModel, PerceptronParams, Sequence, SequenceSource};
pub use qn::{QnIteration, QnModel, QnParams, QnStop};
pub use trainer::TrainedModel;
pub use writer::{
    WritableModel, write_gis, write_gis_writable, write_naive_bayes, write_perceptron, write_qn, write_qn_writable,
};
