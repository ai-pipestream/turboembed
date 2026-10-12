//! The maxent sentence detector, OpenNLP's `SentenceDetectorME`: the
//! model ([`SentenceModel`]) and the annotator that runs it
//! ([`SentenceDetector`]), with [`DetectorStep`] to run it in a pipeline.

pub mod detector;
pub mod model;
pub mod step;

pub use detector::{Candidate, Decision, Scratch, SentenceDetector};
pub use model::SentenceModel;
pub use step::DetectorStep;
