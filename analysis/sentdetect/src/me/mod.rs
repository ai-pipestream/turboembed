//! The maxent sentence detector, OpenNLP's `SentenceDetectorME`: the
//! model ([`SentenceModel`]) and the annotator that runs it
//! ([`SentenceDetector`]).

pub mod detector;
pub mod model;

pub use detector::{Candidate, Decision, Scratch, SentenceDetector};
pub use model::SentenceModel;
