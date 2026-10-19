//! Feedforward neural part-of-speech tagger (`#205`), ported from OpenNLP's
//! `FeedforwardPOSContext`, `FeedforwardPOSModel`, `FeedforwardPOSTagger`, and
//! `FeedforwardPOSTrainer`.

mod context;
mod model;
mod tagger;
mod trainer;

#[cfg(test)]
mod tests;

pub use self::context::{ABSENT, FeedforwardPosContext, PRETRAINED_SLOTS, SLOTS, UNKNOWN};
pub use self::model::FeedforwardPosModel;
pub use self::tagger::FeedforwardPosTagger;
pub use self::trainer::{FeedforwardPosTrainer, FeedforwardSettings};
