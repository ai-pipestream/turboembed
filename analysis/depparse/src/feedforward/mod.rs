//! Feedforward arc-standard parsing and deterministic AdaGrad training.
//! The Java reference uses 14/14/8 features; the expanded template uses 18/18/12.
mod context;
mod model;
mod normalize;
mod package;
mod parser;
mod trainer;
mod wire;
pub use context::{FeatureLayout, FeedforwardContext, FeedforwardFeatureIds};
pub use model::{Activation, FeedforwardDependencyModel, NeuralWeights, Vocabulary};
pub use parser::FeedforwardDependencyParser;
pub use trainer::{FeedforwardDependencyTrainer, FeedforwardSettings};
