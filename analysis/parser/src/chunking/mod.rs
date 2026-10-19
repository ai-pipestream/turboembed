//! Bottom-up shift-reduce chunking constituency parser and its context generators
//! (`opennlp.tools.parser.chunking.*`). The event stream and the trainer are
//! `turbo_train_components::parser::chunking`.

mod context;
mod parser;

pub use context::{BuildContextGenerator, CheckContextGenerator};
pub use parser::Parser;

/// Top-start outcome (`AbstractBottomUpParser.TOP_START = "S-TOP"`).
pub const TOP_START: &str = "S-TOP";
