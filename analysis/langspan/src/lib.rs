//! Model-free Unicode script evidence with exact source spans.
//!
//! Script identification cannot distinguish languages using the same script.
//! Runs therefore report language `und` with zero language confidence until
//! the trained language head in #1525 is available. OpenNLP language models
//! and the n-gram language decoder have been removed under #1511.

mod detector;
mod script_only;
pub mod scripts;
pub use detector::NgramDetector;
pub use script_only::ScriptDetector;
