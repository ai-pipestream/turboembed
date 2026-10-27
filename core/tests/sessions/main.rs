//! Sessions, writes, runs and results through the C interface, on the CPU
//! backend this build links: what each call resolves and reports, every
//! refusal, the session's state, and the encoder's options checked against
//! the same arithmetic written plainly in f64. tests/conformance.rs holds
//! the vectors to the upstream reference; tests/allocations.rs holds the
//! run path to its allocation count.

#[path = "../common/mod.rs"]
mod common;

mod capability;
mod plain;
mod ranges;
mod results;
mod roberta;
mod session;
mod state;
mod support;
mod weights;
mod write_text;
mod write_tokens;
