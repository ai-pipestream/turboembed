//! Late pooling through the C interface on the CPU backend this build
//! links: turbo_embed_write_ranges over the small BERT, with a range of a
//! whole row against that row's own vector, ranges of every kind against
//! the encoder written plainly in f64, the refusals the header lists, the
//! shape of a range run's result, the session's max_ranges and cpu_threads
//! fields, the same write on a static model, and the write that reads
//! the last layers each (turbo_embed_write_layer_ranges).

#[path = "../common/mod.rs"]
mod common;

mod layers;
mod overlap;
mod plain;
mod refusals;
mod session_fields;
mod shape;
mod static_model;
mod stride;
mod support;
mod whole_row;
