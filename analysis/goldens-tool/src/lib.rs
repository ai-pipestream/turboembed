//! The driver that makes the golden layer files in `analysis/goldens/data`:
//! it builds the generator's container, fetches the models and checks
//! their hashes, runs the container and compares what it wrote with the
//! checked in files, and checks the decision point dumps written beside the
//! goldens. `fuzz` runs the generator over a seeded corpus and compares
//! the Rust annotators with what it wrote. The binary `turbo-goldens` is
//! its command line.

pub mod command;
pub mod compare;
pub mod docker;
pub mod dump;
#[cfg(feature = "fuzz")]
pub mod fuzz;
pub mod models;
pub mod output;
pub mod paths;
pub mod recipe;
pub mod sha512;
