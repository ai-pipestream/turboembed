//! The driver that makes the golden layer files in `analysis/goldens/data`:
//! it builds the generator's container, fetches the models and checks
//! their hashes, runs the container and compares what it wrote with the
//! checked in files. The binary `turbo-goldens` is its command line.

pub mod command;
pub mod compare;
pub mod docker;
pub mod models;
pub mod output;
pub mod paths;
pub mod recipe;
pub mod sha512;
