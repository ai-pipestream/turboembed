//! ConTEB's COVID-QA set: questions over research papers cut into
//! chunks, each judged against the one chunk that answers it.
//!
//! The tables are published as Parquet, which nothing in the workspace
//! reads. Fetching gets the Parquet files (once their digests are
//! recorded); the reader reads each one's rows as JSON Lines beside it,
//! with the same name and `.jsonl`, which have to be written from the
//! Parquet outside this crate. The dataset card states no licence, so
//! the set is refused until one is recorded here.

use std::path::Path;

use super::{Digest, RemoteFile, Set, Source, SpanUnit};
use crate::Result;
use crate::read::conteb;
use crate::set::LoadedSet;

pub const SET: Set = Set {
    name: "conteb-covid-qa",
    license: "",
    source: Source::Files(&[
        RemoteFile {
            path: "documents/train-00000-of-00001.parquet",
            url: "https://huggingface.co/datasets/illuin-conteb/covid-qa/resolve/df9373821efad895edf07b50a31ecd652b679b93/documents/train-00000-of-00001.parquet",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: "queries/train-00000-of-00001.parquet",
            url: "https://huggingface.co/datasets/illuin-conteb/covid-qa/resolve/df9373821efad895edf07b50a31ecd652b679b93/queries/train-00000-of-00001.parquet",
            sha256: Digest::NotRecorded,
        },
    ]),
    span_unit: Some(SpanUnit::Chunks),
    read,
};

fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    conteb::read(name, dir, "documents/train-00000-of-00001.jsonl", "queries/train-00000-of-00001.jsonl")
}
